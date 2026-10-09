"""One controlled past-only feature experiment on the frozen 24h XGBoost split.

Usage: python src/feature_engineering_24h.py prepare|train|grid
The 20 cached one-minute real-data partitions, labels, split, XGBoost settings,
class weight, and validation rows are reused without modifying baseline files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from evaluate_alarm_persistence_24h import causal_alarm
from evaluate_event_thresholds_24h import (
    BASELINE, ROOT, SOURCE, event_metrics, load_events, read_json,
    row_and_alarm_metrics, write_csv,
)
from evaluate_threshold_persistence_grid_24h import CONDITIONS, LEVELS, THRESHOLDS
from recover_xgb24_baseline import PROCESS, ROLLING, SENSORS

OUT = ROOT / "output" / "experiments" / "feature_engineering_24h"
MODEL = ROOT / "output" / "models" / "xgb_24h_feature_engineering.joblib"
BASE_FEATURES = read_json(BASELINE / "reproduction_report.json")["features"]
FOCUS = ("FLOWCOOLPRESSURE", "FLOWCOOLFLOWRATE")
SUFFIXES = (
    "diff_1m_clock", "diff_5m_clock", "diff_10m_clock",
    "mean_10m_clock", "std_10m_clock",
    "min_5m_clock", "max_5m_clock",
    "min_10m_clock", "max_10m_clock",
    "min_30m_clock", "max_30m_clock",
    "slope_10m_clock", "slope_30m_clock",
)
ADDED = [f"{sensor}_{suffix}" for sensor in FOCUS for suffix in SUFFIXES]
ALL_FEATURES = [*BASE_FEATURES, *ADDED]
SCORE = OUT / "validation_probabilities.npz"


def category(name: str) -> str:
    if name in SENSORS:
        return "current_sensor"
    if name in PROCESS or name.endswith("_changed"):
        return "process"
    if name.endswith("_diff") or "_diff_" in name:
        return "diff"
    if name.endswith("_per_hour"):
        return "rate"
    if "_mean_" in name:
        return "rolling_mean"
    if "_std_" in name:
        return "rolling_std"
    if "_min_" in name or "_max_" in name:
        return "rolling_min_max"
    if "_slope_" in name:
        return "slope"
    if "_ewma_" in name:
        return "ewma"
    if name in ("pressure_minus_flow", "pressure_div_flow"):
        return "sensor_relation"
    return "other"


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(values, dtype=np.float64)]


def _range_sum(prefix: np.ndarray, left: np.ndarray) -> np.ndarray:
    return prefix[1:] - prefix[left]


def _minmax(values: np.ndarray, left: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trailing clock-window extrema; left[i] is never later than i."""
    low = np.full(len(values), np.nan, dtype=np.float32)
    high = low.copy()
    mins, maxs = deque(), deque()
    for i, value in enumerate(values):
        while mins and mins[0] < left[i]:
            mins.popleft()
        while maxs and maxs[0] < left[i]:
            maxs.popleft()
        if np.isfinite(value):
            while mins and values[mins[-1]] >= value:
                mins.pop()
            while maxs and values[maxs[-1]] <= value:
                maxs.pop()
            mins.append(i)
            maxs.append(i)
        if mins:
            low[i] = values[mins[0]]
            high[i] = values[maxs[0]]
    return low, high


def _session_features(bucket: np.ndarray, values: np.ndarray) -> dict[str, np.ndarray]:
    """All windows use real integer clock minutes within one operating session."""
    n = len(bucket)
    if not n or np.any(np.diff(bucket) <= 0):
        raise AssertionError("Expected strictly increasing one-minute buckets")
    result = {}
    for minutes in (1, 5, 10):
        where = np.searchsorted(bucket, bucket - minutes, side="left")
        exact = (where < n) & (bucket[np.minimum(where, n - 1)] == bucket - minutes)
        diff = np.full(n, np.nan, dtype=np.float32)
        diff[exact] = values[exact] - values[where[exact]]
        result[f"diff_{minutes}m_clock"] = diff
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.0).astype(np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(bucket) > 30) + 1]
    own_start = starts[np.searchsorted(starts, np.arange(n), side="right") - 1]
    minute = (bucket - bucket[own_start]).astype(np.float64)
    p_count = _prefix(finite.astype(np.float64))
    p_x, p_xx = _prefix(safe), _prefix(safe * safe)
    p_t = _prefix(np.where(finite, minute, 0.0))
    p_tt = _prefix(np.where(finite, minute * minute, 0.0))
    p_tx = _prefix(safe * minute)
    for window in (5, 10, 30):
        left = np.searchsorted(bucket, bucket - (window - 1), side="left")
        if np.any(left > np.arange(n)):
            raise AssertionError("A rolling window looks ahead")
        count = _range_sum(p_count, left)
        total = _range_sum(p_x, left)
        if window == 10:
            mean = np.divide(total, count, out=np.full(n, np.nan), where=count > 0)
            variance = np.divide(_range_sum(p_xx, left) - total * mean, count - 1,
                                 out=np.full(n, np.nan), where=count > 1)
            result["mean_10m_clock"] = mean.astype(np.float32)
            result["std_10m_clock"] = np.sqrt(np.maximum(variance, 0)).astype(np.float32)
        result[f"min_{window}m_clock"], result[f"max_{window}m_clock"] = _minmax(values, left)
        if window in (10, 30):
            st = _range_sum(p_t, left)
            denom = count * _range_sum(p_tt, left) - st * st
            numer = count * _range_sum(p_tx, left) - st * total
            slope = np.divide(numer, denom, out=np.full(n, np.nan), where=(count >= 2) & (denom > 0))
            result[f"slope_{window}m_clock"] = (slope * 60).astype(np.float32)
    return result


def _augment(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    time = frame["time"].to_numpy(dtype=np.int64)
    if np.any(np.diff(time) <= 0):
        raise AssertionError("Minute timestamps are not increasing")
    starts = np.r_[0, np.flatnonzero(np.diff(time) > 120) + 1]
    observed_start = frame["restart_after_stop"].to_numpy(dtype=np.int8)
    if np.any(observed_start[starts[1:]] != 1) or int(np.count_nonzero(observed_start)) != len(starts) - 1:
        raise AssertionError("Stop/reset boundaries differ from frozen baseline")
    # Insert virtual clock gaps between sessions. This preserves actual
    # within-session minutes while making a single vectorized rolling pass
    # mathematically equivalent to separate per-session passes.
    bucket = time // 60
    extra = np.maximum(0, 31 - (bucket[starts[1:]] - bucket[starts[1:] - 1]))
    offset = np.r_[0, np.cumsum(extra, dtype=np.int64)]
    session_index = np.searchsorted(starts, np.arange(len(time)), side="right") - 1
    rolling_clock = bucket + offset[session_index]
    if np.any(np.diff(rolling_clock) <= 0) or np.any(rolling_clock[starts[1:]] - rolling_clock[starts[1:] - 1] < 31):
        raise AssertionError("Virtual session separation failed")
    coverage = {}
    for sensor in FOCUS:
        values = frame[sensor].to_numpy(dtype=np.float32)
        outputs = _session_features(rolling_clock, values)
        for suffix, array in outputs.items():
            frame[f"{sensor}_{suffix}"] = array
            coverage[f"{sensor}_{suffix}"] = float(np.isfinite(array).mean())
    if set(ADDED) - set(frame.columns):
        raise AssertionError("Missing added features")
    return frame, {"sessions": len(starts), "finite_fraction": coverage}


def _write_audit() -> None:
    rows = [{"feature": name, "category": category(name), "status": "existing",
             "window_meaning": "observed rows" if any(f"_{w}m" in name for w in (5, 10, 30)) else "current or previous observed value"}
            for name in BASE_FEATURES]
    rows.extend({"feature": name, "category": category(name), "status": "added",
                 "window_meaning": "clock minutes, current and past only; reset after >120s stop"} for name in ADDED)
    write_csv(OUT / "feature_inventory.csv", rows, list(rows[0]))


def prepare() -> None:
    if (OUT / "preparation_report.json").exists() or (OUT / "training_report.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed preparation in {OUT}")
    paths = sorted((BASELINE / "prepared").glob("??_M??.parquet"))
    if len(paths) != 20 or len(BASE_FEATURES) != 85 or len(ADDED) != 26 or len(set(ALL_FEATURES)) != len(ALL_FEATURES):
        raise AssertionError("Feature population changed or duplicated")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prepared").mkdir(exist_ok=True)
    if not (OUT / "feature_inventory.csv").exists():
        _write_audit()
    tool_rows = []
    for path in paths:
        original = pd.read_parquet(path)
        destination = OUT / "prepared" / path.name
        if destination.exists():
            frame = pd.read_parquet(destination)
            if not set(ADDED).issubset(frame.columns):
                raise AssertionError(f"Partial prepared tool: {path.stem}")
            audit = {"sessions": int(frame["restart_after_stop"].sum()) + 1,
                     "finite_fraction": {name: float(np.isfinite(frame[name]).mean()) for name in ADDED}}
            print(f"reused complete prepared {path.stem}", flush=True)
        else:
            frame, audit = _augment(original.copy())
        for col in ("time", "label", "split", "next_event_time"):
            if not frame[col].equals(original[col]):
                raise AssertionError(f"Target/split changed: {path.stem} {col}")
        if any("TTF" in name.upper() or "FAULT" in name.upper() or "EVENT" in name.upper() for name in ALL_FEATURES):
            raise AssertionError("Future/target feature name found")
        if not destination.exists():
            frame.to_parquet(destination, index=False)
        tool_rows.append({"tool": path.stem, "rows": len(frame), "train_rows": int(((frame.split == 0) & (frame.label >= 0)).sum()),
                          "validation_rows": int(((frame.split == 2) & (frame.label >= 0)).sum()),
                          "sessions": audit["sessions"], **audit["finite_fraction"]})
        print(f"prepared {path.stem}: {len(frame):,} rows, {audit['sessions']} sessions", flush=True)
    write_csv(OUT / "feature_coverage_by_tool.csv", tool_rows, list(tool_rows[0]))
    (OUT / "preparation_report.json").write_text(json.dumps({"base_features": len(BASE_FEATURES),
        "added_features": ADDED, "all_features": len(ALL_FEATURES), "focus_sensors": FOCUS,
        "window_definition": "Integer dataset time//60 buckets. W-minute window includes buckets current-(W-1)..current; no gap filling. Exact lag changes require an observation in that earlier bucket.",
        "stop_rule": ">120 second gap resets every feature history, same as baseline",
        "causality": "Every feature uses only indices <= current within its session; train labels/split unchanged; validation may use earlier train history, never the reverse.",
        "tools": len(tool_rows), "train_rows": sum(r["train_rows"] for r in tool_rows),
        "validation_rows": sum(r["validation_rows"] for r in tool_rows)}, indent=2), encoding="utf-8")


def train() -> None:
    if MODEL.exists() or SCORE.exists():
        raise FileExistsError("Experiment model or score already exists")
    baseline = read_json(BASELINE / "reproduction_report.json")
    paths = sorted((OUT / "prepared").glob("??_M??.parquet"))
    if len(paths) != 20:
        raise AssertionError("Prepare all 20 tools first")
    xs, ys, train_events = [], [], set()
    cutoffs = {r["tool"]: r["split_cutoff_time"] for r in read_json(BASELINE / "preparation_report.json")["tools"]}
    for path in paths:
        frame = pd.read_parquet(path, columns=[*ALL_FEATURES, "label", "split", "next_event_time"])
        part = frame.loc[(frame.split == 0) & (frame.label >= 0)]
        xs.append(part[ALL_FEATURES].to_numpy(dtype=np.float32, copy=True))
        ys.append(part.label.to_numpy(dtype=np.int8, copy=True))
        times = part.loc[part.label == 1, "next_event_time"].dropna().unique()
        if np.any(times >= cutoffs[path.stem]):
            raise AssertionError("Train positive event crosses validation cutoff")
        train_events.update((path.stem, int(t)) for t in times)
    x, y = np.concatenate(xs), np.concatenate(ys)
    del xs, ys
    parameters = baseline["xgboost_parameters"]
    if len(y) != 4606868 or int(y.sum()) != 380626 or not np.isclose(parameters["scale_pos_weight"], (len(y)-int(y.sum()))/int(y.sum())):
        raise AssertionError("Training population or class weight differs from baseline")
    model = XGBClassifier(**parameters)
    print(f"fitting unchanged XGBoost on {len(y):,} rows, {x.shape[1]} features", flush=True)
    model.fit(x, y)
    del x, y
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": ALL_FEATURES, "base_parameters": parameters}, MODEL)
    score_parts, truth_parts, time_parts, tool_parts, valid_events = [], [], [], [], set()
    with np.load(SOURCE / "validation_probabilities.npz") as fixed:
        reference_truth, reference_time, reference_tool = (fixed[k].copy() for k in ("truth", "time", "tool_code"))
    for code, path in enumerate(paths):
        frame = pd.read_parquet(path, columns=[*ALL_FEATURES, "time", "label", "split", "next_event_time"])
        part = frame.loc[(frame.split == 2) & (frame.label >= 0)]
        truth_parts.append(part.label.to_numpy(dtype=np.int8, copy=True))
        time_parts.append(part.time.to_numpy(dtype=np.int64, copy=True))
        tool_parts.append(np.full(len(part), code, dtype=np.uint8))
        score_parts.append(model.predict_proba(part[ALL_FEATURES].to_numpy(dtype=np.float32, copy=True))[:, 1].astype(np.float32))
        valid_events.update((path.stem, int(t)) for t in part.loc[part.label == 1, "next_event_time"].dropna().unique())
        print(f"scored {path.stem}: {len(part):,} validation rows", flush=True)
    truth, time, tool, score = map(np.concatenate, (truth_parts, time_parts, tool_parts, score_parts))
    if not (np.array_equal(truth, reference_truth) and np.array_equal(time, reference_time) and np.array_equal(tool, reference_tool)):
        raise AssertionError("Validation rows/labels/order differ from frozen baseline")
    if train_events & valid_events:
        raise AssertionError("Same positive fault event appears in train and validation")
    np.savez_compressed(SCORE, truth=truth, probability=score, time=time, tool_code=tool)
    (OUT / "training_report.json").write_text(json.dumps({"train_rows": 4606868, "train_positive_rows": 380626,
        "validation_rows": len(truth), "validation_positive_rows": int(truth.sum()),
        "parameters": parameters, "validation_alignment_exact": True,
        "train_validation_positive_event_overlap": 0,
        "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "score_sha256": hashlib.sha256(SCORE.read_bytes()).hexdigest()}, indent=2), encoding="utf-8")


def grid() -> None:
    report_path = OUT / "experiment_report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    with np.load(SCORE) as saved, np.load(SOURCE / "validation_probabilities.npz") as frozen:
        truth, score, time, tool = (saved[k].copy() for k in ("truth", "probability", "time", "tool_code"))
        old = frozen["probability"].copy()
    sweep = read_json(SOURCE / "sweep_report.json")
    tool_map = {int(k): v for k, v in sweep["tool_code_map"].items()}
    cutoffs = {r["tool"]: r["split_cutoff_time"] for r in read_json(BASELINE / "preparation_report.json")["tools"]}
    events = load_events(cutoffs, set(tool_map.values()))
    if len(events) != 181:
        raise AssertionError("Fault event population changed")
    reverse = {v: k for k, v in tool_map.items()}
    ap_new = float(average_precision_score(truth, score))
    ap_old = float(average_precision_score(truth, old))
    def evaluate(probability, threshold, condition, window, hits, model_name):
        alarm = causal_alarm(probability >= threshold, time, tool, sorted(tool_map), window, hits)
        binary = alarm.astype(np.uint8)
        row, distribution = row_and_alarm_metrics(.5, truth, binary, time, tool,
                                                   ap_new if model_name == "feature_model" else ap_old, tool_map)
        event, _ = event_metrics(.5, events, binary, time, tool, reverse)
        per_shift = [r["false_alarm_events_per_observed_day"] / 2 for r in distribution]
        return {"model": model_name, "condition": condition,
                "window_minutes": window, "minimum_hits": hits,
                **row, **event, "threshold": threshold,
                "false_alarm_events_per_observed_tool_12h": row["false_alarm_events_10m"] / (len(truth) / 720),
                "tools_over_four_per_observed_12h": int(sum(v > 4 for v in per_shift)),
                "worst_tool_per_observed_12h": float(max(per_shift))}, distribution
    base05, _ = evaluate(old, .5, "1_of_1m", 1, 1, "baseline")
    new05, _ = evaluate(score, .5, "1_of_1m", 1, 1, "feature_model")
    oldfixed, _ = evaluate(old, .36, "2_of_5m", 5, 2, "baseline")
    newfixed, by_tool = evaluate(score, .36, "2_of_5m", 5, 2, "feature_model")
    if (base05["detected_fault_events"], base05["false_alarm_events_10m"],
        oldfixed["detected_fault_events"], oldfixed["false_alarm_events_10m"]) != (112, 3414, 138, 6066):
        raise AssertionError("Frozen baseline/policy not reproduced")
    fixed = [base05, new05, oldfixed, newfixed]
    write_csv(OUT / "fixed_policy_comparison.csv", fixed, list(fixed[0]))
    write_csv(OUT / "fixed_policy_false_alarm_by_tool.csv",
              [{"model": "feature_model", "condition": "2_of_5m", **item, "threshold": .36,
                "false_alarm_events_per_observed_tool_12h": item["false_alarm_events_per_observed_day"] / 2,
                "over_four": item["false_alarm_events_per_observed_day"] / 2 > 4} for item in by_tool],
              ["model", "condition", *by_tool[0].keys(), "false_alarm_events_per_observed_tool_12h", "over_four"])
    grid_rows = []
    for threshold in THRESHOLDS:
        for condition, window, hits in CONDITIONS:
            item, _ = evaluate(score, threshold, condition, window, hits, "feature_model")
            grid_rows.append(item)
            print(f"{threshold:.2f} {condition}: {item['detected_fault_events']}/181, false={item['false_alarm_events_10m']}", flush=True)
    write_csv(OUT / "threshold_persistence_grid.csv", grid_rows, list(grid_rows[0]))
    levels = []
    for level in LEVELS:
        eligible = [r for r in grid_rows if r["fault_event_recall_all_logged"] >= level]
        chosen = min(eligible, key=lambda r: (r["false_alarm_events_10m"], -r["detected_fault_events"])) if eligible else None
        levels.append({"target_recall": level, "candidate_found": chosen is not None,
                       "threshold": chosen["threshold"] if chosen else None,
                       "condition": chosen["condition"] if chosen else None,
                       "detected_fault_events": chosen["detected_fault_events"] if chosen else None,
                       "false_alarm_events_10m": chosen["false_alarm_events_10m"] if chosen else None,
                       "tools_over_four_per_observed_12h": chosen["tools_over_four_per_observed_12h"] if chosen else None})
    write_csv(OUT / "recall_level_best.csv", levels, list(levels[0]))
    report_path.write_text(json.dumps({"purpose": "feature effect on frozen 24h split, separate fixed policy and reused-validation grid",
        "base_features": 85, "added_features": ADDED, "feature_count": len(ALL_FEATURES),
        "ap_baseline": ap_old, "ap_feature_model": ap_new, "ap_change": ap_new - ap_old,
        "threshold_0_5": {"baseline": base05, "feature_model": new05},
        "fixed_0_36_2_of_5m": {"baseline": oldfixed, "feature_model": newfixed},
        "grid_best_by_event_recall": levels,
        "retain_feature_group_for_next_experiment": bool(ap_new > ap_old and
             newfixed["detected_fault_events"] >= oldfixed["detected_fault_events"] and
             newfixed["false_alarm_events_10m"] < oldfixed["false_alarm_events_10m"]),
        "limitations": "Threshold grid reused validation; observed-minute per-12h rates are not actual duty-shift counts. Independent later time validation is required."},
        ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "grid"))
    action = parser.parse_args().action
    {"prepare": prepare, "train": train, "grid": grid}[action]()


if __name__ == "__main__":
    main()
