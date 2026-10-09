"""Reconstruct the 24-hour XGBoost baseline from surviving project evidence.

This is a fresh experiment: it never reads or overwrites a historical model.
Only real train sensor and fault-log files are used.  The old 24-hour source
file is absent, so assumptions that cannot be recovered are written to the
experiment report rather than presented as exact historical settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "output" / "baseline_recovery"
HORIZON = 86400
SEED = 42
SENSORS = [
    "IONGAUGEPRESSURE", "ETCHBEAMVOLTAGE", "ETCHBEAMCURRENT",
    "ETCHSUPPRESSORVOLTAGE", "ETCHSUPPRESSORCURRENT", "FLOWCOOLFLOWRATE",
    "FLOWCOOLPRESSURE", "ETCHGASCHANNEL1READBACK", "ETCHPBNGASREADBACK",
    "FIXTURETILTANGLE", "ROTATIONSPEED", "ACTUALROTATIONANGLE",
    "FIXTURESHUTTERPOSITION", "ETCHSOURCEUSAGE", "ETCHAUXSOURCETIMER",
    "ETCHAUX2SOURCETIMER", "ACTUALSTEPDURATION",
]
ROLLING = [
    "FLOWCOOLPRESSURE", "FLOWCOOLFLOWRATE", "IONGAUGEPRESSURE",
    "ETCHGASCHANNEL1READBACK", "ETCHPBNGASREADBACK",
]
PROCESS = ["stage", "recipe", "recipe_step"]
FAULTS = ["pressure_low", "pressure_high", "leak"]
EXPECTED = {
    "precision": 0.167, "recall": 0.506, "f1": 0.251,
    "balanced_accuracy": 0.674, "average_precision": 0.172,
    "tn": 904772, "fp": 170002, "fn": 33238, "tp": 34100,
    "false_alarm_events_10m": 4052,
}
PARAMETERS = {
    "objective": "binary:logistic", "eval_metric": "logloss",
    "tree_method": "hist", "n_estimators": 120, "max_depth": 5,
    "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_weight": 1, "reg_lambda": 1.0, "n_jobs": 8,
    "random_state": SEED,
}


def _fault_times(log: pd.DataFrame, tool: str, fault: str) -> np.ndarray:
    return np.sort(log.loc[(log.tool_id == tool) & (log.fault_type == fault), "time"].drop_duplicates().to_numpy(dtype=np.float64))


def _next_event(times: np.ndarray, current: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(times, current, side="left")
    result = np.full(len(current), np.inf, dtype=np.float64)
    valid = pos < len(times)
    result[valid] = times[pos[valid]]
    return result


def _rolling_features(frame: pd.DataFrame, sensor: str) -> None:
    group = frame.groupby("session_id", sort=False)[sensor]
    previous = group.shift(1)
    diff = frame[sensor] - previous
    frame[f"{sensor}_diff"] = diff.astype("float32")
    gap = frame["time_diff_seconds"].replace(0, np.nan)
    frame[f"{sensor}_per_hour"] = (diff / gap * 3600).astype("float32")
    for window in (5, 30):
        frame[f"{sensor}_mean_{window}m"] = group.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True).sort_index().to_numpy(dtype=np.float32)
        frame[f"{sensor}_std_{window}m"] = group.rolling(window, min_periods=2).std().reset_index(level=0, drop=True).sort_index().to_numpy(dtype=np.float32)
    before_four = group.shift(4)
    previous_time = frame.groupby("session_id", sort=False)["time"].shift(4)
    frame[f"{sensor}_slope_5m"] = ((frame[sensor] - before_four) / (frame["time"] - previous_time).replace(0, np.nan) * 3600).astype("float32")
    frame[f"{sensor}_ewma_10m"] = group.ewm(span=10, adjust=False).mean().reset_index(level=0, drop=True).sort_index().to_numpy(dtype=np.float32)


def prepare_tool(path: Path, fault_log: pd.DataFrame, cache: Path) -> dict:
    tool = path.name.split("_sensor_ttf")[0]
    meta_path = cache.with_suffix(".json")
    signature = {"source_bytes": path.stat().st_size, "source_mtime_ns": path.stat().st_mtime_ns,
                 "fault_bytes": (DATA / "fault_events.csv").stat().st_size,
                 "fault_mtime_ns": (DATA / "fault_events.csv").stat().st_mtime_ns,
                 "version": 1}
    if cache.exists() and meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old["signature"] == signature:
            print(f"{tool}: prepared cache", flush=True)
            return old["report"]
    started = time.monotonic()
    raw = pd.read_parquet(path, columns=["time", *PROCESS, *SENSORS])
    raw_rows = len(raw)
    if (raw["time"].diff().dropna() < 0).any():
        raise ValueError(f"{tool}: source time decreases")
    raw["minute_bucket"] = (raw["time"] // 60).astype("int64")
    spec = {"time": ("time", "max"), **{c: (c, "last") for c in PROCESS},
            **{c: (c, "mean") for c in SENSORS}}
    for sensor in ROLLING:
        for suffix, method in (("std", "std"), ("min", "min"), ("max", "max")):
            spec[f"{sensor}_{suffix}_1m"] = (sensor, method)
    frame = raw.groupby("minute_bucket", sort=True).agg(**spec).reset_index(drop=True)
    del raw
    frame["time"] = frame["time"].astype("int64")
    gap = frame["time"].diff().fillna(0)
    restart = (gap > 120).astype("int8")
    frame["session_id"] = restart.cumsum().astype("int32")
    frame["time_diff_seconds"] = gap.astype("float32")
    frame["restart_after_stop"] = restart
    session = frame.groupby("session_id", sort=False)
    frame["session_age_minutes"] = ((frame["time"] - session["time"].transform("first")) / 60).astype("float32")
    frame["history_minutes"] = (session.cumcount() + 1).astype("float32")
    for c in PROCESS:
        frame[f"{c}_changed"] = (frame[c] != session[c].shift(1)).astype("int8")
    frame["tool_number"] = int(tool[:2]) * 2 + int(tool[-2:]) - 2
    for sensor in ROLLING:
        _rolling_features(frame, sensor)
    frame["pressure_minus_flow"] = (frame["FLOWCOOLPRESSURE"] - frame["FLOWCOOLFLOWRATE"]).astype("float32")
    frame["pressure_div_flow"] = (frame["FLOWCOOLPRESSURE"] / frame["FLOWCOOLFLOWRATE"].replace(0, np.nan)).astype("float32")

    now = frame["time"].to_numpy(dtype=np.float64)
    events = np.column_stack([_next_event(_fault_times(fault_log, tool, f), now) for f in FAULTS])
    nearest = np.min(events, axis=1)
    label = np.full(len(frame), -1, dtype=np.int8)
    label[now + HORIZON <= now[-1]] = 0
    label[nearest - now <= HORIZON] = 1
    frame["label"] = label
    frame["next_event_time"] = np.where(np.isfinite(nearest), nearest, np.nan)
    cutoff = int(now[int(len(frame) * 0.8)])
    frame["split"] = np.where(now < cutoff - HORIZON, 0, np.where(now >= cutoff, 2, 1)).astype("int8")
    frame.to_parquet(cache, index=False)
    report = {"tool": tool, "raw_rows": raw_rows, "minute_rows": len(frame),
              "split_cutoff_time": cutoff, "train_rows": int((frame.split == 0).sum()),
              "purge_rows": int((frame.split == 1).sum()), "validation_rows": int((frame.split == 2).sum()),
              "train_positive": int(((frame.split == 0) & (frame.label == 1)).sum()),
              "validation_positive": int(((frame.split == 2) & (frame.label == 1)).sum()),
              "validation_negative": int(((frame.split == 2) & (frame.label == 0)).sum()),
              "validation_censored": int(((frame.split == 2) & (frame.label == -1)).sum()),
              "seconds": round(time.monotonic() - started, 1)}
    meta_path.write_text(json.dumps({"signature": signature, "report": report}, indent=2), encoding="utf-8")
    print(f"{tool}: {raw_rows:,} -> {len(frame):,} minute rows; valid +{report['validation_positive']:,}/-{report['validation_negative']:,}", flush=True)
    return report


def _features(columns: list[str]) -> list[str]:
    forbidden = {"time", "minute_bucket", "session_id", "label", "next_event_time", "split"}
    result = [c for c in columns if c not in forbidden]
    if len(result) != 85:
        raise ValueError(f"Expected 85 past/current features; found {len(result)}")
    if any("TTF" in c.upper() or "FAULT" in c.upper() or "EVENT" in c.upper() for c in result):
        raise ValueError("Future/target column found in model features")
    return result


def _load_train(paths: list[Path], features: list[str], cutoffs: dict[str, int]) -> tuple[np.ndarray, np.ndarray, dict]:
    xs, ys, event_sets = [], [], []
    future_events = 0
    for path in paths:
        part = pd.read_parquet(path, columns=[*features, "label", "split", "next_event_time"])
        selected = part[(part.split == 0) & (part.label >= 0)]
        x = selected[features].to_numpy(dtype=np.float32, copy=True)
        np.nan_to_num(x, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
        xs.append(x)
        ys.append(selected.label.to_numpy(dtype=np.int8))
        event_sets.extend((path.stem, int(v)) for v in selected.loc[selected.label == 1, "next_event_time"].dropna().unique())
        future_events += int((selected.loc[selected.label == 1, "next_event_time"] >= cutoffs[path.stem]).sum())
        print(f"train load {path.stem}: {len(selected):,}", flush=True)
    y = np.concatenate(ys)
    return np.concatenate(xs), y, {"training_positive_events": len(set(event_sets)), "training_event_after_cutoff": future_events, "event_keys": set(event_sets)}


def _evaluate(model: XGBClassifier, paths: list[Path], features: list[str]) -> tuple[dict, set]:
    truth_parts, score_parts = [], []
    false_events = 0
    validation_events = set()
    for path in paths:
        part = pd.read_parquet(path, columns=[*features, "time", "label", "split", "next_event_time"])
        valid = part[(part.split == 2) & (part.label >= 0)]
        score = model.predict_proba(valid[features].to_numpy(dtype=np.float32, copy=True))[:, 1]
        y = valid.label.to_numpy(dtype=np.int8)
        pred = score >= 0.5
        false_times = valid.loc[(y == 0) & pred, "time"].to_numpy(dtype=np.float64)
        if len(false_times):
            false_events += int(1 + np.sum(np.diff(false_times) > 600))
        validation_events.update((path.stem, int(v)) for v in valid.loc[y == 1, "next_event_time"].dropna().unique())
        truth_parts.append(y)
        score_parts.append(score)
        print(f"valid {path.stem}: {len(y):,}; false alarm episodes {int(1 + np.sum(np.diff(false_times) > 600)) if len(false_times) else 0:,}", flush=True)
    y = np.concatenate(truth_parts)
    score = np.concatenate(score_parts)
    pred = score >= 0.5
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel()]
    return {"validation_rows": len(y), "positive_rows": int(y.sum()),
            "precision": float(precision_score(y, pred)), "recall": float(recall_score(y, pred)),
            "f1": float(f1_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "average_precision": float(average_precision_score(y, score)),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "false_alarm_events_10m": false_events, "validation_positive_events": len(validation_events)}, validation_events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "train", "all"], nargs="?", default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    prepared_dir = OUT / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(DATA.glob("??_M??_sensor_ttf.parquet"))
    if len(sources) != 20:
        raise ValueError(f"Expected 20 actual train tools; found {len(sources)}")
    fault_log = pd.read_csv(DATA / "fault_events.csv", usecols=["tool_id", "fault_type", "time"])
    if args.action in ("prepare", "all"):
        tool_reports = [prepare_tool(p, fault_log, prepared_dir / f"{p.stem.split('_sensor_ttf')[0]}.parquet") for p in sources]
        summary = {"tools": tool_reports, "raw_rows": sum(r["raw_rows"] for r in tool_reports),
                   "minute_rows": sum(r["minute_rows"] for r in tool_reports),
                   "validation_positive": sum(r["validation_positive"] for r in tool_reports),
                   "validation_negative": sum(r["validation_negative"] for r in tool_reports),
                   "validation_censored": sum(r["validation_censored"] for r in tool_reports)}
        (OUT / "preparation_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("PREP TOTAL", json.dumps({k:v for k,v in summary.items() if k!="tools"}), flush=True)
    if args.action == "prepare":
        return
    paths = [prepared_dir / f"{p.stem.split('_sensor_ttf')[0]}.parquet" for p in sources]
    if not all(p.exists() for p in paths):
        raise FileNotFoundError("Prepared minute files missing; run prepare first")
    first = pd.read_parquet(paths[0])
    features = _features(first.columns.tolist())
    del first
    prep_report = json.loads((OUT / "preparation_report.json").read_text(encoding="utf-8"))
    cutoffs = {r["tool"]: r["split_cutoff_time"] for r in prep_report["tools"]}
    x, y, train_audit = _load_train(paths, features, cutoffs)
    weight = int((y == 0).sum()) / max(int((y == 1).sum()), 1)
    model = XGBClassifier(**PARAMETERS, scale_pos_weight=weight)
    print(f"fitting XGBoost: rows={len(y):,}, positives={int(y.sum()):,}, features={len(features)}, weight={weight:.4f}", flush=True)
    model.fit(x, y)
    del x, y
    model_path = OUT / "xgb_24h_recovery.joblib"
    joblib.dump(model, model_path)
    result, validation_event_keys = _evaluate(model, paths, features)
    comparison = {key: {"reference": value, "reproduced": result[key], "difference": result[key]-value} for key,value in EXPECTED.items()}
    report = {
        "status": "independent reconstruction; exact historical source/model absent",
        "data": "20 real train parquet partitions and fault_events.csv",
        "label": "next logged low/high/leak fault within 86400 wall-clock seconds; 0 only when full follow-up is observed; otherwise censored",
        "split": "per-tool chronological first 80% / last 20%, with 86400-second purge before split",
        "aggregation": "all 4-second rows into one-minute current/past statistics; stops >120 seconds reset rolling state",
        "threshold": 0.5, "random_state": SEED, "features": features,
        "xgboost_parameters": {**PARAMETERS, "scale_pos_weight": weight},
        "unrecoverable_settings": ["exact historical one-minute aggregation and rolling implementation", "exact historical XGBoost depth/learning-rate/subsampling settings"],
        "leakage_audit": {"prohibited_feature_names_present": False,
                          "training_positive_events": train_audit["training_positive_events"],
                          "training_event_after_cutoff": train_audit["training_event_after_cutoff"],
                          "train_validation_positive_event_overlap": len(train_audit["event_keys"] & validation_event_keys),
                          "future_time_or_fault_columns_in_features": False},
        "metrics": result, "comparison_to_agents_md": comparison,
    }
    (OUT / "reproduction_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "metric_comparison.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "reference", "reproduced", "difference"])
        writer.writerows([key, item["reference"], item["reproduced"], item["difference"]] for key,item in comparison.items())
    print("RESULT", json.dumps(result), flush=True)
    print(f"Saved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
