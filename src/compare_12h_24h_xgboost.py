"""Train a matched 12h XGBoost and compare it with the frozen 24h baseline."""

from __future__ import annotations

import csv
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from evaluate_event_thresholds_24h import BASELINE, ROOT, SOURCE, load_events, read_json
from recover_xgb24_baseline import PARAMETERS

OUT = ROOT / "output" / "experiments" / "horizon_comparison_12h_24h"
MODEL_PATH = ROOT / "output" / "models" / "xgb_12h_matched_comparison.joblib"
H12, H24 = 43200, 86400


def confusion(y: np.ndarray, score: np.ndarray) -> dict:
    hit = score >= 0.5
    positive = y == 1
    tp = int(np.count_nonzero(hit & positive))
    fp = int(np.count_nonzero(hit & ~positive))
    fn = int(np.count_nonzero(positive)) - tp
    tn = len(y) - int(np.count_nonzero(positive)) - fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {"validation_rows": len(y), "positive_rows": int(positive.sum()),
            "negative_rows": int((~positive).sum()), "accuracy": (tn + tp) / len(y),
            "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "balanced_accuracy": (recall + specificity) / 2,
            "average_precision": float(average_precision_score(y, score)),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def event_audit(events: list[dict], horizon: int, score: np.ndarray,
                time: np.ndarray, tool: np.ndarray, code_by_tool: dict[str, int]) -> tuple[dict, list[dict]]:
    positions = {name: np.flatnonzero(tool == code) for name, code in code_by_tool.items()}
    details = []
    for event in events:
        idx = positions[event["tool_id"]]
        t = time[idx]
        fault_time = event["fault_time"]
        left = np.searchsorted(t, fault_time - horizon, side="left")
        right = np.searchsorted(t, fault_time, side="left")
        window = idx[left:right]
        alerts = window[score[window] >= 0.5]
        first = int(time[alerts[0]]) if len(alerts) else None
        details.append({"tool_id": event["tool_id"], "fault_type": event["fault_type"],
                        "fault_time": fault_time, "sensor_rows_in_horizon": len(window),
                        "evaluable": bool(len(window)), "detected": bool(len(alerts)),
                        "first_alert_time": first,
                        "lead_time_minutes": (fault_time - first) / 60 if first is not None else None})
    evaluable = sum(row["evaluable"] for row in details)
    detected = sum(row["detected"] for row in details)
    leads = [row["lead_time_minutes"] for row in details if row["detected"]]
    return {"logged_fault_events": len(details), "evaluable_fault_events": evaluable,
            "detected_fault_events": detected, "all_fault_event_recall": detected / len(details),
            "evaluable_fault_event_recall": detected / evaluable if evaluable else None,
            "first_alert_lead_median_minutes": float(np.median(leads)) if leads else None,
            "first_alert_lead_min_minutes": float(np.min(leads)) if leads else None,
            "first_alert_lead_max_minutes": float(np.max(leads)) if leads else None}, details


def false_episodes(y: np.ndarray, score: np.ndarray, time: np.ndarray,
                   tool: np.ndarray, tool_map: dict[int, str]) -> tuple[int, list[dict]]:
    rows = []
    for code, name in sorted(tool_map.items()):
        selected = (tool == code) & (y == 0) & (score >= 0.5)
        t = time[selected]
        count = int(1 + np.count_nonzero(np.diff(t) > 600)) if len(t) else 0
        observed_days = np.count_nonzero(tool == code) / (60 * 24)
        rows.append({"tool_id": name, "false_alarm_events_10m": count,
                     "observed_tool_days_equivalent": observed_days,
                     "false_alarm_events_per_observed_tool_day": count / observed_days if observed_days else None})
    return sum(row["false_alarm_events_10m"] for row in rows), rows


def main() -> None:
    if OUT.exists() or MODEL_PATH.exists():
        raise FileExistsError("Comparison output already exists; do not overwrite it")
    base = read_json(BASELINE / "reproduction_report.json")
    prep = read_json(BASELINE / "preparation_report.json")
    sweep = read_json(SOURCE / "sweep_report.json")
    features = base["features"]
    paths = sorted((BASELINE / "prepared").glob("??_M??.parquet"))
    if len(paths) != 20 or len(features) != 85:
        raise AssertionError("Missing prepared real-data partitions or baseline features")
    with np.load(SOURCE / "validation_probabilities.npz") as saved:
        y24, score24 = saved["truth"], saved["probability"]
        time24, tool24 = saved["time"], saved["tool_code"]
    tool_map = {int(code): name for code, name in sweep["tool_code_map"].items()}
    if [p.stem for p in paths] != [tool_map[k] for k in sorted(tool_map)]:
        raise AssertionError("Prepared tool order differs from saved validation scores")
    if hashlib.sha256((BASELINE / "xgb_24h_recovery.joblib").read_bytes()).hexdigest() != sweep["model_sha256"]:
        raise AssertionError("24h baseline model was changed")

    xs, ys = [], []
    train_event_keys = set()
    for path in paths:
        part = pd.read_parquet(path, columns=[*features, "time", "label", "split", "next_event_time"])
        train = part.loc[(part["split"] == 0) & (part["label"] >= 0)]
        remaining = train["next_event_time"].to_numpy(dtype=np.float64) - train["time"].to_numpy(dtype=np.float64)
        y = (remaining <= H12).astype(np.int8)
        xs.append(train[features].to_numpy(dtype=np.float32, copy=True))
        ys.append(y)
        train_event_keys.update((path.stem, int(t)) for t in train.loc[y == 1, "next_event_time"].dropna().unique())
        print(f"train {path.stem}: {len(y):,} rows; {int(y.sum()):,} positives", flush=True)
        del part, train
    x_train, y_train = np.concatenate(xs), np.concatenate(ys)
    del xs, ys
    weight = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)
    model = XGBClassifier(**PARAMETERS, scale_pos_weight=weight)
    print(f"Fit 12h XGBoost: {len(y_train):,} rows, {int(y_train.sum()):,} positives, weight={weight:.4f}", flush=True)
    model.fit(x_train, y_train)
    train_rows, train_positives = len(y_train), int(y_train.sum())
    del x_train, y_train

    score_parts, y12_parts, time_parts, tool_parts = [], [], [], []
    for code, path in enumerate(paths):
        part = pd.read_parquet(path, columns=[*features, "time", "label", "split", "next_event_time"])
        valid = part.loc[(part["split"] == 2) & (part["label"] >= 0)]
        n = len(valid)
        remaining = valid["next_event_time"].to_numpy(dtype=np.float64) - valid["time"].to_numpy(dtype=np.float64)
        y12_parts.append((remaining <= H12).astype(np.int8))
        score_parts.append(model.predict_proba(valid[features].to_numpy(dtype=np.float32, copy=True))[:, 1].astype(np.float32))
        time_parts.append(valid["time"].to_numpy(dtype=np.int64))
        tool_parts.append(np.full(n, code, dtype=np.uint8))
        print(f"valid {path.stem}: {n:,} rows", flush=True)
        del part, valid
    y12, score12 = np.concatenate(y12_parts), np.concatenate(score_parts)
    time, tool = np.concatenate(time_parts), np.concatenate(tool_parts)
    if not np.array_equal(time, time24) or not np.array_equal(tool, tool24) or len(y12) != len(y24):
        raise AssertionError("12h and 24h validation rows do not align")
    if np.any((y12 == 1) & (y24 == 0)):
        raise AssertionError("12h positive label falls outside the 24h positive label")
    metrics12, metrics24 = confusion(y12, score12), confusion(y24, score24)
    for key in ("precision", "recall", "f1", "balanced_accuracy", "average_precision", "tn", "fp", "fn", "tp"):
        if not np.isclose(metrics24[key], base["metrics"][key], atol=1e-12):
            raise AssertionError(f"24h baseline mismatch: {key}")
    cutoffs = {item["tool"]: item["split_cutoff_time"] for item in prep["tools"]}
    events = load_events(cutoffs, set(tool_map.values()))
    event12, detail12 = event_audit(events, H12, score12, time, tool, {v: k for k, v in tool_map.items()})
    event24, detail24 = event_audit(events, H24, score24, time, tool, {v: k for k, v in tool_map.items()})
    if event24["detected_fault_events"] != 112 or event24["logged_fault_events"] != 181:
        raise AssertionError("24h event baseline mismatch")
    false12, by_tool12 = false_episodes(y12, score12, time, tool, tool_map)
    false24, by_tool24 = false_episodes(y24, score24, time, tool, tool_map)
    if false24 != 3414:
        raise AssertionError("24h false alarm baseline mismatch")
    valid_event_keys = {(r["tool_id"], r["fault_time"]) for r in detail12 if r["evaluable"]}
    if train_event_keys & valid_event_keys:
        raise AssertionError("Training and validation contain the same fault event")
    comparison = []
    for horizon, metrics, event, false in ((12, metrics12, event12, false12), (24, metrics24, event24, false24)):
        comparison.append({"horizon_hours": horizon, "model": "XGBoost", "threshold": 0.5,
                           "train_rows": train_rows if horizon == 12 else None,
                           "train_positive_rows": train_positives if horizon == 12 else None,
                           **metrics, **event, "false_alarm_events_10m": false,
                           "false_alarm_events_per_observed_tool_day": false / (len(y12) / 1440)})
    OUT.mkdir(parents=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    np.savez_compressed(OUT / "validation_probabilities_12h.npz", truth=y12, probability=score12, time=time, tool_code=tool)
    with (OUT / "classification_event_comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    with (OUT / "fault_event_details.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["horizon_hours", *detail12[0]])
        writer.writeheader()
        for horizon, details in ((12, detail12), (24, detail24)):
            writer.writerows({"horizon_hours": horizon, **row} for row in details)
    with (OUT / "false_alarm_by_tool.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["horizon_hours", *by_tool12[0]])
        writer.writeheader()
        for horizon, rows in ((12, by_tool12), (24, by_tool24)):
            writer.writerows({"horizon_hours": horizon, **row} for row in rows)
    report = {"source": "20 real train-derived prepared 1-minute partitions", "horizons_hours": [12, 24],
              "reference_workbook": "LS_team6_3h_6h_검증보고서.xlsx (layout only; metrics not reused)",
              "comparison_design": "Same 85 baseline features, same per-tool 80/20 chronological split and 24h purge, same 1,142,112 labeled validation rows. 12h target is next logged low/high/leak fault within 12h; 24h model and saved probabilities are frozen. 12h model newly trained with 24h XGBoost parameters and independently computed class weight. Both thresholds fixed at 0.5.",
              "fairness_note": "The 12h validation tail is restricted to rows already fully labeled for 24h. This fixes the evaluated time rows but 12h and 24h positive prevalences differ. AP values are therefore not directly comparable without prevalence context.",
              "event_definition": "Unique tool/fault type/time after per-tool validation cutoff; detected if any alarm in [fault_time-horizon, fault_time). Event denominator includes faults without sensor observations; evaluable denominator excludes them.",
              "false_alarm_definition": "Normal-labeled alarm rows merged per tool if consecutive alarms are <=10 minutes apart; observed tool-days are labeled 1-minute validation rows /1440.",
              "model12_path": str(MODEL_PATH), "train_validation_event_overlap": len(train_event_keys & valid_event_keys),
              "xgboost_parameters_12h": {**PARAMETERS, "scale_pos_weight": weight},
              "results": comparison, "regression_models_rerun": False,
              "limitation": "These are validation results, not an independent test. Existing 3h/6h workbook had additional HGB/LSTM and TTF regression results that are not regenerated here."}
    (OUT / "comparison_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
