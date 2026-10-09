"""Matched four-horizon, 1%/100% classification comparison on real train data."""

from __future__ import annotations

import csv
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from compare_12h_24h_xgboost import confusion, event_audit, false_episodes
from evaluate_event_thresholds_24h import BASELINE, ROOT, SOURCE, load_events, read_json
from recover_xgb24_baseline import PARAMETERS

OUT = ROOT / "output" / "experiments" / "horizon_learning_curve_3h_6h_12h_24h"
MODELS = ROOT / "output" / "models"
HOURS = (3, 6, 12, 24)
SEED = 42


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Will not overwrite {OUT}")
    base = read_json(BASELINE / "reproduction_report.json")
    prep = read_json(BASELINE / "preparation_report.json")
    sweep = read_json(SOURCE / "sweep_report.json")
    old12 = read_json(ROOT / "output" / "experiments" / "horizon_comparison_12h_24h" / "comparison_report.json")
    features = base["features"]
    paths = sorted((BASELINE / "prepared").glob("??_M??.parquet"))
    tool_map = {int(k): v for k, v in sweep["tool_code_map"].items()}
    if len(paths) != 20 or [p.stem for p in paths] != [tool_map[k] for k in sorted(tool_map)]:
        raise AssertionError("Prepared 20-tool population is incomplete or misordered")

    train_x_parts, train_remaining_parts = [], []
    val_x_parts, val_remaining_parts, val_time_parts, val_tool_parts, val_24h_parts = [], [], [], [], []
    for code, path in enumerate(paths):
        part = pd.read_parquet(path, columns=[*features, "time", "next_event_time", "label", "split"])
        train = part.loc[(part.split == 0) & (part.label >= 0)]
        valid = part.loc[(part.split == 2) & (part.label >= 0)]
        train_x_parts.append(train[features].to_numpy(dtype=np.float32, copy=True))
        train_remaining_parts.append(train.next_event_time.to_numpy(dtype=np.float64) - train.time.to_numpy(dtype=np.float64))
        val_x_parts.append(valid[features].to_numpy(dtype=np.float32, copy=True))
        val_remaining_parts.append(valid.next_event_time.to_numpy(dtype=np.float64) - valid.time.to_numpy(dtype=np.float64))
        val_time_parts.append(valid.time.to_numpy(dtype=np.int64))
        val_tool_parts.append(np.full(len(valid), code, dtype=np.uint8))
        val_24h_parts.append(valid.label.to_numpy(dtype=np.int8))
        print(f"loaded {path.stem}: train {len(train):,}, validation {len(valid):,}", flush=True)
        del part, train, valid
    x_train = np.concatenate(train_x_parts)
    remaining_train = np.concatenate(train_remaining_parts)
    x_val = np.concatenate(val_x_parts)
    remaining_val = np.concatenate(val_remaining_parts)
    val_time = np.concatenate(val_time_parts)
    val_tool = np.concatenate(val_tool_parts)
    val_24h = np.concatenate(val_24h_parts)
    del train_x_parts, train_remaining_parts, val_x_parts, val_remaining_parts
    gc.collect()
    with np.load(SOURCE / "validation_probabilities.npz") as saved:
        if not (np.array_equal(saved["truth"], val_24h) and np.array_equal(saved["time"], val_time)
                and np.array_equal(saved["tool_code"], val_tool)):
            raise AssertionError("Common validation population does not match frozen 24h scores")
        score24 = saved["probability"].copy()
    old12_score_path = ROOT / "output" / "experiments" / "horizon_comparison_12h_24h" / "validation_probabilities_12h.npz"
    with np.load(old12_score_path) as saved:
        if not (np.array_equal(saved["time"], val_time) and np.array_equal(saved["tool_code"], val_tool)):
            raise AssertionError("Frozen 12h score population differs")
        score12 = saved["probability"].copy()
        y12_saved = saved["truth"].copy()
    cutoffs = {item["tool"]: item["split_cutoff_time"] for item in prep["tools"]}
    events = load_events(cutoffs, set(tool_map.values()))
    if len(events) != 181:
        raise AssertionError("Validation fault event count changed")
    rng = np.random.default_rng(SEED)
    sample_indices = np.sort(rng.choice(len(x_train), size=max(1, round(len(x_train) * .01)), replace=False))

    OUT.mkdir(parents=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    results = []
    event_rows = []
    for hours in HOURS:
        horizon = hours * 3600
        y_train = (remaining_train <= horizon).astype(np.int8)
        y_val = (remaining_val <= horizon).astype(np.int8)
        if hours == 12 and not np.array_equal(y_val, y12_saved):
            raise AssertionError("12h target differs from prior matched experiment")
        if hours == 24 and not np.array_equal(y_val, val_24h):
            raise AssertionError("24h target differs from recovered baseline")
        conditions = (("XGBoost", "1%"), ("XGBoost", "100%"))
        for family, fraction in conditions:
            selected = sample_indices if fraction == "1%" else slice(None)
            if fraction == "100%" and hours == 12:
                score = score12
                model_path = MODELS / "xgb_12h_matched_comparison.joblib"
            elif fraction == "100%" and hours == 24:
                score = score24
                model_path = BASELINE / "xgb_24h_recovery.joblib"
            else:
                y_selected = y_train[selected]
                weight = int((y_selected == 0).sum()) / max(int((y_selected == 1).sum()), 1)
                model = XGBClassifier(**PARAMETERS, scale_pos_weight=weight)
                model.fit(x_train[selected], y_selected)
                score = model.predict_proba(x_val)[:, 1].astype(np.float32)
                model_path = MODELS / f"xgb_{hours}h_{fraction.replace('%', 'pct')}_matched.joblib"
                joblib.dump(model, model_path)
                del model
            row = confusion(y_val, score)
            event, details = event_audit(events, horizon, score, val_time, val_tool,
                                         {v: k for k, v in tool_map.items()})
            false_count, _ = false_episodes(y_val, score, val_time, val_tool, tool_map)
            item = {"horizon_hours": hours, "model": family, "train_fraction": fraction,
                    "train_rows": len(y_train[selected]), "train_positive_rows": int(y_train[selected].sum()),
                    "threshold": .5, **row, **event,
                    "false_alarm_events_10m": false_count,
                    "false_alarm_events_per_observed_tool_day": false_count / (len(y_val) / 1440),
                    "model_path": str(model_path)}
            if fraction == "100%" and hours in (12, 24):
                prior = old12["results"][0 if hours == 12 else 1]
                for key in ("average_precision", "precision", "recall", "f1", "detected_fault_events", "false_alarm_events_10m"):
                    if not np.isclose(item[key], prior[key], atol=1e-12):
                        raise AssertionError(f"{hours}h frozen XGBoost result mismatch: {key}")
            results.append(item)
            event_rows.extend({"horizon_hours": hours, "model": family, "train_fraction": fraction, **detail}
                              for detail in details)
            save_csv(OUT / "classification_event_comparison.csv", results)
            print(f"{hours}h {family} {fraction}: Recall={row['recall']:.3f}, F1={row['f1']:.3f}, events={event['detected_fault_events']}/181, false={false_count}", flush=True)
            if not (fraction == "100%" and hours in (12, 24)):
                del score
            gc.collect()
    if len(results) != 8 or len(event_rows) != 8 * 181:
        raise AssertionError("Incomplete model comparison")
    save_csv(OUT / "fault_event_details.csv", event_rows)
    report = {"data": "Real 20-tool prepared train data derived from 4-second sensors, aggregated to 1-minute rows",
              "hours": HOURS, "models": ["XGBoost 1%", "XGBoost 100%"],
              "training_fraction_definition": "The 1% condition uses one seed-42 uniform row sample of the common chronological training rows for all horizons; 100% uses every eligible training row. Neither condition samples validation.",
              "fixed_validation": "Same 1,142,112 rows, per-tool last 20%, with a 24h pre-cutoff purge; target changes with horizon.",
              "features": "Same 85 features as the recovered 24h baseline; no TTF or fault time in features.",
              "threshold": 0.5, "random_state": SEED,
              "historical_hgb_not_rerun": "The current project's scikit-learn histogram DLL fails to load, including outside sandbox; HGB 1%/100% from the old report is not recreated.",
              "xgboost_parameters": PARAMETERS,
              "historical_report_not_exactly_reproduced": True,
              "note": "The old 3h/6h workbook used different horizon-specific validation populations and its original experiment code for 12h/24h is unavailable. These matched new results must not be presented as its recovered historical scores.",
              "limitations": "Validation is not an untouched final test. No TTF regression models were retrained; row-level AP/F1 across horizons have different positive prevalence. Event recall and false alarm must be read together.",
              "results": results}
    (OUT / "comparison_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} comparison rows to {OUT}", flush=True)


if __name__ == "__main__":
    main()
