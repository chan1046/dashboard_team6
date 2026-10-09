"""Fit full-data HGB on the frozen XGBoost 12h/24h validation design."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from compare_12h_24h_xgboost import confusion, event_audit, false_episodes
from evaluate_event_thresholds_24h import BASELINE, ROOT, SOURCE, load_events, read_json

OUT = ROOT / "output" / "experiments" / "hgb_xgb_matched_12h_24h"
MODEL_DIR = ROOT / "output" / "models"
PARAMS = {"max_iter": 120, "max_leaf_nodes": 31, "learning_rate": .05,
          "l2_regularization": 1.0, "early_stopping": False, "random_state": 42}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    base = read_json(BASELINE / "reproduction_report.json")
    sweep = read_json(SOURCE / "sweep_report.json")
    features = base["features"]
    paths = sorted((BASELINE / "prepared").glob("??_M??.parquet"))
    if len(paths) != 20 or len(features) != 85:
        raise AssertionError("Prepared split/features differ from frozen baseline")
    tool_map = {int(k): v for k, v in sweep["tool_code_map"].items()}
    cutoffs = {r["tool"]: r["split_cutoff_time"] for r in read_json(BASELINE / "preparation_report.json")["tools"]}
    events = load_events(cutoffs, set(tool_map.values()))
    train_x, train_y12, train_y24 = [], [], []
    for path in paths:
        frame = pd.read_parquet(path, columns=[*features, "time", "split", "label", "next_event_time"])
        part = frame.loc[(frame["split"] == 0) & (frame["label"] >= 0)]
        train_x.append(part[features].to_numpy(dtype=np.float32, copy=True))
        train_y24.append(part["label"].to_numpy(dtype=np.uint8, copy=True))
        train_y12.append(((part["next_event_time"] - part["time"]) <= 43200).to_numpy(dtype=np.uint8))
        print(f"loaded {path.stem}: {len(part)} train rows", flush=True)
    x = np.concatenate(train_x)
    y12, y24 = np.concatenate(train_y12), np.concatenate(train_y24)
    del train_x, train_y12, train_y24
    OUT.mkdir(parents=True)
    rows = []
    for hours, truth in ((12, y12), (24, y24)):
        weight = (len(truth) - int(truth.sum())) / int(truth.sum())
        sample_weight = np.where(truth == 1, weight, 1).astype(np.float32)
        model = HistGradientBoostingClassifier(**PARAMS)
        print(f"fitting HGB {hours}h: {len(truth)} full training rows", flush=True)
        model.fit(x, truth, sample_weight=sample_weight)
        model_path = MODEL_DIR / f"hgb_{hours}h_matched_full.joblib"
        if model_path.exists():
            raise FileExistsError(model_path)
        joblib.dump({"model": model, "features": features, "hours": hours,
                     "training_rows": len(truth), "positive_weight": weight}, model_path)
        score_parts, label_parts, time_parts, tool_parts = [], [], [], []
        for code, path in enumerate(paths):
            frame = pd.read_parquet(path, columns=[*features, "time", "split", "label", "next_event_time"])
            part = frame.loc[(frame["split"] == 2) & (frame["label"] >= 0)]
            label = (part["next_event_time"] - part["time"] <= hours * 3600).to_numpy(dtype=np.uint8)
            label_parts.append(label)
            score_parts.append(model.predict_proba(part[features].to_numpy(dtype=np.float32, copy=True))[:, 1].astype(np.float32))
            time_parts.append(part["time"].to_numpy(dtype=np.int64, copy=True))
            tool_parts.append(np.full(len(part), code, dtype=np.uint8))
        label, score = np.concatenate(label_parts), np.concatenate(score_parts)
        time, tool = np.concatenate(time_parts), np.concatenate(tool_parts)
        metrics = confusion(label, score)
        event, _ = event_audit(events, hours * 3600, score, time, tool,
                               {v: k for k, v in tool_map.items()})
        false, _ = false_episodes(label, score, time, tool, tool_map)
        row = {"model": "HGB", "horizon_hours": hours, "threshold": .5,
               "train_rows": len(truth), "train_positive_rows": int(truth.sum()),
               "positive_weight": weight, **metrics, **event,
               "false_alarm_events_10m": false,
               "false_alarm_events_per_observed_tool_12h": false / (len(label) / 720)}
        rows.append(row)
        np.savez_compressed(OUT / f"validation_probabilities_{hours}h.npz",
                            truth=label, probability=score, time=time, tool_code=tool)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    with (OUT / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "report.json").write_text(json.dumps({"design": "same 20 tools, 85 features, full 80% chronological train, fixed validation and purge as XGBoost",
        "parameters": PARAMS, "early_stopping_uses_validation": False,
        "threshold": .5, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
