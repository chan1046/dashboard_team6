"""Full-data HGB-log TTF regression on exactly the existing LSTM eligible rows."""

from __future__ import annotations

import csv
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from compare_ttf_lstm_100pct import load_tool, metric
from recover_xgb24_baseline import FAULTS, ROOT

DATA = ROOT / "output" / "baseline_recovery"
OUT = ROOT / "output" / "experiments" / "hgb_log_ttf_matched"
MODELS = ROOT / "output" / "models"
PARAMS = {"max_iter": 120, "max_leaf_nodes": 31, "learning_rate": .05,
          "l2_regularization": 1.0, "early_stopping": False, "random_state": 42}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    report = json.loads((DATA / "reproduction_report.json").read_text(encoding="utf-8"))
    prep = json.loads((DATA / "preparation_report.json").read_text(encoding="utf-8"))
    features = report["features"]
    cutoffs = {row["tool"]: row["split_cutoff_time"] for row in prep["tools"]}
    paths = sorted((DATA / "prepared").glob("??_M??.parquet"))
    faults = pd.read_csv(ROOT / "data" / "processed" / "fault_events.csv",
                         usecols=["tool_id", "fault_type", "time"])
    if len(paths) != 20:
        raise AssertionError("Expected 20 prepared real-data tools")
    OUT.mkdir(parents=True)
    rows = []
    for fault in FAULTS:
        train_x, train_y = [], []
        for path in paths:
            x, target, train, valid, starts = load_tool(path, features, faults, cutoffs[path.stem], fault)
            train_x.append(x[train])
            train_y.append(np.log1p(target[train]))
            print(f"{fault} loaded {path.stem}: train={len(train):,} valid={len(valid):,}", flush=True)
        x = np.concatenate(train_x)
        y = np.concatenate(train_y)
        n_train = len(y)
        del train_x, train_y
        model = HistGradientBoostingRegressor(**PARAMS)
        print(f"fitting HGB-log {fault}: {len(y):,} full training rows", flush=True)
        model.fit(x, y)
        model_path = MODELS / f"hgb_log_ttf_{fault}_matched_full.joblib"
        if model_path.exists():
            raise FileExistsError(model_path)
        joblib.dump({"model": model, "features": features, "target_transform": "log1p seconds",
                     "fault_type": fault, "training_rows": len(y)}, model_path)
        del x, y
        truths, predictions = [], []
        for path in paths:
            x, target, train, valid, starts = load_tool(path, features, faults, cutoffs[path.stem], fault)
            if len(valid):
                truth = target[valid]
                prediction = np.maximum(0, np.expm1(model.predict(x[valid]).astype(np.float64)))
                truths.append(truth)
                predictions.append(prediction)
        truth, prediction = np.concatenate(truths), np.concatenate(predictions)
        result = {"model": "HGB-log", "fault_type": fault, "train_rows": n_train,
                  "validation_rows": len(truth), **metric(truth, prediction)}
        rows.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    with (OUT / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "report.json").write_text(json.dumps({"design": "same 20 tools, 85 features, full eligible fault-specific train rows, fixed time split and validation as prior LSTM",
        "target": "log1p(TTF seconds), inverse expm1 clipped at zero",
        "regression_train_not_filtered_by_classifier": True,
        "parameters": PARAMS, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
