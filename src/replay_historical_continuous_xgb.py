"""Re-run the exact archived continuous-session 3/6/12/24h XGBoost labels.

The deleted pipeline source is recovered from Codex FileChange history. Its
HGB import is omitted solely because Windows blocks that extension; the
original preparation and label functions are executed unchanged. XGBoost
uses the archived parameter values on all eligible training rows.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
RECOVERY = ROOT / "output" / "experiments" / "historical_exact_replay"
OUT = RECOVERY / "reproduced"
SOURCE_CODE = RECOVERY / "pipeline_continuous_3_6_12_24.py"
MODELS = ROOT / "output" / "models"
HORIZONS = (10800, 21600, 43200, 86400)
EXPECTED = {
    10800: {"validation_rows": 38435, "positive_rows": 3661, "precision": 0.17479441575827118, "recall": 0.24965856323408905, "average_precision": 0.21024880612364089, "detected_events": 50},
    21600: {"validation_rows": 17812, "positive_rows": 3679, "precision": 0.5010771992818671, "recall": 0.7586300625169883, "average_precision": 0.4974422179244688, "detected_events": 74},
    43200: {"validation_rows": 10395, "positive_rows": 3679, "precision": 0.629006266786034, "recall": 0.9548790432182659, "average_precision": 0.7686077125433828, "detected_events": 79},
    86400: {"validation_rows": 5981, "positive_rows": 3679, "precision": 0.7335361877852641, "recall": 0.9173688502310411, "average_precision": 0.9175233757668764, "detected_events": 75},
}


def load_archived_pipeline():
    source = SOURCE_CODE.read_text(encoding="utf-8")
    old_import = "from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor\n"
    if source.count(old_import) != 1:
        raise AssertionError("Archived HGB import changed")
    source = source.replace(old_import, "# HGB import omitted for XGBoost-only replay on this Windows host\n")
    module = types.ModuleType("archived_continuous_pipeline")
    module.__file__ = str(SOURCE_CODE)
    sys.modules[module.__name__] = module
    exec(compile(source, str(SOURCE_CODE), "exec"), module.__dict__)
    return module


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    module = load_archived_pipeline()
    args = SimpleNamespace(source=ROOT / "data" / "processed", output=OUT,
        horizons=list(HORIZONS), reuse_minute_from=None, tools=[],
        stop_gap_seconds=120, train_ratio=.8, seed=42, force_prepare=False)
    if not (OUT / "preparation_report.json").exists():
        module.prepare(args)
    prep = json.loads((OUT / "preparation_report.json").read_text(encoding="utf-8"))
    if prep["raw_rows_total"] != 82189440 or prep["minute_rows_total"] != 5777042:
        raise AssertionError("Actual source/preparation population differs from archived run")
    features = prep["features"]
    MODELS.mkdir(parents=True, exist_ok=True)
    results = []
    files = [OUT / "prepared" / f"{tool}.parquet" for tool in prep["tools"]]
    for h in HORIZONS:
        x_train_parts, y_train_parts, valid_parts = [], [], []
        needed = [*features, "time", "tool_id", "split", f"label_h{h}", f"event_class_h{h}",
                  *[f"event_time_{fault}" for fault in module.FAULTS]]
        for file in files:
            frame = pd.read_parquet(file, columns=needed)
            label = frame[f"label_h{h}"].to_numpy(dtype=np.int8)
            split = frame.split.to_numpy(dtype=np.int8)
            train = (split == 0) & (label >= 0)
            valid = (split == 2) & (label >= 0)
            x_train_parts.append(frame.loc[train, features].to_numpy(dtype=np.float32, copy=True))
            y_train_parts.append(label[train])
            valid_parts.append(frame.loc[valid, needed].copy())
            del frame
        x_train, y_train = np.concatenate(x_train_parts), np.concatenate(y_train_parts)
        validation = pd.concat(valid_parts, ignore_index=True)
        truth = validation[f"label_h{h}"].to_numpy(dtype=np.int8)
        counts = np.bincount(y_train, minlength=2)
        if min(counts) == 0:
            raise AssertionError(f"No positive or negative train rows for {h}")
        model = XGBClassifier(n_estimators=120, max_depth=5, learning_rate=.07,
            subsample=.85, colsample_bytree=.85, tree_method="hist",
            n_jobs=min(12, os.cpu_count() or 4), scale_pos_weight=float(counts[0] / counts[1]),
            eval_metric="logloss", random_state=42)
        print(f"{h}: training {len(y_train):,} full eligible rows, positive={counts[1]:,}", flush=True)
        model.fit(x_train, y_train)
        score = model.predict_proba(validation[features].to_numpy(dtype=np.float32))[:, 1]
        prediction = (score >= .5).astype(np.int8)
        event = module._binary_event_metrics(validation, np.arange(len(validation)), truth,
            prediction, h, validation[f"event_class_h{h}"].to_numpy(dtype=np.int8))
        tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
        result = {"horizon_hours": h // 3600, "horizon_seconds": h,
            "train_rows": len(y_train), "train_positive_rows": int(counts[1]),
            "validation_rows": len(truth), "positive_rows": int(truth.sum()),
            "precision": float(precision_score(truth, prediction, zero_division=0)),
            "recall": float(recall_score(truth, prediction, zero_division=0)),
            "f1": float(f1_score(truth, prediction, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "average_precision": float(average_precision_score(truth, score)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            **event}
        result["matches_archived"] = all(np.isclose(result[key], value, atol=1e-9)
                                         for key, value in EXPECTED[h].items())
        results.append(result)
        save_csv(OUT / "xgb_replay_metrics.csv", results)
        (OUT / "xgb_replay_report.json").write_text(json.dumps(json_safe({
            "archived_source": str(SOURCE_CODE),
            "archived_metric_source": "Codex session rollout-2026-09-22T17-41-36-01a0c847-133e-7a52-8217-d512e5355fe0.jsonl, line 5067",
            "status": "XGBoost replay of archived continuous-session selection; HGB omitted due blocked native module",
            "data": "20 actual train partitions; all 82,189,440 four-second rows contribute to 1-minute preparation",
            "label": "same-session confirmed fault within horizon; rows without complete same-session follow-up censored",
            "split": "per-tool chronological 80/20 with 24-hour pre-cutoff purge",
            "threshold": .5, "random_state": 42, "features": features,
            "xgb_parameters": model.get_params(), "results": results,
            "interpretation": "High recall applies only to the small uncensored continuous-session subset; not wall-clock 12/24h operational performance.",
        }), ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
        model_path = MODELS / f"xgb_continuous_replay_{h//3600}h.joblib"
        if model_path.exists():
            raise FileExistsError(f"Will not overwrite {model_path}")
        joblib.dump(model, model_path)
        print(f"{h}: recall={result['recall']:.4f}, F1={result['f1']:.4f}, AP={result['average_precision']:.4f}, events={result['detected_events']}/{result['events']}, archived_match={result['matches_archived']}", flush=True)
    if len(results) != 4 or not all(r["matches_archived"] for r in results):
        raise AssertionError("One or more archived XGBoost results did not reproduce")


if __name__ == "__main__":
    main()
