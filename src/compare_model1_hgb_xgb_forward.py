"""Fair HGB/XGBoost comparison on the frozen Model-1 forward split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

from evaluate_event_thresholds_24h import event_metrics, row_and_alarm_metrics
from forward_validation_ab_24h import validation_events

ROOT = Path(__file__).resolve().parents[1]
FORWARD = ROOT / "output/experiments/forward_validation_ab_24h"
BASELINE = ROOT / "output/baseline_recovery"
XGB_PATH = ROOT / "output/models/xgb_24h_forward_a.joblib"
OUT = ROOT / "output/experiments/model1_hgb_vs_xgb"
HGB_MODEL = OUT / "hgb_forward_same_split.joblib"
HGB_SCORE = OUT / "hgb_validation_probabilities.npz"
XGB_SCORE = FORWARD / "validation_probabilities_a.npz"
HGB_PARAMS = {"max_iter": 120, "max_leaf_nodes": 31, "learning_rate": .05,
              "l2_regularization": 1.0, "early_stopping": False,
              "random_state": 42}


def load_plan():
    return json.loads((FORWARD / "split_plan.json").read_text(encoding="utf-8"))


def load_artifact():
    payload = joblib.load(XGB_PATH)
    if len(payload["features"]) != 85:
        raise AssertionError("Frozen XGBoost feature count changed")
    return payload


def paths(plan):
    result = [BASELINE / "prepared" / f"{r['tool_id']}.parquet" for r in plan["tools"]]
    if not all(p.exists() for p in result):
        raise FileNotFoundError("Prepared Model-1 artifacts missing")
    return result


def train():
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    plan, payload = load_plan(), load_artifact()
    features = payload["features"]
    expected = sum(r["train_labeled_rows"] for r in plan["tools"])
    x = np.empty((expected, len(features)), dtype=np.float32)
    y = np.empty(expected, dtype=np.uint8)
    cursor = 0
    for row, path in zip(plan["tools"], paths(plan)):
        frame = pd.read_parquet(path, columns=[*features, "time", "label"])
        mask = (frame.time.to_numpy(np.int64) < row["train_end_exclusive_second"]) & (frame.label.to_numpy(np.int8) >= 0)
        part = frame.loc[mask]
        stop = cursor + len(part)
        x[cursor:stop] = part[features].to_numpy(np.float32, copy=True)
        y[cursor:stop] = part.label.to_numpy(np.uint8)
        cursor = stop
        print(f"train load {path.stem}: {len(part):,}", flush=True)
    if cursor != expected:
        raise AssertionError("Train row count drift")
    positive_weight = int((y == 0).sum()) / int((y == 1).sum())
    sample_weight = np.where(y == 1, positive_weight, 1.0).astype(np.float32)
    model = HistGradientBoostingClassifier(**HGB_PARAMS)
    print(f"fit HGB rows={len(y):,}, positive={int(y.sum()):,}, weight={positive_weight:.9f}", flush=True)
    model.fit(x, y, sample_weight=sample_weight)
    del x, sample_weight
    joblib.dump({"model": model, "features": features, "parameters": HGB_PARAMS,
                 "positive_weight": positive_weight,
                 "split_plan": str((FORWARD / 'split_plan.json').relative_to(ROOT))}, HGB_MODEL)

    scores, truth, times, tools = [], [], [], []
    for code, (row, path) in enumerate(zip(plan["tools"], paths(plan))):
        frame = pd.read_parquet(path, columns=[*features, "time", "label"])
        time = frame.time.to_numpy(np.int64)
        label = frame.label.to_numpy(np.int8)
        mask = ((time >= row["validation_start_second"]) &
                (time < row["validation_end_exclusive_second"]) & (label >= 0))
        part = frame.loc[mask]
        scores.append(model.predict_proba(part[features].to_numpy(np.float32, copy=True))[:, 1].astype(np.float32))
        truth.append(part.label.to_numpy(np.uint8))
        times.append(part.time.to_numpy(np.int64))
        tools.append(np.full(len(part), code, dtype=np.uint8))
        print(f"validation score {path.stem}: {len(part):,}", flush=True)
    truth, scores, times, tools = map(np.concatenate, (truth, scores, times, tools))
    if (len(truth), int(truth.sum())) != (1114137, 91664):
        raise AssertionError("Validation population drift")
    np.savez_compressed(HGB_SCORE, truth=truth, probability=scores, time=times, tool_code=tools)
    (OUT / "hgb_training_metadata.json").write_text(json.dumps({
        "parameters": HGB_PARAMS, "train_rows": len(y), "train_positive": int(y.sum()),
        "train_negative": int(len(y)-y.sum()), "positive_weight": positive_weight,
        "validation_rows": len(truth), "validation_positive": int(truth.sum()),
        "feature_count": len(features), "features": features,
        "model_file": HGB_MODEL.name, "score_file": HGB_SCORE.name}, indent=2), encoding="utf-8")


def core_metrics(name, truth, score, threshold):
    pred = score >= threshold
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(truth, pred, labels=[0, 1]).ravel())
    specificity = tn / (tn + fp)
    return {"model": name, "threshold": threshold, "validation_rows": len(truth),
            "positive_rows": int(truth.sum()), "positive_ratio": float(truth.mean()),
            "feature_count": 85, "average_precision": float(average_precision_score(truth, score)),
            "roc_auc": float(roc_auc_score(truth, score)),
            "accuracy": float(accuracy_score(truth, pred)),
            "precision": float(precision_score(truth, pred, zero_division=0)),
            "recall": float(recall_score(truth, pred)), "f1": float(f1_score(truth, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
            "specificity": specificity, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def class_distribution(plan):
    rows = []
    for row, path in zip(plan["tools"], paths(plan)):
        frame = pd.read_parquet(path, columns=["time", "label"])
        t, y = frame.time.to_numpy(np.int64), frame.label.to_numpy(np.int8)
        masks = {"train": (t < row["train_end_exclusive_second"]) & (y >= 0),
                 "validation": (t >= row["validation_start_second"]) & (t < row["validation_end_exclusive_second"]) & (y >= 0),
                 "test": (t >= row["test_start_second"]) & (y >= 0)}
        for split, mask in masks.items():
            rows.append({"tool_id": row["tool_id"], "split": split, "total": int(mask.sum()),
                         "normal": int(((y == 0) & mask).sum()),
                         "positive": int(((y == 1) & mask).sum())})
    return pd.DataFrame(rows)


def evaluate():
    if not HGB_SCORE.exists():
        raise FileNotFoundError("Run train first")
    plan, payload = load_plan(), load_artifact()
    with np.load(HGB_SCORE) as h, np.load(XGB_SCORE) as x:
        ht, hs, htime, htool = (h[k].copy() for k in ("truth", "probability", "time", "tool_code"))
        xt, xs, xtime, xtool = (x[k].copy() for k in ("truth", "probability", "time", "tool_code"))
    if not all(np.array_equal(a, b) for a, b in ((ht, xt), (htime, xtime), (htool, xtool))):
        raise AssertionError("HGB and XGBoost validation rows differ")
    names = {i: row["tool_id"] for i, row in enumerate(plan["tools"])}
    reverse = {v: k for k, v in names.items()}
    events = validation_events(plan)
    comparisons = []
    for name, score, threshold in (("HGB", hs, .5), ("XGBoost", xs, .5), ("XGBoost_operating", xs, .34)):
        row = core_metrics(name, ht, score, threshold)
        alarm_row, _ = row_and_alarm_metrics(threshold, ht, score, htime, htool,
                                              row["average_precision"], names)
        event, _ = event_metrics(threshold, events, score, htime, htool, reverse)
        row.update({"raw_false_alarm_events": alarm_row["false_alarm_events_10m"], **event})
        comparisons.append(row)
    pd.DataFrame(comparisons).to_csv(OUT / "comparison_metrics.csv", index=False, encoding="utf-8-sig")
    for name, filename in (("HGB", "hgb_metrics.json"), ("XGBoost", "xgb_metrics_recheck.json")):
        chosen = [r for r in comparisons if r["model"] == name or (name == "XGBoost" and r["model"].startswith("XGBoost"))]
        (OUT / filename).write_text(json.dumps(chosen, ensure_ascii=False, indent=2), encoding="utf-8")
    dist = class_distribution(plan)
    dist.to_csv(OUT / "class_distribution_by_tool.csv", index=False, encoding="utf-8-sig")
    totals = dist.groupby("split", as_index=False)[["total", "normal", "positive"]].sum()
    totals["positive_ratio"] = totals.positive / totals.total
    totals.to_csv(OUT / "class_distribution_summary.csv", index=False, encoding="utf-8-sig")
    old = json.loads((BASELINE / "reproduction_report.json").read_text(encoding="utf-8"))
    prior_hgb = json.loads((ROOT / "output/experiments/hgb_xgb_matched_12h_24h/report.json").read_text(encoding="utf-8"))
    report = {"same_validation_rows_verified": True, "same_feature_order_verified": payload["features"] == json.loads((FORWARD / "training_a.json").read_text())["feature_names"],
              "existing_hgb": prior_hgb, "presentation_numbers_model_identification": {
                  "precision_17_44_recall_49_30_f1_25_77_balanced_accuracy_67_34_ap_16_68":
                  "Recovered XGBoost, not HGB", "source": "output/baseline_recovery/reproduction_report.json"},
              "current_comparison": comparisons, "class_distribution": totals.to_dict("records"),
              "xgb_existing_ap_matches": bool(np.isclose(comparisons[1]["average_precision"], .4031968707287871)),
              "xgb_operating_event_matches_202_of_264": comparisons[2]["detected_fault_events"] == 202 and comparisons[2]["total_logged_validation_fault_events"] == 264}
    (OUT / "comparison_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["Model 1 HGB vs XGBoost — frozen forward validation comparison", ""]
    for r in comparisons:
        lines.append(f"{r['model']} threshold={r['threshold']}: AP={r['average_precision']:.6f}, ROC-AUC={r['roc_auc']:.6f}, precision={r['precision']:.6f}, recall={r['recall']:.6f}, F1={r['f1']:.6f}, BA={r['balanced_accuracy']:.6f}, events={r['detected_fault_events']}/{r['total_logged_validation_fault_events']}, false alarms={r['raw_false_alarm_events']}")
    lines.extend(["", "The 17.44% precision / 49.30% recall / 16.68% AP presentation baseline is the recovered XGBoost result, not a verified HGB result.",
                  "The old HGB artifact used a different 80/20 split; this experiment retrained HGB on the final forward split."])
    (OUT / "comparison_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"comparison": comparisons, "class_distribution": totals.to_dict("records")}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["train", "evaluate", "all"])
    args = parser.parse_args()
    if args.phase in ("train", "all"):
        train()
    if args.phase in ("evaluate", "all"):
        evaluate()
