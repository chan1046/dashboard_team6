"""Conditional 0–6h vs 6–24h fault imminence on frozen causal features."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from experiment_ttf_interval_model import (BASE_FEATURES, FAULTS, HGB_PARAMS,
                                           PLAN, ROOT, model_samples)

OUT = ROOT / "output/experiments/ttf_imminent_6h"
THRESHOLD = 0.5
PURGE_SECONDS = 86400


def scores(y, probability, weight=None):
    y = np.asarray(y, dtype=np.uint8)
    probability = np.asarray(probability, dtype=float)
    pred = (probability >= THRESHOLD).astype(np.uint8)
    result = {"rows": len(y), "positive_rows": int(y.sum()),
              "roc_auc": np.nan, "pr_auc_ap": np.nan,
              "balanced_accuracy": np.nan, "precision": np.nan,
              "recall_0_6h": np.nan, "f1": np.nan,
              "tn": np.nan, "fp": np.nan, "fn": np.nan, "tp": np.nan}
    if not len(y):
        return result
    matrix = confusion_matrix(y, pred, labels=[0, 1], sample_weight=weight).ravel()
    result.update(dict(zip(("tn", "fp", "fn", "tp"), map(float, matrix))))
    result["precision"] = float(precision_score(y, pred, sample_weight=weight, zero_division=0))
    result["recall_0_6h"] = float(recall_score(y, pred, sample_weight=weight, zero_division=0))
    result["f1"] = float(f1_score(y, pred, sample_weight=weight, zero_division=0))
    if len(np.unique(y)) == 2:
        result["roc_auc"] = float(roc_auc_score(y, probability, sample_weight=weight))
        result["pr_auc_ap"] = float(average_precision_score(y, probability, sample_weight=weight))
        result["balanced_accuracy"] = float(balanced_accuracy_score(y, pred, sample_weight=weight))
    return result


def event_weights(frame):
    return (1 / frame.groupby("event_id").event_id.transform("size")).to_numpy(float)


def fold_assign(frame):
    """Three fixed chronological validation slices with 24h event-safe gaps."""
    times = np.sort(frame.time.unique())
    boundaries = [int(times[int(len(times) * fraction)]) for fraction in (1/3, 2/3)]
    t = frame.time.to_numpy(np.int64)
    f = frame.fault_time.to_numpy(np.int64)
    fold = np.zeros(len(frame), dtype=np.int8)
    fold[(t < boundaries[0] - PURGE_SECONDS) & (f < boundaries[0] - PURGE_SECONDS)] = 1
    fold[(t >= boundaries[0] + PURGE_SECONDS) & (f < boundaries[1] - PURGE_SECONDS)] = 2
    fold[(t >= boundaries[1] + PURGE_SECONDS) & (f >= boundaries[1] + PURGE_SECONDS)] = 3
    return fold, boundaries


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment: {OUT}")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if len(plan["tools"]) != 20 or len(BASE_FEATURES) != 85:
        raise AssertionError("Frozen 20-tool, 85-feature plan changed")
    forbidden = ("ttf", "fault", "next_event", "label", "absolute_time", "lot", "runnum")
    if any(any(word in name.lower() for word in forbidden) for name in BASE_FEATURES):
        raise AssertionError("A target or future feature is present")
    faults = pd.read_csv(FAULTS).sort_values(["tool_id", "time", "fault_type", "fault_event_id"])
    faults = faults.drop_duplicates(["tool_id", "time", "fault_type"], keep="first")
    if faults.duplicated(["tool_id", "time"]).any():
        raise AssertionError("Simultaneous fault types need an explicit assignment policy")
    data = model_samples(faults, plan)
    data["imminent"] = (data.ttf_hours <= 6).astype(np.uint8)
    if not ((data.ttf_hours > 0) & (data.ttf_hours <= 24)).all():
        raise AssertionError("Out-of-scope TTF in sample")
    train = data[data.split == "train"].copy()
    valid = data[data.split == "validation"].copy()
    if set(train.event_id) & set(valid.event_id):
        raise AssertionError("Fault event leaked across train/validation")
    population = data.groupby(["split", "imminent"]).agg(
        rows=("event_id", "size"), unique_fault_events=("event_id", "nunique")
    ).reset_index()
    xtrain = train[BASE_FEATURES].to_numpy(np.float32)
    xvalid = valid[BASE_FEATURES].to_numpy(np.float32)
    ytrain = train.imminent.to_numpy(np.uint8)
    event_train_weight = event_weights(train).copy()
    event_train_weight *= len(event_train_weight) / event_train_weight.sum()
    valid_weight = event_weights(valid)
    valid["validation_fold"], boundaries = fold_assign(valid)
    crossfold = valid.loc[valid.validation_fold > 0].groupby("event_id").validation_fold.nunique()
    if (crossfold > 1).any():
        raise AssertionError("Fault event crossed validation folds")
    metrics = []
    tool_metrics = []
    fold_metrics = []
    event_rows = []
    for model_name, sample_weight in (("A_row", None), ("B_event_balanced", event_train_weight)):
        model = HistGradientBoostingClassifier(**HGB_PARAMS)
        model.fit(xtrain, ytrain, sample_weight=sample_weight)
        probability = model.predict_proba(xvalid)[:, 1]
        valid[f"probability_{model_name}"] = probability
        row_metric = scores(valid.imminent, probability)
        equal_metric = scores(valid.imminent, probability, valid_weight)
        metrics.append({"model": model_name, "evaluation": "row", "threshold": THRESHOLD,
                        "validation_events": valid.event_id.nunique(), **row_metric})
        metrics.append({"model": model_name, "evaluation": "event_equal_weight_rows", "threshold": THRESHOLD,
                        "validation_events": valid.event_id.nunique(), **equal_metric})
        for tool, part in valid.groupby("tool_id", sort=True):
            ix = valid.index.get_indexer(part.index)
            for evaluation, w in (("row", None), ("event_equal_weight_rows", event_weights(part))):
                tool_metrics.append({"model": model_name, "tool_id": tool, "evaluation": evaluation,
                                     "events": part.event_id.nunique(),
                                     **scores(part.imminent, probability[ix], w)})
        for fold in (1, 2, 3):
            part = valid[valid.validation_fold == fold]
            ix = valid.index.get_indexer(part.index)
            for evaluation, w in (("row", None), ("event_equal_weight_rows", event_weights(part))):
                fold_metrics.append({"model": model_name, "fold": fold, "evaluation": evaluation,
                                     "events": part.event_id.nunique(),
                                     "start_time": int(part.time.min()) if len(part) else None,
                                     "end_time": int(part.time.max()) if len(part) else None,
                                     **scores(part.imminent, probability[ix], w)})
        for event_id, part in valid.groupby("event_id", sort=False):
            ix = valid.index.get_indexer(part.index)
            imminent = part.imminent.to_numpy(bool)
            alarm = probability[ix] >= THRESHOLD
            event_rows.append({"model": model_name, "event_id": event_id,
                               "tool_id": part.tool_id.iloc[0], "fault_time": int(part.fault_time.iloc[0]),
                               "fault_type": part.fault_type.iloc[0],
                               "imminent_rows": int(imminent.sum()), "earlier_rows": int((~imminent).sum()),
                               "detected_within_6h": bool(np.any(alarm & imminent)),
                               "alarm_in_6_to_24h": bool(np.any(alarm & ~imminent))})
        print(model_name, json.dumps({"row": row_metric, "event_equal": equal_metric}), flush=True)
    OUT.mkdir(parents=True)
    population.to_csv(OUT / "population.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(tool_metrics).to_csv(OUT / "tool_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_metrics).to_csv(OUT / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(event_rows).to_csv(OUT / "event_detection.csv", index=False, encoding="utf-8-sig")
    valid[["tool_id", "time", "fault_time", "fault_type", "event_id", "ttf_hours", "imminent",
           "validation_fold", "probability_A_row", "probability_B_event_balanced"]].to_csv(
        OUT / "validation_predictions.csv.gz", index=False, compression="gzip")
    report = {"target": "1 if 0 < actual next-fault TTF <=6h, else 0 if 6h<TTF<=24h",
              "selection_caveat": "This is conditional on actual TTF<=24h; true TTF cannot be used to gate Model 2 in operation. End-to-end Model 1+2 performance is not evaluated.",
              "split": "Frozen Model A per-tool chronological train/validation; row and next fault are within the same split; reserved test untouched.",
              "validation_folds": "Three globally chronological slices of the same validation, with 24h internal gaps and each event assigned to at most one slice. Models are not refitted per slice; folds are stability checks, not independent tests.",
              "fold_boundaries_unix_seconds": boundaries,
              "fold_excluded_rows": int((valid.validation_fold == 0).sum()),
              "event_weighting": "Each train event has equal total sample weight. For event-equal evaluation each validation event has total row weight 1. A single event can have both target classes; this is an equal-event-weight row metric, not one binary label per fault.",
              "event_detection": "A fault is detected when any positive prediction occurs in its observed 0-6h window; distinct from event-equal-weight Balanced Accuracy.",
              "features": "Exactly frozen 85 causal Model A features; no TTF, fault timestamp, future/post-fault sensors, or feature changes.",
              "model": "HistGradientBoostingClassifier; identical parameters for A/B; only train sample_weight differs.",
              "parameters": HGB_PARAMS, "threshold": THRESHOLD,
              "train_rows": len(train), "validation_rows": len(valid),
              "train_events": train.event_id.nunique(), "validation_events": valid.event_id.nunique(),
              "class_count_note": "A fault event may contribute rows to both 0-6h and 6-24h; per-class unique event counts therefore need not sum to total events."}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
