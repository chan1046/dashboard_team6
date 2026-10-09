"""Train one fixed-split XGBoost comparator for the final Model 2 RF.

The comparator reuses the frozen partial-A anchor population, 72-feature order,
event balancing, and class balancing from full_fault_type_multi_anchor.py.
Existing Model 2 artifacts are read only; all new outputs use separate paths.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

from audit_full_fault_type_data import PLAN, ROOT, TYPES
from train_full_fault_type_baseline import FEATURES


SOURCE = ROOT / "output/experiments/full_fault_type_multi_anchor"
OUT = ROOT / "output/experiments/full_fault_type_xgboost_comparison"
MODEL_PATH = ROOT / "output/models/full_fault_type_multi_anchor_xgboost_partial_A_event_class.joblib"
MODEL_NAME = "partial_A_event_class"
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "n_jobs": 8,
    "random_state": 42,
}


def metrics(truth: np.ndarray, prediction: np.ndarray, score: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, prediction, labels=np.arange(len(TYPES)), average="macro", zero_division=0
    )
    binary_truth = label_binarize(truth, classes=np.arange(len(TYPES)))
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_ap_ovr": float(average_precision_score(binary_truth, score, average="macro")),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
    }


def aggregate_events(frame: pd.DataFrame) -> pd.DataFrame:
    score_columns = [f"score_{name}" for name in TYPES]
    rows = []
    for event_id, part in frame.groupby("fault_event_id", sort=False):
        scores = part[score_columns].mean().to_numpy(dtype=float)
        rows.append(
            {
                "fault_event_id": int(event_id),
                "tool_id": part.tool_id.iloc[0],
                "fault_type": part.fault_type.iloc[0],
                "prediction": TYPES[int(np.argmax(scores))],
                "anchors": len(part),
                **{column: float(scores[i]) for i, column in enumerate(score_columns)},
            }
        )
    return pd.DataFrame(rows)


def encoded(values: pd.Series) -> np.ndarray:
    lookup = {name: i for i, name in enumerate(TYPES)}
    result = values.map(lookup)
    if result.isna().any():
        raise AssertionError("Unexpected fault type")
    return result.to_numpy(dtype=np.int64)


def main() -> None:
    if OUT.exists() or MODEL_PATH.exists():
        raise FileExistsError("Comparator output already exists; refusing to overwrite")

    anchors = pd.read_csv(SOURCE / "anchor_samples.csv")
    subset = anchors[anchors.eligible_partial_A.astype(bool)].copy()
    train = subset[subset.phase.eq("train")]
    validation = subset[subset.phase.eq("validation")]
    if set(train.fault_event_id) & set(validation.fault_event_id):
        raise AssertionError("Train/validation event overlap")
    if len(FEATURES) != 72 or list(FEATURES) != json.loads(
        (SOURCE / "preparation_config.json").read_text(encoding="utf-8")
    )["features"]:
        raise AssertionError("Frozen feature contract mismatch")

    anchors_per_event = train.fault_event_id.value_counts()
    event_weight = train.fault_event_id.map(lambda value: 1.0 / anchors_per_event[value]).to_numpy()
    event_classes = train.drop_duplicates("fault_event_id").fault_type.value_counts()
    class_weight = {name: len(anchors_per_event) / (len(TYPES) * int(event_classes[name])) for name in TYPES}
    sample_weight = event_weight * train.fault_type.map(class_weight).to_numpy(dtype=float)

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(train[FEATURES].to_numpy(dtype=np.float32), encoded(train.fault_type), sample_weight=sample_weight)
    score = model.predict_proba(validation[FEATURES].to_numpy(dtype=np.float32))
    prediction = np.argmax(score, axis=1)

    xgb_rows = validation[["fault_event_id", "tool_id", "horizon", "horizon_minutes", "fault_type"]].copy()
    xgb_rows["predicted"] = [TYPES[i] for i in prediction]
    for i, name in enumerate(TYPES):
        xgb_rows[f"score_{name}"] = score[:, i]

    rf_rows = pd.read_csv(SOURCE / "validation_predictions.csv")
    rf_rows = rf_rows[rf_rows.model.eq(MODEL_NAME)].copy()
    expected_keys = set(zip(validation.fault_event_id, validation.horizon))
    actual_keys = set(zip(rf_rows.fault_event_id, rf_rows.horizon))
    if expected_keys != actual_keys or len(rf_rows) != len(validation):
        raise AssertionError("RF and XGBoost validation populations differ")

    summaries = []
    event_predictions = []
    for algorithm, rows in (("Random Forest", rf_rows), ("XGBoost", xgb_rows)):
        truth = encoded(rows.fault_type)
        guessed = encoded(rows.predicted)
        scores = rows[[f"score_{name}" for name in TYPES]].to_numpy(dtype=float)
        summaries.append({"algorithm": algorithm, "level": "sample", "samples": len(rows),
                          "unique_events": rows.fault_event_id.nunique(), **metrics(truth, guessed, scores)})

        events = aggregate_events(rows)
        event_truth = encoded(events.fault_type)
        event_guess = encoded(events.prediction)
        event_scores = events[[f"score_{name}" for name in TYPES]].to_numpy(dtype=float)
        summaries.append({"algorithm": algorithm, "level": "event_average_score", "samples": len(events),
                          "unique_events": len(events), **metrics(event_truth, event_guess, event_scores),
                          "correct_events": int((event_truth == event_guess).sum()),
                          "event_detection_rate": float((event_truth == event_guess).mean())})
        events.insert(0, "algorithm", algorithm)
        event_predictions.append(events)

    OUT.mkdir(parents=True)
    pd.DataFrame(summaries).to_csv(OUT / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    xgb_rows.to_csv(OUT / "xgboost_validation_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat(event_predictions, ignore_index=True).to_csv(
        OUT / "event_predictions.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "comparison_scope": "Final Model 2 partial_A event_class only",
        "features": list(FEATURES),
        "feature_count": len(FEATURES),
        "split": str(PLAN.relative_to(ROOT)),
        "weighting": "1/N anchors per event multiplied by inverse event-class frequency",
        "xgboost_parameters": XGB_PARAMS,
        "validation_is_development_not_independent_test": True,
        "ap_definition": "macro one-vs-rest average precision",
        "event_detection_definition": "correct fault-type classification after averaging class scores across event anchors",
    }
    (OUT / "experiment_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": list(FEATURES), "classes": list(TYPES),
                 "parameters": XGB_PARAMS, "class_weight_rule": class_weight,
                 "split": str(PLAN.relative_to(ROOT)), "score_calibrated": False}, MODEL_PATH)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
