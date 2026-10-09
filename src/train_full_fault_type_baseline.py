"""Event-level, strictly pre-fault 30-minute 3-class baseline; Model 1 untouched."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from xgboost import XGBClassifier

from audit_full_fault_type_data import (PLAN, RAW, ROOT, TYPES, MIN_OBSERVATIONS,
                                        MAX_ANCHOR_AGE_SECONDS, phase_of_fault)
from train_simple_full_fault_rul import SENSORS

AUDIT = ROOT / "output/experiments/full_fault_type_data_audit"
OUT = ROOT / "output/experiments/full_fault_type_baseline"
MODELS = ROOT / "output/models"
SENSOR_WINDOW_SECONDS = 1800
ANCHOR_BEFORE_FAULT_SECONDS = 1800
MIN_COVERAGE_SECONDS = 1440
STATISTICS = ("last", "mean", "std", "min", "max", "median", "slope_per_hour", "last_minus_first")
FEATURES = [f"{sensor}_{stat}" for sensor in SENSORS for stat in STATISTICS]
DISPLAY = {"pressure_low": "Low Pressure", "pressure_high": "High Pressure", "leak": "Leak"}


def snapshot(t: np.ndarray, x: np.ndarray, prediction: int, lower: int) -> tuple[dict | None, dict]:
    first = max(int(np.searchsorted(t, prediction - SENSOR_WINDOW_SECONDS, side="left")),
                int(np.searchsorted(t, lower, side="left")))
    end = int(np.searchsorted(t, prediction, side="right"))
    local_t = t[first:end]
    local_x = x[first:end]
    count = len(local_t)
    age = int(prediction - local_t[-1]) if count else None
    span = int(local_t[-1] - local_t[0]) if count else None
    gaps = np.diff(local_t)
    quality = {"sensor_row_count": count, "coverage_duration_seconds": span,
               "median_gap_seconds": float(np.median(gaps)) if len(gaps) else None,
               "max_gap_seconds": int(gaps.max()) if len(gaps) else None,
               "last_sensor_age_seconds": age,
               "first_sensor_timestamp": int(local_t[0]) if count else None,
               "last_sensor_timestamp": int(local_t[-1]) if count else None}
    if (count < MIN_OBSERVATIONS or span is None or span < MIN_COVERAGE_SECONDS
            or age is None or age > MAX_ANCHOR_AGE_SECONDS):
        return None, quality
    if not np.isfinite(local_x).all():
        quality["nonfinite_sensor_values"] = True
        return None, quality
    # Regression is on elapsed real time, so irregular sampling does not
    # silently turn an observation count into a time unit.
    elapsed = (local_t - local_t[0]).astype(np.float64)
    centered = elapsed - elapsed.mean()
    denominator = float(np.dot(centered, centered))
    features = {}
    for j, sensor in enumerate(SENSORS):
        values = local_x[:, j]
        centered_values = values - values.mean()
        slope = float(np.dot(centered, centered_values) / denominator * 3600) if denominator > 0 else 0.0
        features.update({f"{sensor}_last": float(values[-1]),
                         f"{sensor}_mean": float(values.mean()),
                         f"{sensor}_std": float(values.std(ddof=0)),
                         f"{sensor}_min": float(values.min()),
                         f"{sensor}_max": float(values.max()),
                         f"{sensor}_median": float(np.median(values)),
                         f"{sensor}_slope_per_hour": slope,
                         f"{sensor}_last_minus_first": float(values[-1] - values[0])})
    return features, quality


def metric_block(y: np.ndarray, prediction: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=TYPES, zero_division=0)
    return {"accuracy": float(accuracy_score(y, prediction)),
            "macro_precision": float(np.mean(precision)),
            "macro_recall": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            **{f"{name}_recall": float(recall[i]) for i, name in enumerate(TYPES)}}


def write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    for filename in ("full_fault_type_decision_tree.joblib", "full_fault_type_random_forest.joblib",
                     "full_fault_type_xgboost.joblib", "full_fault_type_best.joblib"):
        if (MODELS / filename).exists():
            raise FileExistsError(MODELS / filename)
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    audit = pd.read_csv(AUDIT / "event_availability.csv")
    faults = pd.read_csv(RAW / "fault_events.csv").drop_duplicates(["tool_id", "time", "fault_type"])
    assert len(faults) == 1211 and set(faults.fault_type) == set(TYPES)
    samples, eligibility = [], []
    for item in plan["tools"]:
        tool = item["tool_id"]
        table = pq.read_table(RAW / f"{tool}_sensor_ttf.parquet", columns=["time", *SENSORS])
        t = table.column("time").to_numpy(zero_copy_only=False).astype(np.int64)
        x = np.column_stack([table.column(s).to_numpy(zero_copy_only=False) for s in SENSORS]).astype(np.float64)
        log = faults[faults.tool_id.eq(tool)].sort_values("time", kind="stable")
        previous = None
        for event in log.itertuples():
            fault = int(event.time)
            phase = phase_of_fault(fault, item)
            if phase not in ("train", "validation"):
                previous = fault
                continue
            phase_start = int(t[0]) if phase == "train" else int(item["validation_start_second"])
            lower = max(phase_start, previous + 1 if previous is not None else int(t[0]))
            prediction = fault - ANCHOR_BEFORE_FAULT_SECONDS
            fields, quality = snapshot(t, x, prediction, lower)
            record = {"fault_event_id": int(event.fault_event_id), "tool_id": tool,
                      "phase": phase, "fault_type": event.fault_type,
                      "fault_timestamp": fault, "prediction_timestamp": prediction,
                      "history_start": prediction - SENSOR_WINDOW_SECONDS,
                      "phase_and_cycle_start": lower,
                      "eligible": fields is not None, **quality}
            eligibility.append(record)
            if fields is not None:
                samples.append({**record, **fields})
            previous = fault
        print("extracted", tool, flush=True)
    eligibility = pd.DataFrame(eligibility)
    samples = pd.DataFrame(samples)
    assert not samples.fault_event_id.duplicated().any()
    assert len(samples) and set(samples.fault_type) == set(TYPES)
    assert (samples.last_sensor_timestamp <= samples.prediction_timestamp).all()
    assert (samples.prediction_timestamp < samples.fault_timestamp).all()
    assert (samples.first_sensor_timestamp >= samples.history_start).all()
    assert (samples.first_sensor_timestamp >= samples.phase_and_cycle_start).all()
    train = samples[samples.phase.eq("train")].copy()
    valid = samples[samples.phase.eq("validation")].copy()
    assert not set(train.fault_event_id) & set(valid.fault_event_id)
    assert len(FEATURES) == 72 and not any("TTF" in f or "fault" in f.lower() or "tool" in f.lower() for f in FEATURES)
    matrix_train = train[FEATURES].to_numpy(np.float32)
    matrix_valid = valid[FEATURES].to_numpy(np.float32)
    assert np.isfinite(matrix_train).all() and np.isfinite(matrix_valid).all()
    y_train = train.fault_type.to_numpy()
    y_valid = valid.fault_type.to_numpy()
    # Audit's history_30m is at the final pre-fault sensor timestamp, not
    # at fault-30m. Compare both populations explicitly; do not force counts.
    historical = audit[audit.phase.isin(["train", "validation"])]
    audit_anchor = historical.groupby(["phase", "fault_type"])["horizon_30m_30m_history"].sum()
    actual = samples.groupby(["phase", "fault_type"]).size()
    population = []
    for phase in ("train", "validation"):
        for typ in TYPES:
            k = (phase, typ)
            population.append({"phase": phase, "fault_type": typ,
                               "audit_pre_fault_30m_history": int(historical[(historical.phase == phase) & (historical.fault_type == typ)].history_30m.sum()),
                               "audit_fault_minus_30m_anchor": int(audit_anchor.get(k, 0)),
                               "strict_prediction_window_usable": int(actual.get(k, 0))})
    class_counts = pd.Series(y_train).value_counts().reindex(TYPES, fill_value=0)
    weight_map = {name: len(y_train) / (len(TYPES) * count) for name, count in class_counts.items()}
    sample_weight = np.array([weight_map[y] for y in y_train], dtype=np.float64)
    models = [
        ("dummy", "default", DummyClassifier(strategy="most_frequent")),
        ("decision_tree", "default", DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=42)),
        ("decision_tree", "balanced", DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, class_weight="balanced", random_state=42)),
        ("random_forest", "default", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=2, n_jobs=8, random_state=42)),
        ("random_forest", "balanced", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=2, class_weight="balanced", n_jobs=8, random_state=42)),
        ("xgboost", "default", XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss", n_estimators=250, max_depth=3, learning_rate=.05, subsample=.9, colsample_bytree=.9, tree_method="hist", n_jobs=8, random_state=42)),
        ("xgboost", "balanced", XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss", n_estimators=250, max_depth=3, learning_rate=.05, subsample=.9, colsample_bytree=.9, tree_method="hist", n_jobs=8, random_state=42)),
    ]
    label_to_int = {v: i for i, v in enumerate(TYPES)}
    model_rows, class_rows, confusion_rows, confidence_rows = [], [], [], []
    tool_rows, exclusion_rows, importance_rows, prediction_rows = [], [], [], []
    fitted_models = {}
    for model_type, variant, model in models:
        name = f"{model_type}_{variant}"
        if model_type == "xgboost":
            fit_y = np.array([label_to_int[y] for y in y_train], dtype=np.int32)
            weight = sample_weight if variant == "balanced" else None
            model.fit(matrix_train, fit_y, sample_weight=weight)
            pred_id = model.predict(matrix_valid).astype(int)
            pred = np.array([TYPES[i] for i in pred_id])
            scores = model.predict_proba(matrix_valid)
        else:
            model.fit(matrix_train, y_train)
            pred = model.predict(matrix_valid)
            raw_scores = model.predict_proba(matrix_valid)
            scores = np.column_stack([raw_scores[:, list(model.classes_).index(c)] for c in TYPES])
        fitted_models[name] = model
        block = metric_block(y_valid, pred)
        model_rows.append({"model": name, "train_events": len(train), "validation_events": len(valid),
                           "features": len(FEATURES), **block})
        p, r, f, support = precision_recall_fscore_support(y_valid, pred, labels=TYPES, zero_division=0)
        for i, typ in enumerate(TYPES):
            class_rows.append({"model": name, "fault_type": typ, "precision": float(p[i]),
                               "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])})
        cm = confusion_matrix(y_valid, pred, labels=TYPES)
        for i, actual_type in enumerate(TYPES):
            for j, predicted_type in enumerate(TYPES):
                confusion_rows.append({"model": name, "actual": actual_type,
                                       "predicted": predicted_type, "events": int(cm[i, j])})
        maximum = scores.max(axis=1)
        for threshold in (.5, .6, .7, .8):
            selected = maximum >= threshold
            shown = pred[selected]
            truth = y_valid[selected]
            row = {"model": name, "score_threshold": threshold,
                   "validation_events": len(valid), "shown_events": int(selected.sum()),
                   "withheld_events": int((~selected).sum()),
                   "coverage": float(selected.mean()),
                   "shown_correct_rate": float(np.mean(shown == truth)) if len(shown) else np.nan,
                   "shown_incorrect": int(np.sum(shown != truth))}
            for typ in TYPES:
                row[f"{typ}_shown"] = int(np.sum(shown == typ))
                row[f"{typ}_shown_correct"] = int(np.sum((shown == typ) & (truth == typ)))
            confidence_rows.append(row)
        prediction_rows.extend({"model": name, "fault_event_id": int(event),
                                "tool_id": tool, "actual": actual_type, "predicted": guess,
                                "max_class_score": float(score.max()),
                                **{f"score_{typ}": float(score[i]) for i, typ in enumerate(TYPES)}}
                               for event, tool, actual_type, guess, score in zip(
                                   valid.fault_event_id, valid.tool_id, y_valid, pred, scores))
        for tool in sorted(valid.tool_id.unique()):
            mask = valid.tool_id.to_numpy() == tool
            row = {"model": name, "tool_id": tool, "events": int(mask.sum()),
                   "correct": int(np.sum(y_valid[mask] == pred[mask])),
                   "accuracy": float(np.mean(y_valid[mask] == pred[mask]))}
            for typ in TYPES:
                row[f"actual_{typ}"] = int(np.sum(y_valid[mask] == typ))
                row[f"predicted_{typ}"] = int(np.sum(pred[mask] == typ))
            tool_rows.append(row)
        for excluded in ("03_M02", "09_M02", "06_M02"):
            mask = valid.tool_id.to_numpy() != excluded
            exclusion_rows.append({"model": name, "excluded_tool": excluded,
                                   "events": int(mask.sum()), **metric_block(y_valid[mask], pred[mask])})
        if hasattr(model, "feature_importances_"):
            ranks = np.argsort(model.feature_importances_)[::-1][:20]
            importance_rows.extend({"model": name, "rank": rank + 1, "feature": FEATURES[j],
                                    "sensor": next(s for s in SENSORS if FEATURES[j].startswith(s + "_")),
                                    "importance": float(model.feature_importances_[j])}
                                   for rank, j in enumerate(ranks))
        print(name, json.dumps(block), flush=True)
    OUT.mkdir(parents=True)
    (OUT / "confusion_matrices").mkdir()
    write_csv(OUT / "eligibility.csv", eligibility.to_dict("records"))
    write_csv(OUT / "samples.csv", samples.to_dict("records"))
    write_csv(OUT / "population_comparison.csv", population)
    write_csv(OUT / "model_comparison.csv", model_rows)
    write_csv(OUT / "class_metrics.csv", class_rows)
    write_csv(OUT / "confidence_thresholds.csv", confidence_rows)
    write_csv(OUT / "tool_metrics.csv", tool_rows)
    write_csv(OUT / "tool_exclusion_sensitivity.csv", exclusion_rows)
    write_csv(OUT / "feature_importance.csv", importance_rows)
    write_csv(OUT / "validation_predictions.csv", prediction_rows)
    for name in fitted_models:
        write_csv(OUT / "confusion_matrices" / f"{name}.csv", [x for x in confusion_rows if x["model"] == name])
    # Candidate ranking follows user's priorities; this is development
    # selection on the same Validation, never independent final performance.
    candidates = [r for r in model_rows if not r["model"].startswith("dummy")]
    best = max(candidates, key=lambda r: (r["macro_f1"], r["balanced_accuracy"],
                                          r["pressure_high_recall"], r["leak_recall"]))
    model_path = MODELS / "full_fault_type_best.joblib"
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fitted_models[best["model"]], "model_name": best["model"],
                 "features": FEATURES, "classes": TYPES, "sensor_window_seconds": SENSOR_WINDOW_SECONDS,
                 "prediction_lead_seconds": ANCHOR_BEFORE_FAULT_SECONDS,
                 "quality_rule": ">=60 actual rows, >=24m span, latest row <=60s old, no previous fault or phase crossing",
                 "training_split": str(PLAN.relative_to(ROOT)), "tool_id_used_as_feature": False,
                 "score_calibrated": False}, model_path)
    checks = {"future_sensor_rows": 0, "prediction_at_or_after_fault": 0,
              "cross_split_fault_event_ids": 0, "ttf_feature_columns": 0,
              "fault_or_target_feature_columns": 0, "tool_id_feature_columns": 0,
              "reserved_or_excluded_fit_or_validation_rows": 0}
    write_csv(OUT / "leakage_check.csv", [{"check": key, "violations": value} for key, value in checks.items()])
    (OUT / "best_model.json").write_text(json.dumps({"selected": best,
        "model_path": str(model_path.relative_to(ROOT)), "selection_basis": "validation macro F1, then balanced accuracy, High recall, Leak recall",
        "validation_not_independent_test": True}, indent=2), encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps({"random_state": 42, "features": FEATURES,
        "classes": TYPES, "statistic_definition": "per raw nine-sensor actual pre-anchor 30m window; slope per elapsed hour; last_minus_first",
        "train_class_counts": class_counts.to_dict(), "balanced_sample_weights": weight_map,
        "train_validation_split": str(PLAN.relative_to(ROOT)), "model1_and_rul_unchanged": True,
        "reserved_excluded_unused": True, "dashboard_condition": "Model 2 only after Model 1 warning",
        "score_calibrated": False}, indent=2), encoding="utf-8")
    print("saved", best["model"], len(train), len(valid), flush=True)


if __name__ == "__main__":
    main()
