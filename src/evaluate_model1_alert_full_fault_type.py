"""Evaluate frozen fault-type RF only at frozen Model 1 manager alerts."""
from __future__ import annotations

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

from audit_cooldown_forward_24h import first_after_gap, notify
from audit_full_fault_type_data import PLAN, RAW, ROOT, TYPES
from train_full_fault_type_baseline import FEATURES, SENSORS, snapshot

SOURCE = ROOT / "output/experiments/forward_validation_ab_24h"
POLICY = ROOT / "output/experiments/cooldown_forward_validation_24h/report.json"
MODEL1 = ROOT / "output/models/xgb_24h_forward_a.joblib"
MODEL2 = ROOT / "output/models/full_fault_type_best.joblib"
OUT = ROOT / "output/experiments/model1_alert_full_fault_type_eval"
SCORES = SOURCE / "validation_probabilities_a.npz"
RAW_GAP_SECONDS = 600
COOLDOWN_SECONDS = 1800
THRESHOLD = .34
LEAD_BINS = [(0, 1, "0-1h"), (1, 3, "1-3h"), (3, 6, "3-6h"),
             (6, 12, "6-12h"), (12, 24, "12-24h"), (24, 72, "24-72h"),
             (72, np.inf, "72h+")]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(actual, predicted):
    y = np.asarray(actual)
    p = np.asarray(predicted)
    if not len(y):
        return {"events": 0, "accuracy": np.nan, "macro_f1": np.nan,
                "balanced_accuracy": np.nan,
                **{f"{name}_recall": np.nan for name in TYPES},
                **{f"{name}_support": 0 for name in TYPES}}
    _, recall, _, support = precision_recall_fscore_support(y, p, labels=TYPES, zero_division=0)
    return {"events": len(y), "accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, labels=TYPES, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            **{f"{name}_recall": float(recall[i]) if support[i] else np.nan for i, name in enumerate(TYPES)},
            **{f"{name}_support": int(support[i]) for i, name in enumerate(TYPES)}}


def save(path, frame):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    training = json.loads((SOURCE / "training_a.json").read_text(encoding="utf-8"))
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert policy["model_sha256"] == training["model_sha256"] == sha256(MODEL1)
    assert policy["validation_score_sha256"] == training["validation_score_sha256"] == sha256(SCORES)
    assert "threshold .34" in policy["policy"] and "30m manager notification" in policy["policy"]
    assert policy["validation_only"] and not policy["test_scored"]
    loaded = joblib.load(MODEL2)
    assert loaded["model_name"] == "random_forest_default" and loaded["features"] == FEATURES
    assert loaded["sensor_window_seconds"] == 1800 and loaded["prediction_lead_seconds"] == 1800
    model = loaded["model"]
    assert set(model.classes_) == set(TYPES)
    with np.load(SCORES) as source:
        times, probability, tool_code = [source[key].copy() for key in
                                         ("time", "probability", "tool_code")]
    assert len(times) == training["validation_rows"]
    fault_log = pd.read_csv(RAW / "fault_events.csv").drop_duplicates(["tool_id", "time", "fault_type"])
    if fault_log.groupby(["tool_id", "time"]).size().max() != 1:
        raise AssertionError("Ambiguous simultaneous fault types")
    rows, features = [], []
    raw_count, suppressed_count = 0, 0
    for code, item in enumerate(plan["tools"]):
        tool = item["tool_id"]
        positions = np.flatnonzero(tool_code == code)
        if np.any(times[positions][1:] < times[positions][:-1]):
            raise AssertionError("Unsorted Model 1 validation timestamps")
        hits = positions[probability[positions] >= THRESHOLD]
        raw = first_after_gap(hits, times, RAW_GAP_SECONDS)
        notified, suppressed = notify(raw, times, COOLDOWN_SECONDS)
        raw_count += len(raw)
        suppressed_count += len(suppressed)
        if np.any(np.diff(times[notified]) < COOLDOWN_SECONDS):
            raise AssertionError("Notification cooldown was not preserved")
        if not len(notified):
            continue
        # Read only actual 4-second sensor observations from the validation
        # project Train parquet; no Model 1 or Model 2 retraining.
        table = pq.read_table(RAW / f"{tool}_sensor_ttf.parquet", columns=["time", *SENSORS])
        sensor_times = table.column("time").to_numpy(zero_copy_only=False).astype(np.int64)
        sensor_matrix = np.column_stack([table.column(s).to_numpy(zero_copy_only=False)
                                         for s in SENSORS]).astype(np.float64)
        events = fault_log[fault_log.tool_id.eq(tool)].sort_values("time", kind="stable")
        event_times = events.time.to_numpy(dtype=np.int64)
        event_ids = events.fault_event_id.to_numpy(dtype=np.int64)
        event_types = events.fault_type.to_numpy()
        validation_start = int(item["validation_start_second"])
        validation_end = int(item["validation_end_exclusive_second"])
        for position in notified:
            now = int(times[position])
            if not validation_start <= now < validation_end:
                raise AssertionError("Notification outside frozen Validation")
            next_pos = int(np.searchsorted(event_times, now, side="right"))
            previous = int(event_times[next_pos - 1]) if next_pos else None
            lower = max(validation_start, previous + 1 if previous is not None else validation_start)
            feat, quality = snapshot(sensor_times, sensor_matrix, now, lower)
            next_in_validation = next_pos < len(event_times) and int(event_times[next_pos]) < validation_end
            # Mutually exclusive A/B/C: unknown next fault first, then history.
            category = "C_next_fault_unobserved" if not next_in_validation else (
                       "B_history_insufficient" if feat is None else "A_evaluable")
            lead_hours = (int(event_times[next_pos]) - now) / 3600 if next_in_validation else np.nan
            row = {"tool_id": tool, "alert_timestamp": now,
                   "model1_probability": float(probability[position]),
                   "category": category, "history_sufficient": feat is not None,
                   "next_fault_observed_in_validation": bool(next_in_validation),
                   "next_fault_event_id": int(event_ids[next_pos]) if next_in_validation else np.nan,
                   "next_fault_timestamp": int(event_times[next_pos]) if next_in_validation else np.nan,
                   "actual_fault_type": event_types[next_pos] if next_in_validation else None,
                   "lead_hours": lead_hours, "previous_observed_fault_timestamp": previous,
                   "validation_start": validation_start, "validation_end_exclusive": validation_end,
                   **quality}
            rows.append(row)
            if category == "A_evaluable":
                features.append({"tool_id": tool, "alert_timestamp": now,
                                 "next_fault_event_id": int(event_ids[next_pos]), **feat})
        print(tool, "raw episodes", len(raw), "notifications", len(notified), flush=True)
    population = pd.DataFrame(rows)
    matrix = pd.DataFrame(features)
    assert len(population) and population[["tool_id", "alert_timestamp"]].duplicated().sum() == 0
    assert len(matrix) == population.category.eq("A_evaluable").sum()
    assert (population.loc[population.history_sufficient, "last_sensor_timestamp"] <=
            population.loc[population.history_sufficient, "alert_timestamp"]).all()
    assert not any("ttf" in f.lower() or "fault" in f.lower() or "tool" in f.lower() for f in FEATURES)
    x = matrix[FEATURES].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        raise AssertionError("Nonfinite Model 2 feature")
    predicted = model.predict(x)
    raw_scores = model.predict_proba(x)
    scores = np.column_stack([raw_scores[:, list(model.classes_).index(c)] for c in TYPES])
    score_frame = matrix[["tool_id", "alert_timestamp"]].copy()
    score_frame["predicted_fault_type"] = predicted
    score_frame["max_class_score"] = scores.max(axis=1)
    for j, typ in enumerate(TYPES):
        score_frame[f"score_{typ}"] = scores[:, j]
    evaluated = population.merge(score_frame, on=["tool_id", "alert_timestamp"], how="left", validate="one_to_one")
    evaluated_only = evaluated[evaluated.category.eq("A_evaluable")].copy()
    y = evaluated_only.actual_fault_type.to_numpy()
    p = evaluated_only.predicted_fault_type.to_numpy()
    overall = metrics(y, p)
    class_rows = []
    precision, recall, f1, support = precision_recall_fscore_support(y, p, labels=TYPES, zero_division=0)
    for i, typ in enumerate(TYPES):
        class_rows.append({"fault_type": typ, "precision": float(precision[i]),
                           "recall": float(recall[i]), "f1": float(f1[i]), "support_alerts": int(support[i])})
    cm = confusion_matrix(y, p, labels=TYPES)
    confusion = [{"actual": actual, "predicted": guess, "alerts": int(cm[i, j])}
                 for i, actual in enumerate(TYPES) for j, guess in enumerate(TYPES)]
    event_rows, stability = [], []
    for event_id, part in evaluated_only.sort_values("alert_timestamp").groupby("next_fault_event_id"):
        part = part.sort_values("alert_timestamp")
        first, last = part.iloc[0], part.iloc[-1]
        average = part[[f"score_{typ}" for typ in TYPES]].mean().to_numpy(dtype=float)
        choice = TYPES[int(np.argmax(average))]
        common = {"fault_event_id": int(event_id), "tool_id": first.tool_id,
                  "actual_fault_type": first.actual_fault_type, "alerts": len(part),
                  "first_alert_timestamp": int(first.alert_timestamp),
                  "last_alert_timestamp": int(last.alert_timestamp),
                  "first_lead_hours": float(first.lead_hours),
                  "last_lead_hours": float(last.lead_hours)}
        event_rows.append({**common, "first_prediction": first.predicted_fault_type,
                           "last_prediction": last.predicted_fault_type,
                           "average_score_prediction": choice})
        predicted_sequence = part.predicted_fault_type.to_numpy()
        changes = int(np.sum(predicted_sequence[1:] != predicted_sequence[:-1]))
        stability.append({**common, "prediction_changes": changes,
                          "adjacent_pairs": max(0, len(part)-1),
                          "adjacent_same": max(0, len(part)-1)-changes,
                          "last_correct": bool(last.predicted_fault_type == first.actual_fault_type)})
    event_frame = pd.DataFrame(event_rows)
    event_metrics = []
    for method, column in (("first", "first_prediction"), ("last", "last_prediction"),
                           ("average_score", "average_score_prediction")):
        event_metrics.append({"method": method,
                              **metrics(event_frame.actual_fault_type, event_frame[column])})
    lead_rows = []
    for lo, hi, label in LEAD_BINS:
        mask = (evaluated_only.lead_hours >= lo) & (evaluated_only.lead_hours < hi)
        part = evaluated_only[mask]
        lead_rows.append({"lead_bin": label, "alerts": len(part),
                          "unique_fault_events": part.next_fault_event_id.nunique(),
                          **metrics(part.actual_fault_type, part.predicted_fault_type)})
    confidence = []
    for threshold in (.5, .6, .7, .8):
        shown = evaluated_only[evaluated_only.max_class_score.ge(threshold)]
        row = {"score_threshold": threshold, "evaluable_alerts": len(evaluated_only),
               "shown_alerts": len(shown), "withheld_alerts": len(evaluated_only)-len(shown),
               "coverage": len(shown)/len(evaluated_only) if len(evaluated_only) else np.nan,
               "shown_unique_fault_events": shown.next_fault_event_id.nunique(),
               "shown_correct_rate": float(np.mean(shown.predicted_fault_type == shown.actual_fault_type)) if len(shown) else np.nan,
               "incorrect_alerts": int(np.sum(shown.predicted_fault_type != shown.actual_fault_type))}
        for typ in TYPES:
            row[f"{typ}_shown"] = int(shown.predicted_fault_type.eq(typ).sum())
            row[f"{typ}_shown_correct"] = int((shown.predicted_fault_type.eq(typ) & shown.actual_fault_type.eq(typ)).sum())
        confidence.append(row)
    stable = pd.DataFrame(stability)
    baseline = pd.read_csv(ROOT / "output/experiments/full_fault_type_baseline/model_comparison.csv")
    prior = baseline[baseline.model.eq("random_forest_default")].iloc[0]
    if not (np.isclose(prior.macro_f1, .602548390234413)
            and np.isclose(prior.balanced_accuracy, .6108936725375081)):
        raise AssertionError("Frozen Model 2 30m baseline changed")
    OUT.mkdir(parents=True)
    save(OUT / "manager_alert_population.csv", population)
    save(OUT / "alert_predictions.csv", evaluated)
    save(OUT / "model2_features_at_alerts.csv", matrix)
    save(OUT / "alert_level_metrics.csv", pd.DataFrame([overall]))
    save(OUT / "alert_class_metrics.csv", pd.DataFrame(class_rows))
    save(OUT / "alert_confusion_matrix.csv", pd.DataFrame(confusion))
    save(OUT / "event_predictions.csv", event_frame)
    save(OUT / "event_level_metrics.csv", pd.DataFrame(event_metrics))
    save(OUT / "lead_time_metrics.csv", pd.DataFrame(lead_rows))
    save(OUT / "confidence_thresholds.csv", pd.DataFrame(confidence))
    save(OUT / "repeated_alert_stability.csv", stable)
    check = {"future_sensor_rows": int((population.last_sensor_timestamp > population.alert_timestamp).sum()),
             "ttf_feature_columns": sum("ttf" in f.lower() for f in FEATURES),
             "fault_feature_columns": sum("fault" in f.lower() for f in FEATURES),
             "tool_id_feature_columns": sum("tool" in f.lower() for f in FEATURES),
             "model1_score_sha256_matches_saved": sha256(SCORES) == policy["validation_score_sha256"],
             "model2_file_sha256": sha256(MODEL2)}
    (OUT / "config.json").write_text(json.dumps({"model_training": False,
        "model1_threshold": THRESHOLD, "raw_alarm_gap_seconds": RAW_GAP_SECONDS,
        "notification_cooldown_seconds": COOLDOWN_SECONDS,
        "model1_score_file_sha256": sha256(SCORES), "model1_file_sha256": sha256(MODEL1),
        "model2_file_sha256": sha256(MODEL2), "model2_name": loaded["model_name"],
        "validation_only": True, "raw_alarm_episode_count": raw_count,
        "suppressed_by_cooldown": suppressed_count,
        "admin_alert_count": len(population), "categories": population.category.value_counts().to_dict(),
        "evaluation_scope": "all frozen Model1 administrator notifications in Validation; next fault type only when that next fault occurs before that Tool's Validation end",
        "next_fault_is_target_only": True, "scores_calibrated": False,
        "leakage_checks": check}, indent=2), encoding="utf-8")
    print("admin alerts", len(population), "categories", population.category.value_counts().to_dict(), flush=True)
    print("alert metrics", overall, flush=True)
    print("event metrics", event_metrics, flush=True)


if __name__ == "__main__":
    main()
