"""Frozen-split, actual-observation multi-anchor next-fault-type experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

from audit_full_fault_type_data import PLAN, RAW, ROOT, TYPES, phase_of_fault
from train_full_fault_type_baseline import FEATURES, SENSORS, snapshot

OUT = ROOT / "output/experiments/full_fault_type_multi_anchor"
MODELS = ROOT / "output/models"
BASE = ROOT / "output/experiments/full_fault_type_baseline"
PARTIAL = ROOT / "output/experiments/model2_partial_history_audit"
HORIZONS = [(30, "30m"), (60, "1h"), (180, "3h"), (360, "6h"), (720, "12h"), (1440, "24h")]
RF_PARAMS = {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 2,
             "n_jobs": 8, "random_state": 42}
QUALITY = ("strict", "partial_A")
WEIGHTING = ("event", "event_class")


def summarize(t: np.ndarray, x: np.ndarray) -> dict | None:
    """Exact frozen 72-feature definition; no imputation or extra feature."""
    if len(t) < 2 or not np.isfinite(x).all():
        return None
    elapsed = (t - t[0]).astype(np.float64)
    centered = elapsed - elapsed.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 0:
        return None
    result = {}
    for j, sensor in enumerate(SENSORS):
        values = x[:, j]
        slope = float(np.dot(centered, values - values.mean()) / denominator * 3600)
        result.update({f"{sensor}_last": float(values[-1]),
                       f"{sensor}_mean": float(values.mean()),
                       f"{sensor}_std": float(values.std(ddof=0)),
                       f"{sensor}_min": float(values.min()),
                       f"{sensor}_max": float(values.max()),
                       f"{sensor}_median": float(np.median(values)),
                       f"{sensor}_slope_per_hour": slope,
                       f"{sensor}_last_minus_first": float(values[-1] - values[0])})
    return result


def metric(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=TYPES, zero_division=0)
    return {"accuracy": float(accuracy_score(y, prediction)),
            "macro_precision": float(np.mean(precision)),
            "macro_recall": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            **{f"{typ}_recall": float(recall[i]) if support[i] else np.nan for i, typ in enumerate(TYPES)},
            **{f"{typ}_support": int(support[i]) for i, typ in enumerate(TYPES)}}


def write(name, frame):
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def prepare():
    if OUT.exists():
        raise FileExistsError(OUT)
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    events = pd.read_csv(RAW / "fault_events.csv").drop_duplicates(["tool_id", "time", "fault_type"])
    assert len(events) == 1211 and events.fault_type.value_counts().reindex(TYPES).tolist() == [741, 333, 137]
    expected = pd.read_csv(PARTIAL / "tool_sampling_intervals.csv").set_index("tool_id")["expected_rows_in_30m"].to_dict()
    assert len(expected) == 20
    existing = pd.read_csv(BASE / "samples.csv")
    previous_30m = existing.set_index("fault_event_id")
    anchors = []
    parity_checked = 0
    for item in plan["tools"]:
        tool = item["tool_id"]
        table = pq.read_table(RAW / f"{tool}_sensor_ttf.parquet", columns=["time", *SENSORS])
        times = table.column("time").to_numpy(zero_copy_only=False).astype(np.int64)
        sensor = np.column_stack([table.column(s).to_numpy(zero_copy_only=False)
                                  for s in SENSORS]).astype(np.float64)
        log = events[events.tool_id.eq(tool)].sort_values("time", kind="stable")
        previous = None
        for event in log.itertuples():
            fault = int(event.time)
            phase = phase_of_fault(fault, item)
            if phase not in ("train", "validation"):
                previous = fault
                continue
            phase_start = int(times[0]) if phase == "train" else int(item["validation_start_second"])
            lower = max(phase_start, previous + 1 if previous is not None else int(times[0]))
            for minutes, label in HORIZONS:
                prediction = fault - minutes * 60
                first = max(int(np.searchsorted(times, prediction - 1800, side="left")),
                            int(np.searchsorted(times, lower, side="left")))
                end = int(np.searchsorted(times, prediction, side="right"))
                local_t, local_x = times[first:end], sensor[first:end]
                count = len(local_t)
                span = int(local_t[-1] - local_t[0]) if count else np.nan
                age = int(prediction - local_t[-1]) if count else np.nan
                gap = np.diff(local_t)
                maxgap = int(gap.max()) if len(gap) else np.nan
                ratio = count / expected[tool]
                strict = bool(count >= 60 and span >= 1440 and age <= 60 and np.isfinite(local_x).all())
                partial = bool(ratio >= .5 and span >= 900 and age <= 300 and
                               count >= 2 and np.isfinite(local_x).all())
                if not (strict or partial):
                    continue
                features = summarize(local_t, local_x)
                if features is None:
                    continue
                if not (np.isfinite(np.fromiter(features.values(), dtype=np.float64)).all()
                        and local_t[-1] <= prediction < fault):
                    raise AssertionError("Future sensor or nonfinite feature")
                row = {"fault_event_id": int(event.fault_event_id), "tool_id": tool,
                       "phase": phase, "fault_type": event.fault_type,
                       "fault_timestamp": fault, "prediction_timestamp": prediction,
                       "horizon": label, "horizon_minutes": minutes,
                       "history_start": prediction - 1800,
                       "phase_and_cycle_start": lower,
                       "sensor_row_count": count, "coverage_duration_seconds": span,
                       "last_sensor_age_seconds": age, "median_gap_seconds": float(np.median(gap)) if len(gap) else np.nan,
                       "max_gap_seconds": maxgap, "first_sensor_timestamp": int(local_t[0]),
                       "last_sensor_timestamp": int(local_t[-1]),
                       "observation_ratio": ratio,
                       "eligible_strict": strict, "eligible_partial_A": partial,
                       **features}
                anchors.append(row)
                if label == "30m" and strict and int(event.fault_event_id) in previous_30m.index:
                    old = previous_30m.loc[int(event.fault_event_id)]
                    if old.phase != phase or old.fault_type != event.fault_type:
                        raise AssertionError("Single-anchor metadata drift")
                    for name in FEATURES:
                        if not np.isclose(row[name], old[name], rtol=1e-8, atol=1e-8):
                            raise AssertionError(f"Single-anchor feature drift: {name}")
                    parity_checked += 1
            previous = fault
        print("prepared", tool, flush=True)
    frame = pd.DataFrame(anchors)
    assert len(frame) and parity_checked == len(existing) == 347
    assert not frame.duplicated(["fault_event_id", "horizon"]).any()
    assert (frame.last_sensor_timestamp <= frame.prediction_timestamp).all()
    assert (frame.prediction_timestamp < frame.fault_timestamp).all()
    assert (frame.first_sensor_timestamp >= frame.history_start).all()
    assert (frame.first_sensor_timestamp >= frame.phase_and_cycle_start).all()
    assert not set(frame.loc[frame.phase.eq("train"), "fault_event_id"]) & set(frame.loc[frame.phase.eq("validation"), "fault_event_id"])
    assert not any("ttf" in s.lower() or "tool" in s.lower() or "fault" in s.lower() or "horizon" in s.lower() for s in FEATURES)
    OUT.mkdir(parents=True)
    write("anchor_samples.csv", frame)
    count_rows, overview = [], []
    for condition, field in (("strict", "eligible_strict"), ("partial_A", "eligible_partial_A")):
        subset = frame[frame[field]]
        for phase in ("train", "validation"):
            part = subset[subset.phase.eq(phase)]
            per_event = part.groupby("fault_event_id").size()
            overview.append({"condition": condition, "phase": phase,
                             "samples": len(part), "unique_fault_events": len(per_event),
                             "mean_anchors_per_event": float(per_event.mean()) if len(per_event) else np.nan,
                             "median_anchors_per_event": float(per_event.median()) if len(per_event) else np.nan,
                             "max_anchors_per_event": int(per_event.max()) if len(per_event) else 0,
                             "tools": part.tool_id.nunique()})
            for minutes, label in HORIZONS:
                horizon = part[part.horizon.eq(label)]
                for typ in [*TYPES, "total"]:
                    local = horizon if typ == "total" else horizon[horizon.fault_type.eq(typ)]
                    count_rows.append({"condition": condition, "phase": phase,
                                       "horizon": label, "horizon_minutes": minutes,
                                       "fault_type": typ, "samples": len(local),
                                       "unique_fault_events": local.fault_event_id.nunique(),
                                       "tools": local.tool_id.nunique()})
    write("anchor_counts.csv", pd.DataFrame(count_rows))
    write("dataset_overview.csv", pd.DataFrame(overview))
    (OUT / "preparation_config.json").write_text(json.dumps({
        "model_training": False, "total_original_fault_events": len(events),
        "horizons_minutes": [m for m, _ in HORIZONS],
        "fixed_split": str(PLAN.relative_to(ROOT)),
        "features": FEATURES, "feature_count": len(FEATURES),
        "single_anchor_feature_parity_checked": parity_checked,
        "strict_rule": ">=60 actual observations, >=24m observed span, latest <=60s old, all nine raw sensors finite",
        "partial_A_rule": "actual rows / Train Tool expected rows >=.50, >=15m observed span, latest <=300s old, all nine raw sensors finite",
        "no_interpolation": True, "reserved_or_excluded_unused": True,
        "random_state": 42}, indent=2), encoding="utf-8")
    print(pd.DataFrame(overview).to_string(index=False), flush=True)


def train():
    if not (OUT / "anchor_samples.csv").exists():
        raise FileNotFoundError("Run prepare first")
    frame = pd.read_csv(OUT / "anchor_samples.csv")
    score_file = OUT / "validation_predictions.csv"
    if score_file.exists():
        raise FileExistsError(score_file)
    predictions, training_rows = [], []
    for condition, field in (("strict", "eligible_strict"), ("partial_A", "eligible_partial_A")):
        subset = frame[frame[field]]
        train_data = subset[subset.phase.eq("train")]
        validation = subset[subset.phase.eq("validation")]
        train_events = train_data.fault_event_id.value_counts()
        event_class = train_data.drop_duplicates("fault_event_id").fault_type.value_counts()
        assert set(event_class.index) == set(TYPES)
        event_weight = train_data.fault_event_id.map(lambda x: 1.0 / train_events[x]).to_numpy(dtype=np.float64)
        assert np.allclose(pd.Series(event_weight).groupby(train_data.fault_event_id.to_numpy()).sum().to_numpy(), 1.0)
        class_weight = {typ: len(train_events) / (3 * int(event_class[typ])) for typ in TYPES}
        x_train = train_data[FEATURES].to_numpy(dtype=np.float32)
        x_valid = validation[FEATURES].to_numpy(dtype=np.float32)
        for weight_kind in WEIGHTING:
            model_name = f"{condition}_{weight_kind}"
            model_path = MODELS / f"full_fault_type_multi_anchor_{model_name}.joblib"
            if model_path.exists():
                raise FileExistsError(model_path)
            weight = event_weight if weight_kind == "event" else event_weight * train_data.fault_type.map(class_weight).to_numpy(dtype=float)
            model = RandomForestClassifier(**RF_PARAMS)
            model.fit(x_train, train_data.fault_type.to_numpy(), sample_weight=weight)
            scores_raw = model.predict_proba(x_valid)
            scores = np.column_stack([scores_raw[:, list(model.classes_).index(typ)] for typ in TYPES])
            guessed = model.predict(x_valid)
            predictions.extend({"model": model_name, "condition": condition, "weighting": weight_kind,
                                "fault_event_id": int(row.fault_event_id), "tool_id": row.tool_id,
                                "horizon": row.horizon, "horizon_minutes": int(row.horizon_minutes),
                                "fault_type": row.fault_type, "predicted": prediction,
                                "max_class_score": float(score.max()),
                                **{f"score_{typ}": float(score[j]) for j, typ in enumerate(TYPES)}}
                               for row, prediction, score in zip(validation.itertuples(), guessed, scores))
            MODELS.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "model_name": model_name, "features": FEATURES,
                         "classes": TYPES, "parameters": RF_PARAMS,
                         "training_event_counts": event_class.to_dict(),
                         "event_weight_rule": "1/N usable anchors per Train fault event",
                         "class_weight_rule": class_weight if weight_kind == "event_class" else None,
                         "split": str(PLAN.relative_to(ROOT)), "score_calibrated": False}, model_path)
            training_rows.append({"model": model_name, "train_samples": len(train_data),
                                  "train_unique_events": len(train_events),
                                  "validation_samples": len(validation),
                                  "validation_unique_events": validation.fault_event_id.nunique(),
                                  "class_weight_low": class_weight["pressure_low"] if weight_kind == "event_class" else 1,
                                  "class_weight_high": class_weight["pressure_high"] if weight_kind == "event_class" else 1,
                                  "class_weight_leak": class_weight["leak"] if weight_kind == "event_class" else 1,
                                  "model_path": str(model_path.relative_to(ROOT))})
            print("trained", model_name, len(train_data), len(validation), flush=True)
    write("validation_predictions.csv", pd.DataFrame(predictions))
    write("training_summary.csv", pd.DataFrame(training_rows))


def evaluate():
    if not (OUT / "validation_predictions.csv").exists():
        raise FileNotFoundError("Run train first")
    if (OUT / "model_comparison.csv").exists():
        raise FileExistsError("Evaluation results already exist")
    predictions = pd.read_csv(OUT / "validation_predictions.csv")
    frame = pd.read_csv(OUT / "anchor_samples.csv")
    model_rows, class_rows, confusion_rows, event_rows, event_predictions = [], [], [], [], []
    horizon_rows, cumulative_rows, confidence_rows, tool_rows, exclusion_rows = [], [], [], [], []
    for model_name, part in predictions.groupby("model", sort=False):
        condition = str(part.condition.iloc[0])
        block = metric(part.fault_type, part.predicted)
        model_rows.append({"model": model_name, "condition": condition, "weighting": part.weighting.iloc[0],
                           "validation_samples": len(part), "validation_unique_events": part.fault_event_id.nunique(),
                           **block})
        precision, recall, f1, support = precision_recall_fscore_support(
            part.fault_type, part.predicted, labels=TYPES, zero_division=0)
        for i, typ in enumerate(TYPES):
            class_rows.append({"model": model_name, "fault_type": typ, "precision": float(precision[i]),
                               "recall": float(recall[i]), "f1": float(f1[i]), "support_samples": int(support[i]),
                               "support_events": part[part.fault_type.eq(typ)].fault_event_id.nunique()})
        matrix = confusion_matrix(part.fault_type, part.predicted, labels=TYPES)
        for i, actual in enumerate(TYPES):
            for j, guess in enumerate(TYPES):
                confusion_rows.append({"model": model_name, "actual": actual,
                                       "predicted": guess, "samples": int(matrix[i, j])})
        for _, row in part.groupby("fault_event_id"):
            row = row.sort_values("horizon_minutes", ascending=False)
            first, last = row.iloc[0], row.iloc[-1]
            means = row[[f"score_{typ}" for typ in TYPES]].mean().to_numpy(dtype=float)
            avg_guess = TYPES[int(np.argmax(means))]
            votes = row.predicted.value_counts()
            max_votes = int(votes.max())
            tied = [typ for typ in TYPES if votes.get(typ, 0) == max_votes]
            majority = max(tied, key=lambda typ: means[TYPES.index(typ)])
            event_predictions.append({"model": model_name, "condition": condition,
                "fault_event_id": int(first.fault_event_id), "tool_id": first.tool_id,
                "fault_type": first.fault_type, "anchors": len(row),
                "first_horizon_minutes": int(first.horizon_minutes),
                "last_horizon_minutes": int(last.horizon_minutes),
                "first_prediction": first.predicted, "last_prediction": last.predicted,
                "average_score_prediction": avg_guess, "majority_vote_prediction": majority,
                "max_average_score": float(means.max()),
                **{f"average_score_{typ}": float(means[i]) for i, typ in enumerate(TYPES)}})
        for minutes, label in HORIZONS:
            h = part[part.horizon.eq(label)]
            horizon_rows.append({"model": model_name, "horizon": label, "horizon_minutes": minutes,
                                 "samples": len(h), "unique_fault_events": h.fault_event_id.nunique(),
                                 **metric(h.fault_type, h.predicted)})
        for minutes in (60, 180, 360, 720, 1440):
            h = part[part.horizon_minutes.le(minutes)]
            cumulative_rows.append({"model": model_name, "through_minutes": minutes,
                "samples": len(h), "unique_fault_events": h.fault_event_id.nunique(),
                **metric(h.fault_type, h.predicted)})
        for threshold in (.5, .6, .7, .8):
            shown = part[part.max_class_score.ge(threshold)]
            rec = {"model": model_name, "level": "sample", "score_threshold": threshold,
                   "denominator": len(part), "displayed": len(shown),
                   "displayed_unique_events": shown.fault_event_id.nunique(),
                   "coverage": len(shown)/len(part),
                   "correct_rate": float((shown.fault_type == shown.predicted).mean()) if len(shown) else np.nan,
                   "incorrect": int((shown.fault_type != shown.predicted).sum())}
            for typ in TYPES:
                rec[f"{typ}_displayed"] = int(shown.predicted.eq(typ).sum())
                rec[f"{typ}_correct"] = int((shown.predicted.eq(typ) & shown.fault_type.eq(typ)).sum())
            confidence_rows.append(rec)
    event_frame = pd.DataFrame(event_predictions)
    for model_name, part in event_frame.groupby("model", sort=False):
        for method, field in (("first", "first_prediction"), ("last", "last_prediction"),
                              ("average_score", "average_score_prediction"),
                              ("majority_vote", "majority_vote_prediction")):
            event_rows.append({"model": model_name, "condition": part.condition.iloc[0],
                               "method": method, "unique_fault_events": len(part),
                               **metric(part.fault_type, part[field])})
        for tool in sorted(part.tool_id.unique()):
            local = part[part.tool_id.eq(tool)]
            tool_rows.append({"model": model_name, "tool_id": tool, "unique_events": len(local),
                              "low_events": int(local.fault_type.eq("pressure_low").sum()),
                              "high_events": int(local.fault_type.eq("pressure_high").sum()),
                              "leak_events": int(local.fault_type.eq("leak").sum()),
                              **metric(local.fault_type, local.average_score_prediction)})
        for excluded in ("03_M02", "09_M02", "06_M02"):
            local = part[~part.tool_id.eq(excluded)]
            exclusion_rows.append({"model": model_name, "excluded_tool": excluded,
                                   "unique_events": len(local),
                                   **metric(local.fault_type, local.average_score_prediction)})
        for threshold in (.5, .6, .7, .8):
            shown = part[part.max_average_score.ge(threshold)]
            rec = {"model": model_name, "level": "event_average_score", "score_threshold": threshold,
                   "denominator": len(part), "displayed": len(shown),
                   "displayed_unique_events": len(shown),
                   "coverage": len(shown)/len(part),
                   "correct_rate": float((shown.fault_type == shown.average_score_prediction).mean()) if len(shown) else np.nan,
                   "incorrect": int((shown.fault_type != shown.average_score_prediction).sum())}
            for typ in TYPES:
                rec[f"{typ}_displayed"] = int(shown.average_score_prediction.eq(typ).sum())
                rec[f"{typ}_correct"] = int((shown.average_score_prediction.eq(typ) & shown.fault_type.eq(typ)).sum())
            confidence_rows.append(rec)
    write("model_comparison.csv", pd.DataFrame(model_rows))
    write("class_metrics.csv", pd.DataFrame(class_rows))
    write("event_level_metrics.csv", pd.DataFrame(event_rows))
    write("event_predictions.csv", event_frame)
    write("horizon_metrics.csv", pd.DataFrame(horizon_rows))
    write("cumulative_horizon_metrics.csv", pd.DataFrame(cumulative_rows))
    write("confidence_thresholds.csv", pd.DataFrame(confidence_rows))
    write("tool_metrics.csv", pd.DataFrame(tool_rows))
    write("tool_exclusion_sensitivity.csv", pd.DataFrame(exclusion_rows))
    (OUT / "confusion_matrices").mkdir(exist_ok=False)
    for name in predictions.model.unique():
        pd.DataFrame([r for r in confusion_rows if r["model"] == name]).to_csv(
            OUT / "confusion_matrices" / f"{name}.csv", index=False, encoding="utf-8-sig")
    # Preserve baseline single-anchor performance and compare matched 30m
    # event sets separately from all-event summaries.
    prior = pd.read_csv(BASE / "validation_predictions.csv")
    prior = prior[prior.model.eq("random_forest_default")]
    prior_metrics = pd.read_csv(BASE / "model_comparison.csv")
    prior_metrics = prior_metrics[prior_metrics.model.eq("random_forest_default")].iloc[0]
    matched_rows = []
    for name, part in event_frame.groupby("model"):
        common = part[part.fault_event_id.isin(prior.fault_event_id)]
        old = prior[prior.fault_event_id.isin(common.fault_event_id)]
        assert len(common) == len(old) and set(common.fault_event_id) == set(old.fault_event_id)
        for label, truth, guess in (("single_anchor_on_common_events", old.actual, old.predicted),
                                    ("multi_first_on_common_events", common.fault_type, common.first_prediction),
                                    ("multi_last_on_common_events", common.fault_type, common.last_prediction),
                                    ("multi_average_on_common_events", common.fault_type, common.average_score_prediction)):
            matched_rows.append({"model": name, "comparison": label, "common_events": len(common),
                                 **metric(truth, guess)})
    write("matched_single_anchor_comparison.csv", pd.DataFrame(matched_rows))
    checks = {"sensor_after_prediction": int((frame.last_sensor_timestamp > frame.prediction_timestamp).sum()),
              "prediction_at_or_after_fault": int((frame.prediction_timestamp >= frame.fault_timestamp).sum()),
              "same_event_train_validation": len(set(frame.loc[frame.phase.eq("train"), "fault_event_id"]) &
                                                 set(frame.loc[frame.phase.eq("validation"), "fault_event_id"])),
              "ttf_feature_columns": sum("ttf" in f.lower() for f in FEATURES),
              "horizon_feature_columns": sum("horizon" in f.lower() for f in FEATURES),
              "tool_id_feature_columns": sum("tool" in f.lower() for f in FEATURES),
              "fault_or_target_feature_columns": sum("fault" in f.lower() or "target" in f.lower() for f in FEATURES),
              "reserved_excluded_samples": int((~frame.phase.isin(["train", "validation"])).sum())}
    if any(checks.values()):
        raise AssertionError(checks)
    write("leakage_check.csv", pd.DataFrame([{"check": k, "violations": v} for k, v in checks.items()]))
    average = [r for r in event_rows if r["method"] == "average_score"]
    selected = max(average, key=lambda r: (r["macro_f1"], r["balanced_accuracy"],
                                           r["pressure_high_recall"], r["leak_recall"]))
    chosen_name = selected["model"]
    source_model = MODELS / f"full_fault_type_multi_anchor_{chosen_name}.joblib"
    best_payload = joblib.load(source_model)
    best_payload["selection"] = selected
    target_model = MODELS / "full_fault_type_multi_anchor_best.joblib"
    local_model = OUT / "best_model.joblib"
    if target_model.exists() or local_model.exists():
        raise FileExistsError("Best model target exists")
    joblib.dump(best_payload, target_model)
    joblib.dump(best_payload, local_model)
    (OUT / "best_model_metadata.json").write_text(json.dumps({
        "selected": selected, "model_file": str(target_model.relative_to(ROOT)),
        "local_copy": str(local_model.relative_to(ROOT)),
        "selection_basis": "same Validation event-level average-score macro F1 then balanced accuracy, High and Leak recall",
        "baseline_single_anchor_macro_f1": float(prior_metrics.macro_f1),
        "baseline_single_anchor_balanced_accuracy": float(prior_metrics.balanced_accuracy),
        "validation_is_development_not_independent_test": True,
        "score_calibrated": False}, indent=2), encoding="utf-8")
    print("selected", chosen_name, selected, flush=True)


def event_class_metrics():
    """Rebuild class details from saved validation predictions, without training."""
    source = OUT / "event_predictions.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    rows = []
    for model_name, part in pd.read_csv(source).groupby("model", sort=False):
        for method, field in (("first", "first_prediction"), ("last", "last_prediction"),
                              ("average_score", "average_score_prediction"),
                              ("majority_vote", "majority_vote_prediction")):
            precision, recall, f1, support = precision_recall_fscore_support(
                part.fault_type, part[field], labels=TYPES, zero_division=0)
            for i, typ in enumerate(TYPES):
                rows.append({"model": model_name, "method": method, "fault_type": typ,
                             "precision": float(precision[i]), "recall": float(recall[i]),
                             "f1": float(f1[i]), "support_events": int(support[i])})
    write("event_class_metrics.csv", pd.DataFrame(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "train", "evaluate", "event_class_metrics"])
    args = parser.parse_args()
    {"prepare": prepare, "train": train, "evaluate": evaluate,
     "event_class_metrics": event_class_metrics}[args.phase]()
