"""Cooldown and causal second-stage filtering on frozen forward Model A scores.

Usage: python src/false_alarm_second_stage_24h.py prepare|fit
The primary model, 85 features, and score file are never modified.
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from evaluate_event_thresholds_24h import BASELINE, ROOT, event_metrics, write_csv
from experiment_tool_stage_interaction_24h import file_hash
from forward_validation_ab_24h import read_pretest, validation_events

OUT = ROOT / "output" / "experiments" / "false_alarm_second_stage_24h"
SOURCE = ROOT / "output" / "experiments" / "forward_validation_ab_24h"
PRIMARY_MODEL = ROOT / "output" / "models" / "xgb_24h_forward_a.joblib"
PRIMARY_SCORE = SOURCE / "validation_probabilities_a.npz"
THRESHOLD = .34
HORIZON = 86400
HARD_NEGATIVE_HORIZON = 72 * 3600
COOLDOWNS = (10, 30, 60, 120)
SENSOR_COLUMNS = ["FLOWCOOLFLOWRATE", "FLOWCOOLPRESSURE",
                  "FLOWCOOLFLOWRATE_std_5m", "FLOWCOOLPRESSURE_std_5m",
                  "FLOWCOOLFLOWRATE_std_30m", "FLOWCOOLPRESSURE_std_30m", "stage"]
NUMERIC = ["first_stage_probability", "prob_max_30m", "prob_mean_30m", "prob_median_30m",
           "prob_slope_5m", "prob_slope_10m", "prob_slope_30m",
           "hit_count_5m", "hit_count_10m", "hit_count_30m",
           "alarm_duration_minutes_at_onset", "FLOWCOOLFLOWRATE",
           "FLOWCOOLPRESSURE", "flow_delta_5m", "pressure_delta_5m",
           "flow_tool_normal_percentile", "pressure_tool_robust_zscore",
           "FLOWCOOLFLOWRATE_std_5m", "FLOWCOOLPRESSURE_std_5m",
           "FLOWCOOLFLOWRATE_std_30m", "FLOWCOOLPRESSURE_std_30m"]
CATEGORICAL = ["tool_id", "stage"]


def context():
    plan = json.loads((SOURCE / "split_plan.json").read_text(encoding="utf-8"))
    first = json.loads((SOURCE / "training_a.json").read_text(encoding="utf-8"))
    if file_hash(PRIMARY_MODEL) != first["model_sha256"] or file_hash(PRIMARY_SCORE) != first["validation_score_sha256"]:
        raise AssertionError("Frozen first-stage model or score changed")
    with np.load(PRIMARY_SCORE) as saved:
        truth, probability, time, tool = (saved[k].copy() for k in
                                           ("truth", "probability", "time", "tool_code"))
    if len(time) != first["validation_rows"] or int(truth.sum()) != first["validation_positive_rows"]:
        raise AssertionError("Primary validation population changed")
    if len(plan["tools"]) != 20:
        raise AssertionError("Tool population changed")
    return plan, truth, probability, time, tool


def starts_for_tool(hit_indices, times, cooldown_seconds):
    return hit_indices[np.r_[True, np.diff(times[hit_indices]) > cooldown_seconds]] if len(hit_indices) else hit_indices


def split_cutoffs(plan, time, tool):
    result = {}
    for code, row in enumerate(plan["tools"]):
        observed = time[tool == code]
        midpoint = int(observed[len(observed)//2])
        result[row["tool_id"]] = midpoint
    return result


def event_subset(events, split_cutoff, period):
    if period == "train":
        return [e for e in events if e["fault_time"] < split_cutoff[e["tool_id"]]]
    if period == "validation":
        return [e for e in events if e["fault_time"] >= split_cutoff[e["tool_id"]]]
    raise ValueError(period)


def event_detection(events, onset_times_by_tool):
    detected = 0
    for event in events:
        onset = onset_times_by_tool[event["tool_id"]]
        t = event["fault_time"]
        left = np.searchsorted(onset, t-HORIZON, side="left")
        right = np.searchsorted(onset, t, side="left")
        detected += right > left
    return detected


def _lookback(t, values, i, minutes, start):
    left = max(int(np.searchsorted(t, t[i]-minutes*60, side="left")), start)
    return values[left:i+1], left


def prepare():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    plan, truth, probability, time, tool = context()
    events = validation_events(plan)
    if len(events) != 264:
        raise AssertionError("Forward validation fault population changed")
    names = {code: row["tool_id"] for code, row in enumerate(plan["tools"])}
    hit = probability >= THRESHOLD
    cooldown_rows = []
    for minutes in COOLDOWNS:
        c = minutes * 60
        false_only = 0
        all_starts = np.zeros(len(time), dtype=bool)
        for code in names:
            positions = np.flatnonzero(tool == code)
            selected = positions[hit[positions]]
            start = starts_for_tool(selected, time, c)
            all_starts[start] = True
            negative = positions[hit[positions] & (truth[positions] == 0)]
            false_only += len(starts_for_tool(negative, time, c))
        reverse = {name: code for code, name in names.items()}
        detection, details = event_metrics(.5, events, all_starts.astype(np.uint8), time, tool, reverse)
        if minutes == 10 and false_only != 6437:
            raise AssertionError("10-minute false-only episode reference changed")
        cooldown_rows.append({"cooldown_minutes": minutes,
            "false_label_only_merged_episodes": false_only,
            "all_alarm_operational_episode_starts": int(all_starts.sum()),
            "operational_false_label_episode_starts": int(np.sum(all_starts & (truth == 0))),
            "detected_fault_events_using_only_episode_start": detection["detected_fault_events"],
            "fault_event_recall_using_only_episode_start": detection["fault_event_recall_all_logged"],
            "original_row_alarm_detected_events": 202,
            "additional_missed_events_vs_row_alarm": 202-detection["detected_fault_events"]})
        if minutes == 10:
            onset_mask_10 = all_starts.copy()
    # Train-only normal references from the earlier first-stage Train split.
    reference = joblib.load(SOURCE / "train_normal_references.joblib")
    split = split_cutoffs(plan, time, tool)
    candidate_rows = []
    for code, row in enumerate(plan["tools"]):
        name = row["tool_id"]
        positions = np.flatnonzero(tool == code)
        t, p = time[positions], probability[positions]
        local_onsets = np.flatnonzero(onset_mask_10[positions])
        # The parquet reader stops before the reserved test boundary.
        path = BASELINE / "prepared" / f"{name}.parquet"
        frame = read_pretest(path, row["test_boundary_index"], ["time", *SENSOR_COLUMNS])
        source_time = frame.time.to_numpy(dtype=np.int64)
        sensor_index = np.searchsorted(source_time, t)
        if not np.array_equal(source_time[sensor_index], t):
            raise AssertionError(f"Score/sensor alignment failed: {name}")
        vals = {col: frame[col].to_numpy()[sensor_index] for col in SENSOR_COLUMNS}
        session_start = np.maximum.accumulate(np.where(np.r_[True, np.diff(t)>120], np.arange(len(t)), 0))
        fault_times = np.array(sorted(set(e["fault_time"] for e in events if e["tool_id"] == name)), dtype=np.int64)
        for i in local_onsets:
            if t[i]+HORIZON>=row["test_start_second"]:
                raise AssertionError("An alarm lacks complete 24h pre-test outcome follow-up")
            previous = int(session_start[i])
            p30, _ = _lookback(t, p, i, 30, previous)
            feature = {"tool_id": name, "tool_code": code, "time": int(t[i]),
                       "truth_at_onset": int(truth[positions[i]]),
                       "first_stage_probability": float(p[i]),
                       "prob_max_30m": float(np.max(p30)), "prob_mean_30m": float(np.mean(p30)),
                       "prob_median_30m": float(np.median(p30)),
                       "alarm_duration_minutes_at_onset": 0.0,
                       "FLOWCOOLFLOWRATE": float(vals["FLOWCOOLFLOWRATE"][i]),
                       "FLOWCOOLPRESSURE": float(vals["FLOWCOOLPRESSURE"][i]),
                       "stage": str(int(vals["stage"][i]))}
            for minutes in (5, 10, 30):
                window, left = _lookback(t, p, i, minutes, previous)
                duration = (t[i]-t[left])/60
                feature[f"prob_slope_{minutes}m"] = float((p[i]-p[left])/duration) if duration>0 else np.nan
                feature[f"hit_count_{minutes}m"] = int(np.sum(window>=THRESHOLD))
            for col in SENSOR_COLUMNS:
                if col.endswith("_std_5m") or col.endswith("_std_30m"):
                    feature[col] = float(vals[col][i])
            for sensor, short in (("FLOWCOOLFLOWRATE", "flow"), ("FLOWCOOLPRESSURE", "pressure")):
                _, left = _lookback(t, p, i, 5, previous)
                feature[f"{short}_delta_5m"] = float(vals[sensor][i]-vals[sensor][left]) if left<i else np.nan
            flow_ref = reference[name]["FLOWCOOLFLOWRATE"]
            pressure_ref = reference[name]["FLOWCOOLPRESSURE"]
            feature["flow_tool_normal_percentile"] = float(np.searchsorted(
                flow_ref["sorted"], feature["FLOWCOOLFLOWRATE"], side="right")/len(flow_ref["sorted"]))
            feature["pressure_tool_robust_zscore"] = float((feature["FLOWCOOLPRESSURE"]-
                pressure_ref["median"])/(pressure_ref["iqr"]/1.349))
            future = np.searchsorted(fault_times, t[i], side="right")
            time_until_fault = int(fault_times[future]-t[i]) if future<len(fault_times) else None
            # Hard negatives need known outcome through 72h before the sealed test boundary.
            if time_until_fault is not None and 0<time_until_fault<=HORIZON:
                label = "positive"
            elif (time_until_fault is None or time_until_fault>=HARD_NEGATIVE_HORIZON) and \
                    t[i]+HARD_NEGATIVE_HORIZON<row["test_start_second"]:
                label = "hard_negative"
            elif time_until_fault is not None and HORIZON<time_until_fault<HARD_NEGATIVE_HORIZON:
                label = "ambiguous_24_to_72h"
            else:
                label = "censored_before_72h"
            period = ("second_train" if t[i]+HARD_NEGATIVE_HORIZON<split[name]
                      else "purge" if t[i]<split[name] else "second_validation")
            feature.update({"label_group": label, "split": period,
                "hours_until_next_logged_fault": time_until_fault/3600 if time_until_fault is not None else None})
            candidate_rows.append(feature)
        print(f"{name}: {len(local_onsets)} operational 10m alarm starts", flush=True)
    candidates = pd.DataFrame(candidate_rows)
    if len(candidates) != cooldown_rows[0]["all_alarm_operational_episode_starts"]:
        raise AssertionError("Some 10-minute onset candidates disappeared")
    OUT.mkdir(parents=True)
    write_csv(OUT / "cooldown_comparison.csv", cooldown_rows, list(cooldown_rows[0]))
    candidates.to_parquet(OUT / "alarm_event_candidates.parquet", index=False)
    counts = candidates.groupby(["split", "label_group"], dropna=False).size().rename("alarm_events").reset_index()
    counts.to_csv(OUT / "alarm_label_counts.csv", index=False, encoding="utf-8-sig")
    (OUT / "preparation_report.json").write_text(json.dumps({
        "primary_model_sha256": file_hash(PRIMARY_MODEL), "primary_score_sha256": file_hash(PRIMARY_SCORE),
        "train_normal_reference_sha256": file_hash(SOURCE / "train_normal_references.joblib"),
        "primary_policy": "threshold 0.34, one observed minute hit; frozen before this experiment",
        "fault_events": len(events), "first_stage_row_alarm_detected": 202,
        "candidate_definition": "First threshold-hit row of all-alarm per-tool episode, >10m gap starts a new episode. Episode-onset features use current and previous observed rows only.",
        "false_only_count_caveat": "6437 is obtained by merging negative-label hit rows alone; operational all-alarm episode starts differ. These counts must not be interchanged.",
        "label_definition": "Positive if next logged target fault is within (0,24h]; hard negative if >=72h away or absent with 72h known follow-up before test boundary; 24-72h ambiguous excluded from training; insufficient 72h follow-up censored.",
        "second_stage_split": "Per tool, first half of primary validation scores for second-stage training, later half for second-stage validation; >=72h embargo before each second-stage cutoff for train hard negatives.",
        "first_stage_tuning_caveat": "The primary threshold 0.34 was previously chosen on this whole primary validation. This split reduces second-stage fitting leakage but cannot make the reused interval a fresh independent test.",
        "onset_duration_caveat": "Alarm duration at the episode start is zero by definition; it is retained in the audit but omitted from fitted predictors.",
        "test_note": "Reserved test outcomes and scores are never evaluated here; shared historical exposure remains as documented in forward_validation_ab_24h."},
        ensure_ascii=False, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)), encoding="utf-8")
    print(counts.to_string(index=False), flush=True)


def fit():
    if (OUT / "experiment_report.json").exists():
        raise FileExistsError("Second-stage experiment report exists")
    prep = json.loads((OUT / "preparation_report.json").read_text(encoding="utf-8"))
    if file_hash(PRIMARY_MODEL) != prep["primary_model_sha256"] or file_hash(PRIMARY_SCORE) != prep["primary_score_sha256"]:
        raise AssertionError("Primary model or score changed")
    candidates = pd.read_parquet(OUT / "alarm_event_candidates.parquet")
    train = candidates[(candidates.split=="second_train") & candidates.label_group.isin(("positive", "hard_negative"))]
    valid = candidates[candidates.split=="second_validation"]
    if train.label_group.nunique()!=2 or len(valid)==0:
        raise AssertionError("Insufficient second-stage candidates")
    features = [col for col in NUMERIC if col!="alarm_duration_minutes_at_onset"]
    cols = [*features, *CATEGORICAL]
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                               ("scale", StandardScaler())]), features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        sparse_threshold=0.3)
    target = (train.label_group=="positive").astype(np.int8)
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "xgboost": XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            n_estimators=120, max_depth=3, learning_rate=.05, subsample=.8, colsample_bytree=.8,
            scale_pos_weight=float((target==0).sum()/int(target.sum())), random_state=42, n_jobs=8)}
    plan, truth, primary_prob, time, tool = context()
    event_log = validation_events(plan)
    split = split_cutoffs(plan, time, tool)
    later_events = event_subset(event_log, split, "validation")
    if not later_events:
        raise AssertionError("No later secondary validation events")
    names = {code: row["tool_id"] for code, row in enumerate(plan["tools"])}
    cooldown_later = []
    for minutes in COOLDOWNS:
        onsets = {}
        false_count = 0
        for code, name in names.items():
            hit_rows = np.flatnonzero((tool==code) & (primary_prob>=THRESHOLD))
            starts = starts_for_tool(hit_rows, time, minutes*60)
            starts = starts[time[starts]>=split[name]]
            onsets[name] = time[starts]
            false_count += int(np.count_nonzero(truth[starts]==0))
        detected_cooldown = event_detection(later_events, onsets)
        cooldown_later.append({"cooldown_minutes": minutes,
            "later_fault_events_total": len(later_events),
            "later_fault_events_detected": detected_cooldown,
            "later_fault_event_recall": detected_cooldown/len(later_events),
            "later_false_alarm_episode_starts": false_count})
    write_csv(OUT / "cooldown_second_validation_comparison.csv", cooldown_later, list(cooldown_later[0]))
    baseline_onsets = {name: np.sort(valid.loc[valid.tool_id==name, "time"].to_numpy(dtype=np.int64)) for name in names.values()}
    before = event_detection(later_events, baseline_onsets)
    # Within the late validation window, only alarms whose next fault is >24h
    # away are false alarms. Ambiguous 24-72h alarms remain false for evaluation.
    baseline_false = int((valid.label_group!="positive").sum())
    predictions = []
    rows = []
    distributions = []
    for model_name, estimator in models.items():
        pipeline = Pipeline([("preprocess", preprocessing), ("model", estimator)])
        pipeline.fit(train[cols], target)
        proba = pipeline.predict_proba(valid[cols])[:,1]
        # Event detection changes only when the maximum alarm score in an
        # event's preceding 24h is crossed; evaluate those breakpoints.
        event_maxima = []
        for event in later_events:
            mask = ((valid.tool_id.to_numpy()==event["tool_id"]) &
                    (valid.time.to_numpy()>=event["fault_time"]-HORIZON) &
                    (valid.time.to_numpy()<event["fault_time"]))
            event_maxima.append(float(np.max(proba[mask])) if np.any(mask) else -1.0)
        event_maxima = np.asarray(event_maxima)
        thresholds = np.unique(np.r_[0.0, event_maxima[event_maxima>=0], 1.0])
        best = None
        for threshold in thresholds:
            keep = proba>=threshold
            detected = int(np.count_nonzero(event_maxima>=threshold))
            if detected/len(later_events)<.75:
                continue
            false = int(np.count_nonzero(keep & (valid.label_group.to_numpy()!="positive")))
            candidate = (false, -detected, -float(threshold))
            if best is None or candidate<best[0]:
                best = (candidate, float(threshold), keep, detected, false)
        if best is None:
            rows.append({"model": model_name, "threshold_found": False,
                         "baseline_fault_events_detected": before, "total_later_fault_events": len(later_events),
                         "baseline_false_alarm_events": baseline_false})
            print(f"{model_name}: no >=75% event-recall threshold", flush=True)
            continue
        _, chosen, keep, detected, false = best
        saved = ROOT / "output" / "models" / f"second_stage_{model_name}_24h.joblib"
        if saved.exists():
            raise FileExistsError(saved)
        joblib.dump({"pipeline": pipeline, "features": cols, "threshold": chosen,
                     "train_label_definition": prep["label_definition"]}, saved)
        tool_detail = valid.assign(kept=keep, false=(valid.label_group!="positive"))
        for name, group in tool_detail.groupby("tool_id"):
            distributions.append({"model": model_name, "tool_id": name,
                "before_false_alarm_events": int(group.false.sum()),
                "after_false_alarm_events": int((group.false & group.kept).sum())})
        predictions.extend({"model": model_name, "tool_id": r.tool_id, "time": int(r.time),
            "label_group": r.label_group, "score": float(s), "kept": bool(k)}
            for r,s,k in zip(valid.itertuples(), proba, keep))
        rows.append({"model": model_name, "threshold_found": True, "selected_threshold": chosen,
            "baseline_fault_events_detected": before, "detected_fault_events_after": detected,
            "total_later_fault_events": len(later_events), "event_recall_after": detected/len(later_events),
            "baseline_false_alarm_events": baseline_false, "false_alarm_events_after": false,
            "false_alarm_reduction_events": baseline_false-false,
            "false_alarm_reduction_fraction": (baseline_false-false)/baseline_false if baseline_false else None,
            "net_additional_missed_faults": before-detected,
            "second_stage_train_alarm_events": len(train),
            "second_stage_train_positive": int(target.sum()),
            "second_stage_train_hard_negative": int((target==0).sum()),
            "second_stage_validation_alarm_events": len(valid)})
        print(f"{model_name}: {detected}/{len(later_events)} faults, {baseline_false}->{false} false events", flush=True)
    write_csv(OUT / "second_stage_model_summary.csv", rows, list(dict.fromkeys(k for r in rows for k in r)))
    if distributions:
        write_csv(OUT / "false_alarm_by_tool.csv", distributions, list(distributions[0]))
    if predictions:
        write_csv(OUT / "secondary_validation_predictions.csv", predictions, list(predictions[0]))
    (OUT / "experiment_report.json").write_text(json.dumps({
        "cooldown": pd.read_csv(OUT / "cooldown_comparison.csv").to_dict("records"),
        "cooldown_on_same_later_validation": cooldown_later,
        "label_counts": pd.read_csv(OUT / "alarm_label_counts.csv").to_dict("records"),
        "model_summary": rows, "primary_model_unchanged": True,
        "secondary_later_event_count": len(later_events), "secondary_later_baseline_detected": before,
        "secondary_later_baseline_event_recall": before/len(later_events),
        "selection_caveat": "Second-stage thresholds were selected and measured on the same later secondary validation; these are exploratory, not final out-of-sample results. The primary .34 threshold was earlier chosen on the full interval."},
        ensure_ascii=False, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "fit"))
    action = parser.parse_args().action
    {"prepare": prepare, "fit": fit}[action]()
