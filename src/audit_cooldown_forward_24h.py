"""Audit frozen forward Model A scores under a manager-notification cooldown.

The model's threshold decisions are never changed. A raw alarm is the first
threshold hit after a >10 minute gap for the same tool. The manager receives
that alert unless the last notification for the tool was less than 30 minutes
ago. All threshold hits and raw alarms remain in the audit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_event_thresholds_24h import write_csv
from experiment_tool_stage_interaction_24h import file_hash
from forward_validation_ab_24h import validation_events

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/experiments/forward_validation_ab_24h"
OUT = ROOT / "output/experiments/cooldown_forward_validation_24h"
MODEL = ROOT / "output/models/xgb_24h_forward_a.joblib"
SCORES = SOURCE / "validation_probabilities_a.npz"
THRESHOLD = .34
HORIZON = 86400
GAP = 86400
RAW_GAP = 600
COOLDOWN = 1800
FOLDS = 3


def first_after_gap(indices, times, seconds):
    if not len(indices):
        return indices
    return indices[np.r_[True, np.diff(times[indices]) > seconds]]


def notify(raw_indices, times, seconds):
    kept, suppressed, last = [], [], None
    for index in raw_indices:
        now = int(times[index])
        if last is None or now-last >= seconds:
            kept.append(index)
            last = now
        else:
            suppressed.append(index)
    return np.asarray(kept, dtype=np.int64), np.asarray(suppressed, dtype=np.int64)


def event_details(events, alarm_times):
    rows = []
    for event in events:
        t = event["fault_time"]
        times = alarm_times.get(event["tool_id"], np.empty(0, dtype=np.int64))
        i = int(np.searchsorted(times, t-HORIZON, side="left"))
        first = int(times[i]) if i < len(times) and times[i] < t else None
        rows.append({**event, "detected": first is not None,
                     "first_alert_time": first,
                     "lead_minutes": (t-first)/60 if first is not None else None})
    return rows


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {OUT}")
    plan = json.loads((SOURCE / "split_plan.json").read_text(encoding="utf-8"))
    training = json.loads((SOURCE / "training_a.json").read_text(encoding="utf-8"))
    if file_hash(MODEL) != training["model_sha256"] or file_hash(SCORES) != training["validation_score_sha256"]:
        raise AssertionError("Frozen Model A or validation scores changed")
    with np.load(SCORES) as saved:
        truth, probability, times, tool = (saved[k].copy() for k in
            ("truth", "probability", "time", "tool_code"))
    if len(times) != training["validation_rows"] or int(truth.sum()) != training["validation_positive_rows"]:
        raise AssertionError("Validation population changed")
    if len(plan["tools"]) != 20:
        raise AssertionError("Tool population changed")
    events = validation_events(plan)
    names = {code: item["tool_id"] for code, item in enumerate(plan["tools"])}
    boundaries = {}
    for code, name in names.items():
        local = np.sort(times[tool == code])
        boundaries[name] = np.quantile(local, [0, 1/FOLDS, 2/FOLDS, 1], method="nearest").astype(np.int64)
    fold_rows, event_rows, escalation_rows, exposure_rows = [], [], [], []
    for fold in range(FOLDS):
        raw_by_tool, notified_by_tool, hits_by_tool = {}, {}, {}
        fold_events = []
        false_raw, false_notify, n_rows = 0, 0, 0
        fold_start, fold_end = {}, {}
        for code, name in names.items():
            edges = boundaries[name]
            # Exclude 24h on both sides of every internal fold boundary.
            start = int(edges[fold] + (GAP if fold else 0))
            end = int(edges[fold+1] - (GAP if fold < FOLDS-1 else 0))
            if end <= start:
                raise AssertionError(f"No fold exposure for {name} fold {fold+1}")
            fold_start[name], fold_end[name] = start, end
            exposure_rows.append({"fold": fold+1, "tool_id": name,
                "start_second": start, "end_exclusive_second": end,
                "left_boundary_gap_seconds": GAP if fold else 0,
                "right_boundary_gap_seconds": GAP if fold < FOLDS-1 else 0})
            positions = np.flatnonzero((tool == code) & (times >= start) & (times < end))
            n_rows += len(positions)
            hits = positions[probability[positions] >= THRESHOLD]
            raw = first_after_gap(hits, times, RAW_GAP)
            notice, suppressed = notify(raw, times, COOLDOWN)
            hits_by_tool[name], raw_by_tool[name], notified_by_tool[name] = (times[hits], times[raw], times[notice])
            false_raw += int(np.count_nonzero(truth[raw] == 0))
            false_notify += int(np.count_nonzero(truth[notice] == 0))
            for index in suppressed:
                previous = raw[np.searchsorted(raw, index)-1]
                delta = float(probability[index] - probability[previous])
                escalation_rows.append({"fold": fold+1, "tool_id": name, "time": int(times[index]),
                    "probability": float(probability[index]), "rise_from_previous_raw_alarm": delta,
                    "large_rise_ge_0_2": delta >= .2,
                    "true_positive_row": bool(truth[index] == 1)})
            # Full 24h event lookback must be inside this fold; gap events excluded.
            fold_events.extend(e for e in events if e["tool_id"] == name and
                start+HORIZON <= e["fault_time"] < end)
        hit_details = event_details(fold_events, hits_by_tool)
        raw_details = event_details(fold_events, raw_by_tool)
        notified_details = event_details(fold_events, notified_by_tool)
        raw_detected = sum(d["detected"] for d in raw_details)
        notification_detected = sum(d["detected"] for d in notified_details)
        hit_detected = sum(d["detected"] for d in hit_details)
        leads_a = [d["lead_minutes"] for d in raw_details if d["detected"]]
        leads_b = [d["lead_minutes"] for d in notified_details if d["detected"]]
        fold_rows.append({"fold": fold+1, "scored_rows": n_rows,
            "fault_events": len(fold_events), "row_prediction_detected": hit_detected,
            "raw_alarm_detected": raw_detected, "raw_event_recall": raw_detected/len(fold_events) if fold_events else None,
            "notification_detected": notification_detected,
            "notification_event_recall": notification_detected/len(fold_events) if fold_events else None,
            "raw_false_alarm_events": false_raw, "no_cooldown_false_notifications": false_raw,
            "cooldown_false_notifications": false_notify,
            "false_notification_reduction": false_raw-false_notify,
            "false_notification_reduction_fraction": (false_raw-false_notify)/false_raw if false_raw else None,
            "additional_missed_vs_no_cooldown": raw_detected-notification_detected,
            "raw_first_lead_median_minutes": float(np.median(leads_a)) if leads_a else None,
            "notification_first_lead_median_minutes": float(np.median(leads_b)) if leads_b else None})
        for original, notice, hit in zip(raw_details, notified_details, hit_details):
            event_rows.append({"fold": fold+1, "tool_id": original["tool_id"],
                "fault_type": original["fault_type"], "fault_time": original["fault_time"],
                "row_prediction_detected": hit["detected"],
                "raw_alarm_detected": original["detected"], "notification_detected": notice["detected"],
                "lost_due_to_cooldown": original["detected"] and not notice["detected"],
                "raw_first_lead_minutes": original["lead_minutes"],
                "notification_first_lead_minutes": notice["lead_minutes"]})
    OUT.mkdir(parents=True)
    write_csv(OUT / "fold_metrics.csv", fold_rows, list(fold_rows[0]))
    write_csv(OUT / "fault_event_details.csv", event_rows, list(event_rows[0]))
    write_csv(OUT / "fold_exposure.csv", exposure_rows, list(exposure_rows[0]))
    if escalation_rows:
        write_csv(OUT / "suppressed_alarm_rise_audit.csv", escalation_rows, list(escalation_rows[0]))
    report = {"policy": "Frozen Model A, 85 features, threshold .34, 1/1 minute; raw 10m gap alarm starts; per-tool 30m manager notification cooldown from last sent notification; predictions preserved",
        "folds": FOLDS, "fold_boundary_exclusion_seconds_each_side": GAP,
        "event_full_lookback_seconds": HORIZON,
        "all_validation_fault_events": len(events),
        "fold_evaluable_fault_events": len(event_rows),
        "events_excluded_by_fold_boundary_and_full_lookback": len(events)-len(event_rows),
        "validation_only": True, "test_scored": False,
        "model_sha256": file_hash(MODEL), "validation_score_sha256": file_hash(SCORES),
        "fold_metrics": fold_rows,
        "suppressed_raw_alarm_count": len(escalation_rows),
        "suppressed_large_probability_rise_ge_0_2": sum(r["large_rise_ge_0_2"] for r in escalation_rows),
        "suppressed_true_positive_rows": sum(r["true_positive_row"] for r in escalation_rows),
        "escalation_caveat": "A >=0.2 probability rise is exploratory, measured against the previous raw alarm, not a validated notification override. Warning-to-danger transition cannot be measured from this single 24h risk score; no stage-change proxy was substituted. New fault possibility is audited retrospectively via event-level losses, not inferred from future outcomes online.",
        "independence_caveat": "These chronological folds reuse a previously selected validation set and threshold; they measure stability only, not independent prospective performance. The historically exposed reserved test remains unopened."}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(fold_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
