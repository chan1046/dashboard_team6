"""Evaluate causal k-of-W alarm persistence on saved 24h validation scores.

The recovered XGBoost model and features are untouched. Alarm decisions use
only the current and earlier observed minutes of the same operating session.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate_event_thresholds_24h import (
    BASELINE, FAULT_LOG, ROOT, SOURCE, event_metrics, load_events,
    read_json, row_and_alarm_metrics, write_csv,
)

OUT = ROOT / "output" / "experiments" / "alarm_persistence_24h"
THRESHOLD = 0.5
CONDITIONS = (
    ("1_of_1m", 1, 1),
    ("2_of_3m", 3, 2),
    ("2_of_5m", 5, 2),
    ("3_of_5m", 5, 3),
    ("3_of_10m", 10, 3),
    ("5_of_10m", 10, 5),
)


def causal_alarm(raw_hit: np.ndarray, time: np.ndarray, tool: np.ndarray,
                 tool_codes: list[int], window_minutes: int, minimum_hits: int) -> np.ndarray:
    """One alarm decision per observed row; no gap filling or future rows."""
    if (window_minutes, minimum_hits) == (1, 1):
        return raw_hit.copy()
    result = np.zeros(len(raw_hit), dtype=bool)
    for code in tool_codes:
        positions = np.flatnonzero(tool == code)
        tool_times = time[positions]
        buckets = tool_times // 60
        if len(np.unique(buckets)) != len(buckets):
            raise ValueError(f"Tool code {code}: more than one scored row in a minute")
        breaks = np.flatnonzero(np.diff(tool_times) > 120) + 1
        for start, stop in zip(np.r_[0, breaks], np.r_[breaks, len(positions)]):
            chosen = positions[start:stop]
            minute = buckets[start:stop]
            hit = raw_hit[chosen]
            left = np.searchsorted(minute, minute - (window_minutes - 1), side="left")
            cumulative = np.r_[0, np.cumsum(hit, dtype=np.int32)]
            hit_count = cumulative[1:] - cumulative[left]
            result[chosen] = hit_count >= minimum_hits
    return result


def main() -> None:
    sweep = read_json(SOURCE / "sweep_report.json")
    baseline = read_json(BASELINE / "reproduction_report.json")
    prior_events = read_json(ROOT / "output" / "experiments" / "event_level_threshold_24h" / "event_threshold_report.json")
    preparation = read_json(BASELINE / "preparation_report.json")
    source_scores = SOURCE / "validation_probabilities.npz"
    with np.load(source_scores) as saved:
        truth = saved["truth"]
        probability = saved["probability"]
        time = saved["time"]
        tool = saved["tool_code"]
    if hashlib.sha256((BASELINE / "xgb_24h_recovery.joblib").read_bytes()).hexdigest() != sweep["model_sha256"]:
        raise AssertionError("The baseline model changed after validation probabilities were saved")
    if len(truth) != sweep["validation_rows"] or int(truth.sum()) != sweep["validation_positive_rows"]:
        raise AssertionError("Saved validation population changed")

    tool_map = {int(code): name for code, name in sweep["tool_code_map"].items()}
    reverse_map = {name: code for code, name in tool_map.items()}
    cutoffs = {item["tool"]: item["split_cutoff_time"] for item in preparation["tools"]}
    events = load_events(cutoffs, set(tool_map.values()))
    raw_hit = probability >= THRESHOLD
    summary, event_rows, missed_rows, tool_rows = [], [], [], []
    baseline_event = None
    baseline_false = None
    baseline_leads: dict[tuple[str, str, int], float] | None = None
    for label, window, required in CONDITIONS:
        alarm = causal_alarm(raw_hit, time, tool, sorted(tool_map), window, required)
        # The shared event/false-alarm evaluator accepts a score and threshold.
        # A 0/1 alarm array reproduces exactly the filtered decision sequence.
        alarm_score = alarm.astype(np.uint8)
        row, distribution = row_and_alarm_metrics(0.5, truth, alarm_score, time, tool,
                                                  sweep["average_precision_fixed"], tool_map)
        event, details = event_metrics(0.5, events, alarm_score, time, tool, reverse_map)
        current_leads = {
            (detail["tool_id"], detail["fault_type"], detail["fault_time"]): detail["lead_time_minutes"]
            for detail in details if detail["detected_before_fault"]
        }
        if label == "1_of_1m":
            expected = baseline["metrics"]
            for key in ("tn", "fp", "fn", "tp", "false_alarm_events_10m"):
                if row[key] != expected[key]:
                    raise AssertionError(f"Baseline mismatch: {key}")
            prior = next(item for item in prior_events["summary"] if item["threshold"] == 0.5)
            for key in ("detected_fault_events", "total_logged_validation_fault_events", "evaluable_fault_events"):
                if event[key] != prior[key]:
                    raise AssertionError(f"Prior event-level result mismatch: {key}")
            baseline_event = event["detected_fault_events"]
            baseline_false = row["false_alarm_events_10m"]
            baseline_leads = current_leads
        elif baseline_event is None or baseline_false is None or baseline_leads is None:
            raise AssertionError("Baseline condition must be evaluated first")
        assert baseline_leads is not None
        shared = baseline_leads.keys() & current_leads.keys()
        lead_delays = [baseline_leads[key] - current_leads[key] for key in shared]
        item = {
            "condition": label, "threshold": THRESHOLD,
            "window_minutes": window, "minimum_threshold_exceedances": required,
            **row, **event,
            "false_alarm_events_reduced_vs_baseline": baseline_false - row["false_alarm_events_10m"],
            "false_alarm_reduction_fraction_vs_baseline": (baseline_false - row["false_alarm_events_10m"]) / baseline_false,
            "additional_missed_fault_events_vs_baseline": len(baseline_leads.keys() - current_leads.keys()),
            "newly_detected_fault_events_vs_baseline": len(current_leads.keys() - baseline_leads.keys()),
            "paired_first_alert_delay_median_minutes": float(np.median(lead_delays)) if lead_delays else None,
            "paired_first_alert_delay_max_minutes": float(np.max(lead_delays)) if lead_delays else None,
        }
        summary.append(item)
        for detail in details:
            detail["condition"] = label
            event_rows.append(detail)
            if not detail["detected_before_fault"]:
                missed_rows.append(detail)
        for item_by_tool in distribution:
            item_by_tool["condition"] = label
            tool_rows.append(item_by_tool)
        print(f"{label}: detected={event['detected_fault_events']}/{event['total_logged_validation_fault_events']}, false_alarm_events={row['false_alarm_events_10m']}", flush=True)

    if len(summary) != len(CONDITIONS) or len(event_rows) != len(events) * len(CONDITIONS):
        raise AssertionError("Some condition or fault event was not evaluated")
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "alarm_persistence_summary.csv", summary, list(summary[0]))
    write_csv(OUT / "fault_event_details.csv", event_rows, list(event_rows[0]))
    write_csv(OUT / "missed_fault_events.csv", missed_rows, list(event_rows[0]))
    write_csv(OUT / "false_alarm_by_tool.csv", tool_rows, list(tool_rows[0]))
    report = {
        "source_scores": str(source_scores),
        "source_scores_sha256": hashlib.sha256(source_scores.read_bytes()).hexdigest(),
        "model_sha256": sweep["model_sha256"],
        "model_retrained": False, "features_changed": False,
        "threshold": THRESHOLD,
        "persistence_definition": "At each observed minute, alarm if at least k scores including the current row exceed 0.5 in the current and preceding W-1 clock-minute buckets. Reset after a >120-second genuine tool stop; do not fill absent minutes. Alarm may persist briefly after the last raw exceedance while it remains inside the trailing window.",
        "event_definition": prior_events["event_definition"],
        "false_alarm_definition": prior_events["false_alarm_definition"],
        "observed_hours_definition": prior_events["observed_hours_definition"],
        "no_independent_test_used": True,
        "selection_note": "All conditions are compared on the validation data already used for threshold choice. A selected alarm rule requires a later untouched evaluation period before operational claims.",
        "summary": summary,
    }
    (OUT / "alarm_persistence_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved results to {OUT}", flush=True)


if __name__ == "__main__":
    main()
