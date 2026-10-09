"""Event-level audit of three fixed thresholds on saved 24h validation scores.

No model is loaded or fitted. The only prediction input is the score array
saved by threshold_sweep_xgb24.py and checked against the recovered baseline.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output" / "experiments" / "threshold_sweep_24h"
BASELINE = ROOT / "output" / "baseline_recovery"
DEST = ROOT / "output" / "experiments" / "event_level_threshold_24h"
FAULT_LOG = ROOT / "data" / "processed" / "fault_events.csv"
THRESHOLDS = (0.3406, 0.5, 0.6911)
HORIZON = 86400
FAULT_TYPES = {"pressure_low", "pressure_high", "leak"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_events(cutoffs: dict[str, int], tool_names: set[str]) -> list[dict]:
    result: dict[tuple[str, str, int], dict] = {}
    with FAULT_LOG.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            tool = row["tool_id"]
            fault_type = row["fault_type"]
            if tool not in tool_names or fault_type not in FAULT_TYPES:
                continue
            fault_time = int(row["time"])
            if fault_time < cutoffs[tool]:
                continue
            key = (tool, fault_type, fault_time)
            result[key] = {"tool_id": tool, "fault_type": fault_type, "fault_time": fault_time}
    return sorted(result.values(), key=lambda e: (e["tool_id"], e["fault_time"], e["fault_type"]))


def row_and_alarm_metrics(threshold: float, truth: np.ndarray, score: np.ndarray,
                          time: np.ndarray, tool: np.ndarray, ap: float,
                          tool_map: dict[int, str]) -> tuple[dict, list[dict]]:
    alert = score >= threshold
    positive = truth == 1
    tp = int(np.count_nonzero(alert & positive))
    fp = int(np.count_nonzero(alert & ~positive))
    fn = int(np.count_nonzero(positive)) - tp
    tn = len(truth) - int(np.count_nonzero(positive)) - fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    per_tool = []
    for code, name in sorted(tool_map.items()):
        positions = np.flatnonzero(tool == code)
        false_positions = positions[alert[positions] & ~positive[positions]]
        false_times = time[false_positions]
        # Same rule used in the recovered baseline: compare consecutive false
        # alarm rows for each tool, opening a new episode only after >600 s.
        episode_starts = false_times[np.r_[True, np.diff(false_times) > 600]] if len(false_times) else np.empty(0, dtype=np.int64)
        observed_hours = len(positions) / 60.0
        per_tool.append({
            "threshold": threshold, "tool_id": name,
            "validation_labeled_minute_rows": len(positions),
            "observed_hour_equivalents": observed_hours,
            "false_positive_rows": len(false_positions),
            "false_alarm_events_10m": len(episode_starts),
            "false_alarm_events_per_observed_hour": len(episode_starts) / observed_hours if observed_hours else None,
            "false_alarm_events_per_observed_day": len(episode_starts) * 24 / observed_hours if observed_hours else None,
        })
    false_events = sum(row["false_alarm_events_10m"] for row in per_tool)
    observed_hours_total = len(truth) / 60.0
    metrics = {
        "threshold": threshold, "precision": precision, "recall": recall, "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2, "average_precision": ap,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "false_alarm_events_10m": false_events,
        "observed_hour_equivalents": observed_hours_total,
        "false_alarm_events_per_observed_hour": false_events / observed_hours_total,
        "false_alarm_events_per_observed_day": false_events * 24 / observed_hours_total,
    }
    return metrics, per_tool


def event_metrics(threshold: float, events: list[dict], score: np.ndarray,
                  time: np.ndarray, tool: np.ndarray,
                  code_by_tool: dict[str, int]) -> tuple[dict, list[dict]]:
    positions_by_tool = {name: np.flatnonzero(tool == code) for name, code in code_by_tool.items()}
    details = []
    for event in events:
        positions = positions_by_tool[event["tool_id"]]
        tool_times = time[positions]
        fault_time = event["fault_time"]
        left = int(np.searchsorted(tool_times, fault_time - HORIZON, side="left"))
        right = int(np.searchsorted(tool_times, fault_time, side="left"))
        window = positions[left:right]
        alarms = window[score[window] >= threshold]
        first_time = int(time[alarms[0]]) if len(alarms) else None
        details.append({
            "threshold": threshold, **event,
            "sensor_rows_in_24h_before_fault": len(window),
            "evaluable_from_validation_scores": bool(len(window)),
            "detected_before_fault": bool(len(alarms)),
            "first_alert_time": first_time,
            "lead_time_seconds": fault_time - first_time if first_time is not None else None,
            "lead_time_minutes": (fault_time - first_time) / 60 if first_time is not None else None,
            "miss_reason": "" if len(alarms) else "no_validation_sensor_in_window" if not len(window) else "no_alarm_at_threshold",
        })
    leads = np.array([r["lead_time_minutes"] for r in details if r["lead_time_minutes"] is not None], dtype=float)
    detected = int(sum(r["detected_before_fault"] for r in details))
    evaluable = int(sum(r["evaluable_from_validation_scores"] for r in details))
    result = {
        "total_logged_validation_fault_events": len(details),
        "evaluable_fault_events": evaluable,
        "not_evaluable_no_validation_sensor_in_24h": len(details) - evaluable,
        "detected_fault_events": detected,
        "missed_fault_events_including_no_sensor": len(details) - detected,
        "missed_evaluable_fault_events": evaluable - detected,
        "fault_event_recall_all_logged": detected / len(details) if details else None,
        "fault_event_recall_evaluable": detected / evaluable if evaluable else None,
        "lead_time_median_minutes": float(np.median(leads)) if len(leads) else None,
        "lead_time_min_minutes": float(np.min(leads)) if len(leads) else None,
        "lead_time_max_minutes": float(np.max(leads)) if len(leads) else None,
    }
    return result, details


def main() -> None:
    sweep = read_json(SOURCE / "sweep_report.json")
    baseline = read_json(BASELINE / "reproduction_report.json")
    prep = read_json(BASELINE / "preparation_report.json")
    score_file = SOURCE / "validation_probabilities.npz"
    with np.load(score_file) as saved:
        truth = saved["truth"]
        score = saved["probability"]
        time = saved["time"]
        tool = saved["tool_code"]
    tool_map = {int(code): name for code, name in sweep["tool_code_map"].items()}
    code_by_tool = {name: code for code, name in tool_map.items()}
    if len(truth) != sweep["validation_rows"] or int(truth.sum()) != sweep["validation_positive_rows"]:
        raise AssertionError("Saved validation scores do not match the threshold sweep population")
    if len(tool_map) != 20 or len(score) != len(time) or len(score) != len(tool):
        raise AssertionError("Saved validation score arrays or tool mapping are incomplete")
    if hashlib.sha256((BASELINE / "xgb_24h_recovery.joblib").read_bytes()).hexdigest() != sweep["model_sha256"]:
        raise AssertionError("Recovered model file changed after scores were saved")

    cutoffs = {item["tool"]: item["split_cutoff_time"] for item in prep["tools"]}
    events = load_events(cutoffs, set(tool_map.values()))
    summaries, details, tool_rows = [], [], []
    for threshold in THRESHOLDS:
        row, distribution = row_and_alarm_metrics(threshold, truth, score, time, tool,
                                                  sweep["average_precision_fixed"], tool_map)
        event, event_detail = event_metrics(threshold, events, score, time, tool, code_by_tool)
        row.update(event)
        summaries.append(row)
        tool_rows.extend(distribution)
        details.extend(event_detail)
        print(f"threshold={threshold}: row_recall={row['recall']:.4f}, events={row['detected_fault_events']}/{row['total_logged_validation_fault_events']}, false_alarm_events={row['false_alarm_events_10m']}", flush=True)

    expected = baseline["metrics"]
    middle = summaries[1]
    for key in ("tn", "fp", "fn", "tp", "false_alarm_events_10m"):
        if middle[key] != expected[key]:
            raise AssertionError(f"Threshold 0.5 baseline mismatch for {key}")
    if not np.isclose(middle["recall"], expected["recall"], atol=1e-12):
        raise AssertionError("Threshold 0.5 baseline recall mismatch")
    if not np.isclose(middle["average_precision"], expected["average_precision"], atol=1e-12):
        raise AssertionError("Threshold 0.5 baseline AP mismatch")
    if len(events) * len(THRESHOLDS) != len(details):
        raise AssertionError("Every fault event must appear once for each threshold")

    def paired_lead(threshold_a: float, threshold_b: float) -> dict:
        def index_for(threshold: float) -> dict:
            return {(r["tool_id"], r["fault_type"], r["fault_time"]): r
                    for r in details if r["threshold"] == threshold and r["detected_before_fault"]}
        a, b = index_for(threshold_a), index_for(threshold_b)
        shared = a.keys() & b.keys()
        differences = np.array([a[key]["lead_time_minutes"] - b[key]["lead_time_minutes"] for key in shared], dtype=float)
        return {
            "threshold_a": threshold_a, "threshold_b": threshold_b,
            "detected_by_both": len(shared),
            "detected_only_by_a": len(a.keys() - b.keys()),
            "detected_only_by_b": len(b.keys() - a.keys()),
            "median_first_alert_lead_difference_a_minus_b_minutes": float(np.median(differences)) if len(differences) else None,
            "min_first_alert_lead_difference_a_minus_b_minutes": float(np.min(differences)) if len(differences) else None,
            "max_first_alert_lead_difference_a_minus_b_minutes": float(np.max(differences)) if len(differences) else None,
        }

    DEST.mkdir(parents=True, exist_ok=True)
    write_csv(DEST / "event_threshold_summary.csv", summaries, list(summaries[0]))
    write_csv(DEST / "fault_event_details.csv", details, list(details[0]))
    missed = [row for row in details if not row["detected_before_fault"]]
    write_csv(DEST / "missed_fault_events.csv", missed, list(details[0]))
    write_csv(DEST / "false_alarm_by_tool.csv", tool_rows, list(tool_rows[0]))

    # The final 20% of each tool's chronological train data is exactly the
    # validation period that supplied the saved score array. Censored tail
    # rows lack a complete 24-hour follow-up, so they are not an untouched test.
    censored = sum(item["validation_censored"] for item in prep["tools"])
    data_test_exists = (ROOT / "data" / "test").exists()
    time_audit = {
        "unused_later_labeled_train_period_exists": False,
        "reason": "All labeled rows after each tool's 80% cutoff were used as validation probabilities and in threshold selection.",
        "censored_validation_tail_rows_not_usable_as_24h_test": censored,
        "data_test_directory_exists": data_test_exists,
        "official_test_not_used_for_scored_validation": True,
        "proposed_forward_validation": "For a future final estimate, split each tool chronologically into early model training, later threshold/calibration selection, and still-later untouched test blocks. Purge at least 24 hours at both boundaries, require complete 24-hour outcome follow-up for test rows, and select one threshold before opening the test block. New later labeled data would be preferable to reusing the current validation period.",
    }
    report = {
        "source_scores": str(score_file),
        "source_model_sha256": sweep["model_sha256"],
        "model_retrained": False, "features_changed": False,
        "thresholds_exactly_evaluated": THRESHOLDS,
        "event_definition": "Unique tool_id + fault_type + fault_time from fault_events.csv at or after that tool's validation cutoff. A detection needs at least one scored sensor row with threshold alarm in [fault_time-86400, fault_time); a fault-time row itself is excluded. One alarm may precede multiple nearby fault events.",
        "event_recall_denominators": "All logged validation faults and separately faults with at least one scored validation sensor row in the preceding 24 hours.",
        "false_alarm_definition": "Normal-labeled validation alarm rows; per-tool consecutive false alarms are one episode unless separated by more than 600 seconds.",
        "observed_hours_definition": "Number of labeled validation one-minute rows divided by 60, including positive and negative rows; not full wall-clock operating hours.",
        "row_recall_increase_vs_event_detection": {
            "row_recall_difference_low_vs_0_5": summaries[0]["recall"] - middle["recall"],
            "additional_fault_events_detected_low_vs_0_5": summaries[0]["detected_fault_events"] - middle["detected_fault_events"],
        },
        "paired_first_alert_lead_comparisons": [paired_lead(0.3406, 0.5), paired_lead(0.6911, 0.5)],
        "time_split_audit": time_audit,
        "summary": summaries,
    }
    (DEST / "event_threshold_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(details)} event-threshold rows to {DEST}", flush=True)


if __name__ == "__main__":
    main()
