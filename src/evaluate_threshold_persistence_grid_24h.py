"""Compare threshold and causal persistence pairs on saved 24h validation scores."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from evaluate_alarm_persistence_24h import causal_alarm
from evaluate_event_thresholds_24h import (
    BASELINE, ROOT, SOURCE, event_metrics, load_events, read_json,
    row_and_alarm_metrics, write_csv,
)

OUT = ROOT / "output" / "experiments" / "threshold_persistence_grid_24h"
THRESHOLDS = (0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50)
CONDITIONS = (("1_of_1m", 1, 1), ("2_of_3m", 3, 2), ("2_of_5m", 5, 2),
              ("3_of_5m", 5, 3), ("3_of_10m", 10, 3))
LEVELS = (0.70, 0.75, 0.80)


def detected_keys(details: list[dict]) -> set[tuple[str, str, int]]:
    return {(item["tool_id"], item["fault_type"], item["fault_time"])
            for item in details if item["detected_before_fault"]}


def main() -> None:
    sweep = read_json(SOURCE / "sweep_report.json")
    baseline = read_json(BASELINE / "reproduction_report.json")
    prior = read_json(ROOT / "output" / "experiments" / "event_level_threshold_24h" / "event_threshold_report.json")
    preparation = read_json(BASELINE / "preparation_report.json")
    score_file = SOURCE / "validation_probabilities.npz"
    with np.load(score_file) as saved:
        truth, probability = saved["truth"], saved["probability"]
        time, tool = saved["time"], saved["tool_code"]
    if len(truth) != sweep["validation_rows"] or int(truth.sum()) != sweep["validation_positive_rows"]:
        raise AssertionError("Saved validation population differs from the prior sweep")
    if not (len(truth) == len(probability) == len(time) == len(tool)):
        raise AssertionError("Score array lengths differ")
    if hashlib.sha256((BASELINE / "xgb_24h_recovery.joblib").read_bytes()).hexdigest() != sweep["model_sha256"]:
        raise AssertionError("Recovered model differs from the source of saved scores")
    tool_map = {int(code): name for code, name in sweep["tool_code_map"].items()}
    reverse_map = {name: code for code, name in tool_map.items()}
    cutoffs = {item["tool"]: item["split_cutoff_time"] for item in preparation["tools"]}
    events = load_events(cutoffs, set(tool_map.values()))
    if len(events) != 181:
        raise AssertionError(f"Expected 181 fault events, found {len(events)}")

    def evaluate(alarm: np.ndarray) -> tuple[dict, dict, list[dict], list[dict]]:
        binary = alarm.astype(np.uint8)
        row, by_tool = row_and_alarm_metrics(0.5, truth, binary, time, tool,
                                             sweep["average_precision_fixed"], tool_map)
        event, details = event_metrics(0.5, events, binary, time, tool, reverse_map)
        return row, event, details, by_tool

    base_row, base_event, base_details, _ = evaluate(probability >= 0.5)
    low_row, low_event, _, _ = evaluate(probability >= 0.3406)
    for key in ("tn", "fp", "fn", "tp", "false_alarm_events_10m"):
        if base_row[key] != baseline["metrics"][key]:
            raise AssertionError(f"Recovered baseline mismatch: {key}")
    if (base_event["detected_fault_events"], base_event["evaluable_fault_events"],
            low_event["detected_fault_events"], low_row["false_alarm_events_10m"]) != (112, 163, 145, 8686):
        raise AssertionError("Previous event threshold references were not reproduced")
    baseline_keys = detected_keys(base_details)

    summary, event_rows, missed_rows, tool_rows = [], [], [], []
    for threshold in THRESHOLDS:
        raw_hit = probability >= threshold
        for condition, window, required in CONDITIONS:
            alarm = causal_alarm(raw_hit, time, tool, sorted(tool_map), window, required)
            row, event, details, by_tool = evaluate(alarm)
            keys = detected_keys(details)
            if event["evaluable_fault_events"] != 163:
                raise AssertionError("Evaluable fault denominator changed")
            item = {
                "threshold": threshold, "condition": condition,
                "window_minutes": window, "minimum_threshold_exceedances": required,
                **{key: value for key, value in row.items() if key != "threshold"}, **event,
                "qualifies_recall_target": event["fault_event_recall_all_logged"] >= 0.75 or event["fault_event_recall_evaluable"] >= 0.85,
                "additional_detected_fault_events_vs_0_5": len(keys - baseline_keys),
                "additional_missed_fault_events_vs_0_5": len(baseline_keys - keys),
                "net_detected_fault_events_vs_0_5": len(keys) - len(baseline_keys),
                "false_alarm_change_vs_0_5": row["false_alarm_events_10m"] - base_row["false_alarm_events_10m"],
                "false_alarm_reduction_vs_0_3406": low_row["false_alarm_events_10m"] - row["false_alarm_events_10m"],
                "false_alarm_reduction_pct_vs_0_3406": (low_row["false_alarm_events_10m"] - row["false_alarm_events_10m"]) / low_row["false_alarm_events_10m"],
            }
            summary.append(item)
            for detail in details:
                record = {"grid_threshold": threshold, "condition": condition, **detail}
                event_rows.append(record)
                if not detail["detected_before_fault"]:
                    missed_rows.append(record)
            for distribution in by_tool:
                tool_rows.append({"grid_threshold": threshold, "condition": condition, **distribution})
            print(f"{threshold:.2f} {condition}: {len(keys)}/181 faults, {row['false_alarm_events_10m']} false events", flush=True)

    if len(summary) != 45 or len(event_rows) != 45 * 181:
        raise AssertionError("Incomplete threshold-persistence grid")
    baseline_grid = next(item for item in summary if item["threshold"] == 0.5 and item["condition"] == "1_of_1m")
    if baseline_grid["detected_fault_events"] != 112 or baseline_grid["false_alarm_events_10m"] != 3414:
        raise AssertionError("Grid baseline mismatch")

    def rank(item: dict) -> tuple:
        return (item["false_alarm_events_10m"], -item["detected_fault_events"],
                -item["lead_time_median_minutes"] if item["lead_time_median_minutes"] is not None else 0)

    qualifying = sorted((item for item in summary if item["qualifies_recall_target"]), key=rank)
    level_best = []
    for level in LEVELS:
        eligible = [item for item in summary if item["fault_event_recall_all_logged"] >= level]
        level_best.append({"target_fault_event_recall_all_logged": level,
                           **(min(eligible, key=rank) if eligible else {"candidate_found": False})})
    pareto = sorted((item for item in summary if not any(
        other is not item and
        other["detected_fault_events"] >= item["detected_fault_events"] and
        other["false_alarm_events_10m"] <= item["false_alarm_events_10m"] and
        (other["detected_fault_events"] > item["detected_fault_events"] or
         other["false_alarm_events_10m"] < item["false_alarm_events_10m"])
        for other in summary)), key=lambda item: (item["detected_fault_events"], item["false_alarm_events_10m"]))

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "threshold_persistence_grid.csv", summary, list(summary[0]))
    write_csv(OUT / "qualifying_candidates.csv", qualifying, list(summary[0]))
    write_csv(OUT / "recall_level_best.csv", level_best, list(level_best[0]))
    write_csv(OUT / "pareto_candidates.csv", pareto, list(summary[0]))
    write_csv(OUT / "fault_event_details.csv", event_rows, list(event_rows[0]))
    write_csv(OUT / "missed_fault_events.csv", missed_rows, list(event_rows[0]))
    write_csv(OUT / "false_alarm_by_tool.csv", tool_rows, list(tool_rows[0]))
    report = {
        "source_scores": str(score_file),
        "source_scores_sha256": hashlib.sha256(score_file.read_bytes()).hexdigest(),
        "source_model_sha256": sweep["model_sha256"],
        "model_retrained": False, "features_changed": False,
        "thresholds": THRESHOLDS,
        "persistence_conditions": [{"condition": name, "window_minutes": window, "minimum_hits": required}
                                   for name, window, required in CONDITIONS],
        "persistence_definition": "At each observed minute, require k threshold exceedances in the current and preceding W-1 clock-minute buckets, with only past/current scores. Reset after a >120-second tool stop; do not fill missing minutes.",
        "event_definition": prior["event_definition"],
        "false_alarm_definition": prior["false_alarm_definition"],
        "observed_hours_definition": prior["observed_hours_definition"],
        "comparison_0_3406_alone": {"detected_fault_events": low_event["detected_fault_events"],
                                     "false_alarm_events_10m": low_row["false_alarm_events_10m"]},
        "comparison_0_5_alone": {"detected_fault_events": base_event["detected_fault_events"],
                                  "false_alarm_events_10m": base_row["false_alarm_events_10m"]},
        "target_rule": "All logged fault-event recall >=75% OR evaluable fault-event recall >=85%; among eligible combinations, rank by fewest false alarm events. F1 is not a selection criterion.",
        "qualifying_count": len(qualifying),
        "lowest_false_alarm_qualifying": qualifying[:5],
        "recall_level_best": level_best,
        "pareto_candidates": pareto,
        "selection_limitation": "The same validation scores were used for earlier threshold comparisons. Candidate performance requires a later untouched chronological evaluation period before deployment.",
    }
    (OUT / "threshold_persistence_grid_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(summary)} combinations to {OUT}", flush=True)


if __name__ == "__main__":
    main()
