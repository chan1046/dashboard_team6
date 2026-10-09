"""Threshold/2-of-5 comparison for frozen HGB and XGBoost 24h scores."""

from __future__ import annotations

import csv
import json

import numpy as np

from compare_12h_24h_xgboost import event_audit, false_episodes
from evaluate_alarm_persistence_24h import causal_alarm
from evaluate_event_thresholds_24h import BASELINE, ROOT, SOURCE, load_events, read_json

OUT = ROOT / "output" / "experiments" / "hgb_xgb_persistence_24h"
HGB = ROOT / "output" / "experiments" / "hgb_xgb_matched_12h_24h" / "validation_probabilities_24h.npz"
THRESHOLDS = tuple(np.round(np.arange(.20, .61, .02), 2))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    files = {"HGB": HGB, "XGBoost": SOURCE / "validation_probabilities.npz"}
    tool_map = {int(k): v for k, v in read_json(SOURCE / "sweep_report.json")["tool_code_map"].items()}
    cutoffs = {r["tool"]: r["split_cutoff_time"] for r in read_json(BASELINE / "preparation_report.json")["tools"]}
    events = load_events(cutoffs, set(tool_map.values()))
    reverse = {v: k for k, v in tool_map.items()}
    rows = []
    for model, path in files.items():
        with np.load(path) as saved:
            truth, score, time, tool = (saved[k].copy() for k in ("truth", "probability", "time", "tool_code"))
        for threshold in THRESHOLDS:
            for condition, minutes, hits in (("2_of_5m", 5, 2), ("1_of_1m", 1, 1)):
                if condition == "1_of_1m" and threshold != .5:
                    continue
                alarm = causal_alarm(score >= threshold, time, tool, sorted(tool_map), minutes, hits)
                event, _ = event_audit(events, 86400, alarm.astype(np.uint8), time, tool, reverse)
                false, by_tool = false_episodes(truth, alarm.astype(np.uint8), time, tool, tool_map)
                per_shift = [r["false_alarm_events_per_observed_tool_day"] / 2 for r in by_tool]
                rows.append({"model": model, "threshold": float(threshold), "condition": condition,
                             "detected_fault_events": event["detected_fault_events"],
                             "all_fault_event_recall": event["all_fault_event_recall"],
                             "evaluable_fault_event_recall": event["evaluable_fault_event_recall"],
                             "false_alarm_events_10m": false,
                             "false_alarm_events_per_observed_tool_12h": false / (len(truth) / 720),
                             "tools_over_4_false_alarms_per_observed_12h": sum(v > 4 for v in per_shift),
                             "worst_tool_false_alarms_per_observed_12h": max(per_shift)})
    OUT.mkdir(parents=True)
    with (OUT / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = {model: min((r for r in rows if r["model"] == model and
                        r["condition"] == "2_of_5m" and r["all_fault_event_recall"] >= .75 and
                        r["false_alarm_events_per_observed_tool_12h"] <= 4),
                       key=lambda r: r["false_alarm_events_10m"], default=None)
            for model in files}
    (OUT / "report.json").write_text(json.dumps({"model_retrained": False,
        "source_scores": {k: str(v) for k, v in files.items()},
        "same_validation_rows": True, "target_recall": .75,
        "user_average_false_alarm_cap_per_observed_tool_12h": 4,
        "best_under_average_cap": best,
        "limitation": "An average under four does not guarantee every tool is under four; see per-tool columns. Validation is reused for threshold selection."},
        ensure_ascii=False, indent=2,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value)),
        encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
