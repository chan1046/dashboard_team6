"""Compare frozen 12h/24h scores as one dashboard alarm policy."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from compare_12h_24h_xgboost import event_audit, false_episodes
from evaluate_alarm_persistence_24h import causal_alarm
from evaluate_event_thresholds_24h import BASELINE, ROOT, SOURCE, load_events, read_json

OUT = ROOT / "output" / "experiments" / "dashboard_policy_comparison_12h_24h"
SCORES12 = ROOT / "output" / "experiments" / "horizon_comparison_12h_24h" / "validation_probabilities_12h.npz"
POLICIES = (("baseline", .5, .5, 1, 1),
            ("persistence_036", .36, .36, 5, 2),
            ("persistence_034_036", .34, .36, 5, 2))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    with np.load(SCORES12) as a, np.load(SOURCE / "validation_probabilities.npz") as b:
        y12, p12, time, tool = (a[k].copy() for k in ("truth", "probability", "time", "tool_code"))
        y24, p24, time24, tool24 = (b[k].copy() for k in ("truth", "probability", "time", "tool_code"))
    if not (np.array_equal(time, time24) and np.array_equal(tool, tool24)):
        raise AssertionError("12h/24h validation rows do not align")
    tool_map = {int(k): v for k, v in read_json(SOURCE / "sweep_report.json")["tool_code_map"].items()}
    cutoffs = {r["tool"]: r["split_cutoff_time"] for r in read_json(BASELINE / "preparation_report.json")["tools"]}
    events = load_events(cutoffs, set(tool_map.values()))
    reverse = {v: k for k, v in tool_map.items()}
    rows = []
    for name, t12, t24, window, minimum in POLICIES:
        a12 = causal_alarm(p12 >= t12, time, tool, sorted(tool_map), window, minimum)
        a24 = causal_alarm(p24 >= t24, time, tool, sorted(tool_map), window, minimum)
        combined = a12 | a24
        # The combined action is judged against the 24h truth; 12h is a higher-urgency tier.
        for horizon, truth, alarm, hours in (("12h", y12, a12, 12),
                                              ("24h", y24, a24, 24),
                                              ("any_24h", y24, combined, 24)):
            event, _ = event_audit(events, hours * 3600, alarm.astype(np.uint8), time, tool, reverse)
            false, by_tool = false_episodes(truth, alarm.astype(np.uint8), time, tool, tool_map)
            observed_days = len(truth) / 1440
            rows.append({"policy": name, "horizon": horizon, "threshold_12h": t12,
                         "threshold_24h": t24, "window_minutes": window, "minimum_hits": minimum,
                         **event, "false_alarm_events_10m": false,
                         "false_alarm_events_per_tool_12h": false / observed_days / 2,
                         "maximum_tool_false_alarm_per_observed_12h": max(
                             r["false_alarm_events_per_observed_tool_day"] / 2 for r in by_tool)})
    OUT.mkdir(parents=True)
    with (OUT / "policy_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "policy_report.json").write_text(json.dumps({
        "source_scores_12h": str(SCORES12), "source_scores_24h": str(SOURCE / "validation_probabilities.npz"),
        "model_retrained": False, "features_changed": False,
        "denominator": "Labeled observed minutes / 720 per tool-shift equivalent; not wall-clock shifts",
        "false_alarm_definition": "10-minute merge of alarm rows with negative 12h or 24h label as indicated",
        "limitation": "Policies selected on validation; untouched forward period is still required.",
        "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        print(row["policy"], row["horizon"], row["detected_fault_events"],
              row["false_alarm_events_10m"], round(row["false_alarm_events_per_tool_12h"], 2), flush=True)


if __name__ == "__main__":
    main()
