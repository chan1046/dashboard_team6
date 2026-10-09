"""Compare saved 12h scores with the established 24h alarm grid rules."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from compare_12h_24h_xgboost import confusion, event_audit, false_episodes
from evaluate_alarm_persistence_24h import causal_alarm
from evaluate_event_thresholds_24h import BASELINE, ROOT, SOURCE, load_events, read_json

OUT = ROOT / "output" / "experiments" / "threshold_persistence_grid_12h"
SCORES = ROOT / "output" / "experiments" / "horizon_comparison_12h_24h" / "validation_probabilities_12h.npz"
THRESHOLDS = (0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50)
CONDITIONS = (("1_of_1m", 1, 1), ("2_of_3m", 3, 2), ("2_of_5m", 5, 2),
              ("3_of_5m", 5, 3), ("3_of_10m", 10, 3))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    with np.load(SCORES) as saved:
        truth = saved["truth"].copy()
        probability = saved["probability"].copy()
        time = saved["time"].copy()
        tool = saved["tool_code"].copy()
    prior = read_json(ROOT / "output" / "experiments" / "horizon_comparison_12h_24h" / "comparison_report.json")["results"][0]
    if len(truth) != 1142112 or int(truth.sum()) != 37019:
        raise AssertionError("12h score population changed")
    sweep = read_json(SOURCE / "sweep_report.json")
    tool_map = {int(k): v for k, v in sweep["tool_code_map"].items()}
    cutoffs = {r["tool"]: r["split_cutoff_time"] for r in read_json(BASELINE / "preparation_report.json")["tools"]}
    events = load_events(cutoffs, set(tool_map.values()))
    ap = float(average_precision_score(truth, probability))
    if len(events) != 181 or not np.isclose(ap, prior["average_precision"], atol=1e-12):
        raise AssertionError("Saved 12h model or event population changed")
    summary = []
    for threshold in THRESHOLDS:
        hit = probability >= threshold
        for condition, window, minimum in CONDITIONS:
            alarm = causal_alarm(hit, time, tool, sorted(tool_map), window, minimum)
            row = confusion(truth, alarm.astype(np.float32))
            event, _ = event_audit(events, 43200, alarm.astype(np.float32), time, tool,
                                   {v: k for k, v in tool_map.items()})
            false_count, _ = false_episodes(truth, alarm.astype(np.float32), time, tool, tool_map)
            summary.append({"threshold": threshold, "condition": condition,
                            "window_minutes": window, "minimum_hits": minimum,
                            **{k: v for k, v in row.items() if k != "average_precision"},
                            "average_precision": ap, **event,
                            "false_alarm_events_10m": false_count,
                            "false_alarm_events_per_observed_tool_day": false_count / (len(truth) / 1440)})
            print(f"12h {threshold:.2f} {condition}: events={event['detected_fault_events']}/181 false={false_count}", flush=True)
    baseline = next(r for r in summary if r["threshold"] == .5 and r["condition"] == "1_of_1m")
    for key in ("precision", "recall", "f1", "average_precision", "detected_fault_events", "false_alarm_events_10m"):
        if not np.isclose(baseline[key], prior[key], atol=1e-12):
            raise AssertionError(f"12h baseline not reproduced: {key}")
    eligible = [r for r in summary if r["all_fault_event_recall"] >= .75 or r["evaluable_fault_event_recall"] >= .85]
    best = min(eligible, key=lambda r: r["false_alarm_events_10m"]) if eligible else None
    OUT.mkdir(parents=True)
    write_csv(OUT / "grid.csv", summary)
    (OUT / "report.json").write_text(json.dumps({"model_retrained": False,
        "source_scores": str(SCORES), "validation_rows": len(truth),
        "thresholds": THRESHOLDS, "conditions": CONDITIONS,
        "baseline": baseline, "qualifying_count": len(eligible),
        "lowest_false_alarm_qualifying": best,
        "limitation": "Selection uses the existing validation segment and requires an untouched later time segment for final operational performance."},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": baseline, "qualifying_count": len(eligible), "best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
