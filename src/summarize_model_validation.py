"""Write a reproducible decision sheet from completed real-data experiments."""

from __future__ import annotations

import csv
import json

import pandas as pd

from evaluate_event_thresholds_24h import ROOT

OUT = ROOT / "output" / "experiments" / "model_validation_decision_12h_24h"


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    root = ROOT / "output" / "experiments"
    xgb = json.loads((root / "horizon_comparison_12h_24h" / "comparison_report.json").read_text(encoding="utf-8"))["results"]
    hgb = pd.read_csv(root / "hgb_xgb_matched_12h_24h" / "metrics.csv").to_dict("records")
    classification = []
    for row in [*xgb, *hgb]:
        classification.append({k: row.get(k) for k in ("model", "horizon_hours", "threshold",
            "train_rows", "validation_rows", "average_precision", "precision", "recall", "f1",
            "balanced_accuracy", "tn", "fp", "fn", "tp", "detected_fault_events",
            "logged_fault_events", "evaluable_fault_events", "false_alarm_events_10m")})
    policy = pd.read_csv(root / "hgb_xgb_persistence_24h" / "metrics.csv")
    chosen = policy[(policy.model == "XGBoost") & (policy.threshold == .36) &
                    (policy.condition == "2_of_5m")].iloc[0].to_dict()
    hgb_near = policy[(policy.model == "HGB") & (policy.threshold == .32) &
                      (policy.condition == "2_of_5m")].iloc[0].to_dict()
    regression = pd.concat([
        pd.read_csv(root / "ttf_lstm_100pct_12h_24h" / "metrics.csv"),
        pd.read_csv(root / "hgb_log_ttf_matched" / "metrics.csv")], ignore_index=True)
    regression = regression[["model", "fault_type", "train_rows", "validation_rows",
                             "mae_hours", "rmse_seconds", "r2"]]
    OUT.mkdir(parents=True)
    write_csv(OUT / "classification.csv", classification)
    regression.to_csv(OUT / "regression.csv", index=False, encoding="utf-8-sig")
    write_csv(OUT / "alarm_policy_comparison.csv", [chosen, hgb_near])
    decision = {
        "classification_model_family": "XGBoost 12h and 24h",
        "reason": "On matched full-data chronological validation, XGBoost has higher AP and more detected fault events with fewer 10-minute false alarm events than HGB at threshold 0.5 for both horizons.",
        "candidate_24h_alarm": {"threshold": .36, "rule": "2 of last 5 observed clock minutes",
                                 "detected_fault_events": int(chosen["detected_fault_events"]),
                                 "logged_fault_events": 181,
                                 "false_alarm_events_10m": int(chosen["false_alarm_events_10m"]),
                                 "average_false_alarm_events_per_observed_tool_12h": float(chosen["false_alarm_events_per_observed_tool_12h"]),
                                 "tools_above_four_per_observed_12h": int(chosen["tools_over_4_false_alarms_per_observed_12h"]),
                                 "worst_tool_per_observed_12h": float(chosen["worst_tool_false_alarms_per_observed_12h"])},
        "hgb_same_budget_comparison": {"threshold": .32, "detected_fault_events": int(hgb_near["detected_fault_events"]),
                                       "false_alarm_events_10m": int(hgb_near["false_alarm_events_10m"])},
        "regression_decision": "Neither HGB-log nor LSTM validated for TTF/type display: all type-specific R2 values are negative.",
        "dashboard_status": "Not connected, at user request. XGBoost classification is a research candidate, not an approved live alarm policy.",
        "validation_limit": "Threshold and persistence were selected on the same validation period. No untouched later time period was evaluated. The observed-minute denominator is not a full wall-clock shift.",
        "next_experiment": "Fixed-rule forward validation on a later held-out time segment, reporting fault-event recall and per-tool 12h false-alarm distribution; test tool-specific calibration only after that."}
    (OUT / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
