"""Assemble a truthful 100%-data 12h/24h validation report from executed runs.

Never substitutes archived HGB scores for a new experiment. The classification
probabilities and LSTM metrics must already exist on disk before this runs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from compare_12h_24h_xgboost import confusion
from evaluate_event_thresholds_24h import ROOT

OUT = ROOT / "output" / "experiments" / "model_comparison_100pct_12h_24h"
COMPARE = ROOT / "output" / "experiments" / "horizon_comparison_12h_24h"
LSTM = ROOT / "output" / "experiments" / "ttf_lstm_100pct_12h_24h"
SWEEP = ROOT / "output" / "experiments" / "threshold_sweep_24h"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    prior = json.loads((COMPARE / "comparison_report.json").read_text(encoding="utf-8"))
    if len(prior["results"]) != 2:
        raise AssertionError("Expected exactly 12h/24h prior full-data XGBoost runs")
    with np.load(COMPARE / "validation_probabilities_12h.npz") as x12:
        y12, p12 = x12["truth"].copy(), x12["probability"].copy()
        t12, tool12 = x12["time"].copy(), x12["tool_code"].copy()
    with np.load(SWEEP / "validation_probabilities.npz") as x24:
        y24, p24 = x24["truth"].copy(), x24["probability"].copy()
        t24, tool24 = x24["time"].copy(), x24["tool_code"].copy()
    if not (np.array_equal(t12, t24) and np.array_equal(tool12, tool24)):
        raise AssertionError("Validation windows differ between horizons")
    classifier = []
    for horizon, y, probability, old in zip((12, 24), (y12, y24), (p12, p24), prior["results"]):
        result = confusion(y, probability)
        for name in ("precision", "recall", "f1", "balanced_accuracy", "average_precision", "tn", "fp", "fn", "tp"):
            if not np.isclose(result[name], old[name], atol=1e-12):
                raise AssertionError(f"Saved {horizon}h probability disagrees on {name}")
        classifier.append({"horizon_hours": horizon, "model": "XGBoost", "train_fraction": "100%",
                           "train_rows": 4606868, "threshold": 0.5, **result,
                           "logged_fault_events": old["logged_fault_events"],
                           "evaluable_fault_events": old["evaluable_fault_events"],
                           "detected_fault_events": old["detected_fault_events"],
                           "all_fault_event_recall": old["all_fault_event_recall"],
                           "evaluable_fault_event_recall": old["evaluable_fault_event_recall"],
                           "false_alarm_events_10m": old["false_alarm_events_10m"],
                           "first_alert_lead_median_minutes": old["first_alert_lead_median_minutes"]})
    regression = read_csv(LSTM / "metrics.csv")
    if len(regression) != 3 or {r["fault_type"] for r in regression} != {"pressure_low", "pressure_high", "leak"}:
        raise AssertionError("Three independent LSTM results are required")
    OUT.mkdir(parents=True)
    save_csv(OUT / "classification_metrics.csv", classifier)
    save_csv(OUT / "regression_metrics.csv", regression)
    blocked = [
        {"model": "HistGradientBoostingClassifier", "task": "12h/24h classification", "status": "NOT_RUN",
         "reason": "Windows application control blocked scikit-learn histogram.cp312-win_amd64.pyd (WinError 4551)"},
        {"model": "HistGradientBoostingRegressor-log", "task": "three independent TTF regressions", "status": "NOT_RUN",
         "reason": "Same blocked scikit-learn histogram module"},
    ]
    save_csv(OUT / "unavailable_models.csv", blocked)
    document = {
        "status": "Partial model comparison: real-data XGBoost and LSTM runs; HGB families could not execute on this host",
        "data": "All observed 4-second train rows from 20 tools represented by 1-minute current/past aggregates",
        "split": "Per-tool chronological first 80% train / last 20% validation; 24h purge before validation cutoff",
        "fixed_validation": "The same 1,142,112 fully-labeled minute rows are used for 12h and 24h classification",
        "labels": "Any next logged pressure_low, pressure_high, or leak event within 12h/24h wall-clock time; incomplete follow-up censored",
        "features": "85 current/past process, sensor, rolling, gap, and tool features; no raw TTF, fault time, Lot, runnum, or absolute time",
        "threshold": 0.5, "random_state": 42,
        "classification_source": str(COMPARE / "validation_probabilities_12h.npz") + " and " + str(SWEEP / "validation_probabilities.npz"),
        "regression_source": str(LSTM / "metrics.csv"),
        "classification": classifier, "regression": regression, "unavailable_models": blocked,
        "baseline_24h": "output/baseline_recovery/reproduction_report.json",
        "limitations": [
            "No HGB comparison metric is fabricated or copied from the different historical experiment.",
            "No independent future test interval exists after this tuned validation; dashboard replay is not a final unseen test.",
            "Regression samples are the full eligible rows with a known future same-type fault before train cutoff or within observed validation. Counts differ by fault type.",
            "LSTM TTF error is evaluated independently of classification; no gate restricts training.",
            "TTF model bias can make minimum-prediction fault selection unreliable; connected-system performance has not been established.",
        ],
    }
    (OUT / "report.json").write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"classification": classifier, "regression": regression, "unavailable": blocked}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
