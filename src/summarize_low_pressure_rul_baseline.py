"""Read-only analysis of trained Low Pressure RUL predictions; no retraining."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/experiments/low_pressure_rul_baseline"
AUDIT = ROOT / "output/experiments/low_pressure_rul_audit"
SEED = 42


def main() -> None:
    predictions = pd.read_csv(OUT / "validation_predictions.csv.gz")
    events = pd.read_csv(AUDIT / "low_pressure_events.csv")
    assert predictions.fault_event_id.isin(events.fault_event_id).all()
    by_event = events.set_index("fault_event_id").any_other_between_lows
    predictions["event_has_any_other_fault_between_lows"] = predictions.fault_event_id.map(by_event)
    rows = []
    event_error = {}
    for model in ("Dummy", "Tree", "GRU"):
        pred = predictions[f"{model}_prediction_hours"]
        predictions[f"{model}_absolute_error_hours"] = (pred - predictions.target_ttf_hours).abs()
        event_error[model] = predictions.groupby("fault_event_id")[f"{model}_absolute_error_hours"].mean()
        for scope, part in predictions.groupby("event_has_any_other_fault_between_lows"):
            err = part[f"{model}_absolute_error_hours"]
            rows.append({"model": model, "event_has_other_fault": bool(scope),
                "samples": len(part), "unique_events": part.fault_event_id.nunique(),
                "row_mae_h": float(err.mean()),
                "event_equal_mae_h": float(part.groupby("fault_event_id")[f"{model}_absolute_error_hours"].mean().mean()),
                "actual_ttf_median_h": float(part.target_ttf_hours.median()),
                "prediction_median_h": float(part[f"{model}_prediction_hours"].median())})
    pd.DataFrame(rows).to_csv(OUT / "intervening_fault_disjoint_event_metrics.csv", index=False)
    rng = np.random.default_rng(SEED)
    ids = event_error["Dummy"].index.to_numpy()
    assert all(event_error[m].index.equals(event_error["Dummy"].index) for m in ("Tree", "GRU"))
    sampled = rng.integers(0, len(ids), size=(5000, len(ids)))
    bootstrap = []
    for model in ("Tree", "GRU"):
        difference = (event_error[model] - event_error["Dummy"]).to_numpy()
        means = difference[sampled].mean(axis=1)
        bootstrap.append({"model": model, "comparison": "event MAE minus Dummy; negative is better",
            "events": len(ids), "difference_h": float(difference.mean()),
            "bootstrap_p025_h": float(np.quantile(means, .025)),
            "bootstrap_p975_h": float(np.quantile(means, .975)),
            "resamples": 5000, "seed": SEED})
    pd.DataFrame(bootstrap).to_csv(OUT / "paired_event_bootstrap.csv", index=False)
    config = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    summary = {"train_samples": config["train_samples"], "validation_samples": config["validation_samples"],
        "train_events": config["train_events"], "validation_events": config["validation_events"],
        "train_event_cap": config["train_cap_per_event"],
        "validation_event_with_other_fault": int(predictions.loc[
            predictions.event_has_any_other_fault_between_lows, "fault_event_id"].nunique()),
        "validation_rows_with_future_intermediate_fault": int(predictions.intervening_other_fault.sum()),
        "validation_event_without_other_fault": int(predictions.loc[
            ~predictions.event_has_any_other_fault_between_lows, "fault_event_id"].nunique()),
        "bootstrap": bootstrap}
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
