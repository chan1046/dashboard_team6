"""No-training sensor distribution and within-event trend summaries."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_simple_full_fault_rul import SENSORS


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/experiments/fault_centered_sensor_audit"
HORIZON_ORDER = ["24h", "12h", "6h", "3h", "1h", "30m"]
TOOL_GROUPS = ["03_M02", "09_M01", "08_M01", "other"]


def group_tool(tool: str) -> str:
    return tool if tool in TOOL_GROUPS[:-1] else "other"


def paired(a: pd.DataFrame, sensor: str, earlier: str, later: str) -> dict:
    p = a[a.horizon.isin([earlier, later])].pivot(index="fault_event_id", columns="horizon",
                                                  values=f"last_{sensor}").dropna()
    if not len(p):
        return {"paired_events": 0, "increase_events": 0, "decrease_events": 0,
                "unchanged_events": 0, "increase_pct": None, "decrease_pct": None,
                "median_delta": None, "mean_delta": None}
    delta = p[later] - p[earlier]
    return {"paired_events": len(delta), "increase_events": int((delta > 0).sum()),
            "decrease_events": int((delta < 0).sum()), "unchanged_events": int((delta == 0).sum()),
            "increase_pct": float(100 * (delta > 0).mean()),
            "decrease_pct": float(100 * (delta < 0).mean()),
            "median_delta": float(delta.median()), "mean_delta": float(delta.mean())}


def main() -> None:
    data = pd.read_csv(OUT / "sensor_history_quality.csv")
    valid = data[data.usable_prior_rule & data.source_comparison.eq("exact")].copy()
    assert len(data) == 741 * 6
    assert not data[["future_sensor_violation", "after_fault_sensor_violation",
                     "prediction_at_or_after_fault_violation"]].any().any()
    quality = []
    for horizon in HORIZON_ORDER:
        part = data[data.horizon == horizon]
        present = part[part.sensor_row_count > 0]
        available = part[part.usable_prior_rule]
        quality.append({"horizon": horizon, "all_events": len(part),
            "events_with_at_least_one_sensor": len(present),
            "usable_prior_rule": len(available),
            "count_at_least_60": int((present.sensor_row_count >= 60).sum()),
            "coverage_at_least_20m": int((present.coverage_duration_seconds >= 1200).sum()),
            "coverage_at_least_24m": int((present.coverage_duration_seconds >= 1440).sum()),
            "max_gap_at_most_5m": int((present.max_gap_seconds <= 300).sum()),
            "usable_prior_rule_and_max_gap_at_most_5m": int((available.max_gap_seconds <= 300).sum()),
            "usable_prior_rule_and_max_gap_at_most_90s": int((available.max_gap_seconds <= 90).sum()),
            "row_count_median_when_present": float(present.sensor_row_count.median()) if len(present) else None,
            "coverage_median_seconds_when_present": float(present.coverage_duration_seconds.median()) if len(present) else None,
            "median_sampling_interval_median_seconds_when_present": float(present.median_sampling_interval_seconds.median()) if len(present) else None,
            "max_gap_median_seconds_when_present": float(present.max_gap_seconds.median()) if len(present) else None,
            "max_gap_p90_seconds_when_present": float(present.max_gap_seconds.quantile(.9)) if len(present) else None,
            "sensor_age_median_seconds_when_present": float(present.sensor_age_seconds.median()) if len(present) else None})
    pd.DataFrame(quality).to_csv(OUT / "history_quality_summary.csv", index=False)
    sensor_rows = []
    for horizon in HORIZON_ORDER:
        part = valid[valid.horizon == horizon]
        for sensor in SENSORS:
            s = part[f"last_{sensor}"].dropna()
            sensor_rows.append({"horizon": horizon, "sensor": sensor, "events": len(s),
                "mean": float(s.mean()) if len(s) else None,
                "median": float(s.median()) if len(s) else None,
                "std": float(s.std(ddof=1)) if len(s) > 1 else None,
                "p25": float(s.quantile(.25)) if len(s) else None,
                "p75": float(s.quantile(.75)) if len(s) else None})
    pd.DataFrame(sensor_rows).to_csv(OUT / "sensor_horizon_summary.csv", index=False)
    paired_rows = []
    pairs = [("24h", "30m"), ("12h", "30m"), ("6h", "30m"),
             ("3h", "30m"), ("1h", "30m"), ("6h", "1h")]
    for sensor in SENSORS:
        for earlier, later in pairs:
            paired_rows.append({"sensor": sensor, "earlier": earlier, "later": later,
                                **paired(valid, sensor, earlier, later)})
    pd.DataFrame(paired_rows).to_csv(OUT / "paired_sensor_trends.csv", index=False)
    # Within-event consistency across all available horizons, requiring >=3.
    consistency = []
    for sensor in SENSORS:
        for event, part in valid.groupby("fault_event_id"):
            ordered = part.set_index("horizon").reindex(HORIZON_ORDER)[f"last_{sensor}"].dropna()
            if len(ordered) < 3:
                continue
            differences = np.diff(ordered.to_numpy(dtype=float))
            nonzero = np.sign(differences[differences != 0])
            consistency.append({"sensor": sensor, "fault_event_id": event,
                "tool_id": part.tool_id.iloc[0], "observed_horizons": len(ordered),
                "earliest_horizon": ordered.index[0], "latest_horizon": ordered.index[-1],
                "delta_earliest_to_latest": float(ordered.iloc[-1] - ordered.iloc[0]),
                "monotone_increase": bool(np.all(differences >= 0) and np.any(differences > 0)),
                "monotone_decrease": bool(np.all(differences <= 0) and np.any(differences < 0)),
                "unchanged": bool(np.all(differences == 0)),
                "direction_consistency": float(abs(nonzero.mean())) if len(nonzero) else None})
    consistency_df = pd.DataFrame(consistency)
    consistency_df.to_csv(OUT / "event_internal_direction.csv", index=False)
    consistency_summary = consistency_df.groupby("sensor", as_index=False).agg(
        events=("fault_event_id", "nunique"),
        monotone_increase_events=("monotone_increase", "sum"),
        monotone_decrease_events=("monotone_decrease", "sum"),
        unchanged_events=("unchanged", "sum"),
        median_direction_consistency=("direction_consistency", "median"))
    consistency_summary.to_csv(OUT / "event_direction_summary.csv", index=False)
    valid["tool_group"] = valid.tool_id.map(group_tool)
    tool_rows = []
    for group in TOOL_GROUPS:
        part = valid[valid.tool_group == group]
        for sensor in SENSORS:
            for earlier, later in [("24h", "30m"), ("6h", "30m"), ("1h", "30m")]:
                tool_rows.append({"tool_group": group, "sensor": sensor, "earlier": earlier,
                                  "later": later, **paired(part, sensor, earlier, later)})
    pd.DataFrame(tool_rows).to_csv(OUT / "tool_group_paired_trends.csv", index=False)
    status = data.source_comparison.value_counts().to_dict()
    summary = {"low_fault_events": data.fault_event_id.nunique(), "candidate_horizon_samples": len(data),
        "usable_history_samples": len(valid), "source_ttf_comparison": status,
        "future_sensor_violations": int(data.future_sensor_violation.sum()),
        "after_fault_sensor_violations": int(data.after_fault_sensor_violation.sum()),
        "prediction_at_or_after_fault_violations": int(data.prediction_at_or_after_fault_violation.sum()),
        "source_ttf_mismatch_count": int((data.source_comparison == "mismatch").sum()),
        "source_ttf_max_absolute_difference_seconds": float(data.source_ttf_difference_seconds.abs().max()) if data.source_ttf_difference_seconds.notna().any() else None,
        "manual_event_ids": sorted(pd.read_csv(OUT / "manual_validation_samples.csv").fault_event_id.unique().astype(int).tolist()),
        "manual_tools": sorted(pd.read_csv(OUT / "manual_validation_samples.csv").tool_id.unique().tolist())}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(pd.DataFrame(quality).to_string(index=False))
    print(pd.DataFrame(paired_rows).query('earlier == "24h" and later == "30m"').to_string(index=False))


if __name__ == "__main__":
    main()
