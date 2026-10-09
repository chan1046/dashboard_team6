"""Summarize the causal, no-training Tool-relative fault trend audit."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata

from audit_tool_normalized_fault_trends import (DAY, OUT, PAIRS, PLAN, RAW,
                                                SOURCE, SENSORS, paired, values_stats)


def frozen_tool_baselines(plan: dict, fault_log: pd.DataFrame) -> pd.DataFrame:
    records = []
    for item in plan["tools"]:
        tool = item["tool_id"]
        cutoff = int(item["train_end_exclusive_second"]) - DAY
        table = pq.read_table(RAW / f"{tool}_sensor_ttf.parquet",
                              columns=["time", "FIXTURESHUTTERPOSITION", *SENSORS])
        t = table.column("time").to_numpy(zero_copy_only=False).astype(np.int64)
        fsp = table.column("FIXTURESHUTTERPOSITION").to_numpy(zero_copy_only=False)
        x = np.column_stack([table.column(s).to_numpy(zero_copy_only=False)
                             for s in SENSORS]).astype(np.float64)
        faults = np.sort(fault_log.loc[fault_log.tool_id.eq(tool) &
                                      fault_log.time.lt(int(item["train_end_exclusive_second"])),
                                      "time"].to_numpy(dtype=np.int64))
        previous_pos = np.searchsorted(faults, t, side="right") - 1
        next_pos = np.searchsorted(faults, t, side="right")
        prev_dist = np.full(len(t), np.inf)
        next_dist = np.full(len(t), np.inf)
        has_prev = previous_pos >= 0
        has_next = next_pos < len(faults)
        prev_dist[has_prev] = t[has_prev] - faults[previous_pos[has_prev]]
        next_dist[has_next] = faults[next_pos[has_next]] - t[has_next]
        minute = np.floor_divide(t, 60)
        mask = (np.r_[True, minute[1:] != minute[:-1]] & (t <= cutoff) &
                (fsp == 1) & np.isfinite(x).all(axis=1) &
                (prev_dist > DAY) & (next_dist > DAY))
        ref = x[mask]
        if len(ref) < 60:
            raise AssertionError(f"Insufficient frozen reference for {tool}")
        for j, sensor in enumerate(SENSORS):
            stats = values_stats(ref[:, j])
            records.append({"tool_id": tool, "sensor": sensor,
                            "reference_cutoff": cutoff, "reference_rows": len(ref),
                            "operation_rule": "FSP=1", "normal_rule": ">24h before/after any Train fault",
                            "sampling_rule": "first real observation per wall-clock minute",
                            **stats, "near_zero_median": abs(stats["median"]) <= 1e-9,
                            "near_zero_std": stats["std"] <= 1e-9})
        print("frozen reference", tool, len(ref), flush=True)
    return pd.DataFrame(records)


def main() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    fault_log = pd.read_csv(RAW / "fault_events.csv").drop_duplicates(["tool_id", "time", "fault_type"])
    sample = pd.read_csv(OUT / "relative_samples.csv")
    original = pd.read_csv(SOURCE / "sensor_history_quality.csv")
    by_anchor = OUT / "causal_baseline_by_anchor.csv"
    old = OUT / "tool_baselines.csv"
    if by_anchor.exists():
        frozen = pd.read_csv(old)
        if frozen.tool_id.nunique() != 20 or len(frozen) != 20 * len(SENSORS):
            raise AssertionError("Existing frozen references are incomplete")
    else:
        old.rename(by_anchor)
        frozen = frozen_tool_baselines(plan, fault_log)
        frozen.to_csv(old, index=False)
    assert frozen.tool_id.nunique() == 20 and len(frozen) == 20 * len(SENSORS)
    # The original fault-relative anchors and sensor extraction must not drift.
    assert len(sample) == len(original) == 741 * 6
    assert sample[["tool_id", "fault_event_id", "horizon", "prediction_timestamp"]].equals(
        original[["tool_id", "fault_event_id", "horizon", "prediction_timestamp"]])
    assert (sample.prediction_timestamp < sample.fault_timestamp).all()
    assert (sample.loc[sample.phase.eq("train"), "reference_cutoff"] <=
            sample.loc[sample.phase.eq("train"), "prediction_timestamp"] - DAY).all()
    train_end = {x["tool_id"]: int(x["train_end_exclusive_second"]) for x in plan["tools"]}
    nontrain = sample[~sample.phase.eq("train")]
    assert (nontrain.reference_cutoff == nontrain.tool_id.map(train_end) - DAY).all()
    assert (nontrain.reference_cutoff < nontrain.prediction_timestamp).all()
    assert (sample.sensor_row_count == original.sensor_row_count).all()
    for sensor in SENSORS:
        same = np.isclose(sample[f"raw_last_{sensor}"], original[f"last_{sensor}"], equal_nan=True)
        assert same.all(), sensor
    valid = sample[sample.usable_prior_rule & sample.reference_available].copy()
    valid["horizon_minutes"] = valid.horizon.map({"24h": 1440, "12h": 720,
                                                  "6h": 360, "3h": 180, "1h": 60, "30m": 30})
    associations = []
    grouped = list(valid.groupby("fault_event_id"))
    for kind in ("raw_median", "delta", "ratio", "zscore"):
        for sensor in SENSORS:
            column = f"{kind}_{sensor}"
            rhos_by_phase = {p: [] for p in ("all", "train", "validation", "reserved")}
            for _, event in grouped:
                event = event[["phase", "horizon_minutes", column]].dropna()
                if len(event) < 3:
                    continue
                a = rankdata(event.horizon_minutes.to_numpy(dtype=float))
                b = rankdata(event[column].to_numpy(dtype=float))
                a -= a.mean()
                b -= b.mean()
                norm = np.linalg.norm(a) * np.linalg.norm(b)
                if norm == 0:
                    continue
                rho = float(np.dot(a, b) / norm)
                rhos_by_phase["all"].append(rho)
                phase = str(event.phase.iloc[0])
                if phase in rhos_by_phase:
                    rhos_by_phase[phase].append(rho)
            for phase, rhos in rhos_by_phase.items():
                values = np.asarray(rhos)
                associations.append({"representation": kind, "sensor": sensor, "phase": phase,
                        "events_with_rank": len(values),
                        "median_spearman_ttf_order": float(np.median(values)) if len(values) else np.nan,
                        "mean_spearman_ttf_order": float(np.mean(values)) if len(values) else np.nan,
                        "positive_direction_events": int((values > 0).sum()),
                        "negative_direction_events": int((values < 0).sum()),
                        "positive_direction_pct": float(100 * (values > 0).mean()) if len(values) else np.nan,
                        "negative_direction_pct": float(100 * (values < 0).mean()) if len(values) else np.nan})
    pd.DataFrame(associations).to_csv(OUT / "ttf_order_association.csv", index=False)
    effect = []
    for phase in ("all", "train", "validation", "reserved"):
        part = valid if phase == "all" else valid[valid.phase.eq(phase)]
        for sensor in SENSORS:
            for earlier, later in PAIRS:
                raw = paired(part, f"raw_median_{sensor}", earlier, later)
                for kind in ("delta", "ratio", "zscore"):
                    rel = paired(part, f"{kind}_{sensor}", earlier, later)
                    effect.append({"phase": phase, "sensor": sensor, "earlier": earlier,
                                   "later": later, "representation": kind,
                                   "paired_events": raw["paired_events"],
                                   "raw_increase": raw["increase_events"],
                                   "raw_decrease": raw["decrease_events"],
                                   "relative_increase": rel["increase_events"],
                                   "relative_decrease": rel["decrease_events"],
                                   "raw_dominant_fraction": raw["dominant_fraction"],
                                   "relative_dominant_fraction": rel["dominant_fraction"],
                                   "dominant_fraction_change": rel["dominant_fraction"] - raw["dominant_fraction"]})
    pd.DataFrame(effect).to_csv(OUT / "pair_normalization_effect.csv", index=False)
    baseline = pd.read_csv(by_anchor)
    usable = baseline[baseline.reference_available]
    counts = {"candidate_event_horizons": len(sample), "prior_rule_usable": int(sample.usable_prior_rule.sum()),
              "usable_with_causal_baseline": len(valid), "tool_baseline_count": frozen.tool_id.nunique(),
              "causal_reference_missing": int((~sample.reference_available).sum()),
              "usable_without_reference": int((sample.usable_prior_rule & ~sample.reference_available).sum()),
              "near_zero_median_anchor_sensor": sum(int((usable[f"{s}_median"].abs() <= 1e-9).sum()) for s in SENSORS),
              "near_zero_std_anchor_sensor": sum(int((usable[f"{s}_std"] <= 1e-9).sum()) for s in SENSORS),
              "negative_median_anchor_sensor": sum(int((usable[f"{s}_median"] < 0).sum()) for s in SENSORS),
              "future_reference_violations": 0, "future_sensor_violations": 0}
    (OUT / "summary.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
