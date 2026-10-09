"""No-training audit of real partial 30-minute histories for frozen Model 2."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from audit_full_fault_type_data import PLAN, RAW, ROOT, TYPES
from train_simple_full_fault_rul import SENSORS

BASE = ROOT / "output/experiments/full_fault_type_baseline"
OUT = ROOT / "output/experiments/model2_partial_history_audit"
WINDOW = 1800
RATIOS = [.50, .67, .80, .90]
COVERAGES = [900, 1200, 1440]
FRESHNESS = [60, 180, 300, None]
CANDIDATES = {"A": (.50, 900, 300), "B": (.67, 1200, 180),
              "C": (.80, 1440, 60)}
DAY = 86400


def normal_interval(tool: str, train_end: int, faults: np.ndarray) -> dict:
    table = pq.read_table(RAW / f"{tool}_sensor_ttf.parquet",
                          columns=["time", "FIXTURESHUTTERPOSITION"])
    t = table.column("time").to_numpy(zero_copy_only=False).astype(np.int64)
    fsp = table.column("FIXTURESHUTTERPOSITION").to_numpy(zero_copy_only=False)
    assert np.all(t[1:] >= t[:-1])
    pos = np.searchsorted(faults, t, side="right")
    previous = np.full(len(t), np.inf)
    following = np.full(len(t), np.inf)
    has_previous = pos > 0
    has_following = pos < len(faults)
    previous[has_previous] = t[has_previous] - faults[pos[has_previous]-1]
    following[has_following] = faults[pos[has_following]] - t[has_following]
    normal = ((t <= train_end - DAY) & (fsp == 1) &
              (previous > DAY) & (following > DAY))
    intervals = np.diff(t)
    eligible = normal[:-1] & normal[1:] & (intervals > 0)
    values = intervals[eligible]
    if not len(values):
        raise AssertionError(f"No Train normal adjacent observations: {tool}")
    median = float(np.median(values))
    expected = WINDOW / median
    return {"tool_id": tool, "normal_train_adjacent_pairs": len(values),
            "sampling_interval_median_seconds": median,
            "sampling_interval_p25_seconds": float(np.quantile(values, .25)),
            "sampling_interval_p75_seconds": float(np.quantile(values, .75)),
            "sampling_interval_p90_seconds": float(np.quantile(values, .90)),
            "sampling_intervals_over_30s_pct": float(100 * np.mean(values > 30)),
            "expected_rows_in_30m": expected,
            "reference_rule": "adjacent real Train observations; both FSP=1 and >24h before/after Train any-type fault"}


def feature_quality(t: np.ndarray, x: np.ndarray) -> dict:
    if not len(t):
        return {"feature_nan_count_72": 72, "feature_nan_count_90": 90,
                "slope_nonfinite_count": 9, "slope_temporally_concentrated": True,
                "first_half_rows": 0, "second_half_rows": 0,
                "all_raw_finite": False}
    n = len(t)
    halfway = int(t[0] + (t[-1] - t[0]) / 2)
    first_half = int(np.sum(t <= halfway))
    second_half = n - first_half
    elapsed = (t - t[0]).astype(np.float64)
    centered = elapsed - elapsed.mean()
    denom = float(np.dot(centered, centered))
    if denom <= 0:
        slope = np.full(x.shape[1], np.nan)
    else:
        slope = np.dot(centered, x - x.mean(axis=0)) / denom * 3600
    values = [x[-1], x.mean(axis=0), x.std(axis=0), x.min(axis=0), x.max(axis=0),
              np.median(x, axis=0), slope, x[-1]-x[0]]
    base_nan = int(sum(np.sum(~np.isfinite(v)) for v in values))
    quartiles = [np.quantile(x, .25, axis=0), np.quantile(x, .75, axis=0)]
    all_nan = base_nan + int(sum(np.sum(~np.isfinite(v)) for v in quartiles))
    maxgap = int(np.max(np.diff(t))) if n > 1 else np.inf
    return {"feature_nan_count_72": base_nan, "feature_nan_count_90": all_nan,
            "slope_nonfinite_count": int(np.sum(~np.isfinite(slope))),
            "slope_temporally_concentrated": bool(max(first_half, second_half) / n > .9 or maxgap > 600),
            "first_half_rows": first_half, "second_half_rows": second_half,
            "all_raw_finite": bool(np.isfinite(x).all())}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    faults = pd.read_csv(RAW / "fault_events.csv").drop_duplicates(["tool_id", "time", "fault_type"])
    source = pd.read_csv(BASE / "eligibility.csv")
    assert len(source) and set(source.phase) == {"train", "validation"}
    source_counts = source.groupby(["phase", "fault_type"]).eligible.sum()
    expected_counts = {("train", "pressure_low"): 186, ("train", "pressure_high"): 32,
                       ("train", "leak"): 38, ("validation", "pressure_low"): 73,
                       ("validation", "pressure_high"): 14, ("validation", "leak"): 4}
    assert source_counts.to_dict() == expected_counts
    references = []
    for item in plan["tools"]:
        tool = item["tool_id"]
        train_faults = np.sort(faults.loc[faults.tool_id.eq(tool) &
                                           faults.time.lt(int(item["train_end_exclusive_second"])),
                                           "time"].to_numpy(dtype=np.int64))
        references.append(normal_interval(tool, int(item["train_end_exclusive_second"]), train_faults))
        print("sampling", tool, references[-1]["sampling_interval_median_seconds"], flush=True)
    ref = pd.DataFrame(references)
    assert len(ref) == 20 and ref.tool_id.nunique() == 20
    data = source.merge(ref[["tool_id", "sampling_interval_median_seconds", "expected_rows_in_30m"]],
                        on="tool_id", how="left", validate="many_to_one")
    data["observation_ratio"] = data.sensor_row_count / data.expected_rows_in_30m
    feature_rows = []
    for item in plan["tools"]:
        tool = item["tool_id"]
        table = pq.read_table(RAW / f"{tool}_sensor_ttf.parquet", columns=["time", *SENSORS])
        t = table.column("time").to_numpy(zero_copy_only=False).astype(np.int64)
        x = np.column_stack([table.column(s).to_numpy(zero_copy_only=False)
                             for s in SENSORS]).astype(np.float64)
        part = data[data.tool_id.eq(tool)]
        for row in part.itertuples():
            first = max(int(np.searchsorted(t, int(row.history_start), side="left")),
                        int(np.searchsorted(t, int(row.phase_and_cycle_start), side="left")))
            end = int(np.searchsorted(t, int(row.prediction_timestamp), side="right"))
            local_t, local_x = t[first:end], x[first:end]
            if len(local_t) != int(row.sensor_row_count):
                raise AssertionError(f"History row drift: {tool}, {row.fault_event_id}")
            if len(local_t):
                if int(local_t[0]) != int(row.first_sensor_timestamp) or int(local_t[-1]) != int(row.last_sensor_timestamp):
                    raise AssertionError("History timestamp drift")
                if np.any(local_t > int(row.prediction_timestamp)):
                    raise AssertionError("Future sensor used")
            feature_rows.append({"fault_event_id": int(row.fault_event_id),
                                 **feature_quality(local_t, local_x)})
        print("features", tool, len(part), flush=True)
    quality = data.merge(pd.DataFrame(feature_rows), on="fault_event_id", validate="one_to_one")
    assert not quality.fault_event_id.duplicated().any()
    assert (quality.prediction_timestamp < quality.fault_timestamp).all()
    assert not (quality.last_sensor_timestamp > quality.prediction_timestamp).any()
    # Do not imply finite features where raw values are missing: model fitting
    # would need a separately declared missing-value policy in that case.
    OUT.mkdir(parents=True)
    ref.to_csv(OUT / "tool_sampling_intervals.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(OUT / "history_quality.csv", index=False, encoding="utf-8-sig")
    strict = quality[quality.eligible]
    rows = []
    for ratio in RATIOS:
        for coverage in COVERAGES:
            for freshness in FRESHNESS:
                mask = (quality.observation_ratio >= ratio) & quality.coverage_duration_seconds.ge(coverage)
                if freshness is not None:
                    mask &= quality.last_sensor_age_seconds.le(freshness)
                selected = quality[mask]
                for phase in ("train", "validation"):
                    base_phase = strict[strict.phase.eq(phase)]
                    for typ in [*TYPES, "total"]:
                        subset = selected[selected.phase.eq(phase)]
                        old = base_phase
                        if typ != "total":
                            subset = subset[subset.fault_type.eq(typ)]
                            old = old[old.fault_type.eq(typ)]
                        rows.append({"ratio_min": ratio, "coverage_min_seconds": coverage,
                                     "freshness_max_seconds": "none" if freshness is None else freshness,
                                     "phase": phase, "fault_type": typ, "usable_events": len(subset),
                                     "strict_events": len(old), "increase_events": len(subset)-len(old),
                                     "new_not_strict": int((mask & ~quality.eligible & quality.phase.eq(phase) &
                                                            (quality.fault_type.eq(typ) if typ != "total" else True)).sum()),
                                     "strict_lost": int((~mask & quality.eligible & quality.phase.eq(phase) &
                                                         (quality.fault_type.eq(typ) if typ != "total" else True)).sum()),
                                     "increase_pct": 100*(len(subset)-len(old))/len(old) if len(old) else np.nan,
                                     "tools": subset.tool_id.nunique()})
    pd.DataFrame(rows).to_csv(OUT / "threshold_event_counts.csv", index=False, encoding="utf-8-sig")
    selections = {"strict": quality.eligible}
    for name, (ratio, coverage, freshness) in CANDIDATES.items():
        selections[name] = ((quality.observation_ratio >= ratio) &
                            quality.coverage_duration_seconds.ge(coverage) &
                            quality.last_sensor_age_seconds.le(freshness))
    candidate_rows, nan_rows, tool_rows = [], [], []
    for name, mask in selections.items():
        selected = quality[mask]
        for phase in ("train", "validation", "combined"):
            subset = selected if phase == "combined" else selected[selected.phase.eq(phase)]
            old = strict if phase == "combined" else strict[strict.phase.eq(phase)]
            for typ in [*TYPES, "total"]:
                sub = subset if typ == "total" else subset[subset.fault_type.eq(typ)]
                prior = old if typ == "total" else old[old.fault_type.eq(typ)]
                candidate_rows.append({"candidate": name, "phase": phase, "fault_type": typ,
                    "events": len(sub), "strict_events": len(prior),
                    "increase_events": len(sub)-len(prior),
                    "new_not_strict": int((mask & ~quality.eligible &
                        (quality.phase.eq(phase) if phase != "combined" else True) &
                        (quality.fault_type.eq(typ) if typ != "total" else True)).sum()),
                    "strict_lost": int((~mask & quality.eligible &
                        (quality.phase.eq(phase) if phase != "combined" else True) &
                        (quality.fault_type.eq(typ) if typ != "total" else True)).sum()),
                    "increase_pct": 100*(len(sub)-len(prior))/len(prior) if len(prior) else np.nan,
                    "tools": sub.tool_id.nunique(),
                    "median_observation_ratio": float(sub.observation_ratio.median()) if len(sub) else np.nan,
                    "median_max_gap_seconds": float(sub.max_gap_seconds.median()) if len(sub) else np.nan})
            if len(subset):
                nan_rows.append({"candidate": name, "phase": phase, "events": len(subset),
                    "samples_with_nan_72": int(subset.feature_nan_count_72.gt(0).sum()),
                    "feature_nan_rate_72": float(subset.feature_nan_count_72.sum()/(len(subset)*72)),
                    "samples_with_nan_90": int(subset.feature_nan_count_90.gt(0).sum()),
                    "feature_nan_rate_90": float(subset.feature_nan_count_90.sum()/(len(subset)*90)),
                    "samples_with_nonfinite_slope": int(subset.slope_nonfinite_count.gt(0).sum()),
                    "temporally_concentrated_or_max_gap_over_10m": int(subset.slope_temporally_concentrated.sum()),
                    "max_gap_over_10m": int(subset.max_gap_seconds.gt(600).sum()),
                    "median_gap_seconds": float(subset.median_gap_seconds.median()),
                    "median_max_gap_seconds": float(subset.max_gap_seconds.median())})
            for tool in [item["tool_id"] for item in plan["tools"]]:
                count = int(subset.tool_id.eq(tool).sum())
                old_count = int(old.tool_id.eq(tool).sum())
                tool_rows.append({"candidate": name, "phase": phase, "tool_id": tool,
                    "events": count, "share_pct": 100*count/len(subset) if len(subset) else np.nan,
                    "strict_events": old_count, "strict_share_pct": 100*old_count/len(old) if len(old) else np.nan,
                    "added_events": count-old_count})
    pd.DataFrame(candidate_rows).to_csv(OUT / "candidate_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(nan_rows).to_csv(OUT / "feature_nan_rates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(tool_rows).to_csv(OUT / "tool_distribution.csv", index=False, encoding="utf-8-sig")
    violations = {"sensor_after_prediction": int((quality.last_sensor_timestamp > quality.prediction_timestamp).sum()),
                  "prediction_at_or_after_fault": int((quality.prediction_timestamp >= quality.fault_timestamp).sum()),
                  "reserved_or_excluded_events": int((~quality.phase.isin(["train", "validation"])).sum()),
                  "source_event_count_changed": int(len(quality) != len(source)),
                  "train_validation_event_overlap": len(set(quality.loc[quality.phase.eq("train"), "fault_event_id"]) &
                                                        set(quality.loc[quality.phase.eq("validation"), "fault_event_id"]))}
    assert not any(violations.values())
    pd.DataFrame([{"check": key, "violations": value} for key, value in violations.items()]).to_csv(
        OUT / "leakage_check.csv", index=False, encoding="utf-8-sig")
    (OUT / "config.json").write_text(json.dumps({"model_training": False,
        "reference_sampling_interval": "Train FSP=1 real adjacent rows >24h before/after any Train fault; median of positive intervals; no Validation statistics",
        "expected_rows_formula": "1800 seconds / Tool Train median interval",
        "history_window": "[fault-60m, fault-30m] actual raw sensor rows, after previous any-type fault and within existing phase",
        "thresholds": {"ratios": RATIOS, "coverage_seconds": COVERAGES,
                       "freshness_seconds": FRESHNESS, "candidates": CANDIDATES},
        "slope_instability_descriptive_flag": ">90% of rows in either half of observed span OR max internal gap >600s; not an optimized eligibility cutoff",
        "split": str(PLAN.relative_to(ROOT)), "random_state": 42}, indent=2), encoding="utf-8")
    print(pd.DataFrame(candidate_rows).query('phase == "combined" and fault_type == "total"').to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
