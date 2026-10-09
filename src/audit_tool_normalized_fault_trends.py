"""No-training, causal Tool-normalized audit of fixed Low Pressure horizons."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from audit_fault_centered_sensors import HORIZONS, PLAN, RAW, ROOT, SENSORS

SOURCE = ROOT / "output/experiments/fault_centered_sensor_audit"
OUT = ROOT / "output/experiments/tool_normalized_fault_trend_audit"
PAIRS = [("24h", "12h"), ("12h", "6h"), ("6h", "3h"),
         ("3h", "1h"), ("1h", "30m"), ("24h", "30m")]
GAPS = [None, 30, 60, 120, 300]
MIN_REFERENCE = 60
DAY = 86400


def values_stats(values: np.ndarray) -> dict:
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)), "p25": float(np.quantile(values, .25)),
            "p75": float(np.quantile(values, .75))}


def paired(frame: pd.DataFrame, column: str, earlier: str, later: str) -> dict:
    pivot = frame[frame.horizon.isin((earlier, later))].pivot(
        index="fault_event_id", columns="horizon", values=column).dropna()
    d = pivot[later].to_numpy() - pivot[earlier].to_numpy() if len(pivot) else np.empty(0)
    inc, dec, same = int(np.sum(d > 0)), int(np.sum(d < 0)), int(np.sum(d == 0))
    return {"paired_events": len(d), "increase_events": inc,
            "decrease_events": dec, "unchanged_events": same,
            "dominant_direction": "increase" if inc > dec else "decrease" if dec > inc else "tie",
            "dominant_fraction": max(inc, dec) / len(d) if len(d) else np.nan,
            "median_later_minus_earlier": float(np.median(d)) if len(d) else np.nan}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    original = pd.read_csv(SOURCE / "sensor_history_quality.csv")
    assert len(original) == 741 * 6 and original.fault_event_id.nunique() == 741
    assert not original[["future_sensor_violation", "after_fault_sensor_violation",
                         "prediction_at_or_after_fault_violation"]].any().any()
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    faults = pd.read_csv(RAW / "fault_events.csv").drop_duplicates(["tool_id", "time", "fault_type"])
    OUT.mkdir(parents=True)
    records, baselines = [], []
    for tool_plan in plan["tools"]:
        tool = tool_plan["tool_id"]
        audit = original[original.tool_id.eq(tool)].copy()
        table = pq.read_table(RAW / f"{tool}_sensor_ttf.parquet",
                              columns=["time", "FIXTURESHUTTERPOSITION", *SENSORS])
        times = table.column("time").to_numpy(zero_copy_only=False).astype(np.int64)
        fsp = table.column("FIXTURESHUTTERPOSITION").to_numpy(zero_copy_only=False)
        sensors = np.column_stack([table.column(s).to_numpy(zero_copy_only=False)
                                   for s in SENSORS]).astype(np.float64)
        if np.any(times[1:] < times[:-1]):
            raise AssertionError(f"Unsorted raw observations: {tool}")
        train_end = int(tool_plan["train_end_exclusive_second"])
        tool_faults = np.sort(faults.loc[faults.tool_id.eq(tool) &
                                         faults.time.lt(train_end), "time"].to_numpy(dtype=np.int64))
        # Real 4-second observations only. Sampling first real row per wall-clock minute
        # limits reference size without aggregation, interpolation or synthetic values.
        minute = np.floor_divide(times, 60)
        one_per_minute = np.r_[True, minute[1:] != minute[:-1]]
        previous_pos = np.searchsorted(tool_faults, times, side="right") - 1
        next_pos = np.searchsorted(tool_faults, times, side="right")
        previous_distance = np.full(len(times), np.inf)
        next_distance = np.full(len(times), np.inf)
        has_previous = previous_pos >= 0
        has_next = next_pos < len(tool_faults)
        previous_distance[has_previous] = times[has_previous] - tool_faults[previous_pos[has_previous]]
        next_distance[has_next] = tool_faults[next_pos[has_next]] - times[has_next]
        reference_mask = (one_per_minute & (fsp == 1) & np.isfinite(sensors).all(axis=1)
                          & (times < train_end) & (previous_distance > DAY)
                          & (next_distance > DAY))
        ref_t = times[reference_mask]
        ref_x = sensors[reference_mask]
        order = np.argsort(ref_t, kind="stable")
        ref_t, ref_x = ref_t[order], ref_x[order]
        for row in audit.itertuples(index=False):
            prediction = int(row.prediction_timestamp)
            first = int(np.searchsorted(times, max(int(row.history_start), int(row.phase_and_cycle_start)), side="left"))
            end = int(np.searchsorted(times, prediction, side="right"))
            actual_t = times[first:end]
            actual_x = sensors[first:end]
            if len(actual_t) != int(row.sensor_row_count):
                raise AssertionError(f"Changed history extraction: {tool}, {row.fault_event_id}, {row.horizon}")
            if len(actual_t) and (int(actual_t[-1]) != int(row.last_sensor_timestamp)
                                  or np.any(actual_t > prediction)):
                raise AssertionError("Future sensor or original anchor mismatch")
            # A reference row must be >=24h old when used; only then is its
            # full 24h fault-free status historically knowable at this anchor.
            cutoff = min(prediction - DAY, train_end - DAY)
            n_ref = int(np.searchsorted(ref_t, cutoff, side="right"))
            if row.phase != "train":
                n_ref = int(np.searchsorted(ref_t, train_end - DAY, side="right"))
            reference = ref_x[:n_ref]
            valid_ref = n_ref >= MIN_REFERENCE
            output = {"tool_id": tool, "fault_event_id": int(row.fault_event_id),
                      "phase": row.phase, "horizon": row.horizon,
                      "prediction_timestamp": prediction, "fault_timestamp": int(row.fault_timestamp),
                      "usable_prior_rule": bool(row.usable_prior_rule),
                      "sensor_row_count": int(row.sensor_row_count),
                      "max_gap_seconds": row.max_gap_seconds,
                      "coverage_duration_seconds": row.coverage_duration_seconds,
                      "reference_rows": n_ref, "reference_cutoff": cutoff if row.phase == "train" else train_end - DAY,
                      "reference_available": valid_ref}
            if len(actual_x):
                latest = actual_x[-1]
                window_median = np.median(actual_x, axis=0)
            else:
                latest = np.full(len(SENSORS), np.nan)
                window_median = np.full(len(SENSORS), np.nan)
            for j, sensor in enumerate(SENSORS):
                output[f"raw_last_{sensor}"] = float(latest[j])
                output[f"raw_median_{sensor}"] = float(window_median[j])
                if not valid_ref:
                    for suffix in ("delta", "ratio", "zscore"):
                        output[f"{suffix}_{sensor}"] = np.nan
                    continue
                values = reference[:, j]
                center, mean = float(np.median(values)), float(np.mean(values))
                std = float(np.std(values, ddof=1))
                # Fixed absolute floor only guards zero/near-zero division;
                # no threshold is chosen from Validation outcomes.
                tiny = abs(center) <= 1e-9
                low_std = std <= 1e-9
                output[f"delta_{sensor}"] = float(window_median[j] - center)
                output[f"ratio_{sensor}"] = float(window_median[j] / center) if not tiny else np.nan
                output[f"zscore_{sensor}"] = float((window_median[j] - mean) / std) if not low_std else np.nan
            records.append(output)
            base_stats = ({f"{s}_{name}": float(v) for j, s in enumerate(SENSORS)
                           for name, v in (("median", np.median(reference[:, j])),
                                           ("mean", np.mean(reference[:, j])),
                                           ("std", np.std(reference[:, j], ddof=1)))}
                          if valid_ref else {})
            baselines.append({"tool_id": tool, "fault_event_id": int(row.fault_event_id),
                              "phase": row.phase, "horizon": row.horizon,
                              "prediction_timestamp": prediction, "reference_cutoff": output["reference_cutoff"],
                              "reference_rows": n_ref, "reference_available": valid_ref,
                              **base_stats})
        print("processed", tool, "reference rows", len(ref_t), flush=True)
    frame = pd.DataFrame(records)
    base = pd.DataFrame(baselines)
    assert len(frame) == len(original)
    assert frame[["tool_id", "fault_event_id", "horizon", "prediction_timestamp"]].equals(
        original[["tool_id", "fault_event_id", "horizon", "prediction_timestamp"]])
    frame.to_csv(OUT / "relative_samples.csv", index=False)
    base.to_csv(OUT / "tool_baselines.csv", index=False)
    valid = frame[frame.usable_prior_rule & frame.reference_available].copy()
    kinds = ["raw_last", "raw_median", "delta", "ratio", "zscore"]
    summary = []
    for kind in kinds:
        for sensor in SENSORS:
            column = f"{kind}_{sensor}"
            for _, label in HORIZONS:
                part = valid[valid.horizon.eq(label)][column].dropna().to_numpy()
                summary.append({"representation": kind, "sensor": sensor,
                                "horizon": label, "events": len(part),
                                **(values_stats(part) if len(part) > 1 else
                                   {k: np.nan for k in ("mean", "median", "std", "p25", "p75")})})
    pd.DataFrame(summary).to_csv(OUT / "relative_horizon_summary.csv", index=False)
    paired_rows, tool_rows, gap_rows = [], [], []
    valid["tool_group"] = valid.tool_id.where(valid.tool_id.isin(["03_M02", "09_M01", "08_M01"]), "other")
    for kind in kinds:
        for sensor in SENSORS:
            column = f"{kind}_{sensor}"
            for earlier, later in PAIRS:
                common = {"representation": kind, "sensor": sensor, "earlier": earlier, "later": later}
                paired_rows.append({**common, **paired(valid, column, earlier, later)})
                for group, group_frame in valid.groupby("tool_group"):
                    tool_rows.append({**common, "tool_group": group,
                                      **paired(group_frame, column, earlier, later)})
                for limit in GAPS:
                    subset = valid if limit is None else valid[valid.max_gap_seconds.le(limit)]
                    gap_rows.append({**common, "max_gap_limit_seconds": "none" if limit is None else limit,
                                     "usable_event_horizons": len(subset),
                                     "unique_fault_events": subset.fault_event_id.nunique(),
                                     **paired(subset, column, earlier, later)})
    pd.DataFrame(paired_rows).to_csv(OUT / "paired_relative_trends.csv", index=False)
    pd.DataFrame(tool_rows).to_csv(OUT / "tool_relative_trends.csv", index=False)
    pd.DataFrame(gap_rows).to_csv(OUT / "gap_sensitivity.csv", index=False)
    audit = {"fault_events": 741, "candidate_anchors": len(frame), "usable_with_reference": len(valid),
             "missing_causal_reference_anchors": int((~frame.reference_available).sum()),
             "reference_rule": "Train only; FSP=1; first real observation per wall-clock minute; all nine finite; >24h after previous Train fault and >24h before next Train fault; source rows at least 24h older than each Train anchor; Validation/reserved use frozen Train prefix ending at train_end-24h",
             "reference_min_rows": MIN_REFERENCE, "ratio_denominator_floor": 1e-9,
             "baseline_next_fault_outcome_known_by_anchor": True,
             "models_trained": False, "split_changed": False}
    (OUT / "config.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
