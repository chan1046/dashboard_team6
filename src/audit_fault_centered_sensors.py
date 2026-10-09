"""Validate real pre-fault sensor extraction at six fixed Low Pressure horizons.

No model training. Fault-relative horizons are audit labels only, never features.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from train_simple_full_fault_rul import SENSORS


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/processed"
PLAN = ROOT / "output/experiments/forward_validation_ab_24h/split_plan.json"
OUT = ROOT / "output/experiments/fault_centered_sensor_audit"
TTF_COL = "TTF_FlowCool Pressure Dropped Below Limit"
HORIZONS = [(1440, "24h"), (720, "12h"), (360, "6h"),
            (180, "3h"), (60, "1h"), (30, "30m")]
HISTORY_SECONDS = 1800
PRIOR_MIN_OBSERVATIONS = 60
PRIOR_MIN_COVERAGE_SECONDS = 1440
PRIOR_MAX_SENSOR_AGE_SECONDS = 60


def phase_of(fault: int, item: dict) -> tuple[str, int]:
    if fault < int(item["train_end_exclusive_second"]):
        return "train", 0
    if int(item["validation_start_second"]) <= fault < int(item["validation_end_exclusive_second"]):
        return "validation", int(item["validation_start_second"])
    if fault >= int(item["test_start_second"]):
        return "reserved", int(item["test_start_second"])
    return "excluded_purge", 0


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    raw_faults = pd.read_csv(RAW / "fault_events.csv")
    faults = raw_faults.drop_duplicates(["tool_id", "time", "fault_type"])
    low = faults[faults.fault_type == "pressure_low"].copy()
    assert len(low) == 741 and low.fault_event_id.is_unique
    assert low.fault_name.unique().tolist() == ["FlowCool Pressure Dropped Below Limit"]
    OUT.mkdir(parents=True)
    manual_dir = OUT / "manual_raw_rows"
    manual_dir.mkdir()
    records: list[dict] = []
    manual_windows: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for item in plan["tools"]:
        tool = item["tool_id"]
        table = pq.read_table(RAW / f"{tool}_sensor_ttf.parquet",
                              columns=["time", *SENSORS, TTF_COL])
        t = table.column("time").to_numpy(zero_copy_only=False).astype(np.int64)
        x = np.column_stack([table.column(s).to_numpy(zero_copy_only=False) for s in SENSORS]).astype(np.float64)
        supplied = table.column(TTF_COL).to_numpy(zero_copy_only=False).astype(np.float64)
        assert len(t) and np.all(t[1:] >= t[:-1]), tool
        log = faults[faults.tool_id == tool].sort_values("time", kind="stable")
        previous = None
        for event in log.itertuples():
            f = int(event.time)
            phase, phase_start = phase_of(f, item)
            lower = max(int(t[0]), phase_start, previous + 1 if previous is not None else int(t[0]))
            if event.fault_type != "pressure_low":
                previous = f
                continue
            for horizon_minutes, label in HORIZONS:
                prediction = f - horizon_minutes * 60
                history_start = prediction - HISTORY_SECONDS
                first = max(int(np.searchsorted(t, history_start, side="left")),
                            int(np.searchsorted(t, lower, side="left")))
                end = int(np.searchsorted(t, prediction, side="right"))
                n = max(0, end - first)
                selected_t = t[first:end] if n else np.empty(0, dtype=np.int64)
                selected_x = x[first:end] if n else np.empty((0, len(SENSORS)), dtype=np.float64)
                first_t = int(selected_t[0]) if n else None
                last_t = int(selected_t[-1]) if n else None
                age = int(prediction - last_t) if n else None
                span = int(last_t - first_t) if n else None
                gaps = np.diff(selected_t) if n > 1 else np.empty(0, dtype=np.int64)
                median_interval = float(np.median(gaps)) if len(gaps) else None
                max_gap = int(gaps.max()) if len(gaps) else None
                ttf_seconds = f - prediction
                provided = float(supplied[end - 1]) if n and np.isfinite(supplied[end - 1]) else None
                expected_at_sensor = f - last_t if n else None
                adjusted_source = provided - age if provided is not None and age is not None else None
                source_diff = provided - expected_at_sensor if provided is not None else None
                comparison = ("exact" if source_diff == 0 and adjusted_source == ttf_seconds else
                              "mismatch" if provided is not None else
                              "source_missing" if n else "no_sensor_history")
                sensor_is_finite = bool(np.isfinite(selected_x).all()) if n else False
                usable = bool(n >= PRIOR_MIN_OBSERVATIONS and span is not None and
                    span >= PRIOR_MIN_COVERAGE_SECONDS and age is not None and
                    age <= PRIOR_MAX_SENSOR_AGE_SECONDS and sensor_is_finite)
                leak_future_sensor = bool(n and (selected_t > prediction).any())
                leak_after_fault = bool(n and (selected_t >= f).any())
                leak_order = prediction >= f
                row = {"tool_id": tool, "fault_event_id": int(event.fault_event_id),
                    "fault_name": event.fault_name, "fault_type": event.fault_type,
                    "phase": phase, "fault_timestamp": f,
                    "horizon_minutes": horizon_minutes, "horizon": label,
                    "prediction_timestamp": prediction,
                    "history_start": history_start, "history_end": prediction,
                    "phase_and_cycle_start": lower, "sensor_row_count": n,
                    "first_sensor_timestamp": first_t, "last_sensor_timestamp": last_t,
                    "sensor_age_seconds": age, "median_sampling_interval_seconds": median_interval,
                    "max_gap_seconds": max_gap, "coverage_duration_seconds": span,
                    "all_raw_sensors_finite": sensor_is_finite,
                    "usable_prior_rule": usable,
                    "calculated_ttf_seconds": ttf_seconds,
                    "calculated_ttf_hours": ttf_seconds / 3600,
                    "source_ttf_at_last_sensor_seconds": provided,
                    "source_ttf_adjusted_to_prediction_seconds": adjusted_source,
                    "source_expected_at_last_sensor_seconds": expected_at_sensor,
                    "source_ttf_difference_seconds": source_diff,
                    "source_comparison": comparison,
                    "future_sensor_violation": leak_future_sensor,
                    "after_fault_sensor_violation": leak_after_fault,
                    "prediction_at_or_after_fault_violation": leak_order}
                for j, sensor in enumerate(SENSORS):
                    row[f"last_{sensor}"] = float(selected_x[-1, j]) if n and np.isfinite(selected_x[-1, j]) else None
                records.append(row)
                # The five manual events are selected only after metadata is known.
                # Save candidate first/last rows for every event in compact form;
                # these are later restricted to deterministic five selected events.
                if n:
                    ix = np.unique(np.r_[np.arange(min(3, n)), np.arange(max(0, n - 3), n)])
                    manual_windows[(tool, int(event.fault_event_id), horizon_minutes)] = (selected_t[ix].copy(), selected_x[ix].copy())
            previous = f
        print("audited", tool, flush=True)
    frame = pd.DataFrame(records)
    assert len(frame) == 741 * 6
    assert frame.groupby("fault_event_id").size().eq(6).all()
    assert frame.fault_event_id.nunique() == 741
    frame.to_csv(OUT / "sensor_history_quality.csv", index=False)
    ttf_cols = ["tool_id", "fault_event_id", "phase", "horizon", "fault_timestamp",
                "prediction_timestamp", "last_sensor_timestamp", "sensor_age_seconds",
                "calculated_ttf_seconds", "calculated_ttf_hours",
                "source_ttf_at_last_sensor_seconds", "source_ttf_adjusted_to_prediction_seconds",
                "source_expected_at_last_sensor_seconds", "source_ttf_difference_seconds", "source_comparison"]
    frame[ttf_cols].to_csv(OUT / "ttf_consistency_check.csv", index=False)
    bad = frame[frame[["future_sensor_violation", "after_fault_sensor_violation",
                       "prediction_at_or_after_fault_violation"]].any(axis=1)]
    frame[["tool_id", "fault_event_id", "horizon", "future_sensor_violation",
           "after_fault_sensor_violation", "prediction_at_or_after_fault_violation"]].to_csv(
           OUT / "leakage_check.csv", index=False)
    bad.to_csv(OUT / "leakage_errors.csv", index=False)
    assert len(bad) == 0, "Future information leakage detected; inspect leakage_errors.csv"
    coverage = []
    for minutes, label in HORIZONS:
        part = frame[frame.horizon_minutes == minutes]
        good = part[part.usable_prior_rule]
        coverage.append({"horizon": label, "horizon_minutes": minutes,
            "unique_fault_events": part.fault_event_id.nunique(),
            "usable_events": good.fault_event_id.nunique(),
            "coverage_pct_of_741": 100 * good.fault_event_id.nunique() / 741,
            "tools_with_usable_events": good.tool_id.nunique(),
            "train_usable_events": int((good.phase == "train").sum()),
            "validation_usable_events": int((good.phase == "validation").sum()),
            "reserved_or_excluded_usable_events": int(good.phase.isin(["reserved", "excluded_purge"]).sum())})
    pd.DataFrame(coverage).to_csv(OUT / "horizon_coverage.csv", index=False)
    # Deterministic, diverse manual selection: highest usable horizon count
    # within five fixed Tools chosen before inspecting sensor values.
    selected_tools = ["01_M01", "03_M02", "06_M02", "08_M01", "09_M01"]
    selected_ids = []
    for tool in selected_tools:
        part = frame[frame.tool_id == tool].groupby("fault_event_id").agg(
            usable=("usable_prior_rule", "sum"), fault_time=("fault_timestamp", "first"))
        chosen = part.sort_values(["usable", "fault_time"], ascending=[False, True]).index[0]
        selected_ids.append(int(chosen))
    manual = frame[frame.fault_event_id.isin(selected_ids)].copy()
    assert len(manual) == 30 and manual.tool_id.nunique() == 5
    manual.to_csv(OUT / "manual_validation_samples.csv", index=False)
    raw_rows = []
    for row in manual.itertuples():
        key = (row.tool_id, int(row.fault_event_id), int(row.horizon_minutes))
        if key not in manual_windows:
            continue
        times, sensor_rows = manual_windows[key]
        for position, (time, values) in enumerate(zip(times, sensor_rows)):
            raw_rows.append({"tool_id": row.tool_id, "fault_event_id": int(row.fault_event_id),
                "horizon": row.horizon, "row_order_in_first_last_six": position,
                "sensor_timestamp": int(time), "prediction_timestamp": int(row.prediction_timestamp),
                **{s: float(values[j]) for j, s in enumerate(SENSORS)}})
    pd.DataFrame(raw_rows).to_csv(manual_dir / "first_last_3_rows.csv", index=False)
    config = {"model_training": False, "unique_low_events": 741, "samples": len(frame),
        "fault_relative_horizons_minutes": [m for m, _ in HORIZONS],
        "history_seconds": HISTORY_SECONDS, "raw_sensors": SENSORS,
        "history_rule_for_usable": ">=60 real rows, first-to-last span >=24m, latest sensor <=60s old, all nine sensors finite; phase and previous-any-fault boundaries respected",
        "quality_sensitivity": "20m coverage, <=5m max gap, and minimum row count reported descriptively; not newly optimized",
        "ttf_alignment": "provided TTF belongs to last actual sensor timestamp; subtract sensor age to align with nominal prediction timestamp",
        "reserved_sensor_and_ttf_access": "yes, read-only descriptive audit explicitly covers all 741 original Train faults; reserved outcomes are no longer untouched",
        "manual_selection": "preselected five Tools; event with most prior-rule usable horizons in each, tie earliest fault timestamp",
        "sensor_summary_scope": "last actual sensor value at each usable horizon; one event-horizon observation",
        "random_state": 42}
    (OUT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(coverage).to_string(index=False), flush=True)
    print(frame.source_comparison.value_counts().to_string(), flush=True)
    print("manual event ids", selected_ids, flush=True)


if __name__ == "__main__":
    main()
