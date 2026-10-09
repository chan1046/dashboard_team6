"""Audit whether long sensor gaps can serve as a fault-outage proxy.

Analysis only: this does not calculate loss, change dashboard behavior, or touch
the frozen Model 1/Model 2 training and inference paths.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/processed"
OUT = ROOT / "output/experiments/fault_outage_proxy_audit"
GAP_THRESHOLDS = (30, 60, 120, 300)
LOOKAHEAD_WINDOWS = (60, 300, 600, 1800)
FAULT_TYPES = ("pressure_low", "pressure_high", "leak")


def load_timestamps(tool_id: str) -> np.ndarray:
    values = (
        pq.read_table(RAW / f"{tool_id}_sensor_ttf.parquet", columns=["time"])
        .column("time")
        .to_numpy()
        .astype(np.int64)
    )
    if len(values) < 2 or np.any(np.diff(values) < 0):
        raise ValueError(f"Sensor timestamps are insufficient or unsorted: {tool_id}")
    return values


def mode_smallest(values: np.ndarray) -> int:
    unique, counts = np.unique(values, return_counts=True)
    return int(unique[int(np.argmax(counts))])


def main() -> None:
    # Retain all source rows here: repeated alarms are deduplicated only after
    # they are linked to the exact same outage candidate.
    faults = pd.read_csv(RAW / "fault_events.csv")
    tools = sorted(faults.tool_id.unique())
    timestamps = {tool_id: load_timestamps(tool_id) for tool_id in tools}

    sampling_rows = []
    all_gap_rows = []
    gap_lookup: dict[tuple[int, str], pd.DataFrame] = {}
    for tool_id, times in timestamps.items():
        differences = np.diff(times)
        sampling_rows.append({
            "tool_id": tool_id,
            "sensor_row_count": int(len(times)),
            "diff_count": int(len(differences)),
            "median_seconds": float(np.median(differences)),
            "mode_seconds": mode_smallest(differences),
            "p95_seconds": float(np.quantile(differences, 0.95)),
            "p99_seconds": float(np.quantile(differences, 0.99)),
            "p99_9_seconds": float(np.quantile(differences, 0.999)),
            "max_seconds": int(differences.max()),
        })
        for threshold in GAP_THRESHOLDS:
            indices = np.flatnonzero(differences > threshold)
            gaps = pd.DataFrame({
                "gap_threshold": threshold,
                "tool_id": tool_id,
                "gap_start": times[indices],
                "gap_end": times[indices + 1],
                "gap_seconds": differences[indices],
            })
            gaps["gap_hours"] = gaps.gap_seconds / 3600
            gap_lookup[(threshold, tool_id)] = gaps
            all_gap_rows.append(gaps)

    sampling = pd.DataFrame(sampling_rows)
    gap_candidates = pd.concat(all_gap_rows, ignore_index=True)

    link_rows = []
    for threshold in GAP_THRESHOLDS:
        for lookahead in LOOKAHEAD_WINDOWS:
            for fault in faults.itertuples(index=False):
                gaps = gap_lookup[(threshold, fault.tool_id)]
                starts = gaps.gap_start.to_numpy(dtype=np.int64)
                # Strictly after fault_time, as requested. A gap that already
                # began before the alarm is not matched by this rule.
                index = int(np.searchsorted(starts, int(fault.time), side="right"))
                matched = bool(
                    index < len(starts)
                    and int(starts[index]) <= int(fault.time) + lookahead
                )
                gap = gaps.iloc[index] if matched else None
                link_rows.append({
                    "gap_threshold": threshold,
                    "lookahead_window": lookahead,
                    "fault_event_id": int(fault.fault_event_id),
                    "tool_id": fault.tool_id,
                    "fault_type": fault.fault_type,
                    "fault_time": int(fault.time),
                    "matched": matched,
                    "gap_start": int(gap.gap_start) if matched else None,
                    "gap_end": int(gap.gap_end) if matched else None,
                    "gap_seconds": int(gap.gap_seconds) if matched else None,
                    "gap_hours": float(gap.gap_hours) if matched else None,
                })
    links = pd.DataFrame(link_rows)

    summary_rows = []
    by_type_rows = []
    dedup_rows = []
    for (threshold, lookahead), group in links.groupby(
        ["gap_threshold", "lookahead_window"], sort=True
    ):
        matched = group[group.matched].copy()
        outages = matched.drop_duplicates(["tool_id", "gap_start", "gap_end"])
        hours = outages.gap_hours
        summary_rows.append({
            "gap_threshold": threshold,
            "lookahead_window": lookahead,
            "raw_fault_count": int(len(group)),
            "matched_fault_count": int(len(matched)),
            "unique_outage_count": int(len(outages)),
            "unmatched_fault_count": int((~group.matched).sum()),
            "median_outage_hours": float(hours.median()) if len(hours) else None,
            "p25": float(hours.quantile(0.25)) if len(hours) else None,
            "p75": float(hours.quantile(0.75)) if len(hours) else None,
            "max": float(hours.max()) if len(hours) else None,
        })
        for fault_type in FAULT_TYPES:
            typed = matched[matched.fault_type == fault_type].drop_duplicates(
                ["tool_id", "gap_start", "gap_end"]
            )
            by_type_rows.append({
                "gap_threshold": threshold,
                "lookahead_window": lookahead,
                "fault_type": fault_type,
                "unique_outage_count": int(len(typed)),
            })
        for keys, alarms in matched.groupby(["tool_id", "gap_start", "gap_end"], sort=True):
            tool_id, gap_start, gap_end = keys
            dedup_rows.append({
                "gap_threshold": threshold,
                "lookahead_window": lookahead,
                "tool_id": tool_id,
                "gap_start": int(gap_start),
                "gap_end": int(gap_end),
                "gap_seconds": int(alarms.gap_seconds.iloc[0]),
                "gap_hours": float(alarms.gap_hours.iloc[0]),
                "linked_fault_alarm_count": int(len(alarms)),
                "linked_fault_types": "|".join(sorted(alarms.fault_type.unique())),
                "linked_fault_event_ids": "|".join(map(str, alarms.fault_event_id.tolist())),
            })

    summary = pd.DataFrame(summary_rows)
    by_type = pd.DataFrame(by_type_rows)
    deduplicated = pd.DataFrame(dedup_rows)

    sensitivity_rows = []
    for lookahead in LOOKAHEAD_WINDOWS:
        part = summary[summary.lookahead_window == lookahead].set_index("gap_threshold")
        for lower, upper in zip(GAP_THRESHOLDS, GAP_THRESHOLDS[1:]):
            low = part.loc[lower]
            high = part.loc[upper]
            sensitivity_rows.append({
                "lookahead_window": lookahead,
                "threshold_from": lower,
                "threshold_to": upper,
                "matched_fault_change_pct": 100 * (
                    high.matched_fault_count - low.matched_fault_count
                ) / low.matched_fault_count,
                "unique_outage_change_pct": 100 * (
                    high.unique_outage_count - low.unique_outage_count
                ) / low.unique_outage_count,
                "median_outage_change_pct": 100 * (
                    high.median_outage_hours - low.median_outage_hours
                ) / low.median_outage_hours,
            })
    sensitivity = pd.DataFrame(sensitivity_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    sampling.to_csv(OUT / "sampling_interval_summary.csv", index=False)
    gap_candidates.to_csv(OUT / "outage_gap_candidates.csv", index=False)
    links.to_csv(OUT / "fault_outage_links.csv", index=False)
    summary.to_csv(OUT / "combination_summary.csv", index=False)
    by_type.to_csv(OUT / "fault_type_unique_outages.csv", index=False)
    deduplicated.to_csv(OUT / "deduplicated_outages.csv", index=False)
    sensitivity.to_csv(OUT / "threshold_sensitivity.csv", index=False)

    print("SAMPLING INTERVALS (SECONDS)")
    print(sampling.to_string(index=False))
    print("\nCOMBINATION SUMMARY")
    print(summary.to_string(index=False))
    print("\nFAULT-TYPE UNIQUE OUTAGES")
    print(by_type.to_string(index=False))
    print("\nTHRESHOLD SENSITIVITY")
    print(sensitivity.to_string(index=False))


if __name__ == "__main__":
    main()
