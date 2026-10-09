"""Audit sensor timestamp structure around every unique fault event.

This script is intentionally analysis-only. It does not calculate loss profiles or
touch Model 1/Model 2. The first strictly post-fault sensor observation is retained
without trimming so sampling-interval and long-gap cases remain visible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/processed"
OUT = ROOT / "output/experiments/fault_post_gap_audit"
FAULT_TYPES = ["pressure_low", "pressure_high", "leak"]
CONTEXT_COLUMNS = [
    "fault_event_id",
    "tool_id",
    "fault_type",
    "fault_time",
    "pre_3_time",
    "pre_2_time",
    "pre_1_time",
    "post_1_time",
    "post_2_time",
    "post_3_time",
    "post_4_time",
    "post_5_time",
    "post_fault_gap_seconds",
    "post_fault_gap_hours",
    "sampling_interval_signal",
]


def padded(values: np.ndarray, width: int, *, left: bool = False) -> list[int | None]:
    result = [int(value) for value in values]
    missing = [None] * (width - len(result))
    return missing + result if left else result + missing


def main() -> None:
    faults = (
        pd.read_csv(RAW / "fault_events.csv")
        .drop_duplicates(["tool_id", "time", "fault_type"])
        .sort_values(["tool_id", "time", "fault_type"], kind="stable")
    )
    records: list[dict] = []
    for tool_id, tool_faults in faults.groupby("tool_id", sort=True):
        timestamps = (
            pq.read_table(RAW / f"{tool_id}_sensor_ttf.parquet", columns=["time"])
            .column("time")
            .to_numpy()
            .astype(np.int64)
        )
        if not len(timestamps) or np.any(np.diff(timestamps) < 0):
            raise ValueError(f"Sensor timestamps are empty or unsorted: {tool_id}")

        for event in tool_faults.itertuples(index=False):
            fault_time = int(event.time)
            before_index = int(np.searchsorted(timestamps, fault_time, side="left"))
            after_index = int(np.searchsorted(timestamps, fault_time, side="right"))
            previous = padded(timestamps[max(0, before_index - 3):before_index], 3, left=True)
            following = padded(timestamps[after_index:after_index + 5], 5)
            gap_seconds = following[0] - fault_time if following[0] is not None else None
            records.append(dict(zip(CONTEXT_COLUMNS, [
                int(event.fault_event_id),
                tool_id,
                event.fault_type,
                fault_time,
                *previous,
                *following,
                gap_seconds,
                gap_seconds / 3600 if gap_seconds is not None else None,
                bool(gap_seconds is not None and gap_seconds <= 10),
            ])))

    context = pd.DataFrame(records, columns=CONTEXT_COLUMNS)
    if len(context) != len(faults):
        raise AssertionError("Not every unique fault event was audited")

    summary_rows = []
    for fault_type in FAULT_TYPES:
        values = context.loc[context.fault_type == fault_type, "post_fault_gap_hours"].dropna()
        summary_rows.append({
            "fault_type": fault_type,
            "event_count": int((context.fault_type == fault_type).sum()),
            "valid_gap_count": int(len(values)),
            "median_post_fault_gap_hours": float(values.median()) if len(values) else None,
            "min": float(values.min()) if len(values) else None,
            "p25": float(values.quantile(0.25)) if len(values) else None,
            "mean": float(values.mean()) if len(values) else None,
            "p75": float(values.quantile(0.75)) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
            "sampling_interval_signal_count": int(
                context.loc[context.fault_type == fault_type, "sampling_interval_signal"].sum()
            ),
        })
    summary = pd.DataFrame(summary_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    context.to_csv(OUT / "event_timestamp_context.csv", index=False)
    summary.to_csv(OUT / "gap_summary_unfiltered.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote {len(context):,} event contexts to {OUT / 'event_timestamp_context.csv'}")


if __name__ == "__main__":
    main()
