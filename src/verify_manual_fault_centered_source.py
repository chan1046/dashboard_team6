"""Check five manual examples against original Train sensor and train_ttf CSVs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_simple_full_fault_rul import SENSORS


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/experiments/fault_centered_sensor_audit"
TTF = "TTF_FlowCool Pressure Dropped Below Limit"


def selected_source(path: Path, times: set[int], columns: list[str]) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=100_000):
        found = chunk[chunk.time.isin(times)]
        if len(found):
            parts.append(found)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def main() -> None:
    manual_rows = pd.read_csv(OUT / "manual_raw_rows/first_last_3_rows.csv")
    examples = pd.read_csv(OUT / "manual_validation_samples.csv")
    checks = []
    for tool, part in manual_rows.groupby("tool_id"):
        times = set(part.sensor_timestamp.astype(int))
        source = selected_source(ROOT / f"data/train/{tool}_DC_train.csv", times, ["time", *SENSORS])
        source_by_time = {time: values[SENSORS].to_numpy(dtype=float)
                          for time, values in source.groupby("time")}
        for row in part.itertuples():
            target = np.asarray([getattr(row, sensor) for sensor in SENSORS], dtype=float)
            candidates = source_by_time.get(int(row.sensor_timestamp), np.empty((0, len(SENSORS))))
            matched = bool(any(np.allclose(target, candidate, atol=1e-12, rtol=1e-12, equal_nan=True)
                               for candidate in candidates))
            checks.append({"tool_id": tool, "fault_event_id": int(row.fault_event_id),
                "horizon": row.horizon, "sensor_timestamp": int(row.sensor_timestamp),
                "original_sensor_rows_same_timestamp": len(candidates), "raw_sensor_match": matched})
    sensor_checks = pd.DataFrame(checks)
    source_ttf_checks = []
    for tool, part in examples[examples.last_sensor_timestamp.notna()].groupby("tool_id"):
        times = set(part.last_sensor_timestamp.astype(int))
        source = selected_source(ROOT / f"data/train/train_ttf/{tool}_DC_train_ttf.csv", times,
                                 ["time", TTF])
        by_time = {time: values[TTF].to_numpy(dtype=float) for time, values in source.groupby("time")}
        for row in part.itertuples():
            provided = getattr(row, "source_ttf_at_last_sensor_seconds")
            candidates = by_time.get(int(row.last_sensor_timestamp), np.empty(0))
            matched = bool(any(np.isclose(provided, value, atol=0, rtol=0, equal_nan=True)
                               for value in candidates))
            source_ttf_checks.append({"tool_id": tool, "fault_event_id": int(row.fault_event_id),
                "horizon": row.horizon, "last_sensor_timestamp": int(row.last_sensor_timestamp),
                "original_ttf_rows_same_timestamp": len(candidates), "raw_ttf_match": matched})
    ttf_checks = pd.DataFrame(source_ttf_checks)
    sensor_checks.to_csv(OUT / "manual_raw_sensor_source_check.csv", index=False)
    ttf_checks.to_csv(OUT / "manual_raw_ttf_source_check.csv", index=False)
    result = {"manual_sensor_rows": len(sensor_checks), "sensor_matches": int(sensor_checks.raw_sensor_match.sum()),
        "manual_ttf_comparisons": len(ttf_checks), "ttf_matches": int(ttf_checks.raw_ttf_match.sum())}
    (OUT / "manual_source_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    assert sensor_checks.raw_sensor_match.all()
    assert ttf_checks.raw_ttf_match.all()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
