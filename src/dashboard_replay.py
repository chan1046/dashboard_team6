"""Read-only dashboard data access for saved replay and real history."""
from functools import lru_cache
from pathlib import Path
import csv
import json


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/dashboard"
RAW = ROOT / "data/processed"
REFERENCE_TIME = 27317822
PERIOD_SECONDS = {
    12: 12 * 3600,
    24: 24 * 3600,
    72: 72 * 3600,
    168: 7 * 24 * 3600,
    720: 30 * 24 * 3600,
    4320: 180 * 24 * 3600,
}


@lru_cache(maxsize=1)
def fleet_snapshot():
    return json.loads((OUT / "fleet_snapshot.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def fault_type_replay():
    return json.loads((OUT / "fault_type_replay.json").read_text(encoding="utf-8"))


def replay_results(tools, hours=72):
    fleet = fleet_snapshot()["tools"]
    types = fault_type_replay()["tools"]
    counts = fault_counts(hours, REFERENCE_TIME)
    latest = latest_faults(REFERENCE_TIME)
    results = {}
    for tool in tools:
        item = fleet[tool]
        model1 = dict(item["model1"])
        model2 = dict(types[tool])
        results[tool] = {
            "timestamp": model1.get("prediction_timestamp"),
            "reference_time": REFERENCE_TIME,
            "time_basis": "원본 데이터 경과 초",
            "source": "저장된 실제 검증 결과 재생",
            "data_mode": "replay",
            "model1": model1,
            "model2": model2,
            "priority": item["priority"],
            "fault_count": counts.get(tool, 0),
            "seconds_since_fault": (REFERENCE_TIME - latest[tool]) if tool in latest else None,
        }
    return results


@lru_cache(maxsize=1)
def fault_rows():
    seen, rows = set(), []
    with (RAW / "fault_events.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["tool_id"], int(row["time"]), row["fault_type"])
            if key in seen:
                continue
            seen.add(key)
            row["time"] = int(row["time"])
            rows.append(row)
    return rows


def filtered_faults(hours, cutoff=REFERENCE_TIME, tool=None):
    seconds = PERIOD_SECONDS.get(int(hours))
    return [row for row in fault_rows()
            if row["time"] <= cutoff
            and (seconds is None or row["time"] > cutoff - seconds)
            and (tool is None or row["tool_id"] == tool)]


def fault_counts(hours, cutoff=REFERENCE_TIME):
    counts = {}
    for row in filtered_faults(hours, cutoff):
        counts[row["tool_id"]] = counts.get(row["tool_id"], 0) + 1
    return counts


def latest_faults(cutoff=REFERENCE_TIME):
    latest = {}
    for row in filtered_faults(999, cutoff):
        latest[row["tool_id"]] = max(latest.get(row["tool_id"], row["time"]), row["time"])
    return latest


@lru_cache(maxsize=100)
def sensor_series(tool, sensor, timestamp, seconds=7200, limit=600):
    """Return sampled original rows; no interpolation or generated values."""
    if timestamp is None:
        return []
    import pyarrow.parquet as pq
    table = pq.read_table(
        RAW / f"{tool}_sensor_ttf.parquet", columns=["time", sensor],
        filters=[("time", ">=", int(timestamp) - seconds), ("time", "<=", int(timestamp))])
    times = table.column("time").to_pylist()
    values = table.column(sensor).to_pylist()
    rows = [(int(t), v) for t, v in zip(times, values) if v is not None]
    if len(rows) > limit:
        positions = {round(i * (len(rows) - 1) / (limit - 1)) for i in range(limit)}
        rows = [rows[i] for i in sorted(positions)]
    return rows
