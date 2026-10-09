"""No-training 3-class fault-event and causal sensor-history availability audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/processed"
PLAN = ROOT / "output/experiments/forward_validation_ab_24h/split_plan.json"
OUT = ROOT / "output/experiments/full_fault_type_data_audit"
TYPES = ["pressure_low", "pressure_high", "leak"]
HISTORY_MINUTES = [10, 30, 60, 180]
HORIZON_MINUTES = [30, 60, 180, 360, 720, 1440]
INTERVALS_HOURS = [(0, 3), (3, 6), (6, 12), (12, 24)]
MIN_OBSERVATIONS = 60
MIN_SPAN = .8
MAX_ANCHOR_AGE_SECONDS = 60
GRID_SECONDS = 1800


def phase_of_fault(f: int, item: dict) -> str:
    if f < int(item["train_end_exclusive_second"]):
        return "train"
    if int(item["validation_start_second"]) <= f < int(item["validation_end_exclusive_second"]):
        return "validation"
    return "reserved_or_excluded"


def actual_history(t: np.ndarray, anchor: int, lower_time: int, minutes: int) -> tuple[bool, int | None, int]:
    """Actual observations in a past-only wall-clock window, no interpolation."""
    last = int(np.searchsorted(t, anchor, side="right")) - 1
    if last < 0 or int(t[last]) < lower_time:
        return False, None, 0
    first = max(int(np.searchsorted(t, anchor - minutes * 60, side="left")),
                int(np.searchsorted(t, lower_time, side="left")))
    if last < first:
        return False, None, 0
    n = last - first + 1
    span = int(t[last] - t[first])
    max_gap = int(np.diff(t[first:last+1]).max()) if n > 1 else None
    return bool(n >= MIN_OBSERVATIONS and span >= MIN_SPAN * minutes * 60), max_gap, n


def candidate_grid(t: np.ndarray, start: int, fault: int) -> np.ndarray:
    """Thirty-minute wall-clock grid, anchored to a fresh observed sensor row."""
    grid = np.arange(((start + GRID_SECONDS - 1) // GRID_SECONDS) * GRID_SECONDS,
                     fault, GRID_SECONDS, dtype=np.int64)
    if not len(grid):
        return grid
    idx = np.searchsorted(t, grid, side="right") - 1
    good = (idx >= 0)
    age = np.full(len(grid), np.iinfo(np.int64).max, dtype=np.int64)
    age[good] = grid[good] - t[idx[good]]
    good &= age <= MAX_ANCHOR_AGE_SECONDS
    good &= t[np.maximum(idx, 0)] >= start
    return np.unique(t[idx[good]])


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    raw_faults = pd.read_csv(RAW / "fault_events.csv")
    faults = raw_faults.drop_duplicates(["tool_id", "time", "fault_type"]).copy()
    assert len(faults) == 1211
    assert faults.fault_event_id.is_unique
    assert faults.fault_type.value_counts().reindex(TYPES).tolist() == [741, 333, 137]
    expected_names = {
        "pressure_low": {"FlowCool Pressure Dropped Below Limit"},
        "pressure_high": {"Flowcool Pressure Too High Check Flowcool Pump",
                          "Flowcool Pressure Too High Check Flowcool Pump [NoWaferID]"},
        "leak": {"Flowcool leak"},
    }
    for typ, names in expected_names.items():
        assert set(faults.loc[faults.fault_type == typ, "fault_name"]) == names
    records = []
    for item in plan["tools"]:
        tool = item["tool_id"]
        t = pq.read_table(RAW / f"{tool}_sensor_ttf.parquet", columns=["time"]).column("time").to_numpy().astype(np.int64)
        assert len(t) and np.all(t[1:] >= t[:-1]), tool
        log = faults[faults.tool_id == tool].sort_values("time", kind="stable")
        previous_fault = None
        for event in log.itertuples():
            f = int(event.time)
            phase = phase_of_fault(f, item)
            if phase == "train":
                phase_start = int(t[0])
            elif phase == "validation":
                phase_start = int(item["validation_start_second"])
            elif f >= int(item["test_start_second"]):
                phase_start = int(item["test_start_second"])
            else:
                # Purge is descriptive only; it is never an eligible training phase.
                phase_start = int(t[0])
            lower = max(phase_start, previous_fault + 1 if previous_fault is not None else int(t[0]))
            before = int(np.searchsorted(t, f, side="left")) - 1
            last_sensor = int(t[before]) if before >= 0 and t[before] >= lower else None
            row = {"fault_event_id": int(event.fault_event_id), "tool_id": tool,
                "fault_time": f, "fault_name": event.fault_name,
                "fault_type": event.fault_type, "phase": phase,
                "previous_any_fault_time": previous_fault,
                "phase_and_cycle_start": lower,
                "last_sensor_time": last_sensor,
                "last_sensor_lead_seconds": f - last_sensor if last_sensor is not None else None}
            for minutes in HISTORY_MINUTES:
                if last_sensor is None:
                    ok, gap, obs = False, None, 0
                else:
                    ok, gap, obs = actual_history(t, last_sensor, lower, minutes)
                row[f"history_{minutes}m"] = ok
                row[f"history_{minutes}m_observations"] = obs
                row[f"history_{minutes}m_max_gap_seconds"] = gap
                row[f"history_{minutes}m_gap90"] = bool(ok and gap is not None and gap <= 90)
            if phase in ("train", "validation"):
                for minutes in HORIZON_MINUTES:
                    target = f - minutes * 60
                    idx = int(np.searchsorted(t, target, side="right")) - 1
                    fresh = idx >= 0 and t[idx] >= lower and target - t[idx] <= MAX_ANCHOR_AGE_SECONDS
                    if fresh:
                        ok, gap, obs = actual_history(t, int(t[idx]), lower, 30)
                    else:
                        ok, gap, obs = False, None, 0
                    row[f"horizon_{minutes}m_30m_history"] = bool(fresh and ok)
                    row[f"horizon_{minutes}m_gap90"] = bool(fresh and ok and gap is not None and gap <= 90)
                candidate = candidate_grid(t, lower, f)
                if len(candidate):
                    eligible = []
                    strict = []
                    for anchor in candidate:
                        ok, gap, obs = actual_history(t, int(anchor), lower, 30)
                        if ok:
                            eligible.append(int(anchor))
                            strict.append(bool(gap is not None and gap <= 90))
                    candidate = np.asarray(eligible, dtype=np.int64)
                    strict = np.asarray(strict, dtype=bool)
                else:
                    strict = np.zeros(0, dtype=bool)
                horizon_hours = (f - candidate) / 3600
                for lo, hi in INTERVALS_HOURS:
                    mask = (horizon_hours > lo) & (horizon_hours <= hi)
                    label = f"interval_{lo}_{hi}h"
                    row[label] = bool(mask.any())
                    row[label + "_anchors"] = int(mask.sum())
                    row[label + "_gap90"] = bool((mask & strict).any())
            records.append(row)
            previous_fault = f
        print("audited", tool, len(log), flush=True)
    events = pd.DataFrame(records)
    assert len(events) == 1211 and events.fault_event_id.is_unique
    assert not (set(events.loc[events.phase == "train", "fault_event_id"]) &
                set(events.loc[events.phase == "validation", "fault_event_id"]))
    OUT.mkdir(parents=True)
    events.to_csv(OUT / "event_availability.csv", index=False)
    name_counts = (faults.groupby(["fault_type", "fault_name"]).size()
                   .rename("unique_events").reset_index())
    name_counts.to_csv(OUT / "fault_name_mapping.csv", index=False)
    split_counts = (events.groupby(["fault_type", "phase"]).size().rename("unique_events")
                    .reset_index())
    split_counts.to_csv(OUT / "class_split_counts.csv", index=False)
    coverage = []
    for minutes in HISTORY_MINUTES:
        for typ in TYPES:
            for phase in ("train", "validation", "reserved_or_excluded", "all"):
                part = events[events.fault_type == typ]
                if phase != "all":
                    part = part[part.phase == phase]
                count = int(part[f"history_{minutes}m"].sum())
                strict = int(part[f"history_{minutes}m_gap90"].sum())
                coverage.append({"history_minutes": minutes, "fault_type": typ, "phase": phase,
                    "total_events": len(part), "usable_events": count,
                    "coverage_pct": 100 * count / len(part) if len(part) else None,
                    "usable_gap90_events": strict,
                    "tools_with_usable_events": part.loc[part[f"history_{minutes}m"], "tool_id"].nunique()})
    pd.DataFrame(coverage).to_csv(OUT / "history_coverage.csv", index=False)
    tool_counts = []
    for tool in [item["tool_id"] for item in plan["tools"]]:
        for typ in TYPES:
            part = events[(events.tool_id == tool) & (events.fault_type == typ)]
            tool_counts.append({"tool_id": tool, "fault_type": typ,
                "total_events": len(part), "train_events": int((part.phase == "train").sum()),
                "validation_events": int((part.phase == "validation").sum()),
                "reserved_or_excluded_events": int((part.phase == "reserved_or_excluded").sum()),
                "train_30m_usable": int(part.loc[part.phase == "train", "history_30m"].sum()),
                "validation_30m_usable": int(part.loc[part.phase == "validation", "history_30m"].sum()),
                "train_60m_usable": int(part.loc[part.phase == "train", "history_60m"].sum()),
                "validation_60m_usable": int(part.loc[part.phase == "validation", "history_60m"].sum())})
    pd.DataFrame(tool_counts).to_csv(OUT / "tool_fault_counts.csv", index=False)
    horizons = []
    intervals = []
    for minutes in HORIZON_MINUTES:
        for typ in TYPES:
            for phase in ("train", "validation"):
                part = events[(events.fault_type == typ) & (events.phase == phase)]
                column = f"horizon_{minutes}m_30m_history"
                horizons.append({"horizon_minutes_before_fault": minutes, "fault_type": typ,
                    "phase": phase, "total_events": len(part), "usable_events": int(part[column].sum()),
                    "coverage_pct": 100 * part[column].mean() if len(part) else None,
                    "usable_gap90_events": int(part[f"horizon_{minutes}m_gap90"].sum())})
    for lo, hi in INTERVALS_HOURS:
        label = f"interval_{lo}_{hi}h"
        for typ in TYPES:
            for phase in ("train", "validation"):
                part = events[(events.fault_type == typ) & (events.phase == phase)]
                intervals.append({"interval": f"{lo}-{hi}h", "fault_type": typ, "phase": phase,
                    "total_events": len(part), "usable_events": int(part[label].sum()),
                    "coverage_pct": 100 * part[label].mean() if len(part) else None,
                    "usable_gap90_events": int(part[label + "_gap90"].sum()),
                    "candidate_anchors": int(part[label + "_anchors"].sum())})
    pd.DataFrame(horizons).to_csv(OUT / "horizon_anchor_coverage.csv", index=False)
    pd.DataFrame(intervals).to_csv(OUT / "multi_anchor_interval_coverage.csv", index=False)
    balance = []
    for minutes in (30, 60):
        for phase in ("train", "validation"):
            part = events[events.phase == phase]
            usable = part[part[f"history_{minutes}m"]]
            counts = usable.fault_type.value_counts().reindex(TYPES, fill_value=0)
            majority = int(counts.max())
            total = int(counts.sum())
            share = majority / total if total else 0
            macro_f1 = (2 * share / (1 + share) / 3) if total else None
            for typ in TYPES:
                balance.append({"history_minutes": minutes, "phase": phase, "fault_type": typ,
                    "usable_events": int(counts[typ]), "total_usable_events": total,
                    "class_share_pct": 100 * int(counts[typ]) / total if total else None,
                    "dummy_majority_class": counts.idxmax(),
                    "dummy_majority_accuracy": share,
                    "dummy_macro_f1": macro_f1,
                    "dummy_balanced_accuracy": 1 / 3 if all(counts > 0) else None})
    pd.DataFrame(balance).to_csv(OUT / "class_balance.csv", index=False)
    config = {"model_training": False, "unique_fault_events": len(events),
        "raw_fault_rows": len(raw_faults), "duplicates_removed": len(raw_faults) - len(events),
        "split_definition": str(PLAN.relative_to(ROOT)),
        "history_rule": ">=60 real observations spanning >=80% of requested wall-clock history since previous any-type fault and within phase; no interpolation",
        "horizon_rule": "latest real sensor observation no more than 60s before exact pre-fault horizon; preceding 30m history meets history rule",
        "multi_anchor_rule": "30m wall-clock grid; latest real observation <=60s old; usable preceding 30m history; TTF bin only for audit, never feature",
        "gap90_sensitivity": "max 90s internal observation gap recorded separately; not primary availability",
        "reserved_access": "time column only for descriptive history availability; no reserved sensor values or model evaluation",
        "future_information": "fault time and type used only for event eligibility, phase assignment, and audit labels; no model trained",
        "random_state": 42}
    (OUT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(split_counts.to_string(index=False), flush=True)
    print(pd.DataFrame(coverage).query('phase in ["train", "validation"]').to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
