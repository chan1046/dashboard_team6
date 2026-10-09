"""No-training audit of next-low-pressure-fault TTF on the frozen split."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from audit_simple_full_fault_rul import candidates


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/processed"
PLAN = ROOT / "output/experiments/forward_validation_ab_24h/split_plan.json"
OUT = ROOT / "output/experiments/low_pressure_rul_audit"
HISTORY_MINUTES = [30, 60, 180, 360, 720, 1440]
CADENCE_MINUTES = [10, 30, 60]
BINS_HOURS = [0, 3, 6, 12, 24, 72, 168, np.inf]
BIN_NAMES = ["0-3h", "3-6h", "6-12h", "12-24h", "24-72h", "72-168h", "168h+"]


class TTFDistribution:
    """Exact means/counts, quantiles binned to one-minute resolution."""

    def __init__(self) -> None:
        self.count = 0
        self.events = 0
        self.total_seconds = 0.0
        self.event_mean_hours = 0.0
        self.maximum_seconds = 0
        self.row_hist = np.zeros(1, dtype=np.int64)
        self.event_hist = np.zeros(1, dtype=np.float64)
        self.row_bins = np.zeros(len(BIN_NAMES), dtype=np.int64)
        self.event_bins = np.zeros(len(BIN_NAMES), dtype=np.float64)

    def add(self, seconds: np.ndarray) -> None:
        if not len(seconds):
            return
        self.count += len(seconds)
        self.events += 1
        self.total_seconds += float(seconds.sum(dtype=np.float64))
        self.event_mean_hours += float(seconds.mean()) / 3600
        self.maximum_seconds = max(self.maximum_seconds, int(seconds.max()))
        minute_hist = np.bincount((seconds // 60).astype(np.int64))
        if len(self.row_hist) < len(minute_hist):
            extra = len(minute_hist) - len(self.row_hist)
            self.row_hist = np.pad(self.row_hist, (0, extra))
            self.event_hist = np.pad(self.event_hist, (0, extra))
        self.row_hist[: len(minute_hist)] += minute_hist
        self.event_hist[: len(minute_hist)] += minute_hist / len(seconds)
        h = seconds / 3600
        for j, (lo, hi) in enumerate(zip(BINS_HOURS[:-1], BINS_HOURS[1:])):
            n = int(np.count_nonzero((h >= lo) & (h < hi)))
            self.row_bins[j] += n
            self.event_bins[j] += n / len(seconds)

    @staticmethod
    def quantile(hist: np.ndarray, q: float) -> float | None:
        if not hist.sum():
            return None
        return float(np.searchsorted(np.cumsum(hist), q * hist.sum(), side="left") / 60)

    def summary(self, phase: str, weighting: str) -> dict:
        hist = self.row_hist if weighting == "row" else self.event_hist
        mean = (self.total_seconds / self.count / 3600 if weighting == "row" else
                self.event_mean_hours / self.events) if self.events else None
        return {"phase": phase, "weighting": weighting, "count": self.count if weighting == "row" else self.events,
                "unique_events": self.events, "mean_hours": mean,
                **{f"{name}_hours": self.quantile(hist, q) for name, q in
                   [("p25", .25), ("median", .5), ("p75", .75), ("p90", .9), ("p95", .95)]},
                "max_hours": self.maximum_seconds / 3600 if self.events else None,
                "quantile_resolution": "1 minute"}

    def bins(self, phase: str, weighting: str) -> list[dict]:
        counts = self.row_bins if weighting == "row" else self.event_bins
        total = self.count if weighting == "row" else self.events
        return [{"phase": phase, "weighting": weighting, "ttf_bin": label,
                 "count_or_event_weight": float(counts[j]),
                 "fraction": float(counts[j] / total) if total else None}
                for j, label in enumerate(BIN_NAMES)]


def phase_for_fault(f: int, item: dict) -> str:
    if f < item["train_end_exclusive_second"]:
        return "train"
    if item["validation_start_second"] <= f < item["validation_end_exclusive_second"]:
        return "validation"
    return "reserved_or_purge"


def history_audit(t: np.ndarray, lo: int, hi: int, minutes: int) -> tuple[bool, int | None]:
    if hi <= lo:
        return False, None
    last = hi - 1
    start = max(lo, int(np.searchsorted(t, t[last] - minutes * 60, side="left")))
    segment = t[start:hi]
    maximum_gap = int(np.diff(segment).max()) if len(segment) > 1 else None
    enough = len(segment) >= 60 and t[last] - t[start] >= .8 * minutes * 60
    return bool(enough), maximum_gap


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    faults = pd.read_csv(RAW / "fault_events.csv").drop_duplicates(["tool_id", "time", "fault_type"])
    assert len(faults) == 1211
    low = faults[faults.fault_type == "pressure_low"].copy()
    assert low.fault_name.nunique() == 1
    assert low.fault_name.iloc[0] == "FlowCool Pressure Dropped Below Limit"
    assert low.fault_event_id.nunique() == len(low)
    distributions = {phase: TTFDistribution() for phase in ("train", "validation")}
    event_rows: list[dict] = []
    anchor_rows: list[dict] = []
    cross_rows: list[dict] = []
    for item in plan["tools"]:
        tool = item["tool_id"]
        t = pq.read_table(RAW / f"{tool}_sensor_ttf.parquet", columns=["time"]).column("time").to_numpy().astype(np.int64)
        assert np.all(t[1:] >= t[:-1]), tool
        log = faults[faults.tool_id == tool].sort_values("time", kind="stable")
        lows = log[log.fault_type == "pressure_low"]
        all_times = log.time.to_numpy(dtype=np.int64)
        other = log[log.fault_type != "pressure_low"]
        other_t = other.time.to_numpy(dtype=np.int64)
        high_t = other[other.fault_type == "pressure_high"].time.to_numpy(dtype=np.int64)
        leak_t = other[other.fault_type == "leak"].time.to_numpy(dtype=np.int64)
        prev_low = None
        for event in lows.itertuples():
            f = int(event.time)
            phase = phase_for_fault(f, item)
            phase_start = (int(t[0]) if phase == "train" else
                           int(item["validation_start_second"]) if phase == "validation" else int(t[0]))
            prev_any_idx = int(np.searchsorted(all_times, f, side="left")) - 1
            prev_any = int(all_times[prev_any_idx]) if prev_any_idx >= 0 else None
            low_start = max(phase_start, prev_low + 1 if prev_low is not None else int(t[0]))
            any_start = max(low_start, prev_any + 1 if prev_any is not None else low_start)
            # A retains next-Low target across other faults. B restarts history
            # after the last fault of any type, but keeps the same target.
            lo_a = int(np.searchsorted(t, low_start, side="left"))
            lo_b = int(np.searchsorted(t, any_start, side="left"))
            hi = int(np.searchsorted(t, f, side="left"))
            n_high = int(np.count_nonzero((high_t > (prev_low if prev_low is not None else -1)) & (high_t < f)))
            n_leak = int(np.count_nonzero((leak_t > (prev_low if prev_low is not None else -1)) & (leak_t < f)))
            row = {"tool_id": tool, "fault_event_id": int(event.fault_event_id), "fault_time": f,
                   "phase": phase, "previous_low_fault_time": prev_low, "previous_any_fault_time": prev_any,
                   "previous_low_interval_hours": (f - prev_low) / 3600 if prev_low is not None else None,
                   "other_high_between_lows": n_high, "other_leak_between_lows": n_leak,
                   "any_other_between_lows": bool(n_high + n_leak),
                   "observations_A": max(0, hi - lo_a), "observations_B": max(0, hi - lo_b),
                   "last_sensor_lead_hours": (f - int(t[hi - 1])) / 3600 if hi > lo_a else None}
            for minutes in HISTORY_MINUTES:
                for rule, lo in (("A", lo_a), ("B", lo_b)):
                    ok, max_gap = history_audit(t, lo, hi, minutes)
                    row[f"{rule}_history_{minutes}m"] = ok
                    row[f"{rule}_max_gap_{minutes}m_seconds"] = max_gap
            event_rows.append(row)
            if phase in distributions and hi > lo_a:
                observed = t[lo_a:hi]
                remaining = f - observed
                assert np.all(remaining > 0)
                distributions[phase].add(remaining)
                after_high = np.searchsorted(high_t, observed, side="right")
                after_leak = np.searchsorted(leak_t, observed, side="right")
                between_high = int(np.count_nonzero((after_high < len(high_t)) &
                    (high_t[np.minimum(after_high, len(high_t) - 1)] < f))) if len(high_t) else 0
                between_leak = int(np.count_nonzero((after_leak < len(leak_t)) &
                    (leak_t[np.minimum(after_leak, len(leak_t) - 1)] < f))) if len(leak_t) else 0
                high_mask = (after_high < len(high_t)) & (high_t[np.minimum(after_high, len(high_t)-1)] < f) if len(high_t) else np.zeros(len(observed), bool)
                leak_mask = (after_leak < len(leak_t)) & (leak_t[np.minimum(after_leak, len(leak_t)-1)] < f) if len(leak_t) else np.zeros(len(observed), bool)
                cross_rows.append({"fault_event_id": int(event.fault_event_id), "phase": phase,
                    "rows": len(observed), "rows_high_intervenes": between_high,
                    "rows_leak_intervenes": between_leak,
                    "rows_any_intervenes": int(np.count_nonzero(high_mask | leak_mask))})
                for minutes in (30, 60, 180):
                    for cadence in CADENCE_MINUTES:
                        for rule, lo in (("A", lo_a), ("B", lo_b)):
                            aa = candidates(t, lo, hi, minutes * 60, cadence * 60)
                            anchor_rows.append({"tool_id": tool, "fault_event_id": int(event.fault_event_id),
                                "phase": phase, "history_minutes": minutes,
                                "cadence_minutes": cadence, "rule": rule,
                                "candidate_samples": len(aa), "has_candidate": bool(len(aa))})
            prev_low = f
        print("audited", tool, len(lows), flush=True)
    events = pd.DataFrame(event_rows)
    anchors = pd.DataFrame(anchor_rows)
    crossing = pd.DataFrame(cross_rows)
    assert len(events) == len(low) == 741
    assert events.fault_event_id.is_unique
    assert not (set(events.loc[events.phase == "train", "fault_event_id"]) &
                set(events.loc[events.phase == "validation", "fault_event_id"]))
    OUT.mkdir(parents=True)
    events.to_csv(OUT / "low_pressure_events.csv", index=False)
    anchors.to_csv(OUT / "anchor_counts_by_event.csv", index=False)
    crossing.to_csv(OUT / "intervening_fault_rows.csv", index=False)
    tool_rows = []
    for item in plan["tools"]:
        tool = item["tool_id"]
        part = events[events.tool_id == tool]
        gaps = part.previous_low_interval_hours.dropna()
        tool_rows.append({"tool_id": tool, "total_events": len(part),
            "train_events": int((part.phase == "train").sum()),
            "validation_events": int((part.phase == "validation").sum()),
            "reserved_or_purge_events": int((part.phase == "reserved_or_purge").sum()),
            "first_fault_time": int(part.fault_time.min()) if len(part) else None,
            "last_fault_time": int(part.fault_time.max()) if len(part) else None,
            "low_interval_median_hours": gaps.median(), "low_interval_p75_hours": gaps.quantile(.75),
            "low_interval_p90_hours": gaps.quantile(.9)})
    pd.DataFrame(tool_rows).to_csv(OUT / "tool_event_summary.csv", index=False)
    history = []
    for minutes in HISTORY_MINUTES:
        for rule in ("A", "B"):
            column = f"{rule}_history_{minutes}m"
            for scope, part in [("all", events), ("train", events[events.phase == "train"]),
                                ("validation", events[events.phase == "validation"])]:
                usable = part[part[column]]
                history.append({"rule": rule, "history_minutes": minutes, "scope": scope,
                    "events": len(usable), "total_events": len(part),
                    "coverage_pct": 100 * len(usable) / len(part) if len(part) else None,
                    "tools": usable.tool_id.nunique(),
                    "max_gap_median_seconds": usable[f"{rule}_max_gap_{minutes}m_seconds"].median()})
    pd.DataFrame(history).to_csv(OUT / "history_coverage.csv", index=False)
    dist = []
    bins = []
    for phase, distribution in distributions.items():
        for weight in ("row", "event_equal"):
            dist.append(distribution.summary(phase, weight))
            bins.extend(distribution.bins(phase, weight))
    pd.DataFrame(dist).to_csv(OUT / "ttf_distribution.csv", index=False)
    pd.DataFrame(bins).to_csv(OUT / "ttf_bins.csv", index=False)
    anchor_summary = anchors.groupby(["rule", "history_minutes", "cadence_minutes", "phase"], as_index=False).agg(
        candidate_samples=("candidate_samples", "sum"), usable_events=("has_candidate", "sum"),
        median_samples_per_event=("candidate_samples", "median"),
        p75_samples_per_event=("candidate_samples", lambda x: x.quantile(.75)),
        p90_samples_per_event=("candidate_samples", lambda x: x.quantile(.9)),
        max_samples_per_event=("candidate_samples", "max"))
    anchor_summary.to_csv(OUT / "anchor_summary.csv", index=False)
    config = {"model_training": False, "fault_name": low.fault_name.iloc[0],
        "target": "next Low Pressure fault timestamp minus current observed sensor timestamp; high/leak do not change target",
        "split": str(PLAN.relative_to(ROOT)), "test_use": "no model; time column and full fault log used for descriptive all-event availability only; TTF/anchor distributions limited to Train/Validation",
        "history_rule": "at last pre-fault sensor observation, >=60 real timestamps spanning >=80% of requested wall-clock history; no imputation",
        "rule_A": "low-to-low cycle; intermediate other faults do not reset target or candidate history",
        "rule_B": "same target, candidate history begins strictly after most recent fault of any type",
        "anchor_rule": "latest actual observed timestamp at/before each regular wall-clock grid time, unique; no future observations",
        "ttf_distribution": "all observed raw sensor timestamps in train/validation low-to-low cycles; event equal weighting gives each Low event total weight 1; quantiles resolved to 1 minute",
        "units": "fault and sensor time in seconds, TTF reported in hours", "seed": 42}
    (OUT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print("events", events.phase.value_counts().to_dict(), flush=True)
    print(pd.DataFrame(dist).to_string(index=False), flush=True)
    print(pd.DataFrame(history).query('rule == "A" and scope == "all"').to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
