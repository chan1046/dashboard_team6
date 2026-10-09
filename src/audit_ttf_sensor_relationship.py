"""Descriptive, event-paired TTF/sensor audit using actual pre-test minute data.

No fitting or prediction. TTF is constructed solely as fault_time - observation_time
and is never passed into a feature calculation. Time windows use elapsed seconds,
not row counts, and restart at each recorded operating session.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from recover_xgb24_baseline import SENSORS, ROLLING
from evaluate_event_thresholds_24h import write_csv
from experiment_tool_stage_interaction_24h import file_hash
from forward_validation_ab_24h import validation_events

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "output/baseline_recovery/prepared"
PLAN = ROOT / "output/experiments/forward_validation_ab_24h/split_plan.json"
OUT = ROOT / "output/experiments/ttf_sensor_relationship"
BIN_EDGES_H = [0, 6, 12, 24, 48, 72, 168]
BIN_NAMES = ["0-6h", "6-12h", "12-24h", "24-48h", "48-72h", ">72h_to_7d"]
MAX_H = 168
WINDOWS = (600, 1800)


def causal_features(t: np.ndarray, session: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray]:
    """Past-only real-time differences and moments, separately per session."""
    n = len(t)
    out = {f"diff_{m}m": np.full(n, np.nan, np.float32) for m in (1, 5, 10)}
    for m in (10, 30):
        for stat in ("mean", "std", "slope"):
            out[f"{stat}_{m}m"] = np.full(n, np.nan, np.float32)
    valid = np.isfinite(x)
    boundaries = np.r_[0, np.flatnonzero(np.diff(session) != 0)+1, n]
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if b-a < 2:
            continue
        tt = t[a:b].astype(np.float64)
        xx = np.nan_to_num(x[a:b].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        vv = valid[a:b]
        ix = np.arange(b-a)
        for m in (1, 5, 10):
            target = tt-m*60
            prior = np.searchsorted(tt, target, side="right")-1
            good = (prior >= 0) & vv & vv[np.maximum(prior, 0)]
            delta = np.full(b-a, np.nan, np.float32)
            delta[good] = (xx[good]-xx[prior[good]]).astype(np.float32)
            out[f"diff_{m}m"][a:b] = delta
        rel = tt-tt[0]
        vectors = [vv.astype(np.float64), xx, xx*xx, rel*vv, rel*rel*vv, rel*xx]
        cumulative = [np.r_[0.0, np.cumsum(v)] for v in vectors]
        for m in (10, 30):
            left = np.searchsorted(tt, tt-m*60, side="left")
            count, sx, sx2, st, st2, stx = [c[ix+1]-c[left] for c in cumulative]
            ok = count >= 2
            mean = np.divide(sx, count, out=np.full(b-a, np.nan), where=ok)
            var = np.divide(sx2, count, out=np.full(b-a, np.nan), where=ok)-mean*mean
            denom = st2-st*st/np.maximum(count, 1)
            slope = np.divide(stx-st*sx/np.maximum(count, 1), denom,
                              out=np.full(b-a, np.nan), where=ok & (denom > 0))*3600
            out[f"mean_{m}m"][a:b] = mean.astype(np.float32)
            out[f"std_{m}m"][a:b] = np.sqrt(np.maximum(var, 0)).astype(np.float32)
            out[f"slope_{m}m"][a:b] = slope.astype(np.float32)
    return out


def stats(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "mean": None, "median": None, "std": None,
                "q10": None, "q25": None, "q75": None, "q90": None}
    q = np.quantile(x, [.1, .25, .5, .75, .9])
    return {"n": len(x), "mean": float(np.mean(x)), "median": float(q[2]),
            "std": float(np.std(x)), "q10": float(q[0]), "q25": float(q[1]),
            "q75": float(q[3]), "q90": float(q[4])}


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    events = validation_events(plan)
    by_tool = pd.DataFrame(events).groupby("tool_id")
    descriptions = []
    paired = []
    coverage = []
    normal = []
    for item in plan["tools"]:
        tool = item["tool_id"]
        if tool not in by_tool.groups:
            print(f"{tool}: no validation fault events; excluded from event analysis", flush=True)
            continue
        path = PREP / f"{tool}.parquet"
        frame = pq.read_table(path, columns=["time", "session_id", "label", *SENSORS]).to_pandas()
        t = frame.time.to_numpy(dtype=np.int64)
        if np.any(np.diff(t) < 0):
            raise AssertionError(f"Unsorted observations: {tool}")
        train_normal = (t < item["train_end_exclusive_second"]) & (frame.label.to_numpy()==0)
        refs = {}
        for sensor in SENSORS:
            values = frame[sensor].to_numpy(dtype=np.float64)
            reference = values[train_normal & np.isfinite(values)]
            med, q25, q75 = np.quantile(reference, [.5, .25, .75])
            iqr = q75-q25
            refs[sensor] = (med, iqr)
            normal.append({"tool_id": tool, "sensor": sensor, "train_normal_n": len(reference),
                           "median": float(med), "iqr": float(iqr), "mean": float(reference.mean()),
                           "std": float(reference.std())})
        # Keep a causal 30-minute lookback but avoid processing the irrelevant
        # earlier training history for event features.
        frame = frame.loc[frame.time >= item["validation_start_second"]-1800].reset_index(drop=True)
        t = frame.time.to_numpy(dtype=np.int64)
        # Causal features are calculated once per tool, BEFORE event selection/TTF is introduced.
        features = {sensor: frame[sensor].to_numpy(dtype=np.float32) for sensor in SENSORS}
        session = frame.session_id.to_numpy()
        for sensor in ROLLING:
            raw = features[sensor]
            for key, value in causal_features(t, session, raw).items():
                features[f"{sensor}__{key}"] = value
            med, iqr = refs[sensor]
            features[f"{sensor}__train_normal_robust_z"] = (raw-med)/(iqr/1.349) if iqr>0 else np.full(len(raw), np.nan)
        for event in by_tool.get_group(tool).itertuples():
            fault_t = int(event.fault_time)
            if not (item["validation_start_second"] <= fault_t < item["test_start_second"]):
                raise AssertionError("Event outside validation")
            left = np.searchsorted(t, max(fault_t-MAX_H*3600, item["validation_start_second"]), side="left")
            right = np.searchsorted(t, fault_t, side="left")
            et = t[left:right]
            delta = (fault_t-et)/3600.0
            if len(delta) and (np.any(delta<=0) or np.any(delta>MAX_H)):
                raise AssertionError("TTF event alignment failed")
            bins = np.searchsorted(BIN_EDGES_H, delta, side="left")-1
            # 6h exactly belongs to the nearer bin; bins above 72h are capped at 7d.
            bins = np.clip(bins, 0, len(BIN_NAMES)-1)
            eid = f"{tool}|{event.fault_type}|{fault_t}"
            per_bin_medians = {}
            for bi, label in enumerate(BIN_NAMES):
                mask = bins==bi
                coverage.append({"event_id": eid, "tool_id": tool, "fault_type": event.fault_type,
                    "fault_time": fault_t, "ttf_bin": label, "observed_rows": int(mask.sum())})
                if not np.any(mask):
                    continue
                per_bin_medians[label] = {}
                for feature, array in features.items():
                    result = stats(np.asarray(array[left:right][mask], dtype=np.float64))
                    if result["n"]:
                        descriptions.append({"event_id": eid, "tool_id": tool,
                            "fault_type": event.fault_type, "fault_time": fault_t,
                            "ttf_bin": label, "feature": feature, **result})
                        per_bin_medians[label][feature] = result["median"]
            far = per_bin_medians.get("48-72h", {})
            near = per_bin_medians.get("0-6h", {})
            for feature in set(far) & set(near):
                paired.append({"event_id": eid, "tool_id": tool, "fault_type": event.fault_type,
                    "feature": feature, "near_median": near[feature], "far_median": far[feature],
                    "near_minus_far": near[feature]-far[feature]})
        print(f"{tool}: {len(frame):,} minute rows, {len(by_tool.get_group(tool))} validation events", flush=True)
    desc, pairs = pd.DataFrame(descriptions), pd.DataFrame(paired)
    # Event-balanced summaries; sampled minutes do not dominate events with fewer observations.
    agg = (desc.groupby(["fault_type", "ttf_bin", "feature"], observed=True)
        .agg(events=("event_id", "nunique"), median_of_event_medians=("median", "median"),
             q25_event_median=("median", lambda x: x.quantile(.25)),
             q75_event_median=("median", lambda x: x.quantile(.75)),
             median_within_event_std=("std", "median")).reset_index())
    pattern = (pairs.assign(direction=np.sign(pairs.near_minus_far))
        .groupby(["fault_type", "feature"], observed=True)
        .agg(paired_events=("event_id", "nunique"), median_near_minus_far=("near_minus_far", "median"),
             increases=("direction", lambda x: int((x>0).sum())),
             decreases=("direction", lambda x: int((x<0).sum())),
             unchanged=("direction", lambda x: int((x==0).sum()))).reset_index())
    pattern["direction_consistency"] = pattern[["increases", "decreases"]].max(axis=1)/pattern.paired_events
    tool_patterns = (pairs.groupby(["tool_id", "fault_type", "feature"], observed=True)
        .agg(events=("event_id", "nunique"), median_near_minus_far=("near_minus_far", "median"),
             increases=("near_minus_far", lambda x: int((x>0).sum())),
             decreases=("near_minus_far", lambda x: int((x<0).sum()))).reset_index())
    OUT.mkdir(parents=True)
    for name, data in (("event_bin_feature_stats", desc), ("event_bin_coverage", pd.DataFrame(coverage)),
                       ("event_balanced_bin_summary", agg), ("paired_event_changes", pairs),
                       ("fault_feature_direction", pattern), ("tool_fault_feature_direction", tool_patterns),
                       ("train_only_tool_normal_reference", pd.DataFrame(normal))):
        data.to_csv(OUT/f"{name}.csv", index=False, encoding="utf-8-sig")
    report = {"source": "Existing actual 1-minute prepared sensor observations; fault log defines TTF only as fault_time minus earlier observation time",
        "input_fault_events": len(events), "paired_events_with_both_0_6h_and_48_72h": int(pairs.event_id.nunique()),
        "sensor_count_raw": len(SENSORS), "derived_sensor_count": len(ROLLING),
        "ttf_bins_hours": BIN_NAMES, "far_bin_cap_hours": MAX_H,
        "feature_causality": "All differences, 10/30-minute moments and slopes use current or prior observations within the same existing operating session; no gap filling. Windows use elapsed seconds.",
        "analysis_population": "Forward Model A validation only; no reserved test rows or outcomes inspected",
        "normal_reference": "Per-tool label=0 rows from earlier first-stage Train only; validation never estimates medians/IQR",
        "caveats": ["Descriptive association does not prove regressability or future generalization.",
                    "1-minute aggregation can hide 4-second transients.",
                    "The >72h group is 72h to 7 days, not unlimited time.",
                    "Event windows may overlap when faults are close; paired event patterns are not statistically independent.",
                    "TTF and fault times are targets only, never model or sensor features."],
        "plan_sha256": file_hash(PLAN)}
    (OUT/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"events": len(events), "paired": int(pairs.event_id.nunique()),
                      "feature_rows": len(desc)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
