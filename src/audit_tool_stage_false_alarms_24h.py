"""Read-only tool/stage audit of the frozen 24h XGBoost alarm policy.

Normal sensor references use earlier training rows only. No model fitting,
prediction, feature persistence, or operating-policy selection occurs here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from analyze_false_alarms_24h import THRESHOLD, MIN_HITS, WINDOW
from evaluate_alarm_persistence_24h import causal_alarm
from evaluate_event_thresholds_24h import BASELINE, ROOT, SOURCE, event_metrics, load_events, read_json, row_and_alarm_metrics

OUT = ROOT / "output" / "experiments" / "tool_stage_false_alarm_audit_24h"
SENSORS = ("FLOWCOOLFLOWRATE", "FLOWCOOLPRESSURE")


def save(name: str, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "mean": None, "std": None, "p05": None, "p10": None, "p25": None,
                "median": None, "p75": None, "p90": None, "p95": None}
    q = np.quantile(values, [.05, .1, .25, .5, .75, .9, .95])
    return {"n": int(len(values)), "mean": float(np.mean(values)), "std": float(np.std(values)),
            **dict(zip(("p05", "p10", "p25", "median", "p75", "p90", "p95"), map(float, q)))}


def ks(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return float(ks_2samp(a, b, method="asymp").statistic) if len(a) and len(b) else None


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    sweep = read_json(SOURCE / "sweep_report.json")
    baseline = read_json(BASELINE / "reproduction_report.json")
    prep = read_json(BASELINE / "preparation_report.json")
    if hashlib.sha256((BASELINE / "xgb_24h_recovery.joblib").read_bytes()).hexdigest() != sweep["model_sha256"]:
        raise AssertionError("Frozen baseline model hash changed")
    if len(baseline["features"]) != 85:
        raise AssertionError("Baseline feature count changed")
    with np.load(SOURCE / "validation_probabilities.npz") as source:
        y, p, t, tool = (source[k].copy() for k in ("truth", "probability", "time", "tool_code"))
    if len(y) != 1142112 or int(y.sum()) != 67338:
        raise AssertionError("Validation population changed")
    code_to_name = {int(k): v for k, v in sweep["tool_code_map"].items()}
    name_to_code = {v: k for k, v in code_to_name.items()}
    alarm = causal_alarm(p >= THRESHOLD, t, tool, sorted(code_to_name), WINDOW, MIN_HITS)
    row, tools = row_and_alarm_metrics(.5, y, alarm.astype(np.uint8), t, tool,
                                       sweep["average_precision_fixed"], code_to_name)
    cutoffs = {item["tool"]: item["split_cutoff_time"] for item in prep["tools"]}
    events = load_events(cutoffs, set(code_to_name.values()))
    event_summary, event_details = event_metrics(.5, events, alarm.astype(np.uint8), t, tool, name_to_code)
    if (row["false_alarm_events_10m"], event_summary["detected_fault_events"], len(events)) != (6066, 138, 181):
        raise AssertionError("Policy or event counts changed")

    stages, tool_stages, normal_stats, validation_stats, fixed_values = [], [], [], [], []
    tool_rows, event_rows, special_rows = [], [], []
    feature_arrays: dict[str, dict[str, list[np.ndarray]]] = {
        s: {kind: {label: [] for label in ("fp", "tp")} for kind in
            ("raw", "minus_tool_median", "tool_normal_z", "tool_normal_percentile")}
        for s in SENSORS}
    stage_frames = []
    event_detail_map = {(e["tool_id"], e["fault_type"], int(e["fault_time"])): e for e in event_details}
    for code, name in sorted(code_to_name.items()):
        pos = np.flatnonzero(tool == code)
        tt, yy, pp, aa = t[pos], y[pos], p[pos], alarm[pos]
        frame = pd.read_parquet(BASELINE / "prepared" / f"{name}.parquet",
                                columns=["time", "label", "split", "stage", *SENSORS])
        valid = frame[(frame.split == 2) & (frame.label >= 0)]
        if not np.array_equal(valid.time.to_numpy(dtype=np.int64), tt) or not np.array_equal(valid.label.to_numpy(dtype=np.int8), yy):
            raise AssertionError(f"Feature/score row alignment changed: {name}")
        stage = valid.stage.to_numpy(dtype=np.int32)
        fp = aa & (yy == 0)
        tp = aa & (yy == 1)
        fp_idx = np.flatnonzero(fp)
        breaks = np.r_[0, np.flatnonzero(np.diff(tt[fp_idx]) > 600) + 1] if len(fp_idx) else np.empty(0, dtype=int)
        starts = fp_idx[breaks]
        start_mask = np.zeros(len(pos), dtype=bool)
        start_mask[starts] = True
        if len(starts) != tools[code]["false_alarm_events_10m"]:
            raise AssertionError(f"False episode count changed for {name}")
        stage_frames.append(pd.DataFrame({"stage": stage, "tool_id": name, "observed": 1,
                                          "fp_start": start_mask.astype(np.int8),
                                          "fp_row": fp.astype(np.int8), "tp_row": tp.astype(np.int8)}))
        tool_rows.append({"tool_id": name, "observed_validation_minute_rows": len(pos),
                          "observed_day_equivalents": len(pos)/1440,
                          "false_alarm_events_10m": len(starts),
                          "false_alarm_per_observed_day": len(starts)*1440/len(pos),
                          "fp_alert_rows": int(fp.sum()), "tp_alert_rows": int(tp.sum())})

        train_normal = frame[(frame.split == 0) & (frame.label == 0)]
        if not len(train_normal):
            raise AssertionError(f"No earlier normal reference for {name}")
        for sensor in SENSORS:
            reference = train_normal[sensor].to_numpy(dtype=np.float32)
            reference = reference[np.isfinite(reference)]
            ref_stat = stat(reference)
            normal_stats.append({"tool_id": name, "sensor": sensor, "reference": "earlier_train_normal_1m_mean", **ref_stat})
            val = valid[sensor].to_numpy(dtype=np.float32)
            for label, mask in (("validation_normal", yy == 0), ("FP_alert", fp), ("TP_alert", tp)):
                validation_stats.append({"tool_id": name, "sensor": sensor, "population": label, **stat(val[mask])})
            med, sd = ref_stat["median"], ref_stat["std"]
            sorted_ref = np.sort(reference)
            normal_percentile = np.searchsorted(sorted_ref, val, side="right") / len(sorted_ref)
            transforms = {"raw": val, "minus_tool_median": val - med,
                          "tool_normal_z": (val - ref_stat["mean"]) / sd if sd > 0 else np.full(len(val), np.nan),
                          "tool_normal_percentile": normal_percentile}
            for kind, transformed in transforms.items():
                feature_arrays[sensor][kind]["fp"].append(np.asarray(transformed[fp], dtype=np.float32))
                feature_arrays[sensor][kind]["tp"].append(np.asarray(transformed[tp], dtype=np.float32))
            if name == "03_M02":
                for label, mask in (("FP_alert", fp), ("TP_alert", tp)):
                    special_rows.append({"population": label, "measure": sensor, **stat(val[mask])})
                    special_rows.append({"population": label, "measure": sensor + "_normal_percentile", **stat(normal_percentile[mask])})

        if name == "03_M02":
            for label, mask in (("FP_alert", fp), ("TP_alert", tp)):
                special_rows.append({"population": label, "measure": "probability", **stat(pp[mask])})
                counts = pd.Series(stage[mask]).value_counts()
                for stage_id, count in counts.items():
                    special_rows.append({"population": label, "measure": f"stage={stage_id}", "n": int(count),
                                         "share_of_population": float(count / mask.sum())})

        # Fault stage is the last scored sensor row strictly before the event.
        # Detection can occur in an earlier, different stage; record both.
        for event in (e for e in events if e["tool_id"] == name):
            fault_time = int(event["fault_time"])
            right = int(np.searchsorted(tt, fault_time, side="left"))
            last = right - 1
            last_in_window = last >= 0 and tt[last] >= fault_time - 86400
            pre_stage = int(stage[last]) if last_in_window else None
            key = (name, event["fault_type"], fault_time)
            detail = event_detail_map[key]
            first_time = detail["first_alert_time"]
            first_stage = None
            if first_time is not None:
                first_idx = int(np.searchsorted(tt, first_time, side="left"))
                if first_idx < len(tt) and tt[first_idx] == first_time:
                    first_stage = int(stage[first_idx])
            event_rows.append({"tool_id": name, "fault_type": event["fault_type"],
                               "fault_time_dataset_second": fault_time,
                               "pre_fault_stage_last_scored_row_within_24h": pre_stage,
                               "first_detecting_alarm_stage": first_stage,
                               "evaluable": bool(detail["evaluable_from_validation_scores"]),
                               "detected": bool(detail["detected_before_fault"])})
        print(f"{name}: {len(pos)} rows, {len(starts)} false events", flush=True)

    all_stage = pd.concat(stage_frames, ignore_index=True)
    if len(all_stage) != len(y) or int(all_stage.fp_start.sum()) != 6066:
        raise AssertionError("Stage population does not reproduce validation")
    pre = pd.DataFrame(event_rows)
    per_tool_stage = all_stage.groupby(["tool_id", "stage"], sort=False).agg(
        observed_validation_minute_rows=("observed", "sum"), false_alarm_events_10m=("fp_start", "sum"),
        fp_alert_rows=("fp_row", "sum"), tp_alert_rows=("tp_row", "sum")).reset_index()
    event_agg = pre[pre.pre_fault_stage_last_scored_row_within_24h.notna()].groupby(
        ["tool_id", "pre_fault_stage_last_scored_row_within_24h"]).agg(
        fault_events_assigned=("detected", "size"), detected_fault_events=("detected", "sum")).reset_index()
    event_agg = event_agg.rename(columns={"pre_fault_stage_last_scored_row_within_24h": "stage"})
    per_tool_stage = per_tool_stage.merge(event_agg, on=["tool_id", "stage"], how="left")
    per_tool_stage[["fault_events_assigned", "detected_fault_events"]] = per_tool_stage[["fault_events_assigned", "detected_fault_events"]].fillna(0).astype(int)
    per_tool_stage["false_alarm_per_observed_day"] = per_tool_stage.false_alarm_events_10m * 1440 / per_tool_stage.observed_validation_minute_rows
    per_tool_stage["event_recall_assigned_stage"] = np.where(per_tool_stage.fault_events_assigned > 0,
        per_tool_stage.detected_fault_events / per_tool_stage.fault_events_assigned, np.nan)
    tool_stages = per_tool_stage.sort_values("false_alarm_events_10m", ascending=False).to_dict("records")
    global_stage = per_tool_stage.groupby("stage", as_index=False)[["observed_validation_minute_rows", "false_alarm_events_10m", "fp_alert_rows", "tp_alert_rows", "fault_events_assigned", "detected_fault_events"]].sum()
    global_stage["share_of_observed_rows"] = global_stage.observed_validation_minute_rows / len(y)
    global_stage["share_of_false_alarm_events"] = global_stage.false_alarm_events_10m / 6066
    global_stage["false_alarm_per_observed_day"] = global_stage.false_alarm_events_10m * 1440 / global_stage.observed_validation_minute_rows
    global_stage["event_recall_assigned_stage"] = np.where(global_stage.fault_events_assigned > 0,
        global_stage.detected_fault_events / global_stage.fault_events_assigned, np.nan)
    stages = global_stage.sort_values("false_alarm_events_10m", ascending=False).to_dict("records")
    stage1 = global_stage[global_stage.stage == 1].iloc[0]
    other_rows = len(y) - int(stage1.observed_validation_minute_rows)
    other_false = 6066 - int(stage1.false_alarm_events_10m)
    feature_results = []
    for sensor in SENSORS:
        for kind in feature_arrays[sensor]:
            fp = np.concatenate(feature_arrays[sensor][kind]["fp"])
            tp = np.concatenate(feature_arrays[sensor][kind]["tp"])
            feature_results.append({"sensor": sensor, "representation": kind,
                                    "fp_rows": len(fp), "tp_rows": len(tp),
                                    "pooled_fp_tp_ks": ks(fp, tp),
                                    "fp_median": stat(fp)["median"], "tp_median": stat(tp)["median"]})
    normal_df = pd.DataFrame(normal_stats)
    fixed = []
    for sensor in SENSORS:
        # Same numeric value, different position in each tool's earlier normal distribution.
        group = normal_df[normal_df.sensor == sensor]
        common_value = float(group["median"].median())
        for code, name in sorted(code_to_name.items()):
            frame = pd.read_parquet(BASELINE / "prepared" / f"{name}.parquet",
                                    columns=["label", "split", sensor])
            normal = frame.loc[(frame.split == 0) & (frame.label == 0), sensor].to_numpy(dtype=float)
            normal = normal[np.isfinite(normal)]
            fixed.append({"sensor": sensor, "common_raw_value": common_value, "tool_id": name,
                          "position_in_tool_earlier_normal_cdf": float(np.searchsorted(np.sort(normal), common_value, side="right") / len(normal))})
    OUT.mkdir(parents=True)
    save("stage_summary.csv", stages)
    save("tool_stage_summary.csv", tool_stages)
    save("tool_summary.csv", tool_rows)
    save("fault_event_stage_assignment.csv", event_rows)
    save("tool_normal_sensor_reference.csv", normal_stats)
    save("validation_sensor_distributions.csv", validation_stats)
    save("common_sensor_value_tool_percentile.csv", fixed)
    save("fp_tp_sensor_representation_comparison.csv", feature_results)
    save("03_M02_fp_tp_detail.csv", special_rows)
    if int(per_tool_stage.fault_events_assigned.sum()) != 163:
        raise AssertionError("Expected 163 evaluable events with a pre-fault sensor row")
    summary = {"model_sha256": sweep["model_sha256"], "model_retrained": False,
               "class_weight_changed": False, "threshold_changed": False, "feature_added": False,
               "threshold": THRESHOLD, "persistence": "2_of_5m", "validation_rows": len(y),
               "false_alarm_events_10m": 6066, "fault_events_total": 181,
               "fault_events_evaluable": 163, "fault_events_detected": 138,
               "stage1": stage1.to_dict(), "other_stages_false_alarm_per_observed_day": other_false*1440/other_rows,
               "stage1_vs_other_rate_ratio": float((stage1.false_alarm_events_10m/stage1.observed_validation_minute_rows)/(other_false/other_rows)),
               "top_tool_stage": tool_stages[0], "sensor_representation_comparison": feature_results,
               "sensor_unit": "Current sensor columns are one-minute means of original approximately 4-second measurements; no additional scaling in recovered baseline.",
               "normal_reference": "Per-tool train-split (split=0) label=0 rows only, earlier than validation. It is a retrospective known-normal reference for audit, never a model input.",
               "fault_stage_definition": "Stage on final scored sensor row in [fault_time-24h, fault_time); 18 events lack such a row and have no assigned stage. Detection may have happened in another stage.",
               "false_event_stage_definition": "Stage on first false-positive row of a per-tool 10-minute-merged episode; the episode can span stages.",
               "denominator_caveat": "Observed-day = labeled validation one-minute rows/1440, not wall-clock running days.",
               "interpretation_caveat": "Pooled FP/TP differences may reflect tool mix or stage mix. Per-tool median shifts and z/percentile transforms are monotonic within each tool, so within-tool KS cannot improve by construction; pooled KS is exploratory and not model performance.",
               "raw_value_caveat": "A common numerical sensor value's normal percentile differs across tools, but physical meaning or causal fault risk cannot be established from this comparison alone."}
    # pandas uses NaN for undefined event recall; JSON reports use explicit null.
    def json_safe(value):
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value

    (OUT / "audit_report.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"stage1_share_observed": summary["stage1"]["share_of_observed_rows"],
                      "stage1_share_false": summary["stage1"]["share_of_false_alarm_events"],
                      "stage1_rate_ratio": summary["stage1_vs_other_rate_ratio"],
                      "top_tool_stage": summary["top_tool_stage"],
                      "sensor_representations": feature_results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
