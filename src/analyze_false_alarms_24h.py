"""Audit frozen 24h XGBoost false alarms using real validation features.

No fitting, feature generation, score recomputation, or policy tuning occurs.
False events use the existing 10-minute merge of normal-labeled alert rows.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from evaluate_alarm_persistence_24h import causal_alarm
from evaluate_event_thresholds_24h import (
    BASELINE, FAULT_LOG, ROOT, SOURCE, event_metrics, load_events,
    read_json, row_and_alarm_metrics,
)

OUT = ROOT / "output" / "experiments" / "false_alarm_analysis_24h"
THRESHOLD = 0.36
HOURS = 24
MIN_HITS, WINDOW = 2, 5
FAR_HOURS = 72  # Descriptive audit only, not a new alarm policy.
RATE_DENOMINATOR_MINUTES = 1440


def csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def prob_summary(name: str, values: np.ndarray) -> dict:
    p = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(p, [0, .1, .25, .5, .75, .9, 1]) if len(p) else [None] * 7
    return {"group": name, "rows_or_events": len(p), "mean_probability": float(p.mean()) if len(p) else None,
            **{label: float(value) if value is not None else None for label, value in zip(
                ("min", "p10", "p25", "median", "p75", "p90", "max"), quantiles)},
            "fraction_current_probability_below_0_36": float((p < THRESHOLD).mean()) if len(p) else None}


def nearest_faults(times: np.ndarray, faults: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    previous = np.full(len(times), np.inf, dtype=float)
    following = previous.copy()
    if len(faults):
        ix = np.searchsorted(faults, times, side="right")
        has_previous = ix > 0
        has_following = ix < len(faults)
        previous[has_previous] = (times[has_previous] - faults[ix[has_previous] - 1]) / 3600
        following[has_following] = (faults[ix[has_following]] - times[has_following]) / 3600
    return previous, following


def distance_band(hours: float) -> str:
    if not np.isfinite(hours):
        return "no_fault_on_that_side"
    if hours < 24:
        return "under_24h"
    if hours < 72:
        return "24_to_72h"
    if hours < 168:
        return "72h_to_7d"
    return "7d_or_more"


def _state_rows(tool_id: str, column: str, values: np.ndarray, alarm: np.ndarray,
                truth: np.ndarray, event_start: np.ndarray) -> list[dict]:
    series = pd.Series(values)
    if column == "time_diff_seconds":
        series = pd.cut(series, [-np.inf, 60, 120, 600, np.inf], labels=("0-60s", "61-120s", "121-600s", ">600s"))
    elif column == "history_minutes":
        series = pd.cut(series, [-np.inf, 1, 5, 30, 120, np.inf], labels=("first", "2-5", "6-30", "31-120", ">120"))
    grouped = pd.DataFrame({"state": series.astype(str), "alarm": alarm,
                            "fp": alarm & (truth == 0), "tp": alarm & (truth == 1),
                            "false_event_start": event_start}).groupby("state", dropna=False, sort=False)
    rows = []
    for state, frame in grouped:
        observed = len(frame)
        fp = int(frame.fp.sum())
        episodes = int(frame.false_event_start.sum())
        rows.append({"tool_id": tool_id, "state_feature": column, "state": state,
                     "observed_validation_minutes": observed, "fp_alarm_rows": fp,
                     "tp_alarm_rows": int(frame.tp.sum()), "false_event_starts": episodes,
                     "fp_alarm_rows_per_observed_minute": fp / observed,
                     "false_event_starts_per_observed_day_equivalent": episodes / (observed / 1440)})
    return rows


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    report = read_json(BASELINE / "reproduction_report.json")
    sweep = read_json(SOURCE / "sweep_report.json")
    prior = read_json(ROOT / "output" / "experiments" / "threshold_persistence_grid_24h" / "threshold_persistence_grid_report.json")
    model_file = BASELINE / "xgb_24h_recovery.joblib"
    if hashlib.sha256(model_file.read_bytes()).hexdigest() != sweep["model_sha256"]:
        raise AssertionError("Frozen baseline model changed")
    features = report["features"]
    if len(features) != 85:
        raise AssertionError("Frozen 85-feature schema changed")
    with np.load(SOURCE / "validation_probabilities.npz") as saved:
        truth, probability, time, tool = (saved[k].copy() for k in ("truth", "probability", "time", "tool_code"))
    if len(truth) != 1142112 or int(truth.sum()) != 67338:
        raise AssertionError("Frozen validation population changed")
    tool_map = {int(k): v for k, v in sweep["tool_code_map"].items()}
    reverse = {v: k for k, v in tool_map.items()}
    alarm = causal_alarm(probability >= THRESHOLD, time, tool, sorted(tool_map), WINDOW, MIN_HITS)
    binary = alarm.astype(np.uint8)
    row_metrics, by_tool = row_and_alarm_metrics(.5, truth, binary, time, tool,
                                                 sweep["average_precision_fixed"], tool_map)
    cutoffs = {r["tool"]: r["split_cutoff_time"] for r in read_json(BASELINE / "preparation_report.json")["tools"]}
    events = load_events(cutoffs, set(tool_map.values()))
    event_metrics_result, event_details = event_metrics(.5, events, binary, time, tool, reverse)
    reference = next(x for x in prior["lowest_false_alarm_qualifying"]
                     if x["threshold"] == THRESHOLD and x["condition"] == "2_of_5m")
    if (row_metrics["false_alarm_events_10m"], row_metrics["fp"], row_metrics["tp"],
        event_metrics_result["detected_fault_events"]) != (6066, 267460, 37311, 138):
        raise AssertionError("Fixed policy counts differ from established experiment")
    if len(events) != 181 or reference["false_alarm_events_10m"] != 6066:
        raise AssertionError("Fault event reference changed")
    OUT.mkdir(parents=True)
    faults = pd.read_csv(FAULT_LOG, usecols=["tool_id", "time"]).drop_duplicates()
    feature_by_class: dict[str, dict[str, list[np.ndarray]]] = {name: {"fp": [], "tp": []} for name in features}
    feature_by_tool = []
    state_rows = []
    false_event_rows = []
    tool_rows = []
    relative_phase = np.zeros((24, 3), dtype=np.int64)  # observed, false event starts, FP rows
    progress = np.zeros((4, 3), dtype=np.int64)
    fp_event_probability = []
    for code, name in sorted(tool_map.items()):
        positions = np.flatnonzero(tool == code)
        t, y, p, a = time[positions], truth[positions], probability[positions], alarm[positions]
        fp_local = np.flatnonzero(a & (y == 0))
        tp_local = np.flatnonzero(a & (y == 1))
        false_t = t[fp_local]
        episode_breaks = np.r_[0, np.flatnonzero(np.diff(false_t) > 600) + 1] if len(false_t) else np.empty(0, dtype=int)
        episode_stops = np.r_[episode_breaks[1:], len(false_t)] if len(false_t) else np.empty(0, dtype=int)
        starts_local = fp_local[episode_breaks]
        start_mask = np.zeros(len(t), dtype=bool)
        start_mask[starts_local] = True
        if len(starts_local) != by_tool[code]["false_alarm_events_10m"]:
            raise AssertionError(f"False alarm episode count mismatch for {name}")
        last_fault, next_fault = nearest_faults(t[starts_local],
            np.sort(faults.loc[faults.tool_id == name, "time"].unique().astype(np.int64)))
        nearest = np.minimum(last_fault, next_fault)
        for episode_id, (begin, stop) in enumerate(zip(episode_breaks, episode_stops), 1):
            rows = fp_local[begin:stop]
            start = int(rows[0])
            false_event_rows.append({"tool_id": name, "episode_id_within_tool": episode_id,
                "start_dataset_second": int(t[start]), "end_dataset_second": int(t[rows[-1]]),
                "false_alarm_rows": len(rows), "start_probability": float(p[start]),
                "maximum_probability_in_episode": float(np.max(p[rows])),
                "hours_since_previous_fault": float(last_fault[episode_id-1]) if np.isfinite(last_fault[episode_id-1]) else None,
                "hours_until_next_fault": float(next_fault[episode_id-1]) if np.isfinite(next_fault[episode_id-1]) else None,
                "hours_to_nearest_fault": float(nearest[episode_id-1]) if np.isfinite(nearest[episode_id-1]) else None,
                "nearest_fault_band": distance_band(nearest[episode_id-1]),
                "at_least_72h_from_any_logged_fault": bool(nearest[episode_id-1] >= FAR_HOURS)})
        fp_event_probability.append(p[starts_local])
        for h in range(24):
            phase = ((t % 86400) // 3600) == h
            relative_phase[h] += [int(phase.sum()), int((phase & start_mask).sum()), int((phase & a & (y == 0)).sum())]
        # Equal *ordered observed-row* quartiles within each tool, not wall-clock quarters.
        quarters = np.minimum(np.arange(len(t)) * 4 // len(t), 3)
        for q in range(4):
            selected = quarters == q
            progress[q] += [int(selected.sum()), int((selected & start_mask).sum()), int((selected & a & (y == 0)).sum())]
        path = BASELINE / "prepared" / f"{name}.parquet"
        frame = pd.read_parquet(path, columns=[*features, "time", "label", "split"])
        valid = frame.loc[(frame.split == 2) & (frame.label >= 0)]
        if not np.array_equal(valid.time.to_numpy(dtype=np.int64), t) or not np.array_equal(valid.label.to_numpy(dtype=np.int8), y):
            raise AssertionError(f"Feature rows no longer align with scores: {name}")
        fp_values = valid.iloc[fp_local][features]
        tp_values = valid.iloc[tp_local][features]
        for feature in features:
            f = fp_values[feature].to_numpy(dtype=np.float32, copy=True)
            v = tp_values[feature].to_numpy(dtype=np.float32, copy=True)
            feature_by_class[feature]["fp"].append(f)
            feature_by_class[feature]["tp"].append(v)
            ff, vv = f[np.isfinite(f)], v[np.isfinite(v)]
            if len(ff) >= 20 and len(vv) >= 20:
                iqr = np.subtract(*np.percentile(np.r_[ff, vv], [75, 25]))
                median_diff = float((np.median(vv) - np.median(ff)) / iqr) if iqr > 0 else None
                feature_by_tool.append({"tool_id": name, "feature": feature,
                    "fp_rows": len(ff), "tp_rows": len(vv),
                    "fp_median": float(np.median(ff)), "tp_median": float(np.median(vv)),
                    "tp_minus_fp_median_over_pooled_iqr": median_diff})
        for column in ("stage", "recipe", "recipe_step", "restart_after_stop", "time_diff_seconds", "history_minutes"):
            state_rows.extend(_state_rows(name, column, valid[column].to_numpy(), a, y, start_mask))
        observed_days = len(t) / RATE_DENOMINATOR_MINUTES
        tool_rows.append({"tool_id": name, "observed_validation_minute_rows": len(t),
            "observed_day_equivalents": observed_days, "false_alarm_events_10m": len(starts_local),
            "share_of_all_6066_false_alarm_events": len(starts_local) / 6066,
            "false_alarm_events_per_observed_day": len(starts_local) / observed_days,
            "false_alarm_events_per_observed_12h": len(starts_local) / (observed_days * 2),
            "fp_alarm_rows": len(fp_local), "tp_alarm_rows": len(tp_local),
            "alerted_row_precision": len(tp_local) / (len(tp_local) + len(fp_local)) if len(tp_local) + len(fp_local) else None})
        print(f"{name}: {len(starts_local)} false episodes; FP/TP rows {len(fp_local)}/{len(tp_local)}", flush=True)
    tool_rows.sort(key=lambda r: r["false_alarm_events_10m"], reverse=True)
    top5 = tool_rows[:5]
    top5_count = sum(r["false_alarm_events_10m"] for r in top5)
    if sum(r["false_alarm_events_10m"] for r in tool_rows) != 6066 or len(false_event_rows) != 6066:
        raise AssertionError("False alarm episode population changed")
    csv(OUT / "false_alarm_by_tool.csv", tool_rows)
    csv(OUT / "false_alarm_episodes.csv", false_event_rows)
    csv(OUT / "process_state_distribution_by_tool.csv", state_rows)
    hourly = [{"dataset_relative_phase_hour": hour,
        "observed_validation_minute_rows": int(relative_phase[hour, 0]),
        "false_alarm_episode_starts": int(relative_phase[hour, 1]),
        "fp_alarm_rows": int(relative_phase[hour, 2]),
        "false_alarm_starts_per_observed_day_equivalent": float(relative_phase[hour, 1] / (relative_phase[hour, 0] / 1440))}
        for hour in range(24)]
    csv(OUT / "dataset_relative_hour_distribution.csv", hourly)
    csv(OUT / "validation_progress_quartiles.csv", [{"observed_row_quartile_per_tool": q+1,
        "observed_validation_minute_rows": int(progress[q, 0]),
        "false_alarm_episode_starts": int(progress[q, 1]),
        "fp_alarm_rows": int(progress[q, 2]),
        "false_alarm_starts_per_observed_day_equivalent": float(progress[q, 1] / (progress[q, 0] / 1440))}
        for q in range(4)])
    bands = pd.DataFrame(false_event_rows).groupby("nearest_fault_band", sort=False).size()
    csv(OUT / "distance_to_fault_distribution.csv", [{"nearest_fault_band": band,
        "false_alarm_events_10m": int(count), "share_of_false_alarm_events": int(count)/6066}
        for band, count in bands.items()])
    fp_rows = probability[alarm & (truth == 0)]
    tp_rows = probability[alarm & (truth == 1)]
    fp_starts = np.concatenate(fp_event_probability)
    first_tp = []
    for detail in event_details:
        if detail["detected_before_fault"]:
            code = reverse[detail["tool_id"]]
            positions = np.flatnonzero(tool == code)
            event_time = int(detail["first_alert_time"])
            index = np.searchsorted(time[positions], event_time)
            first_tp.append(float(probability[positions[index]]))
    prob_rows = [prob_summary("FP_alert_rows", fp_rows), prob_summary("TP_alert_rows", tp_rows),
                 prob_summary("FP_episode_starts", fp_starts),
                 prob_summary("first_alert_for_detected_fault_events", np.array(first_tp))]
    csv(OUT / "probability_summary.csv", prob_rows)
    edges = np.linspace(0, 1, 21)
    csv(OUT / "probability_histogram.csv", [{"bin_left": float(edges[i]), "bin_right": float(edges[i+1]),
        "fp_alert_rows": int(np.histogram(fp_rows, edges)[0][i]),
        "tp_alert_rows": int(np.histogram(tp_rows, edges)[0][i]),
        "fp_episode_starts": int(np.histogram(fp_starts, edges)[0][i])} for i in range(20)])
    feature_rows = []
    for feature in features:
        f = np.concatenate(feature_by_class[feature]["fp"])
        t = np.concatenate(feature_by_class[feature]["tp"])
        ff, tt = f[np.isfinite(f)], t[np.isfinite(t)]
        if len(ff) and len(tt):
            ks = ks_2samp(ff, tt, method="asymp").statistic
            fp_q = np.quantile(ff, [.1, .5, .9])
            tp_q = np.quantile(tt, [.1, .5, .9])
        else:
            ks, fp_q, tp_q = None, [None]*3, [None]*3
        within = [r for r in feature_by_tool if r["feature"] == feature and r["tp_minus_fp_median_over_pooled_iqr"] is not None]
        differences = np.array([r["tp_minus_fp_median_over_pooled_iqr"] for r in within])
        feature_rows.append({"feature": feature, "fp_rows": len(f), "tp_rows": len(t),
            "fp_missing_fraction": float(1-len(ff)/len(f)), "tp_missing_fraction": float(1-len(tt)/len(t)),
            "fp_p10": fp_q[0], "fp_median": fp_q[1], "fp_p90": fp_q[2],
            "tp_p10": tp_q[0], "tp_median": tp_q[1], "tp_p90": tp_q[2],
            "ks_distance_all_tools": ks, "tools_with_both_classes_and_variable_feature": len(within),
            "median_within_tool_tp_minus_fp_over_iqr": float(np.median(differences)) if len(differences) else None,
            "fraction_tools_tp_median_above_fp": float((differences > 0).mean()) if len(differences) else None})
    feature_rows.sort(key=lambda r: r["ks_distance_all_tools"] or -1, reverse=True)
    csv(OUT / "feature_fp_tp_distribution.csv", feature_rows)
    csv(OUT / "feature_fp_tp_by_tool.csv", feature_by_tool)
    sensor_names = set(features) - {"stage", "recipe", "recipe_step", "tool_number", "time_diff_seconds",
        "restart_after_stop", "session_age_minutes", "history_minutes", "stage_changed", "recipe_changed", "recipe_step_changed"}
    top_sensors = [r for r in feature_rows if r["feature"] in sensor_names][:10]
    distant = sum(r["at_least_72h_from_any_logged_fault"] for r in false_event_rows)
    summary = {"source_score": str(SOURCE / "validation_probabilities.npz"),
        "source_score_sha256": hashlib.sha256((SOURCE / "validation_probabilities.npz").read_bytes()).hexdigest(),
        "source_model_sha256": sweep["model_sha256"], "retrained": False,
        "features_changed": False, "threshold_changed": False,
        "threshold": THRESHOLD, "persistence": "2 hits in current/previous 4 clock-minute buckets; stop gap >120s resets",
        "false_event_definition": "Per-tool negative-label alarm rows merged while consecutive false alarms are <=600 seconds apart; same as established baseline.",
        "validation_rows": len(truth), "fp_alarm_rows": row_metrics["fp"], "tp_alarm_rows": row_metrics["tp"],
        "fault_events_total": 181, "fault_events_detected": event_metrics_result["detected_fault_events"],
        "false_alarm_events_10m": row_metrics["false_alarm_events_10m"],
        "top_false_alarm_tool": top5[0], "top_five_tools": [r["tool_id"] for r in top5],
        "top_five_false_alarm_events": top5_count, "top_five_share_of_false_alarm_events": top5_count/6066,
        "top_five_share_of_observed_minutes": sum(r["observed_validation_minute_rows"] for r in top5)/len(truth),
        "false_alarm_events_at_least_72h_from_any_logged_fault": distant,
        "share_false_alarm_events_at_least_72h_from_any_logged_fault": distant/6066,
        "fp_episode_start_probability": prob_rows[2], "fp_alert_row_probability": prob_rows[0],
        "tp_alert_row_probability": prob_rows[1], "top_feature_distributions_all_tools": feature_rows[:10],
        "top_sensor_derived_feature_distributions_all_tools": top_sensors,
        "time_caveat": "Dataset time is relative seconds with no verified calendar/time-zone origin; modulo-24h phase is not actual local hour or staff shift.",
        "causality_caveat": "Observed FP/TP feature differences are associations, not sensor causes; repeated adjacent minute rows and tool mix can confound row-level statistics.",
        "distance_caveat": "Distance uses any logged fault of the three target types, including train/validation; >=72h is a descriptive definition, not a new threshold.",
        "denominator_caveat": "Observed-day rate uses labeled validation minute rows/1440, not wall-clock duty days."}
    (OUT / "analysis_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value)), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("false_alarm_events_10m", "top_false_alarm_tool",
        "top_five_share_of_false_alarm_events", "false_alarm_events_at_least_72h_from_any_logged_fault")},
        ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
