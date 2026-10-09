"""A/B/C audit of two tool-normal relative sensors on the frozen 24h split.

Run prepare, train_b, train_c, evaluate in order. Only train label=0 rows
define per-tool reference statistics. No validation reference fitting occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from evaluate_alarm_persistence_24h import causal_alarm
from evaluate_event_thresholds_24h import BASELINE, ROOT, SOURCE, event_metrics, load_events, read_json, row_and_alarm_metrics, write_csv
from evaluate_threshold_persistence_grid_24h import CONDITIONS, THRESHOLDS
from experiment_tool_stage_interaction_24h import file_hash, stage_false_episodes

OUT = ROOT / "output" / "experiments" / "tool_relative_features_24h"
MODELS = ROOT / "output" / "models"
BASE_FEATURES = read_json(BASELINE / "reproduction_report.json")["features"]
EXTRA = ("FLOWCOOLFLOWRATE_tool_percentile", "FLOWCOOLPRESSURE_tool_robust_zscore",
         "FLOWCOOLFLOWRATE_minus_tool_normal_median", "FLOWCOOLPRESSURE_minus_tool_normal_median")
FEATURES = {"B": [*BASE_FEATURES, *EXTRA[:2]], "C": [*BASE_FEATURES, *EXTRA]}
SENSORS = ("FLOWCOOLFLOWRATE", "FLOWCOOLPRESSURE")


def paths_and_checks():
    baseline = read_json(BASELINE / "reproduction_report.json")
    sweep = read_json(SOURCE / "sweep_report.json")
    prep = read_json(BASELINE / "preparation_report.json")
    paths = sorted((BASELINE / "prepared").glob("??_M??.parquet"))
    if len(paths) != 20 or len(BASE_FEATURES) != 85 or baseline["features"] != BASE_FEATURES:
        raise AssertionError("Baseline schema or tool count changed")
    if file_hash(BASELINE / "xgb_24h_recovery.joblib") != sweep["model_sha256"]:
        raise AssertionError("Baseline model hash changed")
    return baseline, sweep, prep, paths


def reference_file():
    return OUT / "train_normal_references.joblib"


def model_file(kind):
    return MODELS / f"xgb_24h_tool_relative_{kind.lower()}.joblib"


def score_file(kind):
    return OUT / f"validation_probabilities_{kind.lower()}.npz"


def prepare():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    _, _, _, paths = paths_and_checks()
    reference, audit = {}, []
    for path in paths:
        frame = pd.read_parquet(path, columns=["split", "label", *SENSORS])
        normal = frame.loc[(frame.split == 0) & (frame.label == 0)]
        if len(normal) < 100:
            raise AssertionError(f"Insufficient train-normal reference: {path.stem}")
        reference[path.stem] = {}
        for sensor in SENSORS:
            values = normal[sensor].to_numpy(dtype=np.float32)
            values = values[np.isfinite(values)]
            if len(values) != len(normal):
                raise AssertionError(f"Missing train normal sensor: {path.stem}, {sensor}")
            quartiles = np.quantile(values, [.25, .5, .75])
            iqr = float(quartiles[2] - quartiles[0])
            if iqr <= 0:
                raise AssertionError(f"Nonpositive train normal IQR: {path.stem}, {sensor}")
            values.sort()
            reference[path.stem][sensor] = {"sorted": values, "median": float(quartiles[1]),
                                            "iqr": iqr}
            audit.append({"tool_id": path.stem, "sensor": sensor, "source_rows": len(frame),
                          "train_label0_reference_rows": len(values), "q25": float(quartiles[0]),
                          "median": float(quartiles[1]), "q75": float(quartiles[2]),
                          "iqr": iqr, "reference_split": "train_only_split0_label0"})
        print(f"reference {path.stem}: {len(normal):,} earlier normal rows", flush=True)
    OUT.mkdir(parents=True)
    joblib.dump(reference, reference_file(), compress=3)
    write_csv(OUT / "train_normal_reference_summary.csv", audit, list(audit[0]))
    (OUT / "reference_report.json").write_text(json.dumps({
        "source": "20 recovered-baseline prepared files, original 1-minute sensor values",
        "normal_definition": "Train split=0 and label=0; full 24-hour future outcome observed with no target fault in that horizon; this does not ensure 72h separation from past/future faults.",
        "alternative_not_applied": "For a later sensitivity check, use only training-period fault logs to exclude train rows within 72h of those faults, and keep candidate reference rows at least 72h before the validation cutoff. Recompute train-only references and compare with the same frozen validation. This is not mixed into the current ablation.",
        "validation_use": "Only the stored train reference is applied; no validation mean/median/IQR/CDF estimation.",
        "unknown_tool_policy": "Raise an error; never compute reference from validation.",
        "robust_z_formula": "(pressure - tool train-normal median) / (tool train-normal IQR / 1.349)",
        "percentile_formula": "empirical right-sided CDF of tool train-normal flow values",
        "baseline_features": 85, "B_added": list(EXTRA[:2]), "C_added": list(EXTRA)}, indent=2), encoding="utf-8")


def make_matrix(frame, tool_name, reference, kind):
    if tool_name not in reference:
        raise KeyError(f"Unknown tool {tool_name}; no validation-derived fallback")
    feature_count = len(FEATURES[kind])
    matrix = np.empty((len(frame), feature_count), dtype=np.float32)
    matrix[:, :85] = frame[BASE_FEATURES].to_numpy(dtype=np.float32, copy=False)
    for sensor in SENSORS:
        if not np.array_equal(matrix[:, BASE_FEATURES.index(sensor)], frame[sensor].to_numpy(dtype=np.float32), equal_nan=True):
            raise AssertionError("Baseline sensor value changed")
    flow = matrix[:, BASE_FEATURES.index("FLOWCOOLFLOWRATE")]
    pressure = matrix[:, BASE_FEATURES.index("FLOWCOOLPRESSURE")]
    fr = reference[tool_name]["FLOWCOOLFLOWRATE"]
    pr = reference[tool_name]["FLOWCOOLPRESSURE"]
    matrix[:, 85] = np.where(np.isfinite(flow),
                              np.searchsorted(fr["sorted"], flow, side="right") / len(fr["sorted"]), np.nan)
    matrix[:, 86] = (pressure - pr["median"]) / (pr["iqr"] / 1.349)
    if kind == "C":
        matrix[:, 87] = flow - fr["median"]
        matrix[:, 88] = pressure - pr["median"]
    return matrix


def train(kind):
    if kind not in FEATURES:
        raise ValueError(kind)
    if model_file(kind).exists() or score_file(kind).exists() or (OUT / f"training_{kind.lower()}.json").exists():
        raise FileExistsError(f"{kind} model or scores already exist")
    baseline, sweep, prep, paths = paths_and_checks()
    reference = joblib.load(reference_file())
    if set(reference) != {path.stem for path in paths}:
        raise AssertionError("Reference tool population changed")
    expected_rows, expected_positives = 4606868, 380626
    matrix = np.empty((expected_rows, len(FEATURES[kind])), dtype=np.float32)
    labels = np.empty(expected_rows, dtype=np.int8)
    cutoffs = {r["tool"]: r["split_cutoff_time"] for r in prep["tools"]}
    training_events = set()
    start = 0
    for path in paths:
        frame = pd.read_parquet(path, columns=[*BASE_FEATURES, "label", "split", "next_event_time"])
        part = frame.loc[(frame.split == 0) & (frame.label >= 0)]
        end = start + len(part)
        matrix[start:end] = make_matrix(part, path.stem, reference, kind)
        labels[start:end] = part.label.to_numpy(dtype=np.int8)
        positive_times = part.loc[part.label == 1, "next_event_time"].dropna().unique()
        if np.any(positive_times >= cutoffs[path.stem]):
            raise AssertionError("Training event crosses validation cutoff")
        training_events.update((path.stem, int(v)) for v in positive_times)
        start = end
        print(f"{kind} loaded {path.stem}: {len(part):,}", flush=True)
    if start != expected_rows or int(labels.sum()) != expected_positives:
        raise AssertionError("Training population changed")
    parameters = baseline["xgboost_parameters"]
    if not np.isclose(parameters["scale_pos_weight"], (expected_rows-expected_positives)/expected_positives):
        raise AssertionError("Class weight changed")
    model = XGBClassifier(**parameters)
    print(f"fitting {kind}: {len(labels):,} rows, {matrix.shape[1]} features", flush=True)
    model.fit(matrix, labels)
    del matrix, labels
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES[kind], "reference": reference,
                 "parameters": parameters}, model_file(kind))
    with np.load(SOURCE / "validation_probabilities.npz") as fixed:
        old_truth, old_time, old_tool = (fixed[k].copy() for k in ("truth", "time", "tool_code"))
    score_parts, truth_parts, time_parts, tool_parts, stage_parts = [], [], [], [], []
    validation_events = set()
    for code, path in enumerate(paths):
        frame = pd.read_parquet(path, columns=[*BASE_FEATURES, "time", "label", "split", "next_event_time"])
        part = frame.loc[(frame.split == 2) & (frame.label >= 0)]
        x = make_matrix(part, path.stem, reference, kind)
        score_parts.append(model.predict_proba(x)[:, 1].astype(np.float32))
        truth_parts.append(part.label.to_numpy(dtype=np.int8))
        time_parts.append(part.time.to_numpy(dtype=np.int64))
        tool_parts.append(np.full(len(part), code, dtype=np.uint8))
        stage_parts.append(part.stage.to_numpy(dtype=np.int32))
        validation_events.update((path.stem, int(v)) for v in part.loc[part.label == 1, "next_event_time"].dropna().unique())
        print(f"{kind} scored {path.stem}: {len(part):,}", flush=True)
    truth, time, tool, probability, stage = map(np.concatenate,
        (truth_parts, time_parts, tool_parts, score_parts, stage_parts))
    if not (np.array_equal(truth, old_truth) and np.array_equal(time, old_time) and np.array_equal(tool, old_tool)):
        raise AssertionError("Validation labels/time/order differ from baseline")
    if training_events & validation_events:
        raise AssertionError("Same positive fault event in train and validation")
    np.savez_compressed(score_file(kind), truth=truth, probability=probability,
                        time=time, tool_code=tool, stage=stage)
    (OUT / f"training_{kind.lower()}.json").write_text(json.dumps({
        "model": kind, "features": FEATURES[kind], "train_rows": expected_rows,
        "train_positive_rows": expected_positives, "validation_rows": len(truth),
        "validation_positive_rows": int(truth.sum()), "validation_alignment_exact": True,
        "train_validation_event_overlap": 0, "parameters": parameters,
        "reference_sha256": file_hash(reference_file()), "model_sha256": file_hash(model_file(kind)),
        "score_sha256": file_hash(score_file(kind)), "baseline_model_sha256": sweep["model_sha256"]},
        indent=2), encoding="utf-8")


def evaluate():
    if (OUT / "experiment_report.json").exists():
        raise FileExistsError("Experiment report already exists")
    baseline, sweep, prep, _ = paths_and_checks()
    for kind in FEATURES:
        report = read_json(OUT / f"training_{kind.lower()}.json")
        if (file_hash(model_file(kind)), file_hash(score_file(kind)), file_hash(reference_file())) != (
                report["model_sha256"], report["score_sha256"], report["reference_sha256"]):
            raise AssertionError(f"{kind} artifact hash changed")
    data = {}
    with np.load(SOURCE / "validation_probabilities.npz") as saved:
        truth, time, tool, data["A"] = (saved[k].copy() for k in ("truth", "time", "tool_code", "probability"))
    with np.load(score_file("B")) as b, np.load(score_file("C")) as c:
        for scores in (b, c):
            if not all(np.array_equal(scores[key], value) for key, value in
                       (("truth", truth), ("time", time), ("tool_code", tool))):
                raise AssertionError("Validation rows differ among A/B/C")
        data["B"] = b["probability"].copy()
        data["C"] = c["probability"].copy()
        stage = b["stage"].copy()
        if not np.array_equal(stage, c["stage"]):
            raise AssertionError("Validation stage differs between B and C")
    ap = {kind: float(average_precision_score(truth, values)) for kind, values in data.items()}
    if not np.isclose(ap["A"], baseline["metrics"]["average_precision"], atol=1e-12):
        raise AssertionError("Baseline AP mismatch")
    tool_map = {int(k): v for k, v in sweep["tool_code_map"].items()}
    reverse = {name: code for code, name in tool_map.items()}
    cutoffs = {r["tool"]: r["split_cutoff_time"] for r in prep["tools"]}
    events = load_events(cutoffs, set(tool_map.values()))
    if len(events) != 181:
        raise AssertionError("Fault event count changed")

    def measures(alarm, kind):
        binary = alarm.astype(np.uint8)
        row, by_tool = row_and_alarm_metrics(.5, truth, binary, time, tool, ap[kind], tool_map)
        event, details = event_metrics(.5, events, binary, time, tool, reverse)
        if event["evaluable_fault_events"] != 163:
            raise AssertionError("Evaluable events changed")
        return {**row, **event}, by_tool, details

    at05, fixed, grids = [], [], []
    by_tool_rows, by_stage_rows, event_rows = [], [], []
    best = {}
    for kind in ("A", "B", "C"):
        scores = data[kind]
        m05, _, _ = measures(scores >= .5, kind)
        if kind == "A" and any(m05[k] != baseline["metrics"][k] for k in
                               ("tn", "fp", "fn", "tp", "false_alarm_events_10m")):
            raise AssertionError("Baseline threshold .5 changed")
        at05.append({"model": kind, "threshold": .5, "condition": "1_of_1m",
                     **{k: v for k, v in m05.items() if k != "threshold"}})
        fixed_alarm = causal_alarm(scores >= .36, time, tool, sorted(tool_map), 5, 2)
        fm, ft, fd = measures(fixed_alarm, kind)
        if kind == "A" and (fm["detected_fault_events"], fm["false_alarm_events_10m"]) != (138, 6066):
            raise AssertionError("Frozen operating candidate changed")
        fixed.append({"model": kind, "threshold": .36, "condition": "2_of_5m",
                      **{k: v for k, v in fm.items() if k != "threshold"}})
        candidates = []
        for threshold in THRESHOLDS:
            raw = scores >= threshold
            for condition, window, hits in CONDITIONS:
                alarm = causal_alarm(raw, time, tool, sorted(tool_map), window, hits)
                metrics, _, _ = measures(alarm, kind)
                item = {"model": kind, "threshold": threshold, "condition": condition,
                        "window_minutes": window, "minimum_hits": hits,
                        **{k: v for k, v in metrics.items() if k != "threshold"},
                        "qualifies_75pct_all_events": metrics["fault_event_recall_all_logged"] >= .75}
                grids.append(item)
                if item["qualifies_75pct_all_events"]:
                    candidates.append(item)
            print(f"grid {kind} threshold {threshold:.2f} done", flush=True)
        best[kind] = min(candidates, key=lambda r: (r["false_alarm_events_10m"],
                          -r["detected_fault_events"])) if candidates else None
        policies = [("fixed", .36, 5, 2)]
        if best[kind]:
            chosen = best[kind]
            policies.append(("grid_best_75pct", chosen["threshold"], chosen["window_minutes"], chosen["minimum_hits"]))
        for label, threshold, window, hits in policies:
            alarm = causal_alarm(scores >= threshold, time, tool, sorted(tool_map), window, hits)
            summary, tool_rows, details = measures(alarm, kind)
            stage_rows = stage_false_episodes(truth, alarm, time, tool, stage, tool_map)
            for r in tool_rows:
                by_tool_rows.append({"model": kind, "policy": label, "threshold": threshold,
                                     "window_minutes": window, "minimum_hits": hits, **r})
            for r in stage_rows:
                by_stage_rows.append({"model": kind, "policy": label, "threshold": threshold,
                                      "window_minutes": window, "minimum_hits": hits, **r})
            event_rows.extend({"model": kind, "policy": label, **r} for r in details)
    write_csv(OUT / "threshold_0_5_comparison.csv", at05, list(at05[0]))
    write_csv(OUT / "fixed_policy_comparison.csv", fixed, list(fixed[0]))
    write_csv(OUT / "threshold_persistence_grid.csv", grids, list(grids[0]))
    write_csv(OUT / "false_alarm_by_tool.csv", by_tool_rows, list(by_tool_rows[0]))
    write_csv(OUT / "false_alarm_by_tool_stage.csv", by_stage_rows, list(by_stage_rows[0]))
    write_csv(OUT / "fault_event_details.csv", event_rows, list(event_rows[0]))
    shifts = []
    for policy in ("fixed", "grid_best_75pct"):
        base = {r["tool_id"]: r["false_alarm_events_10m"] for r in by_tool_rows
                if r["model"] == "A" and r["policy"] == "fixed"}
        for kind in ("B", "C"):
            other = {r["tool_id"]: r["false_alarm_events_10m"] for r in by_tool_rows
                     if r["model"] == kind and r["policy"] == policy}
            for name in sorted(base):
                shifts.append({"model": kind, "policy": policy, "tool_id": name,
                               "baseline_fixed_events": base[name], "new_events": other[name],
                               "change": other[name]-base[name]})
    write_csv(OUT / "false_alarm_tool_shift_vs_baseline.csv", shifts, list(shifts[0]))

    def concentration(kind, policy):
        tr = [r for r in by_tool_rows if r["model"] == kind and r["policy"] == policy]
        sr = [r for r in by_stage_rows if r["model"] == kind and r["policy"] == policy]
        total = sum(r["false_alarm_events_10m"] for r in tr)
        top = sorted(tr, key=lambda r: r["false_alarm_events_10m"], reverse=True)[:5]
        return {"03_M02_total_false_alarm_events": next(r["false_alarm_events_10m"] for r in tr if r["tool_id"] == "03_M02"),
                "03_M02_stage1_false_alarm_events": next((r["false_alarm_events_10m"] for r in sr if r["tool_id"] == "03_M02" and r["stage"] == 1), 0),
                "top_five_tools": [r["tool_id"] for r in top],
                "top_five_false_alarm_events": sum(r["false_alarm_events_10m"] for r in top),
                "top_five_share": sum(r["false_alarm_events_10m"] for r in top)/total if total else None}
    concentrations = {kind: {policy: concentration(kind, policy) for policy in
                      ("fixed", "grid_best_75pct") if any(r["model"] == kind and r["policy"] == policy for r in by_tool_rows)}
                      for kind in ("A", "B", "C")}
    if concentrations["A"]["fixed"]["03_M02_stage1_false_alarm_events"] != 840:
        raise AssertionError("Historical stage attribution changed")
    comparison = []
    for kind in ("A", "B", "C"):
        fm = next(r for r in fixed if r["model"] == kind)
        chosen = best[kind]
        fixed_conc = concentrations[kind]["fixed"]
        best_conc = concentrations[kind].get("grid_best_75pct", {})
        comparison.append({"model": kind, "feature_count": len(BASE_FEATURES) if kind == "A" else len(FEATURES[kind]),
            "average_precision": ap[kind], "fixed_threshold": .36, "fixed_condition": "2_of_5m",
            "fixed_detected_fault_events": fm["detected_fault_events"],
            "fixed_recall_all_events": fm["fault_event_recall_all_logged"],
            "fixed_false_alarm_events": fm["false_alarm_events_10m"],
            "fixed_03_M02_events": fixed_conc["03_M02_total_false_alarm_events"],
            "fixed_03_M02_stage1_events": fixed_conc["03_M02_stage1_false_alarm_events"],
            "fixed_top5_share": fixed_conc["top_five_share"],
            "best_threshold": chosen["threshold"] if chosen else None,
            "best_condition": chosen["condition"] if chosen else None,
            "best_detected_fault_events": chosen["detected_fault_events"] if chosen else None,
            "best_false_alarm_events": chosen["false_alarm_events_10m"] if chosen else None,
            "best_03_M02_events": best_conc.get("03_M02_total_false_alarm_events"),
            "best_03_M02_stage1_events": best_conc.get("03_M02_stage1_false_alarm_events"),
            "best_top5_share": best_conc.get("top_five_share")})
    write_csv(OUT / "model_comparison_summary.csv", comparison, list(comparison[0]))
    report = {"purpose": "Train-normal tool-relative feature ablation on the recovered 24h XGBoost split",
              "baseline_features": 85, "B_added": list(EXTRA[:2]), "C_added": list(EXTRA),
              "model_parameters_unchanged": baseline["xgboost_parameters"],
              "reference_report": str(OUT / "reference_report.json"),
              "ap": ap, "threshold_0_5": at05, "fixed_policy_0_36_2of5": fixed,
              "best_at_least_75pct_all_fault_events": best, "concentrations": concentrations,
              "validation_limit": "Grid selection reuses the same validation; no independent later labeled test exists. Do not call selected results operating performance."}
    (OUT / "experiment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ap": ap,
                      "fixed": [(r["model"], r["detected_fault_events"], r["false_alarm_events_10m"]) for r in fixed],
                      "best": {k: (v["threshold"], v["condition"], v["detected_fault_events"], v["false_alarm_events_10m"]) if v else None for k, v in best.items()},
                      "concentrations": concentrations}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train_b", "train_c", "evaluate"))
    action = parser.parse_args().action
    {"prepare": prepare, "train_b": lambda: train("B"),
     "train_c": lambda: train("C"), "evaluate": evaluate}[action]()
