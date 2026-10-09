"""One-factor 24h XGBoost experiment: native categorical Tool x Stage.

Usage: python src/experiment_tool_stage_interaction_24h.py train|evaluate
Only the 20 frozen prepared real-data files are read. Existing results are not
rewritten. Native XGBoost category codes are fit on training rows only.
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

OUT = ROOT / "output" / "experiments" / "tool_stage_interaction_24h"
MODEL = ROOT / "output" / "models" / "xgb_24h_tool_stage_interaction.joblib"
SCORES = OUT / "validation_probabilities.npz"
NEW_FEATURE = "tool_stage_category"
BASE_FEATURES = read_json(BASELINE / "reproduction_report.json")["features"]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sources():
    baseline = read_json(BASELINE / "reproduction_report.json")
    sweep = read_json(SOURCE / "sweep_report.json")
    prep = read_json(BASELINE / "preparation_report.json")
    paths = sorted((BASELINE / "prepared").glob("??_M??.parquet"))
    if len(paths) != 20 or len(BASE_FEATURES) != 85 or baseline["features"] != BASE_FEATURES:
        raise AssertionError("Recovered baseline schema or tool count changed")
    if file_hash(BASELINE / "xgb_24h_recovery.joblib") != sweep["model_sha256"]:
        raise AssertionError("Recovered baseline model changed")
    return baseline, sweep, prep, paths


def vocabulary(paths):
    """Never inspect validation stage values while constructing category IDs."""
    mapping = {}
    counts = {}
    for path in paths:
        frame = pd.read_parquet(path, columns=["stage", "label", "split"])
        chosen = frame.loc[(frame.split == 0) & (frame.label >= 0), "stage"]
        if chosen.isna().any():
            raise AssertionError(f"Missing training stage: {path.stem}")
        categories = sorted(int(v) for v in chosen.unique())
        for stage in categories:
            mapping[(path.stem, stage)] = len(mapping)
        counts[path.stem] = len(categories)
    return mapping, counts


def as_matrix(frame, path, mapping):
    matrix = np.empty((len(frame), 86), dtype=np.float32)
    matrix[:, :85] = frame[BASE_FEATURES].to_numpy(dtype=np.float32, copy=False)
    np.nan_to_num(matrix[:, :85], copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    stages = frame.stage.to_numpy(dtype=np.int32)
    matrix[:, 85] = np.fromiter((mapping.get((path.stem, int(s)), np.nan) for s in stages),
                                dtype=np.float32, count=len(stages))
    return matrix, int(np.isnan(matrix[:, 85]).sum())


def train():
    if MODEL.exists() or SCORES.exists() or (OUT / "training_report.json").exists():
        raise FileExistsError("Experiment model or scores already exist")
    baseline, sweep, prep, paths = sources()
    mapping, counts = vocabulary(paths)
    parameters = baseline["xgboost_parameters"]
    ntrain = sum(item["train_rows"] for item in prep["tools"])
    # Frozen prep includes censored train rows, which are removed below.
    expected_rows, expected_pos = 4606868, 380626
    x = np.empty((expected_rows, 86), dtype=np.float32)
    y = np.empty(expected_rows, dtype=np.int8)
    start = 0
    train_event_keys = set()
    cutoffs = {item["tool"]: item["split_cutoff_time"] for item in prep["tools"]}
    for path in paths:
        frame = pd.read_parquet(path, columns=[*BASE_FEATURES, "label", "split", "next_event_time"])
        part = frame.loc[(frame.split == 0) & (frame.label >= 0)]
        end = start + len(part)
        values, unseen = as_matrix(part, path, mapping)
        if unseen:
            raise AssertionError("Training category missing from train vocabulary")
        x[start:end] = values
        y[start:end] = part.label.to_numpy(dtype=np.int8)
        times = part.loc[part.label == 1, "next_event_time"].dropna().unique()
        if np.any(times >= cutoffs[path.stem]):
            raise AssertionError("Training event crosses validation boundary")
        train_event_keys.update((path.stem, int(value)) for value in times)
        start = end
        print(f"loaded {path.stem}: {len(part):,} train rows", flush=True)
    if start != expected_rows or int(y.sum()) != expected_pos:
        raise AssertionError("Train rows or positives differ from recovered baseline")
    if not np.isclose(parameters["scale_pos_weight"], (expected_rows-expected_pos)/expected_pos):
        raise AssertionError("Class weight changed")
    # The categorical *type* is input metadata, not a changed tree hyperparameter.
    model = XGBClassifier(**parameters, feature_types=["q"] * 85 + ["c"], enable_categorical=True)
    print(f"fitting {expected_rows:,} real rows with 85 frozen + 1 categorical feature", flush=True)
    model.fit(x, y)
    del x, y
    OUT.mkdir(parents=True)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": [*BASE_FEATURES, NEW_FEATURE],
                 "category_mapping": mapping, "parameters": parameters}, MODEL)
    with np.load(SOURCE / "validation_probabilities.npz") as frozen:
        old_truth, old_time, old_tool = (frozen[k].copy() for k in ("truth", "time", "tool_code"))
    prob_parts, truth_parts, time_parts, tool_parts, stage_parts = [], [], [], [], []
    unseen_rows = {}
    valid_event_keys = set()
    for code, path in enumerate(paths):
        frame = pd.read_parquet(path, columns=[*BASE_FEATURES, "time", "label", "split", "next_event_time"])
        part = frame.loc[(frame.split == 2) & (frame.label >= 0)]
        val_matrix, unseen = as_matrix(part, path, mapping)
        prediction = model.predict_proba(val_matrix)[:, 1].astype(np.float32)
        prob_parts.append(prediction)
        truth_parts.append(part.label.to_numpy(dtype=np.int8))
        time_parts.append(part.time.to_numpy(dtype=np.int64))
        tool_parts.append(np.full(len(part), code, dtype=np.uint8))
        stage_parts.append(part.stage.to_numpy(dtype=np.int32))
        unseen_rows[path.stem] = unseen
        valid_event_keys.update((path.stem, int(value)) for value in part.loc[part.label == 1, "next_event_time"].dropna().unique())
        print(f"scored {path.stem}: {len(part):,} validation rows; {unseen} unseen categories", flush=True)
    truth, time, tool, score, stage = map(np.concatenate,
        (truth_parts, time_parts, tool_parts, prob_parts, stage_parts))
    if not (np.array_equal(truth, old_truth) and np.array_equal(time, old_time) and np.array_equal(tool, old_tool)):
        raise AssertionError("Validation rows, order, or labels differ from frozen baseline")
    if train_event_keys & valid_event_keys:
        raise AssertionError("A positive fault event appears in train and validation")
    np.savez_compressed(SCORES, truth=truth, probability=score, time=time, tool_code=tool, stage=stage)
    report = {"base_features": 85, "new_feature": NEW_FEATURE, "feature_type": "native_xgboost_categorical",
              "category_definition": "(tool_id, integer stage) pairs seen in labeled train rows only; stable sorted IDs; unseen validation pairs become NaN",
              "train_vocabulary_size": len(mapping), "categories_by_tool": counts,
              "validation_unseen_category_rows_by_tool": unseen_rows,
              "validation_unseen_category_rows_total": sum(unseen_rows.values()),
              "train_rows": expected_rows, "train_positive_rows": expected_pos,
              "validation_rows": len(truth), "validation_positive_rows": int(truth.sum()),
              "validation_alignment_exact": True, "train_validation_positive_event_overlap": 0,
              "parameters": parameters, "model_sha256": file_hash(MODEL),
              "score_sha256": file_hash(SCORES), "baseline_model_sha256": sweep["model_sha256"]}
    (OUT / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def stage_false_episodes(truth, alarm, time, tool, stage, tool_map):
    rows = []
    for code, name in sorted(tool_map.items()):
        pos = np.flatnonzero(tool == code)
        fp_idx = pos[alarm[pos] & (truth[pos] == 0)]
        fp_times = time[fp_idx]
        starts = fp_idx[np.r_[True, np.diff(fp_times) > 600]] if len(fp_idx) else np.empty(0, dtype=int)
        values, counts = np.unique(stage[starts], return_counts=True)
        rows.extend({"tool_id": name, "stage": int(value), "false_alarm_events_10m": int(count)}
                    for value, count in zip(values, counts))
    return rows


def evaluate():
    if (OUT / "experiment_report.json").exists():
        raise FileExistsError("Experiment report already exists")
    baseline, sweep, prep, _ = sources()
    training = read_json(OUT / "training_report.json")
    if file_hash(MODEL) != training["model_sha256"] or file_hash(SCORES) != training["score_sha256"]:
        raise AssertionError("Experiment model or scores changed")
    with np.load(SCORES) as new, np.load(SOURCE / "validation_probabilities.npz") as old:
        truth, prob, time, tool, stage = (new[k].copy() for k in ("truth", "probability", "time", "tool_code", "stage"))
        old_prob = old["probability"].copy()
        if not all(np.array_equal(new[k], old[k]) for k in ("truth", "time", "tool_code")):
            raise AssertionError("New and baseline validation rows differ")
    tool_map = {int(code): name for code, name in sweep["tool_code_map"].items()}
    reverse = {name: code for code, name in tool_map.items()}
    cutoffs = {item["tool"]: item["split_cutoff_time"] for item in prep["tools"]}
    events = load_events(cutoffs, set(tool_map.values()))
    if len(events) != 181:
        raise AssertionError("Fault event population changed")
    new_ap = float(average_precision_score(truth, prob))
    old_ap = float(average_precision_score(truth, old_prob))
    if not np.isclose(old_ap, baseline["metrics"]["average_precision"], atol=1e-12):
        raise AssertionError("Baseline AP changed")

    def metrics(binary, ap):
        row, by_tool = row_and_alarm_metrics(.5, truth, binary.astype(np.uint8), time, tool, ap, tool_map)
        event, detail = event_metrics(.5, events, binary.astype(np.uint8), time, tool, reverse)
        if event["evaluable_fault_events"] != 163:
            raise AssertionError("Evaluable event population changed")
        return {**row, **event}, by_tool, detail

    base05, _, _ = metrics(old_prob >= .5, old_ap)
    if any(base05[key] != baseline["metrics"][key] for key in ("tn", "fp", "fn", "tp", "false_alarm_events_10m")):
        raise AssertionError("Recovered threshold .5 baseline changed")
    new05, _, _ = metrics(prob >= .5, new_ap)
    old_fixed_alarm = causal_alarm(old_prob >= .36, time, tool, sorted(tool_map), 5, 2)
    new_fixed_alarm = causal_alarm(prob >= .36, time, tool, sorted(tool_map), 5, 2)
    old_fixed, old_fixed_tools, _ = metrics(old_fixed_alarm, old_ap)
    new_fixed, new_fixed_tools, _ = metrics(new_fixed_alarm, new_ap)
    if (old_fixed["detected_fault_events"], old_fixed["false_alarm_events_10m"]) != (138, 6066):
        raise AssertionError("Frozen candidate policy changed")

    grid = []
    for threshold in THRESHOLDS:
        raw = prob >= threshold
        for condition, window, required in CONDITIONS:
            binary = causal_alarm(raw, time, tool, sorted(tool_map), window, required)
            summary, _, _ = metrics(binary, new_ap)
            grid.append({"threshold": threshold, "condition": condition, "window_minutes": window,
                         "minimum_hits": required, **{k: v for k, v in summary.items() if k != "threshold"},
                         "qualifies_75pct_all_events": summary["fault_event_recall_all_logged"] >= .75})
            print(f"{threshold:.2f} {condition}: {summary['detected_fault_events']}/181, false {summary['false_alarm_events_10m']}", flush=True)
    eligible = [item for item in grid if item["qualifies_75pct_all_events"]]
    chosen = min(eligible, key=lambda item: (item["false_alarm_events_10m"],
                  -item["detected_fault_events"])) if eligible else None

    def distribution(label, threshold, condition, window, required, probability, ap):
        binary = causal_alarm(probability >= threshold, time, tool, sorted(tool_map), window, required)
        summary, by_tool, details = metrics(binary, ap)
        by_stage = stage_false_episodes(truth, binary, time, tool, stage, tool_map)
        for item in by_tool:
            item["policy"] = label
        for item in by_stage:
            item["policy"] = label
        by_tool_sorted = sorted(by_tool, key=lambda item: item["false_alarm_events_10m"], reverse=True)
        top5 = sum(item["false_alarm_events_10m"] for item in by_tool_sorted[:5])
        return summary, by_tool, by_stage, details, {
            "top_five_tools": [item["tool_id"] for item in by_tool_sorted[:5]],
            "top_five_false_alarm_events": top5,
            "top_five_share": top5 / summary["false_alarm_events_10m"] if summary["false_alarm_events_10m"] else None,
            "03_M02_stage1_false_alarm_events": next((item["false_alarm_events_10m"] for item in by_stage
                if item["tool_id"] == "03_M02" and item["stage"] == 1), 0)}

    compared = {}
    tool_rows, stage_rows, fault_rows = [], [], []
    policies = [("baseline_fixed_0.36_2of5", .36, "2_of_5m", 5, 2, old_prob, old_ap),
                ("new_fixed_0.36_2of5", .36, "2_of_5m", 5, 2, prob, new_ap)]
    if chosen:
        policies.append(("new_grid_best_at_least_75pct", chosen["threshold"], chosen["condition"],
                         chosen["window_minutes"], chosen["minimum_hits"], prob, new_ap))
    for label, threshold, condition, window, required, scores, ap in policies:
        summary, by_tool, by_stage, details, concentration = distribution(
            label, threshold, condition, window, required, scores, ap)
        compared[label] = {**summary, "threshold": threshold, "condition": condition, **concentration}
        tool_rows.extend(by_tool)
        stage_rows.extend(by_stage)
        fault_rows.extend({"policy": label, **detail} for detail in details)
    if compared["baseline_fixed_0.36_2of5"]["03_M02_stage1_false_alarm_events"] != 840:
        raise AssertionError("Prior Tool x Stage false alarm count changed")
    base_tool = {r["tool_id"]: r["false_alarm_events_10m"] for r in tool_rows if r["policy"] == policies[0][0]}
    new_tool = {r["tool_id"]: r["false_alarm_events_10m"] for r in tool_rows if r["policy"] == policies[1][0]}
    shifts = [{"tool_id": name, "baseline_false_alarm_events": base_tool[name],
               "new_false_alarm_events": new_tool[name],
               "change": new_tool[name] - base_tool[name]} for name in sorted(base_tool)]
    base_stage = {(r["tool_id"], r["stage"]): r["false_alarm_events_10m"] for r in stage_rows if r["policy"] == policies[0][0]}
    new_stage = {(r["tool_id"], r["stage"]): r["false_alarm_events_10m"] for r in stage_rows if r["policy"] == policies[1][0]}
    stage_shifts = [{"tool_id": name, "stage": stage, "baseline_false_alarm_events": base_stage.get((name, stage), 0),
                    "new_false_alarm_events": new_stage.get((name, stage), 0),
                    "change": new_stage.get((name, stage), 0) - base_stage.get((name, stage), 0)}
                    for name, stage in sorted(base_stage.keys() | new_stage.keys())]
    write_csv(OUT / "threshold_0_5_comparison.csv", [
        {"model": "recovered_baseline", **base05}, {"model": "tool_stage_interaction", **new05}],
        ["model", *base05.keys()])
    write_csv(OUT / "fixed_policy_comparison.csv", [
        {"model": "recovered_baseline", **old_fixed, "threshold": .36, "condition": "2_of_5m"},
        {"model": "tool_stage_interaction", **new_fixed, "threshold": .36, "condition": "2_of_5m"}],
        ["model", *old_fixed.keys(), "condition"])
    write_csv(OUT / "threshold_persistence_grid.csv", grid, list(grid[0]))
    write_csv(OUT / "false_alarm_by_tool.csv", tool_rows, list(tool_rows[0]))
    write_csv(OUT / "false_alarm_by_tool_stage.csv", stage_rows, list(stage_rows[0]))
    write_csv(OUT / "fault_event_details.csv", fault_rows, list(fault_rows[0]))
    write_csv(OUT / "fixed_policy_tool_shift.csv", shifts, list(shifts[0]))
    write_csv(OUT / "fixed_policy_tool_stage_shift.csv", stage_shifts, list(stage_shifts[0]))
    report = {"experiment": "Add only one native categorical Tool x Stage feature to frozen 85-feature baseline",
              "parameters": baseline["xgboost_parameters"], "training": training,
              "ap_baseline": old_ap, "ap_new": new_ap, "ap_change": new_ap - old_ap,
              "threshold_0_5": {"baseline": base05, "new": new05},
              "fixed_policy": {"baseline": compared[policies[0][0]], "new": compared[policies[1][0]]},
              "grid_eligible_count": len(eligible), "grid_best_at_least_75pct_all_events": chosen,
              "grid_best_policy_distribution": compared.get("new_grid_best_at_least_75pct"),
              "fixed_tool_false_alarm_increase_sum": sum(max(0, r["change"]) for r in shifts),
              "fixed_tool_false_alarm_decrease_sum": sum(max(0, -r["change"]) for r in shifts),
              "fixed_new_increased_tools": [r for r in shifts if r["change"] > 0],
              "retain_interaction_feature": bool(chosen and new_ap > old_ap and
                  chosen["false_alarm_events_10m"] < old_fixed["false_alarm_events_10m"] and
                  new_fixed["detected_fault_events"] >= old_fixed["detected_fault_events"]),
              "limitation": "The grid is selected on the same reused validation interval, not an independent future test. Tool/Stage episode attribution uses the first false-positive row and may shift with episode merging."}
    (OUT / "experiment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ap_change": report["ap_change"], "fixed": report["fixed_policy"],
                      "best": chosen}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "evaluate"))
    args = parser.parse_args()
    {"train": train, "evaluate": evaluate}[args.action]()
