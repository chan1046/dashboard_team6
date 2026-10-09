"""Sealed-forward 60/20/20 A/B comparison on the 20 real train partitions.

Run prepare, train_a, train_b, evaluate. Test label/sensor rows are never
loaded by this script. The final 20% was historically used as validation,
so the report never calls it a historically untouched final test.
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from evaluate_alarm_persistence_24h import causal_alarm
from evaluate_event_thresholds_24h import BASELINE, FAULT_LOG, ROOT, event_metrics, row_and_alarm_metrics, write_csv
from evaluate_threshold_persistence_grid_24h import CONDITIONS, THRESHOLDS
from experiment_tool_relative_features_24h import BASE_FEATURES, EXTRA, make_matrix
from experiment_tool_stage_interaction_24h import file_hash

OUT = ROOT / "output" / "experiments" / "forward_validation_ab_24h"
MODELS = ROOT / "output" / "models"
HORIZON = 86400
BATCH = 8192
TOOLS = sorted((BASELINE / "prepared").glob("??_M??.parquet"))
FEATURES = {"A": BASE_FEATURES, "B": [*BASE_FEATURES, *EXTRA[:2]]}


def model_path(kind):
    return MODELS / f"xgb_24h_forward_{kind.lower()}.joblib"


def score_path(kind):
    return OUT / f"validation_probabilities_{kind.lower()}.npz"


def plan_path():
    return OUT / "split_plan.json"


def reference_path():
    return OUT / "train_normal_references.joblib"


def boundary_plan(path):
    parquet = pq.ParquetFile(path)
    n = parquet.metadata.num_rows
    train_index = int(n * .60) // BATCH * BATCH
    test_index = int(n * .80) // BATCH * BATCH
    if not (0 < train_index < test_index < n):
        raise AssertionError(f"Invalid 60/20/20 indices for {path.stem}")
    boundary_times = {}
    cursor = 0
    for batch in parquet.iter_batches(batch_size=BATCH, columns=["time"]):
        if cursor in (train_index, test_index):
            boundary_times[cursor] = int(batch.column(0)[0].as_py())
        if cursor == test_index:
            break  # Only the first test timestamp is read to define its boundary.
        cursor += batch.num_rows
    if set(boundary_times) != {train_index, test_index}:
        raise AssertionError(f"Could not locate time boundaries: {path.stem}")
    return {"tool_id": path.stem, "total_cached_minute_rows_metadata": n,
            "train_boundary_index": train_index, "test_boundary_index": test_index,
            "validation_start_second": boundary_times[train_index],
            "test_start_second": boundary_times[test_index],
            "train_end_exclusive_second": boundary_times[train_index] - HORIZON,
            "validation_end_exclusive_second": boundary_times[test_index] - HORIZON,
            "train_fraction_approx": train_index / n,
            "validation_fraction_before_purge_approx": (test_index-train_index)/n,
            "test_fraction_approx": (n-test_index)/n}


def read_pretest(path, stop_index, columns):
    """Never deserialize a batch containing test labels or sensor values."""
    if stop_index % BATCH:
        raise AssertionError("Test boundary must be a batch boundary")
    batches, cursor = [], 0
    batch_iter = iter(pq.ParquetFile(path).iter_batches(batch_size=BATCH, columns=columns))
    while cursor < stop_index:
        batch = next(batch_iter)
        if cursor + batch.num_rows > stop_index:
            raise AssertionError("A data batch crosses the sealed test boundary")
        batches.append(batch)
        cursor += batch.num_rows
    if cursor != stop_index:
        raise AssertionError(f"Read past/short of test boundary: {path.stem}")
    return pa.Table.from_batches(batches).to_pandas()


def select(frame, row, split):
    time = frame.time.to_numpy(dtype=np.int64)
    if np.any(np.diff(time) <= 0):
        raise AssertionError(f"Time decreases for {row['tool_id']}")
    if not (time[0] < row["validation_start_second"] < row["test_start_second"]):
        raise AssertionError("Boundary order changed")
    if split == "train":
        mask = time < row["train_end_exclusive_second"]
    elif split == "validation":
        mask = (time >= row["validation_start_second"]) & (time < row["validation_end_exclusive_second"])
    else:
        raise ValueError(split)
    mask &= frame.label.to_numpy(dtype=np.int8) >= 0
    return frame.loc[mask]


def prepare():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    if len(TOOLS) != 20 or len(BASE_FEATURES) != 85:
        raise AssertionError("Frozen baseline tool/features changed")
    baseline = json.loads((BASELINE / "reproduction_report.json").read_text(encoding="utf-8"))
    rows = [boundary_plan(path) for path in TOOLS]
    references, counts = {}, []
    fault_log = pd.read_csv(FAULT_LOG, usecols=["tool_id", "fault_type", "time"])
    for row, path in zip(rows, TOOLS):
        frame = read_pretest(path, row["test_boundary_index"],
                             ["time", "label", "next_event_time", "FLOWCOOLFLOWRATE", "FLOWCOOLPRESSURE"])
        train = select(frame, row, "train")
        valid = select(frame, row, "validation")
        if not (len(train) and len(valid)):
            raise AssertionError("Empty train or validation interval")
        if not (int(train.time.max()) + HORIZON < row["validation_start_second"] and
                int(valid.time.max()) + HORIZON < row["test_start_second"]):
            raise AssertionError("Less than 24h purge at split boundary")
        if np.any(train.loc[train.label == 1, "next_event_time"] >= row["validation_start_second"]):
            raise AssertionError("Train positive event crosses validation boundary")
        if np.any(valid.loc[valid.label == 1, "next_event_time"] >= row["test_start_second"]):
            raise AssertionError("Validation positive event crosses test boundary")
        normal = train[train.label == 0]
        if len(normal) < 100:
            raise AssertionError(f"Insufficient train-normal rows {path.stem}")
        references[path.stem] = {}
        for sensor in ("FLOWCOOLFLOWRATE", "FLOWCOOLPRESSURE"):
            values = normal[sensor].to_numpy(dtype=np.float32)
            if not np.all(np.isfinite(values)):
                raise AssertionError("Missing train-normal sensor value")
            q = np.quantile(values, [.25, .5, .75])
            iqr = float(q[2]-q[0])
            if iqr <= 0:
                raise AssertionError("Zero IQR in train-normal reference")
            values.sort()
            references[path.stem][sensor] = {"sorted": values, "median": float(q[1]), "iqr": iqr}
        train_keys = set(int(v) for v in train.loc[train.label == 1, "next_event_time"].dropna().unique())
        val_keys = set(int(v) for v in valid.loc[valid.label == 1, "next_event_time"].dropna().unique())
        if train_keys & val_keys:
            raise AssertionError("Same fault event in train and validation")
        logged = fault_log[(fault_log.tool_id == path.stem) &
                           (fault_log.time >= row["validation_start_second"]) &
                           (fault_log.time < row["test_start_second"]) &
                           (fault_log.fault_type.isin(("pressure_low", "pressure_high", "leak")))]
        row.update({"train_labeled_rows": len(train), "validation_labeled_rows": len(valid),
                    "train_positive_rows": int((train.label == 1).sum()),
                    "validation_positive_rows": int((valid.label == 1).sum()),
                    "train_last_time": int(train.time.max()), "validation_last_time": int(valid.time.max()),
                    "train_validation_event_overlap": 0,
                    "validation_logged_fault_events": len(logged.drop_duplicates(["fault_type", "time"])),
                    "test_label_rows_read": 0, "test_sensor_rows_read": 0})
        counts.extend({"tool_id": path.stem, "sensor": s, "train_normal_rows": len(normal),
                       "median": references[path.stem][s]["median"], "iqr": references[path.stem][s]["iqr"]}
                      for s in references[path.stem])
        print(f"{path.stem}: train {len(train):,}, validation {len(valid):,}; test sealed", flush=True)
    OUT.mkdir(parents=True)
    joblib.dump(references, reference_path(), compress=3)
    write_csv(OUT / "split_by_tool.csv", rows, list(rows[0]))
    write_csv(OUT / "train_normal_reference_summary.csv", counts, list(counts[0]))
    (plan_path()).write_text(json.dumps({"scheme": "approximately 60/20/20 per tool by chronological minute-row count; boundaries aligned down to 8192-row batches",
        "purge": "At least 86400 seconds before train/validation and validation/test boundaries; no overlap of positive next-fault timestamps",
        "test_access": "Only one boundary timestamp per tool and row-count metadata read; no test labels/sensors/probabilities loaded or evaluated",
        "historical_caveat": "The last 20% overlaps the old repeatedly used validation interval. It is sealed in this experiment but not historically untouched; a genuinely independent final test requires new later labeled data.",
        "normal_reference": "Only new train split rows with label=0, separately per tool; validation never estimates a reference",
        "normal_reference_caveat": "Label=0 is fault-free for the next 24h; 72h distance from past faults is not assured. A later sensitivity check can use training-period faults only to enforce 72h exclusion.",
        "baseline_features": 85, "B_added": list(EXTRA[:2]),
        "parameters": baseline["xgboost_parameters"], "tools": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


def train(kind):
    if kind not in FEATURES:
        raise ValueError(kind)
    if model_path(kind).exists() or score_path(kind).exists() or (OUT / f"training_{kind.lower()}.json").exists():
        raise FileExistsError("Forward model or validation score already exists")
    plan = json.loads(plan_path().read_text(encoding="utf-8"))
    rows = plan["tools"]
    references = joblib.load(reference_path())
    expected = sum(r["train_labeled_rows"] for r in rows)
    x = np.empty((expected, len(FEATURES[kind])), dtype=np.float32)
    y = np.empty(expected, dtype=np.int8)
    cursor = 0
    for row, path in zip(rows, TOOLS):
        frame = read_pretest(path, row["test_boundary_index"],
                             [*BASE_FEATURES, "time", "label", "next_event_time"])
        part = select(frame, row, "train")
        stop = cursor + len(part)
        x[cursor:stop] = (part[BASE_FEATURES].to_numpy(dtype=np.float32, copy=True) if kind == "A"
                          else make_matrix(part, path.stem, references, "B"))
        y[cursor:stop] = part.label.to_numpy(dtype=np.int8)
        cursor = stop
        print(f"{kind} loaded {path.stem}: {len(part):,}", flush=True)
    if cursor != expected or int(y.sum()) != sum(r["train_positive_rows"] for r in rows):
        raise AssertionError("Prepared train counts differ")
    parameters = plan["parameters"]
    model = XGBClassifier(**parameters)
    print(f"{kind} fitting {expected:,} train rows / {x.shape[1]} features", flush=True)
    model.fit(x, y)
    del x, y
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES[kind], "parameters": parameters,
                 "train_normal_reference": references if kind == "B" else None}, model_path(kind))
    probability, truth, times, tools, stages = [], [], [], [], []
    for code, (row, path) in enumerate(zip(rows, TOOLS)):
        frame = read_pretest(path, row["test_boundary_index"],
                             [*BASE_FEATURES, "time", "label", "next_event_time"])
        part = select(frame, row, "validation")
        values = (part[BASE_FEATURES].to_numpy(dtype=np.float32, copy=True) if kind == "A"
                  else make_matrix(part, path.stem, references, "B"))
        probability.append(model.predict_proba(values)[:, 1].astype(np.float32))
        truth.append(part.label.to_numpy(dtype=np.int8))
        times.append(part.time.to_numpy(dtype=np.int64))
        tools.append(np.full(len(part), code, dtype=np.uint8))
        stages.append(part.stage.to_numpy(dtype=np.int32))
        print(f"{kind} scored {path.stem}: {len(part):,}", flush=True)
    prob, target, time, tool_code, stage = map(np.concatenate,
        (probability, truth, times, tools, stages))
    if (len(target), int(target.sum())) != (sum(r["validation_labeled_rows"] for r in rows),
                                           sum(r["validation_positive_rows"] for r in rows)):
        raise AssertionError("Validation counts changed")
    np.savez_compressed(score_path(kind), truth=target, probability=prob,
                        time=time, tool_code=tool_code, stage=stage)
    (OUT / f"training_{kind.lower()}.json").write_text(json.dumps({
        "model": kind, "train_rows": expected, "train_positive_rows": sum(r["train_positive_rows"] for r in rows),
        "validation_rows": len(target), "validation_positive_rows": int(target.sum()),
        "feature_names": FEATURES[kind], "parameters": parameters,
        "test_rows_scored": 0, "test_labels_or_sensors_loaded": False,
        "reference_sha256": file_hash(reference_path()), "model_sha256": file_hash(model_path(kind)),
        "validation_score_sha256": file_hash(score_path(kind))}, indent=2), encoding="utf-8")


def validation_events(plan):
    # Read only pre-test fault time/type; never make test outcome reports.
    log = pd.read_csv(FAULT_LOG, usecols=["tool_id", "fault_type", "time"])
    result = []
    for row in plan["tools"]:
        part = log[(log.tool_id == row["tool_id"]) &
                   (log.time >= row["validation_start_second"]) &
                   (log.time < row["test_start_second"]) &
                   (log.fault_type.isin(("pressure_low", "pressure_high", "leak")))]
        result.extend({"tool_id": row["tool_id"], "fault_type": str(item.fault_type),
                       "fault_time": int(item.time)} for item in part.drop_duplicates(["fault_type", "time"]).itertuples())
    return sorted(result, key=lambda x: (x["tool_id"], x["fault_time"], x["fault_type"]))


def evaluate():
    if (OUT / "experiment_report.json").exists():
        raise FileExistsError("Forward report already exists")
    plan = json.loads(plan_path().read_text(encoding="utf-8"))
    training = {kind: json.loads((OUT / f"training_{kind.lower()}.json").read_text()) for kind in ("A", "B")}
    for kind in ("A", "B"):
        if (file_hash(model_path(kind)), file_hash(score_path(kind)), file_hash(reference_path())) != (
            training[kind]["model_sha256"], training[kind]["validation_score_sha256"], training[kind]["reference_sha256"]):
            raise AssertionError(f"{kind} artifact changed")
    with np.load(score_path("A")) as a, np.load(score_path("B")) as b:
        truth, time, tool, stage = (a[key].copy() for key in ("truth", "time", "tool_code", "stage"))
        if not all(np.array_equal(a[key], b[key]) for key in ("truth", "time", "tool_code", "stage")):
            raise AssertionError("A/B validation population differs")
        probabilities = {"A": a["probability"].copy(), "B": b["probability"].copy()}
    ap = {kind: float(average_precision_score(truth, prob)) for kind, prob in probabilities.items()}
    names = {i: row["tool_id"] for i, row in enumerate(plan["tools"])}
    reverse = {name: code for code, name in names.items()}
    events = validation_events(plan)
    if not events:
        raise AssertionError("No validation fault events")

    def measures(alarm, kind):
        binary = alarm.astype(np.uint8)
        row, tools = row_and_alarm_metrics(.5, truth, binary, time, tool, ap[kind], names)
        event, details = event_metrics(.5, events, binary, time, tool, reverse)
        return {**row, **event}, tools, details

    at05, grid, best_by_level, tool_rows = [], [], [], []
    for kind in ("A", "B"):
        scores = probabilities[kind]
        m05, _, _ = measures(scores >= .5, kind)
        at05.append({"model": kind, "threshold": .5, "condition": "1_of_1m",
                     **{key: value for key, value in m05.items() if key != "threshold"}})
        model_grid = []
        for threshold in THRESHOLDS:
            raw = scores >= threshold
            for condition, window, required in CONDITIONS:
                alarms = causal_alarm(raw, time, tool, sorted(names), window, required)
                metrics, _, _ = measures(alarms, kind)
                item = {"model": kind, "threshold": threshold, "condition": condition,
                        "window_minutes": window, "minimum_hits": required,
                        **{key: value for key, value in metrics.items() if key != "threshold"}}
                model_grid.append(item)
                grid.append(item)
            print(f"{kind} grid {threshold:.2f}", flush=True)
        for level in (.70, .75, .80):
            eligible = [item for item in model_grid if item["fault_event_recall_all_logged"] is not None and
                        item["fault_event_recall_all_logged"] >= level]
            chosen = min(eligible, key=lambda item: (item["false_alarm_events_10m"],
                         -item["detected_fault_events"])) if eligible else None
            best_by_level.append({"model": kind, "minimum_all_event_recall": level,
                **(chosen if chosen is not None else {"candidate_found": False})})
            if level == .75 and chosen is not None:
                alarms = causal_alarm(scores >= chosen["threshold"], time, tool, sorted(names),
                                      chosen["window_minutes"], chosen["minimum_hits"])
                _, distribution, _ = measures(alarms, kind)
                tool_rows.extend({"model": kind, "minimum_all_event_recall": level,
                                  "selected_threshold": chosen["threshold"],
                                  "selected_condition": chosen["condition"], **item} for item in distribution)
    write_csv(OUT / "threshold_0_5_comparison.csv", at05, list(at05[0]))
    write_csv(OUT / "validation_threshold_persistence_grid.csv", grid, list(grid[0]))
    pareto = []
    for kind in ("A", "B"):
        candidates = [item for item in grid if item["model"] == kind]
        pareto.extend(item for item in candidates if not any(
            other is not item and
            other["detected_fault_events"] >= item["detected_fault_events"] and
            other["false_alarm_events_10m"] <= item["false_alarm_events_10m"] and
            (other["detected_fault_events"] > item["detected_fault_events"] or
             other["false_alarm_events_10m"] < item["false_alarm_events_10m"])
            for other in candidates))
    write_csv(OUT / "pareto_front.csv", pareto, list(grid[0]))
    columns = list(best_by_level[0])
    for row in best_by_level:
        for key in row:
            if key not in columns:
                columns.append(key)
    write_csv(OUT / "pareto_by_recall_level.csv", best_by_level, columns)
    if tool_rows:
        write_csv(OUT / "false_alarm_by_tool_at_75pct.csv", tool_rows, list(tool_rows[0]))
    concentration = {}
    for kind in ("A", "B"):
        distribution = sorted((r for r in tool_rows if r["model"] == kind),
                              key=lambda r: r["false_alarm_events_10m"], reverse=True)
        concentration[kind] = {"top_five_tools": [r["tool_id"] for r in distribution[:5]],
                               "top_five_share": sum(r["false_alarm_events_10m"] for r in distribution[:5]) /
                                   sum(r["false_alarm_events_10m"] for r in distribution) if distribution else None,
                               "most_false_alarm_tool": distribution[0]["tool_id"] if distribution else None,
                               "most_false_alarm_events": distribution[0]["false_alarm_events_10m"] if distribution else None}
    report = {"scheme": plan["scheme"], "train_rows": training["A"]["train_rows"],
              "validation_rows": training["A"]["validation_rows"],
              "validation_fault_events": len(events),
              "test_start_seconds_by_tool": {r["tool_id"]: r["test_start_second"] for r in plan["tools"]},
              "purge_seconds": HORIZON, "train_validation_positive_event_overlap": 0,
              "test_labels_or_sensors_loaded": False, "test_scores_or_metrics_created": False,
              "historical_test_caveat": plan["historical_caveat"],
              "train_only_reference": plan["normal_reference"],
              "parameters": plan["parameters"], "ap": ap,
              "threshold_0_5": at05, "best_by_recall_level": best_by_level,
              "concentration_at_75pct_best": concentration,
              "selection_limit": "Thresholds and persistence selected only on new validation; reported validation is still a model-selection estimate, not a final test result."}
    (OUT / "experiment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"train_rows": report["train_rows"], "validation_rows": report["validation_rows"],
                      "validation_events": len(events), "ap": ap, "best": best_by_level,
                      "concentration": concentration}, ensure_ascii=False, indent=2), flush=True)


def common_policy_audit():
    """Compare A/B at identical policies, including when B has no 75% candidate."""
    output = OUT / "common_policy_false_alarm_by_tool.csv"
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(plan_path().read_text(encoding="utf-8"))
    with np.load(score_path("A")) as a, np.load(score_path("B")) as b:
        truth, time, tool = (a[k].copy() for k in ("truth", "time", "tool_code"))
        if not all(np.array_equal(a[k], b[k]) for k in ("truth", "time", "tool_code")):
            raise AssertionError("A/B validation rows differ")
        probabilities = {"A": a["probability"].copy(), "B": b["probability"].copy()}
    names = {i: row["tool_id"] for i, row in enumerate(plan["tools"])}
    reverse = {name: code for code, name in names.items()}
    events = validation_events(plan)
    ap = {kind: float(average_precision_score(truth, value)) for kind, value in probabilities.items()}
    summaries, tool_rows = [], []
    for threshold, condition, window, hits in ((.5, "1_of_1m", 1, 1), (.36, "2_of_5m", 5, 2)):
        for kind in ("A", "B"):
            alarm = causal_alarm(probabilities[kind] >= threshold, time, tool, sorted(names), window, hits)
            row, tools = row_and_alarm_metrics(.5, truth, alarm.astype(np.uint8), time, tool, ap[kind], names)
            event, _ = event_metrics(.5, events, alarm.astype(np.uint8), time, tool, reverse)
            summaries.append({"model": kind, "threshold": threshold, "condition": condition,
                              **{k: v for k, v in row.items() if k != "threshold"}, **event})
            tool_rows.extend({"model": kind, **item, "threshold": threshold,
                              "condition": condition} for item in tools)
    write_csv(OUT / "common_policy_metrics.csv", summaries, list(summaries[0]))
    write_csv(output, tool_rows, list(tool_rows[0]))
    report_path = OUT / "experiment_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["common_policy_metrics"] = summaries
    report["max_recall_in_grid"] = {kind: max(item["fault_event_recall_all_logged"] for item in
        pd.read_csv(OUT / "validation_threshold_persistence_grid.csv").query("model == @kind").to_dict("records"))
        for kind in ("A", "B")}
    report["test_access_detail"] = (
        "No test scores or metrics were created. Historical experiments already used the last segment. "
        "During this run, the shared fault-log CSV was loaded for validation filtering, and an initial "
        "pre-test batch reader fetched the first test batch before discarding it; those test rows were not "
        "used in fitting, reference estimation, threshold selection, or reporting. The batch reader is now fixed. "
        "Thus this segment cannot be called strictly untouched. A genuinely untouched final test requires later new labeled data.")
    report["strictly_untouched_test"] = False
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"common_policy": summaries, "max_recall": report["max_recall_in_grid"]},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train_a", "train_b", "evaluate", "common_policy_audit"))
    action = parser.parse_args().action
    {"prepare": prepare, "train_a": lambda: train("A"),
     "train_b": lambda: train("B"), "evaluate": evaluate,
     "common_policy_audit": common_policy_audit}[action]()
