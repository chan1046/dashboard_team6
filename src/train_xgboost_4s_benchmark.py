"""Four-second XGBoost benchmark for 30/60/180-minute fault horizons.

The script keeps the original four-second resolution, builds active-session-aware
labels, samples training rows by tool/class, and evaluates on the untouched last
20 percent of each tool. Official test files are not used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "sensor_ttf"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "xgboost_4s_benchmark"
HORIZONS_MINUTES = [30, 60, 180]
TTF_COLUMNS = [
    "TTF_FlowCool Pressure Dropped Below Limit",
    "TTF_Flowcool Pressure Too High Check Flowcool Pump",
    "TTF_Flowcool leak",
]
CLASS_NAMES = {0: "normal", 1: "pressure_low", 2: "pressure_high", 3: "leak"}
RAW_SENSORS = [
    "IONGAUGEPRESSURE",
    "ETCHBEAMVOLTAGE",
    "ETCHBEAMCURRENT",
    "ETCHSUPPRESSORVOLTAGE",
    "ETCHSUPPRESSORCURRENT",
    "FLOWCOOLFLOWRATE",
    "FLOWCOOLPRESSURE",
    "ETCHGASCHANNEL1READBACK",
    "ETCHPBNGASREADBACK",
    "FIXTURETILTANGLE",
    "ROTATIONSPEED",
    "ACTUALROTATIONANGLE",
    "FIXTURESHUTTERPOSITION",
    "ETCHSOURCEUSAGE",
    "ETCHAUXSOURCETIMER",
    "ETCHAUX2SOURCETIMER",
    "ACTUALSTEPDURATION",
]
CORE_SENSORS = [
    "FLOWCOOLFLOWRATE",
    "FLOWCOOLPRESSURE",
    "IONGAUGEPRESSURE",
    "ETCHGASCHANNEL1READBACK",
    "ETCHPBNGASREADBACK",
]
CATEGORICAL_FEATURES = ["Tool", "stage", "recipe", "recipe_step"]
NUMERIC_FEATURES = [
    *RAW_SENSORS,
    *[f"{name}_diff_4s" for name in CORE_SENSORS],
    "time_diff_seconds",
    "restart_after_stop",
    "session_age_seconds",
    "flowcool_pressure_minus_flow",
    "flowcool_pressure_div_flow",
]
FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


@dataclass
class Timeline:
    time: np.ndarray
    time_diff: np.ndarray
    restart: np.ndarray
    session_age: np.ndarray
    session_end: np.ndarray
    labels: dict[int, np.ndarray]
    event_time: np.ndarray
    event_class: np.ndarray
    cutoff_time: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="4초 원본 XGBoost 30/60/180분 고장 분류 기준 실험"
    )
    parser.add_argument(
        "action", nargs="?", choices=["prepare", "train", "evaluate", "all"], default="all"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tools", nargs="+", default=["all"])
    parser.add_argument("--horizons", default="30,60,180")
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--stop-gap-seconds", type=int, default=120)
    parser.add_argument("--negative-ratio", type=int, default=3)
    parser.add_argument("--max-train-rows", type=int, default=800_000)
    parser.add_argument("--n-estimators", type=int, default=220)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--fault-threshold", type=float, default=0.5)
    parser.add_argument("--false-alert-cooldown-minutes", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--force-prepare", action="store_true")
    return parser.parse_args()


def parse_horizons(value: str) -> list[int]:
    horizons = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not horizons or min(horizons) <= 0:
        raise ValueError("horizons에는 양의 분 단위 정수가 필요합니다.")
    return horizons


def normalize_tool(value: str) -> str:
    cleaned = value.upper().replace("-", "_")
    if not re.fullmatch(r"\d{2}_M\d{2}", cleaned):
        raise ValueError(f"잘못된 설비 이름: {value}")
    return cleaned


def discover_partitions(data_dir: Path, requested: list[str]) -> dict[str, Path]:
    available: dict[str, Path] = {}
    for path in sorted(data_dir.glob("*_sensor_ttf.csv.gz")):
        match = re.fullmatch(r"(\d{2}_M\d{2})_sensor_ttf\.csv", path.stem)
        if match:
            available[normalize_tool(match.group(1))] = path
    if not available:
        raise FileNotFoundError(f"통합 sensor+TTF 파일이 없습니다: {data_dir}")
    selected = sorted(available) if requested == ["all"] else [normalize_tool(x) for x in requested]
    missing = sorted(set(selected) - set(available))
    if missing:
        raise FileNotFoundError(f"통합 파일이 없는 설비: {missing}")
    return {tool: available[tool] for tool in selected}


def build_timeline(
    path: Path,
    horizons: list[int],
    train_ratio: float,
    stop_gap_seconds: int,
) -> Timeline:
    frame = pd.read_csv(path, usecols=["time", *TTF_COLUMNS], low_memory=False)
    time = pd.to_numeric(frame["time"], errors="raise").to_numpy(dtype=np.float64)
    n_rows = len(time)
    if n_rows < 2:
        raise ValueError(f"행이 너무 적습니다: {path}")
    time_diff = np.empty(n_rows, dtype=np.float32)
    time_diff[0] = 0
    time_diff[1:] = np.diff(time).astype(np.float32)
    restart = np.zeros(n_rows, dtype=np.int8)
    restart[0] = 1
    restart[1:] = ((time_diff[1:] > stop_gap_seconds) | (time_diff[1:] < 0)).astype(np.int8)
    starts = np.flatnonzero(restart)
    ends = np.r_[starts[1:] - 1, n_rows - 1]
    session_end = np.empty(n_rows, dtype=np.float64)
    session_age = np.empty(n_rows, dtype=np.float32)
    for start, end in zip(starts, ends, strict=True):
        session_end[start : end + 1] = time[end]
        session_age[start : end + 1] = (time[start : end + 1] - time[start]).astype(np.float32)

    ttf = frame[TTF_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    candidate_event_time = time[:, None] + ttf
    event_is_in_session = (
        np.isfinite(ttf)
        & (ttf >= 0)
        & (candidate_event_time <= session_end[:, None] + stop_gap_seconds)
    )
    valid_ttf = np.where(event_is_in_session, ttf, np.inf)
    event_class = np.argmin(valid_ttf, axis=1).astype(np.int8) + 1
    min_ttf = np.min(valid_ttf, axis=1)
    has_event = np.isfinite(min_ttf)
    event_class[~has_event] = 0
    event_time = np.where(has_event, time + min_ttf, np.nan)

    labels: dict[int, np.ndarray] = {}
    for horizon in horizons:
        horizon_seconds = horizon * 60
        label = np.full(n_rows, -1, dtype=np.int8)
        fully_observed = time + horizon_seconds <= session_end
        label[fully_observed] = 0
        positive = has_event & (min_ttf <= horizon_seconds)
        label[positive] = event_class[positive]
        labels[horizon] = label

    split_position = int(n_rows * train_ratio)
    if split_position <= 0 or split_position >= n_rows:
        raise ValueError(f"train-ratio로 분할할 수 없습니다: {path}")
    cutoff_time = float(time[split_position])
    return Timeline(
        time=time,
        time_diff=time_diff,
        restart=restart,
        session_age=session_age,
        session_end=session_end,
        labels=labels,
        event_time=event_time,
        event_class=event_class,
        cutoff_time=cutoff_time,
    )


def distribute_positive_budget(counts: dict[int, int], budget: int) -> dict[int, int]:
    allocation = {label: 0 for label in (1, 2, 3)}
    remaining = budget
    active = {label for label in allocation if counts.get(label, 0) > 0}
    while active and remaining > 0:
        share = max(1, remaining // len(active))
        progressed = False
        for label in list(active):
            available = counts[label] - allocation[label]
            take = min(available, share, remaining)
            if take > 0:
                allocation[label] += take
                remaining -= take
                progressed = True
            if allocation[label] >= counts[label]:
                active.remove(label)
            if remaining <= 0:
                break
        if not progressed:
            break
    return allocation


def sample_training_indices(
    label: np.ndarray,
    time: np.ndarray,
    cutoff_time: float,
    purge_seconds: int,
    quota: int,
    negative_ratio: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    eligible = (label >= 0) & (time < cutoff_time - purge_seconds)
    counts = {value: int(np.sum(eligible & (label == value))) for value in range(4)}
    positive_budget = max(1, quota // (negative_ratio + 1))
    allocation = distribute_positive_budget(counts, positive_budget)
    selected_parts: list[np.ndarray] = []
    for value in (1, 2, 3):
        candidates = np.flatnonzero(eligible & (label == value))
        take = allocation[value]
        if take:
            selected_parts.append(
                candidates if len(candidates) <= take else rng.choice(candidates, take, replace=False)
            )
    positives_selected = sum(len(part) for part in selected_parts)
    negative_budget = min(counts[0], max(0, quota - positives_selected))
    negative_candidates = np.flatnonzero(eligible & (label == 0))
    if negative_budget:
        selected_parts.append(
            negative_candidates
            if len(negative_candidates) <= negative_budget
            else rng.choice(negative_candidates, negative_budget, replace=False)
        )
    selected = np.sort(np.concatenate(selected_parts)) if selected_parts else np.array([], dtype=np.int64)
    selected_counts = {value: int(np.sum(label[selected] == value)) for value in range(4)}
    return selected, {"available": counts, "selected": selected_counts}


def add_features(
    chunk: pd.DataFrame,
    timeline: Timeline,
    start: int,
    previous_values: dict[str, float],
    stop_gap_seconds: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    end = start + len(chunk)
    chunk = chunk.reset_index(drop=True).copy()
    observed_time = pd.to_numeric(chunk["time"], errors="raise").to_numpy(dtype=np.float64)
    if not np.array_equal(observed_time, timeline.time[start:end]):
        raise ValueError(f"특징 chunk의 time이 timeline과 다릅니다: row {start:,}")
    chunk["time_diff_seconds"] = timeline.time_diff[start:end]
    chunk["restart_after_stop"] = timeline.restart[start:end]
    chunk["session_age_seconds"] = timeline.session_age[start:end]
    for sensor in CORE_SENSORS:
        values = pd.to_numeric(chunk[sensor], errors="coerce")
        diff = values.diff()
        if start > 0 and timeline.time_diff[start] <= stop_gap_seconds and sensor in previous_values:
            diff.iloc[0] = values.iloc[0] - previous_values[sensor]
        else:
            diff.iloc[0] = np.nan
        diff.loc[chunk["restart_after_stop"].astype(bool)] = np.nan
        chunk[f"{sensor}_diff_4s"] = diff.astype("float32")
        if len(values):
            previous_values[sensor] = float(values.iloc[-1])
    pressure = pd.to_numeric(chunk["FLOWCOOLPRESSURE"], errors="coerce")
    flow = pd.to_numeric(chunk["FLOWCOOLFLOWRATE"], errors="coerce")
    chunk["flowcool_pressure_minus_flow"] = (pressure - flow).astype("float32")
    chunk["flowcool_pressure_div_flow"] = (
        pressure / flow.where(flow.abs() > 1e-6)
    ).astype("float32")
    for column in CATEGORICAL_FEATURES:
        chunk[column] = chunk[column].astype("string")
    return chunk[FEATURES], previous_values


def source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def prepare_samples(args: argparse.Namespace, partitions: dict[str, Path], horizons: list[int]) -> dict[str, Any]:
    sample_root = args.output_dir / "samples"
    sample_root.mkdir(parents=True, exist_ok=True)
    max_horizon_seconds = max(horizons) * 60
    quota_per_tool = max(1, args.max_train_rows // len(partitions))
    report: dict[str, Any] = {"tools": {}, "horizons": horizons, "features": FEATURES}

    for tool_position, (tool, path) in enumerate(partitions.items(), start=1):
        print(f"[{tool_position}/{len(partitions)}] {tool}: preparing 4-second labels and samples")
        tool_meta_path = sample_root / f"{tool}.meta.json"
        expected_outputs = [sample_root / f"h{horizon}" / f"{tool}.csv.gz" for horizon in horizons]
        if not args.force_prepare and tool_meta_path.exists() and all(item.exists() for item in expected_outputs):
            metadata = json.loads(tool_meta_path.read_text(encoding="utf-8"))
            if metadata.get("source") == source_signature(path) and metadata.get("settings") == {
                "horizons": horizons,
                "train_ratio": args.train_ratio,
                "stop_gap_seconds": args.stop_gap_seconds,
                "negative_ratio": args.negative_ratio,
                "quota_per_tool": quota_per_tool,
            }:
                report["tools"][tool] = metadata["report"]
                print("    reusing existing samples")
                continue

        timeline = build_timeline(path, horizons, args.train_ratio, args.stop_gap_seconds)
        rng = np.random.default_rng(args.random_state + tool_position)
        selected: dict[int, np.ndarray] = {}
        counts: dict[int, dict[str, Any]] = {}
        for horizon in horizons:
            selected[horizon], counts[horizon] = sample_training_indices(
                timeline.labels[horizon],
                timeline.time,
                timeline.cutoff_time,
                max_horizon_seconds,
                quota_per_tool,
                args.negative_ratio,
                rng,
            )

        writers: dict[int, Any] = {}
        temporary_paths: dict[int, Path] = {}
        try:
            import gzip

            for horizon in horizons:
                folder = sample_root / f"h{horizon}"
                folder.mkdir(parents=True, exist_ok=True)
                output = folder / f"{tool}.csv.gz"
                temporary = output.with_name(f"{output.name}.{os.getpid()}.partial")
                if temporary.exists():
                    temporary.unlink()
                temporary_paths[horizon] = temporary
                writers[horizon] = gzip.open(temporary, "wt", encoding="utf-8", newline="", compresslevel=3)

            usecols = ["time", *RAW_SENSORS, *CATEGORICAL_FEATURES]
            row_start = 0
            previous_values: dict[str, float] = {}
            headers = {horizon: True for horizon in horizons}
            for chunk in pd.read_csv(path, usecols=usecols, chunksize=args.chunk_size, low_memory=False):
                row_end = row_start + len(chunk)
                feature_chunk, previous_values = add_features(
                    chunk, timeline, row_start, previous_values, args.stop_gap_seconds
                )
                for horizon in horizons:
                    indices = selected[horizon]
                    left = np.searchsorted(indices, row_start, side="left")
                    right = np.searchsorted(indices, row_end, side="left")
                    chosen = indices[left:right]
                    if not len(chosen):
                        continue
                    local = chosen - row_start
                    output_chunk = feature_chunk.iloc[local].copy()
                    output_chunk.insert(0, "label", timeline.labels[horizon][chosen])
                    output_chunk.to_csv(writers[horizon], index=False, header=headers[horizon])
                    headers[horizon] = False
                row_start = row_end
            if row_start != len(timeline.time):
                raise ValueError(f"{tool}: 특징 행 수가 timeline과 다릅니다.")
        finally:
            for writer in writers.values():
                writer.close()

        for horizon in horizons:
            output = sample_root / f"h{horizon}" / f"{tool}.csv.gz"
            os.replace(temporary_paths[horizon], output)

        validation_counts = {
            horizon: {
                value: int(
                    np.sum((timeline.time >= timeline.cutoff_time) & (timeline.labels[horizon] == value))
                )
                for value in range(4)
            }
            for horizon in horizons
        }
        tool_report = {
            "rows": len(timeline.time),
            "cutoff_time": timeline.cutoff_time,
            "training": {str(h): counts[h] for h in horizons},
            "validation": {str(h): validation_counts[h] for h in horizons},
        }
        report["tools"][tool] = tool_report
        metadata = {
            "source": source_signature(path),
            "settings": {
                "horizons": horizons,
                "train_ratio": args.train_ratio,
                "stop_gap_seconds": args.stop_gap_seconds,
                "negative_ratio": args.negative_ratio,
                "quota_per_tool": quota_per_tool,
            },
            "report": tool_report,
        }
        tool_meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            "    "
            + ", ".join(
                f"h{h}={sum(counts[h]['selected'].values()):,}" for h in horizons
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def make_pipeline(args: argparse.Namespace) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32),
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        tree_method="hist",
        max_bin=256,
        n_jobs=max(1, (os.cpu_count() or 2) - 1),
        random_state=args.random_state,
        eval_metric="mlogloss",
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def train_models(args: argparse.Namespace, partitions: dict[str, Path], horizons: list[int]) -> dict[str, Any]:
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    for horizon in horizons:
        paths = [args.output_dir / "samples" / f"h{horizon}" / f"{tool}.csv.gz" for tool in partitions]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"h{horizon} 학습 표본 누락: {missing[:3]}")
        print(f"h{horizon}: loading 4-second XGBoost training samples")
        frame = pd.concat((pd.read_csv(path, low_memory=False) for path in paths), ignore_index=True)
        frame = frame.sample(frac=1, random_state=args.random_state + horizon).reset_index(drop=True)
        y = frame.pop("label").astype("int8")
        class_counts = y.value_counts().sort_index().to_dict()
        print(f"    rows={len(frame):,}, classes={class_counts}")
        sample_weight = compute_sample_weight(class_weight="balanced", y=y).astype(np.float32)
        class_weights = {
            CLASS_NAMES[int(label)]: float(sample_weight[y.to_numpy() == label][0])
            for label in class_counts
        }
        pipeline = make_pipeline(args)
        pipeline.fit(frame[FEATURES], y, model__sample_weight=sample_weight)
        model_path = model_dir / f"xgb_4s_h{horizon}.joblib"
        joblib.dump(pipeline, model_path)
        report[str(horizon)] = {
            "training_rows": len(frame),
            "class_counts": {CLASS_NAMES[int(key)]: int(value) for key, value in class_counts.items()},
            "balanced_class_weights": class_weights,
            "model_path": str(model_path),
        }
        del frame, y, pipeline
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def confusion_metrics(matrix: np.ndarray) -> dict[str, Any]:
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else None
    rows = []
    for label in range(4):
        tp = matrix[label, label]
        fp = matrix[:, label].sum() - tp
        fn = matrix[label, :].sum() - tp
        precision = float(tp / (tp + fp)) if tp + fp else 0.0
        recall = float(tp / (tp + fn)) if tp + fn else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        rows.append({"label": CLASS_NAMES[label], "precision": precision, "recall": recall, "f1": f1})
    return {
        "accuracy": accuracy,
        "macro_f1_all_classes": float(np.mean([row["f1"] for row in rows])),
        "macro_f1_fault_classes": float(np.mean([row["f1"] for row in rows[1:]])),
        "macro_recall_fault_classes": float(np.mean([row["recall"] for row in rows[1:]])),
        "per_class": rows,
    }


def approximate_average_precision(pos_hist: np.ndarray, neg_hist: np.ndarray) -> float | None:
    total_positive = float(pos_hist.sum())
    if total_positive == 0:
        return None
    tp = np.cumsum(pos_hist[::-1])
    fp = np.cumsum(neg_hist[::-1])
    precision = np.divide(tp, tp + fp, out=np.ones_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = tp / total_positive
    recall_previous = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_previous) * precision))


def evaluate_models(args: argparse.Namespace, partitions: dict[str, Path], horizons: list[int]) -> dict[str, Any]:
    pipelines = {
        horizon: joblib.load(args.output_dir / "models" / f"xgb_4s_h{horizon}.joblib")
        for horizon in horizons
    }
    bins = 1000
    state: dict[int, dict[str, Any]] = {}
    for horizon in horizons:
        state[horizon] = {
            "confusion": np.zeros((4, 4), dtype=np.int64),
            "pos_hist": np.zeros((4, bins), dtype=np.int64),
            "neg_hist": np.zeros((4, bins), dtype=np.int64),
            "actual_events": {1: set(), 2: set(), 3: set()},
            "detected_events": {1: {}, 2: {}, 3: {}},
            "false_alert_episodes": {1: 0, 2: 0, 3: 0},
            "last_false_alert": {},
            "known_rows": 0,
            "unique_observed_times": 0,
        }

    usecols = ["time", *RAW_SENSORS, *CATEGORICAL_FEATURES]
    for tool_position, (tool, path) in enumerate(partitions.items(), start=1):
        print(f"[{tool_position}/{len(partitions)}] {tool}: evaluating all known rows in the last 20%")
        timeline = build_timeline(path, horizons, args.train_ratio, args.stop_gap_seconds)
        row_start = 0
        previous_values: dict[str, float] = {}
        seen_times = {horizon: set() for horizon in horizons}
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=args.chunk_size, low_memory=False):
            row_end = row_start + len(chunk)
            feature_chunk, previous_values = add_features(
                chunk, timeline, row_start, previous_values, args.stop_gap_seconds
            )
            times = timeline.time[row_start:row_end]
            for horizon in horizons:
                labels = timeline.labels[horizon][row_start:row_end]
                valid = (times >= timeline.cutoff_time) & (labels >= 0)
                if not valid.any():
                    continue
                x_valid = feature_chunk.loc[valid, FEATURES]
                probability = pipelines[horizon].predict_proba(x_valid).astype(np.float32)
                truth = labels[valid].astype(np.int8)
                valid_times = times[valid]
                fault_probability = probability[:, 1:]
                best_fault = np.argmax(fault_probability, axis=1).astype(np.int8) + 1
                best_probability = np.max(fault_probability, axis=1)
                prediction = np.where(best_probability >= args.fault_threshold, best_fault, 0).astype(np.int8)
                np.add.at(state[horizon]["confusion"], (truth, prediction), 1)
                state[horizon]["known_rows"] += len(truth)
                seen_times[horizon].update(np.asarray(valid_times, dtype=np.int64).tolist())

                for label in range(4):
                    score_bin = np.minimum((probability[:, label] * (bins - 1)).astype(np.int32), bins - 1)
                    positive = truth == label
                    state[horizon]["pos_hist"][label] += np.bincount(
                        score_bin[positive], minlength=bins
                    )
                    state[horizon]["neg_hist"][label] += np.bincount(
                        score_bin[~positive], minlength=bins
                    )

                global_indices = np.flatnonzero(valid) + row_start
                positive_rows = truth > 0
                for index, true_label, pred_label, current_time in zip(
                    global_indices[positive_rows],
                    truth[positive_rows],
                    prediction[positive_rows],
                    valid_times[positive_rows],
                    strict=True,
                ):
                    event_time = timeline.event_time[index]
                    key = (tool, int(round(event_time)))
                    state[horizon]["actual_events"][int(true_label)].add(key)
                    if pred_label == true_label:
                        lead = float(event_time - current_time)
                        current = state[horizon]["detected_events"][int(true_label)].get(key, -np.inf)
                        if lead > current:
                            state[horizon]["detected_events"][int(true_label)][key] = lead

                false_rows = (truth == 0) & (prediction > 0)
                for pred_label, current_time in zip(
                    prediction[false_rows], valid_times[false_rows], strict=True
                ):
                    key = (tool, int(pred_label))
                    previous = state[horizon]["last_false_alert"].get(key)
                    if previous is None or current_time - previous > args.false_alert_cooldown_minutes * 60:
                        state[horizon]["false_alert_episodes"][int(pred_label)] += 1
                    state[horizon]["last_false_alert"][key] = float(current_time)
            row_start = row_end
        for horizon in horizons:
            state[horizon]["unique_observed_times"] += len(seen_times[horizon])

    report: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        item = state[horizon]
        matrix = item["confusion"]
        row_metrics = confusion_metrics(matrix)
        ap = {
            CLASS_NAMES[label]: approximate_average_precision(
                item["pos_hist"][label], item["neg_hist"][label]
            )
            for label in (1, 2, 3)
        }
        event_rows = {}
        all_leads: list[float] = []
        for label in (1, 2, 3):
            actual = item["actual_events"][label]
            detected = item["detected_events"][label]
            leads = list(detected.values())
            all_leads.extend(leads)
            event_rows[CLASS_NAMES[label]] = {
                "events": len(actual),
                "detected": len(detected),
                "event_recall": len(detected) / len(actual) if actual else None,
                "median_lead_minutes": float(np.median(leads) / 60) if leads else None,
            }
        operating_hours = item["unique_observed_times"] * 4 / 3600
        false_episodes = int(sum(item["false_alert_episodes"].values()))
        report[str(horizon)] = {
            "known_validation_rows": item["known_rows"],
            "approx_operating_hours": operating_hours,
            "confusion_matrix_labels_normal_low_high_leak": matrix.tolist(),
            **row_metrics,
            "average_precision_fault_classes": ap,
            "event_metrics": event_rows,
            "event_recall_all_faults": (
                sum(row["detected"] for row in event_rows.values())
                / sum(row["events"] for row in event_rows.values())
                if sum(row["events"] for row in event_rows.values())
                else None
            ),
            "median_lead_minutes_all_faults": float(np.median(all_leads) / 60) if all_leads else None,
            "false_alert_episodes": false_episodes,
            "false_alerts_per_operating_hour": false_episodes / operating_hours if operating_hours else None,
            "fault_threshold": args.fault_threshold,
            "false_alert_cooldown_minutes": args.false_alert_cooldown_minutes,
        }
        summary_rows.append(
            {
                "horizon_minutes": horizon,
                "validation_rows": item["known_rows"],
                "accuracy": row_metrics["accuracy"],
                "fault_macro_f1": row_metrics["macro_f1_fault_classes"],
                "fault_macro_recall": row_metrics["macro_recall_fault_classes"],
                "event_recall": report[str(horizon)]["event_recall_all_faults"],
                "median_lead_minutes": report[str(horizon)]["median_lead_minutes_all_faults"],
                "false_alerts_per_hour": report[str(horizon)]["false_alerts_per_operating_hour"],
                "ap_pressure_low": ap["pressure_low"],
                "ap_pressure_high": ap["pressure_high"],
                "ap_leak": ap["leak"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig"
    )
    return report


def main() -> None:
    args = parse_args()
    horizons = parse_horizons(args.horizons)
    if not 0 < args.train_ratio < 1:
        raise ValueError("train-ratio는 0과 1 사이여야 합니다.")
    if args.chunk_size <= 0 or args.max_train_rows <= 0 or args.negative_ratio < 1:
        raise ValueError("chunk-size/max-train-rows는 양수, negative-ratio는 1 이상이어야 합니다.")
    partitions = discover_partitions(args.data_dir, args.tools)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        "tools": list(partitions),
        "tool_count": len(partitions),
        "horizons_minutes": horizons,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "train_ratio": args.train_ratio,
        "stop_gap_seconds": args.stop_gap_seconds,
        "max_train_rows_per_horizon": args.max_train_rows,
        "negative_ratio": args.negative_ratio,
        "official_test_used": False,
    }
    (args.output_dir / "configuration.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.action in {"prepare", "all"}:
        prepare_samples(args, partitions, horizons)
    if args.action in {"train", "all"}:
        train_models(args, partitions, horizons)
    if args.action in {"evaluate", "all"}:
        report = evaluate_models(args, partitions, horizons)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Completed: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
