"""PHM 2018 cooling-fault pipeline: binary risk, three independent TTF models.

The input is the 20 per-tool sensor+TTF parquet files in ../data/processed.
Every 4-second measurement contributes to a 1-minute feature row. TTF and
fault records are used only for targets/audit, never as model inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "data" / "processed"
PREPROCESS_VERSION = "2026-09-23-v2-exact-last-row"
MODEL_VERSION = "2026-09-23-v3-normalized-lstm"
FAULTS = ("pressure_low", "pressure_high", "leak")
TTF = {
    "pressure_low": "TTF_FlowCool Pressure Dropped Below Limit",
    "pressure_high": "TTF_Flowcool Pressure Too High Check Flowcool Pump",
    "leak": "TTF_Flowcool leak",
}
SENSORS = (
    "IONGAUGEPRESSURE", "ETCHBEAMVOLTAGE", "ETCHBEAMCURRENT",
    "ETCHSUPPRESSORVOLTAGE", "ETCHSUPPRESSORCURRENT", "FLOWCOOLFLOWRATE",
    "FLOWCOOLPRESSURE", "ETCHGASCHANNEL1READBACK", "ETCHPBNGASREADBACK",
    "FIXTURETILTANGLE", "ROTATIONSPEED", "ACTUALROTATIONANGLE",
    "FIXTURESHUTTERPOSITION", "ETCHSOURCEUSAGE", "ETCHAUXSOURCETIMER",
    "ETCHAUX2SOURCETIMER", "ACTUALSTEPDURATION",
)
CORE = ("FLOWCOOLPRESSURE", "FLOWCOOLFLOWRATE", "IONGAUGEPRESSURE", "ETCHGASCHANNEL1READBACK", "ETCHPBNGASREADBACK")
CATEGORIES = ("stage", "recipe", "recipe_step")
EXCLUDED_FEATURES = {"time", "minute", "Lot", "runnum", *TTF.values(), "fault_time", "fault_type", "session_id", "split", "tool_id"}
META_COLS = {"time", "minute", "session_id", "session_start", "session_end", "split", "tool_id", "raw_rows", "unique_times", "duplicate_times", "ttf_pressure_low", "ttf_pressure_high", "ttf_leak", "event_time_pressure_low", "event_time_pressure_high", "event_time_leak"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LS_team6 PHM 2018 full-data benchmark")
    p.add_argument("action", choices=("prepare", "train", "evaluate", "all", "audit"), nargs="?", default="all")
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--output", type=Path, default=ROOT / "output_calendar_12_24")
    p.add_argument("--horizons", default="43200,86400", help="seconds, comma-separated; default: 12 and 24 hours")
    p.add_argument("--label-mode", choices=("calendar", "continuous"), default="calendar", help="calendar includes real tool stops in elapsed time")
    p.add_argument("--reuse-minute-from", type=Path, default=None, help="reuse verified 1-minute features from an earlier preparation; labels and time split are recalculated")
    p.add_argument("--tools", nargs="*", default=[])
    p.add_argument("--stop-gap-seconds", type=int, default=120)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hgb-iterations", type=int, default=60)
    p.add_argument("--xgb-trees", type=int, default=120)
    p.add_argument("--lstm-epochs", type=int, default=5)
    p.add_argument("--lstm-seq-len", type=int, default=10)
    p.add_argument("--lstm-batch", type=int, default=1024)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--force-prepare", action="store_true")
    p.add_argument("--skip-lstm", action="store_true", help="for dependency/smoke diagnostics only")
    p.add_argument("--smoke", action="store_true", help="small integration run; never a final report")
    a = p.parse_args()
    a.horizons = sorted({int(x) for x in a.horizons.split(",") if x.strip()})
    if not a.horizons or min(a.horizons) <= 0:
        p.error("--horizons must contain positive seconds")
    if not 0 < a.train_ratio < 1 or a.stop_gap_seconds <= 0:
        p.error("invalid train-ratio or stop-gap-seconds")
    return a


def source_files(args: argparse.Namespace) -> list[Path]:
    files = sorted(args.source.glob("??_M??_sensor_ttf.parquet"))
    if args.tools:
        wanted = set(args.tools)
        files = [f for f in files if f.name.split("_sensor_ttf")[0] in wanted]
    if not files:
        raise FileNotFoundError(f"No sensor+TTF parquet partitions in {args.source}")
    return files


def feature_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if c not in META_COLS and not c.startswith("label_h") and not c.startswith("event_class_h") and c not in EXCLUDED_FEATURES]


def _rolling_features(frame: pd.DataFrame, column: str) -> None:
    by_session = frame.groupby("session_id", sort=False)[column]
    frame[f"{column}_diff"] = by_session.diff().astype("float32")
    seconds = frame["time_diff_seconds"].replace(0, np.nan)
    frame[f"{column}_per_hour"] = (frame[f"{column}_diff"] / seconds * 3600).astype("float32")
    for window in (5, 30):
        rolled = by_session.rolling(window, min_periods=2)
        frame[f"{column}_mean_{window}m"] = rolled.mean().reset_index(level=0, drop=True).astype("float32")
        frame[f"{column}_std_{window}m"] = rolled.std().reset_index(level=0, drop=True).astype("float32")
    before = by_session.shift(5)
    before_time = frame.groupby("session_id", sort=False)["time"].shift(5)
    frame[f"{column}_slope_5m"] = ((frame[column] - before) / ((frame["time"] - before_time) / 60)).astype("float32")
    frame[f"{column}_ewma_10m"] = by_session.transform(lambda x: x.ewm(span=10, adjust=False).mean()).astype("float32")


def _minute_frame(raw: pd.DataFrame, tool: str, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    times = raw["time"].to_numpy(dtype=np.int64, copy=False)
    if len(times) < 2 or np.any(np.diff(times) < 0):
        raise ValueError(f"{tool}: time must be nondecreasing and have at least two rows")
    restarts = np.r_[True, np.diff(times) > args.stop_gap_seconds]
    raw["session_id"] = np.cumsum(restarts, dtype=np.int32)
    raw["minute"] = times // 60
    groups = raw.groupby(["session_id", "minute"], sort=False, observed=True)
    aggs: dict[str, tuple[str, str]] = {
        "time": ("time", "max"),
        "raw_rows": ("time", "size"),
        "unique_times": ("time", "nunique"),
        "stage": ("stage", "last"),
        "recipe": ("recipe", "last"),
        "recipe_step": ("recipe_step", "last"),
    }
    for sensor in SENSORS:
        aggs[sensor] = (sensor, "mean")
    for sensor in CORE:
        aggs[f"{sensor}_std_1m"] = (sensor, "std")
        aggs[f"{sensor}_min_1m"] = (sensor, "min")
        aggs[f"{sensor}_max_1m"] = (sensor, "max")
    for fault in FAULTS:
        aggs[f"ttf_{fault}"] = (TTF[fault], "last")
    frame = groups.agg(**aggs).reset_index()
    # GroupBy.last skips nulls. TTF at the final raw timestamp can be null
    # immediately after a fault, so using the last non-null value invents a
    # new event at the minute's later timestamp. Preserve exact final rows.
    exact_last = groups.tail(1)[["session_id", "minute", *CATEGORIES, *TTF.values()]].reset_index(drop=True)
    if not frame[["session_id", "minute"]].reset_index(drop=True).equals(exact_last[["session_id", "minute"]]):
        raise AssertionError("Minute aggregation and exact-last rows are misaligned")
    for col in CATEGORIES:
        frame[col] = exact_last[col].to_numpy()
    for fault in FAULTS:
        frame[f"ttf_{fault}"] = exact_last[TTF[fault]].to_numpy()
    frame["duplicate_times"] = frame["raw_rows"] - frame["unique_times"]
    frame["tool_id"] = tool
    frame["tool_number"] = int(tool[:2]) * 2 + int(tool[-2:]) - 2
    frame["session_start"] = frame.groupby("session_id", sort=False)["time"].transform("first")
    frame["session_end"] = frame.groupby("session_id", sort=False)["time"].transform("last")
    frame["time_diff_seconds"] = frame.groupby("session_id", sort=False)["time"].diff()
    frame["session_age_minutes"] = ((frame["time"] - frame["session_start"]) / 60).astype("float32")
    frame["history_minutes"] = (frame.groupby("session_id", sort=False).cumcount() + 1).astype("float32")
    frame["restart_after_stop"] = frame["history_minutes"].eq(1).astype("int8")
    for col in CATEGORIES:
        frame[f"{col}_changed"] = frame[col].ne(frame.groupby("session_id", sort=False)[col].shift()).astype("int8")
    for col in SENSORS:
        frame[col] = frame[col].astype("float32")
    for col in CORE:
        _rolling_features(frame, col)
    frame["pressure_minus_flow"] = (frame["FLOWCOOLPRESSURE"] - frame["FLOWCOOLFLOWRATE"]).astype("float32")
    frame["pressure_div_flow"] = (frame["FLOWCOOLPRESSURE"] / frame["FLOWCOOLFLOWRATE"].where(frame["FLOWCOOLFLOWRATE"].abs() > 1e-6)).astype("float32")
    frame["time_diff_seconds"] = frame["time_diff_seconds"].fillna(0).astype("float32")
    frame = frame.replace([np.inf, -np.inf], np.nan)

    # A TTF is observed only if the corresponding fault occurs before the
    # current continuous operating session ends. Otherwise it is censored.
    for fault in FAULTS:
        ttf = frame[f"ttf_{fault}"].to_numpy(dtype=np.float64)
        event_time = frame["time"].to_numpy(dtype=np.float64) + ttf
        observed = np.isfinite(ttf) & (ttf >= 0) & (event_time <= frame["session_end"].to_numpy())
        frame[f"ttf_{fault}"] = np.where(observed, ttf, np.nan).astype("float32")
        frame[f"event_time_{fault}"] = np.where(observed, event_time, np.nan)

    ttf_matrix = frame[[f"ttf_{f}" for f in FAULTS]].to_numpy(dtype=np.float64)
    nearest = np.min(np.where(np.isfinite(ttf_matrix), ttf_matrix, np.inf), axis=1)
    first_type = np.argmin(np.where(np.isfinite(ttf_matrix), ttf_matrix, np.inf), axis=1) + 1
    current_time = frame["time"].to_numpy(dtype=np.float64)
    end_time = frame["session_end"].to_numpy(dtype=np.float64)
    for horizon in args.horizons:
        label = np.full(len(frame), -1, dtype=np.int8)
        label[current_time + horizon <= end_time] = 0
        label[nearest <= horizon] = 1
        frame[f"label_h{horizon}"] = label
        frame[f"event_class_h{horizon}"] = np.where(label == 1, first_type, 0).astype("int8")

    cutoff_index = int(len(frame) * args.train_ratio)
    cutoff_time = int(frame["time"].iloc[cutoff_index])
    # The maximum horizon purge prevents a training label from observing an
    # event in the fixed validation period.
    frame["split"] = np.where(current_time < cutoff_time - max(args.horizons), 0, np.where(current_time >= cutoff_time, 2, 1)).astype("int8")
    report = {
        "tool": tool, "raw_rows": int(len(raw)), "minute_rows": int(len(frame)),
        "duplicate_raw_time_rows": int(np.sum(np.diff(times) == 0)),
        "sessions": int(frame["session_id"].nunique()), "split_cutoff_time": cutoff_time,
        "split_rows": {s: int((frame["split"] == code).sum()) for s, code in (("train", 0), ("purge", 1), ("validation", 2))},
        "horizons": {},
    }
    for horizon in args.horizons:
        report["horizons"][str(horizon)] = {
            s: {
                "normal": int(((frame["split"] == code) & (frame[f"label_h{horizon}"] == 0)).sum()),
                "fault": int(((frame["split"] == code) & (frame[f"label_h{horizon}"] == 1)).sum()),
                "censored": int(((frame["split"] == code) & (frame[f"label_h{horizon}"] == -1)).sum()),
            }
            for s, code in (("train", 0), ("validation", 2))
        }
    return frame, report


def _relabel_prepared(frame: pd.DataFrame, prior_report: dict[str, Any], args: argparse.Namespace, fault_events: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reuse past-only minute features, but regenerate every horizon-dependent target."""
    old_targets = [c for c in frame if c.startswith("label_h") or c.startswith("event_class_h")]
    frame = frame.drop(columns=old_targets)
    current_time = frame["time"].to_numpy(dtype=np.float64)
    if args.label_mode == "calendar":
        if fault_events is None:
            raise ValueError("Calendar labels require the train fault log")
        tool = prior_report["tool"]
        tool_faults = fault_events[fault_events["tool_id"] == tool]
        for fault in FAULTS:
            times = np.sort(tool_faults.loc[tool_faults["fault_type"] == fault, "time"].drop_duplicates().to_numpy(dtype=np.float64))
            position = np.searchsorted(times, current_time, side="left")
            observed = position < len(times)
            next_event = np.full(len(frame), np.nan, dtype=np.float64)
            next_event[observed] = times[position[observed]]
            frame[f"event_time_{fault}"] = next_event
            frame[f"ttf_{fault}"] = (next_event - current_time).astype("float32")
    ttf_matrix = frame[[f"ttf_{f}" for f in FAULTS]].to_numpy(dtype=np.float64)
    finite = np.where(np.isfinite(ttf_matrix), ttf_matrix, np.inf)
    nearest = np.min(finite, axis=1)
    first_type = np.argmin(finite, axis=1) + 1
    end_time = frame["session_end"].to_numpy(dtype=np.float64) if args.label_mode == "continuous" else np.full(len(frame), current_time[-1], dtype=np.float64)
    cutoff_index = int(len(frame) * args.train_ratio)
    cutoff_time = int(frame["time"].iloc[cutoff_index])
    frame["split"] = np.where(current_time < cutoff_time - max(args.horizons), 0, np.where(current_time >= cutoff_time, 2, 1)).astype("int8")
    horizons = {}
    for horizon in args.horizons:
        label = np.full(len(frame), -1, dtype=np.int8)
        label[current_time + horizon <= end_time] = 0
        label[nearest <= horizon] = 1
        frame[f"label_h{horizon}"] = label
        frame[f"event_class_h{horizon}"] = np.where(label == 1, first_type, 0).astype("int8")
        horizons[str(horizon)] = {
            s: {
                "normal": int(((frame["split"] == code) & (label == 0)).sum()),
                "fault": int(((frame["split"] == code) & (label == 1)).sum()),
                "censored": int(((frame["split"] == code) & (label == -1)).sum()),
            }
            for s, code in (("train", 0), ("validation", 2))
        }
    report = {k: prior_report[k] for k in ("tool", "raw_rows", "minute_rows", "duplicate_raw_time_rows", "sessions")}
    report["split_cutoff_time"] = cutoff_time
    report["split_rows"] = {s: int((frame["split"] == code).sum()) for s, code in (("train", 0), ("purge", 1), ("validation", 2))}
    report["horizons"] = horizons
    report["label_mode"] = args.label_mode
    return frame, report


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    files = source_files(args)
    out = args.output / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"resolution": "1-minute rows from 100% of original 4-second measurements", "horizons_seconds": args.horizons, "label_mode": args.label_mode, "stop_gap_seconds": args.stop_gap_seconds, "train_ratio": args.train_ratio, "tools": {}}
    fault_path = args.source / "fault_events.csv"
    if args.label_mode == "calendar" and not fault_path.exists():
        raise FileNotFoundError(f"Calendar labels require {fault_path}")
    fault_events = pd.read_csv(fault_path, usecols=["tool_id", "fault_type", "time"]) if args.label_mode == "calendar" else None
    if args.reuse_minute_from is not None:
        report["reused_one_minute_features_from"] = str(args.reuse_minute_from.resolve())
    feature_list: list[str] | None = None
    columns = ["time", *SENSORS, *CATEGORIES, *TTF.values()]
    for n, source in enumerate(files, 1):
        tool = source.name.split("_sensor_ttf")[0]
        dest = out / f"{tool}.parquet"
        meta = out / f"{tool}.json"
        signature = {"pipeline_version": PREPROCESS_VERSION, "bytes": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns, "horizons": args.horizons, "label_mode": args.label_mode, "fault_bytes": fault_path.stat().st_size if fault_events is not None else None, "fault_mtime_ns": fault_path.stat().st_mtime_ns if fault_events is not None else None, "gap": args.stop_gap_seconds, "train_ratio": args.train_ratio}
        if dest.exists() and meta.exists() and not args.force_prepare:
            prior = json.loads(meta.read_text(encoding="utf-8"))
            if prior.get("signature") == signature:
                report["tools"][tool] = prior["report"]
                feature_list = prior["features"]
                print(f"[{n}/{len(files)}] {tool}: cached", flush=True)
                continue
        if args.reuse_minute_from is not None:
            prior_dir = args.reuse_minute_from.resolve()
            prior_file = prior_dir / f"{tool}.parquet"
            prior_meta = prior_dir / f"{tool}.json"
            prior = json.loads(prior_meta.read_text(encoding="utf-8"))
            old_signature = prior["signature"]
            if old_signature["pipeline_version"] != PREPROCESS_VERSION or old_signature["bytes"] != source.stat().st_size or old_signature["mtime_ns"] != source.stat().st_mtime_ns or old_signature["gap"] != args.stop_gap_seconds or old_signature["train_ratio"] != args.train_ratio:
                raise ValueError(f"{tool}: source or preprocessing settings differ; cannot reuse minute features")
            print(f"[{n}/{len(files)}] {tool}: reusing minute features; recalculating labels and split", flush=True)
            frame, tool_report = _relabel_prepared(pd.read_parquet(prior_file), prior["report"], args, fault_events)
        else:
            print(f"[{n}/{len(files)}] {tool}: aggregating every 4-second row", flush=True)
            raw = pd.read_parquet(source, columns=columns)
            frame, tool_report = _minute_frame(raw, tool, args)
            del raw
            if args.label_mode == "calendar":
                frame, tool_report = _relabel_prepared(frame, tool_report, args, fault_events)
        feature_list = feature_columns(list(frame.columns))
        # Model inputs are explicitly checked against known future/identifier columns.
        if any("ttf" in c.lower() or "event_time" in c.lower() for c in feature_list):
            raise AssertionError("Future TTF/event leaked into features")
        for col in feature_list:
            if frame[col].dtype.kind not in "biuf":
                raise TypeError(f"Feature {col} must be numeric")
            frame[col] = frame[col].astype("float32")
        temp = dest.with_suffix(".partial.parquet")
        frame.to_parquet(temp, index=False, compression="zstd")
        os.replace(temp, dest)
        meta.write_text(json.dumps({"signature": signature, "report": tool_report, "features": feature_list}, ensure_ascii=False, indent=2), encoding="utf-8")
        report["tools"][tool] = tool_report
        print(f"    raw={tool_report['raw_rows']:,}; minute={tool_report['minute_rows']:,}; sessions={tool_report['sessions']:,}", flush=True)
        del frame
    if feature_list is None:
        raise RuntimeError("No features generated")
    report["features"] = feature_list
    report["feature_count"] = len(feature_list)
    report["raw_rows_total"] = sum(x["raw_rows"] for x in report["tools"].values())
    report["minute_rows_total"] = sum(x["minute_rows"] for x in report["tools"].values())
    (args.output / "preparation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def load_prepared(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    manifest_path = args.output / "preparation_report.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run 'prepare' first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["horizons_seconds"] != args.horizons:
        raise ValueError("Requested horizons differ from prepared horizons")
    if manifest.get("label_mode", "continuous") != args.label_mode:
        raise ValueError("Requested label mode differs from prepared label mode")
    features = manifest["features"]
    needed = [*features, "time", "session_id", "split", "tool_id", *[f"ttf_{f}" for f in FAULTS], *[f"event_time_{f}" for f in FAULTS], *[f"label_h{h}" for h in args.horizons], *[f"event_class_h{h}" for h in args.horizons]]
    files = [args.output / "prepared" / f"{tool}.parquet" for tool in manifest["tools"]]
    if any(not p.exists() for p in files):
        raise FileNotFoundError("One or more prepared tool files are missing")
    parts = []
    for n, file in enumerate(files, 1):
        print(f"Loading prepared {n}/{len(files)}: {file.stem}", flush=True)
        parts.append(pd.read_parquet(file, columns=needed))
    frame = pd.concat(parts, ignore_index=True)
    return frame, features


def _sample_1pct(indices: np.ndarray, label: np.ndarray | None, seed: int) -> np.ndarray:
    """Exactly about 1% of eligible rows; stratified for binary classification."""
    rng = np.random.default_rng(seed)
    if label is None:
        take = max(1, int(round(len(indices) * 0.01)))
        return np.sort(rng.choice(indices, size=take, replace=False))
    chosen = []
    for cls in (0, 1):
        pool = indices[label[indices] == cls]
        if len(pool):
            take = max(1, int(round(len(pool) * 0.01)))
            chosen.append(rng.choice(pool, size=min(take, len(pool)), replace=False))
    return np.sort(np.concatenate(chosen)) if chosen else np.array([], dtype=np.int64)


def _atomic_joblib(model: Any, path: Path) -> None:
    temp = path.with_name(path.name + ".partial")
    joblib.dump(model, temp, compress=3)
    os.replace(temp, path)


def _model_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {"model_version": MODEL_VERSION, "horizons": args.horizons, "label_mode": args.label_mode, "hgb_iterations": args.hgb_iterations, "xgb_trees": args.xgb_trees, "lstm_epochs": args.lstm_epochs, "lstm_seq_len": args.lstm_seq_len, "lstm_batch": args.lstm_batch, "seed": args.seed, "tools": [p.name for p in source_files(args)]}


def _save_report(args: argparse.Namespace, report: dict[str, Any]) -> None:
    dest = args.output / "training_report.json"
    temp = dest.with_suffix(".partial.json")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, dest)


def _regression_train_boundary(args: argparse.Namespace, frame: pd.DataFrame) -> np.ndarray:
    prepared = json.loads((args.output / "preparation_report.json").read_text(encoding="utf-8"))
    cutoff = {tool: info["split_cutoff_time"] for tool, info in prepared["tools"].items()}
    return frame["tool_id"].map(cutoff).to_numpy(dtype=np.float64)


def _train_lstm_models(args: argparse.Namespace, frame: pd.DataFrame, features: list[str], report: dict[str, Any]) -> None:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for LSTM. Install requirements.txt") from exc

    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    x = frame[features].to_numpy(dtype=np.float32, copy=False)
    train_base = frame["split"].to_numpy() == 0
    train_boundary = _regression_train_boundary(args, frame)
    # Fit preprocessing on training rows only. Validation distribution never
    # determines the imputation value or normalization scale.
    train_x = x[train_base]
    means = np.nanmean(train_x, axis=0).astype(np.float32)
    stds = np.nanstd(train_x, axis=0).astype(np.float32)
    means[~np.isfinite(means)] = 0
    stds[(~np.isfinite(stds)) | (stds < 1e-6)] = 1
    del train_x
    normalized = np.nan_to_num((x - means) / stds, nan=0, posinf=0, neginf=0).astype(np.float32)
    np.clip(normalized, -10, 10, out=normalized)
    tool = frame["tool_id"].to_numpy()
    session = frame["session_id"].to_numpy()
    boundary = np.r_[True, (tool[1:] != tool[:-1]) | (session[1:] != session[:-1])]
    starts = np.maximum.accumulate(np.where(boundary, np.arange(len(frame)), 0)).astype(np.int64)
    offsets = np.arange(args.lstm_seq_len - 1, -1, -1, dtype=np.int64)

    class Net(nn.Module):
        def __init__(self, n_inputs: int):
            super().__init__()
            self.rnn = nn.LSTM(n_inputs, 24, batch_first=True)
            self.head = nn.Linear(24, 1)

        def forward(self, z: Any) -> Any:
            output, _ = self.rnn(z)
            return self.head(output[:, -1, :]).squeeze(-1)

    def batches(index: np.ndarray, batch_size: int, shuffle: bool, seed: int):
        order = np.random.default_rng(seed).permutation(index) if shuffle else index
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            seq_index = np.maximum(selected[:, None] - offsets[None, :], starts[selected, None])
            yield selected, torch.from_numpy(normalized[seq_index])

    model_dir = args.output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = model_dir / "lstm_scaler.npz"
    np.savez_compressed(scaler_path, mean=means, std=stds, features=np.asarray(features))
    report["lstm_scaler"] = str(scaler_path)
    _save_report(args, report)
    for fault in FAULTS:
        key = f"lstm_100_{fault}"
        path = model_dir / f"{key}.pt"
        if report["models"].get(key, {}).get("path") == str(path) and path.exists():
            print(f"{key}: cached", flush=True)
            continue
        target = frame[f"ttf_{fault}"].to_numpy(dtype=np.float32)
        event_time = frame[f"event_time_{fault}"].to_numpy(dtype=np.float64)
        eligible = np.flatnonzero(train_base & np.isfinite(target) & (target >= 0) & (event_time < train_boundary))
        if args.smoke and len(eligible) > 5000:
            eligible = eligible[:5000]
        if len(eligible) < 10:
            raise ValueError(f"{fault}: fewer than 10 uncensored training targets")
        target_log = np.log1p(target)
        target_mean = float(np.mean(target_log[eligible]))
        target_std = float(np.std(target_log[eligible]))
        if target_std < 1e-6:
            target_std = 1.0
        target_normalized = ((target_log - target_mean) / target_std).astype(np.float32)
        net = Net(len(features))
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
        loss_fn = nn.SmoothL1Loss()
        loss_history: list[float] = []
        start_at = time.monotonic()
        print(f"{key}: training all {len(eligible):,} observed minute rows", flush=True)
        net.train()
        for epoch in range(args.lstm_epochs):
            losses = []
            for selected, sequence in batches(eligible, args.lstm_batch, True, args.seed + epoch):
                prediction = net(sequence)
                answer = torch.from_numpy(target_normalized[selected])
                loss = loss_fn(prediction, answer)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            loss_history.append(float(np.mean(losses)))
            print(f"    epoch {epoch + 1}/{args.lstm_epochs} loss={loss_history[-1]:.4f}", flush=True)
        temp = path.with_suffix(".partial.pt")
        torch.save({"state_dict": net.state_dict(), "input_size": len(features), "hidden_size": 24, "seq_len": args.lstm_seq_len, "target_log_mean": target_mean, "target_log_std": target_std}, temp)
        os.replace(temp, path)
        report["models"][key] = {"path": str(path), "training_rows": int(len(eligible)), "epochs": args.lstm_epochs, "train_loss_normalized_log1p": loss_history, "seconds": round(time.monotonic() - start_at, 1)}
        _save_report(args, report)


def train(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    model_dir = args.output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    params = _model_parameters(args)
    report_path = args.output / "training_report.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        report = previous if previous.get("parameters") == params else {"parameters": params, "models": {}}
    else:
        report = {"parameters": params, "models": {}}
    frame, features = load_prepared(args)
    x = frame[features].to_numpy(dtype=np.float32, copy=False)
    split = frame["split"].to_numpy()
    train_boundary = _regression_train_boundary(args, frame)
    for horizon in args.horizons:
        label = frame[f"label_h{horizon}"].to_numpy(dtype=np.int8)
        full_indices = np.flatnonzero((split == 0) & (label >= 0))
        if args.smoke and len(full_indices) > 5000:
            full_indices = full_indices[:5000]
        small_indices = _sample_1pct(full_indices, label, args.seed + horizon)
        for family, ratio, indices in (("hgb", "1pct", small_indices), ("hgb", "100pct", full_indices), ("xgb", "100pct", full_indices)):
            key = f"binary_{family}_{ratio}_h{horizon}"
            path = model_dir / f"{key}.joblib"
            if report["models"].get(key, {}).get("path") == str(path) and path.exists():
                print(f"{key}: cached", flush=True)
                continue
            y = label[indices]
            counts = np.bincount(y, minlength=2)
            if np.any(counts == 0):
                raise ValueError(f"{key}: both normal and fault labels are required; counts={counts.tolist()}")
            weights = np.where(y == 1, len(y) / (2 * counts[1]), len(y) / (2 * counts[0])).astype(np.float32)
            if family == "hgb":
                model = HistGradientBoostingClassifier(max_iter=args.hgb_iterations, max_leaf_nodes=15, max_bins=64, learning_rate=0.08, l2_regularization=2, early_stopping=False, random_state=args.seed)
                fit_args = {"sample_weight": weights}
            else:
                model = XGBClassifier(n_estimators=args.xgb_trees, max_depth=5, learning_rate=0.07, subsample=0.85, colsample_bytree=0.85, tree_method="hist", n_jobs=min(12, os.cpu_count() or 4), scale_pos_weight=float(counts[0] / counts[1]), eval_metric="logloss", random_state=args.seed)
                fit_args = {}
            start_at = time.monotonic()
            print(f"{key}: training {len(indices):,} rows, positives={counts[1]:,}", flush=True)
            model.fit(x[indices], y, **fit_args)
            _atomic_joblib(model, path)
            report["models"][key] = {"path": str(path), "training_rows": int(len(indices)), "normal_rows": int(counts[0]), "fault_rows": int(counts[1]), "seconds": round(time.monotonic() - start_at, 1)}
            _save_report(args, report)
            del model

    for fault in FAULTS:
        target = frame[f"ttf_{fault}"].to_numpy(dtype=np.float32)
        event_time = frame[f"event_time_{fault}"].to_numpy(dtype=np.float64)
        full_indices = np.flatnonzero((split == 0) & np.isfinite(target) & (target >= 0) & (event_time < train_boundary))
        if args.smoke and len(full_indices) > 5000:
            full_indices = full_indices[:5000]
        small_indices = _sample_1pct(full_indices, None, args.seed + FAULTS.index(fault))
        for ratio, indices in (("1pct", small_indices), ("100pct", full_indices)):
            key = f"ttf_hgb_log_{ratio}_{fault}"
            path = model_dir / f"{key}.joblib"
            if report["models"].get(key, {}).get("path") == str(path) and path.exists():
                print(f"{key}: cached", flush=True)
                continue
            if len(indices) < (1 if args.smoke else 10):
                raise ValueError(f"{key}: too few uncensored targets: {len(indices)}")
            model = HistGradientBoostingRegressor(max_iter=args.hgb_iterations, max_leaf_nodes=15, max_bins=64, learning_rate=0.08, l2_regularization=2, early_stopping=False, random_state=args.seed)
            start_at = time.monotonic()
            print(f"{key}: training {len(indices):,} observed rows", flush=True)
            model.fit(x[indices], np.log1p(target[indices]))
            _atomic_joblib(model, path)
            report["models"][key] = {"path": str(path), "training_rows": int(len(indices)), "seconds": round(time.monotonic() - start_at, 1)}
            _save_report(args, report)
            del model
    if not args.skip_lstm:
        _train_lstm_models(args, frame, features, report)
    report["features"] = features
    report["feature_count"] = len(features)
    report["training_resolution"] = "1-minute feature rows; every source 4-second row aggregated"
    _save_report(args, report)
    return report


def _regression_metrics(truth: np.ndarray, predicted: np.ndarray, event_key: np.ndarray) -> dict[str, Any]:
    if not len(truth):
        return {"rows": 0, "events": 0, "mae_seconds": None, "rmse_seconds": None, "r2": None, "median_absolute_error_seconds": None, "event_mean_mae_seconds": None, **{f"mae_when_actual_ttf_le_{h}_seconds": None for h in (3600, 43200, 86400)}, **{f"rows_when_actual_ttf_le_{h}": 0 for h in (3600, 43200, 86400)}}
    errors = np.abs(truth - predicted)
    by_event = pd.DataFrame({"event": event_key, "error": errors}).groupby("event", sort=False)["error"].mean()
    result = {
        "rows": int(len(truth)), "events": int(len(by_event)),
        "mae_seconds": float(mean_absolute_error(truth, predicted)),
        "rmse_seconds": float(math.sqrt(mean_squared_error(truth, predicted))),
        "r2": float(r2_score(truth, predicted)) if len(truth) > 1 else None,
        "median_absolute_error_seconds": float(np.median(errors)),
        "event_mean_mae_seconds": float(by_event.mean()),
    }
    for horizon in (3600, 43200, 86400):
        near = truth <= horizon
        result[f"mae_when_actual_ttf_le_{horizon}_seconds"] = float(np.mean(errors[near])) if near.any() else None
        result[f"rows_when_actual_ttf_le_{horizon}"] = int(near.sum())
    return result


def _regression_event_rows(model: str, fault: str, tool_ids: np.ndarray, event_times: np.ndarray, truth: np.ndarray, predicted: np.ndarray) -> list[dict[str, Any]]:
    """Keep a separate, auditable TTF error for each real logged event."""
    errors = np.abs(truth - predicted)
    grouped = pd.DataFrame({
        "tool_id": tool_ids, "event_time": event_times.astype(np.int64),
        "actual_ttf_seconds": truth, "predicted_ttf_seconds": predicted,
        "absolute_error_seconds": errors,
        "error_last_12h": np.where(truth <= 43200, errors, np.nan),
        "error_last_24h": np.where(truth <= 86400, errors, np.nan),
        "rows_last_12h": truth <= 43200,
        "rows_last_24h": truth <= 86400,
    }).groupby(["tool_id", "event_time"], sort=False).agg(
        observed_rows=("absolute_error_seconds", "size"),
        mean_actual_ttf_seconds=("actual_ttf_seconds", "mean"),
        mean_predicted_ttf_seconds=("predicted_ttf_seconds", "mean"),
        mae_seconds=("absolute_error_seconds", "mean"),
        mae_last_12h_seconds=("error_last_12h", "mean"),
        mae_last_24h_seconds=("error_last_24h", "mean"),
        rows_last_12h=("rows_last_12h", "sum"),
        rows_last_24h=("rows_last_24h", "sum"),
    ).reset_index()
    grouped.insert(0, "fault_type", fault)
    grouped.insert(0, "model", model)
    return grouped.to_dict("records")


def _binary_event_metrics(frame: pd.DataFrame, indices: np.ndarray, truth: np.ndarray, pred: np.ndarray, horizon: int, actual_type: np.ndarray | None = None, selected_type: np.ndarray | None = None) -> dict[str, Any]:
    tool = frame["tool_id"].to_numpy()[indices]
    row_time = frame["time"].to_numpy(dtype=np.float64)[indices]
    event_time_cols = frame[[f"event_time_{f}" for f in FAULTS]].to_numpy(dtype=np.float64)[indices]
    actual_events: set[tuple[str, int, int]] = set()
    available_lead: dict[tuple[str, int, int], float] = {}
    detected: dict[tuple[str, int, int], float] = {}
    selected_correct: dict[tuple[str, int, int], float] = {}
    actual_by_type = {k: set() for k in (1, 2, 3)}
    detected_by_type = {k: set() for k in (1, 2, 3)}
    selected_by_type = {k: set() for k in (1, 2, 3)}
    types = actual_type if actual_type is not None else np.zeros(len(indices), dtype=np.int8)
    for k in np.flatnonzero(truth == 1):
        fault_index = int(types[k]) - 1
        if fault_index < 0 or fault_index > 2:
            continue
        when = event_time_cols[k, fault_index]
        if not np.isfinite(when):
            continue
        key = (str(tool[k]), fault_index + 1, int(round(when)))
        actual_events.add(key)
        lead = float(when - row_time[k])
        available_lead[key] = max(available_lead.get(key, -np.inf), lead)
        actual_by_type[fault_index + 1].add(key)
        if pred[k] == 1:
            detected[key] = max(detected.get(key, -np.inf), lead)
            detected_by_type[fault_index + 1].add(key)
            if selected_type is not None and selected_type[k] == fault_index + 1:
                selected_correct[key] = max(selected_correct.get(key, -np.inf), lead)
                selected_by_type[fault_index + 1].add(key)
    false_rows = np.flatnonzero((truth == 0) & (pred == 1))
    episodes = 0
    last_false_by_tool: dict[str, float] = {}
    for k in false_rows:
        t = str(tool[k])
        previous = last_false_by_tool.get(t)
        if previous is None or row_time[k] - previous > 600:
            episodes += 1
        last_false_by_tool[t] = row_time[k]
    operating_hours = len(indices) / 60.0
    return {
        "events": len(actual_events), "detected_events": len(detected),
        "event_recall": len(detected) / len(actual_events) if actual_events else None,
        "median_lead_minutes": float(np.median(list(detected.values())) / 60) if detected else None,
        "median_available_lead_minutes": float(np.median(list(available_lead.values())) / 60) if available_lead else None,
        "events_with_observation_lead_ge_60m": sum(v >= 3600 for v in available_lead.values()),
        "events_with_observation_lead_ge_180m": sum(v >= 10800 for v in available_lead.values()),
        "events_with_observation_lead_ge_360m": sum(v >= 21600 for v in available_lead.values()),
        "events_with_observation_lead_ge_720m": sum(v >= 43200 for v in available_lead.values()),
        "events_with_observation_lead_ge_1440m": sum(v >= 86400 for v in available_lead.values()),
        "detected_events_lead_ge_60m": sum(v >= 3600 for v in detected.values()),
        "detected_events_lead_ge_180m": sum(v >= 10800 for v in detected.values()),
        "detected_events_lead_ge_360m": sum(v >= 21600 for v in detected.values()),
        "detected_events_lead_ge_720m": sum(v >= 43200 for v in detected.values()),
        "detected_events_lead_ge_1440m": sum(v >= 86400 for v in detected.values()),
        "type_correct_events": len(selected_correct) if selected_type is not None else None,
        "type_correct_event_recall": len(selected_correct) / len(actual_events) if selected_type is not None and actual_events else None,
        "event_by_type": {
            fault: {
                "events": len(actual_by_type[k]),
                "detected": len(detected_by_type[k]),
                "type_correct": len(selected_by_type[k]) if selected_type is not None else None,
                "type_correct_recall": len(selected_by_type[k]) / len(actual_by_type[k]) if selected_type is not None and actual_by_type[k] else None,
            }
            for k, fault in enumerate(FAULTS, 1)
        },
        "false_alert_episodes_10m_cooldown": episodes,
        "false_alerts_per_observed_hour": episodes / operating_hours if operating_hours else None,
    }


def _calendar_logged_event_metrics(frame: pd.DataFrame, indices: np.ndarray, pred: np.ndarray, horizon: int, logged_events: pd.DataFrame, selected_type: np.ndarray | None = None) -> dict[str, Any]:
    """Count each logged fault once, including faults after a genuine tool stop.

    A fault-time row is excluded: a warning must precede the event. Events with
    no sensor snapshot in the preceding horizon remain in the all-log
    denominator but not the evaluable denominator.
    """
    tool = frame["tool_id"].to_numpy()[indices]
    row_time = frame["time"].to_numpy(dtype=np.float64)[indices]
    positions_by_tool = {name: np.flatnonzero(tool == name) for name in np.unique(tool)}
    available_leads: list[float] = []
    detected_leads: list[float] = []
    typed_leads: list[float] = []
    by_type = {fault: {"events": 0, "detected": 0, "type_correct": 0} for fault in FAULTS}
    for row in logged_events.itertuples(index=False):
        fault = str(row.fault_type)
        if fault not in by_type:
            continue
        positions = positions_by_tool.get(str(row.tool_id))
        if positions is None or len(positions) == 0:
            continue
        times = row_time[positions]
        when = float(row.time)
        left = int(np.searchsorted(times, when - horizon, side="left"))
        right = int(np.searchsorted(times, when, side="left"))
        if left >= right:
            continue
        candidates = positions[left:right]
        available_leads.append(when - float(row_time[candidates[0]]))
        by_type[fault]["events"] += 1
        alerts = candidates[pred[candidates] == 1]
        if len(alerts) == 0:
            continue
        detected_leads.append(when - float(row_time[alerts[0]]))
        by_type[fault]["detected"] += 1
        if selected_type is not None:
            wanted = FAULTS.index(fault) + 1
            typed = alerts[selected_type[alerts] == wanted]
            if len(typed):
                typed_leads.append(when - float(row_time[typed[0]]))
                by_type[fault]["type_correct"] += 1
    evaluable = len(available_leads)
    detected = len(detected_leads)
    typed = len(typed_leads)
    result: dict[str, Any] = {
        "logged_events": int(len(logged_events)),
        "events": evaluable,
        "events_without_prior_sensor_in_horizon": int(len(logged_events) - evaluable),
        "detected_events": detected,
        "event_recall": detected / evaluable if evaluable else None,
        "median_lead_minutes": float(np.median(detected_leads) / 60) if detected_leads else None,
        "median_available_lead_minutes": float(np.median(available_leads) / 60) if available_leads else None,
        "type_correct_events": typed if selected_type is not None else None,
        "type_correct_event_recall": typed / evaluable if selected_type is not None and evaluable else None,
        "event_by_type": {
            fault: {
                **values,
                "type_correct": values["type_correct"] if selected_type is not None else None,
                "type_correct_recall": values["type_correct"] / values["events"] if selected_type is not None and values["events"] else None,
            }
            for fault, values in by_type.items()
        },
    }
    for minutes in (60, 180, 360, 720, 1440):
        seconds = minutes * 60
        result[f"events_with_observation_lead_ge_{minutes}m"] = sum(value >= seconds for value in available_leads)
        result[f"detected_events_lead_ge_{minutes}m"] = sum(value >= seconds for value in detected_leads)
    return result


def _calendar_event_detail_rows(frame: pd.DataFrame, indices: np.ndarray, pred: np.ndarray, horizon: int, logged_events: pd.DataFrame, model_key: str, selected_type: np.ndarray | None = None) -> list[dict[str, Any]]:
    tool = frame["tool_id"].to_numpy()[indices]
    time_values = frame["time"].to_numpy(dtype=np.float64)[indices]
    positions_by_tool = {name: np.flatnonzero(tool == name) for name in np.unique(tool)}
    details: list[dict[str, Any]] = []
    for row in logged_events.itertuples(index=False):
        positions = positions_by_tool.get(str(row.tool_id), np.empty(0, dtype=np.int64))
        times = time_values[positions]
        fault_time = float(row.time)
        left = int(np.searchsorted(times, fault_time - horizon, side="left"))
        right = int(np.searchsorted(times, fault_time, side="left"))
        candidates = positions[left:right]
        alerts = candidates[pred[candidates] == 1]
        first = float(time_values[alerts[0]]) if len(alerts) else None
        item: dict[str, Any] = {
            "model": model_key, "horizon_seconds": horizon,
            "tool_id": str(row.tool_id), "fault_type": str(row.fault_type), "fault_time": int(row.time),
            "sensor_rows_in_pre_fault_window": int(len(candidates)),
            "evaluable": bool(len(candidates)), "detected_before_fault": bool(len(alerts)),
            "first_alert_time": first,
            "first_alert_lead_minutes": (fault_time - first) / 60 if first is not None else None,
        }
        if selected_type is not None:
            correct_code = FAULTS.index(str(row.fault_type)) + 1
            typed = alerts[selected_type[alerts] == correct_code]
            type_time = float(time_values[typed[0]]) if len(typed) else None
            item["type_correct_before_fault"] = bool(len(typed))
            item["first_type_correct_alert_time"] = type_time
            item["first_type_correct_alert_lead_minutes"] = (fault_time - type_time) / 60 if type_time is not None else None
        details.append(item)
    return details


def _fault_log_audit(args: argparse.Namespace, frame: pd.DataFrame) -> dict[str, Any]:
    log_path = args.source / "fault_events.csv"
    if not log_path.exists():
        return {"available": False, "reason": str(log_path)}
    faults = pd.read_csv(log_path)
    relevant_tools = set(frame["tool_id"].unique())
    faults = faults[faults["tool_id"].isin(relevant_tools)]
    prepared = json.loads((args.output / "preparation_report.json").read_text(encoding="utf-8"))
    cutoff = {tool: info["split_cutoff_time"] for tool, info in prepared["tools"].items()}
    faults["is_validation"] = faults["time"] >= faults["tool_id"].map(cutoff)
    report: dict[str, Any] = {"available": True, "fault_log_rows": int(len(faults)), "types": {}}
    for fault in FAULTS:
        expected = set(zip(faults.loc[faults["fault_type"] == fault, "tool_id"].astype(str), faults.loc[faults["fault_type"] == fault, "time"].astype(int)))
        expected_validation = set(zip(faults.loc[(faults["fault_type"] == fault) & faults["is_validation"], "tool_id"].astype(str), faults.loc[(faults["fault_type"] == fault) & faults["is_validation"], "time"].astype(int)))
        event_times = frame[f"event_time_{fault}"].to_numpy(dtype=float)
        finite = np.isfinite(event_times)
        tool_ids = frame["tool_id"].to_numpy()
        observed = {(str(tool_ids[i]), int(round(event_times[i]))) for i in np.flatnonzero(finite)}
        report["types"][fault] = {"fault_log_events": len(expected), "validation_fault_log_events": len(expected_validation), "ttf_confirmed_in_active_session_events": len(observed), "validation_confirmed_active_events": len(expected_validation & observed), "matched_events": len(expected & observed), "log_events_without_confirmed_sensor_row": len(expected - observed), "ttf_event_times_not_in_fault_log": len(observed - expected)}
    report["validation_unique_events_total"] = sum(row["validation_fault_log_events"] for row in report["types"].values())
    return report


def _lstm_predictor(args: argparse.Namespace, frame: pd.DataFrame, features: list[str]):
    import torch
    from torch import nn

    torch.set_num_threads(min(8, os.cpu_count() or 4))
    data = np.load(args.output / "models" / "lstm_scaler.npz")
    expected = data["features"].tolist()
    if expected != features:
        raise ValueError("LSTM scaler feature order differs from prepared data")
    x = frame[features].to_numpy(dtype=np.float32, copy=False)
    z = np.nan_to_num((x - data["mean"]) / data["std"], nan=0, posinf=0, neginf=0).astype(np.float32)
    np.clip(z, -10, 10, out=z)
    tool = frame["tool_id"].to_numpy()
    session = frame["session_id"].to_numpy()
    boundary = np.r_[True, (tool[1:] != tool[:-1]) | (session[1:] != session[:-1])]
    starts = np.maximum.accumulate(np.where(boundary, np.arange(len(frame)), 0)).astype(np.int64)
    offsets = np.arange(args.lstm_seq_len - 1, -1, -1, dtype=np.int64)

    class Net(nn.Module):
        def __init__(self, n_inputs: int):
            super().__init__()
            self.rnn = nn.LSTM(n_inputs, 24, batch_first=True)
            self.head = nn.Linear(24, 1)

        def forward(self, batch: Any) -> Any:
            output, _ = self.rnn(batch)
            return self.head(output[:, -1, :]).squeeze(-1)

    nets = {}
    target_scalers = {}
    for fault in FAULTS:
        checkpoint = torch.load(args.output / "models" / f"lstm_100_{fault}.pt", map_location="cpu", weights_only=True)
        net = Net(len(features))
        net.load_state_dict(checkpoint["state_dict"])
        net.eval()
        nets[fault] = net
        target_scalers[fault] = (float(checkpoint["target_log_mean"]), float(checkpoint["target_log_std"]))

    def predict(fault: str, indices: np.ndarray) -> np.ndarray:
        values = np.empty(len(indices), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(indices), args.lstm_batch):
                chosen = indices[start : start + args.lstm_batch]
                sequence_index = np.maximum(chosen[:, None] - offsets[None, :], starts[chosen, None])
                tensor = torch.from_numpy(z[sequence_index])
                output = nets[fault](tensor).numpy()
                target_mean, target_std = target_scalers[fault]
                values[start : start + len(chosen)] = np.expm1(np.clip(output * target_std + target_mean, 0, 20))
        return values

    return predict


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    frame, features = load_prepared(args)
    report: dict[str, Any] = {"horizons_seconds": args.horizons, "label_mode": args.label_mode, "threshold": args.threshold, "classification": {}, "regression": {}, "joint": {}, "units": {"ttf": "seconds", "lead": "minutes", "false_alerts": "episodes per observed hour"}, "official_test_used": False}
    report["fault_log_audit"] = _fault_log_audit(args, frame)
    logged_events = None
    if args.label_mode == "calendar":
        logged_events = pd.read_csv(args.source / "fault_events.csv", usecols=["tool_id", "fault_type", "time"])
        logged_events = logged_events.drop_duplicates(["tool_id", "fault_type", "time"])
        prepared = json.loads((args.output / "preparation_report.json").read_text(encoding="utf-8"))
        cutoff = {tool: item["split_cutoff_time"] for tool, item in prepared["tools"].items()}
        logged_events = logged_events[logged_events["tool_id"].isin(cutoff) & (logged_events["time"] >= logged_events["tool_id"].map(cutoff))]
    x = frame[features].to_numpy(dtype=np.float32, copy=False)
    split = frame["split"].to_numpy()
    model_dir = args.output / "models"
    classification_rows = []
    regression_rows = []
    joint_rows = []
    event_detail_rows: list[dict[str, Any]] = []
    regression_event_rows: list[dict[str, Any]] = []
    final_positive_rows: dict[int, np.ndarray] = {}
    final_binary: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    xgb_scores: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for horizon in args.horizons:
        truth_all = frame[f"label_h{horizon}"].to_numpy(dtype=np.int8)
        indices = np.flatnonzero((split == 2) & (truth_all >= 0))
        truth = truth_all[indices]
        actual_type = frame[f"event_class_h{horizon}"].to_numpy(dtype=np.int8)[indices]
        for family, ratio in (("hgb", "1pct"), ("hgb", "100pct"), ("xgb", "100pct")):
            key = f"binary_{family}_{ratio}_h{horizon}"
            path = model_dir / f"{key}.joblib"
            if not path.exists():
                raise FileNotFoundError(path)
            model = joblib.load(path)
            start_at = time.monotonic()
            probability = model.predict_proba(x[indices])[:, 1]
            prediction = (probability >= args.threshold).astype(np.int8)
            matrix = confusion_matrix(truth, prediction, labels=[0, 1])
            event = _binary_event_metrics(frame, indices, truth, prediction, horizon, actual_type)
            if logged_events is not None:
                event.update(_calendar_logged_event_metrics(frame, indices, prediction, horizon, logged_events))
                event_detail_rows.extend(_calendar_event_detail_rows(frame, indices, prediction, horizon, logged_events, key))
            metrics = {
                "validation_rows": int(len(indices)), "positive_rows": int(truth.sum()),
                "precision": float(precision_score(truth, prediction, zero_division=0)),
                "recall": float(recall_score(truth, prediction, zero_division=0)),
                "f1": float(f1_score(truth, prediction, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
                "average_precision": float(average_precision_score(truth, probability)) if truth.sum() else None,
                "accuracy": float(np.mean(truth == prediction)),
                "confusion_tn_fp_fn_tp": [int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 0]), int(matrix[1, 1])],
                **event,
            }
            total_logged = report["fault_log_audit"].get("validation_unique_events_total")
            metrics["event_recall_over_all_logged_validation_faults"] = metrics["detected_events"] / total_logged if total_logged else None
            report["classification"][key] = metrics
            classification_rows.append({"model": key, "horizon_seconds": horizon, **metrics})
            print(f"{key}: AP={metrics['average_precision']:.4f}; event_recall={metrics['event_recall']}; seconds={time.monotonic()-start_at:.1f}", flush=True)
            if family == "xgb":
                final_positive_rows[horizon] = indices[prediction == 1]
                final_binary[horizon] = (indices, truth, prediction)
                xgb_scores[horizon] = (indices, probability.astype(np.float32))
            del model, probability, prediction

    if len(args.horizons) == 2:
        short, long = sorted(args.horizons)
        short_indices, short_scores = xgb_scores[short]
        long_indices, long_scores = xgb_scores[long]
        common, short_pos, long_pos = np.intersect1d(short_indices, long_indices, return_indices=True)
        violations = short_scores[short_pos] > long_scores[long_pos] + 1e-6
        report["probability_nesting_audit"] = {
            "common_validation_rows": int(len(common)),
            "raw_score_p_short_gt_p_long_rows": int(violations.sum()),
            "raw_score_p_short_gt_p_long_rate": float(violations.mean()) if len(common) else None,
            "note": "Separate class-weighted classifiers are not calibrated probabilities; apply calibration and enforce P(short) <= P(long) before a risk-tier UI.",
        }

    lstm_predict = None
    if not args.skip_lstm and all((model_dir / f"lstm_100_{f}.pt").exists() for f in FAULTS):
        lstm_predict = _lstm_predictor(args, frame, features)
    for fault in FAULTS:
        target = frame[f"ttf_{fault}"].to_numpy(dtype=np.float32)
        indices = np.flatnonzero((split == 2) & np.isfinite(target) & (target >= 0))
        truth = target[indices]
        event_tool = frame["tool_id"].to_numpy()[indices]
        event_times = frame[f"event_time_{fault}"].to_numpy()[indices].astype(np.int64)
        event_key = np.char.add(np.char.add(event_tool.astype(str), "_"), event_times.astype(str))
        for ratio in ("1pct", "100pct"):
            key = f"ttf_hgb_log_{ratio}_{fault}"
            model = joblib.load(model_dir / f"{key}.joblib")
            predicted = np.expm1(np.clip(model.predict(x[indices]), 0, 20)).astype(np.float32) if len(indices) else np.array([], dtype=np.float32)
            metrics = _regression_metrics(truth, predicted, event_key)
            report["regression"][key] = metrics
            regression_rows.append({"model": key, "fault": fault, **metrics})
            regression_event_rows.extend(_regression_event_rows(key, fault, event_tool, event_times, truth, predicted))
            print(f"{key}: MAE={metrics['mae_seconds']}s; R2={metrics['r2']}", flush=True)
            del model, predicted
        if lstm_predict is not None:
            key = f"lstm_100_{fault}"
            predicted = lstm_predict(fault, indices) if len(indices) else np.array([], dtype=np.float32)
            metrics = _regression_metrics(truth, predicted, event_key)
            report["regression"][key] = metrics
            regression_rows.append({"model": key, "fault": fault, **metrics})
            regression_event_rows.extend(_regression_event_rows(key, fault, event_tool, event_times, truth, predicted))
            print(f"{key}: MAE={metrics['mae_seconds']}s; R2={metrics['r2']}", flush=True)
            del predicted

    for horizon in args.horizons:
        indices, truth, prediction = final_binary[horizon]
        positive_indices = final_positive_rows[horizon]
        for family, ratio in (("hgb_log", "1pct"), ("hgb_log", "100pct"), ("lstm", "100pct")):
            if family == "lstm" and lstm_predict is None:
                continue
            predicted_ttf = []
            for fault in FAULTS:
                if family == "lstm":
                    predicted_ttf.append(lstm_predict(fault, positive_indices))
                else:
                    model = joblib.load(model_dir / f"ttf_hgb_log_{ratio}_{fault}.joblib")
                    predicted_ttf.append(np.expm1(np.clip(model.predict(x[positive_indices]), 0, 20)).astype(np.float32))
                    del model
            predicted_matrix = np.column_stack(predicted_ttf)
            chosen = np.argmin(predicted_matrix, axis=1).astype(np.int8) + 1
            selected_type = np.zeros(len(indices), dtype=np.int8)
            selected_type[prediction == 1] = chosen
            selected_ttf = np.full(len(indices), np.nan, dtype=np.float32)
            selected_ttf[prediction == 1] = np.min(predicted_matrix, axis=1)
            actual_type = frame[f"event_class_h{horizon}"].to_numpy(dtype=np.int8)[indices]
            correct_gate = (truth == 1) & (prediction == 1)
            correct_gate_type = correct_gate & (selected_type == actual_type)
            actual_ttf = frame[[f"ttf_{fault}" for fault in FAULTS]].to_numpy(dtype=np.float32)[indices]
            actual_nearest = np.min(np.where(np.isfinite(actual_ttf), actual_ttf, np.inf), axis=1)
            final_rul_mae = float(np.mean(np.abs(selected_ttf[correct_gate_type] - actual_nearest[correct_gate_type]))) if correct_gate_type.any() else None
            type_accuracy = float(np.mean(selected_type[correct_gate] == actual_type[correct_gate])) if correct_gate.any() else None
            key = f"xgb_100pct_h{horizon}_plus_{family}_{ratio}"
            event = _binary_event_metrics(frame, indices, truth, prediction, horizon, actual_type, selected_type)
            if logged_events is not None:
                event.update(_calendar_logged_event_metrics(frame, indices, prediction, horizon, logged_events, selected_type))
                event_detail_rows.extend(_calendar_event_detail_rows(frame, indices, prediction, horizon, logged_events, key, selected_type))
            item = {"validation_rows": int(len(indices)), "gate_true_positive_rows": int(correct_gate.sum()), "type_accuracy_given_gate_true_positive": type_accuracy, "rul_mae_seconds_when_gate_and_type_correct": final_rul_mae, "rul_rows_when_gate_and_type_correct": int(correct_gate_type.sum()), "selected_type_counts_on_alerts": {fault: int((chosen == k+1).sum()) for k, fault in enumerate(FAULTS)}, **event}
            total_logged = report["fault_log_audit"].get("validation_unique_events_total")
            item["type_correct_event_recall_over_all_logged_validation_faults"] = item["type_correct_events"] / total_logged if total_logged else None
            report["joint"][key] = item
            joint_rows.append({"model": key, "horizon_seconds": horizon, **item})
            print(f"{key}: type_correct_event_recall={item['type_correct_event_recall']}", flush=True)
    report["feature_count"] = len(features)
    report["features"] = features
    report["validation_note"] = "Fixed last 20% of each tool. Models use only first 80% minus max-horizon purge. No official test data. Threshold is fixed, not tuned on validation."
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(classification_rows).to_csv(args.output / "classification_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(regression_rows).to_csv(args.output / "regression_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(joint_rows).to_csv(args.output / "joint_metrics.csv", index=False, encoding="utf-8-sig")
    if event_detail_rows:
        pd.DataFrame(event_detail_rows).to_csv(args.output / "event_level_validation.csv", index=False, encoding="utf-8-sig")
        (args.output / "event_level_validation.json").write_text(json.dumps(event_detail_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if regression_event_rows:
        pd.DataFrame(regression_event_rows).to_csv(args.output / "regression_event_validation.csv", index=False, encoding="utf-8-sig")
    return report


def main() -> None:
    args = parse_args()
    if args.smoke and args.output == ROOT / "output_calendar_12_24":
        args.output = ROOT / "smoke_output"
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action in ("prepare", "all"):
        prepare(args)
    if args.action in ("train", "all"):
        train(args)
    if args.action in ("evaluate", "all"):
        evaluate(args)
    if args.action == "audit":
        frame, _ = load_prepared(args)
        audit = _fault_log_audit(args, frame)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"Completed: {args.output.resolve()}")


if __name__ == "__main__":
    main()
