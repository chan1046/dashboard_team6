"""PHM 2018 냉각계통: 20개 설비 Random Forest 학습 파이프라인.

구조
1. 20개 설비의 sensor/TTF/faults 파일을 자동 탐색하고 manifest를 검증한다.
2. 설비별 원본을 chunk로 읽어 sensor와 TTF의 행 및 time을 대조한다.
3. 동일 시각 행을 삭제하지 않고 모두 사용해 1분 통계로 축약한다.
4. 설비별 특징 파일을 저장해 20개 원본을 동시에 메모리에 올리지 않는다.
5. 각 설비 안에서 시간순 80:20 분할하고 경계 30분을 제거한다.
6. 고장 유형별 TTF Random Forest 회귀 3개와 다음 고장 유형 분류 1개를 학습한다.
7. 전체/설비별 성능, PHM 점수, 변수 중요도와 검증 예측을 저장한다.
8. 테스트 설비를 동일하게 전처리해 대시보드용 분 단위 예측을 만든다.

주의
- fault_name과 TTF 정답은 입력 특성으로 사용하지 않는다.
- Tool은 20개 설비의 차이를 학습하기 위한 범주형 특성으로 사용한다.
- 최종 운영 평가는 untouched 연속 기간에서 별도로 수행해야 한다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_DIR = Path.home() / "Desktop" / "LS_Project_01" / "data" / "train"
DEFAULT_TEST_DIR = Path.home() / "Desktop" / "LS_Project_01" / "data" / "test"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "rf_multi_tool"

TTF_ORIGINAL_TO_SHORT = {
    "TTF_FlowCool Pressure Dropped Below Limit": "ttf_pressure_low",
    "TTF_Flowcool Pressure Too High Check Flowcool Pump": "ttf_pressure_high",
    "TTF_Flowcool leak": "ttf_leak",
}
TTF_COLUMNS = list(TTF_ORIGINAL_TO_SHORT.values())
FAULT_LABELS = {
    "ttf_pressure_low": 0,
    "ttf_pressure_high": 1,
    "ttf_leak": 2,
}
LABEL_TO_FAULT = {0: "pressure_low", 1: "pressure_high", 2: "leak"}
FAULT_NAME_MAP = {
    "FlowCool Pressure Dropped Below Limit": "pressure_low",
    "Flowcool Pressure Too High Check Flowcool Pump": "pressure_high",
    "Flowcool leak": "leak",
}

IDENTITY_COLUMNS = ["time", "Tool", "stage", "Lot", "runnum", "recipe", "recipe_step"]
CORE_ROLLING_SENSORS = [
    "FLOWCOOLFLOWRATE",
    "FLOWCOOLPRESSURE",
    "IONGAUGEPRESSURE",
    "ETCHGASCHANNEL1READBACK",
    "ETCHPBNGASREADBACK",
]
CATEGORICAL_FEATURES = ["Tool", "stage", "recipe", "recipe_step"]
DROP_FROM_FEATURES = {
    "time",
    "Lot",
    "runnum",
    "minute_bucket",
    "representative_ttf",
    "next_fault_type",
    "split",
    "operating_session",
    "session_key",
    "feature_group_key",
    *TTF_COLUMNS,
}


@dataclass(frozen=True)
class ToolFiles:
    tool: str
    sensor: Path
    ttf: Path
    faults: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="20개 냉각 설비 Random Forest TTF 회귀·고장 유형 분류"
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=["manifest", "prepare", "train", "predict", "all"],
        default="all",
        help="실행 단계. all은 prepare→train→predict 순서입니다.",
    )
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--tools",
        nargs="+",
        default=["all"],
        help="all 또는 01_M01 01_M02 ... 형식의 설비 목록",
    )
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--purge-minutes", type=int, default=30)
    parser.add_argument(
        "--stop-gap-seconds",
        "--max-gap-seconds",
        dest="stop_gap_seconds",
        type=int,
        default=120,
        help="이보다 긴 무기록 구간을 설비 정지로 보고 재가동 시 새 운전 세션을 시작합니다.",
    )
    parser.add_argument("--rolling-windows-minutes", default="5,15,30")
    parser.add_argument("--max-train-rows", type=int, default=300_000)
    parser.add_argument(
        "--max-valid-rows",
        type=int,
        default=0,
        help="내부 평가 행 상한. 0이면 마지막 20%% 전체를 평가하여 실제 분포를 유지합니다.",
    )
    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument("--max-depth", type=int, default=18)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--max-raw-rows-per-tool",
        type=int,
        default=0,
        help="0은 전체 행. 양수는 개발용 앞부분 제한이며 운영 학습에 사용하지 마세요.",
    )
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def normalize_tool(value: str) -> str:
    cleaned = value.upper().replace("-", "_")
    if not re.fullmatch(r"\d{2}_M\d{2}", cleaned):
        raise ValueError(f"잘못된 설비 이름: {value}. 예: 01_M01")
    return cleaned


def parse_windows(value: str) -> list[int]:
    windows = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not windows or min(windows) < 2:
        raise ValueError("rolling-windows-minutes에는 2 이상의 정수가 필요합니다.")
    return windows


def ttf_path_for(train_dir: Path, sensor: Path) -> Path:
    candidates = [
        train_dir / "train_ttf" / f"{sensor.stem}_ttf.csv",
        train_dir / "train_ttf" / sensor.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def discover_tools(train_dir: Path, requested: list[str]) -> list[ToolFiles]:
    available: dict[str, Path] = {}
    for sensor in sorted(train_dir.glob("*_DC_train.csv")):
        match = re.fullmatch(r"(\d{2}_M\d{2})_DC_train", sensor.stem, re.IGNORECASE)
        if match:
            available[normalize_tool(match.group(1))] = sensor
    if not available:
        raise FileNotFoundError(f"학습 센서 파일을 찾지 못했습니다: {train_dir}")

    selected = sorted(available) if requested == ["all"] else [normalize_tool(x) for x in requested]
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise FileNotFoundError(f"센서 파일이 없는 설비: {unknown}")

    result: list[ToolFiles] = []
    for tool in selected:
        sensor = available[tool]
        result.append(
            ToolFiles(
                tool=tool,
                sensor=sensor,
                ttf=ttf_path_for(train_dir, sensor),
                faults=train_dir / "train_faults" / f"{tool}_train_fault_data.csv",
            )
        )
    return result


def validate_manifest(tools: list[ToolFiles]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    required_sensor = set(IDENTITY_COLUMNS)
    required_ttf = {"time", *TTF_ORIGINAL_TO_SHORT}
    required_faults = {"time", "fault_name", "Tool"}
    for item in tools:
        record: dict[str, Any] = {
            "tool": item.tool,
            "sensor_path": str(item.sensor),
            "ttf_path": str(item.ttf),
            "faults_path": str(item.faults),
            "sensor_bytes": item.sensor.stat().st_size if item.sensor.exists() else None,
            "ttf_bytes": item.ttf.stat().st_size if item.ttf.exists() else None,
            "faults_bytes": item.faults.stat().st_size if item.faults.exists() else None,
            "status": "OK",
            "message": "",
        }
        missing_files = [str(path) for path in [item.sensor, item.ttf, item.faults] if not path.exists()]
        if missing_files:
            record["status"] = "ERROR"
            record["message"] = "파일 누락: " + "; ".join(missing_files)
            rows.append(record)
            continue
        sensor_columns = set(pd.read_csv(item.sensor, nrows=0).columns)
        ttf_columns = set(pd.read_csv(item.ttf, nrows=0).columns)
        fault_columns = set(pd.read_csv(item.faults, nrows=0).columns)
        missing_columns = {
            "sensor": sorted(required_sensor - sensor_columns),
            "ttf": sorted(required_ttf - ttf_columns),
            "faults": sorted(required_faults - fault_columns),
        }
        bad = {key: value for key, value in missing_columns.items() if value}
        if bad:
            record["status"] = "ERROR"
            record["message"] = f"필수 열 누락: {bad}"
        rows.append(record)
    return pd.DataFrame(rows)


def downcast_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    for column in frame.select_dtypes(include=["float64"]).columns:
        frame[column] = pd.to_numeric(frame[column], downcast="float")
    for column in frame.select_dtypes(include=["int64"]).columns:
        frame[column] = pd.to_numeric(frame[column], downcast="integer")
    return frame


def aggregate_minute_block(frame: pd.DataFrame, sensor_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    aggregations: dict[str, tuple[str, str]] = {
        "time": ("time", "max"),
        "Tool": ("Tool", "last"),
        "stage": ("stage", "last"),
        "Lot": ("Lot", "last"),
        "runnum": ("runnum", "last"),
        "recipe": ("recipe", "last"),
        "recipe_step": ("recipe_step", "last"),
        "raw_rows_1m": ("time", "size"),
        "unique_times_1m": ("time", "nunique"),
        "context_count_1m": ("context_key", "nunique"),
    }
    for target in TTF_COLUMNS:
        aggregations[target] = (target, "last")
    for sensor in sensor_columns:
        aggregations[f"{sensor}_last"] = (sensor, "last")
        aggregations[f"{sensor}_mean_1m"] = (sensor, "mean")
        aggregations[f"{sensor}_std_1m"] = (sensor, "std")

    result = (
        frame.groupby(["minute_bucket", "runnum"], sort=False, as_index=False)
        .agg(**aggregations)
        .sort_values("source_order", kind="stable")
        if "source_order" in frame.columns
        else frame.groupby(["minute_bucket", "runnum"], sort=False, as_index=False).agg(**aggregations)
    )
    result["duplicate_time_rows_1m"] = result["raw_rows_1m"] - result["unique_times_1m"]
    return result


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    valid_any = frame[TTF_COLUMNS].notna().any(axis=1)
    frame["representative_ttf"] = frame[TTF_COLUMNS].min(axis=1, skipna=True)
    frame.loc[~valid_any, "representative_ttf"] = np.nan
    frame["next_fault_type"] = pd.Series(pd.NA, index=frame.index, dtype="Int8")
    if valid_any.any():
        frame.loc[valid_any, "next_fault_type"] = (
            frame.loc[valid_any, TTF_COLUMNS].idxmin(axis=1).map(FAULT_LABELS).astype("int8")
        )
    return frame


def add_causal_features(
    frame: pd.DataFrame,
    windows: list[int],
    stop_gap_seconds: int,
) -> pd.DataFrame:
    frame = frame.reset_index(drop=True)
    frame["Tool"] = frame["Tool"].astype(str)
    frame["run_key"] = frame["Tool"] + "_" + frame["runnum"].astype(str)
    frame["tool_elapsed_time"] = frame["time"] - frame.groupby("Tool")["time"].transform("min")
    frame["time_diff"] = frame.groupby("Tool", sort=False)["time"].diff().fillna(0)
    frame["restart_after_stop"] = (
        frame["time_diff"].gt(stop_gap_seconds) | frame["time_diff"].lt(0)
    ).astype("int8")
    frame["operating_session"] = frame.groupby("Tool", sort=False)[
        "restart_after_stop"
    ].cumsum()
    frame["session_key"] = (
        frame["Tool"] + "_" + frame["operating_session"].astype(str)
    )
    frame["feature_group_key"] = (
        frame["session_key"] + "_" + frame["runnum"].astype(str)
    )
    frame["session_observation_index"] = (
        frame.groupby("feature_group_key", sort=False).cumcount() + 1
    ).astype("int32")
    frame["coverage_ratio_1m"] = (
        pd.to_numeric(frame["unique_times_1m"], errors="coerce") / 15.0
    ).clip(upper=1.0)

    core_columns = [f"{name}_mean_1m" for name in CORE_ROLLING_SENSORS if f"{name}_mean_1m" in frame]
    grouped = frame.groupby("feature_group_key", sort=False)[core_columns]
    differences = grouped.diff()
    for column in core_columns:
        sensor = column.removesuffix("_mean_1m")
        frame[f"{sensor}_diff_1m"] = differences[column].astype("float32")
    for window in windows:
        minimum = max(2, window // 2)
        frame[f"rolling_ready_{window}m"] = (
            frame["session_observation_index"] >= minimum
        ).astype("int8")
        means = grouped.rolling(window, min_periods=minimum).mean().reset_index(level=0, drop=True)
        stds = grouped.rolling(window, min_periods=minimum).std(ddof=0).reset_index(level=0, drop=True)
        shifted = grouped.shift(window)
        for column in core_columns:
            sensor = column.removesuffix("_mean_1m")
            frame[f"{sensor}_mean_{window}m"] = means[column].astype("float32")
            frame[f"{sensor}_std_{window}m"] = stds[column].astype("float32")
            frame[f"{sensor}_delta_{window}m"] = (frame[column] - shifted[column]).astype("float32")

    pressure = "FLOWCOOLPRESSURE_mean_1m"
    flow = "FLOWCOOLFLOWRATE_mean_1m"
    if pressure in frame and flow in frame:
        frame["flowcool_pressure_minus_flow"] = (frame[pressure] - frame[flow]).astype("float32")
        safe_flow = frame[flow].where(frame[flow].abs() > 1e-6)
        frame["flowcool_pressure_div_flow"] = (frame[pressure] / safe_flow).astype("float32")
    return downcast_dataframe(frame)


def exclude_ttf_across_stops(
    frame: pd.DataFrame,
    stop_gap_seconds: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Exclude labels whose fault occurs after the current operating session ended."""
    session_end = frame.groupby("session_key", sort=False)["time"].transform("max")
    excluded: dict[str, int] = {}
    for target in TTF_COLUMNS:
        target_values = pd.to_numeric(frame[target], errors="coerce")
        event_time = pd.to_numeric(frame["time"], errors="coerce") + target_values
        crosses_stop = target_values.notna() & event_time.gt(session_end + stop_gap_seconds)
        excluded[target] = int(crosses_stop.sum())
        frame.loc[crosses_stop, target] = np.nan
    return frame, excluded


def inspect_faults(item: ToolFiles, minute_times: pd.Series) -> dict[str, Any]:
    faults = pd.read_csv(item.faults, low_memory=False).sort_values("time")
    faults["fault_type"] = faults["fault_name"].map(FAULT_NAME_MAP)
    available = np.sort(pd.unique(pd.to_numeric(minute_times, errors="coerce").dropna()))
    fault_times = pd.to_numeric(faults["time"], errors="coerce").dropna().to_numpy()
    if len(available) and len(fault_times):
        positions = np.clip(np.searchsorted(available, fault_times), 0, len(available) - 1)
        left = np.clip(positions - 1, 0, len(available) - 1)
        distance = np.minimum(np.abs(available[positions] - fault_times), np.abs(available[left] - fault_times))
        distance_max = float(distance.max())
        distance_median = float(np.median(distance))
    else:
        distance_max = distance_median = None
    return {
        "tool": item.tool,
        "fault_rows": int(len(faults)),
        "pressure_low": int(faults["fault_type"].eq("pressure_low").sum()),
        "pressure_high": int(faults["fault_type"].eq("pressure_high").sum()),
        "leak": int(faults["fault_type"].eq("leak").sum()),
        "unmapped": int(faults["fault_type"].isna().sum()),
        "nearest_minute_distance_max_seconds": distance_max,
        "nearest_minute_distance_median_seconds": distance_median,
    }


def prepare_tool(
    item: ToolFiles,
    cache_path: Path,
    windows: list[int],
    stop_gap_seconds: int,
    chunk_size: int,
    max_raw_rows: int,
) -> dict[str, Any]:
    sensor_header = pd.read_csv(item.sensor, nrows=0).columns.tolist()
    sensor_columns = [column for column in sensor_header if column not in IDENTITY_COLUMNS]
    sensor_reader = pd.read_csv(item.sensor, chunksize=chunk_size, low_memory=False)
    ttf_reader = pd.read_csv(item.ttf, chunksize=chunk_size, low_memory=False)
    parts: list[pd.DataFrame] = []
    carry = pd.DataFrame()
    raw_rows = 0
    duplicate_time_rows = 0
    time_aligned = True
    truncated = False

    for chunk_index, (sensor_chunk, ttf_chunk) in enumerate(zip_longest(sensor_reader, ttf_reader), start=1):
        if sensor_chunk is None or ttf_chunk is None:
            raise ValueError(f"{item.tool}: sensor와 TTF chunk 수가 다릅니다.")
        if len(sensor_chunk) != len(ttf_chunk):
            raise ValueError(f"{item.tool}: chunk {chunk_index} 행 수가 다릅니다.")
        if max_raw_rows > 0:
            remaining = max_raw_rows - raw_rows
            if remaining <= 0:
                truncated = True
                break
            if len(sensor_chunk) > remaining:
                sensor_chunk = sensor_chunk.iloc[:remaining].copy()
                ttf_chunk = ttf_chunk.iloc[:remaining].copy()
                truncated = True
        sensor_time = pd.to_numeric(sensor_chunk["time"], errors="raise").to_numpy()
        ttf_time = pd.to_numeric(ttf_chunk["time"], errors="raise").to_numpy()
        if not np.array_equal(sensor_time, ttf_time):
            time_aligned = False
            raise ValueError(f"{item.tool}: chunk {chunk_index}의 time 순서가 다릅니다.")

        raw_rows += len(sensor_chunk)
        duplicate_time_rows += int(sensor_chunk["time"].duplicated(keep=False).sum())
        sensor_chunk = sensor_chunk.reset_index(drop=True)
        ttf_chunk = ttf_chunk.reset_index(drop=True).rename(columns=TTF_ORIGINAL_TO_SHORT)
        block = pd.concat([sensor_chunk, ttf_chunk[TTF_COLUMNS]], axis=1)
        block["minute_bucket"] = pd.to_numeric(block["time"], errors="raise").floordiv(60)
        block["context_key"] = (
            block["stage"].astype(str) + "|" + block["recipe"].astype(str) + "|" + block["recipe_step"].astype(str)
        )
        if not carry.empty:
            block = pd.concat([carry, block], ignore_index=True)
        last_bucket = block["minute_bucket"].iloc[-1]
        last_run = block["runnum"].iloc[-1]
        is_last_group = block["minute_bucket"].eq(last_bucket) & block["runnum"].eq(last_run)
        ready = block.loc[~is_last_group]
        carry = block.loc[is_last_group].copy()
        if not ready.empty:
            parts.append(aggregate_minute_block(ready, sensor_columns))
        if max_raw_rows > 0 and raw_rows >= max_raw_rows:
            truncated = True
            break

    if not carry.empty:
        parts.append(aggregate_minute_block(carry, sensor_columns))
    if not parts:
        raise RuntimeError(f"{item.tool}: 전처리 결과가 없습니다.")
    minute_data = pd.concat(parts, ignore_index=True)
    minute_data["Tool"] = item.tool.replace("_", "")
    minute_data = add_causal_features(minute_data, windows, stop_gap_seconds)
    minute_data, excluded_across_stops = exclude_ttf_across_stops(
        minute_data, stop_gap_seconds
    )
    minute_data = add_targets(minute_data)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache = cache_path.with_name(
        f"{cache_path.name}.{os.getpid()}.partial"
    )
    try:
        minute_data.to_csv(temporary_cache, index=False, compression="gzip")
        os.replace(temporary_cache, cache_path)
    finally:
        if temporary_cache.exists():
            temporary_cache.unlink()

    fault_report = inspect_faults(item, minute_data["time"])
    record = {
        "tool": item.tool,
        "raw_rows": raw_rows,
        "prepared_rows": len(minute_data),
        "prepared_columns": len(minute_data.columns),
        "duplicate_time_rows_observed": duplicate_time_rows,
        "sensor_ttf_time_aligned": time_aligned,
        "truncated_for_development": truncated,
        "cache_path": str(cache_path),
        **{
            f"excluded_across_stop_{target}": count
            for target, count in excluded_across_stops.items()
        },
        **{f"fault_{key}": value for key, value in fault_report.items() if key != "tool"},
    }
    cache_metadata = {
        "tool": item.tool,
        "sensor_bytes": item.sensor.stat().st_size,
        "sensor_mtime_ns": item.sensor.stat().st_mtime_ns,
        "ttf_bytes": item.ttf.stat().st_size,
        "ttf_mtime_ns": item.ttf.stat().st_mtime_ns,
        "rolling_windows_minutes": windows,
        "stop_gap_seconds": stop_gap_seconds,
        "max_raw_rows_per_tool": max_raw_rows,
        "complete_source_file": not truncated and max_raw_rows == 0,
        "preparation_record": record,
    }
    metadata_path = cache_path.with_name(cache_path.name + ".meta.json")
    temporary_metadata = metadata_path.with_name(
        f"{metadata_path.name}.{os.getpid()}.partial"
    )
    try:
        temporary_metadata.write_text(
            json.dumps(cache_metadata, ensure_ascii=False, indent=2, default=json_default),
            encoding="utf-8",
        )
        os.replace(temporary_metadata, metadata_path)
    finally:
        if temporary_metadata.exists():
            temporary_metadata.unlink()
    return record


def prepared_path(output_dir: Path, tool: str) -> Path:
    return output_dir / "prepared" / f"{tool}_features.csv.gz"


def valid_complete_cache(
    cache: Path,
    item: ToolFiles,
    windows: list[int],
    stop_gap_seconds: int,
) -> dict[str, Any] | None:
    metadata_path = cache.with_name(cache.name + ".meta.json")
    if not cache.exists() or not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        "tool": item.tool,
        "sensor_bytes": item.sensor.stat().st_size,
        "sensor_mtime_ns": item.sensor.stat().st_mtime_ns,
        "ttf_bytes": item.ttf.stat().st_size,
        "ttf_mtime_ns": item.ttf.stat().st_mtime_ns,
        "rolling_windows_minutes": windows,
        "stop_gap_seconds": stop_gap_seconds,
        "complete_source_file": True,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None
    record = metadata.get("preparation_record")
    return record if isinstance(record, dict) else None


def prepare_all(args: argparse.Namespace, tools: list[ToolFiles], windows: list[int]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for index, item in enumerate(tools, start=1):
        cache = prepared_path(args.output_dir, item.tool)
        cached_record = (
            valid_complete_cache(cache, item, windows, args.stop_gap_seconds)
            if not args.force_prepare and args.max_raw_rows_per_tool == 0
            else None
        )
        if cached_record is not None:
            print(f"[{index}/{len(tools)}] {item.tool}: 기존 특징 캐시 사용")
            records.append(cached_record)
            continue
        print(f"[{index}/{len(tools)}] {item.tool}: 원본 전처리")
        record = prepare_tool(
            item,
            cache,
            windows,
            args.stop_gap_seconds,
            args.chunk_size,
            args.max_raw_rows_per_tool,
        )
        records.append(record)
        print(f"    raw={record['raw_rows']:,}, prepared={record['prepared_rows']:,}")
    report = pd.DataFrame(records)
    report.to_csv(args.output_dir / "preparation_report.csv", index=False, encoding="utf-8-sig")
    return report


def choose_feature_columns(columns: Iterable[str]) -> tuple[list[str], list[str], list[str]]:
    all_columns = list(columns)
    categorical = [column for column in CATEGORICAL_FEATURES if column in all_columns]
    numeric = [
        column
        for column in all_columns
        if column not in DROP_FROM_FEATURES
        and column not in categorical
        and column != "run_key"
    ]
    features = numeric + categorical
    if not features:
        raise ValueError("사용 가능한 입력 특성이 없습니다.")
    return features, numeric, categorical


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def create_pipeline(estimator: Any, numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline([("preprocess", make_preprocessor(numeric, categorical)), ("model", estimator)])


def make_regressor(args: argparse.Namespace, offset: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features="sqrt",
        n_jobs=max(1, min(8, (os.cpu_count() or 4) - 1)),
        random_state=args.random_state + offset,
    )


def make_classifier(args: argparse.Namespace) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=max(1, min(8, (os.cpu_count() or 4) - 1)),
        random_state=args.random_state,
    )


def sample_balanced_by_tool(
    frame: pd.DataFrame,
    max_rows: int,
    random_state: int,
    target: str | None = None,
) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame
    tools = sorted(frame["Tool"].astype(str).unique())
    quota = max(1, max_rows // len(tools))
    parts: list[pd.DataFrame] = []
    for position, tool in enumerate(tools):
        group = frame.loc[frame["Tool"].astype(str).eq(tool)]
        count = min(len(group), quota)
        if target and target in group and group[target].notna().any():
            values = pd.to_numeric(group[target], errors="coerce")
            cutoff = values.quantile(0.20)
            near = group.loc[values.le(cutoff)]
            far = group.loc[values.gt(cutoff)]
            near_n = min(len(near), count // 2)
            far_n = min(len(far), count - near_n)
            picked = [near.sample(near_n, random_state=random_state + position)] if near_n else []
            if far_n:
                picked.append(far.sample(far_n, random_state=random_state + 1000 + position))
            sample = pd.concat(picked) if picked else group.head(0)
            if len(sample) < count:
                remaining = group.drop(index=sample.index)
                sample = pd.concat(
                    [sample, remaining.sample(min(count - len(sample), len(remaining)), random_state=random_state + 2000 + position)]
                )
        else:
            sample = group.sample(count, random_state=random_state + position)
        parts.append(sample)
    combined = pd.concat(parts, ignore_index=True)
    if len(combined) > max_rows:
        combined = combined.sample(max_rows, random_state=random_state)
    return combined.sort_values(["Tool", "time"], kind="stable").reset_index(drop=True)


def load_split_samples(
    args: argparse.Namespace,
    tools: list[ToolFiles],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []
    split_report: dict[str, Any] = {}
    for item in tools:
        cache = prepared_path(args.output_dir, item.tool)
        if not cache.exists():
            raise FileNotFoundError(f"특징 캐시가 없습니다. prepare를 먼저 실행하세요: {cache}")
        try:
            frame = pd.read_csv(cache, compression="gzip", low_memory=False)
        except (pd.errors.ParserError, EOFError, OSError) as exc:
            raise ValueError(
                f"{item.tool}: 특징 캐시를 읽을 수 없습니다: {cache}. "
                f"이 설비만 다시 전처리하세요: prepare --tools {item.tool} --force-prepare"
            ) from exc
        frame = frame.sort_values("time", kind="stable")
        split_position = int(len(frame) * args.train_ratio)
        if split_position <= 0 or split_position >= len(frame):
            raise ValueError(f"{item.tool}: train-ratio로 인해 학습 또는 검증 데이터가 비었습니다.")
        cutoff = float(frame.iloc[split_position]["time"])
        purge_seconds = args.purge_minutes * 60
        train = frame.loc[frame["time"].lt(cutoff - purge_seconds)].copy()
        valid = frame.loc[frame["time"].ge(cutoff)].copy()
        labels_crossing_cutoff: dict[str, int] = {}
        for target in TTF_COLUMNS:
            target_values = pd.to_numeric(train[target], errors="coerce")
            event_time = pd.to_numeric(train["time"], errors="coerce") + target_values
            crosses_cutoff = target_values.notna() & event_time.ge(cutoff)
            labels_crossing_cutoff[target] = int(crosses_cutoff.sum())
            train.loc[crosses_cutoff, target] = np.nan
        train = add_targets(train)
        train["split"] = "train"
        valid["split"] = "validation"
        train_parts.append(train)
        valid_parts.append(valid)
        split_report[item.tool] = {
            "prepared_rows": len(frame),
            "train_rows_before_sampling": len(train),
            "validation_rows_before_sampling": len(valid),
            "cutoff_time": cutoff,
            "purge_minutes": args.purge_minutes,
            "training_labels_excluded_across_cutoff": labels_crossing_cutoff,
        }
    train_all = pd.concat(train_parts, ignore_index=True)
    valid_all = pd.concat(valid_parts, ignore_index=True)
    return train_all, valid_all, split_report


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    truth = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(truth) & np.isfinite(prediction)
    if not mask.any():
        return {"n": 0, "mae_seconds": None, "rmse_seconds": None, "r2": None, "phm_score": None}
    truth = truth[mask]
    prediction = np.maximum(prediction[mask], 0)
    error = np.abs(truth - prediction)
    return {
        "n": int(len(truth)),
        "mae_seconds": float(mean_absolute_error(truth, prediction)),
        "rmse_seconds": float(math.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)) if len(truth) >= 2 else None,
        "median_ae_seconds": float(np.median(error)),
        "within_30_minutes": float(np.mean(error <= 1800)),
        "phm_score": phm_score(truth, prediction),
    }


def phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.maximum(np.asarray(y_pred, dtype=float), 0)
    return float(np.mean(np.exp(-0.001 * truth) * np.abs(truth - prediction)))


def export_feature_importance(model: Pipeline, path: Path) -> None:
    names = model.named_steps["preprocess"].get_feature_names_out()
    values = model.named_steps["model"].feature_importances_
    pd.DataFrame({"feature": names, "importance": values}).sort_values(
        "importance", ascending=False
    ).to_csv(path, index=False, encoding="utf-8-sig")


def train_models(args: argparse.Namespace, tools: list[ToolFiles]) -> dict[str, Any]:
    train_all, valid_all, split_report = load_split_samples(args, tools)
    features, numeric, categorical = choose_feature_columns(train_all.columns)
    model_dir = args.output_dir / "models"
    importance_dir = args.output_dir / "feature_importance"
    model_dir.mkdir(parents=True, exist_ok=True)
    importance_dir.mkdir(parents=True, exist_ok=True)
    print(f"통합 특징 수: {len(features)}, 학습 설비 수: {train_all['Tool'].nunique()}")

    metrics: dict[str, Any] = {
        "tools": [item.tool for item in tools],
        "tool_count": len(tools),
        "split": split_report,
        "features": {"all": features, "numeric": numeric, "categorical": categorical},
        "regression": {},
        "classification": {},
    }
    validation_output = valid_all[["Tool", "time", *TTF_COLUMNS, "representative_ttf", "next_fault_type"]].copy()
    regressors: dict[str, Pipeline] = {}

    regression_rows: list[dict[str, Any]] = []
    for offset, target in enumerate(TTF_COLUMNS):
        target_train = train_all.loc[train_all[target].notna()].copy()
        target_valid = valid_all.loc[valid_all[target].notna()].copy()
        target_train = sample_balanced_by_tool(
            target_train, args.max_train_rows, args.random_state + offset, target=target
        )
        target_valid = sample_balanced_by_tool(
            target_valid, args.max_valid_rows, args.random_state + 100 + offset, target=target
        )
        if target_train.empty or target_valid.empty:
            raise ValueError(f"{target}: 학습 또는 검증 정답이 없습니다.")
        print(f"회귀 {target}: train={len(target_train):,}, valid={len(target_valid):,}")
        model = create_pipeline(make_regressor(args, offset), numeric, categorical)
        model.fit(target_train[features], target_train[target])
        prediction = np.maximum(model.predict(target_valid[features]), 0)
        overall = regression_metrics(target_valid[target], prediction)
        regression_rows.append({"target": target, "scope": "ALL", **overall})
        for tool_name, group in target_valid.assign(prediction=prediction).groupby("Tool", sort=True):
            regression_rows.append(
                {"target": target, "scope": str(tool_name), **regression_metrics(group[target], group["prediction"].to_numpy())}
            )
        model_path = model_dir / f"rf_{target}.joblib"
        joblib.dump(model, model_path)
        export_feature_importance(model, importance_dir / f"rf_{target}_importance.csv")
        regressors[target] = model
        metrics["regression"][target] = {"train_rows": len(target_train), "validation_rows": len(target_valid), **overall}

    regression_table = pd.DataFrame(regression_rows)
    regression_table.to_csv(args.output_dir / "regression_metrics.csv", index=False, encoding="utf-8-sig")

    classifier_train = train_all.loc[train_all["next_fault_type"].notna()].copy()
    classifier_valid = valid_all.loc[valid_all["next_fault_type"].notna()].copy()
    classifier_train = sample_balanced_by_tool(classifier_train, args.max_train_rows, args.random_state)
    classifier_valid = sample_balanced_by_tool(classifier_valid, args.max_valid_rows, args.random_state + 100)
    classifier_train["next_fault_type"] = classifier_train["next_fault_type"].astype(int)
    classifier_valid["next_fault_type"] = classifier_valid["next_fault_type"].astype(int)
    if classifier_train["next_fault_type"].nunique() < 2:
        raise ValueError("통합 학습 데이터의 고장 유형이 2개 미만입니다.")
    print(f"분류 next_fault_type: train={len(classifier_train):,}, valid={len(classifier_valid):,}")
    classifier = create_pipeline(make_classifier(args), numeric, categorical)
    classifier.fit(classifier_train[features], classifier_train["next_fault_type"])
    class_prediction = classifier.predict(classifier_valid[features]).astype(int)
    class_truth = classifier_valid["next_fault_type"].to_numpy(dtype=int)

    classification_rows: list[dict[str, Any]] = []
    evaluated = classifier_valid.assign(prediction=class_prediction)
    for scope, group in [("ALL", evaluated), *[(str(k), v) for k, v in evaluated.groupby("Tool", sort=True)]]:
        truth = group["next_fault_type"].to_numpy(dtype=int)
        pred = group["prediction"].to_numpy(dtype=int)
        classification_rows.append(
            {
                "scope": scope,
                "n": len(group),
                "accuracy": accuracy_score(truth, pred),
                "macro_f1": f1_score(truth, pred, average="macro", zero_division=0),
                "weighted_f1": f1_score(truth, pred, average="weighted", zero_division=0),
                "macro_precision": precision_score(truth, pred, average="macro", zero_division=0),
                "macro_recall": recall_score(truth, pred, average="macro", zero_division=0),
            }
        )
    pd.DataFrame(classification_rows).to_csv(
        args.output_dir / "classification_metrics.csv", index=False, encoding="utf-8-sig"
    )
    labels = [0, 1, 2]
    metrics["classification"] = {
        "train_rows": len(classifier_train),
        "validation_rows": len(classifier_valid),
        **classification_rows[0],
        "confusion_matrix_labels_0_1_2": confusion_matrix(class_truth, class_prediction, labels=labels).tolist(),
        "classification_report": classification_report(
            class_truth,
            class_prediction,
            labels=labels,
            target_names=[LABEL_TO_FAULT[label] for label in labels],
            output_dict=True,
            zero_division=0,
        ),
    }
    joblib.dump(classifier, model_dir / "rf_next_fault_type.joblib")
    export_feature_importance(classifier, importance_dir / "rf_next_fault_type_importance.csv")

    validation_class = classifier_valid[["Tool", "time", "next_fault_type"]].copy()
    validation_class["pred_next_fault_type"] = class_prediction
    probabilities = classifier.predict_proba(classifier_valid[features])
    for position, label in enumerate(classifier.classes_):
        validation_class[f"prob_{LABEL_TO_FAULT[int(label)]}"] = probabilities[:, position]
    validation_class.to_csv(
        args.output_dir / "validation_classification_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    metrics["model_paths"] = {
        **{target: str(model_dir / f"rf_{target}.joblib") for target in TTF_COLUMNS},
        "next_fault_type": str(model_dir / "rf_next_fault_type.joblib"),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8"
    )
    (args.output_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "features": features,
                "numeric_features": numeric,
                "categorical_features": categorical,
                "rolling_windows_minutes": parse_windows(args.rolling_windows_minutes),
                "stop_gap_seconds": args.stop_gap_seconds,
                "tool_count": len(tools),
                "tools": [item.tool for item in tools],
                "aggregation_minutes": 1,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    raise TypeError(f"JSON 변환 불가: {type(value)}")


def prepare_test_frame(
    path: Path,
    windows: list[int],
    stop_gap_seconds: int,
) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    missing = set(IDENTITY_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name}: 테스트 필수 열 누락 {sorted(missing)}")
    sensors = [column for column in raw.columns if column not in IDENTITY_COLUMNS]
    raw["minute_bucket"] = pd.to_numeric(raw["time"], errors="raise").floordiv(60)
    raw["context_key"] = (
        raw["stage"].astype(str) + "|" + raw["recipe"].astype(str) + "|" + raw["recipe_step"].astype(str)
    )
    for target in TTF_COLUMNS:
        raw[target] = np.nan
    minute = aggregate_minute_block(raw, sensors)
    minute["Tool"] = str(raw["Tool"].iloc[0])
    minute = add_causal_features(minute, windows, stop_gap_seconds)
    return add_targets(minute)


def predict_test(args: argparse.Namespace) -> None:
    metadata_path = args.output_dir / "model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("model_metadata.json이 없습니다. train을 먼저 실행하세요.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = metadata["features"]
    windows = metadata["rolling_windows_minutes"]
    stop_gap_seconds = metadata["stop_gap_seconds"]
    model_dir = args.output_dir / "models"
    regressors = {target: joblib.load(model_dir / f"rf_{target}.joblib") for target in TTF_COLUMNS}
    classifier = joblib.load(model_dir / "rf_next_fault_type.joblib")
    prediction_dir = args.output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.test_dir.glob("*_DC_test.csv"))
    if not paths:
        print(f"경고: 테스트 CSV가 없습니다: {args.test_dir}")
        return
    for path in paths:
        print(f"테스트 예측: {path.name}")
        frame = prepare_test_frame(path, windows, stop_gap_seconds)
        for feature in features:
            if feature not in frame:
                frame[feature] = np.nan
        result = frame[
            [
                "time",
                "Tool",
                "stage",
                "runnum",
                "recipe",
                "recipe_step",
                "restart_after_stop",
                "session_observation_index",
                *[f"rolling_ready_{window}m" for window in windows],
            ]
        ].copy()
        prediction_matrix: list[np.ndarray] = []
        for target in TTF_COLUMNS:
            prediction = np.maximum(regressors[target].predict(frame[features]), 0)
            result[f"pred_{target}_seconds"] = prediction
            result[f"pred_{target}_minutes"] = prediction / 60.0
            prediction_matrix.append(prediction)
        matrix = np.column_stack(prediction_matrix)
        result["pred_representative_ttf_seconds"] = matrix.min(axis=1)
        result["fault_type_from_min_ttf"] = [LABEL_TO_FAULT[int(index)] for index in matrix.argmin(axis=1)]
        predicted_class = classifier.predict(frame[features]).astype(int)
        result["classifier_fault_type"] = [LABEL_TO_FAULT[int(label)] for label in predicted_class]
        probabilities = classifier.predict_proba(frame[features])
        for position, label in enumerate(classifier.classes_):
            result[f"prob_{LABEL_TO_FAULT[int(label)]}"] = probabilities[:, position]
        result.to_csv(
            prediction_dir / f"{path.stem}_dashboard_prediction.csv.gz",
            index=False,
            compression="gzip",
        )


def main() -> None:
    args = parse_args()
    if not 0 < args.train_ratio < 1:
        raise ValueError("train-ratio는 0과 1 사이여야 합니다.")
    if args.chunk_size <= 0 or args.purge_minutes < 0 or args.stop_gap_seconds < 60:
        raise ValueError(
            "chunk-size는 양수, purge-minutes는 0 이상, stop-gap-seconds는 60 이상이어야 합니다."
        )
    windows = parse_windows(args.rolling_windows_minutes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tools = discover_tools(args.train_dir, args.tools)
    manifest = validate_manifest(tools)
    manifest.to_csv(args.output_dir / "tool_manifest.csv", index=False, encoding="utf-8-sig")
    print(manifest[["tool", "status", "message"]].to_string(index=False))
    if manifest["status"].ne("OK").any():
        raise RuntimeError("manifest 오류를 수정한 후 다시 실행하세요.")
    print(f"검증된 설비: {len(tools)}대")
    if args.action == "manifest":
        return
    if args.action in {"prepare", "all"}:
        prepare_all(args, tools, windows)
    if args.action in {"train", "all"}:
        train_models(args, tools)
    if args.action in {"predict", "all"} and not args.skip_test:
        predict_test(args)
    print(f"완료: {args.output_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except ImportError as error:
        expected = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
        raise RuntimeError(f"학습 패키지가 없습니다. {expected}로 실행하세요.") from error
