from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd


FAULT_TARGETS = {
    "pressure_drop": "TTF_FlowCool Pressure Dropped Below Limit",
    "pressure_high": "TTF_Flowcool Pressure Too High Check Flowcool Pump",
    "leak": "TTF_Flowcool leak",
}

SENSOR_COLUMNS = [
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
]

CONTEXT_COLUMNS = ["Tool", "stage", "runnum", "recipe", "recipe_step"]
READ_CHUNK_SIZE = 500_000
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an event-grouped XGBoost predictive-maintenance baseline."
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path.home() / "Desktop" / "LS_Project_01" / "data" / "train",
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        default=["02_M01"],
        help="Tool file prefixes, for example: 02_M01 03_M01 04_M02",
    )
    parser.add_argument(
        "--fault", choices=sorted(FAULT_TARGETS), default="pressure_drop"
    )
    parser.add_argument("--horizon-minutes", type=int, default=30)
    parser.add_argument(
        "--windows-minutes",
        default="5,15,30",
        help="Comma-separated rolling windows after one-minute aggregation.",
    )
    parser.add_argument(
        "--sample-minutes",
        type=int,
        default=1,
        help="Keep one feature row every N minutes after aggregation.",
    )
    parser.add_argument(
        "--negative-ratio",
        type=int,
        default=10,
        help="Maximum negative rows retained per positive row within each fault cycle.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "xgboost_baseline",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create training_samples.csv.gz without fitting XGBoost.",
    )
    return parser.parse_args()


def normalize_tool_name(tool: str) -> str:
    return tool.upper().replace("-", "_")


def sensor_path_for(train_dir: Path, tool: str) -> Path:
    return train_dir / f"{normalize_tool_name(tool)}_DC_train.csv"


def ttf_path_for(train_dir: Path, sensor_path: Path) -> Path:
    ttf_dir = train_dir / "train_ttf"
    candidates = [
        ttf_dir / f"{sensor_path.stem}_ttf.csv",
        ttf_dir / sensor_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"TTF file not found for {sensor_path.name}. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def aggregate_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    named_aggregations: dict[str, tuple[str, str]] = {
        "time": ("time", "max"),
        "Tool": ("Tool", "last"),
        "stage": ("stage", "last"),
        "recipe": ("recipe", "last"),
        "recipe_step": ("recipe_step", "last"),
        "ttf": ("ttf", "last"),
        "context_count_1m": ("context_key", "nunique"),
    }
    for sensor in SENSOR_COLUMNS:
        named_aggregations[f"{sensor}_last"] = (sensor, "last")
        named_aggregations[f"{sensor}_mean_1m"] = (sensor, "mean")
        named_aggregations[f"{sensor}_std_1m"] = (sensor, "std")
        named_aggregations[f"{sensor}_min_1m"] = (sensor, "min")
        named_aggregations[f"{sensor}_max_1m"] = (sensor, "max")

    return (
        frame.groupby(["minute_bucket", "runnum"], as_index=False, sort=False)
        .agg(**named_aggregations)
        .sort_values("time")
    )


def read_minute_data(
    sensor_path: Path,
    ttf_path: Path,
    ttf_column: str,
) -> pd.DataFrame:
    sensor_usecols = ["time", *CONTEXT_COLUMNS, *SENSOR_COLUMNS]
    ttf_usecols = ["time", ttf_column]
    sensor_reader = pd.read_csv(
        sensor_path, usecols=sensor_usecols, chunksize=READ_CHUNK_SIZE
    )
    ttf_reader = pd.read_csv(ttf_path, usecols=ttf_usecols, chunksize=READ_CHUNK_SIZE)

    aggregated_parts: list[pd.DataFrame] = []
    carry = pd.DataFrame()
    for chunk_number, pair in enumerate(
        zip_longest(sensor_reader, ttf_reader), start=1
    ):
        sensor_chunk, ttf_chunk = pair
        if sensor_chunk is None or ttf_chunk is None:
            raise ValueError("Sensor and TTF files contain different numbers of rows.")
        if len(sensor_chunk) != len(ttf_chunk):
            raise ValueError(
                f"Chunk {chunk_number}: sensor rows and TTF rows do not match."
            )
        if not np.array_equal(
            sensor_chunk["time"].to_numpy(), ttf_chunk["time"].to_numpy()
        ):
            raise ValueError(f"Chunk {chunk_number}: sensor and TTF times do not align.")

        sensor_chunk = sensor_chunk.copy()
        sensor_chunk["ttf"] = pd.to_numeric(
            ttf_chunk[ttf_column], errors="coerce"
        ).to_numpy()
        sensor_chunk["minute_bucket"] = sensor_chunk["time"].floordiv(60)
        sensor_chunk["context_key"] = (
            sensor_chunk["stage"].astype(str)
            + "|"
            + sensor_chunk["recipe"].astype(str)
            + "|"
            + sensor_chunk["recipe_step"].astype(str)
        )
        if not carry.empty:
            sensor_chunk = pd.concat([carry, sensor_chunk], ignore_index=True)

        last_bucket = sensor_chunk["minute_bucket"].iloc[-1]
        ready = sensor_chunk.loc[sensor_chunk["minute_bucket"].ne(last_bucket)]
        carry = sensor_chunk.loc[sensor_chunk["minute_bucket"].eq(last_bucket)].copy()
        if not ready.empty:
            aggregated_parts.append(aggregate_minutes(ready))

    if not carry.empty:
        aggregated_parts.append(aggregate_minutes(carry))
    if not aggregated_parts:
        raise RuntimeError(f"No rows read from {sensor_path}")

    minute_data = pd.concat(aggregated_parts, ignore_index=True)
    return minute_data.sort_values("time", ignore_index=True)


def add_rolling_features(
    minute_data: pd.DataFrame,
    windows_minutes: list[int],
) -> pd.DataFrame:
    data = minute_data.copy()
    data["run_key"] = data["Tool"].astype(str) + "_" + data["runnum"].astype(str)
    mean_columns = [f"{sensor}_mean_1m" for sensor in SENSOR_COLUMNS]
    grouped = data.groupby("run_key", sort=False)[mean_columns]
    feature_blocks: list[pd.DataFrame] = []

    for window in windows_minutes:
        minimum_rows = max(2, window // 2)
        rolling_mean = (
            grouped.rolling(window, min_periods=minimum_rows)
            .mean()
            .reset_index(level=0, drop=True)
        )
        rolling_std = (
            grouped.rolling(window, min_periods=minimum_rows)
            .std()
            .reset_index(level=0, drop=True)
        )
        rolling_min = (
            grouped.rolling(window, min_periods=minimum_rows)
            .min()
            .reset_index(level=0, drop=True)
        )
        rolling_max = (
            grouped.rolling(window, min_periods=minimum_rows)
            .max()
            .reset_index(level=0, drop=True)
        )
        shifted = grouped.shift(window)

        window_features: dict[str, pd.Series] = {}
        for mean_column in mean_columns:
            sensor = mean_column.removesuffix("_mean_1m")
            window_features[f"{sensor}_mean_{window}m"] = rolling_mean[mean_column]
            window_features[f"{sensor}_std_{window}m"] = rolling_std[mean_column]
            window_features[f"{sensor}_min_{window}m"] = rolling_min[mean_column]
            window_features[f"{sensor}_max_{window}m"] = rolling_max[mean_column]
            window_features[f"{sensor}_delta_{window}m"] = (
                data[mean_column] - shifted[mean_column]
            )
        feature_blocks.append(pd.DataFrame(window_features, index=data.index))

    return pd.concat([data, *feature_blocks], axis=1).copy()


def label_and_sample(
    data: pd.DataFrame,
    horizon_minutes: int,
    sample_minutes: int,
    negative_ratio: int,
    random_state: int,
) -> pd.DataFrame:
    horizon_seconds = horizon_minutes * 60
    usable = data.loc[data["ttf"].notna() & data["ttf"].ge(0)].copy()
    usable = usable.loc[usable["minute_bucket"].mod(sample_minutes).eq(0)]
    usable["target"] = usable["ttf"].le(horizon_seconds).astype("int8")
    usable["event_time"] = np.rint(usable["time"] + usable["ttf"]).astype("int64")
    usable["event_group"] = (
        usable["Tool"].astype(str) + "_" + usable["event_time"].astype(str)
    )

    sampled_groups: list[pd.DataFrame] = []
    for _, cycle in usable.groupby("event_group", sort=False):
        positives = cycle.loc[cycle["target"].eq(1)]
        if positives.empty:
            continue
        negatives = cycle.loc[cycle["target"].eq(0)]
        negative_count = min(len(negatives), len(positives) * negative_ratio)
        if negative_count:
            negatives = negatives.sample(
                n=negative_count, random_state=random_state
            )
            sampled_groups.append(pd.concat([positives, negatives]))
        else:
            sampled_groups.append(positives)

    if not sampled_groups:
        raise RuntimeError("No labeled fault cycles with positive samples were found.")
    return (
        pd.concat(sampled_groups, ignore_index=True)
        .sort_values(["event_group", "time"])
        .reset_index(drop=True)
    )


def prepare_training_data(args: argparse.Namespace) -> pd.DataFrame:
    windows = sorted(
        {int(value.strip()) for value in args.windows_minutes.split(",") if value.strip()}
    )
    if not windows or min(windows) <= 0:
        raise ValueError("--windows-minutes must contain positive integers.")
    if args.sample_minutes <= 0 or args.negative_ratio <= 0:
        raise ValueError("Sampling parameters must be positive.")

    ttf_column = FAULT_TARGETS[args.fault]
    prepared: list[pd.DataFrame] = []
    for tool in args.tools:
        sensor_path = sensor_path_for(args.train_dir, tool)
        if not sensor_path.exists():
            raise FileNotFoundError(sensor_path)
        ttf_path = ttf_path_for(args.train_dir, sensor_path)
        print(f"Preparing {sensor_path.name} with {ttf_path.name} ...")
        minute_data = read_minute_data(sensor_path, ttf_path, ttf_column)
        featured = add_rolling_features(minute_data, windows)
        sampled = label_and_sample(
            featured,
            horizon_minutes=args.horizon_minutes,
            sample_minutes=args.sample_minutes,
            negative_ratio=args.negative_ratio,
            random_state=args.random_state,
        )
        prepared.append(sampled)
        print(
            f"  rows={len(sampled):,}, positives={int(sampled['target'].sum()):,}, "
            f"fault cycles={sampled['event_group'].nunique():,}"
        )

    training_data = pd.concat(prepared, ignore_index=True)
    return training_data.sort_values(["Tool", "time"], ignore_index=True)


def train_model(
    training_data: pd.DataFrame,
    output_dir: Path,
    test_size: float,
    random_state: int,
) -> dict[str, object]:
    try:
        import joblib
        from sklearn.compose import ColumnTransformer
        from sklearn.metrics import (
            average_precision_score,
            confusion_matrix,
            precision_recall_curve,
            roc_auc_score,
        )
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from xgboost import XGBClassifier
    except ImportError as error:
        expected_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
        raise RuntimeError(
            "Training dependencies are missing from the Python interpreter that launched "
            f"this script. Current interpreter: {sys.executable}. "
            f"Run with the project interpreter: {expected_python}"
        ) from error

    excluded = {
        "target",
        "ttf",
        "time",
        "minute_bucket",
        "event_time",
        "event_group",
        "runnum",
        "run_key",
    }
    categorical_columns = ["Tool", "stage", "recipe", "recipe_step"]
    feature_columns = [
        column for column in training_data.columns if column not in excluded
    ]
    numeric_columns = [
        column for column in feature_columns if column not in categorical_columns
    ]

    if training_data["event_group"].nunique() < 4:
        raise RuntimeError(
            "At least four fault cycles are required for a grouped train/validation split."
        )
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_index, valid_index = next(
        splitter.split(
            training_data[feature_columns],
            training_data["target"],
            groups=training_data["event_group"],
        )
    )
    train = training_data.iloc[train_index].copy()
    valid = training_data.iloc[valid_index].copy()
    if train["target"].nunique() < 2 or valid["target"].nunique() < 2:
        raise RuntimeError(
            "Grouped split produced a single-class partition. Add more tools/fault cycles."
        )

    positive_count = int(train["target"].sum())
    negative_count = len(train) - positive_count
    scale_pos_weight = negative_count / max(positive_count, 1)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical_columns,
            ),
            ("numeric", "passthrough", numeric_columns),
        ],
        remainder="drop",
    )
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=min(4, max(1, (os.cpu_count() or 2) - 1)),
        random_state=random_state,
    )
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )
    pipeline.fit(train[feature_columns], train["target"])

    probabilities = pipeline.predict_proba(valid[feature_columns])[:, 1]
    precision, recall, thresholds = precision_recall_curve(
        valid["target"], probabilities
    )
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    best_position = int(np.nanargmax(f1))
    threshold = float(thresholds[best_position])
    predictions = probabilities >= threshold

    validation_result = valid[
        ["time", "Tool", "runnum", "event_group", "ttf", "target"]
    ].copy()
    validation_result["fault_probability"] = probabilities
    validation_result["prediction"] = predictions.astype("int8")
    validation_result.to_csv(
        output_dir / "validation_predictions.csv.gz", index=False, compression="gzip"
    )

    positive_events = validation_result.loc[validation_result["target"].eq(1)]
    detected_events = (
        positive_events.groupby("event_group")["prediction"].max().sum()
    )
    event_count = positive_events["event_group"].nunique()
    negative_validation = validation_result.loc[validation_result["target"].eq(0)]
    false_positive_rows = int(negative_validation["prediction"].sum())

    transformed_names = pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()
    feature_importance = pd.DataFrame(
        {
            "feature": transformed_names,
            "importance": pipeline.named_steps[
                "classifier"
            ].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    joblib.dump(pipeline, output_dir / "xgboost_pipeline.joblib")

    matrix = confusion_matrix(valid["target"], predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    validation_positive_rate = float(valid["target"].mean())
    average_precision = float(
        average_precision_score(valid["target"], probabilities)
    )
    report: dict[str, object] = {
        "training_rows": len(train),
        "validation_rows": len(valid),
        "training_fault_cycles": int(train["event_group"].nunique()),
        "validation_fault_cycles": int(valid["event_group"].nunique()),
        "training_positive_rows": positive_count,
        "scale_pos_weight": scale_pos_weight,
        "validation_positive_rate": validation_positive_rate,
        "average_precision": average_precision,
        "average_precision_lift_over_prevalence": float(
            average_precision / max(validation_positive_rate, 1e-12)
        ),
        "roc_auc": float(roc_auc_score(valid["target"], probabilities)),
        "selected_threshold": threshold,
        "threshold_f1": float(f1[best_position]),
        "confusion_matrix_tn_fp_fn_tp": matrix.ravel().tolist(),
        "threshold_precision": float(
            true_positive / max(true_positive + false_positive, 1)
        ),
        "threshold_recall": float(
            true_positive / max(true_positive + false_negative, 1)
        ),
        "threshold_false_positive_rate": float(
            false_positive / max(false_positive + true_negative, 1)
        ),
        "event_recall": float(detected_events / max(event_count, 1)),
        "detected_events": int(detected_events),
        "validation_positive_events": int(event_count),
        "false_positive_rows": false_positive_rows,
        "note": (
            "The threshold maximizes row-level F1 on the same validation split. "
            "Negative rows were downsampled, so this report cannot estimate production "
            "false alarms per operating hour. Validate a fixed threshold on untouched "
            "chronological data before deployment."
        ),
    }
    return report


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_data = prepare_training_data(args)
    sample_path = args.output_dir / "training_samples.csv.gz"
    training_data.to_csv(sample_path, index=False, compression="gzip")
    print(
        f"Prepared {len(training_data):,} rows, "
        f"{int(training_data['target'].sum()):,} positives, "
        f"{training_data['event_group'].nunique():,} fault cycles."
    )
    print(f"Training samples: {sample_path.resolve()}")

    if args.prepare_only:
        return

    report = train_model(
        training_data,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    report.update(
        {
            "fault": args.fault,
            "ttf_column": FAULT_TARGETS[args.fault],
            "horizon_minutes": args.horizon_minutes,
            "tools": [normalize_tool_name(tool) for tool in args.tools],
            "sample_minutes": args.sample_minutes,
            "negative_ratio": args.negative_ratio,
        }
    )
    report_path = args.output_dir / "model_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Model report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
