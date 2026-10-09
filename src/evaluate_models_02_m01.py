from __future__ import annotations

import json
import math
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path.home() / "Desktop" / "LS_Project_01" / "data" / "train"
SENSOR_PATH = RAW_DIR / "02_M01_DC_train.csv"
TTF_PATH = RAW_DIR / "train_ttf" / "02_M01_DC_train_ttf.csv"
SAMPLES_PATH = ROOT / "output" / "xgboost_baseline" / "training_samples.csv.gz"
OUTPUT_DIR = ROOT / "output" / "model_validation_02_M01"

TTF_COLUMNS = [
    "TTF_FlowCool Pressure Dropped Below Limit",
    "TTF_Flowcool Pressure Too High Check Flowcool Pump",
    "TTF_Flowcool leak",
]
ID_COLUMNS = {"target", "ttf", "time", "minute_bucket", "event_time", "event_group", "runnum", "run_key"}
CATEGORICAL_COLUMNS = ["Tool", "stage", "recipe", "recipe_step"]


def audit_raw_files(chunk_size: int = 500_000) -> tuple[pd.DataFrame, dict[str, object]]:
    sensor_reader = pd.read_csv(SENSOR_PATH, chunksize=chunk_size)
    ttf_reader = pd.read_csv(TTF_PATH, chunksize=chunk_size)
    stats: dict[str, dict[str, float]] = {}
    categories = {key: set() for key in ["Tool", "stage", "Lot", "runnum", "recipe", "recipe_step"]}
    rows = 0
    aligned = True
    non_increasing = 0
    gap_gt_4 = 0
    interval_counts: Counter[int] = Counter()
    first_time = None
    last_time = None
    previous_time = None
    ttf_zero_counts = {column: 0 for column in TTF_COLUMNS}
    ttf_missing_counts = {column: 0 for column in TTF_COLUMNS}
    columns: list[str] = []

    while True:
        try:
            sensor_chunk = next(sensor_reader)
        except StopIteration:
            sensor_chunk = None
        try:
            ttf_chunk = next(ttf_reader)
        except StopIteration:
            ttf_chunk = None
        if sensor_chunk is None and ttf_chunk is None:
            break
        if sensor_chunk is None or ttf_chunk is None or len(sensor_chunk) != len(ttf_chunk):
            aligned = False
            break
        if not columns:
            columns = sensor_chunk.columns.tolist()
        aligned = aligned and np.array_equal(sensor_chunk["time"].to_numpy(), ttf_chunk["time"].to_numpy())
        rows += len(sensor_chunk)

        times = sensor_chunk["time"].to_numpy(dtype=np.int64)
        if first_time is None:
            first_time = int(times[0])
        combined = times if previous_time is None else np.concatenate([[previous_time], times])
        diffs = np.diff(combined)
        non_increasing += int(np.sum(diffs <= 0))
        gap_gt_4 += int(np.sum(diffs > 4))
        interval_counts.update(map(int, diffs.tolist()))
        previous_time = int(times[-1])
        last_time = previous_time

        for name in categories:
            categories[name].update(sensor_chunk[name].dropna().unique().tolist())

        for name in sensor_chunk.columns:
            series = sensor_chunk[name]
            entry = stats.setdefault(name, {"missing": 0.0, "inf": 0.0, "count": 0.0, "sum": 0.0, "sumsq": 0.0, "min": math.inf, "max": -math.inf})
            entry["missing"] += float(series.isna().sum())
            if pd.api.types.is_numeric_dtype(series):
                values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                finite = values[np.isfinite(values)]
                entry["inf"] += float(np.isinf(values).sum())
                if finite.size:
                    entry["count"] += float(finite.size)
                    entry["sum"] += float(finite.sum())
                    entry["sumsq"] += float(np.square(finite).sum())
                    entry["min"] = min(entry["min"], float(finite.min()))
                    entry["max"] = max(entry["max"], float(finite.max()))

        for name in TTF_COLUMNS:
            numeric = pd.to_numeric(ttf_chunk[name], errors="coerce")
            ttf_zero_counts[name] += int(numeric.eq(0).sum())
            ttf_missing_counts[name] += int(numeric.isna().sum())

    rows_out: list[dict[str, object]] = []
    for name in columns:
        entry = stats[name]
        count = int(entry["count"])
        mean = entry["sum"] / count if count else np.nan
        variance = max(entry["sumsq"] / count - mean * mean, 0.0) if count else np.nan
        role = "식별/시간" if name in {"time", "Tool", "Lot", "runnum"} else ("공정 조건" if name in {"stage", "recipe", "recipe_step"} else "센서")
        rows_out.append(
            {
                "field": name,
                "role": role,
                "rows": rows,
                "missing_count": int(entry["missing"]),
                "missing_rate": entry["missing"] / max(rows, 1),
                "infinite_count": int(entry["inf"]),
                "minimum": None if entry["min"] == math.inf else entry["min"],
                "maximum": None if entry["max"] == -math.inf else entry["max"],
                "mean": mean if count else None,
                "std": math.sqrt(variance) if count else None,
                "unique_categories": len(categories[name]) if name in categories else None,
                "constant_flag": bool(count and entry["min"] == entry["max"]),
            }
        )

    most_common_interval = interval_counts.most_common(1)[0] if interval_counts else (None, 0)
    summary = {
        "sensor_file": str(SENSOR_PATH),
        "ttf_file": str(TTF_PATH),
        "raw_rows": rows,
        "raw_columns": len(columns),
        "first_time_seconds": first_time,
        "last_time_seconds": last_time,
        "duration_days": (last_time - first_time) / 86400 if first_time is not None and last_time is not None else None,
        "sensor_ttf_row_and_time_alignment": aligned,
        "non_increasing_time_intervals": non_increasing,
        "gaps_over_4_seconds": gap_gt_4,
        "most_common_interval_seconds": most_common_interval[0],
        "most_common_interval_share": most_common_interval[1] / max(rows - 1, 1),
        "fault_event_counts": ttf_zero_counts,
        "ttf_missing_counts": ttf_missing_counts,
        "category_cardinality": {key: len(value) for key, value in categories.items()},
    }
    return pd.DataFrame(rows_out), summary


def build_preprocessor(data: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    features = [column for column in data.columns if column not in ID_COLUMNS]
    categorical = [column for column in CATEGORICAL_COLUMNS if column in features]
    numeric = [column for column in features if column not in categorical]
    transformer = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return transformer, features, numeric


def choose_threshold_oof(pipeline: Pipeline, train: pd.DataFrame, features: list[str]) -> float:
    groups = train["event_group"].to_numpy()
    y = train["target"].to_numpy()
    probabilities = np.full(len(train), np.nan)
    splitter = GroupKFold(n_splits=min(3, train["event_group"].nunique()))
    for fit_idx, score_idx in splitter.split(train[features], y, groups):
        fold_model = clone(pipeline)
        fold_model.fit(train.iloc[fit_idx][features], y[fit_idx])
        probabilities[score_idx] = fold_model.predict_proba(train.iloc[score_idx][features])[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def classification_evaluation(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    preprocessor, features, _ = build_preprocessor(data)
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, valid_idx = next(split.split(data[features], data["target"], groups=data["event_group"]))
    train = data.iloc[train_idx].copy()
    valid = data.iloc[valid_idx].copy()
    ratio = (len(train) - int(train["target"].sum())) / max(int(train["target"].sum()), 1)

    models = {
        "Dummy (prevalence)": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=350, min_samples_leaf=4, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_estimators=300, learning_rate=0.05, max_depth=4, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=ratio, tree_method="hist", n_jobs=4, random_state=42),
        "MLP (64-32)": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", early_stopping=True, validation_fraction=0.2, max_iter=500, random_state=42),
    }
    rows: list[dict[str, object]] = []
    predictions_all: list[pd.DataFrame] = []
    for model_name, estimator in models.items():
        pipe = Pipeline([("preprocessor", clone(preprocessor)), ("model", estimator)])
        threshold = 0.5 if model_name.startswith("Dummy") else choose_threshold_oof(pipe, train, features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(train[features], train["target"])
        probability = pipe.predict_proba(valid[features])[:, 1]
        prediction = (probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(valid["target"], prediction, labels=[0, 1]).ravel()
        positive = valid["target"].eq(1)
        event_detected = (
            pd.DataFrame({"event_group": valid.loc[positive, "event_group"], "prediction": prediction[positive.to_numpy()]})
            .groupby("event_group")["prediction"]
            .max()
        )
        rows.append(
            {
                "model": model_name,
                "model_family": "신경망" if model_name.startswith("MLP") else "머신러닝",
                "train_rows": len(train),
                "validation_rows": len(valid),
                "train_events": train["event_group"].nunique(),
                "validation_events": valid["event_group"].nunique(),
                "validation_positive_rate": valid["target"].mean(),
                "threshold_from_train_cv": threshold,
                "pr_auc": average_precision_score(valid["target"], probability),
                "roc_auc": roc_auc_score(valid["target"], probability),
                "brier_score": brier_score_loss(valid["target"], probability),
                "precision": precision_score(valid["target"], prediction, zero_division=0),
                "recall": recall_score(valid["target"], prediction, zero_division=0),
                "f1": f1_score(valid["target"], prediction, zero_division=0),
                "specificity": tn / max(tn + fp, 1),
                "false_positive_rate": fp / max(tn + fp, 1),
                "event_recall": event_detected.mean() if len(event_detected) else np.nan,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
        detail = valid[["time", "Tool", "runnum", "event_group", "ttf", "target"]].copy()
        detail.insert(0, "model", model_name)
        detail["probability"] = probability
        detail["threshold"] = threshold
        detail["prediction"] = prediction
        predictions_all.append(detail)

    split_summary = {
        "train_event_groups": sorted(train["event_group"].unique().tolist()),
        "validation_event_groups": sorted(valid["event_group"].unique().tolist()),
        "train_rows": len(train),
        "validation_rows": len(valid),
        "train_positive_rows": int(train["target"].sum()),
        "validation_positive_rows": int(valid["target"].sum()),
    }
    return pd.DataFrame(rows), pd.concat(predictions_all, ignore_index=True), split_summary


def regression_evaluation(data: pd.DataFrame, validation_groups: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    positive = data.loc[data["target"].eq(1)].copy()
    positive["ttf_minutes"] = positive["ttf"] / 60.0
    train = positive.loc[~positive["event_group"].isin(validation_groups)].copy()
    valid = positive.loc[positive["event_group"].isin(validation_groups)].copy()
    preprocessor, features, _ = build_preprocessor(positive)
    models = {
        "Dummy Median": DummyRegressor(strategy="median"),
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(n_estimators=350, min_samples_leaf=3, n_jobs=-1, random_state=42),
        "XGBoost": XGBRegressor(objective="reg:squarederror", n_estimators=300, learning_rate=0.04, max_depth=3, min_child_weight=4, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, tree_method="hist", n_jobs=4, random_state=42),
        "MLP (64-32)": MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", early_stopping=True, validation_fraction=0.2, max_iter=800, random_state=42),
    }
    rows: list[dict[str, object]] = []
    predictions_all: list[pd.DataFrame] = []
    for model_name, estimator in models.items():
        pipe = Pipeline([("preprocessor", clone(preprocessor)), ("model", estimator)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(train[features], train["ttf_minutes"])
        prediction = np.clip(pipe.predict(valid[features]), 0.0, 30.0)
        actual = valid["ttf_minutes"].to_numpy()
        absolute_error = np.abs(actual - prediction)
        event_mae = (
            pd.DataFrame({"event_group": valid["event_group"].to_numpy(), "absolute_error": absolute_error})
            .groupby("event_group")["absolute_error"]
            .mean()
        )
        rows.append(
            {
                "model": model_name,
                "model_family": "신경망" if model_name.startswith("MLP") else "머신러닝",
                "train_rows": len(train),
                "validation_rows": len(valid),
                "train_events": train["event_group"].nunique(),
                "validation_events": valid["event_group"].nunique(),
                "mae_minutes": mean_absolute_error(actual, prediction),
                "rmse_minutes": math.sqrt(mean_squared_error(actual, prediction)),
                "median_ae_minutes": median_absolute_error(actual, prediction),
                "r2": r2_score(actual, prediction),
                "within_5_minutes": float(np.mean(absolute_error <= 5)),
                "within_10_minutes": float(np.mean(absolute_error <= 10)),
                "event_macro_mae_minutes": event_mae.mean(),
            }
        )
        detail = valid[["time", "Tool", "runnum", "event_group", "ttf", "ttf_minutes"]].copy()
        detail.insert(0, "model", model_name)
        detail["predicted_ttf_minutes"] = prediction
        detail["absolute_error_minutes"] = absolute_error
        predictions_all.append(detail)
    return pd.DataFrame(rows), pd.concat(predictions_all, ignore_index=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Auditing raw 02_M01 sensor and TTF files...")
    quality, raw_summary = audit_raw_files()
    quality.to_csv(OUTPUT_DIR / "data_quality.csv", index=False)

    print("Loading prepared one-minute feature samples...")
    data = pd.read_csv(SAMPLES_PATH)
    classification, class_predictions, split_summary = classification_evaluation(data)
    validation_groups = split_summary["validation_event_groups"]
    regression, reg_predictions = regression_evaluation(data, validation_groups)

    classification = classification.sort_values(["pr_auc", "event_recall", "f1"], ascending=False).reset_index(drop=True)
    regression = regression.sort_values(["mae_minutes", "rmse_minutes"]).reset_index(drop=True)
    classification.to_csv(OUTPUT_DIR / "classification_metrics.csv", index=False)
    regression.to_csv(OUTPUT_DIR / "regression_metrics.csv", index=False)
    class_predictions.to_csv(OUTPUT_DIR / "classification_predictions.csv.gz", index=False, compression="gzip")
    reg_predictions.to_csv(OUTPUT_DIR / "regression_predictions.csv.gz", index=False, compression="gzip")

    result = {
        "scope": "02_M01 pilot; pressure-drop fault; 30-minute horizon",
        "raw_data": raw_summary,
        "prepared_data": {
            "rows": len(data),
            "columns": len(data.columns),
            "positive_rows": int(data["target"].sum()),
            "positive_rate": float(data["target"].mean()),
            "fault_cycles": int(data["event_group"].nunique()),
            "aggregation_minutes": 1,
            "rolling_windows_minutes": [5, 15, 30],
            "negative_sampling_ratio": 10,
        },
        "split": split_summary,
        "selected_classification_model_by_pr_auc": classification.iloc[0]["model"],
        "selected_regression_model_by_mae": regression.iloc[0]["model"],
        "limitations": [
            "Only one tool and one fault type were evaluated.",
            "The held-out set contains two fault cycles, so uncertainty is high.",
            "Negative rows were downsampled, so production false alarms per hour cannot be estimated.",
            "TTF regression is evaluated only on rows within 30 minutes of a known fault.",
            "Deep learning is represented by a small feed-forward MLP; sequence models require the full event dataset.",
        ],
    }
    (OUTPUT_DIR / "validation_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(classification.to_string(index=False))
    print(regression.to_string(index=False))
    print(f"Saved validation outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
