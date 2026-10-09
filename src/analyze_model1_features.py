"""Read-only Model-1 feature correlation and saved-XGBoost importance audit."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, pointbiserialr
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
FORWARD = ROOT / "output/experiments/forward_validation_ab_24h"
PREPARED = ROOT / "output/baseline_recovery/prepared"
MODEL_PATH = ROOT / "output/models/xgb_24h_forward_a.joblib"
OUT = ROOT / "output/experiments/model1_feature_analysis"
SENSORS = [
    "IONGAUGEPRESSURE", "ETCHBEAMVOLTAGE", "ETCHBEAMCURRENT",
    "ETCHSUPPRESSORVOLTAGE", "ETCHSUPPRESSORCURRENT", "FLOWCOOLFLOWRATE",
    "FLOWCOOLPRESSURE", "ETCHGASCHANNEL1READBACK", "ETCHPBNGASREADBACK",
    "FIXTURETILTANGLE", "ROTATIONSPEED", "ACTUALROTATIONANGLE",
    "FIXTURESHUTTERPOSITION", "ETCHSOURCEUSAGE", "ETCHAUXSOURCETIMER",
    "ETCHAUX2SOURCETIMER", "ACTUALSTEPDURATION",
]


def source_sensor(feature: str) -> str:
    if feature in ("pressure_minus_flow", "pressure_div_flow"):
        return "FLOWCOOLPRESSURE+FLOWCOOLFLOWRATE"
    for sensor in sorted(SENSORS, key=len, reverse=True):
        if feature == sensor or feature.startswith(sensor + "_"):
            return sensor
    return "process/session"


def feature_group(feature: str) -> str:
    if feature in SENSORS:
        return "current/raw"
    if feature.endswith(("_std_1m", "_min_1m", "_max_1m")):
        return "1m_statistics"
    if feature.endswith(("_mean_5m", "_std_5m", "_mean_30m", "_std_30m")):
        return "rolling"
    if feature.endswith("_ewma_10m"):
        return "ewma"
    if feature.endswith(("_diff", "_per_hour", "_slope_5m")):
        return "change/slope"
    if feature in ("pressure_minus_flow", "pressure_div_flow"):
        return "interaction"
    return "process/session"


def new_state(n: int) -> dict[str, np.ndarray]:
    return {name: np.zeros(n, dtype=np.float64) for name in
            ("count", "sum_x", "sum_y", "sum_xx", "sum_yy", "sum_xy")}


def update_state(state: dict[str, np.ndarray], frame: pd.DataFrame,
                 features: list[str], mask: np.ndarray) -> None:
    selected = frame.loc[mask, [*features, "label"]]
    for start in range(0, len(selected), 50000):
        part = selected.iloc[start:start + 50000]
        x = part[features].to_numpy(dtype=np.float64, copy=True)
        y = part.label.to_numpy(dtype=np.float64)
        finite = np.isfinite(x)
        safe = np.where(finite, x, 0.0)
        state["count"] += finite.sum(axis=0)
        state["sum_x"] += safe.sum(axis=0)
        state["sum_y"] += (finite * y[:, None]).sum(axis=0)
        state["sum_xx"] += (safe * safe).sum(axis=0)
        state["sum_yy"] += (finite * (y[:, None] ** 2)).sum(axis=0)
        state["sum_xy"] += (safe * y[:, None]).sum(axis=0)


def correlations(state: dict[str, np.ndarray]) -> np.ndarray:
    n = state["count"]
    numerator = n * state["sum_xy"] - state["sum_x"] * state["sum_y"]
    denominator = np.sqrt((n * state["sum_xx"] - state["sum_x"] ** 2) *
                          (n * state["sum_yy"] - state["sum_y"] ** 2))
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                     where=denominator > 0)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    payload = joblib.load(MODEL_PATH)
    model, features = payload["model"], list(payload["features"])
    training = json.loads((FORWARD / "training_a.json").read_text(encoding="utf-8"))
    plan = json.loads((FORWARD / "split_plan.json").read_text(encoding="utf-8"))
    if len(features) != 85 or features != training["feature_names"]:
        raise AssertionError("Final artifact/training feature order mismatch")
    states = {"train": new_state(85), "validation": new_state(85)}
    counts = {"train": [0, 0], "validation": [0, 0]}

    # Fixed global Validation sample indices: uniform without replacement.
    validation_total = sum(int(r["validation_labeled_rows"]) for r in plan["tools"])
    rng = np.random.default_rng(42)
    global_pick = np.sort(rng.choice(validation_total, size=min(100000, validation_total), replace=False))
    sampled_x, sampled_y = [], []
    validation_cursor = 0

    for row in plan["tools"]:
        path = PREPARED / f"{row['tool_id']}.parquet"
        frame = pd.read_parquet(path, columns=[*features, "time", "label"])
        time = frame.time.to_numpy(dtype=np.int64)
        label = frame.label.to_numpy(dtype=np.int8)
        masks = {
            "train": (time < row["train_end_exclusive_second"]) & (label >= 0),
            "validation": ((time >= row["validation_start_second"]) &
                           (time < row["validation_end_exclusive_second"]) & (label >= 0)),
        }
        for split, mask in masks.items():
            update_state(states[split], frame, features, mask)
            counts[split][0] += int(mask.sum())
            counts[split][1] += int(((label == 1) & mask).sum())

        valid = frame.loc[masks["validation"], [*features, "label"]]
        stop = validation_cursor + len(valid)
        left = np.searchsorted(global_pick, validation_cursor, side="left")
        right = np.searchsorted(global_pick, stop, side="left")
        local = global_pick[left:right] - validation_cursor
        if len(local):
            sampled_x.append(valid.iloc[local][features].to_numpy(dtype=np.float32, copy=True))
            sampled_y.append(valid.iloc[local].label.to_numpy(dtype=np.uint8))
        validation_cursor = stop
        print(f"analysed {row['tool_id']}", flush=True)
    if validation_cursor != validation_total:
        raise AssertionError("Validation row count drift")

    train_corr = correlations(states["train"])
    valid_corr = correlations(states["validation"])
    metadata = pd.DataFrame({"feature": features,
                             "source_sensor": [source_sensor(f) for f in features],
                             "feature_group": [feature_group(f) for f in features],
                             "train_correlation": train_corr,
                             "validation_correlation": valid_corr,
                             "abs_train_correlation": np.abs(train_corr),
                             "abs_validation_correlation": np.abs(valid_corr),
                             "train_nonmissing_rows": states["train"]["count"].astype(np.int64),
                             "validation_nonmissing_rows": states["validation"]["count"].astype(np.int64)})
    OUT.mkdir(parents=True)
    metadata.sort_values("abs_validation_correlation", ascending=False).to_csv(
        OUT / "all85_target_correlation.csv", index=False, encoding="utf-8-sig")
    sensor_corr = metadata[metadata.feature.isin(SENSORS)].copy()
    sensor_corr["direction"] = np.where(sensor_corr.validation_correlation.abs() < .05, "near-zero",
                                np.where(sensor_corr.validation_correlation > 0, "positive", "negative"))
    sensor_corr[["feature", "train_correlation", "validation_correlation",
                 "abs_train_correlation", "abs_validation_correlation", "direction"]].rename(
        columns={"feature": "sensor"}).to_csv(OUT / "sensor_target_correlation.csv",
                                                index=False, encoding="utf-8-sig")

    booster = model.get_booster()
    raw_gain = booster.get_score(importance_type="gain")
    gains = np.zeros(85, dtype=np.float64)
    for i, feature in enumerate(features):
        gains[i] = float(raw_gain.get(feature, raw_gain.get(f"f{i}", 0.0)))
    gain_sum = gains.sum()
    gain_share = gains / gain_sum * 100 if gain_sum else gains
    gain = metadata[["feature", "source_sensor", "feature_group"]].copy()
    gain["gain_importance"] = gains
    gain["gain_share_percent"] = gain_share
    gain = gain.sort_values("gain_share_percent", ascending=False).reset_index(drop=True)
    gain.insert(0, "rank", np.arange(1, 86))
    gain.to_csv(OUT / "xgb_gain_importance_all85.csv", index=False, encoding="utf-8-sig")
    gain.head(20).to_csv(OUT / "xgb_gain_importance_top20.csv", index=False, encoding="utf-8-sig")

    x_sample, y_sample = np.concatenate(sampled_x), np.concatenate(sampled_y)
    if len(y_sample) != len(global_pick):
        raise AssertionError("Permutation sample construction failed")
    print(f"permutation sample={len(y_sample):,}, positives={int(y_sample.sum()):,}", flush=True)
    permutation = permutation_importance(model, x_sample, y_sample, scoring="average_precision",
                                         n_repeats=3, random_state=42, n_jobs=1)
    perm = metadata[["feature", "source_sensor", "feature_group"]].copy()
    perm["permutation_importance_mean"] = permutation.importances_mean
    perm["permutation_importance_std"] = permutation.importances_std
    perm["sample_rows"] = len(y_sample)
    perm["sample_positive_rows"] = int(y_sample.sum())
    perm.sort_values("permutation_importance_mean", ascending=False).to_csv(
        OUT / "permutation_importance.csv", index=False, encoding="utf-8-sig")

    integrated = metadata.merge(gain[["feature", "gain_importance", "gain_share_percent"]], on="feature").merge(
        perm[["feature", "permutation_importance_mean", "permutation_importance_std"]], on="feature")
    integrated = integrated.sort_values("gain_share_percent", ascending=False).reset_index(drop=True)
    integrated.insert(0, "rank", np.arange(1, 86))

    sensor_rows = []
    for sensor in SENSORS:
        related = integrated[integrated.source_sensor.eq(sensor)].sort_values("gain_share_percent", ascending=False)
        raw = metadata[metadata.feature.eq(sensor)].iloc[0]
        best = related.iloc[0]
        sensor_rows.append({"sensor": sensor, "train_target_correlation": raw.train_correlation,
                            "validation_target_correlation": raw.validation_correlation,
                            "highest_importance_feature": best.feature,
                            "highest_feature_gain_share_percent": best.gain_share_percent,
                            "total_sensor_related_gain_share_percent": related.gain_share_percent.sum(),
                            "related_feature_count": len(related)})
    sensor_summary = pd.DataFrame(sensor_rows).sort_values("total_sensor_related_gain_share_percent", ascending=False)
    sensor_summary.to_csv(OUT / "sensor_level_importance_summary.csv", index=False, encoding="utf-8-sig")

    # Numerical confirmation that point-biserial and Pearson are identical.
    checks = []
    for sensor in SENSORS:
        j = features.index(sensor)
        ok = np.isfinite(x_sample[:, j])
        p = float(pearsonr(x_sample[ok, j], y_sample[ok]).statistic)
        b = float(pointbiserialr(y_sample[ok], x_sample[ok, j]).statistic)
        checks.append(abs(p-b))

    top_corr = sensor_corr.sort_values("abs_validation_correlation", ascending=False).head(5)
    top_gain = integrated.head(20)
    top_sensor = sensor_summary.head(5)
    lines = ["Model 1 feature analysis (actual frozen artifacts)", "",
             f"Train rows={counts['train'][0]:,}; positives={counts['train'][1]:,}; ratio={counts['train'][1]/counts['train'][0]:.6%}",
             f"Validation rows={counts['validation'][0]:,}; positives={counts['validation'][1]:,}; ratio={counts['validation'][1]/counts['validation'][0]:.6%}",
             "Direction rule: abs(Validation correlation) < 0.05 => near-zero; otherwise sign determines positive/negative.",
             f"Point-biserial/Pearson maximum absolute difference on the fixed sample={max(checks):.12g}",
             f"Permutation importance: uniform Validation sample without replacement, rows={len(y_sample):,}, random_state=42, repeats=3, scoring=average_precision.",
             "Interaction gains are not double-counted in either single-sensor total.", "",
             "Top 5 raw sensors by absolute Validation target correlation:"]
    lines.extend(f"{i+1}. {r.feature}: train={r.train_correlation:.8f}, validation={r.validation_correlation:.8f}"
                 for i, r in enumerate(top_corr.itertuples()))
    lines.extend(["", "Top 20 integrated features by XGBoost gain share:"])
    lines.extend(f"{r.rank}. {r.feature} | {r.source_sensor} | {r.feature_group} | train_corr={r.train_correlation:.8f} | validation_corr={r.validation_correlation:.8f} | gain_share={r.gain_share_percent:.6f}% | permutation={r.permutation_importance_mean:.8f}"
                 for r in top_gain.itertuples())
    lines.extend(["", "Top 5 sensors by summed single-sensor feature gain share:"])
    lines.extend(f"{i+1}. {r.sensor}: total={r.total_sensor_related_gain_share_percent:.6f}%, best={r.highest_importance_feature} ({r.highest_feature_gain_share_percent:.6f}%)"
                 for i, r in enumerate(top_sensor.itertuples()))
    (OUT / "feature_analysis_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
