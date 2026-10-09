"""Train a next-Low-Pressure RUL baseline on the frozen Train/Validation split."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from audit_simple_full_fault_rul import candidates
from train_simple_full_fault_rul import SENSORS


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/processed"
PLAN = ROOT / "output/experiments/forward_validation_ab_24h/split_plan.json"
AUDIT = ROOT / "output/experiments/low_pressure_rul_audit"
OUT = ROOT / "output/experiments/low_pressure_rul_baseline"
MODELS = ROOT / "output/models"
SEED = 42
HISTORY_SECONDS = 1800
CADENCE_SECONDS = 1800
SEQUENCE_POINTS = 60
TRAIN_EVENT_CAP = 32
BINS = [0, 3, 6, 12, 24, 72, 168, np.inf]
BIN_NAMES = ["0-3h", "3-6h", "6-12h", "12-24h", "24-72h", "72-168h", "168h+"]


class LowGRU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(9, 32, batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.gru(x)[1][-1]).squeeze(-1)


def build() -> tuple[dict, pd.DataFrame]:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    faults = pd.read_csv(RAW / "fault_events.csv").drop_duplicates(["tool_id", "time", "fault_type"])
    assert len(faults) == 1211
    low = faults[faults.fault_type == "pressure_low"]
    assert len(low) == 741 and low.fault_name.unique().tolist() == ["FlowCool Pressure Dropped Below Limit"]
    data = {phase: {"x": [], "y": [], "meta": []} for phase in ("train", "validation")}
    counts = []
    for item in plan["tools"]:
        tool = item["tool_id"]
        end = int(item["validation_end_exclusive_second"])
        frame = pq.read_table(RAW / f"{tool}_sensor_ttf.parquet", columns=["time"] + SENSORS,
                              filters=[("time", "<", end)]).to_pandas()
        t = frame.time.to_numpy(dtype=np.int64)
        x = frame[SENSORS].to_numpy(dtype=np.float32)
        assert np.all(t[1:] >= t[:-1]), tool
        log = faults[faults.tool_id == tool].sort_values("time", kind="stable")
        other_t = log.loc[log.fault_type != "pressure_low", "time"].to_numpy(dtype=np.int64)
        prev_low = None
        for event in log[log.fault_type == "pressure_low"].itertuples():
            f = int(event.time)
            if f >= end:
                break
            if f < item["train_end_exclusive_second"]:
                phase = "train"
                phase_start = int(t[0])
                phase_end = int(item["train_end_exclusive_second"])
            elif item["validation_start_second"] <= f < end:
                phase = "validation"
                phase_start = int(item["validation_start_second"])
                phase_end = end
            else:
                prev_low = f
                continue
            lo = max(int(np.searchsorted(t, phase_start, side="left")),
                     0 if prev_low is None else int(np.searchsorted(t, prev_low, side="right")))
            hi = min(int(np.searchsorted(t, f, side="left")),
                     int(np.searchsorted(t, phase_end, side="left")))
            anchors = candidates(t, lo, hi, HISTORY_SECONDS, CADENCE_SECONDS)
            original_count = len(anchors)
            if phase == "train" and original_count > TRAIN_EVENT_CAP:
                selected = np.unique(np.rint(np.linspace(0, original_count - 1, TRAIN_EVENT_CAP)).astype(int))
                anchors = anchors[selected]
            counts.append({"tool_id": tool, "phase": phase, "fault_event_id": int(event.fault_event_id),
                           "candidate_samples": original_count, "retained_samples": len(anchors)})
            for a in anchors:
                a = int(a)
                start = max(lo, int(np.searchsorted(t, t[a] - HISTORY_SECONDS, side="left")))
                assert a - start + 1 >= 60 and t[a] - t[start] >= .8 * HISTORY_SECONDS
                idx = np.linspace(start, a, SEQUENCE_POINTS, dtype=np.int64)
                assert len(idx) == SEQUENCE_POINTS and t[idx[-1]] <= t[a] < f
                assert prev_low is None or t[idx[0]] > prev_low
                assert t[idx[0]] >= phase_start
                target = (f - int(t[a])) / 3600
                assert target > 0
                next_other_idx = int(np.searchsorted(other_t, t[a], side="right"))
                intervenes = bool(next_other_idx < len(other_t) and other_t[next_other_idx] < f)
                gaps = np.diff(t[start:a+1])
                data[phase]["x"].append(x[idx].copy())
                data[phase]["y"].append(target)
                data[phase]["meta"].append({"tool_id": tool, "fault_event_id": int(event.fault_event_id),
                    "anchor_time": int(t[a]), "fault_time": f, "target_ttf_hours": target,
                    "intervening_other_fault": intervenes, "history_first_time": int(t[start]),
                    "history_elapsed_seconds": int(t[a] - t[start]),
                    "history_max_internal_gap_seconds": int(gaps.max()) if len(gaps) else 0,
                    "phase": phase})
            prev_low = f
        print("built", tool, flush=True)
    for phase in data:
        data[phase]["x"] = np.stack(data[phase]["x"]).astype(np.float32)
        data[phase]["y"] = np.asarray(data[phase]["y"], dtype=np.float32)
        data[phase]["meta"] = pd.DataFrame(data[phase]["meta"])
    return data, pd.DataFrame(counts)


def summary_features(x: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    current = x[:, -1, :]
    first = x[:, 0, :]
    mean = np.nanmean(x, axis=1)
    std = np.nanstd(x, axis=1)
    minimum = np.nanmin(x, axis=1)
    maximum = np.nanmax(x, axis=1)
    elapsed = np.maximum(meta.history_elapsed_seconds.to_numpy(dtype=np.float32) / 3600, 1e-6)
    slope = (current - first) / elapsed[:, None]
    return np.concatenate([current, mean, std, minimum, maximum, slope], axis=1).astype(np.float32)


def event_weights(ids: pd.Series) -> np.ndarray:
    counts = ids.value_counts()
    w = ids.map(1 / counts).to_numpy(dtype=np.float32)
    return w * (len(w) / w.sum())


def standardize(x: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num((x - center) / scale, nan=0, posinf=0, neginf=0), -20, 20).astype(np.float32)


def fit_gru(x: np.ndarray, y: np.ndarray, weights: np.ndarray, vx: np.ndarray,
            target_scale: float) -> tuple[LowGRU, dict, np.ndarray]:
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    flat = x.reshape(-1, x.shape[-1])
    center = np.nanmedian(flat, axis=0).astype(np.float32)
    scale = np.maximum(np.nanpercentile(flat, 75, axis=0) - np.nanpercentile(flat, 25, axis=0), 1e-5).astype(np.float32)
    tx = standardize(x, center, scale)
    val = standardize(vx, center, scale)
    dataset = TensorDataset(torch.from_numpy(tx), torch.from_numpy(y / target_scale),
                            torch.from_numpy(weights.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    model = LowGRU()
    optim = torch.optim.Adam(model.parameters(), lr=.001)
    losses = []
    for epoch in range(5):
        model.train()
        batch_losses = []
        for bx, by, bw in loader:
            optim.zero_grad()
            pred = model(bx)
            loss = (nn.functional.smooth_l1_loss(pred, by, reduction="none") * bw).sum() / bw.sum()
            loss.backward()
            optim.step()
            batch_losses.append(float(loss))
        losses.append(float(np.mean(batch_losses)))
        print("GRU epoch", epoch + 1, losses[-1], flush=True)
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(val), 512):
            predictions.append(model(torch.from_numpy(val[i:i+512])).numpy())
    result = np.maximum(0, np.concatenate(predictions) * target_scale)
    scaler = {"center": center, "scale": scale, "target_scale": target_scale, "losses": losses}
    return model, scaler, result


def safe_corr(y: np.ndarray, p: np.ndarray, method: str) -> float | None:
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return None
    value = pearsonr(y, p).statistic if method == "pearson" else spearmanr(y, p).statistic
    return float(value) if np.isfinite(value) else None


def metrics(meta: pd.DataFrame, p: np.ndarray) -> dict:
    y = meta.target_ttf_hours.to_numpy(dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if not len(y):
        return {"rows": 0, "events": 0}
    error = p - y
    ae = np.abs(error)
    frame = pd.DataFrame({"event": meta.fault_event_id.to_numpy(), "abs_error": ae,
                          "squared_error": error ** 2, "actual": y, "predicted": p})
    event = frame.groupby("event", sort=False).agg(mae=("abs_error", "mean"),
        mse=("squared_error", "mean"), actual_median=("actual", "median"),
        predicted_median=("predicted", "median"))
    return {"rows": len(y), "events": len(event), "row_mae_h": float(ae.mean()),
        "row_rmse_h": float(np.sqrt(np.mean(error ** 2))),
        "row_r2": float(r2_score(y, p)) if len(y) > 1 else None,
        "row_smape_pct": float(np.mean(200 * ae / (np.abs(y) + np.abs(p) + 1e-6))),
        "event_equal_mae_h": float(event.mae.mean()),
        "event_equal_rmse_h": float(np.sqrt(event.mse.mean())),
        "event_median_mae_h": float(event.mae.median()),
        "row_pearson": safe_corr(y, p, "pearson"),
        "row_spearman": safe_corr(y, p, "spearman"),
        "event_pearson": safe_corr(event.actual_median.to_numpy(), event.predicted_median.to_numpy(), "pearson"),
        "event_spearman": safe_corr(event.actual_median.to_numpy(), event.predicted_median.to_numpy(), "spearman"),
        "prediction_median_h": float(np.median(p))}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    for path in (MODELS / "low_pressure_rul_tree.joblib", MODELS / "low_pressure_rul_gru.pt",
                 MODELS / "low_pressure_rul_scaler.pkl"):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    audit = pd.read_csv(AUDIT / "anchor_summary.csv")
    data, counts = build()
    tr, va = data["train"], data["validation"]
    trmeta, vameta = tr["meta"], va["meta"]
    assert set(trmeta.fault_event_id).isdisjoint(set(vameta.fault_event_id))
    for phase, part in (("train", tr), ("validation", va)):
        expected = audit[(audit.rule == "A") & (audit.history_minutes == 30) &
                         (audit.cadence_minutes == 30) & (audit.phase == phase)]
        assert int(expected.candidate_samples.iloc[0]) == int(counts[counts.phase == phase].candidate_samples.sum())
        assert int(expected.usable_events.iloc[0]) == part["meta"].fault_event_id.nunique()
    assert len(trmeta) <= 32 * trmeta.fault_event_id.nunique()
    assert len(counts[counts.phase == "train"]) == 461
    assert len(counts[counts.phase == "validation"]) == 186
    weights = event_weights(trmeta.fault_event_id)
    sums = pd.DataFrame({"event": trmeta.fault_event_id, "weight": weights}).groupby("event").weight.sum()
    assert np.allclose(sums.to_numpy(), sums.iloc[0], rtol=1e-5)
    event_median = trmeta.groupby("fault_event_id").target_ttf_hours.median()
    dummy_hour = float(event_median.median())
    dummy = np.full(len(vameta), dummy_hour, dtype=np.float32)
    tree = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=31,
        learning_rate=.05, random_state=SEED, l2_regularization=1)
    tree.fit(summary_features(tr["x"], trmeta), tr["y"], sample_weight=weights)
    tree_pred = np.maximum(0, tree.predict(summary_features(va["x"], vameta)))
    gru, scaler, gru_pred = fit_gru(tr["x"], tr["y"], weights, va["x"], dummy_hour)
    predictions = {"Dummy": dummy, "Tree": tree_pred, "GRU": gru_pred}
    OUT.mkdir(parents=True)
    counts.to_csv(OUT / "anchor_counts_by_event.csv", index=False)
    trmeta.to_csv(OUT / "train_sample_metadata.csv.gz", index=False, compression="gzip")
    validation = vameta.copy()
    for name, pred in predictions.items():
        validation[name + "_prediction_hours"] = pred
    validation.to_csv(OUT / "validation_predictions.csv.gz", index=False, compression="gzip")
    joblib.dump({"model": tree, "sensors": SENSORS, "feature_order":
        [f"{stat}_{sensor}" for stat in ("current", "mean", "std", "min", "max", "slope_per_hour") for sensor in SENSORS],
        "history_seconds": HISTORY_SECONDS}, MODELS / "low_pressure_rul_tree.joblib")
    with (MODELS / "low_pressure_rul_scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle)
    torch.save({"state_dict": gru.state_dict(), "architecture": "GRU9x32+linear1",
        "sensor_order": SENSORS, "history_seconds": HISTORY_SECONDS,
        "sequence_points": SEQUENCE_POINTS, "scaler_file": "low_pressure_rul_scaler.pkl"},
        MODELS / "low_pressure_rul_gru.pt")
    metric_rows, bin_rows, tool_rows, middle_rows = [], [], [], []
    for name, pred in predictions.items():
        for subset, mask in [("all", np.ones(len(vameta), dtype=bool))] + [
            (f"ttf_le_{h}h", va["y"] <= h) for h in (24, 12, 6, 3)]:
            metric_rows.append({"model": name, "subset": subset,
                                **metrics(vameta.loc[mask], np.asarray(pred)[mask])})
        for lo, hi, label in zip(BINS[:-1], BINS[1:], BIN_NAMES):
            mask = (va["y"] >= lo) & (va["y"] < hi)
            actual = va["y"][mask]
            forecast = np.asarray(pred)[mask]
            bin_rows.append({"model": name, "ttf_bin": label, "samples": len(actual),
                "unique_events": vameta.loc[mask, "fault_event_id"].nunique(),
                "actual_median_h": float(np.median(actual)) if len(actual) else None,
                "prediction_median_h": float(np.median(forecast)) if len(actual) else None,
                "mae_h": float(np.mean(np.abs(forecast - actual))) if len(actual) else None,
                "mean_bias_h": float(np.mean(forecast - actual)) if len(actual) else None})
        for tool, indices in vameta.groupby("tool_id").groups.items():
            ix = np.asarray(list(indices), dtype=np.int64)
            tool_rows.append({"model": name, "tool_id": tool, **metrics(vameta.iloc[ix], np.asarray(pred)[ix])})
        for label, mask in [("03_M02", vameta.tool_id.eq("03_M02")),
                            ("excluding_03_M02", vameta.tool_id.ne("03_M02")),
                            ("no_other_fault", ~vameta.intervening_other_fault),
                            ("other_fault_before_low", vameta.intervening_other_fault)]:
            middle_rows.append({"model": name, "group": label,
                                **metrics(vameta.loc[mask], np.asarray(pred)[mask])})
    pd.DataFrame(metric_rows).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(OUT / "ttf_bin_metrics.csv", index=False)
    pd.DataFrame(tool_rows).to_csv(OUT / "tool_metrics.csv", index=False)
    pd.DataFrame(middle_rows).to_csv(OUT / "tool_and_intervening_fault_metrics.csv", index=False)
    config = {"split": str(PLAN.relative_to(ROOT)), "target": "next Low Pressure fault minus anchor, hours",
        "fault_name": "FlowCool Pressure Dropped Below Limit", "intervening_fault": "target unchanged; flag metadata only",
        "history_seconds": HISTORY_SECONDS, "cadence_seconds": CADENCE_SECONDS,
        "sequence_points": SEQUENCE_POINTS, "train_cap_per_event": TRAIN_EVENT_CAP,
        "train_selection": "up to 32 uniformly spaced chronological candidate anchors per event, covering its observed TTF range",
        "validation_selection": "all eligible 30-minute anchors; no cap; same causal 30-minute history rule",
        "event_weight": "1/N per retained Train event sample, globally normalized to mean 1",
        "dummy": "median of each Train event's retained-anchor median TTF",
        "dummy_hours": dummy_hour, "tree": "HGB 120 trees, 31 leaves, learning rate .05, l2=1, seed 42",
        "gru": "one-layer GRU hidden32, 5 epochs, Adam .001, weighted SmoothL1 on TTF/Train dummy median",
        "sensor_order": SENSORS, "random_state": SEED, "no_model1_change": True,
        "reserved_or_purge_access": "none for training or validation; raw parquet filter time < validation_end",
        "train_samples": len(trmeta), "validation_samples": len(vameta),
        "train_events": trmeta.fault_event_id.nunique(),
        "validation_events": vameta.fault_event_id.nunique()}
    (OUT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: config[k] for k in ("train_samples", "validation_samples", "train_events", "validation_events", "dummy_hours")}, indent=2))
    print(pd.DataFrame(metric_rows).query('subset == "all"').to_string(index=False))


if __name__ == "__main__":
    main()
