"""Fit independent 5-minute LSTM TTF models on all eligible real train minutes.

This experiment uses the frozen 20-tool chronological 80/20 split and 24-hour
purge from the recovered classifier. Targets are derived from the real fault
log, in wall-clock seconds. Train targets are restricted to events before each
tool's split cutoff so validation fault events never occur in training labels.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from recover_xgb24_baseline import ROOT, FAULTS


DATA = ROOT / "output" / "baseline_recovery"
OUT = ROOT / "output" / "experiments" / "ttf_lstm_100pct_12h_24h"
MODELS = ROOT / "output" / "models"
SEED = 42
SEQ = 5
EPOCHS = 3
BATCH = 2048


class TtfLstm(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.lstm = nn.LSTM(width, 16, batch_first=True)
        self.head = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states, _ = self.lstm(x)
        return self.head(states[:, -1]).squeeze(-1)


def next_fault(now: np.ndarray, faults: np.ndarray) -> np.ndarray:
    result = np.full(len(now), np.nan, dtype=np.float64)
    indices = np.searchsorted(faults, now, side="left")
    okay = indices < len(faults)
    result[okay] = faults[indices[okay]]
    return result


def load_tool(path: Path, features: list[str], fault_log: pd.DataFrame,
              cutoff: int, fault: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_parquet(path, columns=[*features, "time", "split", "label"])
    time = frame.time.to_numpy(dtype=np.int64)
    split = frame.split.to_numpy(dtype=np.int8)
    event_times = np.sort(fault_log.loc[
        (fault_log.tool_id == path.stem) & (fault_log.fault_type == fault), "time"
    ].drop_duplicates().to_numpy(dtype=np.float64))
    target_event = next_fault(time, event_times)
    train = (split == 0) & np.isfinite(target_event) & (target_event < cutoff)
    valid = (split == 2) & (frame.label.to_numpy() >= 0) & np.isfinite(target_event) & (target_event <= time[-1])
    # A stop longer than two minutes starts a new history. No sequence crosses it.
    starts = np.maximum.accumulate(np.where(np.r_[True, np.diff(time) > 120], np.arange(len(time)), 0))
    x = frame[features].to_numpy(dtype=np.float32, copy=True)
    return x, (target_event - time).astype(np.float32), np.flatnonzero(train), np.flatnonzero(valid), starts


def batches(x: np.ndarray, starts: np.ndarray, indices: np.ndarray,
            mean: np.ndarray, std: np.ndarray):
    for begin in range(0, len(indices), BATCH):
        selected = indices[begin:begin + BATCH]
        steps = np.maximum(selected[:, None] - np.arange(SEQ - 1, -1, -1), starts[selected, None])
        values = x[steps]
        values = np.clip((values - mean) / std, -10, 10)
        np.nan_to_num(values, copy=False, nan=0, posinf=0, neginf=0)
        yield selected, torch.from_numpy(values)


def metric(y: np.ndarray, estimate: np.ndarray) -> dict:
    error = estimate - y
    mse = float(np.mean(error.astype(np.float64) ** 2))
    denominator = float(np.sum((y - float(np.mean(y))) ** 2, dtype=np.float64))
    return {"rows": len(y), "mae_seconds": float(np.mean(np.abs(error))),
            "mae_hours": float(np.mean(np.abs(error)) / 3600),
            "rmse_seconds": float(np.sqrt(mse)),
            "r2": float(1 - np.sum(error.astype(np.float64) ** 2) / denominator) if denominator > 0 else None}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    report = json.loads((DATA / "reproduction_report.json").read_text(encoding="utf-8"))
    prep = json.loads((DATA / "preparation_report.json").read_text(encoding="utf-8"))
    features = report["features"]
    cutoffs = {item["tool"]: item["split_cutoff_time"] for item in prep["tools"]}
    files = sorted((DATA / "prepared").glob("??_M??.parquet"))
    if len(files) != 20:
        raise ValueError("Expected all 20 real train tools")
    fault_log = pd.read_csv(ROOT / "data" / "processed" / "fault_events.csv", usecols=["tool_id", "fault_type", "time"])
    OUT.mkdir(parents=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    summary = []
    for fault in FAULTS:
        # Statistics come only from eligible training rows for this fault.
        sum_x = np.zeros(len(features), dtype=np.float64)
        sum_sq = np.zeros(len(features), dtype=np.float64)
        counts = np.zeros(len(features), dtype=np.int64)
        train_rows = 0
        for path in files:
            x, target, train, valid, starts = load_tool(path, features, fault_log, cutoffs[path.stem], fault)
            finite = np.isfinite(x[train])
            values = np.where(finite, x[train], 0).astype(np.float64)
            sum_x += values.sum(axis=0)
            sum_sq += np.square(values).sum(axis=0)
            counts += finite.sum(axis=0)
            train_rows += len(train)
            print(f"{fault} stats {path.stem}: train={len(train):,}, valid={len(valid):,}", flush=True)
        mean = sum_x / np.maximum(counts, 1)
        variance = sum_sq / np.maximum(counts, 1) - mean * mean
        std = np.sqrt(np.maximum(variance, 1e-6))
        mean, std = mean.astype(np.float32), std.astype(np.float32)
        model = TtfLstm(len(features))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(EPOCHS):
            model.train()
            total_loss, total_rows = 0.0, 0
            for path in files:
                x, target, train, valid, starts = load_tool(path, features, fault_log, cutoffs[path.stem], fault)
                # Every eligible training minute is used once per epoch.
                rng = np.random.default_rng(SEED + epoch + 100 * FAULTS.index(fault) + 1000 * int(path.stem[:2]))
                rng.shuffle(train)
                for selected, sequence in batches(x, starts, train, mean, std):
                    expected = torch.from_numpy(np.log1p(target[selected]))
                    optimizer.zero_grad()
                    output = model(sequence)
                    loss = nn.functional.smooth_l1_loss(output, expected)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss) * len(selected)
                    total_rows += len(selected)
            print(f"{fault} epoch {epoch+1}/{EPOCHS}: rows={total_rows:,}, log-loss={total_loss/total_rows:.4f}", flush=True)
        model.eval()
        all_truth, all_pred = [], []
        for path in files:
            x, target, train, valid, starts = load_tool(path, features, fault_log, cutoffs[path.stem], fault)
            estimates = []
            with torch.no_grad():
                for selected, sequence in batches(x, starts, valid, mean, std):
                    estimates.append(np.expm1(model(sequence).numpy()))
            if estimates:
                all_truth.append(target[valid])
                all_pred.append(np.concatenate(estimates))
        truth = np.concatenate(all_truth)
        prediction = np.concatenate(all_pred)
        row = {"model": "LSTM", "fault_type": fault, "train_rows": train_rows,
               "validation_rows": len(truth), "epochs": EPOCHS, "sequence_minutes": SEQ,
               **metric(truth, prediction)}
        summary.append(row)
        torch.save({"state_dict": model.state_dict(), "features": features,
                    "mean": mean, "std": std, "sequence_minutes": SEQ,
                    "fault_type": fault, "epochs": EPOCHS}, MODELS / f"lstm_100pct_{fault}.pt")
        print(json.dumps(row), flush=True)
    with (OUT / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (OUT / "conditions.json").write_text(json.dumps({
        "source": "20 real train tools, 1-minute aggregation of all observed 4-second rows",
        "split": "per-tool chronological first 80% / last 20%, 24-hour pre-cutoff purge",
        "training_targets": "next real same-type logged fault before validation cutoff; all valid train rows, no classifier gate",
        "validation_targets": "next real same-type logged fault within observed series; censored rows excluded",
        "features": features, "epochs": EPOCHS, "sequence_minutes": SEQ, "seed": SEED,
        "limitations": "Validation target population differs by fault type; no independent final test; full 4-second rows are aggregated to minutes.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
