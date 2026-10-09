"""Evaluate thresholds on the saved 24h recovered baseline without fitting."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output" / "baseline_recovery"
OUT = ROOT / "output" / "experiments" / "threshold_sweep_24h"
TARGET_RECALLS = (0.50, 0.60, 0.70)


def load_validation(features: list[str], model) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels, scores, times, tools = [], [], [], []
    paths = sorted((BASE / "prepared").glob("??_M??.parquet"))
    if len(paths) != 20:
        raise ValueError(f"Expected 20 prepared real-data partitions; found {len(paths)}")
    for code, path in enumerate(paths):
        frame = pd.read_parquet(path, columns=[*features, "time", "label", "split"])
        valid = frame.loc[(frame["split"] == 2) & (frame["label"] >= 0)]
        if not valid["time"].is_monotonic_increasing:
            raise ValueError(f"{path.stem}: validation time is not sorted")
        x = valid[features].to_numpy(dtype=np.float32, copy=True)
        probability = model.predict_proba(x)[:, 1].astype(np.float32)
        labels.append(valid["label"].to_numpy(dtype=np.int8))
        scores.append(probability)
        times.append(valid["time"].to_numpy(dtype=np.int64))
        tools.append(np.full(len(valid), code, dtype=np.uint8))
        print(f"{path.stem}: {len(valid):,} validation probabilities", flush=True)
    return tuple(np.concatenate(parts) for parts in (labels, scores, times, tools))


def exact_candidate_thresholds(score: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Use unique observed scores, counting all rows tied at each threshold."""
    order = np.argsort(score, kind="stable")[::-1]
    sorted_score = score[order]
    sorted_truth = truth[order]
    end = np.flatnonzero(np.r_[sorted_score[1:] != sorted_score[:-1], True])
    tp = np.cumsum(sorted_truth, dtype=np.int64)[end]
    fp = (end + 1) - tp
    recall = tp / int(truth.sum())
    f1 = 2 * tp / np.maximum(2 * tp + fp + (int(truth.sum()) - tp), 1)
    result = {f"recall_{int(target*100)}pct": float(sorted_score[int(end[np.argmin(np.abs(recall - target))])]) for target in TARGET_RECALLS}
    result["maximum_f1"] = float(sorted_score[int(end[int(np.argmax(f1))])])
    return result


def metrics_at(threshold: float, truth: np.ndarray, score: np.ndarray, time: np.ndarray, tool: np.ndarray, ap: float) -> dict:
    alarm = score >= threshold
    positive = truth == 1
    tp = int(np.count_nonzero(alarm & positive))
    fp = int(np.count_nonzero(alarm & ~positive))
    fn = int(positive.sum()) - tp
    tn = len(truth) - int(positive.sum()) - fp
    false_positions = np.flatnonzero(alarm & ~positive)
    if len(false_positions):
        false_time = time[false_positions]
        false_tool = tool[false_positions]
        false_episodes = int(1 + np.count_nonzero((false_tool[1:] != false_tool[:-1]) | (np.diff(false_time) > 600)))
    else:
        false_episodes = 0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": float(threshold), "precision": precision, "recall": recall,
        "f1": f1, "balanced_accuracy": (recall + specificity) / 2,
        "average_precision": ap, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "false_alarm_events_10m": false_episodes,
    }


def assert_baseline(actual: dict, expected: dict, ap: float) -> None:
    for key in ("tn", "fp", "fn", "tp", "false_alarm_events_10m"):
        if actual[key] != expected[key]:
            raise AssertionError(f"Saved-model baseline mismatch for {key}: {actual[key]} != {expected[key]}")
    for key in ("precision", "recall", "f1", "balanced_accuracy"):
        if not np.isclose(actual[key], expected[key], atol=1e-12, rtol=0):
            raise AssertionError(f"Saved-model baseline mismatch for {key}: {actual[key]} != {expected[key]}")
    if not np.isclose(ap, expected["average_precision"], atol=1e-12, rtol=0):
        raise AssertionError(f"Saved-model AP mismatch: {ap} != {expected['average_precision']}")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    baseline_report = json.loads((BASE / "reproduction_report.json").read_text(encoding="utf-8"))
    features = baseline_report["features"]
    if len(features) != 85:
        raise ValueError("Recovered baseline feature count changed")
    model_path = BASE / "xgb_24h_recovery.joblib"
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    model = joblib.load(model_path)
    truth, score, time, tool = load_validation(features, model)
    if len(truth) != baseline_report["metrics"]["validation_rows"] or int(truth.sum()) != baseline_report["metrics"]["positive_rows"]:
        raise AssertionError("Validation population differs from recovered baseline")
    ap = float(average_precision_score(truth, score))
    baseline = metrics_at(0.5, truth, score, time, tool, ap)
    assert_baseline(baseline, baseline_report["metrics"], ap)
    print("threshold 0.5 exactly matches recovered baseline", flush=True)

    exact = exact_candidate_thresholds(score, truth)
    grid = np.linspace(0.0, 1.0, 201, dtype=np.float64)
    thresholds = sorted({float(t) for t in grid} | {0.5} | set(exact.values()))
    rows = [metrics_at(t, truth, score, time, tool, ap) for t in thresholds]
    for row in rows:
        row["delta_recall_vs_0_5"] = row["recall"] - baseline["recall"]
        row["delta_precision_vs_0_5"] = row["precision"] - baseline["precision"]
        row["delta_false_alarm_events_vs_0_5"] = row["false_alarm_events_10m"] - baseline["false_alarm_events_10m"]
    selected = []
    for name, threshold in [("baseline_0_5", 0.5), *exact.items()]:
        row = next(row for row in rows if row["threshold"] == threshold)
        selected.append({"candidate": name, **row})

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "validation_probabilities.npz", truth=truth, probability=score, time=time, tool_code=tool)
    write_csv(OUT / "threshold_metrics.csv", rows)
    write_csv(OUT / "selected_thresholds.csv", selected)
    report = {
        "model_path": str(model_path), "model_sha256": model_hash,
        "model_retrained": False, "features_changed": False,
        "source_report": str(BASE / "reproduction_report.json"),
        "tool_code_map": {str(i): path.stem for i, path in enumerate(sorted((BASE / "prepared").glob("??_M??.parquet")))},
        "validation_rows": len(truth), "validation_positive_rows": int(truth.sum()),
        "threshold_grid": "0.000 to 1.000 inclusive at 0.005 increments, plus exact observed-score candidates",
        "alarm_merge": "per tool, each normal-row alarm starts a new episode only when >600 seconds after the prior false alarm",
        "average_precision_fixed": ap,
        "baseline_0_5_exact_match": True,
        "candidate_thresholds": exact,
        "selected": selected,
        "interpretation_note": "Candidate thresholds and maximum F1 were selected on this same validation set. Treat these as exploratory; verify a chosen threshold on a later untouched period before claiming improved generalization.",
    }
    (OUT / "sweep_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in selected:
        print(f"{row['candidate']}: threshold={row['threshold']:.8f}, recall={row['recall']:.4f}, precision={row['precision']:.4f}, F1={row['f1']:.4f}, false_events={row['false_alarm_events_10m']}", flush=True)
    print(f"Saved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
