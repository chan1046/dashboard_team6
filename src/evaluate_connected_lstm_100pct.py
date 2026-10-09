"""Audit minimum-predicted-TTF fault typing behind frozen 12h/24h XGBoost gates."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from compare_ttf_lstm_100pct import TtfLstm, batches, next_fault
from recover_xgb24_baseline import FAULTS, ROOT

OUT = ROOT / "output" / "experiments" / "connected_lstm_100pct_12h_24h"
PREP = ROOT / "output" / "baseline_recovery"
MODELS = ROOT / "output" / "models"
COMPARE = ROOT / "output" / "experiments" / "horizon_comparison_12h_24h"
SWEEP = ROOT / "output" / "experiments" / "threshold_sweep_24h"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    torch.set_num_threads(4)
    with np.load(COMPARE / "validation_probabilities_12h.npz") as saved:
        p12, t12, tool12 = saved["probability"].copy(), saved["time"].copy(), saved["tool_code"].copy()
    with np.load(SWEEP / "validation_probabilities.npz") as saved:
        p24, t24, tool24 = saved["probability"].copy(), saved["time"].copy(), saved["tool_code"].copy()
    if not (np.array_equal(t12, t24) and np.array_equal(tool12, tool24)):
        raise AssertionError("Classification validation rows do not align")
    source = sorted((PREP / "prepared").glob("??_M??.parquet"))
    if len(source) != 20:
        raise AssertionError("Missing one or more real tools")
    report = json.loads((PREP / "reproduction_report.json").read_text(encoding="utf-8"))
    features = report["features"]
    prep = json.loads((PREP / "preparation_report.json").read_text(encoding="utf-8"))
    cutoffs = {r["tool"]: r["split_cutoff_time"] for r in prep["tools"]}
    logs = pd.read_csv(ROOT / "data" / "processed" / "fault_events.csv", usecols=["tool_id", "fault_type", "time"])
    models = []
    for fault in FAULTS:
        state = torch.load(MODELS / f"lstm_100pct_{fault}.pt", map_location="cpu", weights_only=False)
        if state["features"] != features:
            raise AssertionError("LSTM feature order differs from baseline")
        network = TtfLstm(len(features))
        network.load_state_dict(state["state_dict"])
        network.eval()
        models.append((network, state["mean"], state["std"]))
    event_rows = []
    aggregate = {h: {"alarm_positive_rows": 0, "type_correct_rows": 0,
                     "rul_absolute_error_sum": 0.0, "rul_rows": 0} for h in (12, 24)}
    for code, path in enumerate(source):
        frame = pd.read_parquet(path, columns=[*features, "time", "split", "label"])
        full_time = frame.time.to_numpy(dtype=np.int64)
        valid_idx = np.flatnonzero((frame.split.to_numpy() == 2) & (frame.label.to_numpy() >= 0))
        t = full_time[valid_idx]
        if not np.array_equal(t, t12[tool12 == code]):
            raise AssertionError(f"{path.stem}: validation time differs from frozen probabilities")
        x = frame[features].to_numpy(dtype=np.float32, copy=True)
        starts = np.maximum.accumulate(np.where(np.r_[True, np.diff(full_time) > 120], np.arange(len(full_time)), 0))
        estimates = []
        for model, mean, std in models:
            parts = []
            with torch.no_grad():
                for selected, sequence in batches(x, starts, valid_idx, mean, std):
                    parts.append(np.expm1(model(sequence).numpy()))
            estimates.append(np.concatenate(parts))
        estimate = np.column_stack(estimates)
        chosen = np.argmin(estimate, axis=1)
        true_events = []
        for fault in FAULTS:
            fault_times = np.sort(logs.loc[(logs.tool_id == path.stem) & (logs.fault_type == fault), "time"].drop_duplicates().to_numpy(dtype=np.float64))
            true_events.append(next_fault(t, fault_times))
        true_events = np.column_stack(true_events)
        nearest_type = np.argmin(np.where(np.isfinite(true_events), true_events, np.inf), axis=1)
        nearest_time = true_events[np.arange(len(t)), nearest_type]
        true_ttf = nearest_time - t
        for h, score in ((12, p12[tool12 == code]), (24, p24[tool24 == code])):
            positive_alarm = (score >= 0.5) & np.isfinite(true_ttf) & (true_ttf <= h * 3600)
            aggregate[h]["alarm_positive_rows"] += int(positive_alarm.sum())
            aggregate[h]["type_correct_rows"] += int(np.sum(chosen[positive_alarm] == nearest_type[positive_alarm]))
            aggregate[h]["rul_absolute_error_sum"] += float(np.sum(np.abs(estimate[positive_alarm, chosen[positive_alarm]] - true_ttf[positive_alarm]), dtype=np.float64))
            aggregate[h]["rul_rows"] += int(positive_alarm.sum())
        logged = logs.loc[(logs.tool_id == path.stem) & (logs.time >= cutoffs[path.stem])]
        for _, event in logged.iterrows():
            fault_index = FAULTS.index(event.fault_type)
            for h, score in ((12, p12[tool12 == code]), (24, p24[tool24 == code])):
                left = np.searchsorted(t, event.time - h * 3600, side="left")
                right = np.searchsorted(t, event.time, side="left")
                alert = np.flatnonzero(score[left:right] >= 0.5)
                first = left + alert[0] if len(alert) else None
                event_rows.append({"horizon_hours": h, "tool_id": path.stem, "fault_type": event.fault_type,
                                   "fault_time": int(event.time), "evaluable": bool(right > left),
                                   "detected": first is not None,
                                   "first_alert_lead_minutes": (event.time - t[first]) / 60 if first is not None else None,
                                   "predicted_type_at_first_alert": FAULTS[int(chosen[first])] if first is not None else None,
                                   "type_correct_at_first_alert": bool(chosen[first] == fault_index) if first is not None else False,
                                   "predicted_ttf_hours_at_first_alert": float(estimate[first, chosen[first]] / 3600) if first is not None else None})
        print(f"inferred {path.stem}: validation rows={len(t):,}", flush=True)
    if len(event_rows) != 2 * 181:
        raise AssertionError(f"Expected 181 logged validation fault events per horizon, got {len(event_rows)}")
    summary = []
    for h in (12, 24):
        events = [r for r in event_rows if r["horizon_hours"] == h]
        detected = sum(r["detected"] for r in events)
        correct = sum(r["type_correct_at_first_alert"] for r in events)
        counted = aggregate[h]
        summary.append({"horizon_hours": h, "logged_fault_events": len(events),
                        "detected_fault_events": detected, "type_correct_events": correct,
                        "type_accuracy_among_detected_events": correct / detected if detected else None,
                        "type_correct_over_all_fault_events": correct / len(events),
                        "alarm_positive_rows": counted["alarm_positive_rows"],
                        "row_type_accuracy_on_detected_positives": counted["type_correct_rows"] / counted["alarm_positive_rows"] if counted["alarm_positive_rows"] else None,
                        "selected_min_ttf_mae_hours_on_detected_positives": counted["rul_absolute_error_sum"] / counted["rul_rows"] / 3600 if counted["rul_rows"] else None})
    OUT.mkdir(parents=True)
    write_csv(OUT / "connected_metrics.csv", summary)
    write_csv(OUT / "fault_event_details.csv", event_rows)
    (OUT / "conditions.json").write_text(json.dumps({"threshold": 0.5,
        "models": "Frozen full-data 12/24h XGBoost gates + independently trained full-data LSTM TTF models",
        "type_rule": "argmin of three predicted wall-clock TTF values at first alarm before each fault",
        "limitation": "Validation only; this selection rule was not tuned on an independent test. Large TTF regression error may bias argmin."},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
