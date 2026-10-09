"""Build dashboard replay files from frozen validation predictions only."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from dashboard_inference import TOOLS, status_from_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/dashboard"
REFERENCE_TIME = 27317822
EXCLUDED_TOOL = "02_M01"


def nearest_index(times, indices, timestamp):
    ordered = indices[np.argsort(times[indices])]
    local = times[ordered]
    position = int(np.searchsorted(local, timestamp))
    candidates = []
    if position < len(local):
        candidates.append(ordered[position])
    if position:
        candidates.append(ordered[position - 1])
    return min(candidates, key=lambda i: (abs(int(times[i]) - timestamp), int(times[i])))


def build_fleet():
    source = ROOT / "output/experiments/forward_validation_ab_24h/validation_probabilities_a.npz"
    with np.load(source) as saved:
        times = saved["time"].astype(np.int64)
        codes = saved["tool_code"].astype(int)
        scores = saved["probability"]
        truth = saved["truth"].astype(int)
    tools = {}
    for code, tool in enumerate(TOOLS):
        if tool == EXCLUDED_TOOL:
            tools[tool] = {
                "model1": {"available": False, "prediction_timestamp": None,
                           "timestamp_gap_seconds": None, "risk_score": None,
                           "status": "판단 보류", "actual_label": None,
                           "reason": "비가동/점검으로 해당 시점 데이터 없음"},
                "priority": None,
            }
            continue
        index = nearest_index(times, np.flatnonzero(codes == code), REFERENCE_TIME)
        score = float(scores[index])
        tools[tool] = {
            "model1": {"available": True, "prediction_timestamp": int(times[index]),
                       "timestamp_gap_seconds": abs(int(times[index]) - REFERENCE_TIME),
                       "risk_score": score, "status": status_from_score(score),
                       "actual_label": int(truth[index])},
            "priority": None,
        }
    ranked = sorted((t for t in TOOLS if tools[t]["model1"]["available"]),
                    key=lambda t: (-tools[t]["model1"]["risk_score"], t))
    for priority, tool in enumerate(ranked, 1):
        tools[tool]["priority"] = priority
    return {
        "metadata": {"mode": "저장된 실제 검증 결과 재생", "reference_time": REFERENCE_TIME,
                     "model": "Model 1", "source": str(source.relative_to(ROOT)).replace("\\", "/"),
                     "same_exact_timestamp": False},
        "tools": tools,
    }


def build_fault_types(model1_source):
    folder = ROOT / "output/experiments/full_fault_type_multi_anchor"
    predictions = pd.read_csv(folder / "validation_predictions.csv")
    anchors = pd.read_csv(folder / "anchor_samples.csv")
    predictions = predictions[predictions.model.eq("partial_A_event_class")].copy()
    keys = ["fault_event_id", "tool_id", "horizon"]
    predictions = predictions.merge(
        anchors[keys + ["prediction_timestamp"]], on=keys, how="left", validate="one_to_one")

    with np.load(model1_source) as saved:
        times = saved["time"].astype(np.int64)
        codes = saved["tool_code"].astype(int)

    tools = {}
    for code, tool in enumerate(TOOLS):
        rows = predictions[predictions.tool_id.eq(tool)].sort_values(
            ["prediction_timestamp", "fault_event_id"], ascending=[False, False])
        if rows.empty:
            tools[tool] = {"available": False, "prediction_timestamp": None,
                           "predicted_class": None, "status": "판단 보류",
                           "reason": "최종 Model 2 검증 결과 없음"}
            continue
        row = rows.iloc[0]
        timestamp = int(row.prediction_timestamp)
        index = nearest_index(times, np.flatnonzero(codes == code), timestamp)
        tools[tool] = {
            "available": True, "prediction_timestamp": timestamp,
            "predicted_class": {"pressure_low": "Low Pressure", "pressure_high": "High Pressure",
                                "leak": "Leak"}[row.predicted],
            "class_key": row.predicted, "actual_fault_type": row.fault_type,
            "fault_event_id": int(row.fault_event_id), "horizon": row.horizon,
            "nearest_model1_timestamp": int(times[index]),
            "timestamp_gap_seconds": abs(int(times[index]) - timestamp),
        }
    return {
        "metadata": {"model": "full_fault_type_multi_anchor_best.joblib",
                     "source": "output/experiments/full_fault_type_multi_anchor/validation_predictions.csv",
                     "replay_mode": "설비별 최신 저장 검증 결과", "same_timestamp_as_fleet": False,
                     "selected_model": "partial_A_event_class"},
        "tools": tools,
    }


def validate(fleet, fault_types):
    counts = {s: sum(x["model1"]["status"] == s for x in fleet["tools"].values())
              for s in ("정상", "주의", "위험", "판단 보류")}
    assert counts == {"정상": 12, "주의": 5, "위험": 2, "판단 보류": 1}
    assert fleet["tools"]["02_M01"]["priority"] is None
    assert [t for t in sorted(TOOLS, key=lambda t: fleet["tools"][t]["priority"] or 999)[:3]] == [
        "03_M02", "03_M01", "08_M01"]
    available = [t for t, x in fault_types["tools"].items() if x["available"]]
    missing = [t for t, x in fault_types["tools"].items() if not x["available"]]
    assert len(available) == 17
    assert missing == ["07_M02", "08_M02", "10_M01"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model1_source = ROOT / "output/experiments/forward_validation_ab_24h/validation_probabilities_a.npz"
    fleet = build_fleet()
    fault_types = build_fault_types(model1_source)
    validate(fleet, fault_types)
    (OUT / "fleet_snapshot.json").write_text(json.dumps(fleet, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "fault_type_replay.json").write_text(json.dumps(fault_types, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"fleet": len(fleet["tools"]),
                      "model2_available": sum(x["available"] for x in fault_types["tools"].values())},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
