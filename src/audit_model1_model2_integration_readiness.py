"""Read-only audit of whether frozen validation scores permit Model 1→2 evaluation.

No model fitting, rescoring, threshold selection or access to reserved test.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE1 = ROOT / "output/experiments/forward_validation_ab_24h"
SOURCE2 = ROOT / "output/experiments/ttf_imminent_6h"
OUT = ROOT / "output/experiments/model1_model2_integration"
THRESHOLD1 = .34
THRESHOLD2 = .5


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def label_audit():
    """Label Model 1 triggers only where a full 6h validation lookahead exists."""
    destination = OUT / "trigger_6h_label_audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    plan = json.loads((SOURCE1 / "split_plan.json").read_text(encoding="utf-8"))
    faults = pd.read_csv(ROOT / "data/processed/fault_events.csv")
    faults = faults.drop_duplicates(["tool_id", "time", "fault_type"])
    with np.load(SOURCE1 / "validation_probabilities_a.npz") as saved:
        times, codes, scores = (saved[k].copy() for k in ("time", "tool_code", "probability"))
    total, evaluable, imminent = 0, 0, 0
    by_tool = []
    for code, item in enumerate(plan["tools"]):
        tool = item["tool_id"]
        chosen = (codes == code) & (scores >= THRESHOLD1)
        t = times[chosen]
        complete = t + 21600 < item["validation_end_exclusive_second"]
        tf = t[complete]
        fault_times = np.sort(faults.loc[
            (faults.tool_id == tool) &
            (faults.time < item["validation_end_exclusive_second"]), "time"].to_numpy(np.int64))
        next_index = np.searchsorted(fault_times, tf, side="right")
        next_time = np.full(len(tf), np.iinfo(np.int64).max, dtype=np.int64)
        exists = next_index < len(fault_times)
        next_time[exists] = fault_times[next_index[exists]]
        positives = int(np.count_nonzero(next_time - tf <= 21600))
        by_tool.append({"tool_id": tool, "triggers": len(t), "fully_observable_6h_triggers": len(tf),
                        "actual_fault_within_6h": positives})
        total += len(t); evaluable += len(tf); imminent += positives
    result = {"model1_triggers": total, "fully_observable_6h_triggers": evaluable,
              "excluded_at_validation_end": total - evaluable,
              "actual_fault_within_6h": imminent,
              "imminent_fraction": imminent / evaluable if evaluable else None,
              "rule": "The next unique logged fault for the same tool must occur within 6h of the trigger, inside the frozen validation interval. Triggers without a full 6h validation lookahead are excluded, not labeled negative.",
              "by_tool": by_tool}
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "by_tool"}, indent=2))


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    plan_path = SOURCE1 / "split_plan.json"
    training_path = SOURCE1 / "training_a.json"
    model_path = ROOT / "output/models/xgb_24h_forward_a.joblib"
    score_path = SOURCE1 / "validation_probabilities_a.npz"
    model2_path = SOURCE2 / "validation_predictions.csv.gz"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if digest(model_path) != training["model_sha256"] or digest(score_path) != training["validation_score_sha256"]:
        raise AssertionError("Frozen Model 1 artifact changed")
    with np.load(score_path) as saved:
        times = saved["time"].copy()
        tool_code = saved["tool_code"].copy()
        p1 = saved["probability"].copy()
    names = {i: item["tool_id"] for i, item in enumerate(plan["tools"])}
    triggers = pd.DataFrame({"tool_id": pd.Series(tool_code).map(names),
                             "time": times, "model1_probability": p1})
    triggers = triggers[triggers.model1_probability >= THRESHOLD1].copy()
    if triggers.duplicated(["tool_id", "time"]).any():
        raise AssertionError("Model 1 duplicate tool/time trigger")
    stored = pd.read_csv(model2_path, usecols=["tool_id", "time", "probability_B_event_balanced", "imminent"])
    if stored.duplicated(["tool_id", "time"]).any():
        raise AssertionError("Stored Model 2 duplicate tool/time score")
    joined = triggers.merge(stored, on=["tool_id", "time"], how="left", validate="one_to_one", indicator=True)
    found = joined._merge.eq("both")
    rows = [{"tool_id": name, "model1_triggers": len(part),
             "model2_scores_available": int(part._merge.eq("both").sum()),
             "model2_scores_missing": int(part._merge.eq("left_only").sum()),
             "coverage_fraction": float(part._merge.eq("both").mean())}
            for name, part in joined.groupby("tool_id", sort=True)]
    available = joined[found]
    report = {
        "status": "blocked_missing_frozen_model2_artifact_and_full_timeline_scores",
        "model1_threshold": THRESHOLD1, "model2_threshold": THRESHOLD2,
        "model1_triggers": len(joined),
        "model2_scores_at_trigger": int(found.sum()),
        "model2_scores_missing_at_trigger": int((~found).sum()),
        "model2_trigger_coverage_fraction": float(found.mean()),
        "stored_model2_positive_fraction_at_covered_triggers": float(available.imminent.mean()) if len(available) else None,
        "stored_model2_imminent_signal_fraction_at_covered_triggers": float((available.probability_B_event_balanced >= THRESHOLD2).mean()) if len(available) else None,
        "coverage_caveat": "Stored Model 2 scores exist only for 5-minute sampled rows selected with actual next-fault TTF<=24h. Covered triggers are label-selected and cannot represent the full validation timeline or false-urgent rate.",
        "missing_artifact": "The fitted B_event_balanced HistGradientBoostingClassifier was not persisted in output/models or this experiment. Its saved validation scores cannot generate predictions for missing trigger rows.",
        "next_action_if_authorized": "Recreate exactly the prior deterministic fit, save the fitted model, then score every Model 1 trigger without TTF filtering. This would be model training and is intentionally not done in this audit.",
        "model1_model_sha256": digest(model_path),
        "model1_validation_scores_sha256": digest(score_path),
        "model2_partial_predictions_sha256": digest(model2_path),
        "test_used": False,
    }
    OUT.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(OUT / "trigger_score_coverage_by_tool.csv", index=False, encoding="utf-8-sig")
    (OUT / "readiness_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    label_audit() if "--labels-only" in sys.argv else main()
