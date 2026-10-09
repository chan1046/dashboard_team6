"""Read saved predictions to audit event counts and 24h operating relevance."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from train_rul_paper_multioutput import OUT, TYPES


def main():
    dest=OUT/"summary_report.json"
    if dest.exists():
        raise FileExistsError(dest)
    candidates=pd.read_csv(OUT/"fault_event_sequence_availability.csv")
    meta=pd.read_csv(OUT/"sequence_metadata_and_targets.csv.gz")
    pred=pd.read_csv(OUT/"validation_predictions.csv.gz")
    metrics=pd.read_csv(OUT/"metrics.csv")
    event_counts={}
    ids={}
    for split in ("train","validation"):
        part=meta[meta.split==split]
        by_type={}
        for typ in TYPES:
            col=f"event_id_{typ}"
            by_type[typ]=int(part[col].nunique())
        event_counts[split]={"by_type":by_type,"total_distinct_typed_fault_events":sum(by_type.values()),
                             "windows":len(part)}
    near=[]
    for (model,typ),group in pred.groupby(["model","fault_type"]):
        sub=group[(group.actual_ttf_h>0)&(group.actual_ttf_h<=24)]
        if not len(sub):
            continue
        near.append({"model":model,"fault_type":typ,"rows":len(sub),"events":sub.event_id.nunique(),
            "mae_h":float(sub.abs_error_h.mean()),
            "event_equal_mae_h":float(sub.groupby("event_id").abs_error_h.mean().mean()),
            "within_1h":float((sub.abs_error_h<=1).mean()),
            "within_2h":float((sub.abs_error_h<=2).mean())})
    pd.DataFrame(near).to_csv(OUT/"within_24h_diagnostic_metrics.csv",index=False,encoding="utf-8-sig")
    macro=metrics.groupby("model").agg(macro_mae_h=("mae_h","mean"),
        macro_event_equal_mae_h=("event_equal_mae_h","mean"),
        macro_rmse_h=("rmse_h","mean")).to_dict(orient="index")
    near_frame=pd.DataFrame(near)
    near_macro=near_frame.groupby("model").agg(macro_mae_h=("mae_h","mean"),
        macro_event_equal_mae_h=("event_equal_mae_h","mean")).to_dict(orient="index")
    summary={"raw_unique_fault_events":1211,
        "events_with_2000_shutter1_observations_in_type_cycle_upper":int(candidates.has_2000_operating_obs.sum()),
        "candidate_events_by_type":candidates.groupby("fault_type").has_2000_operating_obs.sum().astype(int).to_dict(),
        "actual_used_events":event_counts,
        "macro_metrics_all_known_rul":macro,
        "within_24h_diagnostic_macro":near_macro,
        "within_24h_scope":"Diagnostic subset selected only after all validation predictions. Not the primary metric or training selection; exact 24h event counts are small and no Model 1 scores used.",
        "old_conditional_population":"Previous 304 train/116 validation events were next-any-fault in true 0-6h, sampled every 10 minutes from 1-minute data. The new counts are multi-type, all-known-RUL 4-second operating-window events; descriptive population change only, not matched improvement.",
        "model1_connection_performed":False,
        "model1_connection_reason":"Despite lower MAE than Dummy for some types, all full-range R² remain negative and 24h diagnostic errors are not reliable. The predeclared gate requires Model 2 to be meaningfully better for 24h risk triage."}
    dest.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
