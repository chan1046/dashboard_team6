"""Score frozen Model 1 triggers with the newly saved Sequence RUL model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiment_sequence_rul_baseline import (ROOT, OUT, MODELS, PLAN, FAULTS,
    SCORES1, FEATURES, GRURegressor, load_tool, predict_gru)
from experiment_tool_stage_interaction_24h import file_hash

THRESHOLD1=.34


def main():
    destination=OUT/"model1_trigger_connection_report.json"
    if destination.exists():
        raise FileExistsError(destination)
    plan=json.loads(PLAN.read_text(encoding="utf-8"))
    cfg=json.loads((MODELS/"sequence_rul_gru_baseline_config.json").read_text(encoding="utf-8"))
    source=ROOT/"output/experiments/forward_validation_ab_24h"
    training=json.loads((source/"training_a.json").read_text(encoding="utf-8"))
    if file_hash(MODELS/"xgb_24h_forward_a.joblib")!=training["model_sha256"] or file_hash(SCORES1)!=training["validation_score_sha256"]:
        raise AssertionError("Frozen Model 1 artifacts changed")
    if cfg["feature_names"]!=FEATURES:
        raise AssertionError("Saved feature config changed")
    payload=torch.load(cfg["model_path"],map_location="cpu",weights_only=False)
    model=GRURegressor(len(FEATURES));model.load_state_dict(payload["state_dict"])
    center=payload["center"];scale=payload["scale"]
    length=int(cfg["sequence_minutes"])
    with np.load(SCORES1) as saved:
        t_all,tool_all,p1_all=(saved[k].copy() for k in ("time","tool_code","probability"))
    faults=pd.read_csv(FAULTS).drop_duplicates(["tool_id","time","fault_type"])
    rows=[]; event_rows=[]; available=0; total=0; complete=0
    for code,item in enumerate(plan["tools"]):
        d=load_tool(item,faults)
        chosen=(tool_all==code)&(p1_all>=THRESHOLD1)
        times=t_all[chosen]; p1=p1_all[chosen]
        total+=len(times)
        index=np.searchsorted(d["t"],times)
        aligned=(index<len(d["t"]))&(d["t"][np.minimum(index,len(d["t"])-1)]==times)
        okay=aligned.copy()
        okay[aligned] &= index[aligned]>=length-1
        eligible=np.flatnonzero(okay)
        okay[eligible] &= d["run_start"][index[eligible]]<=index[eligible]-length+1
        eligible=np.flatnonzero(okay)
        if len(eligible):
            positions=index[eligible,None]-np.arange(length-1,-1,-1)[None,:]
            okfinite=np.isfinite(d["v"][positions]).all(axis=(1,2))
            okay[eligible] &= okfinite
        selected=np.flatnonzero(okay)
        available+=len(selected)
        predictions=np.full(len(times),np.nan,dtype=np.float32)
        for start in range(0,len(selected),2048):
            group=selected[start:start+2048]
            pos=index[group,None]-np.arange(length-1,-1,-1)[None,:]
            x=((d["v"][pos]-center)/scale).astype(np.float32)
            predictions[group]=np.clip(predict_gru(model,x)*24,0,cfg["degradation_window_h"])
        ft=np.sort(faults.loc[(faults.tool_id==d["tool"]) &
                              (faults.time<item["validation_end_exclusive_second"]),"time"].to_numpy(np.int64))
        next_i=np.searchsorted(ft,times,side="right")
        next_t=np.full(len(times),np.iinfo(np.int64).max,dtype=np.int64)
        exists=next_i<len(ft)
        next_t[exists]=ft[next_i[exists]]
        observable=(times+21600<item["validation_end_exclusive_second"])
        complete+=int(observable.sum())
        ttf=np.where(exists,(next_t-times)/3600,np.nan)
        part=pd.DataFrame({"tool_id":d["tool"],"time":times,"model1_probability":p1,
                           "sequence_available":okay,"predicted_rul_h":predictions,
                           "next_fault_time":np.where(exists,next_t,-1),
                           "actual_ttf_h":ttf,"fully_observable_6h":observable})
        rows.append(part)
        # The first Model 1 threshold crossing within each event's 24h window.
        for fault_time in ft:
            f=part[(part.time<fault_time)&(part.time>=fault_time-86400)]
            if f.empty:
                continue
            scored=f[f.sequence_available]
            event_rows.append({"tool_id":d["tool"],"fault_time":int(fault_time),
                "first_model1_lead_h":float((fault_time-f.time.iloc[0])/3600),
                "model1_trigger_rows":len(f),"scored_trigger_rows":len(scored),
                "first_scored_rul_h":float(scored.predicted_rul_h.iloc[0]) if len(scored) else None,
                "last_scored_rul_h":float(scored.predicted_rul_h.iloc[-1]) if len(scored) else None,
                "estimated_rul_change_h":float(scored.predicted_rul_h.iloc[-1]-scored.predicted_rul_h.iloc[0]) if len(scored)>1 else None})
        print(d["tool"],len(times),len(selected),flush=True)
    frame=pd.concat(rows,ignore_index=True)
    frame.to_csv(OUT/"model1_trigger_sequence_predictions.csv.gz",index=False,compression="gzip")
    pd.DataFrame(event_rows).to_csv(OUT/"model1_first_alarm_rul_trajectory.csv",index=False,encoding="utf-8-sig")
    positive=frame[(frame.sequence_available)&(frame.actual_ttf_h>0)&(frame.actual_ttf_h<=6)]
    residual=positive.predicted_rul_h.to_numpy()-positive.actual_ttf_h.to_numpy()
    quantiles=np.quantile(np.abs(residual),[.5,.8,.9]) if len(residual) else [np.nan]*3
    by_tool=frame.groupby("tool_id").agg(triggers=("time","size"),sequence_available=("sequence_available","sum"))
    by_tool["available_fraction"]=by_tool.sequence_available/by_tool.triggers
    by_tool.to_csv(OUT/"model1_trigger_availability_by_tool.csv",encoding="utf-8-sig")
    report={"model1_trigger_rows":total,"sequence_available_rows":available,
            "sequence_available_fraction":available/total,"sequence_unavailable_fraction":1-available/total,
            "complete_6h_lookahead_rows":complete,
            "actual_fault_within_6h_scored_rows":len(positive),
            "mae_h_at_fault_bound_triggers_0_6h":float(np.mean(np.abs(residual))) if len(residual) else None,
            "absolute_residual_p50_p80_p90_h":list(map(float,quantiles)),
            "model1_detected_fault_events":len(event_rows),
            "events_with_at_least_one_rul_estimate":sum(r["scored_trigger_rows"]>0 for r in event_rows),
            "conditional_scope":"RUL MAE is measured only for Model 1 triggers with an available sequence and an actual fault <=6h. No-fault triggers have no finite RUL target; model outputs there are extrapolation and should not be displayed as exact fault times.",
            "model1_unchanged":True,"test_used":False}
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
