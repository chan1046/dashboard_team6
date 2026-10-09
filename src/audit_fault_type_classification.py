"""Fault-event coverage and a conditional 6h-lead multiclass diagnosis baseline.

One snapshot per event. Fault times/types select the research population and
targets only; the classifier receives current/past sensor values exclusively.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

from compare_hgb_xgb_12h_24h import PARAMS as HGB_PARAMS
from experiment_tool_stage_interaction_24h import file_hash
from forward_validation_ab_24h import read_pretest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/experiments/fault_type_classification"
SURVIVAL = ROOT / "output/experiments/survival_feasibility"
PLAN = ROOT / "output/experiments/forward_validation_ab_24h/split_plan.json"
PREP = ROOT / "output/baseline_recovery/prepared"
FAULT_LOG = ROOT / "data/processed/fault_events.csv"
CLASSES = ["pressure_low", "pressure_high", "leak"]
TARGET_LEAD_SECONDS = 6*3600
TOLERANCE_SECONDS = 3600
SENSORS = ["FLOWCOOLPRESSURE", "FLOWCOOLFLOWRATE", "IONGAUGEPRESSURE"]
FEATURES = [*SENSORS, "FLOWCOOLPRESSURE_mean_past_30m",
            "FLOWCOOLFLOWRATE_mean_past_30m", "IONGAUGEPRESSURE_std_past_10m"]


def find_snapshot(t, previous_fault, fault_time):
    target = fault_time-TARGET_LEAD_SECONDS
    index = np.searchsorted(t,target)
    candidates = [i for i in (index-1,index) if 0<=i<len(t) and
                  previous_fault<t[i]<fault_time and abs(int(t[i])-target)<=TOLERANCE_SECONDS]
    return min(candidates,key=lambda i:abs(int(t[i])-target)) if candidates else None


def make_snapshot(t, session, values, index, previous_fault):
    now=int(t[index])
    start=max(previous_fault,now-1800)
    left=int(np.searchsorted(t,start,side="right" if start==previous_fault else "left"))
    local=np.arange(left,index+1)
    local=local[session[local]==session[index]]
    pressure=values["FLOWCOOLPRESSURE"]
    flow=values["FLOWCOOLFLOWRATE"]
    ion=values["IONGAUGEPRESSURE"]
    window10=local[t[local]>=now-600]
    return {"FLOWCOOLPRESSURE":float(pressure[index]),
            "FLOWCOOLFLOWRATE":float(flow[index]),
            "IONGAUGEPRESSURE":float(ion[index]),
            "FLOWCOOLPRESSURE_mean_past_30m":float(np.nanmean(pressure[local])) if len(local) else np.nan,
            "FLOWCOOLFLOWRATE_mean_past_30m":float(np.nanmean(flow[local])) if len(local) else np.nan,
            "IONGAUGEPRESSURE_std_past_10m":float(np.nanstd(ion[window10])) if len(window10)>=2 else np.nan,
            "history_rows_30m":len(local)}


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    original=pd.read_csv(FAULT_LOG)
    events=original.sort_values(["tool_id","time","fault_type","fault_event_id"]).drop_duplicates(
        ["tool_id","time","fault_type"],keep="first")
    if len(events)!=1211 or len(original)!=1236:
        raise AssertionError("Fault deduplication reference changed")
    cycles=pd.read_csv(SURVIVAL/"failure_cycles.csv")
    observed=cycles[cycles.event_observed].copy()
    if len(observed)!=1207:
        raise AssertionError("Cycle audit reference changed")
    coverage=[]
    for row in observed.itertuples():
        item={"tool_id":row.tool_id,"fault_type":row.fault_type,"fault_time":int(row.fault_time),
              "event_id":f"{row.tool_id}|{row.fault_type}|{int(row.fault_time)}",
              "any_pre_fault_sensor_since_previous_fault":bool(row.sensor_rows_strictly_inside>0)}
        for h in (1,6,12,24):
            item[f"sensor_within_{h}h"]=(pd.notna(row.gap_from_last_sensor_to_cycle_end_seconds) and
                                         row.gap_from_last_sensor_to_cycle_end_seconds<=h*3600)
        coverage.append(item)
    coverage=pd.DataFrame(coverage)
    # Preserve every unique fault, including four events predating the first
    # sensor record. They have no usable pre-fault history by definition.
    all_keys=events[["tool_id","fault_type","time"]].rename(columns={"time":"fault_time"}).copy()
    all_keys["event_id"]=(all_keys.tool_id+"|"+all_keys.fault_type+"|"+
                          all_keys.fault_time.astype(str))
    coverage=all_keys.merge(coverage,on=["tool_id","fault_type","fault_time","event_id"],how="left",validate="one_to_one")
    coverage["in_sensor_range"]=coverage.any_pre_fault_sensor_since_previous_fault.notna()
    for column in ["any_pre_fault_sensor_since_previous_fault",*[f"sensor_within_{h}h" for h in (1,6,12,24)]]:
        coverage[column]=coverage[column].fillna(False).astype(bool)
    plan=json.loads(PLAN.read_text(encoding="utf-8"))
    snapshots=[]
    for item in plan["tools"]:
        name=item["tool_id"]
        full_events=events[events.tool_id==name].sort_values("time")
        path=PREP/f"{name}.parquet"
        frame=read_pretest(path,item["test_boundary_index"],["time","session_id",*SENSORS])
        t=frame.time.to_numpy(np.int64)
        session=frame.session_id.to_numpy()
        values={sensor:frame[sensor].to_numpy(float) for sensor in SENSORS}
        prior_fault=None
        for event in full_events.itertuples():
            fault_time=int(event.time)
            if fault_time<item["train_end_exclusive_second"]:
                split="train"
            elif item["validation_start_second"]<=fault_time<item["validation_end_exclusive_second"]:
                split="validation"
            else:
                split="excluded"
            previous=prior_fault if prior_fault is not None else int(t[0])-1
            if split!="excluded" and fault_time>t[0]:
                index=find_snapshot(t,previous,fault_time)
                if index is not None:
                    # Snapshot and its target event remain within the same
                    # per-tool chronological partition; no split-edge leak.
                    valid_origin=(split=="train" and t[index]<item["train_end_exclusive_second"] or
                                  split=="validation" and t[index]>=item["validation_start_second"])
                    if valid_origin:
                        fields=make_snapshot(t,session,values,index,previous)
                        snapshots.append({"tool_id":name,"fault_type":event.fault_type,
                            "fault_time":fault_time,"event_id":f"{name}|{event.fault_type}|{fault_time}",
                            "snapshot_time":int(t[index]),"lead_hours":(fault_time-int(t[index]))/3600,
                            "split":split,**fields})
            prior_fault=fault_time
        print(f"{name}: {len(full_events)} events, snapshot audit complete",flush=True)
    samples=pd.DataFrame(snapshots)
    if samples.event_id.duplicated().any() or not set(samples.fault_type)==set(CLASSES):
        raise AssertionError("Duplicate event snapshot or missing class")
    train=samples[samples.split=="train"]
    valid=samples[samples.split=="validation"]
    if set(train.event_id)&set(valid.event_id):
        raise AssertionError("Same event appears in both splits")
    if set(train.fault_type)!=set(CLASSES) or set(valid.fault_type)!=set(CLASSES):
        raise AssertionError("One fault type absent from train/validation")
    models={"Dummy_most_frequent":DummyClassifier(strategy="most_frequent"),
            "HGB_balanced":HistGradientBoostingClassifier(**HGB_PARAMS,class_weight="balanced")}
    result, per_class, confusion_rows, prediction_rows=[],[],[],[]
    for model_name,model in models.items():
        model.fit(train[FEATURES].to_numpy(np.float32),train.fault_type.to_numpy())
        pred=model.predict(valid[FEATURES].to_numpy(np.float32))
        proba=model.predict_proba(valid[FEATURES].to_numpy(np.float32))
        order=list(model.classes_)
        result.append({"model":model_name,"train_events":len(train),"validation_events":len(valid),
            "accuracy":float(np.mean(pred==valid.fault_type.to_numpy())),
            "macro_f1":float(f1_score(valid.fault_type,pred,labels=CLASSES,average="macro",zero_division=0)),
            "balanced_accuracy":float(balanced_accuracy_score(valid.fault_type,pred))})
        precision,recall,f1,support=precision_recall_fscore_support(
            valid.fault_type,pred,labels=CLASSES,zero_division=0)
        for name,p,r,f,n in zip(CLASSES,precision,recall,f1,support):
            per_class.append({"model":model_name,"fault_type":name,"precision":float(p),
                              "recall":float(r),"f1":float(f),"validation_events":int(n)})
        matrix=confusion_matrix(valid.fault_type,pred,labels=CLASSES)
        for i,truth in enumerate(CLASSES):
            for j,guess in enumerate(CLASSES):
                confusion_rows.append({"model":model_name,"actual":truth,"predicted":guess,"events":int(matrix[i,j])})
        for row,guess,probs in zip(valid.itertuples(),pred,proba):
            prediction_rows.append({"model":model_name,"tool_id":row.tool_id,"event_id":row.event_id,
                "fault_time":row.fault_time,"snapshot_time":row.snapshot_time,
                "actual":row.fault_type,"predicted":guess,
                **{f"prob_{name}":float(probs[order.index(name)]) for name in CLASSES}})
        print(json.dumps(result[-1]),flush=True)
    OUT.mkdir(parents=True)
    coverage.to_csv(OUT/"event_sensor_coverage.csv",index=False,encoding="utf-8-sig")
    samples.to_csv(OUT/"event_snapshots.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(result).to_csv(OUT/"metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(per_class).to_csv(OUT/"per_type_metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(confusion_rows).to_csv(OUT/"confusion_matrix.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(OUT/"validation_predictions.csv",index=False,encoding="utf-8-sig")
    tool_counts=events.pivot_table(index="tool_id",columns="fault_type",values="fault_event_id",aggfunc="count",fill_value=0).reindex(columns=CLASSES,fill_value=0)
    tool_counts["total"]=tool_counts.sum(axis=1)
    tool_counts.to_csv(OUT/"fault_type_by_tool.csv",encoding="utf-8-sig")
    coverage_summary={"total_unique_fault_events":len(events),
        "original_fault_rows":len(original),"duplicates_removed":len(original)-len(events),
        "type_counts":events.groupby("fault_type").size().to_dict(),
        "events_in_sensor_range":int(coverage.in_sensor_range.sum()),
        "any_sensor_history_since_previous_fault":int(coverage.any_pre_fault_sensor_since_previous_fault.sum()),
        **{f"any_sensor_within_{h}h":int(coverage[f"sensor_within_{h}h"].sum()) for h in (1,6,12,24)},
        "conditional_6h_snapshot_events":len(samples),
        "snapshot_train_events":len(train),"snapshot_validation_events":len(valid),
        "snapshot_type_by_split":samples.groupby(["split","fault_type"]).size().unstack(fill_value=0).to_dict("index")}
    report={"coverage":coverage_summary,
        "snapshot":"One actual minute observation closest to exactly 6h before each event within ±1h, after prior same-tool fault; only current and earlier same-session sensors",
        "conditional_caveat":"True future fault time selects the research snapshot. Online Model 2 cannot select that time; this is not gated end-to-end Model 1+2 performance.",
        "split":"Existing Model A per-tool chronological Train/Validation, with >=24h boundary purge; no event is shared. Reserved test not deserialized.",
        "features":FEATURES,"model_parameters":HGB_PARAMS,
        "forbidden_as_feature":["fault_name","fault_type","TTF","fault_time","post_fault_sensor","future_sensor","absolute_time","tool_id"],
        "gating":"Only display Model 2 probabilities after Model 1 actually raises risk; no forced diagnosis on Model 1-negative rows. Gated performance is not assessed in this experiment.",
        "fault_log_sha256":file_hash(FAULT_LOG),"split_plan_sha256":file_hash(PLAN),
        "model_1_unchanged":True,"metrics":result}
    (OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
