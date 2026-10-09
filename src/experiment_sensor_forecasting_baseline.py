"""Forecast FlowCool sensor values 30/60 min ahead from causal minute history.

Uses the existing chronological Model A train/validation boundaries and never
opens the reserved test segment. TTF and faults are evaluation labels only.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from audit_ttf_sensor_relationship import causal_features
from experiment_tool_stage_interaction_24h import file_hash
from forward_validation_ab_24h import read_pretest

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT/"output/baseline_recovery/prepared"
PLAN = ROOT/"output/experiments/forward_validation_ab_24h/split_plan.json"
OUT = ROOT/"output/experiments/sensor_forecasting_baseline"
SENSORS = ("FLOWCOOLPRESSURE","FLOWCOOLFLOWRATE")
HORIZONS = (1800,3600)
SAMPLE_MINUTES = 5
TARGET_TOLERANCE_SECONDS = 60
PARAMS = {"objective":"reg:squarederror", "n_estimators":120, "max_depth":4,
          "learning_rate":.05, "subsample":.8,"colsample_bytree":.8,
          "min_child_weight":4,"reg_lambda":2.,"tree_method":"hist",
          "random_state":42,"n_jobs":8}


def one_sensor_features(t, session, value):
    # The shared function uses elapsed seconds and resets at each session.
    history = causal_features(t,session,value)
    result = {"current":value}
    for minutes in (1,5,10):
        result[f"lag_{minutes}m"] = value-history[f"diff_{minutes}m"]
    for minutes in (10,30):
        for statistic in ("mean","std"):
            result[f"{statistic}_{minutes}m"] = history[f"{statistic}_{minutes}m"]
    result["slope_10m"] = history["slope_10m"]
    # Actual elapsed 5-minute rolling moments, current and preceding values.
    mean5, std5 = np.full(len(t),np.nan,np.float32),np.full(len(t),np.nan,np.float32)
    bounds=np.r_[0,np.flatnonzero(np.diff(session)!=0)+1,len(t)]
    for a,b in zip(bounds[:-1],bounds[1:]):
        tt=t[a:b]
        x=value[a:b].astype(float)
        finite=np.isfinite(x)
        x=np.where(finite,x,0.)
        left=np.searchsorted(tt,tt-300,side="left")
        idx=np.arange(b-a)
        sums=[np.r_[0.,np.cumsum(z)] for z in (finite.astype(float),x,x*x)]
        n,s,s2=[z[idx+1]-z[left] for z in sums]
        good=n>=2
        m=np.divide(s,n,out=np.full(b-a,np.nan),where=good)
        v=np.divide(s2,n,out=np.full(b-a,np.nan),where=good)-m*m
        mean5[a:b]=m.astype(np.float32)
        std5[a:b]=np.sqrt(np.maximum(v,0)).astype(np.float32)
    result["mean_5m"],result["std_5m"]=mean5,std5
    return result


def targets(t, session, value, horizon):
    # First actual observation at/after desired time, within 60 s. Never
    # interpolate across a stopped machine or use the future value as input.
    target_time=t+horizon
    index=np.searchsorted(t,target_time,side="left")
    exists=index<len(t)
    safe=np.minimum(index,len(t)-1)
    eligible=exists & (t[safe]-target_time>=0) & (t[safe]-target_time<=TARGET_TOLERANCE_SECONDS) & (session[safe]==session)
    return safe, eligible


def prepare(plan):
    parts=[]
    audit=[]
    cols=["time","session_id","stage","next_event_time",*SENSORS]
    for code,item in enumerate(plan["tools"]):
        name=item["tool_id"]
        path=PREP/f"{name}.parquet"
        frame=read_pretest(path,item["test_boundary_index"],cols)
        t=frame.time.to_numpy(np.int64)
        if np.any(np.diff(t)<0):
            raise AssertionError(f"Unsorted time: {name}")
        session=frame.session_id.to_numpy()
        features={}
        for sensor in SENSORS:
            values=frame[sensor].to_numpy(np.float32)
            for suffix,v in one_sensor_features(t,session,values).items():
                features[f"{sensor}_{suffix}"]=v
        sample=(t//60)%SAMPLE_MINUTES==0
        split=np.full(len(t),"excluded",object)
        split[t<item["train_end_exclusive_second"]]="train"
        split[(t>=item["validation_start_second"])&(t<item["validation_end_exclusive_second"])]="validation"
        next_event=frame.next_event_time.to_numpy(float)
        near_fault=np.isfinite(next_event)&(next_event>t)&(next_event-t<=86400)
        for horizon in HORIZONS:
            index,eligible=targets(t,session,frame[SENSORS[0]].to_numpy(np.float32),horizon)
            # Both origin and label time must remain in one temporal partition.
            target_boundary=np.where(split=="train",item["train_end_exclusive_second"],
                              np.where(split=="validation",item["validation_end_exclusive_second"],-1))
            eligible &= sample & (split!="excluded") & (t[index]<target_boundary)
            if np.any(eligible & (split[index]!=split)):
                raise AssertionError(f"Future target crossed split: {name}")
            for sensor in SENSORS:
                values=frame[sensor].to_numpy(np.float32)
                good=eligible & np.isfinite(values) & np.isfinite(values[index])
                if not np.any(good):
                    continue
                rows=pd.DataFrame({"tool_id":name,"tool_code":code,"stage":frame.stage.to_numpy()[good],
                    "time":t[good],"target_time":t[index[good]],
                    "split":split[good],"horizon_minutes":horizon//60,
                    "sensor":sensor,"current_value":values[good],
                    "future_value":values[index[good]],"pre_fault_24h":near_fault[good]})
                for key,v in features.items():
                    rows[key]=v[good]
                parts.append(rows)
                audit.append({"tool_id":name,"sensor":sensor,"horizon_minutes":horizon//60,
                              "train_rows":int(np.count_nonzero(good&(split=="train"))),
                              "validation_rows":int(np.count_nonzero(good&(split=="validation")))})
        print(f"{name}: {len(frame):,} minute rows",flush=True)
    return pd.concat(parts,ignore_index=True),pd.DataFrame(audit)


def metric(y,p):
    return {"mae":float(mean_absolute_error(y,p)),
            "rmse":float(np.sqrt(mean_squared_error(y,p))),
            "r2":float(r2_score(y,p)) if len(y)>1 else None}


def chart(frame,sensor,horizon):
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    shown=frame.iloc[::max(1,len(frame)//5000)]
    axes[0].scatter(shown.future_value,shown.predicted_value,s=3,alpha=.2)
    lo=min(shown.future_value.min(),shown.predicted_value.min())
    hi=max(shown.future_value.max(),shown.predicted_value.max())
    axes[0].plot([lo,hi],[lo,hi],"k--",linewidth=1)
    axes[0].set(xlabel="Actual future value",ylabel="Forecast",title=f"{sensor} +{horizon}m")
    axes[1].scatter(shown.future_value,shown.predicted_value-shown.future_value,s=3,alpha=.2)
    axes[1].axhline(0,color="black",linestyle="--",linewidth=1)
    axes[1].set(xlabel="Actual future value",ylabel="Forecast residual")
    fig.tight_layout()
    fig.savefig(OUT/f"actual_vs_predicted_{sensor}_{horizon}m.png",dpi=150)
    plt.close(fig)


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {OUT}")
    plan=json.loads(PLAN.read_text(encoding="utf-8"))
    if len(plan["tools"])!=20:
        raise AssertionError("Expected all 20 tools")
    data,audit=prepare(plan)
    OUT.mkdir(parents=True)
    audit.to_csv(OUT/"eligible_rows_by_tool.csv",index=False,encoding="utf-8-sig")
    feature_cols=[c for c in data if c.endswith(("_current","_lag_1m","_lag_5m","_lag_10m","_mean_5m","_mean_10m","_mean_30m","_std_5m","_std_10m","_std_30m","_slope_10m"))]
    feature_cols=[*feature_cols,"stage"]
    # Tool identities are known from the earlier training plan, not inferred
    # from validation targets. They are included as fixed one-hot columns.
    for code in range(len(plan["tools"])):
        col=f"tool_{code:02d}"
        data[col]=(data.tool_code==code).astype(np.int8)
        feature_cols.append(col)
    summary,detail,direction,predictions=[],[],[],[]
    for sensor in SENSORS:
        for horizon in (30,60):
            group=data[(data.sensor==sensor)&(data.horizon_minutes==horizon)]
            train=group[group.split=="train"]
            valid=group[group.split=="validation"].copy()
            if len(train)<1000 or len(valid)<1000:
                raise AssertionError(f"Insufficient rows: {sensor} {horizon}")
            model=XGBRegressor(**PARAMS)
            model.fit(train[feature_cols].to_numpy(np.float32),train.future_value.to_numpy(np.float32))
            valid["predicted_value"]=model.predict(valid[feature_cols].to_numpy(np.float32))
            valid["persistence_value"]=valid.current_value
            actual_change=valid.future_value.to_numpy(float)-valid.current_value.to_numpy(float)
            train_change=train.future_value.to_numpy(float)-train.current_value.to_numpy(float)
            # One scale-specific flat threshold learned exclusively from Train.
            epsilon=float(.1*np.std(train_change))
            actual_direction=np.where(actual_change>epsilon,1,np.where(actual_change<-epsilon,-1,0))
            predicted_change=valid.predicted_value.to_numpy(float)-valid.current_value.to_numpy(float)
            forecast_direction=np.where(predicted_change>epsilon,1,np.where(predicted_change<-epsilon,-1,0))
            nonflat=actual_direction!=0
            for label,column in (("Persistence","persistence_value"),("XGBoost","predicted_value")):
                result={"sensor":sensor,"horizon_minutes":horizon,"model":label,
                    "train_rows":len(train),"validation_rows":len(valid),
                    "direction_flat_tolerance_train_only":epsilon,
                    **metric(valid.future_value,valid[column])}
                pred_dir=np.zeros(len(valid),dtype=int) if label=="Persistence" else forecast_direction
                result["direction_accuracy_3class"]=float(np.mean(pred_dir==actual_direction))
                result["direction_accuracy_actual_nonflat"]=float(np.mean(pred_dir[nonflat]==actual_direction[nonflat])) if nonflat.any() else None
                result["actual_nonflat_fraction"]=float(np.mean(nonflat))
                summary.append(result)
                for condition,sub in (("normal",valid[~valid.pre_fault_24h]),("pre_fault_24h",valid[valid.pre_fault_24h])):
                    if len(sub):
                        detail.append({"sensor":sensor,"horizon_minutes":horizon,"model":label,
                            "segment":condition,"rows":len(sub),**metric(sub.future_value,sub[column])})
                for tool_id,sub in valid.groupby("tool_id"):
                    detail.append({"sensor":sensor,"horizon_minutes":horizon,"model":label,
                        "segment":f"tool:{tool_id}","rows":len(sub),**metric(sub.future_value,sub[column])})
            predicted_move=forecast_direction!=0
            direction.append({"sensor":sensor,"horizon_minutes":horizon,"epsilon_train_only":epsilon,
                "actual_up":int((actual_direction==1).sum()),"actual_down":int((actual_direction==-1).sum()),
                "actual_stable":int((actual_direction==0).sum()),
                "predicted_move_fraction":float(np.mean(predicted_move)),
                "precision_when_change_predicted":float(np.mean(forecast_direction[predicted_move]==actual_direction[predicted_move])) if predicted_move.any() else None,
                "false_move_rate_on_stable":float(np.mean(predicted_move[actual_direction==0])) if np.any(actual_direction==0) else None})
            predictions.append(valid[["tool_id","time","target_time","sensor","horizon_minutes","split",
                "current_value","future_value","predicted_value","persistence_value","pre_fault_24h"]])
            chart(valid,sensor,horizon)
            print(json.dumps([r for r in summary if r["sensor"]==sensor and r["horizon_minutes"]==horizon]),flush=True)
    pd.DataFrame(summary).to_csv(OUT/"overall_metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(detail).to_csv(OUT/"segment_and_tool_metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(direction).to_csv(OUT/"direction_class_counts.csv",index=False,encoding="utf-8-sig")
    pd.concat(predictions).to_csv(OUT/"validation_predictions.csv.gz",index=False,compression="gzip")
    report={"scope":"Existing prepared actual 1-minute data, 20 tools, reused forward Model A Train/Validation only; no reserved Test",
        "sample_rule":"Deterministic clock-minute mod 5 == 0 at forecast origin; future target searched in full minute history",
        "horizons_minutes":[30,60],"target_match":"First actual observation at/after requested horizon, no more than 60 seconds late, in same continuous session",
        "split_purge":"Existing per-tool >=24h train/validation and validation/test purges, plus target time constrained to its origin split boundary",
        "feature_rule":"Current/past sensors only; lag and rolling windows use elapsed seconds and same-session history. Tool one-hot identities fixed from Train plan. No TTF, fault time, next_event_time, or future sensor in feature_cols.",
        "normal_vs_prefault":"pre_fault_24h is evaluation grouping using next logged fault from prepared data; it is not a model feature",
        "direction_rule":"Three classes up/stable/down; stable tolerance = 10% of Train target-delta standard deviation per sensor/horizon. Both all-row and actual-nonflat accuracy reported.",
        "model_parameters":PARAMS,"features":feature_cols,
        "plan_sha256":file_hash(PLAN),"first_stage_model_untouched":True,
        "interpretation":"High value R2 can arise from persistence and does not imply useful trend prediction. Historical validation has been reused and is not independent final test."}
    (OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
