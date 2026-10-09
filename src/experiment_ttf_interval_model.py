"""Four-bin time-to-fault research classifier on frozen Model A features.

Full raw 4-second data are counted; fitting uses deterministic 5-minute
samples from existing causal 1-minute features in the old forward Train and
Validation only. TTF/fault time are targets, never features.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

from compare_hgb_xgb_12h_24h import PARAMS as HGB_PARAMS
from experiment_tool_relative_features_24h import BASE_FEATURES
from experiment_tool_stage_interaction_24h import file_hash
from forward_validation_ab_24h import read_pretest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output/experiments/ttf_interval_model"
PLAN=ROOT/"output/experiments/forward_validation_ab_24h/split_plan.json"
FAULTS=ROOT/"data/processed/fault_events.csv"
RAW=ROOT/"data/processed"
PREP=ROOT/"output/baseline_recovery/prepared"
LABELS=["0-3h","3-6h","6-12h","12-24h"]
EDGES=np.asarray([10800,21600,43200,86400],dtype=np.int64)
SAMPLE_MINUTES=5


def bin_index(ttf_seconds):
    return np.searchsorted(EDGES,ttf_seconds,side="left").astype(np.int8)


def full_raw_counts(fault_log,plan):
    rows=[]
    for item in plan["tools"]:
        tool=item["tool_id"]
        log=fault_log[fault_log.tool_id==tool].sort_values("time")
        ft=log.time.to_numpy(np.int64)
        typ=log.fault_type.to_numpy()
        if len(ft)==0:
            continue
        path=RAW/f"{tool}_sensor_ttf.parquet"
        t=pq.read_table(path,columns=["time"]).column("time").to_numpy().astype(np.int64)
        pos=np.searchsorted(ft,t,side="right")
        existing=pos<len(ft)
        safe=np.minimum(pos,len(ft)-1)
        delta=ft[safe]-t
        eligible=existing&(delta>0)&(delta<=86400)
        b=bin_index(delta[eligible])
        p=pos[eligible]
        for i,label in enumerate(LABELS):
            selected=p[b==i]
            if not len(selected):
                continue
            for fault_type in ("pressure_low","pressure_high","leak"):
                chosen=selected[typ[selected]==fault_type]
                if len(chosen):
                    rows.append({"tool_id":tool,"interval":label,"fault_type":fault_type,
                        "raw_4s_rows":len(chosen),"unique_fault_events":len(np.unique(chosen))})
        print(f"raw {tool}: {len(t):,} sensor rows",flush=True)
    return pd.DataFrame(rows)


def model_samples(fault_log,plan):
    parts=[]
    cols=[*BASE_FEATURES,"time"]
    for item in plan["tools"]:
        tool=item["tool_id"]
        log=fault_log[fault_log.tool_id==tool].sort_values("time")
        ft=log.time.to_numpy(np.int64)
        if not len(ft):
            continue
        frame=read_pretest(PREP/f"{tool}.parquet",item["test_boundary_index"],cols)
        t=frame.time.to_numpy(np.int64)
        pos=np.searchsorted(ft,t,side="right")
        existing=pos<len(ft)
        safe=np.minimum(pos,len(ft)-1)
        next_fault=ft[safe]
        delta=next_fault-t
        eligible=existing&(delta>0)&(delta<=86400)&((t//60)%SAMPLE_MINUTES==0)
        train=eligible&(next_fault<item["train_end_exclusive_second"])&(t<item["train_end_exclusive_second"])
        valid=eligible&(next_fault>=item["validation_start_second"])&(next_fault<item["validation_end_exclusive_second"])&(
            t>=item["validation_start_second"])&(t<item["validation_end_exclusive_second"])
        chosen=train|valid
        if not np.any(chosen):
            continue
        b=bin_index(delta[chosen])
        if np.any(b>3):
            raise AssertionError("Out-of-horizon bin")
        meta=pd.DataFrame({"tool_id":tool,"time":t[chosen],"fault_time":next_fault[chosen],
            "fault_type":log.fault_type.to_numpy()[pos[chosen]],
            "ttf_hours":delta[chosen]/3600,
            "class_index":b,"interval":[LABELS[x] for x in b],
            "split":np.where(train[chosen],"train","validation")})
        meta["event_id"]=meta.tool_id+"|"+meta.fault_time.astype(str)
        for feature in BASE_FEATURES:
            meta[feature]=frame.loc[chosen,feature].to_numpy(dtype=np.float32)
        parts.append(meta)
        print(f"model {tool}: train={int(train.sum()):,} validation={int(valid.sum()):,}",flush=True)
    data=pd.concat(parts,ignore_index=True)
    if set(data[data.split=="train"].event_id)&set(data[data.split=="validation"].event_id):
        raise AssertionError("Event appears on both sides of split")
    return data


def classification_metrics(y,pred):
    return {"macro_f1":float(f1_score(y,pred,labels=np.arange(4),average="macro",zero_division=0)),
        "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "accuracy":float(np.mean(y==pred)),
        "within_one_bin_accuracy":float(np.mean(np.abs(y-pred)<=1)),
        "mean_absolute_bin_error":float(np.mean(np.abs(y-pred))),
        "two_or_more_bins_error_fraction":float(np.mean(np.abs(y-pred)>=2))}


def calibration(y,proba):
    onehot=np.eye(4)[y]
    brier=float(np.mean(np.sum((proba-onehot)**2,axis=1)))
    confidence=proba.max(axis=1)
    correct=(proba.argmax(axis=1)==y).astype(float)
    bins=[]
    ece=0.
    for i in range(10):
        lo,hi=i/10,(i+1)/10
        mask=(confidence>=lo)&(confidence<hi if i<9 else confidence<=hi)
        if np.any(mask):
            accuracy=float(correct[mask].mean()); conf=float(confidence[mask].mean())
            ece+=mask.mean()*abs(accuracy-conf)
            bins.append({"confidence_bin":f"{lo:.1f}-{hi:.1f}","rows":int(mask.sum()),
                         "mean_confidence":conf,"observed_accuracy":accuracy})
    per_class=[{"interval":LABELS[i],"mean_probability":float(proba[:,i].mean()),
                "observed_fraction":float((y==i).mean()),
                "brier_binary":float(np.mean((proba[:,i]-onehot[:,i])**2))} for i in range(4)]
    return brier,ece,bins,per_class


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    plan=json.loads(PLAN.read_text(encoding="utf-8"))
    if len(plan["tools"])!=20 or len(BASE_FEATURES)!=85:
        raise AssertionError("Expected 20 tools and frozen 85 features")
    forbidden=("ttf","fault","next_event","label","absolute_time","lot","runnum")
    if any(any(word in name.lower() for word in forbidden) for name in BASE_FEATURES):
        raise AssertionError("Future/target feature detected")
    fault_log=pd.read_csv(FAULTS).sort_values(["tool_id","time","fault_type","fault_event_id"]).drop_duplicates(
        ["tool_id","time","fault_type"],keep="first")
    if len(fault_log)!=1211 or fault_log.duplicated(["tool_id","time"]).any():
        raise AssertionError("Unexpected fault count or simultaneous events")
    counts=full_raw_counts(fault_log,plan)
    data=model_samples(fault_log,plan)
    train=data[data.split=="train"]
    valid=data[data.split=="validation"].copy()
    if set(train.class_index)!=set(range(4)) or set(valid.class_index)!=set(range(4)):
        raise AssertionError("Missing train/validation class")
    representative=train.groupby("class_index").ttf_hours.median().reindex(range(4)).to_numpy(float)
    Xtrain=train[BASE_FEATURES].to_numpy(np.float32)
    Xvalid=valid[BASE_FEATURES].to_numpy(np.float32)
    ytrain=train.class_index.to_numpy(np.int8)
    yvalid=valid.class_index.to_numpy(np.int8)
    event_weights=(1/valid.groupby("event_id").event_id.transform("size")).to_numpy(float)
    models={"Dummy_prior":DummyClassifier(strategy="prior"),
        "HGB_balanced":HistGradientBoostingClassifier(**HGB_PARAMS,class_weight="balanced")}
    metrics_rows=[]; class_rows=[]; confusion_rows=[]; adjacency_rows=[]; prediction_frames=[]; calibration_rows=[]; confidence_rows=[]
    for name,model in models.items():
        model.fit(Xtrain,ytrain)
        proba=model.predict_proba(Xvalid)
        if not np.array_equal(model.classes_,np.arange(4)):
            raise AssertionError("Class probability order changed")
        pred=model.predict(Xvalid)
        expected_hours=proba@representative
        error=expected_hours-valid.ttf_hours.to_numpy(float)
        m={"model":name,"training_rows":len(train),"validation_rows":len(valid),
            "training_events":train.event_id.nunique(),"validation_events":valid.event_id.nunique(),
            **classification_metrics(yvalid,pred),
            "event_equal_weight_macro_f1":float(f1_score(yvalid,pred,labels=np.arange(4),average="macro",sample_weight=event_weights,zero_division=0)),
            "event_equal_weight_balanced_accuracy":float(balanced_accuracy_score(yvalid,pred,sample_weight=event_weights)),
            "event_equal_weight_within_one_bin":float(np.average(np.abs(yvalid-pred)<=1,weights=event_weights)),
            "reference_time_mae_hours":float(np.mean(np.abs(error))),
            "reference_time_rmse_hours":float(np.sqrt(np.mean(error**2))),
            "event_equal_weight_reference_time_mae_hours":float(np.average(np.abs(error),weights=event_weights)),
            "reference_time_median_abs_error_hours":float(np.median(np.abs(error))),
            "reference_time_bias_hours":float(np.mean(error))}
        brier,ece,bins,per_class=calibration(yvalid,proba)
        m["multiclass_brier"]=brier; m["confidence_ece_10_bins"]=ece
        metrics_rows.append(m)
        precision,recall,f1,support=precision_recall_fscore_support(yvalid,pred,labels=np.arange(4),zero_division=0)
        for i in range(4):
            class_rows.append({"model":name,"interval":LABELS[i],"precision":float(precision[i]),
                "recall":float(recall[i]),"f1":float(f1[i]),"validation_rows":int(support[i])})
        mat=confusion_matrix(yvalid,pred,labels=np.arange(4))
        for i in range(4):
            for j in range(4):
                confusion_rows.append({"model":name,"actual":LABELS[i],"predicted":LABELS[j],"rows":int(mat[i,j]),
                    "ordinal_distance":abs(i-j)})
        for i in range(3):
            adjacency_rows.append({"model":name,"pair":f"{LABELS[i]} vs {LABELS[i+1]}",
                "earlier_to_later":int(mat[i,i+1]),"later_to_earlier":int(mat[i+1,i]),
                "total_adjacent_confusions":int(mat[i,i+1]+mat[i+1,i])})
        for row in bins: confidence_rows.append({"model":name,**row})
        for row in per_class: calibration_rows.append({"model":name,**row})
        pred_frame=valid[["tool_id","time","fault_time","fault_type","event_id","ttf_hours","interval"]].copy()
        pred_frame["model"]=name
        pred_frame["predicted_interval"]=[LABELS[i] for i in pred]
        pred_frame["ordinal_error_bins"]=np.abs(yvalid-pred)
        pred_frame["reference_expected_hours"]=expected_hours
        pred_frame["reference_residual_hours"]=error
        for i,label in enumerate(LABELS):pred_frame[f"prob_{label}"]=proba[:,i]
        prediction_frames.append(pred_frame)
        print(json.dumps(m),flush=True)
    OUT.mkdir(parents=True)
    counts.to_csv(OUT/"full_raw_counts_by_tool_type_interval.csv",index=False,encoding="utf-8-sig")
    counts.groupby("interval",sort=False).agg(raw_4s_rows=("raw_4s_rows","sum"),
        unique_fault_events=("unique_fault_events","sum")).reindex(LABELS).reset_index().to_csv(
        OUT/"full_raw_interval_summary.csv",index=False,encoding="utf-8-sig")
    counts.groupby(["tool_id","interval"],sort=False)[["raw_4s_rows","unique_fault_events"]].sum().reset_index().to_csv(
        OUT/"full_raw_by_tool_interval.csv",index=False,encoding="utf-8-sig")
    counts.groupby(["fault_type","interval"],sort=False)[["raw_4s_rows","unique_fault_events"]].sum().reset_index().to_csv(
        OUT/"full_raw_by_fault_type_interval.csv",index=False,encoding="utf-8-sig")
    data.groupby(["split","interval"],sort=False).agg(rows=("event_id","size"),events=("event_id","nunique")).reset_index().to_csv(
        OUT/"model_population_by_interval.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame({"interval":LABELS,"train_ttf_median_hours":representative}).to_csv(
        OUT/"train_representative_times.csv",index=False,encoding="utf-8-sig")
    for filename,rows in (("metrics.csv",metrics_rows),("per_interval_metrics.csv",class_rows),
        ("confusion_matrix.csv",confusion_rows),("adjacent_confusions.csv",adjacency_rows),
        ("calibration_by_class.csv",calibration_rows),("calibration_confidence_bins.csv",confidence_rows)):
        pd.DataFrame(rows).to_csv(OUT/filename,index=False,encoding="utf-8-sig")
    predictions=pd.concat(prediction_frames,ignore_index=True)
    predictions.to_csv(OUT/"validation_predictions.csv.gz",index=False,compression="gzip")
    predictions.groupby("model").reference_residual_hours.quantile([0,.1,.25,.5,.75,.9,1]).unstack().to_csv(
        OUT/"reference_time_residual_quantiles.csv",encoding="utf-8-sig")
    plot=predictions[predictions.model=="HGB_balanced"].iloc[::max(1,len(valid)//5000)]
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    ax[0].scatter(plot.ttf_hours,plot.reference_expected_hours,s=3,alpha=.2)
    ax[0].plot([0,24],[0,24],"k--",linewidth=1)
    ax[0].set(xlabel="Actual TTF (h)",ylabel="Probability-weighted reference time (h)")
    ax[1].hist(plot.reference_residual_hours,bins=50)
    ax[1].set(xlabel="Reference time minus actual (h)",ylabel="Rows")
    fig.tight_layout();fig.savefig(OUT/"reference_time_scatter_residual.png",dpi=150);plt.close(fig)
    report={"scope":"All actual raw 4-second records counted; no artificial rows. Classifier trained on deterministic 5-minute samples of existing 1-minute prepared observations, Train/Validation only.",
        "raw_fault_events_unique":len(fault_log),"intervals":LABELS,
        "target":"Next unique logged fault for same tool within 24h; target bin and TTF from fault time minus current time. No post-fault or future sensor feature.",
        "split":"Frozen Model A per-tool chronological Train/Validation with >=24h purge; event timestamp and feature-row timestamp must both be in the same partition. Same event cannot cross split. Reserved test not read by model.",
        "features":"Exactly existing 85 causal Model A features, unchanged; no new feature engineering or hyperparameter search.",
        "representative_time_rule":"Median observed TTF in sampled Train rows for each interval; validation TTF never used.",
        "reference_time_rule":"Sum over four class probabilities times Train representative hours; illustrative estimate, not validated exact RUL.",
        "calibration_rule":"Validation multiclass Brier, max-confidence ECE in 10 bins, and mean class probability vs observed fraction; no probability calibrator fitted.",
        "selection_caveat":"Rows selected because the true next fault occurs within 24h. Operational Model 2 must be gated by Model 1 predictions; end-to-end performance not measured here.",
        "test_caveat":"This validation was previously used for model selection; not an independent final test. The historically exposed final partition remains unopened by this model experiment.",
        "train_representative_hours":dict(zip(LABELS,representative)),
        "hgb_parameters":HGB_PARAMS,"baseline_feature_count":len(BASE_FEATURES),
        "fault_log_sha256":file_hash(FAULTS),"split_plan_sha256":file_hash(PLAN),
        "model_1_unchanged":True,"metrics":metrics_rows}
    (OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":main()
