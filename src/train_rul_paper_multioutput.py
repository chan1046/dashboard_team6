"""Paper-inspired raw 4-second, three-head RUL baseline on frozen forward split.

This is a developmental adaptation, not an exact reproduction of the 2018
architecture. No Model 1 training, no final-test rows in fitting/evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/experiments/rul_paper_reproduction_baseline"
MODEL_DIR = ROOT / "output/models"
PLAN = ROOT / "output/experiments/forward_validation_ab_24h/split_plan.json"
FAULTS = ROOT / "data/processed/fault_events.csv"
RAW = ROOT / "data/processed"
SENSORS = ["IONGAUGEPRESSURE", "ETCHBEAMVOLTAGE", "ETCHBEAMCURRENT",
           "ETCHSUPPRESSORVOLTAGE", "ETCHSUPPRESSORCURRENT",
           "FLOWCOOLFLOWRATE", "FLOWCOOLPRESSURE",
           "ETCHGASCHANNEL1READBACK", "ETCHPBNGASREADBACK"]
TYPES = ["pressure_low", "pressure_high", "leak"]
LENGTH, OVERLAP, STRIDE, RESAMPLE_STEP = 2000, 500, 1500, 20
STEPS = LENGTH // RESAMPLE_STEP
SEED = 42


class MultiHeadGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(len(SENSORS), 32, batch_first=True)
        self.head = nn.Linear(32, len(TYPES))

    def forward(self, x):
        return self.head(self.gru(x)[1][-1])


def read_raw_before_boundary(path, boundary):
    columns = ["time", "FIXTURESHUTTERPOSITION", *SENSORS]
    table = pq.read_table(path, columns=columns, filters=[("time", "<", boundary)])
    frame = table.to_pandas()
    if len(frame) and (frame.time >= boundary).any():
        raise AssertionError("Reserved test sensor row returned by filter")
    return frame


def construct(plan, faults):
    sequences, rows, event_candidates = [], [], []
    for item in plan["tools"]:
        name = item["tool_id"]
        frame = read_raw_before_boundary(RAW/f"{name}_sensor_ttf.parquet",
                                         item["validation_end_exclusive_second"])
        frame = frame.sort_values("time", kind="stable")
        time_all = frame.time.to_numpy(np.int64)
        operating = frame.FIXTURESHUTTERPOSITION.to_numpy() == 1
        t = time_all[operating]
        values = frame.loc[operating, SENSORS].to_numpy(np.float32)
        # The paper's operating filter removes down-time measurements. Missing
        # sensor values are not synthesized; a window containing any is dropped.
        valid_sensor = np.isfinite(values).all(axis=1)
        # The reserved final partition must not even enter availability counts.
        log = faults[(faults.tool_id == name) &
                     (faults.time < item["validation_end_exclusive_second"])]
        type_times = {typ: np.sort(log.loc[log.fault_type == typ, "time"].to_numpy(np.int64)) for typ in TYPES}
        for typ in TYPES:
            ft = type_times[typ]
            if len(ft):
                start = np.searchsorted(t, np.r_[t[0]-1, ft[:-1]], side="right")
                end = np.searchsorted(t, ft, side="left")
                event_candidates.extend({"tool_id":name, "fault_type":typ,"fault_time":int(f),
                    "operating_obs_since_previous_same_type":int(b-a),
                    "has_2000_operating_obs":bool(b-a>=LENGTH)}
                    for f,a,b in zip(ft,start,end))
        # Per-tool chronological windows with exactly 500 operating observations
        # overlap. They may straddle stops; no values are imputed across gaps.
        n_windows = max(0, (len(t)-LENGTH)//STRIDE+1)
        kept = {"train":0,"validation":0}
        for wi in range(n_windows):
            left = wi*STRIDE
            right = left+LENGTH-1
            anchor = int(t[right]); start_time=int(t[left])
            if anchor < item["train_end_exclusive_second"]:
                split="train"
            elif (start_time >= item["validation_start_second"] and
                  anchor < item["validation_end_exclusive_second"]):
                split="validation"
            else:
                continue
            if not valid_sensor[left:left+LENGTH].all():
                continue
            target = np.full(3, np.nan, dtype=np.float32)
            events = [None]*3
            split_end = (item["train_end_exclusive_second"] if split=="train"
                         else item["validation_end_exclusive_second"])
            for j,typ in enumerate(TYPES):
                ft = type_times[typ]
                pos = int(np.searchsorted(ft,anchor,side="right"))
                if pos < len(ft) and int(ft[pos]) < split_end:
                    # A head's entire history must belong to its current
                    # same-type run-to-fault cycle. Otherwise the previous
                    # fault is inside the input sequence.
                    previous_same_type = int(ft[pos-1]) if pos else None
                    if previous_same_type is None or start_time > previous_same_type:
                        target[j] = (int(ft[pos])-anchor)/3600
                        events[j] = f"{name}|{typ}|{int(ft[pos])}"
            if not np.isfinite(target).any():
                # Preserve the censored population count through candidate
                # metadata; all-masked rows have no supervised loss.
                continue
            raw_gap = np.diff(time_all[np.searchsorted(time_all,start_time):
                                           np.searchsorted(time_all,anchor,side="right")])
            operating_gap = np.diff(t[left:left+LENGTH])
            selected = values[left:right+1:RESAMPLE_STEP]
            assert selected.shape == (STEPS,len(SENSORS))
            sequences.append(selected)
            rows.append({"tool_id":name,"split":split,"start_time":start_time,"time":anchor,
                "max_gap_seconds":int(operating_gap.max()),
                "max_raw_gap_seconds":int(raw_gap.max()) if len(raw_gap) else 0,
                "elapsed_seconds":anchor-start_time,
                "session_transition":bool(np.any(raw_gap>90)),
                **{f"ttf_h_{typ}":target[j] for j,typ in enumerate(TYPES)},
                **{f"event_id_{typ}":events[j] for j,typ in enumerate(TYPES)}})
            kept[split]+=1
        print(name,"operating",len(t),"windows",n_windows,"kept",kept,flush=True)
    x=np.stack(sequences).astype(np.float32)
    meta=pd.DataFrame(rows)
    candidate=pd.DataFrame(event_candidates)
    for typ in TYPES:
        key=f"event_id_{typ}"
        a=set(meta.loc[meta.split=="train",key].dropna())
        b=set(meta.loc[meta.split=="validation",key].dropna())
        if a&b:
            raise AssertionError(f"Fault event crosses Train/Validation: {typ}")
    return x,meta,candidate


def predict_network(model,x):
    model.eval();result=[]
    with torch.no_grad():
        for start in range(0,len(x),512):
            result.append(model(torch.from_numpy(x[start:start+512])).numpy())
    return np.vstack(result)


def metrics(y,p):
    error=np.abs(y-p)
    return {"mae_h":float(mean_absolute_error(y,p)),
            "rmse_h":float(np.sqrt(mean_squared_error(y,p))),
            "r2":float(r2_score(y,p)) if len(y)>1 else None,
            "within_1h":float(np.mean(error<=1)),"within_2h":float(np.mean(error<=2))}


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    model_path=MODEL_DIR/"rul_paper_multioutput_gru.pt"
    tree_path=MODEL_DIR/"rul_paper_tree_baseline.joblib"
    if model_path.exists() or tree_path.exists():
        raise FileExistsError("Refusing to overwrite trained model")
    plan=json.loads(PLAN.read_text(encoding="utf-8"))
    if len(plan["tools"])!=20:
        raise AssertionError("Frozen split changed")
    forbidden=("ttf","fault","next","time","lot","runnum")
    if any(any(token in name.lower() for token in forbidden) for name in SENSORS):
        raise AssertionError("Future feature in raw sensor list")
    faults=pd.read_csv(FAULTS).drop_duplicates(["tool_id","time","fault_type"])
    if len(faults)!=1211 or faults.duplicated(["tool_id","time"]).any():
        raise AssertionError("Fault population changed")
    np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
    x,meta,candidates=construct(plan,faults)
    train=meta.split.eq("train").to_numpy()
    valid=meta.split.eq("validation").to_numpy()
    if not train.any() or not valid.any():
        raise AssertionError("Missing train/validation windows")
    center=np.median(x[train].reshape(-1,len(SENSORS)),axis=0).astype(np.float32)
    iqr=(np.percentile(x[train].reshape(-1,len(SENSORS)),75,axis=0)-
         np.percentile(x[train].reshape(-1,len(SENSORS)),25,axis=0)).astype(np.float32)
    scale=np.where(iqr<1e-6,1,iqr).astype(np.float32)
    xt=((x[train]-center)/scale).astype(np.float32)
    xv=((x[valid]-center)/scale).astype(np.float32)
    yt=meta.loc[train,[f"ttf_h_{typ}" for typ in TYPES]].to_numpy(np.float32)
    yv=meta.loc[valid,[f"ttf_h_{typ}" for typ in TYPES]].to_numpy(np.float32)
    mask=np.isfinite(yt)
    # log1p keeps actual measured hours as target while moderating the large
    # range. A missing/censored target never contributes to the loss.
    target=np.log1p(np.nan_to_num(yt,nan=0)).astype(np.float32)
    dataset=TensorDataset(torch.from_numpy(xt),torch.from_numpy(target),
                          torch.from_numpy(mask.astype(np.float32)))
    loader=DataLoader(dataset,batch_size=128,shuffle=True,
                      generator=torch.Generator().manual_seed(SEED))
    model=MultiHeadGRU();opt=torch.optim.Adam(model.parameters(),lr=.001)
    for epoch in range(5):
        losses=[]
        model.train()
        for xb,yb,mb in loader:
            opt.zero_grad();pred=model(xb)
            squared=(pred-yb).square()*mb
            # Equal mean contribution from each type (not row-frequency driven).
            per_type=squared.sum(dim=0)/mb.sum(dim=0).clamp(min=1)
            loss=per_type.mean();loss.backward();opt.step()
            losses.append(float(loss))
        print("GRU epoch",epoch+1,"masked loss",float(np.mean(losses)),flush=True)
    pseq=np.expm1(predict_network(model,xv))
    pseq=np.maximum(pseq,0)
    flat_train=np.concatenate([xt[:,-1,:],xt.mean(axis=1),xt.std(axis=1)],axis=1)
    flat_valid=np.concatenate([xv[:,-1,:],xv.mean(axis=1),xv.std(axis=1)],axis=1)
    pbase=np.full_like(yv,np.nan)
    ptree=np.full_like(yv,np.nan)
    trees={}
    for j,typ in enumerate(TYPES):
        known=np.isfinite(yt[:,j])
        if known.sum()<10:
            continue
        pbase[:,j]=float(np.median(yt[known,j]))
        tree=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=31,learning_rate=.05,
            l2_regularization=1,early_stopping=False,random_state=SEED)
        tree.fit(flat_train[known],np.log1p(yt[known,j]))
        ptree[:,j]=np.maximum(0,np.expm1(tree.predict(flat_valid)))
        trees[typ]=tree
    OUT.mkdir(parents=True)
    candidates.to_csv(OUT/"fault_event_sequence_availability.csv",index=False,encoding="utf-8-sig")
    meta.to_csv(OUT/"sequence_metadata_and_targets.csv.gz",index=False,compression="gzip")
    metric_rows=[];event_rows=[];tool_rows=[];fold_rows=[];prediction_rows=[]
    valid_meta=meta.loc[valid].reset_index(drop=True)
    for model_name,prob in (("Dummy",pbase),("Tree_HGB",ptree),("Sequence_GRU",pseq)):
        for j,typ in enumerate(TYPES):
            known=np.isfinite(yv[:,j])&np.isfinite(prob[:,j])
            if not known.any():
                continue
            part=valid_meta.loc[known].copy()
            part["actual_ttf_h"]=yv[known,j]
            part["predicted_ttf_h"]=prob[known,j]
            part["abs_error_h"]=np.abs(part.actual_ttf_h-part.predicted_ttf_h)
            part["fault_type"]=typ
            part["event_id"]=part[f"event_id_{typ}"]
            part["model"]=model_name
            prediction_rows.append(part[["tool_id","time","event_id","fault_type","model",
                                          "actual_ttf_h","predicted_ttf_h","abs_error_h"]])
            per_event=part.groupby("event_id").abs_error_h.mean()
            result={"model":model_name,"fault_type":typ,"validation_rows":len(part),
                "validation_events":part.event_id.nunique(),
                "event_equal_mae_h":float(per_event.mean()),
                **metrics(part.actual_ttf_h.to_numpy(),part.predicted_ttf_h.to_numpy())}
            metric_rows.append(result)
            for eid,error in per_event.items():
                event_rows.append({"model":model_name,"fault_type":typ,"event_id":eid,
                                   "event_mae_h":float(error)})
            for tool,g in part.groupby("tool_id"):
                tool_rows.append({"model":model_name,"fault_type":typ,"tool_id":tool,
                                  "rows":len(g),"events":g.event_id.nunique(),
                                  "mae_h":float(g.abs_error_h.mean()),
                                  "event_equal_mae_h":float(g.groupby("event_id").abs_error_h.mean().mean())})
            unique=np.sort(part.event_id.unique())
            event_time={eid:int(eid.split("|")[-1]) for eid in unique}
            boundaries=np.quantile(list(event_time.values()),[1/3,2/3],method="nearest")
            part["fold"]=1+np.searchsorted(boundaries,
                part.event_id.map(event_time).to_numpy(),side="right")
            for fold,g in part.groupby("fold"):
                fold_rows.append({"model":model_name,"fault_type":typ,"fold":int(fold),
                                  "rows":len(g),"events":g.event_id.nunique(),
                                  "mae_h":float(g.abs_error_h.mean()),
                                  "event_equal_mae_h":float(g.groupby("event_id").abs_error_h.mean().mean())})
            print(json.dumps(result),flush=True)
    pd.DataFrame(metric_rows).to_csv(OUT/"metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(event_rows).to_csv(OUT/"event_metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(tool_rows).to_csv(OUT/"tool_metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(OUT/"fold_metrics.csv",index=False,encoding="utf-8-sig")
    pd.concat(prediction_rows,ignore_index=True).to_csv(OUT/"validation_predictions.csv.gz",index=False,compression="gzip")
    torch.save({"state_dict":model.state_dict(),"scaler_center":center,"scaler_iqr":scale},model_path)
    joblib.dump({"models":trees,"scaler_center":center,"scaler_iqr":scale,
                 "features":SENSORS},tree_path)
    counts={typ:int(mask[:,j].sum()) for j,typ in enumerate(TYPES)}
    config={"input_data":"actual raw 4-second sensor parquet, shutter=1 observations",
        "sensor_columns":SENSORS,"fault_types":TYPES,"sequence_observations":LENGTH,
        "overlap_observations":OVERLAP,"stride_observations":STRIDE,
        "sequence_resampling":"every 20th of 2000 raw operating observations -> 100 GRU steps; all 2000 raw observations retained only for window eligibility and gap metadata",
        "target":"three separate next-same-type-fault elapsed hours; censored when no same-type fault observed within same split",
        "mask":"finite target only; masked squared error normalized by valid count per type, then mean of three type losses",
        "target_transform":"log1p(hours) for fit; expm1 and clamp at zero for inference; no artificial far-horizon target",
        "scaler":"Train-only global sensor median and IQR; scale=1 for IQR<1e-6",
        "split":"Frozen per-tool chronological Model A Train/Validation with >=24h boundary purge; window start/end and target event inside same split; reserved Test unused",
        "architecture":"shared one-layer GRU, 32 hidden units, three linear outputs, Adam .001, five fixed epochs, random_state 42",
        "gap":"No >90s rejection; no synthetic observations. Actual max operating/raw gap, elapsed time, raw >90s session transition saved per window as metadata, not model feature.",
        "model_file":str(model_path),"tree_model_file":str(tree_path),
        "train_windows":int(train.sum()),"validation_windows":int(valid.sum()),
        "train_valid_target_counts":counts,"validation_valid_target_counts":{
            typ:int(np.isfinite(yv[:,j]).sum()) for j,typ in enumerate(TYPES)},
        "caveat":"Developmental paper-inspired baseline, not exact paper architecture or official challenge score; validation is reused development data."}
    (OUT/"experiment_config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Saved",OUT,flush=True)


if __name__=="__main__":
    main()
