"""Causal sequence RUL research on the frozen, pre-test chronological split.

Run once. No Model 1 fitting, no test scoring, and no TTF/fault metadata in X.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from forward_validation_ab_24h import read_pretest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/experiments/sequence_rul_baseline"
MODELS = ROOT / "output/models"
PLAN = ROOT / "output/experiments/forward_validation_ab_24h/split_plan.json"
FAULTS = ROOT / "data/processed/fault_events.csv"
PREP = ROOT / "output/baseline_recovery/prepared"
SCORES1 = ROOT / "output/experiments/forward_validation_ab_24h/validation_probabilities_a.npz"
SENSORS = ["FLOWCOOLPRESSURE", "FLOWCOOLFLOWRATE", "IONGAUGEPRESSURE",
           "ETCHGASCHANNEL1READBACK", "ETCHPBNGASREADBACK"]
FEATURES = SENSORS + ["stage"]
WINDOWS = [6, 12, 24]
LENGTHS = [30, 60, 120]
ANCHOR_SECONDS = 600
MAX_GAP_SECONDS = 90
SEED = 42


class GRURegressor(nn.Module):
    def __init__(self, dims=6):
        super().__init__()
        self.gru = nn.GRU(dims, 16, batch_first=True)
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        return self.head(self.gru(x)[1][-1]).squeeze(-1)


def predict_gru(model, x, batch=512):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            chunks.append(model(torch.from_numpy(x[start:start+batch])).numpy())
    return np.concatenate(chunks) if chunks else np.empty(0)


def train_gru(x, y, center, scale):
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    model = GRURegressor(x.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    dataset = TensorDataset(torch.from_numpy(((x-center)/scale).astype(np.float32)),
                            torch.from_numpy((y/24).astype(np.float32)))
    loader = DataLoader(dataset, batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    model.train()
    for epoch in range(3):
        losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = nn.functional.smooth_l1_loss(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        print(f"  GRU epoch {epoch+1}: loss {np.mean(losses):.5f}", flush=True)
    return model


def summary(x):
    # All statistics refer solely to the observed history ending at prediction time.
    last = x[:, -1, :]
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    slope = (last-x[:, 0, :]) / max(1, x.shape[1]-1)
    return np.concatenate([last, mean, std, slope], axis=1).astype(np.float32)


def measures(y, p):
    err = np.abs(y-p)
    return {"mae_h": float(mean_absolute_error(y,p)),
            "rmse_h": float(np.sqrt(mean_squared_error(y,p))),
            "r2": float(r2_score(y,p)) if len(y)>1 else None,
            "within_1h": float(np.mean(err<=1)),
            "within_2h": float(np.mean(err<=2))}


def load_tool(item, fault_log):
    tool = item["tool_id"]
    columns = ["time", "session_id", *FEATURES]
    frame = read_pretest(PREP/f"{tool}.parquet", item["test_boundary_index"], columns)
    t = frame.time.to_numpy(np.int64)
    session = frame.session_id.to_numpy(np.int64)
    values = frame[FEATURES].to_numpy(np.float32)
    bad = ~np.isfinite(values)
    values[bad] = np.nan
    # Gap/session boundaries reset history. 90s tolerates minute jitter only.
    breaks = np.r_[True, (np.diff(t)>MAX_GAP_SECONDS) | (np.diff(t)<=0) | (np.diff(session)!=0)]
    run_start = np.maximum.accumulate(np.where(breaks, np.arange(len(t)), 0))
    log = fault_log[fault_log.tool_id==tool].sort_values("time")
    ft = log.time.to_numpy(np.int64)
    pos = np.searchsorted(ft,t,side="right")
    exists = pos<len(ft)
    next_fault = np.full(len(t),np.iinfo(np.int64).max,dtype=np.int64)
    next_fault[exists] = ft[pos[exists]]
    delta = next_fault-t
    train = (t<item["train_end_exclusive_second"]) & (next_fault<item["train_end_exclusive_second"])
    valid = ((t>=item["validation_start_second"]) & (t<item["validation_end_exclusive_second"])
             & (next_fault>=item["validation_start_second"])
             & (next_fault<item["validation_end_exclusive_second"]))
    return {"tool":tool,"t":t,"v":values,"run_start":run_start,"next_fault":next_fault,
            "delta":delta,"train":train,"valid":valid,
            "fault_type":log.fault_type.to_numpy(),"fault_pos":pos,"item":item}


def samples(tools, length, window):
    xs, metas = [], []
    excluded_gap = 0
    for d in tools:
        t, v = d["t"], d["v"]
        in_window = (d["delta"]>0)&(d["delta"]<=window*3600)&(d["train"]|d["valid"])
        candidates = np.flatnonzero(in_window & ((t//60)%10==0))
        starts = candidates-length+1
        okay = (starts>=0)&(d["run_start"][candidates]<=starts)
        # Keep all 1-minute history rows; reject any non-finite raw feature.
        excluded_gap += int((~okay).sum())
        indices = candidates[okay]
        for i in indices:
            x = v[i-length+1:i+1]
            if not np.isfinite(x).all():
                continue
            xs.append(x)
            ft = int(d["next_fault"][i])
            metas.append((d["tool"],int(t[i]),ft,f"{d['tool']}|{ft}",
                          str(d["fault_type"][d["fault_pos"][i]]),
                          "train" if d["train"][i] else "validation",d["delta"][i]/3600))
    x = np.stack(xs).astype(np.float32)
    meta = pd.DataFrame(metas,columns=["tool_id","time","fault_time","event_id","fault_type","split","ttf_h"])
    return x,meta,excluded_gap


def evaluate(model_name, window, length, part, prediction, metric_rows, detail_rows, predictions):
    y = part.ttf_h.to_numpy(float)
    prediction = np.clip(np.asarray(prediction,float),0,window)
    error = np.abs(y-prediction)
    by_event = pd.DataFrame({"event_id":part.event_id.to_numpy(),"error":error}).groupby("event_id").error.mean()
    row = {"window_h":window,"sequence_minutes":length,"model":model_name,
           "validation_rows":len(part),"validation_events":part.event_id.nunique(),
           "event_equal_mae_h":float(by_event.mean()),**measures(y,prediction)}
    metric_rows.append(row)
    print(json.dumps(row),flush=True)
    p = part[["tool_id","event_id","fault_type","time","fault_time","ttf_h"]].copy()
    p["predicted_h"] = prediction
    p["abs_error_h"] = error
    p["model"] = model_name
    p["window_h"] = window
    p["sequence_minutes"] = length
    predictions.append(p)
    for kind,column in (("tool","tool_id"),("fault_type","fault_type"),("fold","fold")):
        if column == "fold":
            ordered = np.sort(part.time.unique())
            edges = [ordered[int(len(ordered)*q)] for q in (1/3,2/3)]
            p["fold"] = 1+np.searchsorted(edges,p.time.to_numpy(),side="right")
        for value,g in p.groupby(column):
            detail_rows.append({"window_h":window,"sequence_minutes":length,"model":model_name,
                                "group_type":kind,"group":value,"rows":len(g),
                                "events":g.event_id.nunique(),"mae_h":float(g.abs_error_h.mean()),
                                "event_equal_mae_h":float(g.groupby("event_id").abs_error_h.mean().mean())})


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if len(plan["tools"])!=20:
        raise AssertionError("Expected frozen 20-tool split")
    forbidden = ("ttf","fault","next","time","lot","runnum")
    if any(any(s in f.lower() for s in forbidden) for f in FEATURES):
        raise AssertionError("Target/future metadata in sequence features")
    faults = pd.read_csv(FAULTS).drop_duplicates(["tool_id","time","fault_type"])
    if faults.duplicated(["tool_id","time"]).any():
        raise AssertionError("Ambiguous simultaneous faults")
    torch.manual_seed(SEED); np.random.seed(SEED)
    tools = [load_tool(item,faults) for item in plan["tools"]]
    baseline = pd.read_csv(ROOT/"output/experiments/ttf_0_6h_regression/metrics.csv")
    OUT.mkdir(parents=True)
    baseline[baseline.model.isin(["Dummy_median","XGBoost"])].to_csv(OUT/"prior_0_6h_baselines.csv",index=False)
    metric_rows=[]; detail_rows=[]; predictions=[]; availability=[]; best=None
    for length in LENGTHS:
        for window in WINDOWS:
            x,meta,excluded = samples(tools,length,window)
            tr=meta.split.eq("train").to_numpy(); va=meta.split.eq("validation").to_numpy()
            if set(meta.loc[tr,"event_id"]) & set(meta.loc[va,"event_id"]):
                raise AssertionError("Event crosses train/validation")
            train,valid=meta.loc[tr],meta.loc[va]
            if not len(train) or not len(valid):
                raise AssertionError("No training/validation samples")
            availability.append({"window_h":window,"sequence_minutes":length,
                                 "train_rows":len(train),"train_events":train.event_id.nunique(),
                                 "validation_rows":len(valid),"validation_events":valid.event_id.nunique(),
                                 "excluded_for_gap_or_short_history":excluded})
            center=np.median(x[tr].reshape(-1,x.shape[-1]),axis=0).astype(np.float32)
            scale=(np.percentile(x[tr].reshape(-1,x.shape[-1]),75,axis=0)-
                   np.percentile(x[tr].reshape(-1,x.shape[-1]),25,axis=0)).astype(np.float32)
            scale=np.where(scale<1e-6,1,scale).astype(np.float32)
            xt=((x[tr]-center)/scale).astype(np.float32)
            xv=((x[va]-center)/scale).astype(np.float32)
            yt=train.ttf_h.to_numpy(float)
            dummy=DummyRegressor(strategy="median").fit(summary(xt),yt)
            tree=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=31,learning_rate=.05,
                                               l2_regularization=1,early_stopping=False,random_state=SEED)
            tree.fit(summary(xt),yt)
            gru=train_gru(x[tr],yt,center,scale)
            for name,p in (("Dummy",dummy.predict(summary(xv))),
                           ("Tree_HGB",tree.predict(summary(xv))),
                           ("Sequence_GRU",predict_gru(gru,xv)*24)):
                evaluate(name,window,length,valid,p,metric_rows,detail_rows,predictions)
            score=metric_rows[-1]["event_equal_mae_h"]
            if best is None or score<best[0]:
                best=(score,window,length,gru.state_dict(),center,scale,tree)
            del x,xt,xv,gru
    pd.DataFrame(availability).to_csv(OUT/"sample_availability.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(metric_rows).to_csv(OUT/"metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(detail_rows).to_csv(OUT/"group_metrics.csv",index=False,encoding="utf-8-sig")
    pd.concat(predictions,ignore_index=True).to_csv(OUT/"validation_predictions.csv.gz",index=False,compression="gzip")
    score,window,length,state,center,scale,tree=best
    model_path=MODELS/"sequence_rul_gru_baseline.pt"
    config_path=MODELS/"sequence_rul_gru_baseline_config.json"
    if model_path.exists() or config_path.exists():
        raise FileExistsError("Sequence RUL model/config already exists")
    torch.save({"state_dict":state,"center":center,"scale":scale},model_path)
    config={"architecture":"GRU 16 hidden units, 3 epochs, smooth L1 loss, target TTF hours /24",
            "feature_names":FEATURES,"sequence_minutes":length,"degradation_window_h":window,
            "max_gap_seconds":MAX_GAP_SECONDS,"same_session_only":True,
            "anchor_interval_seconds":ANCHOR_SECONDS,"random_state":SEED,
            "model_path":str(model_path),"event_equal_validation_mae_h":score,
            "conditional_selection_caveat":"Model trained only on true pre-fault windows; full-timeline scores are extrapolations."}
    config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    report={"config":config,"method":"Frozen per-tool chronological Model A train/validation; 24h boundary purge; no test access. Every sample sequence ends at its prediction timestamp and remains within one uninterrupted session with <=90s spacing.",
            "selection":"Small fixed GRU architecture; 9 window-length combinations; best by validation event-equal MAE. Reused validation is developmental, not independent test.",
            "features":"Five raw sensors plus current/past stage; no 85 derived rolling/lag fields duplicated in sequence. No TTF, next fault, absolute time, tool ID or event ID in X.",
            "prior_baseline":"prior_0_6h_baselines.csv; different split/population and therefore not a directly matched improvement test."}
    (OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("selected",json.dumps(config),flush=True)


if __name__=="__main__":
    main()
