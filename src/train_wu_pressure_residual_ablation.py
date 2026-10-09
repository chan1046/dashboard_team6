"""Train-only Wu-style pressure residual and matched, strictly segmented RUL A/B.

Development study only. No existing Model 1 or 85-feature artifact is read.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_rul_paper_multioutput import MultiHeadGRU, SENSORS, TYPES, SEED

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/experiments/wu_pressure_residual_ablation'
MODEL_DIR=ROOT/'output/models'
RAW=ROOT/'data/processed'
PLAN=ROOT/'output/experiments/forward_validation_ab_24h/split_plan.json'
WINDOWS=OUT/'strict_sequence_target_availability.csv'
PRESSURE='FLOWCOOLPRESSURE'
FLOW='FLOWCOOLFLOWRATE'
HOUR=3600
DAY=86400


class PressureMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(1,10),nn.Tanh(),nn.Linear(10,50),
                               nn.Tanh(),nn.Linear(50,1))

    def forward(self,x):
        return self.net(x)


class MultiHeadGRU10(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru=nn.GRU(10,32,batch_first=True)
        self.head=nn.Linear(32,3)

    def forward(self,x):
        return self.head(self.gru(x)[1][-1])


def nearest_fault_distance(times, faults):
    pos=np.searchsorted(faults,times,side='right')
    after=np.full(len(times),np.inf,dtype=np.float64)
    before=np.full(len(times),np.inf,dtype=np.float64)
    yes=pos<len(faults)
    after[yes]=faults[pos[yes]]-times[yes]
    yes=pos>0
    before[yes]=times[yes]-faults[pos[yes]-1]
    return before,after


def read_tool(item,cols):
    tab=pq.read_table(RAW/f"{item['tool_id']}_sensor_ttf.parquet",
        columns=list(dict.fromkeys(['time','FIXTURESHUTTERPOSITION',*cols])),
        filters=[('time','<',item['validation_end_exclusive_second'])])
    f=tab.to_pandas()
    if len(f) and f.time.max()>=item['validation_end_exclusive_second']:
        raise AssertionError('Reserved Test row read')
    return f


def normal_mask(f,item,all_faults,phase):
    t=f.time.to_numpy(np.int64)
    operating=f.FIXTURESHUTTERPOSITION.to_numpy()==1
    before,after=nearest_fault_distance(t,all_faults)
    if phase=='train':
        inside=t<int(item['train_end_exclusive_second'])-DAY
    else:
        inside=(t>=int(item['validation_start_second']))&(t<int(item['validation_end_exclusive_second'])-DAY)
    return inside&operating&(before>DAY)&(after>DAY)&np.isfinite(f[FLOW])&np.isfinite(f[PRESSURE])


def pressure_predict(model,flow,scaler):
    out=np.empty(len(flow),dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0,len(flow),65536):
            values=((flow[i:i+65536]-scaler['flow_mean'])/scaler['flow_std']).astype(np.float32)
            pred=model(torch.from_numpy(values[:,None])).numpy().ravel()
            out[i:i+65536]=pred*scaler['pressure_std']+scaler['pressure_mean']
    return out


def train_pressure(plan,faults):
    q,p=[],[];per_tool=[]
    for item in plan['tools']:
        f=read_tool(item,[FLOW,PRESSURE]);tool=item['tool_id']
        ft=np.sort(faults.loc[faults.tool_id==tool,'time'].unique())
        eligible=normal_mask(f,item,ft,'train')
        chosen=eligible&(np.arange(len(f))%256==0)
        q.append(f.loc[chosen,FLOW].to_numpy(np.float32))
        p.append(f.loc[chosen,PRESSURE].to_numpy(np.float32))
        per_tool.append({'tool_id':tool,'train_normal_operating_rows':int(eligible.sum()),
                         'mlp_training_sample_rows':int(chosen.sum())})
        print('pressure sample',tool,int(chosen.sum()),flush=True)
    q=np.concatenate(q);p=np.concatenate(p)
    if len(q)<100:raise AssertionError('Too few Train-normal pressure pairs')
    scaler={'flow_mean':float(q.mean()),'flow_std':float(max(q.std(),1e-6)),
            'pressure_mean':float(p.mean()),'pressure_std':float(max(p.std(),1e-6))}
    x=((q-scaler['flow_mean'])/scaler['flow_std']).astype(np.float32)[:,None]
    y=((p-scaler['pressure_mean'])/scaler['pressure_std']).astype(np.float32)[:,None]
    model=PressureMLP();opt=torch.optim.Adam(model.parameters(),lr=.001)
    loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),
        batch_size=1024,shuffle=True,generator=torch.Generator().manual_seed(SEED))
    for epoch in range(8):
        model.train();losses=[]
        for a,b in loader:
            opt.zero_grad();loss=((model(a)-b)**2).mean();loss.backward();opt.step()
            losses.append(float(loss))
        print('MLP epoch',epoch+1,'MSE',float(np.mean(losses)),flush=True)
    torch.save({'state_dict':model.state_dict(),'input':[FLOW],'target':PRESSURE,
                'hidden':[10,50],'activation':'tanh','random_state':SEED},
               MODEL_DIR/'wu_normal_pressure_mlp.pt')
    joblib.dump(scaler,MODEL_DIR/'wu_normal_pressure_scaler.joblib')
    pd.DataFrame(per_tool).to_csv(OUT/'pressure_training_population.csv',index=False)
    return model,scaler


def build_residuals_and_sequences(plan,faults,mlp,scaler):
    windows=pd.read_csv(WINDOWS)
    xraw=[];xplus=[];metas=[];eda=[];validation_stats=[];tool_normal=[]
    for item in plan['tools']:
        tool=item['tool_id'];f=read_tool(item,SENSORS)
        t=f.time.to_numpy(np.int64);flow=f[FLOW].to_numpy(np.float32)
        pressure=f[PRESSURE].to_numpy(np.float32)
        finite=np.isfinite(flow)&np.isfinite(pressure)
        residual=np.full(len(f),np.nan,dtype=np.float32)
        residual[finite]=pressure_predict(mlp,flow[finite],scaler)-pressure[finite]
        ft=np.sort(faults.loc[faults.tool_id==tool,'time'].unique())
        normal=normal_mask(f,item,ft,'validation')
        vals=residual[normal]
        if len(vals):
            validation_stats.append({'tool_id':tool,'normal_validation_rows':len(vals),
              'sum_abs_error':float(np.abs(vals).sum()),'sum_sq_error':float(np.square(vals).sum()),
              'sum_actual':float(pressure[normal].sum()),'sum_actual_sq':float(np.square(pressure[normal]).sum())})
            sample=vals[np.arange(len(vals))%64==0]
            tool_normal.append({'tool_id':tool,'sampled_normal_rows':len(sample),
                'residual_mean':float(vals.mean()),'residual_std':float(vals.std()),
                'residual_median_sample':float(np.median(sample)),'abs_residual_median_sample':float(np.median(np.abs(sample)))})
        # Residuals are inferred from contemporaneous q and p only. Fault times
        # enter the following retrospective descriptive labels, never the MLP.
        selected=(np.arange(len(f))%64==0)&(f.FIXTURESHUTTERPOSITION.to_numpy()==1)&finite
        selected &= ((t<int(item['train_end_exclusive_second']))|
                     (t>=int(item['validation_start_second'])))
        si=np.flatnonzero(selected)
        if len(si):
            part=pd.DataFrame({'tool_id':tool,'time':t[si],
                'phase':np.where(t[si]<item['train_end_exclusive_second'],'train','validation'),
                'residual':residual[si]})
            for typ in TYPES:
                type_ft=np.sort(faults.loc[(faults.tool_id==tool)&(faults.fault_type==typ),'time'].unique())
                _,d=nearest_fault_distance(t[si],type_ft)
                part[f'next_{typ}_h']=d/3600
            eda.append(part)
        wg=windows[windows.tool_id==tool]
        for row in wg.itertuples():
            left=int(row.start_raw_row)
            right=int(row.end_raw_row)
            if right>=len(t) or int(t[left])!=int(row.start_time) or int(t[right])!=int(row.time):
                raise AssertionError('Window raw-row alignment mismatch')
            if not np.isfinite(residual[left:right+1]).all():continue
            raw=f.iloc[left:right+1][SENSORS].to_numpy(np.float32)
            if not np.isfinite(raw).all():continue
            xa=raw[::20]
            xb=np.column_stack([xa,residual[left:right+1:20]])
            if xa.shape!=(100,9) or xb.shape!=(100,10):raise AssertionError('Shape mismatch')
            xraw.append(xa);xplus.append(xb)
            metas.append({c:getattr(row,c) for c in windows.columns})
        print('residual',tool,'operating',int((f.FIXTURESHUTTERPOSITION==1).sum()),flush=True)
    stats=pd.DataFrame(validation_stats);stats.to_csv(OUT/'pressure_validation_by_tool.csv',index=False)
    pd.DataFrame(tool_normal).to_csv(OUT/'normal_residual_by_tool.csv',index=False)
    sampled=pd.concat(eda,ignore_index=True)
    sampled.to_csv(OUT/'residual_eda_sample.csv.gz',index=False,compression='gzip')
    return np.stack(xraw),np.stack(xplus),pd.DataFrame(metas),stats,sampled


def pressure_metrics(stats):
    n=stats.normal_validation_rows.sum()
    mae=stats.sum_abs_error.sum()/n
    rmse=np.sqrt(stats.sum_sq_error.sum()/n)
    sse=stats.sum_sq_error.sum()
    sst=stats.sum_actual_sq.sum()-stats.sum_actual.sum()**2/n
    return {'normal_validation_rows':int(n),'mae':float(mae),'rmse':float(rmse),
            'r2':float(1-sse/sst) if sst>0 else None}


def residual_eda(sampled):
    rows=[]
    for typ in TYPES:
        dist=sampled[f'next_{typ}_h']
        bins=[(0,1,'0-1h'),(1,3,'1-3h'),(3,6,'3-6h'),(6,12,'6-12h'),(12,24,'12-24h'),(24,np.inf,'>24h')]
        for low,high,name in bins:
            v=sampled.loc[(dist>low)&(dist<=high),'residual'].dropna().to_numpy()
            if len(v)==0:continue
            rows.append({'fault_type':typ,'bin':name,'sampled_rows':len(v),
              'mean':float(v.mean()),'median':float(np.median(v)),
              'std':float(v.std()),'abs_mean':float(np.abs(v).mean()),
              'abs_median':float(np.median(np.abs(v))),
              'q10':float(np.quantile(v,.1)),'q90':float(np.quantile(v,.9))})
    pd.DataFrame(rows).to_csv(OUT/'residual_by_fault_distance.csv',index=False)
    return rows


def train_gru(x,meta,name):
    train=meta.phase.eq('train').to_numpy();valid=meta.phase.eq('validation').to_numpy()
    center=np.median(x[train].reshape(-1,x.shape[-1]),axis=0).astype(np.float32)
    q=np.percentile(x[train].reshape(-1,x.shape[-1]),[25,75],axis=0)
    scale=np.where(q[1]-q[0]<1e-6,1,q[1]-q[0]).astype(np.float32)
    xt=((x[train]-center)/scale).astype(np.float32)
    xv=((x[valid]-center)/scale).astype(np.float32)
    yt=meta.loc[train,[f'ttf_h_{t}' for t in TYPES]].to_numpy(np.float32)
    yv=meta.loc[valid,[f'ttf_h_{t}' for t in TYPES]].to_numpy(np.float32)
    mask=np.isfinite(yt).astype(np.float32)
    target=np.log1p(np.nan_to_num(yt,nan=0)).astype(np.float32)
    loader=DataLoader(TensorDataset(torch.from_numpy(xt),torch.from_numpy(target),torch.from_numpy(mask)),
        batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(SEED))
    torch.manual_seed(SEED)
    model=MultiHeadGRU() if name=='A' else MultiHeadGRU10()
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    for epoch in range(5):
        model.train()
        for a,b,m in loader:
            opt.zero_grad();pred=model(a)
            loss=(((pred-b)**2*m).sum(0)/m.sum(0).clamp(min=1)).mean()
            loss.backward();opt.step()
    model.eval();preds=[]
    with torch.no_grad():
        for i in range(0,len(xv),512):preds.append(model(torch.from_numpy(xv[i:i+512])).numpy())
    pred=np.maximum(0,np.expm1(np.concatenate(preds)))
    model_path=MODEL_DIR/f'wu_residual_rul_gru_{name.lower()}.pt'
    torch.save({'state_dict':model.state_dict(),'scaler_center':center,'scaler_iqr':scale,
                'features':SENSORS+(['wu_pressure_residual'] if name=='B' else []),
                'sequence_shape':list(x.shape[1:]),'architecture':'GRU hidden32 one layer, 3 heads',
                'optimizer':'Adam 0.001','epochs':5,'target':'log1p hours with valid-head mask'},model_path)
    joblib.dump({'center':center,'iqr':scale},MODEL_DIR/f'wu_residual_rul_scaler_{name.lower()}.joblib')
    return yv,pred,valid,yt


def metric(y,p,events):
    e=np.abs(y-p);ss=np.sum((y-y.mean())**2)
    return {'rows':len(y),'events':len(np.unique(events)),
       'mae_h':float(e.mean()),'rmse_h':float(np.sqrt(np.mean((y-p)**2))),
       'r2':float(1-np.sum((y-p)**2)/ss) if ss>0 else None,
       'smape_pct':float(np.mean(200*e/np.maximum(y+p,1e-6))),
       'within_1h':float(np.mean(e<=1)),'within_2h':float(np.mean(e<=2)),
       'event_equal_mae_h':float(pd.DataFrame({'id':events,'e':e}).groupby('id').e.mean().mean())}


def evaluate(xa,xb,meta):
    metric_rows=[];tool_rows=[];fold_rows=[];pred_rows=[]
    for name,x in (('A',xa),('B',xb)):
        yv,pred,valid,yt=train_gru(x,meta,name)
        vm=meta.loc[valid].reset_index(drop=True)
        for j,typ in enumerate(TYPES):
            known=np.isfinite(yv[:,j]);events=vm.loc[known,f'event_{typ}'].to_numpy()
            if not known.any():continue
            for model_name,values in (('GRU',pred[known,j]),
               ('Dummy',np.full(known.sum(),np.median(yt[np.isfinite(yt[:,j]),j])))):
                y=yv[known,j];m=metric(y,values,events)
                metric_rows.append({'set':name,'model':model_name,'fault_type':typ,**m})
                part=vm.loc[known,['tool_id','time',f'event_{typ}']].copy().reset_index(drop=True)
                part['actual']=y;part['prediction']=values;part['abs_error']=np.abs(y-values)
                part['set']=name;part['model']=model_name;part['fault_type']=typ
                pred_rows.append(part)
                for tool,g in part.groupby('tool_id'):
                    tool_rows.append({'set':name,'model':model_name,'fault_type':typ,'tool_id':tool,
                        **metric(g.actual.to_numpy(),g.prediction.to_numpy(),g[f'event_{typ}'].to_numpy())})
                times=part[f'event_{typ}'].str.split('|').str[-1].astype(int)
                unique=np.sort(times.unique())
                boundaries=np.quantile(unique,[1/3,2/3]) if len(unique)>=3 else []
                part['fold']=np.searchsorted(boundaries,times,side='right')+1
                for fold,g in part.groupby('fold'):
                    fold_rows.append({'set':name,'model':model_name,'fault_type':typ,'fold':int(fold),
                        **metric(g.actual.to_numpy(),g.prediction.to_numpy(),g[f'event_{typ}'].to_numpy())})
    pd.DataFrame(metric_rows).to_csv(OUT/'rul_metrics.csv',index=False)
    pd.DataFrame(tool_rows).to_csv(OUT/'rul_tool_metrics.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(OUT/'rul_fold_metrics.csv',index=False)
    pd.concat(pred_rows,ignore_index=True).to_csv(OUT/'rul_predictions.csv',index=False)
    return metric_rows


def main():
    if not WINDOWS.exists():raise FileNotFoundError('Run segment and target audit first')
    for path in (MODEL_DIR/'wu_residual_rul_gru_a.pt',MODEL_DIR/'wu_residual_rul_gru_b.pt'):
        if path.exists():raise FileExistsError(f'Refusing to overwrite {path}')
    torch.manual_seed(SEED);np.random.seed(SEED);torch.set_num_threads(4)
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    faults=pd.read_csv(RAW/'fault_events.csv').drop_duplicates(['tool_id','time','fault_type'])
    mlp_path=MODEL_DIR/'wu_normal_pressure_mlp.pt'
    scaler_path=MODEL_DIR/'wu_normal_pressure_scaler.joblib'
    if mlp_path.exists() and scaler_path.exists():
        mlp=PressureMLP()
        mlp.load_state_dict(torch.load(mlp_path,map_location='cpu',weights_only=False)['state_dict'])
        scaler=joblib.load(scaler_path)
    elif not mlp_path.exists() and not scaler_path.exists():
        mlp,scaler=train_pressure(plan,faults)
    else:
        raise RuntimeError('Incomplete saved pressure model pair')
    xa,xb,meta,stats,sampled=build_residuals_and_sequences(plan,faults,mlp,scaler)
    meta.to_csv(OUT/'matched_sequence_metadata.csv',index=False)
    pm=pressure_metrics(stats);eda=residual_eda(sampled)
    metrics=evaluate(xa,xb,meta)
    config={'source':'real raw four-second sensor parquet','operating_segments':str(OUT/'operating_segments.csv.gz'),
        'sequence_windows':str(WINDOWS),'normal':'FSP=1; >24h from previous and next any-type fault; full 24h future within frozen Train',
        'pressure_mlp_input':[FLOW],'pressure_mlp_target':PRESSURE,
        'normal_fit_subsampling':'deterministic raw row index modulo 256 == 0; full eligible Train-normal population counted',
        'residual':'MLP(flow)-actual pressure; computed for all finite Train/Validation operating observations without fault/TTF input',
        'eda_subsampling':'deterministic raw row index modulo 64 == 0; quantiles and fault-distance distributions are sampled',
        'pressure_validation_metric_population':'all finite Validation operating observations >24h from any fault, with full 24h horizon observed',
        'split_plan':str(PLAN),'model1':'untouched','engineered_85':'unused',
        'gru':'same architecture, Adam 0.001, five epochs, log1p target, head mask, and 2000/500 window rule as previous paper baseline; only channels differ',
        'validation_reuse':'development validation, not independent final test',
        'pressure_metrics':pm,'matched_windows':len(meta),'rul_metric_rows':len(metrics)}
    (OUT/'experiment_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('PRESSURE',pm,'WINDOWS',len(meta),'RUL',len(metrics),flush=True)


if __name__=='__main__':main()
