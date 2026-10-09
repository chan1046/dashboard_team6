"""Real-data, unrestricted next-fault TTF baseline; never reads reserved Test sensors."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from audit_simple_full_fault_rul import candidates

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/experiments/simple_full_fault_rul'
MODELS = ROOT / 'output/models'
RAW = ROOT / 'data/processed'
PLAN = ROOT / 'output/experiments/forward_validation_ab_24h/split_plan.json'
SENSORS = ['IONGAUGEPRESSURE','ETCHBEAMVOLTAGE','ETCHBEAMCURRENT',
           'ETCHSUPPRESSORVOLTAGE','ETCHSUPPRESSORCURRENT',
           'FLOWCOOLFLOWRATE','FLOWCOOLPRESSURE',
           'ETCHGASCHANNEL1READBACK','ETCHPBNGASREADBACK']
SEED=42
HISTORY_SECONDS=3600
CADENCE_SECONDS=1800
SEQUENCE_POINTS=60
TRAIN_EVENT_CAP=64
MIN_OBSERVATIONS=60
MIN_SPAN_FRACTION=.8

def build_samples():
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    faults=pd.read_csv(RAW/'fault_events.csv').drop_duplicates(['tool_id','time','fault_type'])
    assert len(faults)==1211
    rng=np.random.default_rng(SEED)
    parts={p:{'x':[],'y':[],'meta':[]} for p in ('train','validation')}
    source_audit=[]
    for item in plan['tools']:
        tool=item['tool_id']
        end=int(item['validation_end_exclusive_second'])
        table=pq.read_table(RAW/f'{tool}_sensor_ttf.parquet',columns=['time']+SENSORS,
                            filters=[('time','<',end)])
        frame=table.to_pandas()
        t=frame['time'].to_numpy(dtype=np.int64)
        x=frame[SENSORS].to_numpy(dtype=np.float32)
        assert len(t)==0 or np.all(t[1:]>=t[:-1]),f'unsorted {tool}'
        log=faults.loc[faults.tool_id==tool].sort_values('time',kind='stable')
        prev=None
        for event in log.itertuples():
            f=int(event.time)
            if f>=end:break
            if f<int(item['train_end_exclusive_second']):
                phase='train'; phase_start=int(t[0]);phase_end=int(item['train_end_exclusive_second'])
            elif int(item['validation_start_second'])<=f<end:
                phase='validation';phase_start=int(item['validation_start_second']);phase_end=end
            else:
                prev=f;continue
            lo=max(0 if prev is None else int(np.searchsorted(t,prev,side='right')),
                   int(np.searchsorted(t,phase_start,side='left')))
            hi=min(int(np.searchsorted(t,f,side='left')),
                   int(np.searchsorted(t,phase_end,side='left')))
            anchors=candidates(t,lo,hi,HISTORY_SECONDS,CADENCE_SECONDS)
            raw_count=len(anchors)
            if phase=='train' and raw_count>TRAIN_EVENT_CAP:
                anchors=np.sort(rng.choice(anchors,size=TRAIN_EVENT_CAP,replace=False))
            source_audit.append({'tool_id':tool,'fault_event_id':int(event.fault_event_id),
              'fault_type':event.fault_type,'phase':phase,'candidate_anchors':raw_count,
              'retained_anchors':len(anchors),'previous_fault_time':prev,'fault_time':f})
            for a in anchors:
                a=int(a)
                st=int(np.searchsorted(t,t[a]-HISTORY_SECONDS,side='left'))
                st=max(st,lo)
                assert a-st+1>=MIN_OBSERVATIONS
                assert t[a]-t[st]>=MIN_SPAN_FRACTION*HISTORY_SECONDS
                assert t[a]<f and (prev is None or t[st]>prev)
                idx=np.linspace(st,a,SEQUENCE_POINTS,dtype=np.int64)
                seq=x[idx].copy()
                target=(f-int(t[a]))/3600
                assert target>0
                gaps=np.diff(t[st:a+1])
                parts[phase]['x'].append(seq)
                parts[phase]['y'].append(target)
                parts[phase]['meta'].append({'tool_id':tool,
                  'fault_event_id':int(event.fault_event_id),'fault_type':event.fault_type,
                  'anchor_time':int(t[a]),'fault_time':f,'target_ttf_hours':target,
                  'history_first_time':int(t[st]),'history_elapsed_seconds':int(t[a]-t[st]),
                  'max_internal_gap_seconds':int(gaps.max()) if len(gaps) else 0,
                  'downtime_gaps_over_60s':int((gaps>60).sum()),
                  'phase':phase})
            prev=f
        print('samples',tool,len(log),flush=True)
    for phase in parts:
        parts[phase]['x']=np.stack(parts[phase]['x']).astype(np.float32)
        parts[phase]['y']=np.asarray(parts[phase]['y'],dtype=np.float32)
        parts[phase]['meta']=pd.DataFrame(parts[phase]['meta'])
    return parts,pd.DataFrame(source_audit)

def summarize(x):
    last=x[:,-1,:]
    mean=x.mean(axis=1);std=x.std(axis=1)
    mini=x.min(axis=1);maxi=x.max(axis=1)
    slope=(x[:,-1,:]-x[:,0,:])/1.0 # sensor units per requested history hour
    return np.concatenate([last,mean,std,mini,maxi,slope],axis=1).astype(np.float32)

def metrics(y,p):
    e=np.abs(y-p)
    return {'rows':len(y),'mae_hours':float(e.mean()),
      'rmse_hours':float(np.sqrt(mean_squared_error(y,p))),
      'r2':float(r2_score(y,p)) if len(y)>1 else None,
      'smape_pct':float(np.mean(200*e/(np.abs(y)+np.abs(p)+1e-6))),
      **{f'within_{h}h_pct':float(100*np.mean(e<=h)) for h in (1,2,3,6)}}

def train_gru(xtrain,ytrain,weights,xval):
    torch.manual_seed(SEED);torch.set_num_threads(4)
    flat=xtrain.reshape(-1,xtrain.shape[-1])
    center=np.nanmedian(flat,axis=0).astype(np.float32)
    q25=np.nanpercentile(flat,25,axis=0).astype(np.float32)
    q75=np.nanpercentile(flat,75,axis=0).astype(np.float32)
    scale=np.maximum(q75-q25,1e-5)
    def transform(x):
        z=(x-center)/scale
        return np.clip(np.nan_to_num(z,nan=0,posinf=0,neginf=0),-20,20).astype(np.float32)
    tx=transform(xtrain);vx=transform(xval)
    target_scale=float(np.median(ytrain))
    class GRU(nn.Module):
        def __init__(self):
            super().__init__();self.gru=nn.GRU(len(SENSORS),32,batch_first=True)
            self.head=nn.Linear(32,1)
        def forward(self,x):
            return self.head(self.gru(x)[1][-1]).squeeze(-1)
    model=GRU();optim=torch.optim.Adam(model.parameters(),lr=0.001)
    ds=TensorDataset(torch.from_numpy(tx),torch.from_numpy(ytrain/target_scale),
                     torch.from_numpy(weights.astype(np.float32)))
    loader=DataLoader(ds,batch_size=256,shuffle=True)
    loss_history=[]
    for epoch in range(5):
        model.train();total=0
        for bx,by,bw in loader:
            optim.zero_grad();pred=model(bx)
            loss=(nn.functional.smooth_l1_loss(pred,by,reduction='none')*bw).sum()/bw.sum()
            loss.backward();optim.step();total+=float(loss)*len(bx)
        loss_history.append(total/len(ds));print('GRU epoch',epoch+1,loss_history[-1],flush=True)
    model.eval();pred=[]
    with torch.no_grad():
        for i in range(0,len(vx),512):pred.append(model(torch.from_numpy(vx[i:i+512])).numpy())
    prediction=np.maximum(0,np.concatenate(pred)*target_scale)
    return model,{'center':center,'scale':scale,'target_scale':target_scale,
                  'loss_history':loss_history},prediction

def main():
    if not (OUT/'availability_config.json').exists():
        raise FileNotFoundError('Run audit_simple_full_fault_rul.py first')
    # Only our own new experiment output names; existing baselines are never touched.
    for name in ('sample_metadata.csv.gz','metrics.csv','validation_predictions.csv.gz'):
        if (OUT/name).exists():raise FileExistsError(f'Refusing to overwrite {OUT/name}')
    MODELS.mkdir(parents=True,exist_ok=True)
    parts,sample_audit=build_samples()
    tr=parts['train'];va=parts['validation']
    assert set(tr['meta'].fault_event_id).isdisjoint(set(va['meta'].fault_event_id))
    assert np.max(tr['meta'].anchor_time)<np.max(tr['meta'].fault_time)
    sample_audit.to_csv(OUT/'selected_anchor_counts_by_event.csv',index=False)
    metadata=pd.concat([tr['meta'],va['meta']],ignore_index=True)
    metadata.to_csv(OUT/'sample_metadata.csv.gz',index=False,compression='gzip')
    train_counts=tr['meta'].fault_event_id.value_counts()
    weights=tr['meta'].fault_event_id.map(1/train_counts).to_numpy(dtype=np.float32)
    weights*=len(weights)/weights.sum()
    dummy=np.full(len(va['y']),float(np.median(tr['y'])),dtype=np.float32)
    sx=summarize(tr['x']);vx=summarize(va['x'])
    tree=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=31,
           learning_rate=.05,random_state=SEED,l2_regularization=1)
    tree.fit(sx,tr['y'],sample_weight=weights)
    tree_pred=np.maximum(0,tree.predict(vx))
    gru,scaler,gru_pred=train_gru(tr['x'],tr['y'],weights,va['x'])
    preds={'Dummy':dummy,'Tree_HGB_event_balanced':tree_pred,'GRU_event_balanced':gru_pred}
    detail=va['meta'].copy()
    for name,pred in preds.items():detail[name]=pred
    detail.to_csv(OUT/'validation_predictions.csv.gz',index=False,compression='gzip')
    rows=[]
    for name,pred in preds.items():
        allm=metrics(va['y'],pred)
        event_mae=pd.DataFrame({'id':va['meta'].fault_event_id,'error':np.abs(va['y']-pred)}).groupby('id').error.mean()
        allm['event_equal_mae_hours']=float(event_mae.mean())
        rows.append({'model':name,'subset':'all',**allm})
        for lo,hi,label in [(0,3,'0-3h'),(3,6,'3-6h'),(6,12,'6-12h'),(12,24,'12-24h'),(24,np.inf,'24h+')]:
            mask=(va['y']>lo if lo else va['y']>=lo)&(va['y']<=hi if hi==np.inf else va['y']<hi)
            if mask.any():rows.append({'model':name,'subset':label,**metrics(va['y'][mask],pred[mask])})
        for h in (24,12,6,3):
            mask=va['y']<=h
            if mask.any():rows.append({'model':name,'subset':f'within_{h}h',**metrics(va['y'][mask],pred[mask])})
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    bygroup=[]
    for name,pred in preds.items():
        temp=va['meta'][['tool_id','fault_type','fault_event_id']].copy()
        temp['error']=np.abs(va['y']-pred)
        for group in ('tool_id','fault_type'):
            for key,part in temp.groupby(group):
                bygroup.append({'model':name,'group':group,'value':key,'rows':len(part),
                  'events':part.fault_event_id.nunique(),'mae_hours':part.error.mean(),
                  'event_equal_mae_hours':part.groupby('fault_event_id').error.mean().mean()})
    pd.DataFrame(bygroup).to_csv(OUT/'metrics_by_tool_fault_type.csv',index=False)
    tr_event=tr['meta'].fault_event_id.drop_duplicates()
    v_event=va['meta'].fault_event_id.drop_duplicates()
    config={'history_minutes':60,'prediction_interval_minutes':30,
      'sequence_points':SEQUENCE_POINTS,'train_event_cap':TRAIN_EVENT_CAP,
      'train_events':len(tr_event),'validation_events':len(v_event),
      'train_samples':len(tr['y']),'validation_samples':len(va['y']),
      'ttf_unit':'hours; next any-type fault; unbounded; no target filtering',
      'train_ttf_quantiles_hours':{str(q):float(np.quantile(tr['y'],q)) for q in (.5,.75,.9,.95,1)},
      'raw_sensors':SENSORS,'random_state':SEED,
      'sequence_rule':'60 evenly spaced actual observed indices over prior 1h; >=60 observations and >=80% observed elapsed span; no fill; gaps recorded',
      'anchor_rule':'latest observation at or before each 30min wall-clock grid, deduplicated',
      'split':'frozen per-Tool forward plan; >=24h purge; fault event in exactly one phase; all history restricted to same phase and prior-fault cycle; reserved Test sensor/label never read',
      'event_weight':'training event cap64 uniform seeded; weight per retained event = 1/its retained count; all Validation anchors retained',
      'tree':'HistGradientBoostingRegressor, 6 summary statistics of 9 raw sensors, max_iter120/max_leaf_nodes31/lr0.05/l2=1',
      'gru':'one layer 32 hidden, Adam lr0.001, 5 epochs, smooth L1 on target/Train median, no target clipping; prediction constrained >=0',
      'train_last_fault_time':int(tr['meta'].fault_time.max()),
      'validation_first_fault_time':int(va['meta'].fault_time.min())}
    (OUT/'training_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    with open(MODELS/'simple_full_fault_rul_tree.pkl','wb') as f:pickle.dump(tree,f)
    torch.save({'state_dict':gru.state_dict(),'config':config},MODELS/'simple_full_fault_rul_gru.pt')
    with open(MODELS/'simple_full_fault_rul_scaler.pkl','wb') as f:pickle.dump(scaler,f)
    print(json.dumps(config,indent=2),flush=True)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':main()
