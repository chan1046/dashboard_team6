"""Compare cap64 legacy, all-anchor event weighting, and 32-anchor equal-event RUL."""
from __future__ import annotations
import json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error,r2_score
from scipy.stats import pearsonr,spearmanr
from audit_simple_full_fault_rul import candidates
from train_simple_full_fault_rul import ROOT,RAW,PLAN,SENSORS,summarize

BASE=ROOT/'output/experiments/simple_full_fault_rul'
OUT=ROOT/'output/experiments/event_balanced_rul'
MODELS=ROOT/'output/models'
SEED=42
BINS=[0,3,6,12,24,72,np.inf]
NAMES=['0-3h','3-6h','6-12h','12-24h','24-72h','72h+']

class SmallGRU(nn.Module):
    def __init__(self):
        super().__init__();self.gru=nn.GRU(9,32,batch_first=True);self.head=nn.Linear(32,1)
    def forward(self,x):return self.head(self.gru(x)[1][-1]).squeeze(-1)

def build():
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    faults=pd.read_csv(RAW/'fault_events.csv').drop_duplicates(['tool_id','time','fault_type'])
    rng=np.random.default_rng(SEED)
    result={p:{'x':[],'y':[],'meta':[],'a':[],'c':[]} for p in ('train','validation')}
    for item in plan['tools']:
        tool=item['tool_id'];end=int(item['validation_end_exclusive_second'])
        frame=pq.read_table(RAW/f'{tool}_sensor_ttf.parquet',columns=['time']+SENSORS,
                            filters=[('time','<',end)]).to_pandas()
        t=frame.time.to_numpy(dtype=np.int64)
        x=frame[SENSORS].to_numpy(dtype=np.float32)
        prev=None
        for event in faults[faults.tool_id==tool].sort_values('time',kind='stable').itertuples():
            f=int(event.time)
            if f>=end:break
            if f<int(item['train_end_exclusive_second']):
                phase='train';start=int(t[0]);stop=int(item['train_end_exclusive_second'])
            elif int(item['validation_start_second'])<=f<end:
                phase='validation';start=int(item['validation_start_second']);stop=end
            else:prev=f;continue
            lo=max(0 if prev is None else int(np.searchsorted(t,prev,side='right')),
                   int(np.searchsorted(t,start,side='left')))
            hi=min(int(np.searchsorted(t,f,side='left')),
                   int(np.searchsorted(t,stop,side='left')))
            anchors=candidates(t,lo,hi,3600,1800)
            n=len(anchors)
            if n:
                base=len(result[phase]['y'])
                if phase=='train':
                    a=np.arange(n)
                    if n>64:a=np.sort(rng.choice(a,size=64,replace=False))
                    c=np.unique(np.rint(np.linspace(0,n-1,min(32,n))).astype(int))
                    result[phase]['a'].extend((base+a).tolist())
                    result[phase]['c'].extend((base+c).tolist())
                for anchor in anchors:
                    anchor=int(anchor)
                    st=max(lo,int(np.searchsorted(t,t[anchor]-3600,side='left')))
                    ix=np.linspace(st,anchor,60,dtype=np.int64)
                    assert len(ix)==60 and t[anchor]<f and (prev is None or t[st]>prev)
                    assert t[anchor]-t[st]>=2880
                    target=(f-int(t[anchor]))/3600
                    result[phase]['x'].append(x[ix].copy())
                    result[phase]['y'].append(target)
                    result[phase]['meta'].append((tool,int(event.fault_event_id),event.fault_type,
                                                  int(t[anchor]),f,target))
            prev=f
        print('dataset',tool,flush=True)
    columns=['tool_id','fault_event_id','fault_type','anchor_time','fault_time','target_ttf_hours']
    for p in result:
        result[p]['x']=np.stack(result[p]['x']).astype(np.float32)
        result[p]['y']=np.asarray(result[p]['y'],dtype=np.float32)
        result[p]['meta']=pd.DataFrame(result[p]['meta'],columns=columns)
        result[p]['a']=np.asarray(result[p]['a'],dtype=np.int64)
        result[p]['c']=np.asarray(result[p]['c'],dtype=np.int64)
    return result

def weights(meta):
    n=meta.fault_event_id.value_counts()
    w=meta.fault_event_id.map(1/n).to_numpy(dtype=np.float32)
    return w*(len(w)/w.sum())

def train_tree(x,y,w,vx):
    model=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=31,
           learning_rate=.05,random_state=SEED,l2_regularization=1)
    model.fit(summarize(x),y,sample_weight=w)
    return model,np.maximum(0,model.predict(summarize(vx)))

def train_gru(x,y,w,vx,legacy_scaler):
    torch.manual_seed(SEED);torch.set_num_threads(4)
    center=legacy_scaler['center'];scale=legacy_scaler['scale']
    target_scale=float(legacy_scaler['target_scale'])
    def z(a):return np.clip(np.nan_to_num((a-center)/scale,nan=0,posinf=0,neginf=0),-20,20).astype(np.float32)
    ds=TensorDataset(torch.from_numpy(z(x)),torch.from_numpy(y/target_scale),torch.from_numpy(w))
    dl=DataLoader(ds,batch_size=256,shuffle=True)
    model=SmallGRU();optim=torch.optim.Adam(model.parameters(),lr=.001)
    for ep in range(5):
        model.train();losses=[]
        for bx,by,bw in dl:
            optim.zero_grad();p=model(bx)
            loss=(nn.functional.smooth_l1_loss(p,by,reduction='none')*bw).sum()/bw.sum()
            loss.backward();optim.step();losses.append(float(loss))
        print('GRU epoch',ep+1,'loss',np.mean(losses),flush=True)
    model.eval();out=[];val=z(vx)
    with torch.no_grad():
        for i in range(0,len(val),512):out.append(model(torch.from_numpy(val[i:i+512])).numpy())
    return model,np.maximum(0,np.concatenate(out)*target_scale)

def metrics(y,p,ids):
    e=p-y;abs_e=np.abs(e)
    frame=pd.DataFrame({'id':ids,'abs':abs_e,'squared':e**2})
    ev=frame.groupby('id').agg(mae=('abs','mean'),mse=('squared','mean'))
    return {'rows':len(y),'events':len(ev),'row_mae_h':float(abs_e.mean()),
      'row_rmse_h':float(np.sqrt(np.mean(e**2))),
      'row_r2':float(r2_score(y,p)) if len(y)>1 else np.nan,
      'event_equal_mae_h':float(ev.mae.mean()),
      'event_equal_rmse_h':float(np.sqrt(ev.mse.mean())),
      'pearson':float(pearsonr(y,p).statistic) if len(y)>2 and np.std(p)>0 else np.nan,
      'spearman':float(spearmanr(y,p).statistic) if len(y)>2 and np.std(p)>0 else np.nan}

def main():
    if OUT.exists():raise FileExistsError(f'Refusing to overwrite {OUT}')
    OUT.mkdir(parents=True)
    data=build();tr=data['train'];va=data['validation']
    old=pd.read_csv(BASE/'validation_predictions.csv.gz')
    oldtrain=pd.read_csv(BASE/'sample_metadata.csv.gz').query('phase=="train"').reset_index(drop=True)
    assert len(old)==len(va['y']) and np.allclose(old.target_ttf_hours,va['y'])
    assert np.array_equal(old.fault_event_id,va['meta'].fault_event_id)
    assert np.array_equal(old.anchor_time,va['meta'].anchor_time)
    assert len(tr['a'])==len(oldtrain)
    assert np.array_equal(tr['meta'].iloc[tr['a']].anchor_time.to_numpy(),oldtrain.anchor_time.to_numpy())
    assert set(tr['meta'].fault_event_id).isdisjoint(set(va['meta'].fault_event_id))
    with open(MODELS/'simple_full_fault_rul_scaler.pkl','rb') as f:legacy_scaler=pickle.load(f)
    variants={'A_cap64_legacy':tr['a'],'B_all_weighted':np.arange(len(tr['y'])),
              'C_equal32':tr['c']}
    pred=va['meta'].copy();trainrows=[];counts=[]
    for variant,ix in variants.items():
        selected=tr['meta'].iloc[ix].reset_index(drop=True)
        selected_y=tr['y'][ix]
        w=weights(selected)
        selected_counts=selected.fault_event_id.value_counts()
        counts.append({'variant':variant,'train_samples':len(ix),
          'train_events':selected.fault_event_id.nunique(),
          'min_samples_per_event':int(selected_counts.min()),
          'median_samples_per_event':float(selected_counts.median()),
          'max_samples_per_event':int(selected_counts.max()),
          'train_ttf_median_h':float(np.median(selected_y)),
          'weighted_event_mass_min':float(pd.DataFrame({'id':selected.fault_event_id,'w':w}).groupby('id').w.sum().min()),
          'weighted_event_mass_max':float(pd.DataFrame({'id':selected.fault_event_id,'w':w}).groupby('id').w.sum().max())})
        if variant=='A_cap64_legacy':
            d=np.full(len(va['y']),np.median(selected_y),dtype=np.float32)
            tree=old.Tree_HGB_event_balanced.to_numpy(dtype=np.float32)
            gru=old.GRU_event_balanced.to_numpy(dtype=np.float32)
        else:
            d=np.full(len(va['y']),np.median(selected_y),dtype=np.float32)
            tree_model,tree=train_tree(tr['x'][ix],selected_y,w,va['x'])
            with open(MODELS/f'event_balanced_rul_tree_{variant}.pkl','wb') as f:pickle.dump(tree_model,f)
            gru_model,gru=train_gru(tr['x'][ix],selected_y,w,va['x'],legacy_scaler)
            torch.save({'state_dict':gru_model.state_dict(),'variant':variant,
              'legacy_scaler_path':str(MODELS/'simple_full_fault_rul_scaler.pkl'),
              'sensors':SENSORS,'sequence_points':60,'history_minutes':60},
              MODELS/f'event_balanced_rul_gru_{variant}.pt')
        for model,p in [('Dummy',d),('Tree',tree),('GRU',gru)]:
            pred[f'{variant}__{model}']=p
            for subset,mask in [('all',np.ones(len(va['y']),dtype=bool))]+[
                (f'within_{h}h',va['y']<=h) for h in (24,12,6,3)]:
                trainrows.append({'variant':variant,'model':model,'subset':subset,
                  **metrics(va['y'][mask],p[mask],va['meta'].fault_event_id.to_numpy()[mask])})
        print('trained',variant,flush=True)
    pred.to_csv(OUT/'validation_predictions.csv.gz',index=False,compression='gzip')
    pd.DataFrame(trainrows).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(counts).to_csv(OUT/'train_variant_counts.csv',index=False)
    binrows=[];group_rows=[]
    y=va['y'];meta=va['meta']
    for variant in variants:
        for model in ('Dummy','Tree','GRU'):
            p=pred[f'{variant}__{model}'].to_numpy()
            for lo,hi,name in zip(BINS[:-1],BINS[1:],NAMES):
                mask=(y>=lo)&(y<hi)
                yy=y[mask];pp=p[mask]
                binrows.append({'variant':variant,'model':model,'ttf_bin':name,
                  'rows':len(yy),'events':meta.loc[mask,'fault_event_id'].nunique(),
                  'actual_median_h':float(np.median(yy)),
                  'prediction_median_h':float(np.median(pp)),
                  'mae_h':float(np.mean(np.abs(pp-yy))),
                  'mean_bias_h':float(np.mean(pp-yy)),
                  'median_bias_h':float(np.median(pp-yy))})
            temp=meta[['tool_id','fault_type','fault_event_id']].copy()
            temp['abs_error']=np.abs(p-y)
            for group in ('tool_id','fault_type'):
                for key,part in temp.groupby(group):
                    group_rows.append({'variant':variant,'model':model,'group':group,'value':key,
                      'rows':len(part),'events':part.fault_event_id.nunique(),
                      'row_mae_h':float(part.abs_error.mean()),
                      'event_equal_mae_h':float(part.groupby('fault_event_id').abs_error.mean().mean())})
    pd.DataFrame(binrows).to_csv(OUT/'ttf_bin_prediction_bias.csv',index=False)
    pd.DataFrame(group_rows).to_csv(OUT/'tool_fault_type_metrics.csv',index=False)
    config={'split':str(PLAN),'validation_rows':len(y),'validation_events':meta.fault_event_id.nunique(),
      'ttf_unit':'unbounded hours; next any-type fault','sensor_input':SENSORS,
      'history_minutes':60,'prediction_interval_minutes':30,'sequence_points':60,
      'A':'exact stored legacy baseline: random maximum64 samples/event plus 1/N weights; existing tree and GRU predictions reused, no retraining A',
      'B':'all eligible 30-minute Train anchors; sample weight=1/N per event',
      'C':'up to 32 equally spaced chronological Train anchors per event, which equally represent monotone TTF extent; 1/N weights retained so short events have equal total mass',
      'weight_normalization':'1/N multiplied by common sample-count/event-count factor; every event has same total weight',
      'architecture':'legacy HGB parameters and one-layer GRU32 Adam 0.001 5 epochs batch256 smooth L1',
      'gru_scaler':'legacy A Train sensor scaler and target scale fixed for B/C, avoiding a separate preprocessing change',
      'test_access':'none for model fitting/evaluation','random_state':SEED,
      'selection':'C uses only Train event chronology, no Validation TTF selection'}
    (OUT/'experiment_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(pd.DataFrame(counts).to_string(index=False),flush=True)
    print(pd.DataFrame(trainrows).query('subset=="all"').to_string(index=False),flush=True)

if __name__=='__main__':main()
