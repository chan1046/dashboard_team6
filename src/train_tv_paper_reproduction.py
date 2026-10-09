"""TV et al. T2V and clipped multi-head LSTM on frozen Rule B windows."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import r2_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/experiments/tv_paper_reproduction'
MODEL=ROOT/'output/models'
OLD=ROOT/'output/experiments/rul_rule_b_baseline'
PLAN=ROOT/'output/experiments/forward_validation_ab_24h/split_plan.json'
RAW=ROOT/'data/processed'
SENSORS=['IONGAUGEPRESSURE','ETCHBEAMVOLTAGE','ETCHBEAMCURRENT',
 'ETCHSUPPRESSORVOLTAGE','ETCHSUPPRESSORCURRENT','FLOWCOOLFLOWRATE',
 'FLOWCOOLPRESSURE','ETCHGASCHANNEL1READBACK','ETCHPBNGASREADBACK']
TYPES=['pressure_low','pressure_high','leak']
SEED=42; RU=500; EPOCHS=5; HIDDEN=50; DISTANT_MULTIPLIER=10

class LSTM(nn.Module):
    def __init__(self,features):
        super().__init__()
        self.rnn=nn.LSTM(features,HIDDEN,num_layers=2,dropout=.4,batch_first=True)
        self.head=nn.Linear(HIDDEN,3)
    def forward(self,x):return self.head(self.rnn(x)[0][:,-1,:])

def get_selection():
    f=pd.read_csv(OLD/'B_operating_axis_sequence_metadata.csv.gz')
    target=f[[f'ttf_h_{t}' for t in TYPES]].to_numpy(np.float64)*3600
    near=np.nanmin(target,axis=1)<RU
    train=f.phase.eq('train').to_numpy()
    tr_near=np.flatnonzero(train&near)
    tr_far=np.flatnonzero(train&~near)
    if len(tr_near)!=6:raise AssertionError('Near-failure population changed')
    rng=np.random.default_rng(SEED)
    far_sample=rng.choice(tr_far,size=min(len(tr_far),DISTANT_MULTIPLIER*len(tr_near)),replace=False)
    selected=np.sort(np.r_[tr_near,far_sample,np.flatnonzero(~train)])
    chosen=f.iloc[selected].copy()
    chosen['source_row']=selected
    chosen['near_failure_any_type']=near[selected]
    summary={'all_rule_b_usable_windows':len(f),'train_all':int(train.sum()),
      'train_near_any_type':len(tr_near),'train_distant_all':len(tr_far),
      'train_distant_sampled':len(far_sample),'train_sampled_total':len(tr_near)+len(far_sample),
      'validation_all_unsampled':int((~train).sum()),
      'validation_near_any_type':int((~train&near).sum()),
      'sampling':'all near (<500 seconds in any observed head) plus 10x as many randomly selected distant Train windows; paper does not give exact distant ratio',
      'seed':SEED,
      'train_near_per_type':{t:int(np.sum(train&(target[:,j]<RU))) for j,t in enumerate(TYPES)},
      'validation_near_per_type':{t:int(np.sum(~train&(target[:,j]<RU))) for j,t in enumerate(TYPES)}}
    return chosen,summary

def read_selected_windows(plan,selection):
    result={}; shape=[]
    for item in plan['tools']:
        tool=item['tool_id']; sub=selection.loc[selection.tool_id.eq(tool)]
        if sub.empty:continue
        table=pq.read_table(RAW/f'{tool}_sensor_ttf.parquet',
          columns=['time','FIXTURESHUTTERPOSITION',*SENSORS],
          filters=[('time','<',item['validation_end_exclusive_second'])])
        df=table.to_pandas()
        if (df.time>=item['validation_end_exclusive_second']).any():
            raise AssertionError('Test row returned')
        op=df.FIXTURESHUTTERPOSITION.to_numpy()==1
        t=df.time.to_numpy(np.int64)[op]
        s=df.loc[op,SENSORS].to_numpy(np.float32)
        del df,table
        # Recreate the exact audit Rule B segment starts, then match both
        # start/end and phase. This avoids ambiguous timestamp-only matching.
        phase=np.select([t<item['train_end_exclusive_second'],
                         t>=item['validation_start_second']],[0,2],default=1)
        ft=pd.read_csv(RAW/'fault_events.csv',usecols=['tool_id','time'])
        ft=np.sort(ft.loc[(ft.tool_id==tool)&
               (ft.time<item['validation_end_exclusive_second']),'time'].unique())
        prev=np.r_[t[0]-1,t[:-1]]
        crossed=np.searchsorted(ft,t,side='right')>np.searchsorted(ft,prev,side='right')
        starts=(phase!=np.r_[-1,phase[:-1]])|crossed|(t-prev<0)
        starts[0]=True
        bounds=np.r_[np.flatnonzero(starts),len(t)]
        lookup={}
        for row in sub.itertuples():
            lookup.setdefault((int(row.start_time),int(row.time),row.phase),[]).append(row.source_row)
        for a,b in zip(bounds[:-1],bounds[1:]):
            if phase[a]==1 or b-a<2000:continue
            for left in range(a,b-2000+1,1500):
                right=left+1999
                key=(int(t[left]),int(t[right]),'train' if phase[a]==0 else 'validation')
                ids=lookup.get(key)
                if not ids:continue
                idx=ids.pop()
                window=s[left:right+1]
                if window.shape!=(2000,9) or not np.isfinite(window).all():
                    raise AssertionError('Selected raw sequence differs from prior usable sequence')
                down=window[::2]
                t2v=down.reshape(100,10,9).reshape(100,90)
                old=window[::20]
                if down.shape!=(1000,9) or t2v.shape!=(100,90) or old.shape!=(100,9):
                    raise AssertionError('T2V shape incorrect')
                result[idx]=(old.copy(),t2v.copy())
        if any(lookup.values()):raise AssertionError(f'Unmatched audited Rule B window for {tool}')
        print(tool,'selected sequences',len(sub),flush=True)
        del t,s
    if len(result)!=len(selection):raise AssertionError('Window selection incomplete')
    order=selection.source_row.to_numpy()
    old=np.stack([result[i][0] for i in order])
    tv=np.stack([result[i][1] for i in order])
    print('shape raw', (2000,9),'d1', (1000,9),'d2 T2V',tv.shape,'old',old.shape,flush=True)
    return old,tv

def train_model(x,targets,mask,name):
    torch.manual_seed(SEED)
    model=LSTM(x.shape[2]); opt=torch.optim.Adam(model.parameters(),lr=.005)
    ds=TensorDataset(torch.from_numpy(x),torch.from_numpy(targets),torch.from_numpy(mask))
    loader=DataLoader(ds,batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(SEED))
    for epoch in range(EPOCHS):
        model.train(); losses=[]
        for xb,yb,mb in loader:
            opt.zero_grad()
            sq=(model(xb)-yb).square()*mb
            loss=(sq.sum(0)/mb.sum(0).clamp(min=1)).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();opt.step();losses.append(float(loss.detach()))
        print(name,'epoch',epoch+1,'loss',np.mean(losses),flush=True)
    return model

def predict(model,x):
    model.eval(); out=[]
    with torch.no_grad():
        for a in range(0,len(x),256):out.append(model(torch.from_numpy(x[a:a+256])).numpy())
    return np.vstack(out)

def metrics(y,p):
    e=np.abs(y-p); d=np.abs(y)+np.abs(p)
    return {'mae':float(e.mean()),'rmse':float(np.sqrt(np.mean((y-p)**2))),
       'r2':float(r2_score(y,p)) if len(y)>1 else np.nan,
       'smape_pct':float((2*e/np.maximum(d,1e-9)).mean()*100)}

def main():
    if not (OUT/'unit_audit.json').exists():raise FileNotFoundError('Audit RUL units first')
    existing=[MODEL/f'tv_{n}_lstm.pt' for n in ['subsample_unclipped','t2v_unclipped','t2v_clipped']]
    if any(p.exists() for p in existing) or (OUT/'metrics.csv').exists():
        raise FileExistsError('Refusing to overwrite prior TV result or model')
    torch.set_num_threads(4)
    selection,sampling=get_selection()
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    old,tv=read_selected_windows(plan,selection)
    train=selection.phase.eq('train').to_numpy(); valid=~train
    if train.sum()!=sampling['train_sampled_total'] or valid.sum()!=sampling['validation_all_unsampled']:
        raise AssertionError('Split mismatch')
    old_config=json.loads((OLD/'experiment_config.json').read_text(encoding='utf-8'))['B_raw9']['scaler']
    center=np.array(old_config['sensor_center'],dtype=np.float32)
    scale=np.array(old_config['sensor_iqr'],dtype=np.float32)
    old=(old-center)/scale
    tv=(tv-center[None,None,:].repeat(10,axis=2))/scale[None,None,:].repeat(10,axis=2)
    y=selection[[f'ttf_h_{t}' for t in TYPES]].to_numpy(np.float32)*3600
    mask=np.isfinite(y).astype(np.float32)
    clipped=np.minimum(np.nan_to_num(y,nan=0),RU).astype(np.float32)
    normalized=clipped/RU
    unbounded=np.nan_to_num(y,nan=0).astype(np.float32)/RU
    models={}
    for name,x,tar in [('subsample_unclipped',old,unbounded),
                        ('t2v_unclipped',tv,unbounded),('t2v_clipped',tv,normalized)]:
        model=train_model(x[train].astype(np.float32),tar[train],mask[train],name)
        models[name]=predict(model,x[valid].astype(np.float32))*RU
        MODEL.mkdir(parents=True,exist_ok=True)
        torch.save({'state_dict':model.state_dict(),'input_features':x.shape[2],
           'sensors':SENSORS,'scaler_center':center,'scaler_iqr':scale,
           'name':name,'r_u_seconds':RU,'target_normalization_divisor_seconds':RU,
           'target_clipped':name.endswith('_clipped'),'hidden_size':HIDDEN,
           'layers':2,'dropout':.4,'seed':SEED},MODEL/f'tv_{name}_lstm.pt')
    median_raw=np.array([np.nanmedian(y[train,j]) for j in range(3)],np.float32)
    median_clip=np.array([np.nanmedian(np.minimum(y[train,j],RU)) for j in range(3)],np.float32)
    models['dummy_unclipped']=np.broadcast_to(median_raw,(valid.sum(),3)).copy()
    models['dummy_clipped']=np.broadcast_to(median_clip,(valid.sum(),3)).copy()
    # Reuse the prior Rule B GRU predictions only as contextual baseline.
    oldp=pd.read_csv(OLD/'validation_predictions.csv.gz')
    metadata=selection.loc[valid].reset_index(drop=True)
    prediction=[]; metric=[]; event=[]; tool=[]; fold=[]
    for name,p in models.items():
        isclip=name.endswith('_clipped')
        for j,typ in enumerate(TYPES):
            known=mask[valid,j].astype(bool)
            part=metadata.loc[known,['tool_id','start_time','time',f'event_id_{typ}']].copy()
            part.columns=['tool_id','start_time','time','event_id']
            part['condition']=name;part['fault_type']=typ
            true=y[valid,j][known]
            target=np.minimum(true,RU) if isclip else true
            estimate=np.clip(p[known,j],0,RU) if isclip else np.maximum(p[known,j],0)
            part['actual_seconds']=true
            part['evaluation_target_seconds']=target
            part['predicted_seconds']=estimate
            part['target_clipped']=isclip
            part['abs_error_seconds']=np.abs(target-estimate)
            prediction.append(part)
            pe=part.groupby('event_id').abs_error_seconds.mean()
            near=true<RU
            row={'condition':name,'fault_type':typ,'samples':len(part),
               'events':part.event_id.nunique(),'event_equal_mae_seconds':float(pe.mean()),
               'near_samples':int(near.sum()),'near_mae_seconds':float(np.abs(target[near]-estimate[near]).mean()) if near.any() else np.nan,
               **metrics(target,estimate),
               'normalized_mae':float(np.mean(np.abs(target-estimate))/RU),
               'normalized_rmse':float(np.sqrt(np.mean((target-estimate)**2))/RU)}
            metric.append(row)
            event.extend({'condition':name,'fault_type':typ,'event_id':eid,'event_mae_seconds':v} for eid,v in pe.items())
            for tool_id,g in part.groupby('tool_id'):
                tool.append({'condition':name,'fault_type':typ,'tool_id':tool_id,'samples':len(g),'events':g.event_id.nunique(),
                    'mae_seconds':g.abs_error_seconds.mean(),'event_equal_mae_seconds':g.groupby('event_id').abs_error_seconds.mean().mean()})
            et=part.event_id.str.rsplit('|',n=1).str[-1].astype(int)
            bounds=np.quantile(et.unique(),[1/3,2/3],method='nearest')
            part['fold']=1+np.searchsorted(bounds,et.to_numpy(),side='right')
            for k,g in part.groupby('fold'):
                fold.append({'condition':name,'fault_type':typ,'fold':int(k),'samples':len(g),'events':g.event_id.nunique(),
                   'mae_seconds':g.abs_error_seconds.mean(),'event_equal_mae_seconds':g.groupby('event_id').abs_error_seconds.mean().mean()})
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(metric).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(event).to_csv(OUT/'event_metrics.csv',index=False)
    pd.DataFrame(tool).to_csv(OUT/'tool_metrics.csv',index=False)
    pd.DataFrame(fold).to_csv(OUT/'fold_metrics.csv',index=False)
    pd.concat(prediction,ignore_index=True).to_csv(OUT/'validation_predictions.csv.gz',index=False,compression='gzip')
    selection[['tool_id','phase','start_time','time','near_failure_any_type',
               *[f'event_id_{t}' for t in TYPES],*[f'ttf_h_{t}' for t in TYPES]]].to_csv(
               OUT/'sampled_window_metadata.csv.gz',index=False,compression='gzip')
    (OUT/'sampling_config.json').write_text(json.dumps(sampling,indent=2),encoding='utf-8')
    config={'paper_source':'https://papers.phmsociety.org/index.php/phmconf/article/download/589/phmc_18_589',
      'window':'Rule B audited 2000 observations, 500 overlap, Tool/fault/split boundaries',
      'shape':{'raw':[2000,9],'downsample_d1_2':[1000,9],'t2v_d2_10':[100,90],
               'comparison_subsample':[100,9]},
      'clipping':{'r_u_seconds':RU,'normalized_target':'min(raw_RUL_seconds,500)/500'},
      'unclipped_ablation':'raw_RUL_seconds/500 (no clipping); same sampling and LSTM structure',
      'sampling':sampling,'split':str(PLAN),'test':'reserved, unread',
      'scaling':'Train-only sensor median/IQR reused from Rule B baseline',
      'mask':'only observed next same-type fault targets within frozen split; three-head masked MSE',
      'architecture':{'layers':2,'hidden':HIDDEN,'dropout':.4,'adam_lr':.005,'batch':32,
                      'epochs':EPOCHS,'hidden_selection':'smallest paper candidate prespecified; no Validation tuning'},
      'seed':SEED,'metadata':'elapsed and max gap retained only; not model features',
      'model1':'unchanged'}
    (OUT/'experiment_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(pd.DataFrame(metric).to_string(index=False),flush=True)

if __name__=='__main__':main()
