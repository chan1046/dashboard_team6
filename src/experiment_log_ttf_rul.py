"""A-only GRU target-transform ablation; existing Model 1 and raw-TTF GRU untouched."""
from __future__ import annotations
import json,pickle
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import r2_score
from experiment_event_balanced_rul import ROOT,BASE,MODELS,build,weights,SmallGRU
from train_simple_full_fault_rul import PLAN,SENSORS

OUT=ROOT/'output/experiments/log_ttf_rul'
SEED=42
BINS=[0,3,6,12,24,72,168,np.inf]
NAMES=['0-3h','3-6h','6-12h','12-24h','24-72h','72-168h','168h+']

def train_log_gru(x,y,w,vx,scaler):
    torch.manual_seed(SEED);torch.set_num_threads(4)
    center=scaler['center'];scale=scaler['scale']
    def z(a):
        return np.clip(np.nan_to_num((a-center)/scale,nan=0,posinf=0,neginf=0),-20,20).astype(np.float32)
    train=z(x);validation=z(vx)
    logy=np.log1p(y).astype(np.float32)
    ds=TensorDataset(torch.from_numpy(train),torch.from_numpy(logy),torch.from_numpy(w))
    dl=DataLoader(ds,batch_size=256,shuffle=True)
    model=SmallGRU();optim=torch.optim.Adam(model.parameters(),lr=.001)
    losses=[]
    for epoch in range(5):
        model.train();batch_losses=[]
        for bx,by,bw in dl:
            optim.zero_grad();pred=model(bx)
            loss=(nn.functional.smooth_l1_loss(pred,by,reduction='none')*bw).sum()/bw.sum()
            loss.backward();optim.step();batch_losses.append(float(loss))
        losses.append(float(np.mean(batch_losses)))
        print('log GRU epoch',epoch+1,losses[-1],flush=True)
    model.eval();raw=[]
    with torch.no_grad():
        for i in range(0,len(validation),512):
            raw.append(model(torch.from_numpy(validation[i:i+512])).numpy())
    logpred=np.concatenate(raw).astype(np.float64)
    hours=np.expm1(logpred)
    negative=int(np.sum(hours<0))
    hours=np.maximum(hours,0)
    return model,logpred,hours,negative,losses

def metrics(y,p,ids):
    error=p-y;ae=np.abs(error)
    event=pd.DataFrame({'event_id':ids,'ae':ae,'se':error**2}).groupby('event_id').agg(
      event_mae=('ae','mean'),event_mse=('se','mean'),event_median_ae=('ae','median'))
    return {'rows':len(y),'events':len(event),'row_mae_h':float(ae.mean()),
      'row_rmse_h':float(np.sqrt(np.mean(error**2))),
      'row_r2':float(r2_score(y,p)) if len(y)>1 else np.nan,
      'row_smape_pct':float(np.mean(200*ae/(np.abs(y)+np.abs(p)+1e-6))),
      'event_equal_mae_h':float(event.event_mae.mean()),
      'event_equal_rmse_h':float(np.sqrt(event.event_mse.mean())),
      'event_median_absolute_error_h':float(event.event_median_ae.median()),
      'median_event_mae_h':float(event.event_mae.median()),
      'prediction_median_h':float(np.median(p)),
      'pearson':float(pearsonr(y,p).statistic) if len(y)>2 and np.ptp(p)>1e-8 else np.nan,
      'spearman':float(spearmanr(y,p).statistic) if len(y)>2 and np.ptp(p)>1e-8 else np.nan}

def main():
    if OUT.exists():raise FileExistsError(f'Refusing to overwrite {OUT}')
    OUT.mkdir(parents=True)
    d=build();tr=d['train'];va=d['validation'];ix=tr['a']
    baseline=pd.read_csv(BASE/'validation_predictions.csv.gz')
    oldtrain=pd.read_csv(BASE/'sample_metadata.csv.gz').query('phase=="train"').reset_index(drop=True)
    assert len(ix)==18580 and len(va['y'])==22518
    assert tr['meta'].iloc[ix].fault_event_id.nunique()==553
    assert va['meta'].fault_event_id.nunique()==201
    assert np.array_equal(oldtrain.anchor_time.to_numpy(),tr['meta'].iloc[ix].anchor_time.to_numpy())
    assert np.array_equal(oldtrain.fault_event_id.to_numpy(),tr['meta'].iloc[ix].fault_event_id.to_numpy())
    assert np.array_equal(baseline.anchor_time.to_numpy(),va['meta'].anchor_time.to_numpy())
    assert np.array_equal(baseline.fault_event_id.to_numpy(),va['meta'].fault_event_id.to_numpy())
    assert np.allclose(baseline.target_ttf_hours.to_numpy(),va['y'])
    assert set(tr['meta'].iloc[ix].fault_event_id).isdisjoint(set(va['meta'].fault_event_id))
    with open(MODELS/'simple_full_fault_rul_scaler.pkl','rb') as f:scaler=pickle.load(f)
    selected=tr['meta'].iloc[ix].reset_index(drop=True)
    w=weights(selected)
    mass=pd.DataFrame({'id':selected.fault_event_id,'w':w}).groupby('id').w.sum()
    assert np.allclose(mass,mass.iloc[0],rtol=1e-6)
    model,logpred,hours,negative,losses=train_log_gru(tr['x'][ix],tr['y'][ix],w,
                                                     va['x'],scaler)
    with open(MODELS/'log_ttf_rul_scaler.pkl','wb') as f:pickle.dump({
      'sensor_center':scaler['center'],'sensor_scale':scaler['scale'],
      'sensor_clip':[-20,20],
      'target_transform':'np.log1p(TTF_hours)',
      'target_inverse':'max(0,np.expm1(predicted_log1p_TTF))'},f)
    torch.save({'state_dict':model.state_dict(),'architecture':'SmallGRU(9,32,1)',
      'scaler_path':str(MODELS/'log_ttf_rul_scaler.pkl'),
      'target_transform':'log1p(TTF_hours)',
      'inverse_transform':'maximum(0, expm1(predicted_log1p_TTF))',
      'sensors':SENSORS,'history_minutes':60,'sequence_points':60},
      MODELS/'log_ttf_rul_gru.pt')
    pred=va['meta'].copy()
    pred['Dummy']=baseline['Dummy'].to_numpy(dtype=np.float64)
    pred['Raw_GRU']=baseline['GRU_event_balanced'].to_numpy(dtype=np.float64)
    pred['Log_GRU']=hours
    pred['Log_GRU_log_target_prediction']=logpred
    pred.to_csv(OUT/'validation_predictions.csv.gz',index=False,compression='gzip')
    y=va['y'].astype(np.float64);ids=va['meta'].fault_event_id.to_numpy()
    rows=[];near=[];bybin=[];bygroup=[]
    for name in ('Dummy','Raw_GRU','Log_GRU'):
        p=pred[name].to_numpy(dtype=np.float64)
        rows.append({'model':name,'subset':'all',**metrics(y,p,ids)})
        for h in (24,12,6,3):
            mask=y<=h
            near.append({'model':name,'subset':f'TTF<={h}h',**metrics(y[mask],p[mask],ids[mask])})
        for lo,hi,label in zip(BINS[:-1],BINS[1:],NAMES):
            mask=(y>=lo)&(y<hi);yy=y[mask];pp=p[mask]
            bybin.append({'model':name,'ttf_bin':label,'rows':len(yy),
              'events':len(np.unique(ids[mask])),
              'actual_median_h':float(np.median(yy)),
              'prediction_median_h':float(np.median(pp)),
              'prediction_p25_h':float(np.percentile(pp,25)),
              'prediction_p75_h':float(np.percentile(pp,75)),
              'mae_h':float(np.mean(np.abs(pp-yy))),
              'mean_bias_h':float(np.mean(pp-yy)),
              'median_bias_h':float(np.median(pp-yy))})
        temp=va['meta'][['tool_id','fault_type','fault_event_id']].copy()
        temp['absolute_error_h']=np.abs(p-y)
        for group in ('tool_id','fault_type'):
            for key,part in temp.groupby(group):
                ev=part.groupby('fault_event_id').absolute_error_h.mean()
                bygroup.append({'model':name,'group':group,'value':key,
                  'rows':len(part),'events':len(ev),
                  'row_mae_h':float(part.absolute_error_h.mean()),
                  'event_equal_mae_h':float(ev.mean())})
    allmetrics=pd.DataFrame(rows);subsets=pd.DataFrame(near)
    for table in (allmetrics,subsets):
        dummy=table.loc[table.model=='Dummy'].iloc[0] if table is allmetrics else None
        if table is subsets:
            lookup=table[table.model=='Dummy'].set_index('subset').row_mae_h
            table['dummy_row_mae_improvement_pct']=100*(1-table.row_mae_h/table.subset.map(lookup))
            lookup2=table[table.model=='Dummy'].set_index('subset').event_equal_mae_h
            table['dummy_event_mae_improvement_pct']=100*(1-table.event_equal_mae_h/table.subset.map(lookup2))
    allmetrics.to_csv(OUT/'overall_metrics.csv',index=False)
    subsets.to_csv(OUT/'near_fault_metrics.csv',index=False)
    pd.DataFrame(bybin).to_csv(OUT/'ttf_bin_calibration.csv',index=False)
    pd.DataFrame(bygroup).to_csv(OUT/'tool_fault_type_metrics.csv',index=False)
    config={'train_samples':len(ix),'train_events':selected.fault_event_id.nunique(),
      'validation_samples':len(y),'validation_events':len(np.unique(ids)),
      'split_plan':str(PLAN),'history_minutes':60,'prediction_interval_minutes':30,
      'raw_sensors':SENSORS,'sequence_points':60,'random_state':SEED,
      'sample_selection':'exact A legacy indices, max64 randomly selected per event using seed42',
      'event_weight':'1/N within each event, multiplied by common factor to mean1; all event masses equal',
      'architecture':'one-layer GRU32, linear head, Adam lr0.001, batch256, 5 epochs, weighted smooth L1',
      'sensor_scaler':'unchanged Train-only A median/IQR scaler; saved duplicate in output/models/log_ttf_rul_scaler.pkl',
      'baseline_target':'TTF_hours / Train median, existing stored GRU model and probabilities used unchanged',
      'log_target':'log1p(TTF_hours), no further target scaling',
      'inverse_transform':'expm1(predicted_log1p_TTF); negative inverse results clipped to 0 hours',
      'negative_inverse_prediction_count':negative,
      'training_loss_by_epoch':losses,'test_access':'none',
      'feature_change':False,'split_change':False,'sampling_change':False}
    (OUT/'experiment_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(json.dumps({'negative_inverse_prediction_count':negative},indent=2),flush=True)
    print(allmetrics.to_string(index=False),flush=True)
    print(pd.DataFrame(bybin).query('model != "Dummy"').to_string(index=False),flush=True)

if __name__=='__main__':main()
