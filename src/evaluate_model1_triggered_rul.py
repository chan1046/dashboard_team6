"""Frozen Model 1 manager notifications -> frozen raw/log GRU inference, no fitting."""
from __future__ import annotations
import hashlib,json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy.stats import pearsonr,spearmanr
from experiment_event_balanced_rul import ROOT,MODELS,SmallGRU
from train_simple_full_fault_rul import RAW,PLAN,SENSORS

SOURCE=ROOT/'output/experiments/forward_validation_ab_24h'
COOLDOWN_SOURCE=ROOT/'output/experiments/cooldown_forward_validation_24h'
OUT=ROOT/'output/experiments/model1_triggered_rul'
SCORES=SOURCE/'validation_probabilities_a.npz'
THRESHOLD=.34
RAW_GAP_SECONDS=600
COOLDOWN_SECONDS=1800
HISTORY_SECONDS=3600
HISTORY_MIN_OBS=60
HISTORY_MIN_SPAN=2880
BINS=[0,3,6,12,24,72,np.inf]
NAMES=['0-3h','3-6h','6-12h','12-24h','24-72h','72h+']

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def metric(y,p,ids):
    ae=np.abs(p-y);e=p-y
    frame=pd.DataFrame({'id':ids,'ae':ae,'se':e**2})
    ev=frame.groupby('id').agg(mae=('ae','mean'),mse=('se','mean'))
    return {'alerts':len(y),'fault_events':len(ev),'mae_h':float(ae.mean()),
      'rmse_h':float(np.sqrt(np.mean(e**2))),'median_ae_h':float(np.median(ae)),
      'mean_bias_h':float(np.mean(e)),
      'event_equal_mae_h':float(ev.mae.mean()),
      'event_equal_rmse_h':float(np.sqrt(ev.mse.mean())),
      'prediction_median_h':float(np.median(p)),
      'pearson':float(pearsonr(y,p).statistic) if len(y)>2 and np.ptp(p)>1e-8 else np.nan,
      'spearman':float(spearmanr(y,p).statistic) if len(y)>2 and np.ptp(p)>1e-8 else np.nan}

def select_manager_notifications(times,prob,positions):
    hits=positions[prob[positions]>=THRESHOLD]
    raw=hits[np.r_[True,np.diff(times[hits])>RAW_GAP_SECONDS]] if len(hits) else hits
    notices=[];last=None
    for i in raw:
        if last is None or int(times[i])-last>=COOLDOWN_SECONDS:
            notices.append(i);last=int(times[i])
    return hits,raw,np.asarray(notices,dtype=np.int64)

def load_models():
    with open(MODELS/'simple_full_fault_rul_scaler.pkl','rb') as f:scaler=pickle.load(f)
    raw=SmallGRU();raw.load_state_dict(torch.load(MODELS/'simple_full_fault_rul_gru.pt',
                                                   map_location='cpu',weights_only=False)['state_dict'])
    log=SmallGRU();log.load_state_dict(torch.load(MODELS/'log_ttf_rul_gru.pt',
                                                   map_location='cpu',weights_only=False)['state_dict'])
    raw.eval();log.eval();torch.set_num_threads(4)
    return raw,log,scaler

def infer(sequences,raw,log,scaler):
    x=np.stack(sequences).astype(np.float32)
    x=np.clip(np.nan_to_num((x-scaler['center'])/scaler['scale'],
                            nan=0,posinf=0,neginf=0),-20,20).astype(np.float32)
    raws=[];logs=[]
    with torch.no_grad():
        for i in range(0,len(x),512):
            inp=torch.from_numpy(x[i:i+512])
            raws.append(raw(inp).numpy())
            logs.append(log(inp).numpy())
    rawpred=np.maximum(0,np.concatenate(raws)*float(scaler['target_scale']))
    loghours=np.expm1(np.concatenate(logs).astype(np.float64))
    negative=int(np.count_nonzero(loghours<0))
    return rawpred,np.maximum(0,loghours),negative

def main():
    if OUT.exists():raise FileExistsError(f'Refusing to overwrite {OUT}')
    OUT.mkdir(parents=True)
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    saved=json.loads((SOURCE/'training_a.json').read_text(encoding='utf-8'))
    policy=json.loads((COOLDOWN_SOURCE/'report.json').read_text(encoding='utf-8'))
    if saved['validation_score_sha256']!=digest(SCORES):raise AssertionError('Model 1 scores changed')
    if policy['validation_score_sha256']!=digest(SCORES):raise AssertionError('Cooldown score source changed')
    if saved['model_sha256']!=digest(MODELS/'xgb_24h_forward_a.joblib'):
        raise AssertionError('Frozen Model 1 changed')
    assert 'threshold .34' in policy['policy'] and '30m' in policy['policy']
    with np.load(SCORES) as z:
        time=z['time'].astype(np.int64);tool_code=z['tool_code'];prob=z['probability']
    faults=pd.read_csv(RAW/'fault_events.csv').drop_duplicates(['tool_id','time','fault_type'])
    raw_model,log_model,scaler=load_models()
    notices=[];sequences=[];seq_notice_index=[];counts=[];all_events=[]
    for code,item in enumerate(plan['tools']):
        tool=item['tool_id'];vstart=int(item['validation_start_second'])
        vend=int(item['validation_end_exclusive_second'])
        pos=np.flatnonzero(tool_code==code)
        assert np.all((time[pos]>=vstart)&(time[pos]<vend))
        hits,raw,notify=select_manager_notifications(time,prob,pos)
        counts.append({'tool_id':tool,'threshold_hits':len(hits),
                       'raw_alarm_episode_starts':len(raw),'manager_notifications':len(notify)})
        log=faults[faults.tool_id==tool].sort_values('time',kind='stable')
        valfault=log[(log.time>=vstart)&(log.time<vend)].copy()
        all_events.extend(valfault[['tool_id','fault_event_id','fault_type','time']].to_dict('records'))
        ft=log.time.to_numpy(dtype=np.int64)
        val_ft=valfault.time.to_numpy(dtype=np.int64)
        val_ids=valfault.fault_event_id.to_numpy(dtype=np.int64)
        val_types=valfault.fault_type.to_numpy()
        frame=pq.read_table(RAW/f'{tool}_sensor_ttf.parquet',columns=['time']+SENSORS,
                            filters=[('time','<',vend)]).to_pandas()
        stime=frame.time.to_numpy(dtype=np.int64)
        sx=frame[SENSORS].to_numpy(dtype=np.float32)
        for index in notify:
            t=int(time[index]);j=int(np.searchsorted(val_ft,t,side='right'))
            event_available=j<len(val_ft)
            f=int(val_ft[j]) if event_available else None
            event_id=int(val_ids[j]) if event_available else None
            typ=str(val_types[j]) if event_available else None
            prev_idx=int(np.searchsorted(ft,t,side='right'))-1
            prev=int(ft[prev_idx]) if prev_idx>=0 else None
            end_ix=int(np.searchsorted(stime,t,side='right'))-1
            valid=False;span=None;age=None;nobs=0;maxgap=None
            if end_ix>=0:
                age=t-int(stime[end_ix])
                lo=max(int(np.searchsorted(stime,t-HISTORY_SECONDS,side='left')),
                       int(np.searchsorted(stime,vstart,side='left')),
                       0 if prev is None else int(np.searchsorted(stime,prev,side='right')))
                nobs=end_ix-lo+1
                if nobs>=HISTORY_MIN_OBS:
                    span=int(stime[end_ix]-stime[lo])
                    valid=span>=HISTORY_MIN_SPAN
                    if valid:
                        idx=np.linspace(lo,end_ix,60,dtype=np.int64)
                        seq=sx[idx].copy()
                        gaps=np.diff(stime[lo:end_ix+1])
                        maxgap=int(gaps.max()) if len(gaps) else 0
                        assert stime[end_ix]<=t and stime[lo]>=t-HISTORY_SECONDS
                        assert prev is None or stime[lo]>prev
                        assert stime[lo]>=vstart
            status='A_evaluable' if event_available and valid else (
                   'B_history_insufficient' if event_available else 'C_no_validation_next_fault')
            row={'tool_id':tool,'model1_time':t,'model1_probability':float(prob[index]),
              'next_fault_time':f,'next_fault_event_id':event_id,'fault_type':typ,
              'actual_ttf_h':(f-t)/3600 if f is not None else np.nan,
              'within_24h_before_fault':bool(f is not None and 0<f-t<=86400),
              'status':status,'last_sensor_time':int(stime[end_ix]) if end_ix>=0 else None,
              'sensor_age_seconds':age,'history_observations':max(0,nobs),
              'history_observed_span_seconds':span,'max_internal_gap_seconds':maxgap,
              'raw_gru_rul_h':np.nan,'log_gru_rul_h':np.nan}
            notices.append(row)
            if status=='A_evaluable':
                seq_notice_index.append(len(notices)-1);sequences.append(seq)
        print('notifications',tool,len(notify),flush=True)
    if sequences:
        rawpred,logpred,negative=infer(sequences,raw_model,log_model,scaler)
        for k,r,l in zip(seq_notice_index,rawpred,logpred):
            notices[k]['raw_gru_rul_h']=float(r);notices[k]['log_gru_rul_h']=float(l)
    else:negative=0
    table=pd.DataFrame(notices)
    table.to_csv(OUT/'manager_notifications.csv.gz',index=False,compression='gzip')
    pd.DataFrame(counts).to_csv(OUT/'model1_alert_counts_by_tool.csv',index=False)
    evaluable=table[table.status=='A_evaluable'].copy()
    assert evaluable.raw_gru_rul_h.notna().all() and evaluable.log_gru_rul_h.notna().all()
    def make_metrics(frame,scope):
        y=frame.actual_ttf_h.to_numpy();ids=frame.next_fault_event_id.to_numpy()
        return [{'scope':scope,'model':name,**metric(y,frame[col].to_numpy(),ids)}
            for name,col in [('Raw_GRU','raw_gru_rul_h'),('Log_GRU','log_gru_rul_h')]] if len(frame) else []
    overall=make_metrics(evaluable,'all_evaluable_alerts')
    pd.DataFrame(overall).to_csv(OUT/'overall_metrics.csv',index=False)
    ttfbins=[];binmetrics=[]
    evaluable['ttf_bin']=pd.cut(evaluable.actual_ttf_h,BINS,labels=NAMES,right=False)
    for label in NAMES:
        part=evaluable[evaluable.ttf_bin==label]
        if not len(part):continue
        ttfbins.append({'ttf_bin':label,'alerts':len(part),
          'fault_events':part.next_fault_event_id.nunique(),
          'actual_median_h':part.actual_ttf_h.median(),
          'actual_p25_h':part.actual_ttf_h.quantile(.25),
          'actual_p75_h':part.actual_ttf_h.quantile(.75)})
        for name,col in [('Raw_GRU','raw_gru_rul_h'),('Log_GRU','log_gru_rul_h')]:
            err=part[col]-part.actual_ttf_h
            binmetrics.append({'ttf_bin':label,'model':name,'alerts':len(part),
              'fault_events':part.next_fault_event_id.nunique(),
              'actual_median_h':part.actual_ttf_h.median(),
              'prediction_median_h':part[col].median(),
              'mae_h':np.abs(err).mean(),'mean_bias_h':err.mean()})
    pd.DataFrame(ttfbins).to_csv(OUT/'actual_ttf_at_alert_bins.csv',index=False)
    pd.DataFrame(binmetrics).to_csv(OUT/'ttf_bin_model_comparison.csv',index=False)
    first=evaluable.sort_values(['tool_id','next_fault_event_id','model1_time']).drop_duplicates('next_fault_event_id',keep='first')
    last=evaluable.sort_values(['tool_id','next_fault_event_id','model1_time']).drop_duplicates('next_fault_event_id',keep='last')
    first.to_csv(OUT/'first_evaluable_alert_by_event.csv',index=False)
    last.to_csv(OUT/'last_evaluable_alert_by_event.csv',index=False)
    first_metrics=make_metrics(first,'first_evaluable_alert')
    last_metrics=make_metrics(last,'last_evaluable_alert')
    pd.DataFrame(first_metrics+last_metrics).to_csv(OUT/'first_last_metrics.csv',index=False)
    # True first manager notification is evaluated separately: it can lack RUL history.
    with_fault=table[table.next_fault_event_id.notna()].copy()
    first_actual=with_fault.sort_values(['tool_id','next_fault_event_id','model1_time']).drop_duplicates('next_fault_event_id',keep='first')
    last_actual=with_fault.sort_values(['tool_id','next_fault_event_id','model1_time']).drop_duplicates('next_fault_event_id',keep='last')
    first_actual.to_csv(OUT/'first_actual_notification_by_event.csv',index=False)
    last_actual.to_csv(OUT/'last_actual_notification_by_event.csv',index=False)
    pair=first.merge(last,on='next_fault_event_id',suffixes=('_first','_last'))
    pair=pair[pair.model1_time_last>pair.model1_time_first].copy()
    pair['elapsed_h']=(pair.model1_time_last-pair.model1_time_first)/3600
    for name,col in [('raw','raw_gru_rul_h'),('log','log_gru_rul_h')]:
        pair[f'{name}_prediction_drop_h']=pair[f'{col}_first']-pair[f'{col}_last']
        pair[f'{name}_first_ae_h']=np.abs(pair[f'{col}_first']-pair.actual_ttf_h_first)
        pair[f'{name}_last_ae_h']=np.abs(pair[f'{col}_last']-pair.actual_ttf_h_last)
    pair.to_csv(OUT/'first_last_paired_events.csv',index=False)
    mono=[]
    for event,part in evaluable.sort_values('model1_time').groupby('next_fault_event_id'):
        if len(part)<2:continue
        dt=np.diff(part.model1_time.to_numpy())/3600
        for name,col in [('Raw_GRU','raw_gru_rul_h'),('Log_GRU','log_gru_rul_h')]:
            change=np.diff(part[col].to_numpy())
            for i in range(len(change)):
                mono.append({'fault_event_id':event,'model':name,'elapsed_h':dt[i],
                  'prediction_change_h':change[i],
                  'decreased':change[i]<0,'increased':change[i]>0,
                  'clock_consistency_absolute_error_h':abs(change[i]+dt[i]),
                  'predicted_drop_over_elapsed':-change[i]/dt[i] if dt[i]>0 else np.nan})
    monotable=pd.DataFrame(mono);monotable.to_csv(OUT/'consecutive_alert_changes.csv',index=False)
    mons=monotable.groupby('model').agg(pairs=('decreased','size'),events=('fault_event_id','nunique'),
      decreased_pct=('decreased',lambda x:100*x.mean()),
      increased_pct=('increased',lambda x:100*x.mean()),
      median_predicted_drop_over_elapsed=('predicted_drop_over_elapsed','median'),
      median_clock_consistency_error_h=('clock_consistency_absolute_error_h','median')).reset_index()
    mons.to_csv(OUT/'monotonicity_summary.csv',index=False)
    urgent=[]
    for actual_cut in (24,72):
        part=evaluable[evaluable.actual_ttf_h>actual_cut]
        for name,col in [('Raw_GRU','raw_gru_rul_h'),('Log_GRU','log_gru_rul_h')]:
            for predicted_cut in (3,6,12,24):
                urgent.append({'actual_ttf_gt_h':actual_cut,'model':name,
                  'predicted_rul_le_h':predicted_cut,'alerts':len(part),
                  'fault_events':part.next_fault_event_id.nunique(),
                  'overurgent_alerts':int((part[col]<=predicted_cut).sum()),
                  'overurgent_pct':100*(part[col]<=predicted_cut).mean(),
                  'overurgent_events':part.loc[part[col]<=predicted_cut,'next_fault_event_id'].nunique()})
    pd.DataFrame(urgent).to_csv(OUT/'overurgent_rates.csv',index=False)
    near=[]
    for h in (24,12,6,3):near+=make_metrics(evaluable[evaluable.actual_ttf_h<=h],f'actual_ttf_le_{h}h')
    pd.DataFrame(near).to_csv(OUT/'near_fault_alert_metrics.csv',index=False)
    bygroup=[]
    for group in ('tool_id','fault_type'):
        for key,part in evaluable.groupby(group):
            for name,col in [('Raw_GRU','raw_gru_rul_h'),('Log_GRU','log_gru_rul_h')]:
                m=metric(part.actual_ttf_h.to_numpy(),part[col].to_numpy(),part.next_fault_event_id.to_numpy())
                bygroup.append({'group':group,'value':key,'model':name,
                  'small_event_count_caution':m['fault_events']<10,**m})
    pd.DataFrame(bygroup).to_csv(OUT/'tool_fault_type_metrics.csv',index=False)
    population=pd.DataFrame(all_events).rename(columns={'time':'fault_time'})
    popids=set(population.fault_event_id)
    pre=table[table.within_24h_before_fault]
    detected=set(pre.next_fault_event_id.dropna().astype(int))
    usable=set(pre[pre.status=='A_evaluable'].next_fault_event_id.dropna().astype(int))
    funnel=[{'stage':'validation_fault_events','events':len(popids),'fraction_of_all':1.0},
      {'stage':'model1_manager_notification_within24h','events':len(detected),'fraction_of_all':len(detected)/len(popids)},
      {'stage':'prealert_with_rul_history','events':len(usable),'fraction_of_all':len(usable)/len(popids)},
      {'stage':'paired_rul_prediction_evaluable','events':len(usable),'fraction_of_all':len(usable)/len(popids)}]
    pd.DataFrame(funnel).to_csv(OUT/'detection_to_rul_funnel.csv',index=False)
    summary={'model1_threshold':THRESHOLD,'raw_alarm_gap_seconds':RAW_GAP_SECONDS,
      'manager_cooldown_seconds':COOLDOWN_SECONDS,'threshold_hits':int(sum(z['threshold_hits'] for z in counts)),
      'raw_alarm_episode_starts':int(sum(z['raw_alarm_episode_starts'] for z in counts)),
      'manager_notifications':len(table),'status_counts':{k:int(v) for k,v in table.status.value_counts().items()},
      'all_validation_faults_within_validation_end':len(popids),
      'original_model1_report_faults_until_test_start':264,
      'all_alerts_with_evaluable_rul':len(evaluable),
      'all_next_fault_events_with_evaluable_rul':int(evaluable.next_fault_event_id.nunique()),
      'first_actual_notification_events':len(first_actual),
      'first_actual_notification_rul_history_available':int((first_actual.status=='A_evaluable').sum()),
      'last_actual_notification_rul_history_available':int((last_actual.status=='A_evaluable').sum()),
      'first_last_two_evaluable_alert_events':len(pair),
      'negative_log_inverse_predictions_clipped':negative,
      'model1_score_sha256':digest(SCORES),
      'rul_source_models_sha256':{'raw_gru':digest(MODELS/'simple_full_fault_rul_gru.pt'),
                                  'log_gru':digest(MODELS/'log_ttf_rul_gru.pt')},
      'source_policy':policy['policy'],
      'history_rule':'at alert time, only earlier raw observations within 1h and same validation interval/next-fault cycle; >=60 observations spanning >=48m; select 60 evenly spaced observed indices; no fill',
      'label_scope':'next logged fault strictly before per-Tool validation_end_exclusive_second; no reserved test outcome or sensor used',
      'rul_model_fitting':False,'model1_fitting':False,
      'caveat':'RUL models trained on 30-minute grid anchors; manager notification timestamps are different and involve nearest prior sensor observation. This is deployment-time sampling shift. Validation is development data, not final independent test.'}
    (OUT/'experiment_config_and_counts.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('threshold_hits','raw_alarm_episode_starts','manager_notifications','status_counts','all_validation_faults_within_validation_end','all_alerts_with_evaluable_rul','all_next_fault_events_with_evaluable_rul')},indent=2),flush=True)
    print(pd.DataFrame(overall).to_string(index=False),flush=True)
    print(pd.DataFrame(funnel).to_string(index=False),flush=True)

if __name__=='__main__':main()
