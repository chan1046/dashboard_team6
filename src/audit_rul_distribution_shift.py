"""Read-only audit of the existing simple full-fault RUL experiment."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import ks_2samp
from audit_simple_full_fault_rul import candidates
from train_simple_full_fault_rul import SENSORS, ROOT, RAW, PLAN

BASE=ROOT/'output/experiments/simple_full_fault_rul'
OUT=ROOT/'output/experiments/rul_distribution_shift_audit'
BINS=[0,3,6,12,24,72,168,np.inf]
NAMES=['0-3h','3-6h','6-12h','12-24h','24-72h','72-168h','168h+']

def weighted_quantile(values,weights,q):
    a=np.asarray(values,dtype=float);w=np.asarray(weights,dtype=float)
    ok=np.isfinite(a)&np.isfinite(w)&(w>0)
    a=a[ok];w=w[ok]
    if not len(a):return np.nan
    ix=np.argsort(a);a=a[ix];w=w[ix]
    return float(a[min(len(a)-1,np.searchsorted(np.cumsum(w),q*w.sum(),side='left'))])

def describe(values,weights=None):
    a=np.asarray(values,dtype=float)
    w=np.ones(len(a)) if weights is None else np.asarray(weights,dtype=float)
    valid=np.isfinite(a)&np.isfinite(w)&(w>0)
    a=a[valid];w=w[valid]
    return {'count':len(a),'mean':float(np.average(a,weights=w)) if len(a) else np.nan,
      **{name:weighted_quantile(a,w,q) for name,q in [('p25',.25),('median',.5),('p75',.75),('p90',.9),('p95',.95),('max',1)]}}

def add_weights(frame):
    result=frame.copy()
    sizes=result.groupby('fault_event_id').fault_event_id.transform('size')
    result['event_weight']=1/sizes
    result['ttf_bin']=pd.cut(result.target_ttf_hours,BINS,labels=NAMES,right=False)
    return result

def get_uncapped_train(plan,faults):
    values=[];rows=[];ranges={}
    for item in plan['tools']:
        tool=item['tool_id'];path=RAW/f'{tool}_sensor_ttf.parquet'
        t=pq.read_table(path,columns=['time']).column('time').to_numpy(zero_copy_only=False).astype(np.int64)
        ranges[tool]=(int(t[0]),int(t[-1]))
        log=faults[faults.tool_id==tool].sort_values('time',kind='stable')
        prev=None
        for event in log.itertuples():
            f=int(event.time)
            if f>=int(item['train_end_exclusive_second']):break
            lo=max(0 if prev is None else int(np.searchsorted(t,prev,side='right')),0)
            hi=min(int(np.searchsorted(t,f,side='left')),int(np.searchsorted(t,int(item['train_end_exclusive_second']),side='left')))
            a=candidates(t,lo,hi,3600,1800)
            if len(a):
                y=(f-t[a])/3600
                values.extend(y.tolist())
                rows.append({'tool_id':tool,'fault_event_id':int(event.fault_event_id),
                  'candidate_count':len(a),'uncapped_event_median_ttf':float(np.median(y))})
            prev=f
        print('time audit',tool,flush=True)
    return np.asarray(values),pd.DataFrame(rows),ranges

def event_table(frame):
    return frame.groupby(['fault_event_id','tool_id','fault_type'],observed=True).agg(
      samples=('target_ttf_hours','size'),median_ttf_hours=('target_ttf_hours','median'),
      mean_ttf_hours=('target_ttf_hours','mean'),first_anchor=('anchor_time','min'),
      last_anchor=('anchor_time','max'),fault_time=('fault_time','first')).reset_index()

def main():
    if (OUT/'audit_config.json').exists():raise FileExistsError(f'Refusing to overwrite completed audit {OUT}')
    OUT.mkdir(parents=True,exist_ok=True)
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    faults=pd.read_csv(RAW/'fault_events.csv').drop_duplicates(['tool_id','time','fault_type'])
    meta=add_weights(pd.read_csv(BASE/'sample_metadata.csv.gz'))
    pred=pd.read_csv(BASE/'validation_predictions.csv.gz')
    assert len(pred)==len(meta[meta.phase=='validation'])
    assert set(meta[meta.phase=='train'].fault_event_id).isdisjoint(set(meta[meta.phase=='validation'].fault_event_id))
    uncapped,raw_counts,ranges=get_uncapped_train(plan,faults)
    raw_counts.to_csv(OUT/'uncapped_train_counts_by_event.csv',index=False)
    summary=[];bins=[];events={}
    for phase,part in meta.groupby('phase'):
        e=event_table(part);events[phase]=e
        for basis,w in [('row',np.ones(len(part))),('event_equal',part.event_weight.to_numpy())]:
            summary.append({'phase':phase,'basis':basis,**describe(part.target_ttf_hours,w),
              'unique_events':part.fault_event_id.nunique()})
            for label in NAMES:
                mask=(part.ttf_bin==label).to_numpy()
                bins.append({'phase':phase,'basis':basis,'ttf_bin':label,
                  'weighted_share_pct':float(100*w[mask].sum()/w.sum()),'rows':int(mask.sum()),
                  'events_with_rows':int(part.loc[mask,'fault_event_id'].nunique())})
    for basis,arr in [('row',uncapped),('event_median_once',raw_counts.uncapped_event_median_ttf.to_numpy())]:
        summary.append({'phase':'train_uncapped','basis':basis,**describe(arr),
          'unique_events':len(raw_counts)})
        cats=pd.cut(arr,BINS,labels=NAMES,right=False)
        for label in NAMES:bins.append({'phase':'train_uncapped','basis':basis,'ttf_bin':label,
          'weighted_share_pct':float(100*np.mean(cats==label)),'rows':int(np.sum(cats==label)),
          'events_with_rows':np.nan})
    pd.DataFrame(summary).to_csv(OUT/'ttf_distribution_summary.csv',index=False)
    pd.DataFrame(bins).to_csv(OUT/'ttf_bin_shares.csv',index=False)
    all_events=pd.concat([e.assign(phase=p) for p,e in events.items()])
    all_events.to_csv(OUT/'event_level_ttf.csv',index=False)

    # Actual inter-fault intervals, independent of sampled anchor counts.
    flog=faults.sort_values(['tool_id','time'],kind='stable').copy()
    flog['previous_fault_time']=flog.groupby('tool_id').time.shift()
    flog['fault_interval_hours']=(flog.time-flog.previous_fault_time)/3600
    entries=[]
    for item in plan['tools']:
        tool=item['tool_id'];tt=flog[flog.tool_id==tool].copy()
        tt['phase']=np.select([tt.time<item['train_end_exclusive_second'],
            (tt.time>=item['validation_start_second'])&(tt.time<item['validation_end_exclusive_second'])],
            ['train','validation'],default='reserved_or_purge')
        start,end=ranges[tool];span=max(1,end-start)
        tt['time_quartile']=np.minimum(4,1+np.floor(4*(tt.time-start)/span).astype(int))
        entries.append(tt)
    flog=pd.concat(entries,ignore_index=True)
    flog.to_csv(OUT/'fault_intervals_all_events.csv',index=False)
    quart=flog.groupby(['tool_id','time_quartile'],observed=True).agg(
      fault_events=('fault_event_id','size'),median_fault_interval_hours=('fault_interval_hours','median'),
      mean_fault_interval_hours=('fault_interval_hours','mean')).reset_index()
    e2=all_events.copy()
    e2['time_quartile']=[min(4,1+int(4*(f-ranges[t][0])/max(1,ranges[t][1]-ranges[t][0])))
                          for t,f in zip(e2.tool_id,e2.fault_time)]
    e_q=e2.groupby(['tool_id','time_quartile'],observed=True).agg(
        usable_events=('fault_event_id','size'),median_ttf_hours=('median_ttf_hours','median')).reset_index()
    quart=quart.merge(e_q,on=['tool_id','time_quartile'],how='left')
    quart.to_csv(OUT/'time_quartiles_by_tool.csv',index=False)
    global_q=flog.groupby('time_quartile').agg(fault_events=('fault_event_id','size'),
       median_fault_interval_hours=('fault_interval_hours','median'),mean_fault_interval_hours=('fault_interval_hours','mean')).reset_index()
    global_q=global_q.merge(e2.groupby('time_quartile').median_ttf_hours.agg(['median','count']).reset_index(),on='time_quartile',how='left')
    global_q.rename(columns={'median':'event_median_ttf_hours','count':'usable_events'},inplace=True)
    global_q.to_csv(OUT/'time_quartiles_aggregate.csv',index=False)

    toolrows=[];typerows=[]
    for group,target in [('tool_id',toolrows),('fault_type',typerows)]:
        for key in sorted(flog[group].unique()):
            rec={group:key}
            for phase in ('train','validation'):
                s=meta[(meta.phase==phase)&(meta[group]==key)]
                e=events[phase][events[phase][group]==key]
                fi=flog[(flog.phase==phase)&(flog[group]==key)]
                rec[f'{phase}_events']=len(e)
                rec[f'{phase}_all_fault_events']=len(fi)
                rec[f'{phase}_samples']=len(s)
                rec[f'{phase}_row_median_ttf_hours']=s.target_ttf_hours.median()
                rec[f'{phase}_event_equal_median_ttf_hours']=weighted_quantile(s.target_ttf_hours,s.event_weight,.5)
                rec[f'{phase}_event_median_of_medians_hours']=e.median_ttf_hours.median()
                rec[f'{phase}_event_p75_hours']=e.median_ttf_hours.quantile(.75)
                rec[f'{phase}_event_p90_hours']=e.median_ttf_hours.quantile(.9)
                rec[f'{phase}_mean_fault_interval_hours']=fi.fault_interval_hours.mean()
                rec[f'{phase}_median_fault_interval_hours']=fi.fault_interval_hours.median()
            rec['event_equal_median_shift_hours']=rec['validation_event_equal_median_ttf_hours']-rec['train_event_equal_median_ttf_hours']
            target.append(rec)
    pd.DataFrame(toolrows).to_csv(OUT/'tool_comparison.csv',index=False)
    pd.DataFrame(typerows).to_csv(OUT/'fault_type_comparison.csv',index=False)

    # Actual current sensor values at the same saved anchor timestamps.
    sensor_rows=[]
    for item in plan['tools']:
        tool=item['tool_id'];wanted=meta[meta.tool_id==tool]
        if wanted.empty:continue
        table=pq.read_table(RAW/f'{tool}_sensor_ttf.parquet',columns=['time']+SENSORS,
                            filters=[('time','<',int(item['validation_end_exclusive_second']))])
        frame=table.to_pandas();t=frame.time.to_numpy(dtype=np.int64)
        idx=np.searchsorted(t,wanted.anchor_time.to_numpy(dtype=np.int64),side='right')-1
        assert np.all(t[idx]==wanted.anchor_time.to_numpy(dtype=np.int64))
        picked=frame.iloc[idx][SENSORS].reset_index(drop=True)
        picked.insert(0,'phase',wanted.phase.to_numpy());picked.insert(0,'tool_id',tool)
        sensor_rows.append(picked)
        print('sensor audit',tool,flush=True)
    sensor=pd.concat(sensor_rows,ignore_index=True)
    srows=[]
    for tool,part in sensor.groupby('tool_id'):
        tr=part[part.phase=='train'];va=part[part.phase=='validation']
        for col in SENSORS:
            a=tr[col].to_numpy(dtype=float);b=va[col].to_numpy(dtype=float)
            a=a[np.isfinite(a)];b=b[np.isfinite(b)]
            if not len(a) or not len(b):continue
            med=float(np.median(a));iqr=float(np.percentile(a,75)-np.percentile(a,25))
            scale=iqr if iqr>0 else 1.0
            # A common Train-derived monotone transform leaves KS unchanged.
            srows.append({'tool_id':tool,'sensor':col,'train_rows':len(a),'validation_rows':len(b),
              'train_median':med,'validation_median':float(np.median(b)),
              'train_iqr':iqr,'validation_iqr':float(np.percentile(b,75)-np.percentile(b,25)),
              'raw_ks':float(ks_2samp(a,b,method='asymp').statistic),
              'train_scaled_train_median':float(np.median((a-med)/scale)),
              'train_scaled_validation_median':float(np.median((b-med)/scale)),
              'train_scaled_train_iqr':float((np.percentile(a,75)-np.percentile(a,25))/scale),
              'train_scaled_validation_iqr':float((np.percentile(b,75)-np.percentile(b,25))/scale),
              'train_scaled_ks':float(ks_2samp((a-med)/scale,(b-med)/scale,method='asymp').statistic)})
    pd.DataFrame(srows).to_csv(OUT/'sensor_shift_by_tool.csv',index=False)
    aggregate=[]
    for col in SENSORS:
        tr=sensor[sensor.phase=='train'][col].dropna().to_numpy(dtype=float)
        va=sensor[sensor.phase=='validation'][col].dropna().to_numpy(dtype=float)
        per=pd.DataFrame(srows).query('sensor==@col')
        aggregate.append({'sensor':col,'train_median':float(np.median(tr)),
          'validation_median':float(np.median(va)),
          'train_iqr':float(np.percentile(tr,75)-np.percentile(tr,25)),
          'validation_iqr':float(np.percentile(va,75)-np.percentile(va,25)),
          'pooled_raw_ks':float(ks_2samp(tr,va,method='asymp').statistic),
          'median_tool_raw_ks':float(per.raw_ks.median()),
          'median_tool_train_scaled_validation_median':float(per.train_scaled_validation_median.median())})
    pd.DataFrame(aggregate).to_csv(OUT/'sensor_shift_summary.csv',index=False)

    pred['ttf_bin']=pd.cut(pred.target_ttf_hours,[0,3,6,12,24,72,np.inf],
       labels=['0-3h','3-6h','6-12h','12-24h','24-72h','72h+'],right=False)
    pred['error']=pred.GRU_event_balanced-pred.target_ttf_hours
    bias=pred.groupby('ttf_bin',observed=True).agg(rows=('error','size'),events=('fault_event_id','nunique'),
      actual_median_hours=('target_ttf_hours','median'),prediction_median_hours=('GRU_event_balanced','median'),
      mean_bias_hours=('error','mean'),median_bias_hours=('error','median'),
      mae_hours=('error',lambda z:float(np.abs(z).mean()))).reset_index()
    bias.to_csv(OUT/'gru_prediction_bias_by_ttf_bin.csv',index=False)
    config={'no_model_training':True,'source_experiment':str(BASE),'data':'real existing Train/Validation samples; raw sensor 9 at saved anchors; time-only 20-Tool scan for uncapped anchors and observation span',
      'event_equal_definition':'each event contributes total weight 1 over its saved prediction timestamps',
      'fault_interval_definition':'current fault timestamp minus previous same-Tool fault timestamp; first fault interval unavailable',
      'time_quartile_definition':'each Tool original sensor observation start/end wall-clock span divided into four equal-duration intervals',
      'sensor_normalization':'Train per-Tool median/IQR only; same affine transform applied to Train and Validation, so KS is mathematically invariant',
      'reserved_test':'no reserved Test sensor values used for sensor distributions, models or validation; time column across full raw training file used only for observation span and fault chronology audit',
      'no_split_change':True}
    (OUT/'audit_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print(bias.to_string(index=False),flush=True)

if __name__=='__main__':main()
