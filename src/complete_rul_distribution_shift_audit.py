"""Additional read-only composition and pooled sensor-standardization checks."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import ks_2samp
from audit_rul_distribution_shift import OUT,BASE,RAW,PLAN,SENSORS,weighted_quantile

def main():
    if (OUT/'sensor_pooled_normalized.csv').exists():raise FileExistsError(OUT/'sensor_pooled_normalized.csv')
    m=pd.read_csv(BASE/'sample_metadata.csv.gz')
    e=pd.read_csv(OUT/'event_level_ttf.csv')
    rows=[]
    train=e[e.phase=='train'];val=e[e.phase=='validation']
    rows.append({'comparison':'event_median_unadjusted','train_median_h':train.median_ttf_hours.median(),
      'validation_median_h':val.median_ttf_hours.median(),'validation_weighted_mean_h':val.median_ttf_hours.mean()})
    for g in ('fault_type','tool_id'):
        ts=train[g].value_counts(normalize=True);vs=val[g].value_counts(normalize=True)
        common=set(ts.index)&set(vs.index)
        v=val[val[g].isin(common)].copy();t=train[train[g].isin(common)].copy()
        ts=ts.loc[list(common)];ts/=ts.sum();vs=vs.loc[list(common)];vs/=vs.sum()
        w=v[g].map((ts/vs).to_dict()).to_numpy()
        rows.append({'comparison':f'event_median_{g}_common_support_standardized',
          'common_groups':len(common),'train_events':len(t),'validation_events':len(v),
          'train_median_h':float(t.median_ttf_hours.median()),
          'validation_median_h':float(v.median_ttf_hours.median()),
          'validation_reweighted_median_h':weighted_quantile(v.median_ttf_hours,w,.5),
          'validation_weighted_mean_h':float(np.average(v.median_ttf_hours,weights=w))})
    pd.DataFrame(rows).to_csv(OUT/'composition_standardization.csv',index=False)
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    chunks=[]
    for item in plan['tools']:
        tool=item['tool_id'];wanted=m[m.tool_id==tool]
        if wanted.empty:continue
        frame=pq.read_table(RAW/f'{tool}_sensor_ttf.parquet',columns=['time']+SENSORS,
          filters=[('time','<',int(item['validation_end_exclusive_second']))]).to_pandas()
        tt=frame.time.to_numpy(dtype=np.int64)
        ii=np.searchsorted(tt,wanted.anchor_time.to_numpy(dtype=np.int64),side='right')-1
        assert np.all(tt[ii]==wanted.anchor_time.to_numpy(dtype=np.int64))
        x=frame.iloc[ii][SENSORS].reset_index(drop=True).copy()
        x['phase']=wanted.phase.to_numpy()
        if not (x.phase=='train').any():
            print('no Train reference; exclude from pooled normalization',tool,flush=True)
            continue
        for s in SENSORS:
            tr=x.loc[x.phase=='train',s].dropna().to_numpy(dtype=float)
            median=np.median(tr);iqr=np.percentile(tr,75)-np.percentile(tr,25)
            x[s+'_scaled']=(x[s]-median)/(iqr if iqr>0 else 1)
        chunks.append(x)
        print('pooled normalized',tool,flush=True)
    full=pd.concat(chunks,ignore_index=True)
    out=[]
    for s in SENSORS:
        tr=full[full.phase=='train'];va=full[full.phase=='validation']
        a=tr[s].dropna().to_numpy();b=va[s].dropna().to_numpy()
        az=tr[s+'_scaled'].dropna().to_numpy();bz=va[s+'_scaled'].dropna().to_numpy()
        out.append({'sensor':s,'pooled_raw_ks':float(ks_2samp(a,b,method='asymp').statistic),
          'pooled_tool_train_scaled_ks':float(ks_2samp(az,bz,method='asymp').statistic),
          'pooled_raw_train_median':float(np.median(a)),'pooled_raw_validation_median':float(np.median(b)),
          'pooled_scaled_train_median':float(np.median(az)),
          'pooled_scaled_validation_median':float(np.median(bz))})
    pd.DataFrame(out).to_csv(OUT/'sensor_pooled_normalized.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(out).to_string(index=False))

if __name__=='__main__':main()
