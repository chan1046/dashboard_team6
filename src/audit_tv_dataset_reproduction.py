"""Read-only TV 2018 window/TTF construction audit. No model fitting."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/experiments/tv_dataset_reproduction_audit'
RAW=ROOT/'data/processed'
PLAN=ROOT/'output/experiments/forward_validation_ab_24h/split_plan.json'
FAULT=RAW/'fault_events.csv'
TYPES=['pressure_low','pressure_high','leak']
TTF=['TTF_FlowCool Pressure Dropped Below Limit',
     'TTF_Flowcool Pressure Too High Check Flowcool Pump','TTF_Flowcool leak']
SENSORS=['IONGAUGEPRESSURE','ETCHBEAMVOLTAGE','ETCHBEAMCURRENT',
 'ETCHSUPPRESSORVOLTAGE','ETCHSUPPRESSORCURRENT','FLOWCOOLFLOWRATE',
 'FLOWCOOLPRESSURE','ETCHGASCHANNEL1READBACK','ETCHPBNGASREADBACK']
LENGTH=2000; RU=500; SEED=42

def create_sums():
    return {'windows':0,'sensor_finite_windows':0,'target_available_windows':0,
       'usable_windows':0,'direct_near_windows':0,'source_near_windows':0,
       'usable_direct_near_windows':0,'usable_source_near_windows':0,
       'cross_any_fault_windows':0,'cross_same_type_fault_windows':0,
       'post_fault_observation_windows':0,
       **{f'direct_near_{t}':0 for t in TYPES},
       **{f'source_near_{t}':0 for t in TYPES}}

def add(dst,src):
    for k,v in src.items():dst[k]+=int(v)

def count_windows(t,phase,valid_sensor,provided,ft_all,ft_by,item,stride,cut_fault):
    counts={'train':create_sums(),'validation':create_sums()}
    if len(t)<LENGTH:return counts,[]
    previous=np.r_[t[0]-1,t[:-1]]
    crossing=(np.searchsorted(ft_all,t,side='right')>
              np.searchsorted(ft_all,previous,side='right'))
    starts=(phase!=np.r_[-1,phase[:-1]])|(t-previous<0)
    if cut_fault:starts|=crossing
    starts[0]=True
    boundaries=np.r_[np.flatnonzero(starts),len(t)]
    invalid_prefix=np.r_[0,np.cumsum(~valid_sensor,dtype=np.int64)]
    selected=[]
    for a,b in zip(boundaries[:-1],boundaries[1:]):
        if phase[a]==1 or b-a<LENGTH:continue
        left=np.arange(a,b-LENGTH+1,stride,dtype=np.int64)
        right=left+LENGTH-1
        st=t[left];anchor=t[right]
        partition='train' if phase[a]==0 else 'validation'
        end=(item['train_end_exclusive_second'] if phase[a]==0
             else item['validation_end_exclusive_second'])
        sensor_ok=(invalid_prefix[right+1]-invalid_prefix[left])==0
        has_target=np.zeros(len(left),dtype=bool)
        direct_near=np.zeros(len(left),dtype=bool)
        source_near=np.zeros(len(left),dtype=bool)
        same_cross=np.zeros(len(left),dtype=bool)
        per_direct={};per_source={}
        for j,typ in enumerate(TYPES):
            ft=ft_by[typ]
            pos=np.searchsorted(ft,anchor,side='right')
            future=np.full(len(left),np.iinfo(np.int64).max,dtype=np.int64)
            yes=pos<len(ft);future[yes]=ft[pos[yes]]
            previous_same=np.full(len(left),np.iinfo(np.int64).min,dtype=np.int64)
            had=pos>0;previous_same[had]=ft[pos[had]-1]
            eligible=(future<end)
            if cut_fault:eligible &= st>previous_same
            target=np.where(eligible,future-anchor,np.nan)
            has_target|=eligible
            near=eligible&(target<RU)
            per_direct[typ]=near
            direct_near|=near
            source=provided[right,j]
            src_near=np.isfinite(source)&(source<RU)
            per_source[typ]=src_near
            source_near|=src_near
            same_cross|=(np.searchsorted(ft,anchor,side='right')>
                          np.searchsorted(ft,st,side='right'))
        crosses=(np.searchsorted(ft_all,anchor,side='right')>
                 np.searchsorted(ft_all,st,side='right'))
        # If an earlier fault is inside the sequence, observations after it
        # appear in the feature history. No observation after the NEXT target
        # fault is used because the target is strictly after the anchor.
        c=counts[partition]
        c['windows']+=len(left)
        c['sensor_finite_windows']+=int(sensor_ok.sum())
        c['target_available_windows']+=int(has_target.sum())
        c['usable_windows']+=int((sensor_ok&has_target).sum())
        c['direct_near_windows']+=int(direct_near.sum())
        c['source_near_windows']+=int(source_near.sum())
        c['usable_direct_near_windows']+=int((sensor_ok&has_target&direct_near).sum())
        c['usable_source_near_windows']+=int((sensor_ok&has_target&source_near).sum())
        c['cross_any_fault_windows']+=int(crosses.sum())
        c['cross_same_type_fault_windows']+=int(same_cross.sum())
        c['post_fault_observation_windows']+=int(crosses.sum())
        for typ in TYPES:
            c[f'direct_near_{typ}']+=int(per_direct[typ].sum())
            c[f'source_near_{typ}']+=int(per_source[typ].sum())
        # Sample timestamp/value comparisons independently of window targets.
        if len(right):
            take=np.linspace(0,len(right)-1,min(50,len(right)),dtype=int)
            selected.extend(right[take].tolist())
    return counts,selected

def compare_sample(tool,t,provided,ft_by,phase,item,rng):
    # 1000 random operating timestamps per Tool, plus every supplied near
    # timestamp in this pre-Test partition (the latter may be many rows).
    chosen=np.unique(np.r_[rng.choice(len(t),size=min(1000,len(t)),replace=False),
                          np.flatnonzero(np.isfinite(provided).any(axis=1)&
                                (np.nan_to_num(provided,nan=np.inf)<RU).any(axis=1))])
    result=[]
    for j,typ in enumerate(TYPES):
        ft=ft_by[typ]
        anchor=t[chosen]
        # Supplied train_ttf uses 0 at the logged fault timestamp. The
        # comparison therefore treats a fault at the same timestamp as next.
        pos=np.searchsorted(ft,anchor,side='left')
        next_time=np.full(len(chosen),np.iinfo(np.int64).max,dtype=np.int64)
        ok=pos<len(ft);next_time[ok]=ft[pos[ok]]
        calculated=np.where(ok,next_time-anchor,np.nan)
        supplied=provided[chosen,j]
        for idx,src,direct in zip(chosen,supplied,calculated):
            result.append({'tool_id':tool,'time':int(t[idx]),'phase':int(phase[idx]),
              'fault_type':typ,'provided_ttf':float(src) if np.isfinite(src) else np.nan,
              'direct_ttf':float(direct) if np.isfinite(direct) else np.nan})
    return result

def check_raw_csv(tool,processed):
    # Direct provenance check against the actual train/train_ttf CSV. The
    # processed parquet alone is never treated as proof of source identity.
    raw=pd.read_csv(ROOT/f'data/train/train_ttf/{tool}_DC_train_ttf.csv',nrows=50)
    parsed=processed[['time',*TTF]].drop_duplicates('time',keep='first')
    joined=raw.merge(parsed,on='time',how='left',suffixes=('_csv','_parquet'),validate='many_to_one')
    good=np.ones(len(joined),dtype=bool)
    for col in TTF:
        x=joined[col+'_csv'].to_numpy(float);y=joined[col+'_parquet'].to_numpy(float)
        good &= (np.isclose(x,y,atol=0,rtol=0)|(~np.isfinite(x)&~np.isfinite(y)))
    return {'tool_id':tool,'csv_rows_checked':len(raw),'matched_rows':int(good.sum())}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    faults=pd.read_csv(FAULT).drop_duplicates(['tool_id','time','fault_type'])
    rng=np.random.default_rng(SEED)
    variants=[('stride1500_fault',1500,True),('stride500_fault',500,True),
      ('stride1500_no_fault',1500,False),('stride500_no_fault',500,False)]
    rows=[];samples=[];csvchecks=[]
    for item in plan['tools']:
        tool=item['tool_id']
        table=pq.read_table(RAW/f'{tool}_sensor_ttf.parquet',
          columns=['time','FIXTURESHUTTERPOSITION',*TTF,*SENSORS],
          filters=[('time','<',item['validation_end_exclusive_second'])])
        full=table.to_pandas()
        if (full.time>=item['validation_end_exclusive_second']).any():
            raise AssertionError('Reserved Test row unexpectedly read')
        csvchecks.append(check_raw_csv(tool,full.head(1000)))
        op=full.FIXTURESHUTTERPOSITION.to_numpy()==1
        t=full.time.to_numpy(np.int64)[op]
        provided=full.loc[op,TTF].to_numpy(np.float64)
        valid=full.loc[op,SENSORS].notna().all(axis=1).to_numpy()
        phase=np.select([t<item['train_end_exclusive_second'],
                         t>=item['validation_start_second']],[0,2],default=1)
        log=faults.loc[(faults.tool_id==tool)&
                       (faults.time<item['validation_end_exclusive_second'])]
        allft=np.sort(log.time.unique().astype(np.int64))
        by={typ:np.sort(log.loc[log.fault_type==typ,'time'].unique().astype(np.int64))
            for typ in TYPES}
        full_log=faults.loc[faults.tool_id==tool]
        full_by={typ:np.sort(full_log.loc[full_log.fault_type==typ,'time'].unique().astype(np.int64))
                 for typ in TYPES}
        samples.extend(compare_sample(tool,t,provided,full_by,phase,item,rng))
        for label,stride,boundary in variants:
            counts,_=count_windows(t,phase,valid,provided,allft,by,item,stride,boundary)
            for ph,stats in counts.items():
                rows.append({'variant':label,'tool_id':tool,'phase':ph,
                  'window_length':LENGTH,'stride':stride,'overlap':LENGTH-stride,
                  'fault_boundary':boundary,**stats})
        print(tool,'done',flush=True)
    detail=pd.DataFrame(rows)
    detail.to_csv(OUT/'window_counts_by_tool_phase.csv',index=False)
    numeric=[c for c in detail.columns if c not in
             ['variant','tool_id','phase','window_length','stride','overlap','fault_boundary']]
    summary=detail.groupby(['variant','phase'],as_index=False)[numeric].sum()
    summary.to_csv(OUT/'window_counts_by_variant_phase.csv',index=False)
    total=detail.groupby('variant',as_index=False)[numeric].sum()
    total.to_csv(OUT/'window_counts_by_variant_total.csv',index=False)
    sample=pd.DataFrame(samples)
    sample['both_finite']=sample.provided_ttf.notna()&sample.direct_ttf.notna()
    sample['both_nan']=sample.provided_ttf.isna()&sample.direct_ttf.isna()
    sample['difference']=sample.provided_ttf-sample.direct_ttf
    sample['exact_match']=sample.both_finite&(sample.difference==0)
    sample.to_csv(OUT/'ttf_comparison_sample.csv.gz',index=False,compression='gzip')
    bytype=sample.groupby('fault_type').agg(samples=('time','size'),
      both_finite=('both_finite','sum'),exact_finite=('exact_match','sum'),
      both_nan=('both_nan','sum'),provided_nan=('provided_ttf',lambda x:int(x.isna().sum())),
      direct_nan=('direct_ttf',lambda x:int(x.isna().sum())),
      difference_min=('difference','min'),difference_median=('difference','median'),
      difference_max=('difference','max')).reset_index()
    bytype['exact_finite_pct']=100*bytype.exact_finite/bytype.both_finite
    bytype.to_csv(OUT/'ttf_comparison_by_fault_type.csv',index=False)
    pd.DataFrame(csvchecks).to_csv(OUT/'raw_csv_parquet_provenance_check.csv',index=False)
    config={'model_training':False,'raw_data_only':True,'length':LENGTH,
      'variants':[{'name':a,'stride':s,'fault_boundary':b} for a,s,b in variants],
      'split':'frozen forward Train/Validation, reserved Test sensor rows unread',
      'time_reversal':'split for all variants to preserve chronology',
      'source_ttf':'original train_ttf columns merged into processed parquet, raw CSV parity checked',
      'direct_ttf':'window count uses next strictly future same-type fault within same phase as prior baseline; raw TTF sample comparison uses next fault at or after timestamp across full logged Train to match provided zero at fault',
      'source_comparison':'1000 random FSP=1 observations per Tool plus all supplied TTF<500 observations, pre-Test',
      'fault_crossing':'counts any logged fault strictly after window start and no later than window end; those windows include post-earlier-fault observations, but not observations after their next target fault',
      'paper_source':'https://papers.phmsociety.org/index.php/phmconf/article/download/589/phmc_18_589'}
    (OUT/'audit_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(total.to_string(index=False),flush=True)
    print(bytype.to_string(index=False),flush=True)

if __name__=='__main__':main()
