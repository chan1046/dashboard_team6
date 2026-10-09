"""Read-only 2018 TV window semantics and past-fault-history audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/experiments/tv_window_semantics_audit'
RAW = ROOT/'data/processed'
PLAN = ROOT/'output/experiments/forward_validation_ab_24h/split_plan.json'
FAULTS = RAW/'fault_events.csv'
TYPES = ['pressure_low','pressure_high','leak']
SENSORS = ['IONGAUGEPRESSURE','ETCHBEAMVOLTAGE','ETCHBEAMCURRENT',
 'ETCHSUPPRESSORVOLTAGE','ETCHSUPPRESSORCURRENT','FLOWCOOLFLOWRATE',
 'FLOWCOOLPRESSURE','ETCHGASCHANNEL1READBACK','ETCHPBNGASREADBACK']
VARIANTS = [('A_stride1500_fault',1500,True),('B_stride500_fault',500,True),
            ('C_stride1500_no_fault',1500,False),('D_stride500_no_fault',500,False)]
LEN = 2000
RU = 500


def read_tool(item):
    tab=pq.read_table(RAW/f"{item['tool_id']}_sensor_ttf.parquet",
      columns=['time','FIXTURESHUTTERPOSITION','stage',*SENSORS],
      filters=[('time','<',item['validation_end_exclusive_second'])])
    frame=tab.to_pandas()
    if (frame.time >= item['validation_end_exclusive_second']).any():
        raise AssertionError('Reserved Test sensor row returned')
    op=frame.FIXTURESHUTTERPOSITION.to_numpy()==1
    return (frame.time.to_numpy(np.int64)[op],
            frame.loc[op,'stage'].to_numpy(),
            frame.loc[op,SENSORS].to_numpy(np.float32))


def count_windows(t,phase,ft_all,by_type,item,stride,split_fault,tool):
    prev=np.r_[t[0]-1,t[:-1]]
    changes=phase!=np.r_[-1,phase[:-1]]
    crossed=(np.searchsorted(ft_all,t,side='right')>
             np.searchsorted(ft_all,prev,side='right'))
    starts=changes|(t-prev<0)
    if split_fault:starts|=crossed
    starts[0]=True
    bounds=np.r_[np.flatnonzero(starts),len(t)]
    counts=[];examples=[];event_ids={'train':set(),'validation':set()}
    for a,b in zip(bounds[:-1],bounds[1:]):
        if phase[a]==1 or b-a<LEN:continue
        left=np.arange(a,b-LEN+1,stride,dtype=np.int64)
        right=left+LEN-1
        start=t[left];end=t[right]
        partition='train' if phase[a]==0 else 'validation'
        phase_end=(item['train_end_exclusive_second'] if phase[a]==0 else
                   item['validation_end_exclusive_second'])
        inside_lo=np.searchsorted(ft_all,start,side='right')
        inside_hi=np.searchsorted(ft_all,end,side='right')
        inside_n=inside_hi-inside_lo
        one=inside_n==1;multiple=inside_n>=2;none=inside_n==0
        first=np.full(len(left),np.iinfo(np.int64).max,dtype=np.int64)
        last=np.full(len(left),np.iinfo(np.int64).max,dtype=np.int64)
        yes=inside_n>0
        first[yes]=ft_all[inside_lo[yes]]
        last[yes]=ft_all[inside_hi[yes]-1]
        after_first=yes&(end>first)
        after_last=yes&(end>last)
        current_target_after_fault=np.zeros(len(left),dtype=bool)
        near=np.zeros(len(left),dtype=bool)
        available=np.zeros(len(left),dtype=bool)
        near_type={typ:np.zeros(len(left),dtype=bool) for typ in TYPES}
        for typ in TYPES:
            ft=by_type[typ]
            pos=np.searchsorted(ft,end,side='right')
            future=np.full(len(left),np.iinfo(np.int64).max,dtype=np.int64)
            has=pos<len(ft);future[has]=ft[pos[has]]
            previous_same=np.full(len(left),np.iinfo(np.int64).min,dtype=np.int64)
            had=pos>0;previous_same[had]=ft[pos[had]-1]
            ok=future<phase_end
            if split_fault:ok &= start>previous_same
            available|=ok
            yes_near=ok&((future-end)<RU)
            near|=yes_near
            near_type[typ]=yes_near
            # The target is a strict future fault. This explicit assertion
            # detects a target that has already occurred before input end.
            current_target_after_fault |= ok&(future<=end)
            for event_time in np.unique(future[ok]):
                event_ids[partition].add((tool,typ,int(event_time)))
        if current_target_after_fault.any():
            raise AssertionError('Target-fault post-event sensor leakage')
        counts.append({'variant':('fault' if split_fault else 'no_fault')+f'_stride{stride}',
           'tool_id':tool,'phase':partition,'windows':len(left),
           'usable_target_windows':int(available.sum()),
           'near_any':int(near.sum()),
           **{f'near_{typ}':int(near_type[typ].sum()) for typ in TYPES},
           'zero_fault_inside':int(none.sum()),'one_fault_inside':int(one.sum()),
           'two_or_more_faults_inside':int(multiple.sum()),
           'post_first_fault_sensor':int(after_first.sum()),
           'post_last_fault_sensor':int(after_last.sum()),
           'target_fault_at_or_before_input_end':int(current_target_after_fault.sum())})
        for k in np.flatnonzero(yes):
            examples.append({'variant':('fault' if split_fault else 'no_fault')+f'_stride{stride}',
              'tool_id':tool,'phase':partition,'start_time':int(start[k]),
              'end_time':int(end[k]),'faults_inside':int(inside_n[k]),
              'first_fault_inside':int(first[k]),'last_fault_inside':int(last[k]),
              'post_first_fault_sensor':bool(after_first[k]),
              'post_last_fault_sensor':bool(after_last[k]),
              'target_fault_after_input_end':True,
              'near_any':bool(near[k])})
    return counts,examples,event_ids


def transition_metrics(t,stage,sensors,ft,tool):
    """Describe *past* fault signatures; do not make or fit a feature."""
    if len(t)<2:return []
    # Index of first retained operating observation at/after each fault.
    idx=np.searchsorted(t,ft,side='left')
    idx=idx[(idx>0)&(idx<len(t))]
    if not len(idx):return []
    # Nine-channel largest adjacent change. Compare each fault crossing to
    # tool-specific non-crossing adjacent operating changes (descriptive).
    delta=np.nanmax(np.abs(np.diff(sensors,axis=0)),axis=1)
    prev=np.r_[t[0]-1,t[:-1]]
    event_cross=(np.searchsorted(ft,t,side='right')>
                 np.searchsorted(ft,prev,side='right'))
    ordinary=delta[~event_cross[1:]]
    ordinary=ordinary[np.isfinite(ordinary)]
    q50=float(np.median(ordinary)) if len(ordinary) else np.nan
    q95=float(np.quantile(ordinary,.95)) if len(ordinary) else np.nan
    q99=float(np.quantile(ordinary,.99)) if len(ordinary) else np.nan
    rows=[]
    for i in idx:
        crossed=ft[(ft>t[i-1])&(ft<=t[i])]
        if not len(crossed):continue
        jump=float(delta[i-1])
        rows.append({'tool_id':tool,'fault_time':int(crossed[-1]),
          'operating_before':int(t[i-1]),'operating_after':int(t[i]),
          'gap_seconds':int(t[i]-t[i-1]),
          'stage_changed':bool(stage[i]!=stage[i-1]),
          'stage_numeric_decrease':bool(pd.notna(stage[i]) and pd.notna(stage[i-1]) and
                                         isinstance(stage[i],(int,float,np.integer,np.floating)) and
                                         isinstance(stage[i-1],(int,float,np.integer,np.floating)) and
                                         stage[i]<stage[i-1]),
          'max_abs_raw9_jump':jump,'ordinary_jump_median':q50,
          'ordinary_jump_p95':q95,'ordinary_jump_p99':q99,
          'jump_above_ordinary_p99':bool(jump>q99) if np.isfinite(q99) else False})
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    faults=pd.read_csv(FAULTS).drop_duplicates(['tool_id','time','fault_type'])
    rows=[];crossing=[];unique=[];transitions=[]
    for item in plan['tools']:
        tool=item['tool_id']
        t,stage,sensors=read_tool(item)
        phase=np.select([t<item['train_end_exclusive_second'],
                         t>=item['validation_start_second']],[0,2],default=1)
        log=faults.loc[(faults.tool_id==tool)&
                       (faults.time<item['validation_end_exclusive_second'])]
        ft=np.sort(log.time.unique().astype(np.int64))
        by={typ:np.sort(log.loc[log.fault_type==typ,'time'].unique().astype(np.int64))
            for typ in TYPES}
        transitions.extend(transition_metrics(t,stage,sensors,ft,tool))
        for _,stride,boundary in VARIANTS:
            r,c,e=count_windows(t,phase,ft,by,item,stride,boundary,tool)
            rows.extend(r);crossing.extend(c)
            for ph,ids in e.items():
                unique.extend({'variant':('fault' if boundary else 'no_fault')+f'_stride{stride}',
                   'phase':ph,'tool_id':x[0],'fault_type':x[1],
                   'fault_time':x[2]} for x in ids)
        print('semantics',tool,flush=True)
    counts=pd.DataFrame(rows)
    counts.to_csv(OUT/'window_counts_by_tool_phase.csv',index=False)
    numeric=[c for c in counts.columns if c not in ['variant','tool_id','phase']]
    phase=counts.groupby(['variant','phase'],as_index=False)[numeric].sum()
    phase.to_csv(OUT/'window_counts_by_variant_phase.csv',index=False)
    total=counts.groupby('variant',as_index=False)[numeric].sum()
    total.to_csv(OUT/'window_counts_by_variant_total.csv',index=False)
    events=pd.DataFrame(unique).drop_duplicates()
    events.to_csv(OUT/'covered_target_events.csv',index=False)
    ec=events.groupby(['variant','phase','fault_type']).size().rename('unique_events').reset_index()
    ec.to_csv(OUT/'unique_events_by_variant_phase_type.csv',index=False)
    cr=pd.DataFrame(crossing)
    cr.to_csv(OUT/'fault_crossing_window_metadata.csv.gz',index=False,compression='gzip')
    tr=pd.DataFrame(transitions).drop_duplicates(['tool_id','fault_time'])
    tr.to_csv(OUT/'fault_transition_signatures.csv',index=False)
    cfg={'model_training':False,'data':'real pre-reserved-Test 20 Tool FSP=1 observations',
      'window_length':LEN,'variants':VARIANTS,'target':'next strictly later same-type fault within frozen split',
      'near_threshold_raw_ttf_units':RU,
      'past_fault_signature':'stage change/decrease, elapsed gap, max absolute raw-nine sensor jump around first operating observations across a logged fault; ordinary jump quantiles from same Tool nonfault transitions',
      'future_leakage':'no window with target fault <= input end; no post-target sensor',
      'split':'frozen per-Tool chronological Train/Validation with >=24h purge; Test sensor rows unread'}
    (OUT/'audit_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    print(total.to_string(index=False),flush=True)
    print(ec.to_string(index=False),flush=True)
    print('transition rows',len(tr),flush=True)

if __name__=='__main__':main()
