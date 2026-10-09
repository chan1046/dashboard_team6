"""Audit real 20-Tool, 1211-fault inter-fault history and anchor coverage."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/experiments/simple_full_fault_rul'
RAW=ROOT/'data/processed'
PLAN=ROOT/'output/experiments/forward_validation_ab_24h/split_plan.json'
FAULT=RAW/'fault_events.csv'
HISTORY_MINUTES=[30,60,180,360,720,1440]
CADENCE_MINUTES=[10,30,60]
MIN_OBSERVATIONS=60
MIN_SPAN_FRACTION=.8

def phase_of_fault(f,item):
    if f<item['train_end_exclusive_second']:return 'train'
    if item['validation_start_second']<=f<item['validation_end_exclusive_second']:return 'validation'
    return 'reserved_or_purge'

def candidates(time,lo,hi,history_s,cadence_s):
    """Actual observed anchors at/before regular wall-clock grid; no imputation."""
    t=time[lo:hi]
    if len(t)<MIN_OBSERVATIONS:return np.empty(0,dtype=np.int64)
    grid=np.arange(((t[0]+cadence_s-1)//cadence_s)*cadence_s,
                   t[-1]+1,cadence_s,dtype=np.int64)
    if not len(grid):return np.empty(0,dtype=np.int64)
    a=np.unique(np.searchsorted(t,grid,side='right')-1)
    a=a[a>=0]
    start=np.searchsorted(t,t[a]-history_s,side='left')
    valid=(((a-start+1)>=MIN_OBSERVATIONS)&
           ((t[a]-t[start])>=MIN_SPAN_FRACTION*history_s))
    return lo+a[valid]

def main():
    if OUT.exists():raise FileExistsError(f'Refusing to overwrite {OUT}')
    OUT.mkdir(parents=True)
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    faults=pd.read_csv(FAULT).drop_duplicates(['tool_id','time','fault_type'])
    if len(faults)!=1211:raise AssertionError('Fault population changed')
    availability=[];grid_rows=[]
    for item in plan['tools']:
        tool=item['tool_id']
        table=pq.read_table(RAW/f'{tool}_sensor_ttf.parquet',columns=['time'])
        t=table.column('time').to_numpy(zero_copy_only=False).astype(np.int64)
        log=faults.loc[faults.tool_id==tool].sort_values('time',kind='stable')
        prev=None
        for event in log.itertuples():
            f=int(event.time)
            lo=0 if prev is None else int(np.searchsorted(t,prev,side='right'))
            hi=int(np.searchsorted(t,f,side='left'))
            in_cycle=t[lo:hi]
            last=int(in_cycle[-1]) if len(in_cycle) else None
            phase=phase_of_fault(f,item)
            row={'tool_id':tool,'fault_event_id':int(event.fault_event_id),
              'fault_time':f,'fault_type':event.fault_type,'phase':phase,
              'previous_fault_time':prev,'cycle_observations':len(in_cycle),
              'last_sensor_time':last,'last_sensor_lead_seconds':f-last if last is not None else np.nan,
              'cycle_elapsed_seconds':int(in_cycle[-1]-in_cycle[0]) if len(in_cycle) else np.nan}
            for minutes in HISTORY_MINUTES:
                h=minutes*60
                if len(in_cycle)>=MIN_OBSERVATIONS:
                    last_idx=len(in_cycle)-1
                    st=np.searchsorted(in_cycle,in_cycle[last_idx]-h,side='left')
                    enough=(last_idx-st+1>=MIN_OBSERVATIONS and
                            in_cycle[last_idx]-in_cycle[st]>=MIN_SPAN_FRACTION*h)
                else:enough=False
                row[f'history_{minutes}m_at_last_observation']=bool(enough)
                row[f'history_{minutes}m_within24h_of_fault']=bool(enough and f-last<=86400) if last is not None else False
            availability.append(row)
            if phase!='reserved_or_purge' and len(in_cycle):
                boundary_start=(t[0] if phase=='train' else item['validation_start_second'])
                boundary_end=(item['train_end_exclusive_second'] if phase=='train' else
                              item['validation_end_exclusive_second'])
                phase_lo=max(lo,int(np.searchsorted(t,boundary_start,side='left')))
                phase_hi=min(hi,int(np.searchsorted(t,boundary_end,side='left')))
                for minutes in [30,60,180,360]:
                    for cadence in CADENCE_MINUTES:
                        anchors=candidates(t,phase_lo,phase_hi,minutes*60,cadence*60)
                        grid_rows.append({'tool_id':tool,'fault_event_id':int(event.fault_event_id),
                          'phase':phase,'fault_type':event.fault_type,
                          'history_minutes':minutes,'cadence_minutes':cadence,
                          'candidate_samples':len(anchors),'has_candidate':len(anchors)>0,
                          'candidate_within24h':int(np.sum(f-t[anchors]<=86400))})
            prev=f
        print('audit',tool,len(log),flush=True)
    a=pd.DataFrame(availability);g=pd.DataFrame(grid_rows)
    a.to_csv(OUT/'fault_history_availability.csv',index=False)
    g.to_csv(OUT/'candidate_anchor_counts_by_event.csv',index=False)
    coverage=[]
    for minutes in HISTORY_MINUTES:
        for scope,col in [('at_last_observation',f'history_{minutes}m_at_last_observation'),
                          ('last_observation_within24h',f'history_{minutes}m_within24h_of_fault')]:
            coverage.append({'history_minutes':minutes,'scope':scope,'total_fault_events':len(a),
              'usable_events':int(a[col].sum()),'coverage_pct':100*a[col].mean()})
    pd.DataFrame(coverage).to_csv(OUT/'history_coverage_summary.csv',index=False)
    bytool=[]
    for minutes in HISTORY_MINUTES:
        col=f'history_{minutes}m_at_last_observation'
        for (tool,typ),part in a.groupby(['tool_id','fault_type']):
            bytool.append({'history_minutes':minutes,'tool_id':tool,'fault_type':typ,
              'total_events':len(part),'usable_events':int(part[col].sum()),
              'coverage_pct':100*part[col].mean()})
    pd.DataFrame(bytool).to_csv(OUT/'history_coverage_by_tool_fault_type.csv',index=False)
    aggregate=g.groupby(['history_minutes','cadence_minutes','phase']).agg(
      candidate_samples=('candidate_samples','sum'),usable_events=('has_candidate','sum'),
      samples_within24h=('candidate_within24h','sum')).reset_index()
    aggregate.to_csv(OUT/'history_cadence_candidate_summary.csv',index=False)
    config={'model_training':False,'all_fault_events':len(a),
      'data_read_for_availability':'time column of original Train parquet across all 20 Tools, including the reserved later time axis; no reserved sensor or TTF values read',
      'model_candidate_scope':'only frozen Train/Validation fault events and sensor times; no reserved/purge event in fitting or evaluation',
      'history_rule':f'within previous-any-fault to current-fault cycle, >= {MIN_OBSERVATIONS} observed sensor timestamps and >= {MIN_SPAN_FRACTION:.0%} of requested wall-clock span at anchor; no synthetic downtime rows',
      'anchor_rule':'latest actual observation at/before each grid time, de-duplicated; no future sensor input',
      'fault_label':'next any-type logged fault; unique fault_event_id',
      'reserved_test':'time-only availability audit, no model use'}
    (OUT/'availability_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(pd.DataFrame(coverage).to_string(index=False),flush=True)
    print(aggregate.to_string(index=False),flush=True)

if __name__=='__main__':main()
