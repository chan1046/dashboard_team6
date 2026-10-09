"""Freeze raw four-second operating-segment boundaries before RUL fitting."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/experiments/wu_pressure_residual_ablation'
PLAN=ROOT/'output/experiments/forward_validation_ab_24h/split_plan.json'
RAW=ROOT/'data/processed'
FAULTS=RAW/'fault_events.csv'
MAX_GAP_SECONDS=12
LENGTH=2000
STRIDE=1500


def main():
    if OUT.exists():raise FileExistsError(f'Refusing to overwrite {OUT}')
    OUT.mkdir(parents=True)
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    faults=pd.read_csv(FAULTS).drop_duplicates(['tool_id','time','fault_type'])
    rows=[];counts=[]
    for item in plan['tools']:
        tool=item['tool_id'];end=int(item['validation_end_exclusive_second'])
        tab=pq.read_table(RAW/f'{tool}_sensor_ttf.parquet',
                          columns=['time','FIXTURESHUTTERPOSITION'],filters=[('time','<',end)])
        f=tab.to_pandas();t=f.time.to_numpy(np.int64)
        operating=f.FIXTURESHUTTERPOSITION.to_numpy()==1
        phase=np.select([t<int(item['train_end_exclusive_second']),
                         t>=int(item['validation_start_second'])],[0,2],default=1)
        ft=np.sort(faults.loc[(faults.tool_id==tool)&(faults.time<end),'time'].unique().astype(np.int64))
        if len(t)==0:continue
        prev=np.r_[t[0]-1,t[:-1]]
        gap=t-prev
        crossed=(np.searchsorted(ft,t,side='right')>
                 np.searchsorted(ft,prev,side='right'))
        prior_operating=np.r_[False,operating[:-1]]
        new=operating & ((~prior_operating)|(gap<=0)|(gap>MAX_GAP_SECONDS)|
                         (phase!=np.r_[-1,phase[:-1]])|crossed)
        sid=np.cumsum(new,dtype=np.int64)
        selected=np.flatnonzero(operating)
        groups=pd.DataFrame({'sid':sid[selected],'time':t[selected],
                             'phase':phase[selected],'row_idx':selected})
        for segment_id,g in groups.groupby('sid',sort=False):
            ts=g.time.to_numpy();n=len(g)
            maxgap=int(np.diff(ts).max()) if n>1 else 0
            if maxgap>MAX_GAP_SECONDS:raise AssertionError('Cross-stop segment')
            rows.append({'tool_id':tool,'segment_id':f'{tool}|{segment_id}',
                         'phase':('train','purge','validation')[int(g.phase.iloc[0])],
                         'start_time':int(ts[0]),'end_time':int(ts[-1]),
                         'start_raw_row':int(g.row_idx.iloc[0]),
                         'end_raw_row':int(g.row_idx.iloc[-1]),
                         'observations':n,'elapsed_seconds':int(ts[-1]-ts[0]),
                         'max_internal_gap_seconds':maxgap,
                         'eligible_2000':n>=LENGTH,
                         'sequence_windows':max(0,(n-LENGTH)//STRIDE+1)})
        counts.append({'tool_id':tool,'raw_rows_pretest':len(t),
                       'operating_rows':int(operating.sum()),
                       'operating_segments':int(new.sum()),
                       'operating_to_downtime_transitions':int(np.sum(operating[:-1]&~operating[1:])),
                       'downtime_to_operating_transitions':int(np.sum(~operating[:-1]&operating[1:])),
                       'operating_gap_over_12s':int(np.sum(operating&prior_operating&(gap>MAX_GAP_SECONDS))),
                       'operating_gap_over_120s':int(np.sum(operating&prior_operating&(gap>120))),
                       'fault_boundaries_with_operating_row':int(np.sum(operating&crossed))})
        print(tool,'segments',int(new.sum()),flush=True)
    seg=pd.DataFrame(rows);seg.to_csv(OUT/'operating_segments.csv.gz',index=False,compression='gzip')
    pd.DataFrame(counts).to_csv(OUT/'tool_segment_summary.csv',index=False)
    config={'operating':'FIXTURESHUTTERPOSITION == 1',
            'new_segment_when':'previous raw row not operating; timestamp <= prior or >12 seconds after prior; frozen split phase changes; any fault event crossed',
            'max_gap_seconds':MAX_GAP_SECONDS,'window_observations':LENGTH,
            'overlap_observations':500,'stride_observations':STRIDE,
            'no_fill':'downtime omitted; never concatenate segments',
            'split_plan':str(PLAN),'reserved_test':'sensor rows at/after validation_end_exclusive_second not read'}
    (OUT/'operating_segment_definition.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('TOTAL',len(seg),'eligible',int(seg.eligible_2000.sum()),'windows',int(seg.sequence_windows.sum()),flush=True)


if __name__=='__main__':main()
