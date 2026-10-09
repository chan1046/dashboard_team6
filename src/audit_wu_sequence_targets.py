"""Read-only target coverage of strictly within-segment 2,000-point windows."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/experiments/wu_pressure_residual_ablation'
RAW=ROOT/'data/processed'
PLAN=ROOT/'output/experiments/forward_validation_ab_24h/split_plan.json'
TYPES=('pressure_low','pressure_high','leak')


def main():
    segments=pd.read_csv(OUT/'operating_segments.csv.gz')
    plan={x['tool_id']:x for x in json.loads(PLAN.read_text(encoding='utf-8'))['tools']}
    faults=pd.read_csv(RAW/'fault_events.csv').drop_duplicates(['tool_id','time','fault_type'])
    rows=[]
    for tool,g in segments[(segments.sequence_windows>0)&segments.phase.isin(['train','validation'])].groupby('tool_id'):
        item=plan[tool]
        t=pq.read_table(RAW/f'{tool}_sensor_ttf.parquet',columns=['time'],
            filters=[('time','<',item['validation_end_exclusive_second'])]).column('time').to_numpy()
        for row in g.itertuples():
            for k in range(int(row.sequence_windows)):
                left=int(row.start_raw_row)+1500*k
                right=left+1999
                if right>int(row.end_raw_row):raise AssertionError('Window crosses segment')
                start=int(t[left]);anchor=int(t[right])
                if anchor-start>1999*12:raise AssertionError('Gap rule broken')
                entry={'tool_id':tool,'phase':row.phase,'segment_id':row.segment_id,
                       'start_time':start,'time':anchor,'start_raw_row':left,
                       'end_raw_row':right,'max_gap_s':int(np.diff(t[left:right+1]).max())}
                end=(item['train_end_exclusive_second'] if row.phase=='train'
                     else item['validation_end_exclusive_second'])
                for typ in TYPES:
                    ft=np.sort(faults.loc[(faults.tool_id==tool)&(faults.fault_type==typ),'time'].unique())
                    p=int(np.searchsorted(ft,anchor,side='right'))
                    next_time=int(ft[p]) if p<len(ft) else None
                    prev_time=int(ft[p-1]) if p else None
                    valid=(next_time is not None and next_time<end and
                           (prev_time is None or start>prev_time))
                    entry[f'event_{typ}']=f'{tool}|{typ}|{next_time}' if valid else None
                    entry[f'ttf_h_{typ}']=(next_time-anchor)/3600 if valid else np.nan
                rows.append(entry)
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'strict_sequence_target_availability.csv',index=False)
    print(frame.groupby('phase').size().to_string())
    for typ in TYPES:
        print(typ,frame.groupby('phase')[f'event_{typ}'].nunique().to_dict(),
              'rows',frame.groupby('phase')[f'ttf_h_{typ}'].count().to_dict())


if __name__=='__main__':main()
