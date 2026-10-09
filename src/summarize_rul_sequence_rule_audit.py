"""Summarize saved sequence-rule audit; no model fit or source mutation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/experiments/rul_sequence_rule_audit'
FAULTS=ROOT/'data/processed/fault_events.csv'
PLAN=ROOT/'output/experiments/forward_validation_ab_24h/split_plan.json'
TYPES=('pressure_low','pressure_high','leak')


def main():
    cause=pd.read_csv(OUT/'segment_break_causes_by_tool.csv').groupby('cause',as_index=False)[['exclusive_breaks','multilabel_count']].sum()
    cause['pct_of_456703']=cause.exclusive_breaks/cause.exclusive_breaks.sum()*100
    if cause.exclusive_breaks.sum()!=456703:raise AssertionError('Strict breaks do not reconcile')
    cause.to_csv(OUT/'segment_break_cause_summary.csv',index=False)
    gap=pd.read_csv(OUT/'operating_gap_bins_by_tool.csv').groupby('gap_bin',as_index=False)['count'].sum()
    gap['pct']=gap['count']/gap['count'].sum()*100
    gap.to_csv(OUT/'operating_gap_distribution.csv',index=False)
    w=pd.read_csv(OUT/'window_metadata.csv.gz')
    rows=[]
    for (rule,phase),g in w.groupby(['rule','phase']):
        ids=set()
        type_count={}
        for typ in TYPES:
            e=set(g[f'event_{typ}'].dropna())
            ids.update(e);type_count[f'{typ}_events']=len(e)
        rows.append({'rule':rule,'phase':phase,'sequences':len(g),
                     'unique_fault_events':len(ids),**type_count,
                     'tools_with_window':g.tool_id.nunique()})
    pd.DataFrame(rows).to_csv(OUT/'rule_phase_coverage.csv',index=False)
    duration=[]
    for rule,g in w.groupby('rule'):
        hours=g.elapsed_seconds.to_numpy()/3600
        duration.append({'rule':rule,'windows':len(g),'median_h':float(np.median(hours)),
            'p25_h':float(np.quantile(hours,.25)),'p75_h':float(np.quantile(hours,.75)),
            'p90_h':float(np.quantile(hours,.90)),'min_h':float(hours.min()),
            'max_h':float(hours.max()),'over_24h_windows':int(np.sum(hours>24))})
    pd.DataFrame(duration).to_csv(OUT/'window_elapsed_overall.csv',index=False)
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    faults=pd.read_csv(FAULTS).drop_duplicates(['tool_id','time','fault_type'])
    phase_fault=[]
    for item in plan['tools']:
        f=faults[faults.tool_id==item['tool_id']]
        phase_fault.extend([{'tool_id':item['tool_id'],'fault_type':x.fault_type,
            'time':x.time,'phase':('train' if x.time<item['train_end_exclusive_second']
              else 'validation' if x.time>=item['validation_start_second'] and
                   x.time<item['validation_end_exclusive_second']
              else 'reserved_test_or_purge')} for x in f.itertuples()])
    pd.DataFrame(phase_fault).groupby(['phase','fault_type']).size().rename('fault_events').reset_index().to_csv(
        OUT/'source_fault_population_by_phase.csv',index=False)
    tool=pd.read_csv(OUT/'operating_structure_by_tool.csv')
    b=w[w.rule=='B_operating_axis']
    tool_windows=b.groupby('tool_id').agg(B_windows=('rule','size'),
        B_elapsed_median_h=('elapsed_seconds',lambda v:float(np.median(v)/3600)),
        B_elapsed_max_h=('elapsed_seconds',lambda v:float(np.max(v)/3600))).reset_index()
    tool.merge(tool_windows,on='tool_id',how='left').to_csv(OUT/'operating_structure_and_B_windows_by_tool.csv',index=False)
    print(cause.to_string(index=False));print(gap.to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
