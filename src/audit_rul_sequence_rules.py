"""Audit operating-axis window coverage only; never fit or score an ML model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/experiments/rul_sequence_rule_audit'
RAW=ROOT/'data/processed'
PLAN=ROOT/'output/experiments/forward_validation_ab_24h/split_plan.json'
FAULTS=RAW/'fault_events.csv'
LENGTH=2000
STRIDE=1500
TYPES=('pressure_low','pressure_high','leak')
GAP_BINS=[(-np.inf,0,'<=0s'),(0,8,'0-8s'),(8,30,'8-30s'),(30,60,'30-60s'),
          (60,300,'1-5min'),(300,1800,'5-30min'),(1800,np.inf,'>30min')]


def read_times(item):
    tab=pq.read_table(RAW/f"{item['tool_id']}_sensor_ttf.parquet",
        columns=['time','FIXTURESHUTTERPOSITION'],
        filters=[('time','<',item['validation_end_exclusive_second'])])
    f=tab.to_pandas()
    return f.time.to_numpy(np.int64),f.FIXTURESHUTTERPOSITION.to_numpy()==1


def phase_ids(t,item):
    return np.select([t<item['train_end_exclusive_second'],
                      t>=item['validation_start_second']],[0,2],default=1).astype(np.int8)


def pass1(plan,faults):
    if OUT.exists():raise FileExistsError(f'Refusing to overwrite {OUT}')
    OUT.mkdir(parents=True)
    causes=[];gaps=[];per_tool=[];exact=[]
    for item in plan['tools']:
        tool=item['tool_id'];t,op=read_times(item)
        ph=phase_ids(t,item)
        ft=np.sort(faults.loc[(faults.tool_id==tool)&
            (faults.time<item['validation_end_exclusive_second']),'time'].unique())
        prev=np.r_[t[0]-1,t[:-1]];delta=t-prev
        prior=np.r_[False,op[:-1]]
        fsp=op&~prior
        tool_first=np.zeros(len(t),dtype=bool)
        tool_first[np.flatnonzero(op)[0]]=True
        gap=op&prior&(delta>12)&(delta<=120)
        session=op&prior&(delta>120)
        invalid=op&prior&(delta<=0)
        split=op&(ph!=np.r_[-1,ph[:-1]])
        fault=op&(np.searchsorted(ft,t,side='right')>
                  np.searchsorted(ft,prev,side='right'))
        starts=fsp|gap|session|invalid|split|fault
        # Existing implementation's segment count must reconcile exactly.
        exclusive={
            'Tool first operating row':tool_first,
            'FSP transition':fsp&~tool_first,
            'timestamp gap >12-120s':gap&~fsp,
            'session gap >120s':session&~fsp,
            'non-increasing time':invalid&~fsp&~gap&~session,
            'frozen split boundary':split&~fsp&~gap&~session&~invalid,
            'fault boundary':fault&~fsp&~gap&~session&~invalid&~split}
        accounted=np.zeros(len(t),dtype=bool)
        for label,flag in exclusive.items():
            n=int(flag.sum());accounted|=flag
            causes.append({'tool_id':tool,'cause':label,'exclusive_breaks':n,
                'multilabel_count':int({'Tool first operating row':tool_first,
                  'FSP transition':fsp,'timestamp gap >12-120s':gap,
                  'session gap >120s':session,'non-increasing time':invalid,
                  'frozen split boundary':split,'fault boundary':fault}[label].sum())})
        if not np.array_equal(accounted,starts):raise AssertionError('Cause decomposition failed')
        op_t=t[op];op_gap=np.diff(op_t)
        for lo,hi,label in GAP_BINS:
            count=int(((op_gap>lo)&(op_gap<=hi)).sum())
            gaps.append({'tool_id':tool,'gap_bin':label,'count':count})
        # Distribution around possible valleys; exact positive short-gap modes.
        bins=np.array([0,8,12,20,30,60,120,300,600,900,1800,3600,7200,14400,86400,np.iinfo(np.int64).max])
        hist,_=np.histogram(op_gap,bins=bins)
        for left,right,count in zip(bins[:-1],bins[1:],hist):
            exact.append({'tool_id':tool,'left_exclusive_s':int(left),
                'right_inclusive_s':int(right),'count':int(count)})
        per_tool.append({'tool_id':tool,'raw_rows_pretest':len(t),'operating_rows':int(op.sum()),
            'operating_wall_clock_elapsed_s':int(op_t[-1]-op_t[0]),
            'strict_breaks_recomputed':int(starts.sum()),
            'operating_gap_count':len(op_gap),
            'operating_gap_median_s':float(np.median(op_gap)),
            'operating_gap_p90_s':float(np.quantile(op_gap,.9)),
            'operating_gap_p99_s':float(np.quantile(op_gap,.99))})
        print('pass1',tool,int(starts.sum()),flush=True)
    pd.DataFrame(causes).to_csv(OUT/'segment_break_causes_by_tool.csv',index=False)
    pd.DataFrame(gaps).to_csv(OUT/'operating_gap_bins_by_tool.csv',index=False)
    pd.DataFrame(exact).to_csv(OUT/'operating_gap_histogram_by_tool.csv',index=False)
    pd.DataFrame(per_tool).to_csv(OUT/'operating_structure_by_tool.csv',index=False)
    old=pd.read_csv(ROOT/'output/experiments/wu_pressure_residual_ablation/tool_segment_summary.csv')
    chk=pd.DataFrame(per_tool).merge(old[['tool_id','operating_segments']],on='tool_id')
    if not (chk.strict_breaks_recomputed==chk.operating_segments).all():
        raise AssertionError('Current strict segment count not reproduced')
    print('TOTAL breaks',sum(x['strict_breaks_recomputed'] for x in per_tool),flush=True)


def windows_for_rule(t,phase,ft_by_type,all_faults,tool,rule,max_gap,rolling_gap_max):
    # Each Tool is processed independently. An index boundary is imposed at
    # every fault, split phase, and (for C) oversized operating-observation gap.
    n=len(t)
    if n<LENGTH:return [],[]
    prior=np.r_[t[0]-1,t[:-1]]
    gap=t-prior
    split_change=phase!=np.r_[-1,phase[:-1]]
    fault_cross=(np.searchsorted(all_faults,t,side='right')>
                 np.searchsorted(all_faults,prior,side='right'))
    # Repeated timestamps are kept as distinct provided observations with a
    # zero-second metadata gap. Only actual time reversal breaks the axis.
    starts=split_change|fault_cross|(gap<0)
    starts[0]=True
    if max_gap is not None:starts|=gap>max_gap
    boundaries=np.r_[np.flatnonzero(starts),n]
    rows=[];durations=[]
    for a,b in zip(boundaries[:-1],boundaries[1:]):
        if int(phase[a])==1 or b-a<LENGTH:continue
        left=np.arange(a,b-LENGTH+1,STRIDE,dtype=np.int64)
        right=left+LENGTH-1
        start=t[left];anchor=t[right]
        elapsed=anchor-start
        phase_end=(CURRENT_ITEM['train_end_exclusive_second'] if phase[a]==0
                   else CURRENT_ITEM['validation_end_exclusive_second'])
        ids={}
        for typ in TYPES:
            f=ft_by_type[typ]
            pos=np.searchsorted(f,anchor,side='right')
            future=np.full(len(anchor),np.iinfo(np.int64).max,dtype=np.int64)
            has=pos<len(f)
            future[has]=f[pos[has]]
            previous=np.full(len(anchor),np.iinfo(np.int64).min,dtype=np.int64)
            has_prev=pos>0
            previous[has_prev]=f[pos[has_prev]-1]
            eligible=(future<phase_end)&(start>previous)
            ids[typ]=np.where(eligible,future,-1)
        # Max observed gap is metadata, not synthetic missing sensor values.
        # Segment boundaries above ensure every C gap is <= max_gap.
        for k in range(len(left)):
            ent={'rule':rule,'tool_id':tool,'phase':('train' if phase[a]==0 else 'validation'),
                 'start_time':int(start[k]),'time':int(anchor[k]),
                 'elapsed_seconds':int(elapsed[k]),'max_operating_gap_s':int(rolling_gap_max[right[k]])}
            for typ in TYPES:
                future=int(ids[typ][k])
                ent[f'event_{typ}']=f'{tool}|{typ}|{future}' if future>=0 else None
            rows.append(ent)
        durations.extend(elapsed.tolist())
    return rows,durations


CURRENT_ITEM=None


def pass2(plan,faults,thresholds):
    if not OUT.exists():raise FileNotFoundError('Run pass1 first')
    old=pd.read_csv(ROOT/'output/experiments/wu_pressure_residual_ablation/strict_sequence_target_availability.csv')
    allrows=[]
    for r in old.itertuples():
        ent={'rule':'A_strict','tool_id':r.tool_id,'phase':r.phase,
             'start_time':r.start_time,'time':r.time,
             'elapsed_seconds':int(r.time-r.start_time),'max_operating_gap_s':r.max_gap_s,
             **{f'event_{typ}':getattr(r,f'event_{typ}') for typ in TYPES}}
        allrows.append(ent)
    for item in plan['tools']:
        global CURRENT_ITEM
        CURRENT_ITEM=item
        tool=item['tool_id'];raw_t,op=read_times(item)
        t=raw_t[op];phase=phase_ids(t,item)
        gaps=np.diff(t,prepend=t[0]);gaps[0]=0
        rolling_max=pd.Series(gaps).rolling(LENGTH-1,min_periods=1).max().to_numpy()
        all_faults=np.sort(faults.loc[(faults.tool_id==tool)&
            (faults.time<item['validation_end_exclusive_second']),'time'].unique().astype(np.int64))
        by={typ:np.sort(faults.loc[(faults.tool_id==tool)&
            (faults.fault_type==typ)&(faults.time<item['validation_end_exclusive_second']),'time'].unique().astype(np.int64))
            for typ in TYPES}
        for rule,max_gap in [('B_operating_axis',None),*((f'C_gap_{v}s',v) for v in thresholds)]:
            rows,_=windows_for_rule(t,phase,by,all_faults,tool,rule,max_gap,rolling_max)
            allrows.extend(rows)
        print('pass2',tool,'windows cumulative',len(allrows),flush=True)
    allframe=pd.DataFrame(allrows)
    allframe.to_csv(OUT/'window_metadata.csv.gz',index=False,compression='gzip')
    total_faults=len(faults)
    summary=[];eventrows=[];durations=[]
    for rule,g in allframe.groupby('rule',sort=False):
        unique=set()
        counts={}
        for typ in TYPES:
            ids=set(g[f'event_{typ}'].dropna())
            unique.update(ids)
            counts[typ]=len(ids)
            for eid in ids:
                eventrows.append({'rule':rule,'event_id':eid,'tool_id':eid.split('|')[0],
                                  'fault_type':typ,'fault_time':int(eid.split('|')[-1])})
        train=g.phase.eq('train').sum();val=g.phase.eq('validation').sum()
        summary.append({'rule':rule,'total_sequences':len(g),'train_sequences':int(train),
            'validation_sequences':int(val),'unique_fault_events':len(unique),
            'of_1211_pct':100*len(unique)/total_faults,
            **{f'{typ}_events':counts[typ] for typ in TYPES}})
        for phase,sub in g.groupby('phase'):
            dur=sub.elapsed_seconds.to_numpy()/3600
            durations.append({'rule':rule,'phase':phase,'windows':len(dur),
                'median_h':float(np.median(dur)),'p25_h':float(np.quantile(dur,.25)),
                'p75_h':float(np.quantile(dur,.75)),'p90_h':float(np.quantile(dur,.9)),
                'min_h':float(dur.min()),'max_h':float(dur.max()),
                'over_24h_pct':float(np.mean(dur>24)*100)})
    pd.DataFrame(summary).to_csv(OUT/'rule_coverage.csv',index=False)
    pd.DataFrame(eventrows).to_csv(OUT/'covered_fault_events.csv',index=False)
    pd.DataFrame(durations).to_csv(OUT/'window_elapsed_distribution.csv',index=False)
    config={'model_training':False,'real_data_only':True,'window_observations':LENGTH,
        'overlap_observations':500,'stride_observations':STRIDE,
        'input_selection':'FSP=1 raw observations; no interpolation or synthetic rows',
        'A':'reused frozen strict 12s segments and within-segment windows',
        'B':'operating-index axis split at Tool, frozen Train/Validation phase, and any fault; gaps retained as metadata',
        'C':'same as B plus gap threshold sensitivity; thresholds in seconds',
        'C_thresholds':thresholds,'target':'next same-type fault within same split and after window start; no event crosses a split',
        'reserved_test':'no sensor rows at or after validation_end read',
        'denominator_1211':'all deduplicated fault log; numerator uses pre-reserved-Test Train/Validation only'}
    (OUT/'audit_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['gaps','coverage'])
    p.add_argument('--gap-seconds',nargs='*',type=int,default=[])
    args=p.parse_args()
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    faults=pd.read_csv(FAULTS).drop_duplicates(['tool_id','time','fault_type'])
    if len(faults)!=1211:raise AssertionError('Fault population changed')
    if args.phase=='gaps':pass1(plan,faults)
    else:
        if not args.gap_seconds:raise ValueError('Choose C sensitivity thresholds from saved gap histogram')
        pass2(plan,faults,args.gap_seconds)


if __name__=='__main__':main()
