"""Verify challenge TTF/timestamp units before paper-style target clipping."""
from pathlib import Path
import csv
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/experiments/tv_paper_reproduction'
FAULT=ROOT/'data/processed/fault_events.csv'
TTF=ROOT/'data/train/train_ttf/01_M01_DC_train_ttf.csv'

def main():
    if OUT.exists():raise FileExistsError('Refusing to overwrite TV experiment')
    with TTF.open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle)
        a=next(reader);b=next(reader)
    fault=pd.read_csv(FAULT)
    t0=int(a['time']);t1=int(b['time'])
    col='TTF_FlowCool Pressure Dropped Below Limit'
    v0=float(a[col]);v1=float(b[col])
    ft=int(fault.loc[(fault.tool_id=='01_M01')&(fault.fault_type=='pressure_low'),'time'].min())
    if not (t1-t0==4 and v0-v1==4 and ft-t0==v0):
        raise AssertionError('Raw timestamp and TTF units did not reconcile')
    OUT.mkdir(parents=True)
    report={'raw_timestamp_unit':'seconds on challenge operating time axis',
      'evidence':{'tool':'01_M01','time_0':t0,'time_1':t1,'ttf_0':v0,'ttf_1':v1,
                  'first_low_fault_time':ft,'fault_minus_observation':ft-t0},
      'project_prior_target_unit':'hours = (fault_time - sequence_end_time)/3600',
      'paper_rul_unit':'same challenge timestamp/TTF unit (seconds); paper does not explicitly name SI seconds, but uses the provided challenge TTF values',
      'paper_ru_raw_units':500,'paper_ru_hours':500/3600,
      'source':'https://papers.phmsociety.org/index.php/phmconf/article/download/589/phmc_18_589',
      'caveat':'The fault timestamp is the reported operator shutdown, not necessarily the physical onset of failure.'}
    (OUT/'unit_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
