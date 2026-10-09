"""Read-only audit of source fault coverage and warning-window attrition."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/experiments/degradation_warning_rul'
PLAN=ROOT/'output/experiments/forward_validation_ab_24h/split_plan.json'
FAULTS=ROOT/'data/processed/fault_events.csv'
OLD=ROOT/'output/experiments/rul_paper_reproduction_baseline/fault_event_sequence_availability.csv'
TYPES=('pressure_low','pressure_high','leak')


def main():
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    faults=pd.read_csv(FAULTS).drop_duplicates(['tool_id','time','fault_type'])
    availability=pd.read_csv(OLD)
    w=pd.read_csv(OUT/'warning_events.csv')
    p=pd.read_csv(OUT/'detector_predictions.csv.gz')
    rows=[]
    for typ in TYPES:
        logs=[]
        for item in plan['tools']:
            f=faults[(faults.tool_id==item['tool_id'])&(faults.fault_type==typ)&
                 (faults.time>=item['validation_start_second'])&
                 (faults.time<item['validation_end_exclusive_second'])]
            logs.extend(zip(f.tool_id,f.time))
        all_events=set(logs)
        a=availability[availability.fault_type==typ]
        a_events={(r.tool_id,int(r.fault_time)) for r in a.itertuples() if r.has_2000_operating_obs}
        available=all_events&a_events
        v=w[(w.phase=='validation')&(w.fault_type==typ)]
        window={(r.tool_id,int(r.event_id.split('|')[-1])) for r in v.itertuples()}
        warned={(r.tool_id,int(r.event_id.split('|')[-1])) for r in v.itertuples() if r.warned}
        rows.append({'fault_type':typ,'logged_validation_faults':len(all_events),
                     'has_2000_operating_obs_since_previous_type':len(available),
                     'has_eligible_sparse_window':len(window),
                     'warned':len(warned),
                     'lost_before_2000_obs':len(all_events-available),
                     'lost_after_2000_obs_before_sparse_window':len(available-window),
                     'sparse_window_no_warning':len(window-warned),
                     'non_labelled_validation_window_rows':int(((p.phase=='validation')&p[f'label_{typ}'].isna()).sum()),
                     'median_window_max_operating_gap_s':float(p.loc[p.phase=='validation','max_operating_gap_s'].median())})
    report=pd.DataFrame(rows)
    report.to_csv(OUT/'coverage_audit.csv',index=False)
    # All source detector inputs are named raw sensors in the source file. These
    # saved scores cannot alone prove correct online time alignment; flag design.
    observations={'validation_window_rows':int((p.phase=='validation').sum()),
        'detector_train_window_rows':int((p.phase=='detector_train').sum()),
        'rul_train_window_rows':int((p.phase=='rul_train').sum()),
        'large_gap_validation_windows_gt90s':int(((p.phase=='validation')&(p.max_operating_gap_s>90)).sum()),
        'fraction_large_gap_validation_windows':float(((p.phase=='validation')&(p.max_operating_gap_s>90)).sum()/(p.phase=='validation').sum()),
        'warning_stride_operating_observations':1500,
        'nominal_warning_stride_minutes_at_4s':100,
        'known_exclusions':['2000 operating observations required','1500-observation stride',
          '24h split embargo and internal Train split','phase-end 24h detector-label censoring',
          'RUL only after 0.5 warning'],'leakage_status':'Static source-code audit finds no TTF/fault/time model input and no validation fit; full raw timestamp replay was not independently recomputed.'}
    (OUT/'coverage_audit_notes.json').write_text(json.dumps(observations,indent=2),encoding='utf-8')
    print(report.to_string(index=False));print(json.dumps(observations,indent=2))


if __name__=='__main__':main()
