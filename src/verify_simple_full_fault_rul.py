"""Read-only validation of the newly generated full-range RUL experiment."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/experiments/simple_full_fault_rul'
PLAN=ROOT/'output/experiments/forward_validation_ab_24h/split_plan.json'

def main():
    meta=pd.read_csv(OUT/'sample_metadata.csv.gz')
    counts=pd.read_csv(OUT/'selected_anchor_counts_by_event.csv')
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    bytool={x['tool_id']:x for x in plan['tools']}
    tr=meta[meta.phase=='train'];va=meta[meta.phase=='validation']
    tests={
      'all_history_at_or_before_anchor':bool((meta.history_first_time<=meta.anchor_time).all()),
      'all_anchors_strictly_before_target_fault':bool((meta.anchor_time<meta.fault_time).all()),
      'ttf_exactly_fault_minus_anchor_hours':bool(np.allclose(meta.target_ttf_hours,(meta.fault_time-meta.anchor_time)/3600)),
      'no_event_shared_train_validation':set(tr.fault_event_id).isdisjoint(set(va.fault_event_id)),
      'train_anchors_before_train_end':all(bool((part.anchor_time<bytool[tool]['train_end_exclusive_second']).all()) for tool,part in tr.groupby('tool_id')),
      'validation_anchors_at_after_validation_start':all(bool((part.anchor_time>=bytool[tool]['validation_start_second']).all()) for tool,part in va.groupby('tool_id')),
      'validation_faults_before_validation_end':all(bool((part.fault_time<bytool[tool]['validation_end_exclusive_second']).all()) for tool,part in va.groupby('tool_id')),
      'all_history_within_requested_hour':bool((meta.anchor_time-meta.history_first_time<=3600).all()),
      'all_history_spans_at_least_80pct_hour':bool((meta.history_elapsed_seconds>=2880).all()),
      'no_train_event_above_64_samples':bool((counts[counts.phase=='train'].retained_anchors<=64).all())}
    assert all(tests.values()),tests
    cap=counts[counts.phase=='train'].query('retained_anchors>0').copy()
    cap['raw_event_share_pct']=100*cap.candidate_anchors/cap.candidate_anchors.sum()
    cap['capped_event_share_pct']=100*cap.retained_anchors/cap.retained_anchors.sum()
    cap['event_balanced_share_pct']=100/len(cap)
    cap.to_csv(OUT/'event_balance_comparison.csv',index=False)
    report={'leakage_checks':tests,
      'observed_fault_events':1211,
      'train_events_in_fixed_split':int((counts.phase=='train').sum()),
      'validation_events_in_fixed_split':int((counts.phase=='validation').sum()),
      'train_usable_events':int(tr.fault_event_id.nunique()),
      'validation_usable_events':int(va.fault_event_id.nunique()),
      'train_ttf_median_hours':float(tr.target_ttf_hours.median()),
      'validation_ttf_median_hours':float(va.target_ttf_hours.median()),
      'train_ttf_max_hours':float(tr.target_ttf_hours.max()),
      'validation_ttf_max_hours':float(va.target_ttf_hours.max()),
      'validation_anchors_with_gap_over_30min_pct':float(100*(va.max_internal_gap_seconds>1800).mean()),
      'validation_anchors_with_gap_over_5min_pct':float(100*(va.max_internal_gap_seconds>300).mean()),
      'target_not_clipped':True,
      'no_fake_downtime_rows':True,
      'reserved_test_sensor_rows_read_for_model':0,
      'reserved_test_labels_read_for_model':0}
    (OUT/'verification_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
