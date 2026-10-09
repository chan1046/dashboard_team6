"""Summarize the saved, unretrained degradation-warning experiment."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/experiments/degradation_warning_rul'
PLAN = ROOT / 'output/experiments/forward_validation_ab_24h/split_plan.json'
FAULTS = ROOT / 'data/processed/fault_events.csv'
TYPES = ('pressure_low','pressure_high','leak')


def main():
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    faults=pd.read_csv(FAULTS).drop_duplicates(['tool_id','time','fault_type'])
    pred=pd.read_csv(OUT/'detector_predictions.csv.gz')
    warnings=pd.read_csv(OUT/'warning_events.csv')
    metrics=pd.read_csv(OUT/'rul_metrics.csv')
    result=[];false_rows=[]
    for typ in TYPES:
        actual=0
        for item in plan['tools']:
            actual += len(faults[(faults.tool_id==item['tool_id']) & (faults.fault_type==typ) &
                (faults.time>=item['validation_start_second']) &
                (faults.time<item['validation_end_exclusive_second'])])
        w=warnings[(warnings.fault_type==typ)&(warnings.phase=='validation')]
        normal=pred[(pred.phase=='validation')&(pred[f'label_{typ}']==0)&
                    (pred[f'warning_{typ}']==True)].copy()
        normal=normal.sort_values(['tool_id','time'])
        normal['new_episode']=(normal.tool_id.ne(normal.tool_id.shift())|
                               normal.time.sub(normal.time.shift()).gt(600))
        episodes=int(normal.new_episode.sum())
        false_rows.append({'fault_type':typ,'false_warning_rows':len(normal),
                           'false_warning_episodes_10m':episodes})
        result.append({'fault_type':typ,'all_logged_validation_faults':actual,
            'fault_events_with_eligible_history':len(w),
            'warned_fault_events':int(w.warned.sum()),
            'event_recall_all':float(w.warned.sum()/actual) if actual else None,
            'event_recall_eligible':float(w.warned.mean()) if len(w) else None,
            'lead_median_h':float(w.lead_time_h.median()) if w.warned.any() else None,
            'lead_min_h':float(w.lead_time_h.min()) if w.warned.any() else None,
            'lead_max_h':float(w.lead_time_h.max()) if w.warned.any() else None,
            'false_warning_rows':len(normal),'false_warning_episodes_10m':episodes})
    pd.DataFrame(result).to_csv(OUT/'event_summary.csv',index=False)
    pd.DataFrame(false_rows).to_csv(OUT/'false_warning_summary.csv',index=False)
    # The two arms share exactly the same validation examples by construction.
    # Fold slices are score reporting only; models were not refit by fold.
    ab=metrics.pivot_table(index=['fault_type','model'],columns='arm',values='mae_h')
    ab['mae_change_after_minus_without_h']=ab.get('after_warning')-ab.get('without_warning')
    ab.to_csv(OUT/'matched_ablation_mae.csv')
    caveats={
        'degradation_label':'24h pre-fault proxy, not measured onset',
        'selection':'RUL validation rows are restricted to warned faults, so results do not measure all-fault RUL quality',
        'ablation':'both training arms scored on identical post-warning validation rows; training populations differ by design',
        'folds':'three chronological validation event slices of a single fitted model; not three refits',
        'comparison_to_old':'old raw-sequence baseline used the whole Train partition and a different validation row population; numerical MAE is not directly comparable',
        'horizon':'detector label is censored in last 24h of each partition',
        'status':'development validation previously reused across project; no untouched final test claim'}
    (OUT/'interpretation.json').write_text(json.dumps(caveats,indent=2),encoding='utf-8')
    print(pd.DataFrame(result).to_string(index=False))


if __name__=='__main__':main()
