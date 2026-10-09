"""Reconcile row hits, alarm starts and manager notices; true first/last alert QA."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from evaluate_model1_triggered_rul import (OUT,SOURCE,SCORES,RAW,PLAN,
    select_manager_notifications,metric)

def main():
    if (OUT/'policy_detection_reconciliation.csv').exists():
        raise FileExistsError(OUT/'policy_detection_reconciliation.csv')
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    with np.load(SCORES) as z:
        t=z['time'];code=z['tool_code'];p=z['probability']
    faults=pd.read_csv(RAW/'fault_events.csv').drop_duplicates(['tool_id','time','fault_type'])
    records=[]
    for i,item in enumerate(plan['tools']):
        tool=item['tool_id'];positions=np.flatnonzero(code==i)
        hits,starts,notices=select_manager_notifications(t,p,positions)
        for boundary_name,end in [('RUL_exact_validation_end',item['validation_end_exclusive_second']),
                                  ('Model1_report_until_test_start',item['test_start_second'])]:
            log=faults[(faults.tool_id==tool)&(faults.time>=item['validation_start_second'])&
                       (faults.time<end)]
            for e in log.itertuples():
                f=int(e.time)
                row={'boundary':boundary_name,'tool_id':tool,'fault_event_id':int(e.fault_event_id),
                     'fault_type':e.fault_type,'fault_time':f}
                for name,indices in [('row_hit',hits),('raw_alarm_start',starts),('manager_notification',notices)]:
                    tt=t[indices];j=np.searchsorted(tt,f-86400,side='left')
                    row[name]=bool(j<len(tt) and tt[j]<f)
                records.append(row)
    df=pd.DataFrame(records)
    df.to_csv(OUT/'policy_detection_reconciliation.csv',index=False)
    totals=[]
    for scope,part in df.groupby('boundary'):
        totals.append({'boundary':scope,'fault_events':len(part),
          **{name+'_detected_events':int(part[name].sum()) for name in
             ('row_hit','raw_alarm_start','manager_notification')}})
    pd.DataFrame(totals).to_csv(OUT/'policy_detection_summary.csv',index=False)
    first=pd.read_csv(OUT/'first_actual_notification_by_event.csv')
    last=pd.read_csv(OUT/'last_actual_notification_by_event.csv')
    rows=[]
    for scope,frame in [('true_first_notification',first),('true_last_notification',last)]:
        valid=frame[frame.status=='A_evaluable']
        for model,col in [('Raw_GRU','raw_gru_rul_h'),('Log_GRU','log_gru_rul_h')]:
            rows.append({'scope':scope,'total_events_with_notification':len(frame),
              'history_available_events':len(valid),'model':model,
              **metric(valid.actual_ttf_h.to_numpy(),valid[col].to_numpy(),
                       valid.next_fault_event_id.to_numpy())})
    pd.DataFrame(rows).to_csv(OUT/'true_first_last_metrics.csv',index=False)
    both=first[first.status=='A_evaluable'].merge(last[last.status=='A_evaluable'],
            on='next_fault_event_id',suffixes=('_first','_last'))
    both=both[both.model1_time_last>both.model1_time_first]
    summary=[]
    for model,col in [('Raw_GRU','raw_gru_rul_h'),('Log_GRU','log_gru_rul_h')]:
        fe=np.abs(both[f'{col}_first']-both.actual_ttf_h_first)
        le=np.abs(both[f'{col}_last']-both.actual_ttf_h_last)
        elapsed=(both.model1_time_last-both.model1_time_first)/3600
        drop=both[f'{col}_first']-both[f'{col}_last']
        summary.append({'model':model,'both_true_first_last_evaluable_events':len(both),
          'first_mae_h':float(fe.mean()),'last_mae_h':float(le.mean()),
          'last_error_lower_pct':float(100*(le<fe).mean()),
          'median_elapsed_h':float(elapsed.median()),
          'median_predicted_drop_h':float(drop.median())})
    pd.DataFrame(summary).to_csv(OUT/'true_first_last_paired_summary.csv',index=False)
    print(pd.DataFrame(totals).to_string(index=False),flush=True)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':main()
