"""Keep Model 1 any-prealert detection distinct from next-fault RUL matching."""
from __future__ import annotations
import pandas as pd
from evaluate_model1_triggered_rul import OUT

def main():
    events=pd.read_csv(OUT/'policy_detection_reconciliation.csv')
    events=events[events.boundary=='RUL_exact_validation_end'].copy()
    alerts=pd.read_csv(OUT/'manager_notifications.csv.gz')
    fields=[]
    for event in events.itertuples():
        p=alerts[(alerts.tool_id==event.tool_id)&
                 (alerts.model1_time>=event.fault_time-86400)&
                 (alerts.model1_time<event.fault_time)]
        sufficient=p[p.status=='A_evaluable']
        matched=sufficient[sufficient.next_fault_event_id==event.fault_event_id]
        fields.append({'fault_event_id':event.fault_event_id,'tool_id':event.tool_id,
          'fault_type':event.fault_type,'fault_time':event.fault_time,
          'manager_prealert':len(p)>0,'prealert_with_history':len(sufficient)>0,
          'prealert_matched_to_this_next_fault':len(matched)>0,
          'prealert_count':len(p),'history_available_prealert_count':len(sufficient),
          'matched_next_fault_prealert_count':len(matched)})
    details=pd.DataFrame(fields)
    assert details.manager_prealert.sum()==events.manager_notification.sum()
    summary=[{'stage':'validation_fault_events','events':len(details),'fraction_of_all':1.0},
      {'stage':'Model1_any_manager_notification_in_pre24h','events':int(details.manager_prealert.sum()),
        'fraction_of_all':float(details.manager_prealert.mean())},
      {'stage':'pre24h_notification_with_rul_history','events':int(details.prealert_with_history.sum()),
        'fraction_of_all':float(details.prealert_with_history.mean())},
      {'stage':'pre24h_notification_next_fault_matched_and_rul_evaluable',
        'events':int(details.prealert_matched_to_this_next_fault.sum()),
        'fraction_of_all':float(details.prealert_matched_to_this_next_fault.mean())}]
    details.to_csv(OUT/'funnel_event_details.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'detection_to_rul_funnel.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False))

if __name__=='__main__':main()
