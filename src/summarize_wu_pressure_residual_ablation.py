"""Summarize existing Wu ablation outputs without fitting or retuning."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/experiments/wu_pressure_residual_ablation'
TYPES=('pressure_low','pressure_high','leak')


def main():
    metrics=pd.read_csv(OUT/'rul_metrics.csv')
    gru=metrics[metrics.model=='GRU']
    a=gru[gru.set=='A'].set_index('fault_type')
    b=gru[gru.set=='B'].set_index('fault_type')
    rows=[]
    for typ in TYPES:
        if typ not in a.index or typ not in b.index:
            rows.append({'fault_type':typ,'status':'no same-sample validation target'});continue
        x=a.loc[typ];y=b.loc[typ]
        rows.append({'fault_type':typ,'status':'scored but underpowered',
          'validation_events':int(x.events),'validation_rows':int(x.rows),
          'A_mae_h':x.mae_h,'B_mae_h':y.mae_h,'mae_reduction_pct':100*(x.mae_h-y.mae_h)/x.mae_h,
          'A_rmse_h':x.rmse_h,'B_rmse_h':y.rmse_h,'rmse_reduction_pct':100*(x.rmse_h-y.rmse_h)/x.rmse_h,
          'A_smape_pct':x.smape_pct,'B_smape_pct':y.smape_pct,
          'smape_reduction_pct':100*(x.smape_pct-y.smape_pct)/x.smape_pct,
          'A_event_equal_mae_h':x.event_equal_mae_h,'B_event_equal_mae_h':y.event_equal_mae_h})
    pd.DataFrame(rows).to_csv(OUT/'rul_ablation_comparison.csv',index=False)
    eda=pd.read_csv(OUT/'residual_eda_sample.csv.gz')
    event_rows=[]
    for typ in TYPES:
        dist=eda[f'next_{typ}_h'].to_numpy()
        valid=np.isfinite(dist)&(dist>0)&(dist<=24)
        sub=eda.loc[valid,['tool_id','time','residual']].copy()
        sub['dist_h']=dist[valid]
        sub['fault_time']=(sub.time.to_numpy()+np.rint(sub.dist_h.to_numpy()*3600).astype(np.int64))
        sub['abs_residual']=sub.residual.abs()
        for (tool,ft),g in sub.groupby(['tool_id','fault_time']):
            near=g.loc[g.dist_h<=1,'abs_residual']
            far=g.loc[(g.dist_h>12)&(g.dist_h<=24),'abs_residual']
            event_rows.append({'fault_type':typ,'tool_id':tool,'fault_time':ft,
              'near_0_1h_rows':len(near),'far_12_24h_rows':len(far),
              'near_abs_median':float(near.median()) if len(near) else np.nan,
              'far_abs_median':float(far.median()) if len(far) else np.nan,
              'near_minus_far_abs_median':float(near.median()-far.median()) if len(near) and len(far) else np.nan})
    events=pd.DataFrame(event_rows)
    events.to_csv(OUT/'residual_event_trends.csv',index=False)
    stability=[]
    for typ,g in events.groupby('fault_type'):
        paired=g.dropna(subset=['near_minus_far_abs_median'])
        stability.append({'fault_type':typ,'events_with_any_0_24h_sample':len(g),
          'events_with_both_0_1h_and_12_24h':len(paired),
          'increased_abs_residual_events':int((paired.near_minus_far_abs_median>0).sum()),
          'median_event_change':float(paired.near_minus_far_abs_median.median()) if len(paired) else np.nan})
    pd.DataFrame(stability).to_csv(OUT/'residual_event_trend_summary.csv',index=False)
    macro={'A_macro_mae_evaluable_types_h':float(a.mae_h.mean()),
           'B_macro_mae_evaluable_types_h':float(b.mae_h.mean()),
           'macro_fault_types':list(a.index),
           'warning':'Macro excludes leak because validation has zero eligible leak event; temporal folds cannot be meaningfully assessed with 1-2 events.'}
    (OUT/'macro_summary.json').write_text(json.dumps(macro,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(stability).to_string(index=False));print(macro)


if __name__=='__main__':main()
