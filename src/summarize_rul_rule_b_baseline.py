"""Summarize fixed Rule B/C GRU experiment without fitting a model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/experiments/rul_rule_b_baseline'
AUDIT=ROOT/'output/experiments/rul_sequence_rule_audit'
TYPES=['pressure_low','pressure_high','leak']

def main():
    m=pd.read_csv(OUT/'metrics.csv')
    pred=pd.read_csv(OUT/'validation_predictions.csv.gz')
    fold=pd.read_csv(OUT/'fold_metrics.csv')
    audit=pd.read_csv(AUDIT/'rule_coverage.csv')
    coverage=[]
    for rule,label in [('B_operating_axis','Rule B'),('C_gap_1800s','Rule C >30min')]:
        f=pd.read_csv(OUT/f'{rule}_sequence_metadata.csv.gz')
        long=[]
        for typ in TYPES:
            z=f.loc[f[f'event_id_{typ}'].notna(), ['tool_id','phase','start_time','time',
                'elapsed_seconds','max_internal_gap_s','large_gap_count_over_30min',
                f'event_id_{typ}',f'ttf_h_{typ}']].copy()
            z.columns=['tool_id','phase','sequence_start_time','sequence_end_time',
                'actual_elapsed_seconds','max_internal_gap_seconds',
                'large_gap_count_over_30min','fault_event_id','target_rul_hours']
            z['fault_type']=typ
            long.append(z)
        pd.concat(long,ignore_index=True).to_csv(
            OUT/f'{rule}_sequence_target_metadata_long.csv.gz',index=False,compression='gzip')
        for phase,g in f.groupby('phase'):
            ids=set()
            row={'rule':label,'phase':phase,'usable_sequences':len(g)}
            for typ in TYPES:
                ids.update(g[f'event_id_{typ}'].dropna())
                row[f'{typ}_events']=g[f'event_id_{typ}'].nunique()
                row[f'{typ}_target_samples']=g[f'ttf_h_{typ}'].notna().sum()
            row['unique_fault_events']=len(ids)
            coverage.append(row)
    cov=pd.DataFrame(coverage)
    cov.to_csv(OUT/'usable_coverage.csv',index=False)
    # The two B variants are paired by exact window. Rule C uses the same
    # event population where possible, but its shifted stride makes rows differ.
    subsets=[]
    b=pred.loc[pred.condition=='B_raw9']
    c=pred.loc[pred.condition=='C1800_raw9']
    for typ in TYPES:
        common=set(b.loc[b.fault_type==typ,'event_id']) & set(c.loc[c.fault_type==typ,'event_id'])
        for condition in ['B_raw9_Dummy','B_raw9','B_raw9_time','C1800_raw9_Dummy','C1800_raw9']:
            g=pred.loc[(pred.condition==condition)&(pred.fault_type==typ)&
                       pred.event_id.isin(common)]
            if g.empty:continue
            err=np.abs(g.actual_ttf_h-g.predicted_ttf_h)
            subsets.append({'condition':condition,'fault_type':typ,'common_events':len(common),
                'samples':len(g),'mae_h':err.mean(),
                'event_equal_mae_h':g.assign(error=err).groupby('event_id').error.mean().mean()})
    pd.DataFrame(subsets).to_csv(OUT/'common_event_subset_metrics.csv',index=False)
    macro=m.loc[~m.condition.str.endswith('_Dummy')].groupby('condition').agg(
        macro_mae_h=('mae_h','mean'),macro_event_mae_h=('event_equal_mae_h','mean'),
        macro_smape_pct=('smape_pct','mean')).reset_index()
    macro.to_csv(OUT/'macro_metrics.csv',index=False)
    # No fold refit: these are chronological thirds of the same Validation.
    strong=(m.loc[~m.condition.str.endswith('_Dummy'),'r2']>0).sum()
    report=f'''# Rule B/C raw-sensor GRU baseline (development validation)

All inputs are real FSP=1 observations from 20 Tools before the reserved Test boundary. Frozen per-Tool chronological Train/Validation split, at least 24h purge, and audit fault boundaries are retained. No Model 1 artifact or prior result was changed. No downtime row was filled. The 2,000-observation windows overlap by 500 observations. **To keep this fixed CPU baseline comparable with the earlier paper-inspired GRU, the network consumes every 20th observation (100 time steps) of each complete 2,000-point window.** Thus this is a Rule B windowing experiment, not a full-resolution 2,000-step paper reproduction. Time metadata summarizes the complete 2,000-point window and uses Train-only log1p median/IQR scaling. Raw sensor scaling also uses Train only. Event IDs, times, and TTF never enter the raw-9 input; the time-input variant adds only observed elapsed and max gap, not target time.

## Coverage

The time-only audit found B 25,310 windows/634 unique covered events and C >30min 9,833/485; strict A found 46/12. After excluding all-masked targets and nonfinite selected sensor observations, the actual usable counts are:

```text
{cov.to_string(index=False)}
```

Target is **next fault of the same type**, in hours from window end. Each of three heads is masked when the corresponding type's future fault is not observable within the same phase. Every window remains within one Tool, phase, and any-type fault interval. There is no target-time-based 0–6h screening. The full unbounded RUL target can therefore be hundreds of hours; hourly MAE is not comparable with an earlier true-TTF-selected 0–6h conditional experiment.

## Validation metrics (hours, unless indicated)

```text
{m[['condition','fault_type','validation_samples','validation_events','mae_h','rmse_h','r2','smape_pct','within_1h','within_2h','event_equal_mae_h']].round(3).to_string(index=False)}
```

Fault-type macro metrics:

```text
{macro.round(3).to_string(index=False)}
```

The B raw and B time models share identical Validation windows. Rule C uses the same frozen time interval but different windows. Their common-event subset is in `common_event_subset_metrics.csv`; it equalizes events, **not timestamps**, so it is sensitivity analysis rather than a paired row-level causal comparison. Tool metrics and validation chronological thirds are in `tool_metrics.csv` and `fold_metrics.csv`. These thirds are stability slices of one fitted model, **not separate forward retrain folds**. The Validation interval has been used repeatedly in this project; none of these are independent final Test numbers.

Only {strong} of 9 GRU type/condition R² values are positive. Absolute MAEs remain hundreds of hours and ±1h/±2h fractions are near zero. B time metadata does not consistently lower type-wise or event-wise MAE. C changes both coverage and window duration, so any C difference cannot be attributed solely to the gap cutoff on identical samples. The strict baseline covered too few validation events for a fair same-sample accuracy comparison; the defensible improvement is **coverage**, not RUL accuracy.

## Reproducibility and limitations

`experiment_config.json` records the rule, model architecture, target masking, scaler, split and model paths. `validation_predictions.csv.gz` contains the actual validation predictions, event IDs, timestamps and target values for rechecking metrics. The reserved Test region was not read. A future decisive experiment should compare full-resolution or explicitly time-aware sequence encoders on an independent later labeled period before a dashboard displays an expected failure time.
'''
    (OUT/'analysis_report.md').write_text(report,encoding='utf-8')
    print(cov.to_string(index=False))
    print(macro.to_string(index=False))

if __name__=='__main__':main()
