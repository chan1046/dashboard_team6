"""Produce audit comparisons for TV reproduction without retraining."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/experiments/tv_paper_reproduction'
OLD=ROOT/'output/experiments/rul_rule_b_baseline'
TYPES=['pressure_low','pressure_high','leak']

def main():
    cfg=json.loads((OUT/'experiment_config.json').read_text(encoding='utf-8'))
    units=json.loads((OUT/'unit_audit.json').read_text(encoding='utf-8'))
    m=pd.read_csv(OUT/'metrics.csv')
    p=pd.read_csv(OUT/'validation_predictions.csv.gz')
    old=pd.read_csv(OLD/'validation_predictions.csv.gz')
    old=old.loc[old.condition.eq('B_raw9')].copy()
    old['actual_seconds']=old.actual_ttf_h*3600
    old['predicted_seconds']=old.predicted_ttf_h*3600
    old['fault_type']=old.fault_type.astype(str)
    # Exact same Validation windows, identified by Tool/start/end/type/event.
    source=p.loc[p.condition.eq('t2v_unclipped'),['tool_id','start_time','time','fault_type','event_id','actual_seconds']]
    paired=source.merge(old[['tool_id','start_time','time','fault_type','event_id','predicted_seconds']],
       on=['tool_id','start_time','time','fault_type','event_id'],how='left',validate='one_to_one')
    if paired.predicted_seconds.isna().any():raise AssertionError('Previous GRU validation sample mismatch')
    old_metric=[]
    for typ,g in paired.groupby('fault_type'):
        error=np.abs(g.actual_seconds-g.predicted_seconds)
        old_metric.append({'condition':'prior_B_GRU_raw9_unclipped','fault_type':typ,
         'samples':len(g),'events':g.event_id.nunique(),'mae_seconds':error.mean(),
         'event_equal_mae_seconds':g.assign(error=error).groupby('event_id').error.mean().mean()})
    pd.DataFrame(old_metric).to_csv(OUT/'prior_gru_same_validation_comparison.csv',index=False)
    # Paper's post-hoc [100,150] raw-unit display gate. Never use this
    # validation-conditioned subset to claim an improved all-row MAE.
    gate=[]
    for typ,g in p.loc[p.condition.eq('t2v_clipped')].groupby('fault_type'):
        keep=g.predicted_seconds.between(100,150,inclusive='both')
        gate.append({'fault_type':typ,'validation_samples':len(g),'shown_samples':int(keep.sum()),
          'shown_pct':100*keep.mean(),'shown_events':g.loc[keep,'event_id'].nunique(),
          'paper_rule_seconds':'show only prediction in [100,150], otherwise NaN'})
    pd.DataFrame(gate).to_csv(OUT/'paper_100_150_inference_gate_sensitivity.csv',index=False)
    macro=m.groupby('condition').agg(macro_mae_seconds=('mae','mean'),
         macro_event_mae_seconds=('event_equal_mae_seconds','mean'),
         macro_smape_pct=('smape_pct','mean')).reset_index()
    macro.to_csv(OUT/'macro_metrics.csv',index=False)
    fold=pd.read_csv(OUT/'fold_metrics.csv')
    selected=pd.read_csv(OUT/'sampled_window_metadata.csv.gz')
    event_counts=[]
    for phase,g in selected.groupby('phase'):
        for typ in TYPES:
            event_counts.append({'phase':phase,'fault_type':typ,
              'sampled_windows_with_target':int(g[f'event_id_{typ}'].notna().sum()),
              'unique_fault_events':int(g[f'event_id_{typ}'].nunique())})
    ec=pd.DataFrame(event_counts)
    ec.to_csv(OUT/'sampled_train_validation_event_counts.csv',index=False)
    report=f'''# TV et al. 2018 RUL preprocessing reproduction: development result

Source: [TV et al., *Recurrent Neural Networks for Online Remaining Useful Life Estimation in Ion Mill Etching System*](https://papers.phmsociety.org/index.php/phmconf/article/download/589/phmc_18_589), §§5.2, 6.1–6.2. The paper specifies FSP=1, nine sensors, 2,000-point windows/500 overlap, d1=2, d2=10 T2V, r_u=500, random distant-window sampling, and 2-layer LSTM. It does **not** give the distant sampling ratio. Its RUL display rule [100,150] is checked separately; it is not used to select validation samples for metrics. This is a paper-style adaptation to our frozen forward split, not an official challenge score.

## Unit audit before fitting

The first 01_M01 low-pressure example has sensor time 3,283,106, next time 3,283,110, TTF 3,250,078 then 3,250,074, and first logged low-pressure fault time 6,533,184. Therefore TTF equals fault time minus observation time exactly, and both advance by four per four-second measurement. The paper uses the supplied challenge TTF values, so r_u=500 corresponds to **500 dataset seconds = 8 min 20 sec = 0.138889 hours**. The paper does not explicitly call the anonymized time unit SI seconds; this correspondence is established by this dataset's four-second cadence and TTF arithmetic. Our previous Rule B GRU used **hours** and unbounded RUL. The 500 cap is therefore not 500 hours. Fault time marks operator shutdown, not verified physiological failure onset.

## Reproduction choices and coverage

Rule B uses the audited real operating windows only. No downtime interpolation, Model 1 retraining, engineered 85 features, Wu residual, or time metadata as a model feature. Shapes logged during extraction: raw `(2000,9)` → `d1=2: (1000,9)` → `d2=10: (100,90)`. The 100×9 subsample is a separate ablation; the T2V path preserves all 1,000 downsampled sensor points by concatenating consecutive groups of ten.

There are {cfg['sampling']['all_rule_b_usable_windows']:,} finite-sensor and at-least-one-target windows, of which {cfg['sampling']['train_all']:,} are Train and {cfg['sampling']['validation_all_unsampled']:,} Validation. Train has **{cfg['sampling']['train_near_any_type']}** windows with any known RUL <500 seconds and {cfg['sampling']['train_distant_all']:,} distant windows. All six near windows were retained, and {cfg['sampling']['train_distant_sampled']} distant windows were sampled with seed 42, leaving **{cfg['sampling']['train_sampled_total']} Train windows**. The 10:1 distant:near ratio is our fixed adaptation; the paper did not publish its ratio. Validation was not sampled. Train near counts by type: {cfg['sampling']['train_near_per_type']}; Validation near counts: {cfg['sampling']['validation_near_per_type']}. Thus the high-pressure head has **zero near examples in both Train and Validation** and the leak head has zero near examples in Validation. After clipping, almost every target is exactly 500 seconds. A low clipped MAE can therefore be achieved by the constant-500 Dummy; it is not proof of useful RUL prediction.

Hidden size 50 was chosen before viewing validation because it is the smallest of the paper's 50/100/150/200 candidates. No hidden-size sweep used Validation. Models have two LSTM layers, dropout 0.4, Adam 0.005, batch 32 and five fixed epochs. There is no claim to reproduce the paper's up-to-2500-iteration early stopping. All three ablations use the **same 66 sampled Train and full Validation windows** and identical network settings; the only differences are subsampling vs T2V and unclipped vs clipped targets. Targets missing for a fault type are masked from loss. Prior B GRU is loaded only for same-window context and was trained on many more Train windows.

Actually represented fault events after Train sampling versus unsampled Validation:

```text
{ec.to_string(index=False)}
```

The target is computed from the next logged same-type fault inside the frozen split; the published challenge's supplied TTF files may additionally censor values outside their specified horizon. Thus our target availability and exact window count need not match the paper's approximately 0.1M windows. This is a documented adaptation, not an undisclosed reproduction claim.

## Validation metrics

Units below are **seconds**. For unclipped variants, error is against uncapped actual RUL. For clipped variants, error is against `min(actual RUL,500)` and normalized MAE/RMSE divide by 500; these are different target questions and cannot be compared as like-for-like absolute RUL accuracy.

```text
{m[['condition','fault_type','samples','events','mae','rmse','r2','smape_pct','event_equal_mae_seconds','near_samples','near_mae_seconds','normalized_mae']].round(3).to_string(index=False)}
```

Fault-type macro summaries:

```text
{macro.round(3).to_string(index=False)}
```

The 100×90 T2V **unclipped** model has nearly the same very poor error as the 100×9 **unclipped** model on the same samples. The clipped model yields a much smaller capped-target MAE but loses against the 500-second Dummy for all types. High and leak clipped validation targets are constant, making their R² values degenerate; they must not be interpreted as forecasting skill. On the three near low-pressure windows, clipped T2V still has substantial error. For uncapped real RUL, the previous GRU trained on all 15,076 Train windows is context only (`prior_gru_same_validation_comparison.csv`), not an isolated T2V effect.

Time-ordered thirds of Validation (`fold_metrics.csv`) and Tool-level errors (`tool_metrics.csv`) are descriptive slices of the same fitted models; no forward refitting took place. The reserved Test partition was not opened. The paper's [100,150]-second output gate is in `paper_100_150_inference_gate_sensitivity.csv`; it cannot turn conditional shown-only accuracy into all-row performance. The paper's challenge S1/S2 involve NaN submission decisions and a different challenge split, so its published scores are not directly comparable with these MAE/RMSE results.

## Decision

The exact-unit, paper-style cap is only **8 min 20 sec**, while this project seeks hours-scale early warning. With six near Train windows, none for high pressure, the experiment does not establish an RUL model suitable for dashboard Model 2. The visible reduction in capped MAE is principally target saturation; T2V alone did not improve the uncapped target. Preserve this result as an honest reproduction attempt and use new later labeled data or a differently justified decision horizon before claiming a useful time-to-failure estimator.
'''
    (OUT/'analysis_report.md').write_text(report,encoding='utf-8')
    print(macro.to_string(index=False))
    print(pd.DataFrame(gate).to_string(index=False))

if __name__=='__main__':main()
