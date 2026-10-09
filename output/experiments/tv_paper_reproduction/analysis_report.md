# TV et al. 2018 RUL preprocessing reproduction: development result

Source: [TV et al., *Recurrent Neural Networks for Online Remaining Useful Life Estimation in Ion Mill Etching System*](https://papers.phmsociety.org/index.php/phmconf/article/download/589/phmc_18_589), §§5.2, 6.1–6.2. The paper specifies FSP=1, nine sensors, 2,000-point windows/500 overlap, d1=2, d2=10 T2V, r_u=500, random distant-window sampling, and 2-layer LSTM. It does **not** give the distant sampling ratio. Its RUL display rule [100,150] is checked separately; it is not used to select validation samples for metrics. This is a paper-style adaptation to our frozen forward split, not an official challenge score.

## Unit audit before fitting

The first 01_M01 low-pressure example has sensor time 3,283,106, next time 3,283,110, TTF 3,250,078 then 3,250,074, and first logged low-pressure fault time 6,533,184. Therefore TTF equals fault time minus observation time exactly, and both advance by four per four-second measurement. The paper uses the supplied challenge TTF values, so r_u=500 corresponds to **500 dataset seconds = 8 min 20 sec = 0.138889 hours**. The paper does not explicitly call the anonymized time unit SI seconds; this correspondence is established by this dataset's four-second cadence and TTF arithmetic. Our previous Rule B GRU used **hours** and unbounded RUL. The 500 cap is therefore not 500 hours. Fault time marks operator shutdown, not verified physiological failure onset.

## Reproduction choices and coverage

Rule B uses the audited real operating windows only. No downtime interpolation, Model 1 retraining, engineered 85 features, Wu residual, or time metadata as a model feature. Shapes logged during extraction: raw `(2000,9)` → `d1=2: (1000,9)` → `d2=10: (100,90)`. The 100×9 subsample is a separate ablation; the T2V path preserves all 1,000 downsampled sensor points by concatenating consecutive groups of ten.

There are 19,053 finite-sensor and at-least-one-target windows, of which 15,076 are Train and 3,977 Validation. Train has **6** windows with any known RUL <500 seconds and 15,070 distant windows. All six near windows were retained, and 60 distant windows were sampled with seed 42, leaving **66 Train windows**. The 10:1 distant:near ratio is our fixed adaptation; the paper did not publish its ratio. Validation was not sampled. Train near counts by type: {'pressure_low': 5, 'pressure_high': 0, 'leak': 1}; Validation near counts: {'pressure_low': 3, 'pressure_high': 0, 'leak': 0}. Thus the high-pressure head has **zero near examples in both Train and Validation** and the leak head has zero near examples in Validation. After clipping, almost every target is exactly 500 seconds. A low clipped MAE can therefore be achieved by the constant-500 Dummy; it is not proof of useful RUL prediction.

Hidden size 50 was chosen before viewing validation because it is the smallest of the paper's 50/100/150/200 candidates. No hidden-size sweep used Validation. Models have two LSTM layers, dropout 0.4, Adam 0.005, batch 32 and five fixed epochs. There is no claim to reproduce the paper's up-to-2500-iteration early stopping. All three ablations use the **same 66 sampled Train and full Validation windows** and identical network settings; the only differences are subsampling vs T2V and unclipped vs clipped targets. Targets missing for a fault type are masked from loss. Prior B GRU is loaded only for same-window context and was trained on many more Train windows.

Actually represented fault events after Train sampling versus unsampled Validation:

```text
     phase    fault_type  sampled_windows_with_target  unique_fault_events
     train  pressure_low                           49                   24
     train pressure_high                           47                   22
     train          leak                           33                   14
validation  pressure_low                         3297                  116
validation pressure_high                         1811                   30
validation          leak                         1117                   10
```

The target is computed from the next logged same-type fault inside the frozen split; the published challenge's supplied TTF files may additionally censor values outside their specified horizon. Thus our target availability and exact window count need not match the paper's approximately 0.1M windows. This is a documented adaptation, not an undisclosed reproduction claim.

## Validation metrics

Units below are **seconds**. For unclipped variants, error is against uncapped actual RUL. For clipped variants, error is against `min(actual RUL,500)` and normalized MAE/RMSE divide by 500; these are different target questions and cannot be compared as like-for-like absolute RUL accuracy.

```text
          condition    fault_type  samples  events         mae        rmse      r2  smape_pct  event_equal_mae_seconds  near_samples  near_mae_seconds  normalized_mae
subsample_unclipped  pressure_low     3297     116 1651434.375 2180445.750  -1.345    195.861               306263.656             3          1454.284        3302.869
subsample_unclipped pressure_high     1811      30 1859778.875 2330243.500  -1.755    197.824               902629.688             0               NaN        3719.558
subsample_unclipped          leak     1117      10 1829017.125 2418801.500  -1.335    197.719              1006493.000             0               NaN        3658.034
      t2v_unclipped  pressure_low     3297     116 1651106.250 2180190.000  -1.345    195.314               305954.094             3          1757.934        3302.212
      t2v_unclipped pressure_high     1811      30 1860393.250 2330732.500  -1.756    198.297               903238.438             0               NaN        3720.787
      t2v_unclipped          leak     1117      10 1829184.000 2418939.000  -1.336    197.884              1006651.188             0               NaN        3658.368
        t2v_clipped  pressure_low     3297     116      85.546      92.949 -92.389     19.244                   82.623             3           201.337           0.171
        t2v_clipped pressure_high     1811      30      66.022      86.671   0.000     15.532                   64.359             0               NaN           0.132
        t2v_clipped          leak     1117      10      21.293      27.279   0.000      4.414                   22.432             0               NaN           0.043
    dummy_unclipped  pressure_low     3297     116 1305796.000 1489345.500  -0.094     84.805              1898554.375             3       2089940.000        2611.592
    dummy_unclipped pressure_high     1811      30 2247530.000 2537779.000  -2.267     90.367              3104608.750             0               NaN        4495.060
    dummy_unclipped          leak     1117      10 2479475.000 2837757.000  -2.214     99.166              3230080.500             0               NaN        4958.950
      dummy_clipped  pressure_low     3297     116       0.244       9.621  -0.001      0.081                    0.637             3           268.000           0.000
      dummy_clipped pressure_high     1811      30       0.000       0.000   1.000      0.000                    0.000             0               NaN           0.000
      dummy_clipped          leak     1117      10       0.000       0.000   1.000      0.000                    0.000             0               NaN           0.000
```

Fault-type macro summaries:

```text
          condition  macro_mae_seconds  macro_event_mae_seconds  macro_smape_pct
      dummy_clipped              0.081                    0.212            0.027
    dummy_unclipped        2010933.667              2744414.542           91.446
subsample_unclipped        1780076.792               738462.115          197.135
        t2v_clipped             57.620                   56.471           13.063
      t2v_unclipped        1780227.833               738614.573          197.165
```

The 100×90 T2V **unclipped** model has nearly the same very poor error as the 100×9 **unclipped** model on the same samples. The clipped model yields a much smaller capped-target MAE but loses against the 500-second Dummy for all types. High and leak clipped validation targets are constant, making their R² values degenerate; they must not be interpreted as forecasting skill. On the three near low-pressure windows, clipped T2V still has substantial error. For uncapped real RUL, the previous GRU trained on all 15,076 Train windows is context only (`prior_gru_same_validation_comparison.csv`), not an isolated T2V effect.

Time-ordered thirds of Validation (`fold_metrics.csv`) and Tool-level errors (`tool_metrics.csv`) are descriptive slices of the same fitted models; no forward refitting took place. The reserved Test partition was not opened. The paper's [100,150]-second output gate is in `paper_100_150_inference_gate_sensitivity.csv`; it cannot turn conditional shown-only accuracy into all-row performance. The paper's challenge S1/S2 involve NaN submission decisions and a different challenge split, so its published scores are not directly comparable with these MAE/RMSE results.

## Decision

The exact-unit, paper-style cap is only **8 min 20 sec**, while this project seeks hours-scale early warning. With six near Train windows, none for high pressure, the experiment does not establish an RUL model suitable for dashboard Model 2. The visible reduction in capped MAE is principally target saturation; T2V alone did not improve the uncapped target. Preserve this result as an honest reproduction attempt and use new later labeled data or a differently justified decision horizon before claiming a useful time-to-failure estimator.
