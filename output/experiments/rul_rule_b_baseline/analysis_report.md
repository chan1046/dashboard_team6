# Rule B/C raw-sensor GRU baseline (development validation)

All inputs are real FSP=1 observations from 20 Tools before the reserved Test boundary. Frozen per-Tool chronological Train/Validation split, at least 24h purge, and audit fault boundaries are retained. No Model 1 artifact or prior result was changed. No downtime row was filled. The 2,000-observation windows overlap by 500 observations. **To keep this fixed CPU baseline comparable with the earlier paper-inspired GRU, the network consumes every 20th observation (100 time steps) of each complete 2,000-point window.** Thus this is a Rule B windowing experiment, not a full-resolution 2,000-step paper reproduction. Time metadata summarizes the complete 2,000-point window and uses Train-only log1p median/IQR scaling. Raw sensor scaling also uses Train only. Event IDs, times, and TTF never enter the raw-9 input; the time-input variant adds only observed elapsed and max gap, not target time.

## Coverage

The time-only audit found B 25,310 windows/634 unique covered events and C >30min 9,833/485; strict A found 46/12. After excluding all-masked targets and nonfinite selected sensor observations, the actual usable counts are:

```text
         rule      phase  usable_sequences  pressure_low_events  pressure_low_target_samples  pressure_high_events  pressure_high_target_samples  leak_events  leak_target_samples  unique_fault_events
       Rule B      train             15076                  323                        11410                   107                         10373           48                 8725                  478
       Rule B validation              3977                  116                         3297                    30                          1811           10                 1117                  156
Rule C >30min      train              6182                  253                         4642                    86                          3961           38                 4144                  377
Rule C >30min validation              1580                   75                         1432                    23                           630           10                  706                  108
```

Target is **next fault of the same type**, in hours from window end. Each of three heads is masked when the corresponding type's future fault is not observable within the same phase. Every window remains within one Tool, phase, and any-type fault interval. There is no target-time-based 0–6h screening. The full unbounded RUL target can therefore be hundreds of hours; hourly MAE is not comparable with an earlier true-TTF-selected 0–6h conditional experiment.

## Validation metrics (hours, unless indicated)

```text
        condition    fault_type  validation_samples  validation_events   mae_h  rmse_h     r2  smape_pct  within_1h  within_2h  event_equal_mae_h
     B_raw9_Dummy  pressure_low                3297                116 333.288 396.054 -0.003     84.781      0.000      0.001            401.006
     B_raw9_Dummy pressure_high                1811                 30 458.177 520.005 -0.778     79.991      0.001      0.001            641.714
     B_raw9_Dummy          leak                1117                 10 737.494 842.279 -2.670    101.310      0.001      0.001            955.223
           B_raw9  pressure_low                3297                116 328.833 423.717 -0.148     89.005      0.003      0.004            284.496
           B_raw9 pressure_high                1811                 30 352.908 407.758 -0.093     72.636      0.002      0.003            449.755
           B_raw9          leak                1117                 10 545.111 617.558 -0.973     92.662      0.000      0.001            691.661
B_raw9_time_Dummy  pressure_low                3297                116 333.288 396.054 -0.003     84.781      0.000      0.001            401.006
B_raw9_time_Dummy pressure_high                1811                 30 458.177 520.005 -0.778     79.991      0.001      0.001            641.714
B_raw9_time_Dummy          leak                1117                 10 737.494 842.279 -2.670    101.310      0.001      0.001            955.223
      B_raw9_time  pressure_low                3297                116 327.951 421.218 -0.134     88.451      0.002      0.005            292.445
      B_raw9_time pressure_high                1811                 30 353.773 407.790 -0.093     71.581      0.002      0.004            459.119
      B_raw9_time          leak                1117                 10 547.216 614.855 -0.956     92.867      0.001      0.001            705.177
 C1800_raw9_Dummy  pressure_low                1432                 75 330.829 392.631 -0.001     79.024      0.001      0.002            412.124
 C1800_raw9_Dummy pressure_high                 630                 23 502.868 581.589 -0.984     75.783      0.000      0.000            717.541
 C1800_raw9_Dummy          leak                 706                 10 650.036 740.516 -1.816     98.215      0.001      0.001            834.520
       C1800_raw9  pressure_low                1432                 75 330.993 432.684 -0.215     85.527      0.002      0.003            282.699
       C1800_raw9 pressure_high                 630                 23 339.478 406.334  0.032     63.319      0.003      0.003            422.289
       C1800_raw9          leak                 706                 10 504.888 565.936 -0.645     91.161      0.003      0.003            639.398
```

Fault-type macro metrics:

```text
  condition  macro_mae_h  macro_event_mae_h  macro_smape_pct
     B_raw9      408.951            475.304           84.768
B_raw9_time      409.647            485.580           84.300
 C1800_raw9      391.787            448.129           80.002
```

The B raw and B time models share identical Validation windows. Rule C uses the same frozen time interval but different windows. Their common-event subset is in `common_event_subset_metrics.csv`; it equalizes events, **not timestamps**, so it is sensitivity analysis rather than a paired row-level causal comparison. Tool metrics and validation chronological thirds are in `tool_metrics.csv` and `fold_metrics.csv`. These thirds are stability slices of one fitted model, **not separate forward retrain folds**. The Validation interval has been used repeatedly in this project; none of these are independent final Test numbers.

Only 1 of 9 GRU type/condition R² values are positive. Absolute MAEs remain hundreds of hours and ±1h/±2h fractions are near zero. B time metadata does not consistently lower type-wise or event-wise MAE. C changes both coverage and window duration, so any C difference cannot be attributed solely to the gap cutoff on identical samples. The strict baseline covered too few validation events for a fair same-sample accuracy comparison; the defensible improvement is **coverage**, not RUL accuracy.

## Reproducibility and limitations

`experiment_config.json` records the rule, model architecture, target masking, scaler, split and model paths. `validation_predictions.csv.gz` contains the actual validation predictions, event IDs, timestamps and target values for rechecking metrics. The reserved Test region was not read. A future decisive experiment should compare full-resolution or explicitly time-aware sequence encoders on an independent later labeled period before a dashboard displays an expected failure time.
