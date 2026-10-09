# Degradation warning → RUL: development validation

**Coverage audit correction (see `coverage_audit.csv`): This experiment is not a valid basis for deciding that the challenge data cannot support RUL.** Only 78/186, 3/59, and 1/17 logged Validation fault events received warnings. The 2,000-operating-observation requirement already removes 68 low-pressure, 28 high-pressure, and 6 leak events. Another 10/1/1 events have no eligible sparse window. Moreover 6,391/6,424 (99.5%) Validation windows contain an operating-observation gap above 90 seconds; the median maximum gap is 4,964 seconds. Thus these sequences routinely bridge actual equipment stops. The ~100-minute warning stride misses short-lived changes. These are serious construction limitations, not evidence of a sensor-information limit. Existing metrics are retained as an audit trail but should not be used to select Model 2.

This is a paper-inspired two-stage experiment on actual PHM 2018 data, not an exact paper reproduction. Existing 24h XGBoost Model 1 was not accessed or retrained. The frozen forward Train/Validation boundaries were reused; reserved Test sensor rows were not read. Validation in this project has been used before, so these are development results.

## Construction and leakage controls

- Only 4-second raw observations with `FIXTURESHUTTERPOSITION=1` enter the dataset. No stop interval is filled. A 2,000-operating-observation window advances by 1,500 observations (usually about 100 minutes); the GRU sees every 20th raw observation, while Random Forest uses current, mean, standard deviation, and end-minus-start summaries of the raw window. Consequently this **is not per-4-second online warning evaluation**. A window can span an actual shutdown; its elapsed gap is recorded, not imputed.
- The degradation label is an explicitly chosen proxy: the next same-type fault occurs within 24 hours. It does not identify a measured physical degradation onset. The final 24 hours of each phase have no detector label because the entire forward horizon is not observable inside that phase. Fault timestamps and TTF are never model inputs.
- Each Tool's Train time is divided into an earlier detector-fit region and a later RUL-fit region, with 24-hour embargo on both sides of the internal boundary. The detector selects later-Train RUL rows using out-of-sample predictions. Validation follows the frozen boundaries. Same-type faults in the input window are excluded from the corresponding RUL target. Features use only current and earlier observations.
- Detector threshold is fixed at 0.5. For a retrospectively linked fault event, RUL samples begin with the first warning in the 24-hour pre-fault period and end before the fault. Fault events with no warning are excluded and counted. This conditioning uses event time for retrospective evaluation; deployment would have to maintain warning state online.
- The no-warning and after-warning RUL models are evaluated on **the identical after-warning Validation windows**. They have different Train populations by design. This tests the training/selection structure, but does not estimate RUL quality for faults that never received a warning.

## Detector

See `detector_metrics.csv` and `event_summary.csv`. Row-level AP: low pressure 0.329, high pressure 0.0268, leak 0.00735. On all logged Validation faults, warnings precede 78/186 low-pressure, 3/59 high-pressure, and 1/17 leak events. Among faults with an eligible raw window, these are 78/108, 3/30, and 1/10. Median lead times are 16.31, 16.15, and 7.71 hours, respectively. Normal false warning rows are 476, 193, and 294; 10-minute grouping yields the same counts because evaluated windows are far apart. These counts should not be interpreted as notification volume from a 4-second online detector.

## Matched RUL comparison

See `rul_metrics.csv`, `matched_ablation_mae.csv`, `tool_metrics.csv`, and `fold_metrics.csv`. Low-pressure Validation has 217 post-warning windows from 78 events. The no-warning vs after-warning MAE is: Dummy 221.92→5.87 h; Tree 28.06→6.03 h; GRU 16.11→10.75 h. Yet the post-warning Dummy is **better** than both trained models, and all post-warning R² values are negative. The after-warning Tree has RMSE 7.30 h, SMAPE 65.0%, and only 18.4% within ±2 h. Its event-equal MAE is 6.09 h. The after-warning GRU has RMSE 12.64 h, SMAPE 180.6%, and event-equal MAE 9.48 h.

High-pressure Validation contains only 6 post-warning windows / 3 events. Its GRU worsens from 8.66 to 13.12 h MAE. Leak has only 2 windows / 1 event and no warning-positive Train event, so an after-warning RUL model is unavailable. These sample sizes cannot support a stable conclusion.

For low pressure, after-warning Tree MAE in three chronological **score slices** is 6.05/6.26/5.79 h, and after-warning GRU is 11.46/10.63/10.11 h. These are not three independently refitted forward folds. Tool-level breakdown is in `tool_metrics.csv`; only four Tools contribute low-pressure warned Validation events, and tool imbalance remains material.

## Interpretation

Warning selection makes the low-pressure RUL target range much shorter and reduces MAE versus broad-window training on the same selected Validation rows. It does **not** demonstrate a useful learned RUL predictor: Dummy still wins, low-pressure event coverage is 78/186, high-pressure/leak coverage is very poor, and sparse window stride limits warning evaluation. The earlier whole-Train paper-style baseline has a different Train and Validation population, so its MAE must not be compared directly as a controlled improvement. This two-stage structure should not replace Model 2 or be shown as a reliable dashboard time estimate yet.
