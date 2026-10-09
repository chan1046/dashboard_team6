# TV 2018 dataset construction audit — no model training

The [TV et al. 2018 paper, §6.1](https://papers.phmsociety.org/index.php/phmconf/article/download/589/phmc_18_589) says each Tool's FSP=1 observations are divided into 2,000-point windows with **500 points of overlap**, and supplied Train time-to-failure values make the three-target RUL vector. It reports approximately **0.1 million windows** and approximately **150 windows** with at least one target <500. Its wording supports **stride 1,500** mathematically. Stride 500 is an alternative interpretation tested here, not confirmed paper implementation. The paper does not state a fault-event segmentation rule. It does not state the exact code used to align TTF rows to operating windows, which observations lacking all targets were excluded from its 0.1M figure, or its total FSP=1 row count after all cleaning.

## Current implementation and ablations

Window length 2,000, overlap 500, stride 1,500 are confirmed in `src/audit_rul_sequence_rules.py` and `src/train_rul_rule_b_baseline.py`. Counting uses exactly the existing per-Tool chronological FSP=1 order and frozen Train/Validation boundaries with at least 24h purge. The reserved later segment was not used for labels or model selection. No interpolation, feature engineering or model fitting was performed. `all_windows` counts every length-valid window. `usable_windows` requires nine nonmissing sensor values and at least one observed next same-type target within that frozen phase. In this data, all sampled sensor windows are finite; **the 25,310 → 19,053 reduction comes from all three targets being unavailable in the restricted phase, not from sensor gaps**.

```text
                                                 condition                                                scope  all_windows  usable_windows  near_direct  near_source
                       Paper, original Train (approximate)                                       original paper       100000             NaN        150.0          NaN
                                          stride1500_fault                        our pre-Test Train+Validation        25310         19053.0          9.0         10.0
                                       stride1500_no_fault                        our pre-Test Train+Validation        25909         19634.0         18.0         18.0
                                           stride500_fault                        our pre-Test Train+Validation        75304         56588.0         44.0         45.0
                                        stride500_no_fault                        our pre-Test Train+Validation        77686         58858.0         55.0         55.0
full original Train, Tool-only, stride1500, time/FSP count includes reserved later segment, no target inspected        33406             NaN          NaN          NaN
 full original Train, Tool-only, stride500, time/FSP count includes reserved later segment, no target inspected       100199             NaN          NaN          NaN
```

Train/Validation phase counts and per-fault near counts are in `window_counts_by_variant_phase.csv` and `window_counts_by_variant_total.csv`. Current Rule B near count is **6 Train + 3 Validation = 9** when calculated from the next strictly later logged fault, or **7 + 3 = 10** from the supplied TTF at window end. The one-window discrepancy is at an exact fault timestamp: provided TTF is zero, whereas the prior baseline intentionally seeks the next **strictly later** fault. That one row does not explain the paper's near-window count.

Removing fault boundaries increases all windows by **599** at stride 1,500, versus **2382** at stride 500. It allows **894** and **2,691** windows, respectively, to straddle an earlier logged fault. Those windows contain sensor observations after that **earlier** fault but never after the **next target fault** because the target is strictly later than the window end. They mix operating cycles, so they require a deliberate cycle-definition decision before model training. Stride interpretation has the larger effect on total and near window counts; fault boundaries have a smaller but real effect on near windows.

## Supplied train_ttf versus direct timestamp arithmetic

The processed parquet's TTF columns were checked directly against **1,000 rows from the original `train/train_ttf` CSVs** (50 per Tool): **1,000/1,000 rows match exactly**, including NaNs. Then 42,753 pre-Test operating timestamps (1,000 random per Tool plus all source near-TTF observations) were compared for each of three types against `next same-type logged fault at or after time − time` using the **full available fault log only for this read-only target audit**:

```text
   fault_type  samples  both_finite  exact_finite  both_nan  provided_nan  direct_nan  difference_min  difference_median  difference_max  exact_finite_pct
         leak    42753        28073         28073     14680         14680       14680             0.0                0.0             0.0             100.0
pressure_high    42753        36132         36132      6621          6621        6621             0.0                0.0             0.0             100.0
 pressure_low    42753        39656         39656      3097          3097        3097             0.0                0.0             0.0             100.0
```

All **103,861 finite comparisons** agree exactly, and all **24,398 jointly missing values** agree. Thus no meaningful unit conversion or arithmetic mismatch exists. A supplied TTF of 500 corresponds to a 500-unit timestamp difference (about 8m20s given the 4-second source cadence), not 500 hours. The paper uses the provided Train TTF values (§6.1); our prior baseline recomputed them from fault timestamps, which differs only for the exact-fault-time convention and restricted split horizon in this audit. Raw CSV source/provenance evidence is in `raw_csv_parquet_provenance_check.csv`.

## FSP and data-range effect

Our pre-Test subset has **39,043,817** FSP=1 observations, as recorded by the sequence rule audit. Counting the **full original Train file's** time and FSP columns only, without reading its later sensor values, TTF, or labels, yields **50,133,540** FSP=1 observations and **33,406** / **100,199** Tool-only windows at stride 1,500 / 500. These are structural upper-scope comparisons, **not** near-window counts or permission to tune on reserved labels. The full original unfiltered raw row count is **82,189,440**, versus **164,311** raw stride-500 windows before FSP filtering. The paper and current audit both apply FSP=1, so FSP filtering itself is not an identified discrepancy.

## Conclusion

Within the frozen pre-Test subset, **stride 500 plus no fault boundary** is numerically closest of the four tested combinations to the paper's approximately 100,000 total windows and 150 near windows, but still yields only **77,686 / 55**. Full-file time/FSP counts bring stride-500 Tool-only total windows much closer to the paper's 0.1M, so **coverage range and stride** explain most of the window-count gap. However, the paper explicitly says *overlap 500*, which corresponds to stride 1,500; matching a reported approximate number is not evidence that the authors used stride 500. The source TTF arithmetic agrees exactly and is not the main cause. The fault boundary explains part of the near-count gap, but its removal mixes cycles. The remaining near-window gap likely reflects both the pre-Test range and different window eligibility/target conventions; the paper's exact implementation is **unconfirmed**. No model should be retrained merely because a variant matches the paper's approximate counts. First specify the intended paper-faithful stride/target alignment and how fault-crossing windows are handled, then verify on a dataset with enough truly near-failure observations.
