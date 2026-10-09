# TV et al. 2018 window semantics audit — no model training

Primary source: [TV et al., 2018, §§5.1–6.1](https://papers.phmsociety.org/index.php/phmconf/article/download/589/phmc_18_589). The authors say they retain FSP=1, use nine sensors, and split each Tool's data into overlapping **2,000-point windows with 500-point overlap**. They obtain each three-type RUL target vector from the provided Train TTF values. The mathematical stride of a literal 500-point overlap is **1,500 points**. The PDF contains no explicit `stride` value, no `sliding window` implementation loop, and no instruction to cut at a logged fault or maintenance boundary. It does not say that stride is 500. The approximately 0.1M / 150 numbers are descriptive, not source code.

The [publisher article page](https://www.papers.phmsociety.org/index.php/phmconf/article/view/589) provides a PDF but no listed supplemental code. Targeted searches for the exact title, `LSTM-MR`, named authors, and TCS Research found **no verified author/team public window-generation implementation**. A [third-party PHM challenge repository](https://github.com/ninja1mmm/2018-phm-data-challenge) exists, but it is not attributable to the paper authors and cannot establish their stride, clipping, target, or boundary semantics. Public implementation status: **unconfirmed**.

## Four-rule comparison on the same frozen pre-Test range

Targets are the next strictly future same-type logged fault within the same Train/Validation phase; no Validation/Test labels are used as features. Near means at least one observable type has RUL <500 raw timestamp units. All rules use 2,000 FSP=1 operating observations, Tool boundaries, and chronological split/purge boundaries. They differ only in stride and whether each fault cuts the operating sequence.

```text
            variant  windows  near_any  near_pressure_low  near_pressure_high  near_leak  zero_fault_inside  one_fault_inside  two_or_more_faults_inside  post_first_fault_sensor  target_fault_at_or_before_input_end  train  validation  unique_target_events
   fault_stride1500    25310         9                  8                   0          1              25310                 0                          0                        0                                    0  19044        6266                   634
    fault_stride500    75304        44                 28                   5         11              75304                 0                          0                        0                                    0  56680       18624                   637
no_fault_stride1500    25909        18                 17                   1          0              25015               679                        215                      894                                    0  19477        6432                   761
 no_fault_stride500    77686        55                 45                   6          6              74995              2057                        634                     2691                                    0  58412       19274                   816
```

Detailed Tool × phase and type-wise unique event counts are in `window_counts_by_tool_phase.csv` and `unique_events_by_variant_phase_type.csv`. For example, the exact current rule (fault boundary, stride 1,500) has **25,310 total / 9 near / 634 unique target events**; no-fault stride 500 has **77,686 / 55 / 816** on this same restricted range. The earlier dataset audit additionally found **100,199** total Tool-only stride-500 windows over the full original Train *time/FSP axis*, without reading its reserved later TTF values. That number is close to the paper's approximate 0.1M, but it does not prove their implementation.

## Fault-crossing semantics and possible information transfer

Counts of input windows containing logged faults:

```text
            variant  zero_fault_inside  one_fault_inside  two_or_more_faults_inside  post_first_fault_sensor  target_fault_at_or_before_input_end
   fault_stride1500              25310                 0                          0                        0                                    0
    fault_stride500              75304                 0                          0                        0                                    0
no_fault_stride1500              25015               679                        215                      894                                    0
 no_fault_stride500              74995              2057                        634                     2691                                    0
```

The categories are **not all mutually exclusive**: category C (two or more faults) is a subset of D (post-fault sensor observations). A is no fault inside. B, an input that reaches or passes its **current next target fault**, is zero by explicit timestamp assertion: target time is strictly greater than input end. Therefore removing fault boundaries does **not** leak sensor readings from the future of the chosen target event. It *does* allow the model to see that one or more **earlier** fault/maintenance cycles have occurred, through the history after those events. This is past information in deployment, but can be a shortcut if the intended task is a single run-to-failure cycle. A future exact-fault-time target from provided TTF=0 would need its own anchor convention; it was not silently mixed into this next-future target audit.

Across 823 unique faults with adjacent retained operating observations on both sides, **585** have stage change, **539** stage decrease, **589** operating gap >30 min, and **580** have a maximum raw-nine sensor jump above the same Tool's 99th percentile for non-fault adjacent operating transitions. Median fault-crossing gap is **6728 seconds**. Stage and timestamp are **not** among the nine TV input sensors, but the nine raw sensor sequence can still encode an abrupt post-maintenance change. The >99th-percentile comparison is descriptive, not a fitted classifier or a claim that every jump is caused by maintenance.

Window-level cue overlap among the fault-crossing windows:

```text
            variant  windows  windows_with_matched_transition  stage_change  stage_decrease  raw9_jump_above_tool_p99  gap_over_30min
no_fault_stride1500      894                              894           690             635                       688             692
 no_fault_stride500     2691                             2691          2078            1924                      2065            2109
```

## Paper-number fit versus method fit

- **Numerical fit:** On our pre-Test range, stride 500 without a fault boundary is closest of the four to 0.1M / 150, but still reaches only 77,686 / 55. Full original Train time/FSP reaches 100,199 Tool-only stride-500 windows; its near count was deliberately not opened.
- **Textual fit:** Literal `overlap of 500 points` favors stride **1,500**. The paper does not specify a fault boundary. Hence no-fault stride-1,500 is the closest *textual* reconstruction, while the question of a hidden cycle cut remains unresolved.
- **Risk:** All four rules exclude post-**target** sensor data. Removing a fault boundary mixes earlier fault cycles and introduces measurable maintenance-state cues. It is not classic future-data leakage, but is a genuine shortcut/distribution risk and may change the scientific question. Numeric closeness alone cannot resolve the author's undisclosed code.

## Recommendation

For the **next actual prognostic training rule**, choose **stride 1,500 with a fault boundary**. It honors the paper's literal 500-point overlap, keeps strict time order, and prevents the model from reading post-*earlier*-fault sensor changes as an implicit maintenance-state cue. It is a deliberately safer adaptation, **not an exact published implementation**. A Tool-only stride-1,500 construction should remain a *read-only paper-text sensitivity dataset* until the meaning of its 894 fault-crossing windows is established. No candidate fully meets every requested criterion: the safer rule has only **9** pre-Test <500 near windows, and even the Tool-only text-faithful rule has only 18. Hence do **not** start another type-wise RUL training run yet; obtain more genuine near-failure observations or change the separately justified prediction question first. Do not choose stride 500 solely because it reproduces the approximate 100k count.
