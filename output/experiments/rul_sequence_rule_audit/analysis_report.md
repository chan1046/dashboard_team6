# PHM 2018 RUL operating-observation sequence rule audit (no model training)

This audit uses only actual 4-second raw time and `FIXTURESHUTTERPOSITION` for 20 Tools, the deduplicated 1,211-event fault log, and the frozen forward Train/Validation boundaries. No sensor values were interpolated, no ML/DL model was fitted or scored, no existing baseline was changed, and no reserved Test sensor rows were read. The all-event denominator includes reserved Test/purge faults; this distinction is explicit below. Detailed outputs: `segment_break_cause_summary.csv`, `operating_gap_distribution.csv`, `operating_structure_and_B_windows_by_tool.csv`, `rule_coverage.csv`, `rule_phase_coverage.csv`, `window_elapsed_overall.csv`, and `window_metadata.csv.gz`.

## Why the strict rule made 456,703 segments

The existing strict implementation starts a new segment on a non-FSP=1 → FSP=1 transition, raw timestamp gap >12 seconds, non-increasing/duplicate raw time, frozen split boundary, or crossed fault. Recomputed starts equal the stored segment count **exactly**. Mutually exclusive attribution uses the priority Tool start → FSP transition → gap → non-increasing time → split → fault; `multilabel_count` separately shows coincident causes.

| Exclusive cause | Breaks | Share |
|---|---:|---:|
| FSP state transition into operating | 410,304 | 89.840% |
| Non-increasing raw timestamp (duplicate or reversed time) | 44,758 | 9.800% |
| Raw operating-row gap >12–120s | 920 | 0.201% |
| Raw operating-row gap >120s (session-like) | 531 | 0.116% |
| Fault boundary | 159 | 0.035% |
| Frozen split boundary | 11 | 0.002% |
| First operating row of each Tool | 20 | 0.004% |

Tool changes cannot occur within a per-Tool source file; 20 Tool starts are counted explicitly. There is no other unexplained cause. “Session change” in this audit means a raw operating-row gap above 120 seconds; it is already included in the time-gap total rather than double counted. The former 12-second strict rule was an analyst choice, not stated in the TV paper. Duplicate timestamps must be diagnosed separately; they do not demonstrate a backwards-moving clock or missing sensor values by themselves.

## Operating-observation structure

Before reserved Test, there are **64,109,241 raw rows** and **39,043,817 FSP=1 observations** across 20 Tools. For each Tool, the median and 90th-percentile *adjacent FSP=1 observation* gap are 4 seconds; tool-specific wall-clock spans, observations, gap quantiles, and Rule B window counts are in `operating_structure_and_B_windows_by_tool.csv`. Keeping only FSP=1 removes downtime rows but does **not** erase elapsed time between the retained observations.

| Adjacent retained-observation gap | Count | Share |
|---|---:|---:|
| ≤0s (repeated/non-increasing) | 44,767 | 0.115% |
| >0–8s | 38,587,068 | 98.830% |
| >8–30s | 17,008 | 0.044% |
| >30–60s | 111,234 | 0.285% |
| >1–5min | 28,334 | 0.073% |
| >5–30min | 222,699 | 0.570% |
| >30min | 32,687 | 0.084% |

The finer histogram has **34,307** gaps in 60–120s, only **641** in 120–300s, then **192,210** in 300–600s. Thus 120–300s is a genuine low-frequency interval. Cutting at either edge (120 or 300 seconds) nevertheless discards almost all useful 2,000-point windows. A 30-minute cutoff trims only the long-gap tail; **there is no sharp distribution valley at exactly 1,800 seconds**. It is retained as a sensitivity candidate, not presented as a paper-prescribed or uniquely optimal threshold.

## What the TV et al. paper actually says

The [primary paper, Section 6.1](https://papers.phmsociety.org/index.php/phmconf/article/download/589/phmc_18_589) says: select only observations whose fixture shutter position is 1, use nine named numeric sensors, *then* divide each Tool's data into overlapping windows of **2,000 points, 500 overlap**. Therefore 2,000 refers to **retained operating observations**, not 2,000 raw rows including downtime. Its wording supports sequential indexing after FSP screening. It does **not** explicitly specify a maximum wall-clock gap, a shutter-cycle boundary, a downtime interpolation method, or a requirement that the 2,000 points be 8,000 consecutive seconds. Those implementation details are **not confirmable from the paper**. The paper's approximately 0.1M windows are not directly comparable with our restricted pre-Test Train/Validation and fault-boundary rule.

## Fixed coverage comparison

All rules use 2,000 retained observations, 500 overlap (stride 1,500), per-Tool chronology, the same frozen Train/Validation intervals and event-linkage check: a window and its next same-type fault must lie in the same split, and the window starts after the previous same-type fault. No window crosses an intervening **any-type** fault. No feature values, training labels, or model probabilities were inspected to select a rule. Rule B and C retain duplicate-time rows as provided observations with zero-second metadata gap; they break only for actual time reversal, Tool/split/fault boundary and, for C, the nominated long-gap cutoff.

| Rule | Train windows | Validation windows | All windows | Unique covered events | Low / High / Leak | Share of all 1,211 faults |
|---|---:|---:|---:|---:|---:|---:|
| A: previous strict segment | 14 | 32 | **46** | **12** | 5 / 4 / 3 | 0.99% |
| B: operating-observation axis | 19,044 | 6,266 | **25,310** | **634** | 439 / 137 / 58 | 52.35% |
| C: gap >120s | 20 | 35 | **55** | **14** | 6 / 5 / 3 | 1.16% |
| C: gap >300s | 20 | 38 | **58** | **15** | 6 / 6 / 3 | 1.24% |
| C: gap >1,800s | 7,591 | 2,242 | **9,833** | **485** | 328 / 109 / 48 | 40.05% |

The frozen Train interval contains 751 recorded faults, Validation 262, and reserved Test/purge 198. Because this audit did not read reserved Test sensor rows, the percentages of all 1,211 are **pre-Test coverage**, not an estimate that the remaining 198 cannot be covered. B covers 478 Train + 156 Validation events; C-1,800 covers 377 + 108. Per-phase and per-type counts are in `rule_phase_coverage.csv`. The absolute number of logged fault events and the number of usable sequence-linked events must not be conflated.

## Wall-clock meaning of 2,000 observations

| Rule | Median | p25 | p75 | p90 | Min–max | Windows >24h |
|---|---:|---:|---:|---:|---:|---:|
| A | 2.22h | 2.22h | 2.22h | 2.22h | 2.22–2.22h | 0 |
| B | **6.10h** | 4.21h | 8.69h | 13.79h | 2.22–**273.20h** | **760** |
| C >120s | 2.22h | 2.22h | 2.22h | 2.23h | 2.22–2.30h | 0 |
| C >300s | 2.22h | 2.22h | 2.22h | 2.23h | 2.22–2.32h | 0 |
| C >1,800s | **3.06h** | 2.68h | 4.61h | 5.42h | 2.22–10.52h | 0 |

Rule B's longest window spans **11.38 days** of wall-clock time despite containing only 2,000 operating observations. Its median maximum *single* internal gap is 3,904 seconds and maximum is 952,850 seconds. This is a variable-length-time history, not a continuous 2.22-hour signal. C-1,800's maximum internal gap is 1,800 seconds. Per-window elapsed time and maximum gap are saved in `window_metadata.csv.gz`; stop duration is never replaced by fabricated sensor readings. For downstream use, a model must receive or explicitly handle elapsed-time/gap metadata, and results should be stratified by duration.

## Recommendation

For the **next paper-style dataset reconstruction**, use **Rule B as the primary operating-observation definition**: it follows the published FSP=1 → 2,000 operating-point window order most closely, preserves real observation order and gap metadata, covers 634 pre-Test fault events, and never interpolates downtime. **Do not feed it to a regular-time GRU as if all 2,000 points were consecutive 4-second samples**; the 760 cross-day windows require explicit gap-aware handling or separate duration-stratified evaluation. Keep C-1,800 as a predeclared sensitivity dataset because it retains 485 events while bounding wall-clock history to 10.52h. The 1,800-second threshold is an operational long-gap sensitivity cutoff, not a discovered paper parameter or unique natural minimum. A or C-120/300 would repeat the over-exclusion problem.

The previous strict rule **did discard data excessively** for a paper-style operating-observation analysis: 12 versus 634 covered events, driven mainly by breaking at each shutter transition and duplicate timestamp. This conclusion concerns dataset coverage only; it predicts no RUL model performance.
