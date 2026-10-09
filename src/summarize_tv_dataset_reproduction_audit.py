"""Summarize TV window audit and full Train coverage without opening held-out TTF."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/experiments/tv_dataset_reproduction_audit'
RAW=ROOT/'data/processed'
PLAN=ROOT/'output/experiments/forward_validation_ab_24h/split_plan.json'

def windows_on_axis(t,stride):
    if len(t)<2000:return 0
    starts=np.r_[True,np.diff(t)<0]
    bounds=np.r_[np.flatnonzero(starts),len(t)]
    lengths=np.diff(bounds)
    return int(np.maximum(0,1+(lengths-2000)//stride)[lengths>=2000].sum())

def full_range_count(plan):
    rows=[]
    for item in plan['tools']:
        tool=item['tool_id']
        tab=pq.read_table(RAW/f'{tool}_sensor_ttf.parquet',columns=['time','FIXTURESHUTTERPOSITION'])
        df=tab.to_pandas(); t=df.time.to_numpy(np.int64)
        op=t[df.FIXTURESHUTTERPOSITION.to_numpy()==1]
        rows.append({'tool_id':tool,'all_original_train_rows':len(t),
          'all_original_train_operating_rows':len(op),
          'full_tool_only_stride1500_windows':windows_on_axis(op,1500),
          'full_tool_only_stride500_windows':windows_on_axis(op,500),
          'full_unfiltered_raw_stride500_windows':windows_on_axis(t,500)})
        print('full-range time/FSP only',tool,flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'full_original_train_time_fsp_only_counts.csv',index=False)
    return frame

def main():
    if not (OUT/'window_counts_by_variant_total.csv').exists():
        raise FileNotFoundError('Run audit_tv_dataset_reproduction.py first')
    plan=json.loads((ROOT/'output/experiments/forward_validation_ab_24h/split_plan.json').read_text(encoding='utf-8'))
    full=full_range_count(plan)
    total=pd.read_csv(OUT/'window_counts_by_variant_total.csv')
    phase=pd.read_csv(OUT/'window_counts_by_variant_phase.csv')
    ttf=pd.read_csv(OUT/'ttf_comparison_by_fault_type.csv')
    provenance=pd.read_csv(OUT/'raw_csv_parquet_provenance_check.csv')
    if provenance.matched_rows.sum()!=1000:raise AssertionError('Raw CSV provenance mismatch')
    if not (ttf.exact_finite_pct==100).all():raise AssertionError('TTF mismatch')
    base=total.loc[total.variant=='stride1500_fault'].iloc[0]
    if int(base.windows)!=25310 or int(base.usable_windows)!=19053 or int(base.direct_near_windows)!=9:
        raise AssertionError('Prior Rule B numbers did not reproduce')
    comparison=pd.DataFrame([
      {'condition':'Paper, original Train (approximate)','scope':'original paper','all_windows':100000,
        'usable_windows':np.nan,'near_direct':150,'near_source':np.nan},
      *[{'condition':r.variant,'scope':'our pre-Test Train+Validation',
         'all_windows':int(r.windows),'usable_windows':int(r.usable_windows),
         'near_direct':int(r.direct_near_windows),'near_source':int(r.source_near_windows)}
         for r in total.itertuples()],
      {'condition':'full original Train, Tool-only, stride1500, time/FSP count',
       'scope':'includes reserved later segment, no target inspected',
       'all_windows':int(full.full_tool_only_stride1500_windows.sum()),
       'usable_windows':np.nan,'near_direct':np.nan,'near_source':np.nan},
      {'condition':'full original Train, Tool-only, stride500, time/FSP count',
       'scope':'includes reserved later segment, no target inspected',
       'all_windows':int(full.full_tool_only_stride500_windows.sum()),
       'usable_windows':np.nan,'near_direct':np.nan,'near_source':np.nan}])
    comparison.to_csv(OUT/'paper_vs_current_comparison.csv',index=False)
    report=f'''# TV 2018 dataset construction audit — no model training

The [TV et al. 2018 paper, §6.1](https://papers.phmsociety.org/index.php/phmconf/article/download/589/phmc_18_589) says each Tool's FSP=1 observations are divided into 2,000-point windows with **500 points of overlap**, and supplied Train time-to-failure values make the three-target RUL vector. It reports approximately **0.1 million windows** and approximately **150 windows** with at least one target <500. Its wording supports **stride 1,500** mathematically. Stride 500 is an alternative interpretation tested here, not confirmed paper implementation. The paper does not state a fault-event segmentation rule. It does not state the exact code used to align TTF rows to operating windows, which observations lacking all targets were excluded from its 0.1M figure, or its total FSP=1 row count after all cleaning.

## Current implementation and ablations

Window length 2,000, overlap 500, stride 1,500 are confirmed in `src/audit_rul_sequence_rules.py` and `src/train_rul_rule_b_baseline.py`. Counting uses exactly the existing per-Tool chronological FSP=1 order and frozen Train/Validation boundaries with at least 24h purge. The reserved later segment was not used for labels or model selection. No interpolation, feature engineering or model fitting was performed. `all_windows` counts every length-valid window. `usable_windows` requires nine nonmissing sensor values and at least one observed next same-type target within that frozen phase. In this data, all sampled sensor windows are finite; **the 25,310 → 19,053 reduction comes from all three targets being unavailable in the restricted phase, not from sensor gaps**.

```text
{comparison.to_string(index=False)}
```

Train/Validation phase counts and per-fault near counts are in `window_counts_by_variant_phase.csv` and `window_counts_by_variant_total.csv`. Current Rule B near count is **6 Train + 3 Validation = 9** when calculated from the next strictly later logged fault, or **7 + 3 = 10** from the supplied TTF at window end. The one-window discrepancy is at an exact fault timestamp: provided TTF is zero, whereas the prior baseline intentionally seeks the next **strictly later** fault. That one row does not explain the paper's near-window count.

Removing fault boundaries increases all windows by **{int(total.loc[total.variant=='stride1500_no_fault','windows'].iloc[0]-base.windows)}** at stride 1,500, versus **{int(total.loc[total.variant=='stride500_no_fault','windows'].iloc[0]-total.loc[total.variant=='stride500_fault','windows'].iloc[0])}** at stride 500. It allows **894** and **2,691** windows, respectively, to straddle an earlier logged fault. Those windows contain sensor observations after that **earlier** fault but never after the **next target fault** because the target is strictly later than the window end. They mix operating cycles, so they require a deliberate cycle-definition decision before model training. Stride interpretation has the larger effect on total and near window counts; fault boundaries have a smaller but real effect on near windows.

## Supplied train_ttf versus direct timestamp arithmetic

The processed parquet's TTF columns were checked directly against **1,000 rows from the original `train/train_ttf` CSVs** (50 per Tool): **1,000/1,000 rows match exactly**, including NaNs. Then {int(ttf.samples.iloc[0]):,} pre-Test operating timestamps (1,000 random per Tool plus all source near-TTF observations) were compared for each of three types against `next same-type logged fault at or after time − time` using the **full available fault log only for this read-only target audit**:

```text
{ttf.to_string(index=False)}
```

All **{int(ttf.exact_finite.sum()):,} finite comparisons** agree exactly, and all **{int(ttf.both_nan.sum()):,} jointly missing values** agree. Thus no meaningful unit conversion or arithmetic mismatch exists. A supplied TTF of 500 corresponds to a 500-unit timestamp difference (about 8m20s given the 4-second source cadence), not 500 hours. The paper uses the provided Train TTF values (§6.1); our prior baseline recomputed them from fault timestamps, which differs only for the exact-fault-time convention and restricted split horizon in this audit. Raw CSV source/provenance evidence is in `raw_csv_parquet_provenance_check.csv`.

## FSP and data-range effect

Our pre-Test subset has **39,043,817** FSP=1 observations, as recorded by the sequence rule audit. Counting the **full original Train file's** time and FSP columns only, without reading its later sensor values, TTF, or labels, yields **{int(full.all_original_train_operating_rows.sum()):,}** FSP=1 observations and **{int(full.full_tool_only_stride1500_windows.sum()):,}** / **{int(full.full_tool_only_stride500_windows.sum()):,}** Tool-only windows at stride 1,500 / 500. These are structural upper-scope comparisons, **not** near-window counts or permission to tune on reserved labels. The full original unfiltered raw row count is **{int(full.all_original_train_rows.sum()):,}**, versus **{int(full.full_unfiltered_raw_stride500_windows.sum()):,}** raw stride-500 windows before FSP filtering. The paper and current audit both apply FSP=1, so FSP filtering itself is not an identified discrepancy.

## Conclusion

Within the frozen pre-Test subset, **stride 500 plus no fault boundary** is numerically closest of the four tested combinations to the paper's approximately 100,000 total windows and 150 near windows, but still yields only **{int(total.loc[total.variant=='stride500_no_fault','windows'].iloc[0]):,} / {int(total.loc[total.variant=='stride500_no_fault','direct_near_windows'].iloc[0])}**. Full-file time/FSP counts bring stride-500 Tool-only total windows much closer to the paper's 0.1M, so **coverage range and stride** explain most of the window-count gap. However, the paper explicitly says *overlap 500*, which corresponds to stride 1,500; matching a reported approximate number is not evidence that the authors used stride 500. The source TTF arithmetic agrees exactly and is not the main cause. The fault boundary explains part of the near-count gap, but its removal mixes cycles. The remaining near-window gap likely reflects both the pre-Test range and different window eligibility/target conventions; the paper's exact implementation is **unconfirmed**. No model should be retrained merely because a variant matches the paper's approximate counts. First specify the intended paper-faithful stride/target alignment and how fault-crossing windows are handled, then verify on a dataset with enough truly near-failure observations.
'''
    (OUT/'analysis_report.md').write_text(report,encoding='utf-8')
    print(comparison.to_string(index=False))

if __name__=='__main__':main()
