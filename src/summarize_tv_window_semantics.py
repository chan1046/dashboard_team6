"""Summarize paper evidence and the read-only window semantics audit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/experiments/tv_window_semantics_audit'
PRIOR=ROOT/'output/experiments/tv_dataset_reproduction_audit'
FAULT=ROOT/'data/processed/fault_events.csv'
PAPER='https://papers.phmsociety.org/index.php/phmconf/article/download/589/phmc_18_589'

def cue_windows(cross,transitions,faults):
    result=[]
    for tool,g in cross.groupby('tool_id'):
        ft=np.sort(faults.loc[faults.tool_id==tool,'time'].unique().astype(np.int64))
        tr=transitions.loc[transitions.tool_id==tool].set_index('fault_time')
        for row in g.itertuples():
            inside=ft[(ft>row.start_time)&(ft<=row.end_time)]
            match=tr.reindex(inside)
            result.append({'variant':row.variant,'tool_id':tool,'start_time':row.start_time,
                'end_time':row.end_time,'faults_inside':len(inside),
                'matched_fault_transitions':int(match.gap_seconds.notna().sum()),
                'any_stage_change_at_fault':bool(match.stage_changed.fillna(False).any()),
                'any_stage_decrease_at_fault':bool(match.stage_numeric_decrease.fillna(False).any()),
                'any_sensor_jump_above_tool_p99':bool(match.jump_above_ordinary_p99.fillna(False).any()),
                'any_fault_gap_over_30min':bool((match.gap_seconds>1800).fillna(False).any())})
    return pd.DataFrame(result)

def main():
    counts=pd.read_csv(OUT/'window_counts_by_variant_total.csv')
    phase=pd.read_csv(OUT/'window_counts_by_variant_phase.csv')
    events=pd.read_csv(OUT/'covered_target_events.csv')
    cross=pd.read_csv(OUT/'fault_crossing_window_metadata.csv.gz')
    transitions=pd.read_csv(OUT/'fault_transition_signatures.csv')
    faults=pd.read_csv(FAULT).drop_duplicates(['tool_id','time','fault_type'])
    cue=cue_windows(cross,transitions,faults)
    cue.to_csv(OUT/'fault_crossing_window_cues.csv',index=False)
    cues=cue.groupby('variant').agg(windows=('start_time','size'),
        windows_with_matched_transition=('matched_fault_transitions',lambda x:int((x>0).sum())),
        stage_change=('any_stage_change_at_fault','sum'),
        stage_decrease=('any_stage_decrease_at_fault','sum'),
        raw9_jump_above_tool_p99=('any_sensor_jump_above_tool_p99','sum'),
        gap_over_30min=('any_fault_gap_over_30min','sum')).reset_index()
    cues.to_csv(OUT/'fault_crossing_cue_summary.csv',index=False)
    summary=counts[['variant','windows','near_any','near_pressure_low','near_pressure_high','near_leak',
       'zero_fault_inside','one_fault_inside','two_or_more_faults_inside',
       'post_first_fault_sensor','target_fault_at_or_before_input_end']].copy()
    summary=summary.merge(phase.pivot(index='variant',columns='phase',values='windows').reset_index(),on='variant')
    ec=events.groupby('variant').size().rename('unique_target_events').reset_index()
    summary=summary.merge(ec,on='variant')
    summary.to_csv(OUT/'four_rule_comparison.csv',index=False)
    # Internal reconciliation with the independent previous audit.
    prior=pd.read_csv(PRIOR/'window_counts_by_variant_total.csv')
    names={'fault_stride1500':'stride1500_fault','fault_stride500':'stride500_fault',
           'no_fault_stride1500':'stride1500_no_fault','no_fault_stride500':'stride500_no_fault'}
    for row in summary.itertuples():
        old=prior.loc[prior.variant==names[row.variant]].iloc[0]
        if row.windows!=old.windows or row.near_any!=old.direct_near_windows:
            raise AssertionError('Independent audit mismatch')
    evidence=[
      {'topic':'window length and operating filter','confirmed':'FSP=1 observations; Tool-wise windows of length 2000','not_confirmed':'Whether tool timeline is additionally split at fault or maintenance'},
      {'topic':'overlap and stride','confirmed':'Text says overlap of 500 points; arithmetic stride is 1500','not_confirmed':'No explicit stride variable/code; approximate 0.1M count resembles stride 500 on full Train'},
      {'topic':'sliding window','confirmed':'Overlapping windows stated','not_confirmed':'Term sliding window and exact start-index loop not present'},
      {'topic':'target','confirmed':'RUL vector from provided Train time-to-failure values, three fault types, cap 500','not_confirmed':'Exact row alignment and all-NaN window handling code'},
      {'topic':'implementation','confirmed':'Publisher page provides paper PDF','not_confirmed':'No author or TCS companion code/supplementary artifact located in searched publisher/GitHub listings'}]
    pd.DataFrame(evidence).to_csv(OUT/'paper_claims_and_unknowns.csv',index=False)
    config={'model_training':False,'source_paper':PAPER,
      'public_implementation_search':'publisher article page and targeted GitHub/web queries for title, LSTM-MR, named authors/TCS; no verified author implementation',
      'third_party_repos':'https://github.com/ninja1mmm/2018-phm-data-challenge is an unrelated third-party challenge solution, not evidence of TV et al. implementation',
      'four_rule_comparison':'same pre-Test Train/Validation and same next strictly future same-type target; raw target <500',
      'A':'no logged fault strictly inside input window',
      'B':'target event at/before input end; impossible under strictly-future target, verified zero',
      'C':'at least two logged faults inside input window',
      'D':'at least one fault inside input window and observed sensor after it; overlaps C',
      'cue_audit':'descriptive observations, not proof of target leakage or causal fault signature'}
    (OUT/'audit_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    report=f'''# TV et al. 2018 window semantics audit — no model training

Primary source: [TV et al., 2018, §§5.1–6.1]({PAPER}). The authors say they retain FSP=1, use nine sensors, and split each Tool's data into overlapping **2,000-point windows with 500-point overlap**. They obtain each three-type RUL target vector from the provided Train TTF values. The mathematical stride of a literal 500-point overlap is **1,500 points**. The PDF contains no explicit `stride` value, no `sliding window` implementation loop, and no instruction to cut at a logged fault or maintenance boundary. It does not say that stride is 500. The approximately 0.1M / 150 numbers are descriptive, not source code.

The [publisher article page](https://www.papers.phmsociety.org/index.php/phmconf/article/view/589) provides a PDF but no listed supplemental code. Targeted searches for the exact title, `LSTM-MR`, named authors, and TCS Research found **no verified author/team public window-generation implementation**. A [third-party PHM challenge repository](https://github.com/ninja1mmm/2018-phm-data-challenge) exists, but it is not attributable to the paper authors and cannot establish their stride, clipping, target, or boundary semantics. Public implementation status: **unconfirmed**.

## Four-rule comparison on the same frozen pre-Test range

Targets are the next strictly future same-type logged fault within the same Train/Validation phase; no Validation/Test labels are used as features. Near means at least one observable type has RUL <500 raw timestamp units. All rules use 2,000 FSP=1 operating observations, Tool boundaries, and chronological split/purge boundaries. They differ only in stride and whether each fault cuts the operating sequence.

```text
{summary.to_string(index=False)}
```

Detailed Tool × phase and type-wise unique event counts are in `window_counts_by_tool_phase.csv` and `unique_events_by_variant_phase_type.csv`. For example, the exact current rule (fault boundary, stride 1,500) has **25,310 total / 9 near / 634 unique target events**; no-fault stride 500 has **77,686 / 55 / 816** on this same restricted range. The earlier dataset audit additionally found **100,199** total Tool-only stride-500 windows over the full original Train *time/FSP axis*, without reading its reserved later TTF values. That number is close to the paper's approximate 0.1M, but it does not prove their implementation.

## Fault-crossing semantics and possible information transfer

Counts of input windows containing logged faults:

```text
{summary[['variant','zero_fault_inside','one_fault_inside','two_or_more_faults_inside','post_first_fault_sensor','target_fault_at_or_before_input_end']].to_string(index=False)}
```

The categories are **not all mutually exclusive**: category C (two or more faults) is a subset of D (post-fault sensor observations). A is no fault inside. B, an input that reaches or passes its **current next target fault**, is zero by explicit timestamp assertion: target time is strictly greater than input end. Therefore removing fault boundaries does **not** leak sensor readings from the future of the chosen target event. It *does* allow the model to see that one or more **earlier** fault/maintenance cycles have occurred, through the history after those events. This is past information in deployment, but can be a shortcut if the intended task is a single run-to-failure cycle. A future exact-fault-time target from provided TTF=0 would need its own anchor convention; it was not silently mixed into this next-future target audit.

Across {len(transitions)} unique faults with adjacent retained operating observations on both sides, **{int(transitions.stage_changed.sum())}** have stage change, **{int(transitions.stage_numeric_decrease.sum())}** stage decrease, **{int((transitions.gap_seconds>1800).sum())}** operating gap >30 min, and **{int(transitions.jump_above_ordinary_p99.sum())}** have a maximum raw-nine sensor jump above the same Tool's 99th percentile for non-fault adjacent operating transitions. Median fault-crossing gap is **{float(transitions.gap_seconds.median()):.0f} seconds**. Stage and timestamp are **not** among the nine TV input sensors, but the nine raw sensor sequence can still encode an abrupt post-maintenance change. The >99th-percentile comparison is descriptive, not a fitted classifier or a claim that every jump is caused by maintenance.

Window-level cue overlap among the fault-crossing windows:

```text
{cues.to_string(index=False)}
```

## Paper-number fit versus method fit

- **Numerical fit:** On our pre-Test range, stride 500 without a fault boundary is closest of the four to 0.1M / 150, but still reaches only 77,686 / 55. Full original Train time/FSP reaches 100,199 Tool-only stride-500 windows; its near count was deliberately not opened.
- **Textual fit:** Literal `overlap of 500 points` favors stride **1,500**. The paper does not specify a fault boundary. Hence no-fault stride-1,500 is the closest *textual* reconstruction, while the question of a hidden cycle cut remains unresolved.
- **Risk:** All four rules exclude post-**target** sensor data. Removing a fault boundary mixes earlier fault cycles and introduces measurable maintenance-state cues. It is not classic future-data leakage, but is a genuine shortcut/distribution risk and may change the scientific question. Numeric closeness alone cannot resolve the author's undisclosed code.

## Recommendation

For the **next actual prognostic training rule**, choose **stride 1,500 with a fault boundary**. It honors the paper's literal 500-point overlap, keeps strict time order, and prevents the model from reading post-*earlier*-fault sensor changes as an implicit maintenance-state cue. It is a deliberately safer adaptation, **not an exact published implementation**. A Tool-only stride-1,500 construction should remain a *read-only paper-text sensitivity dataset* until the meaning of its 894 fault-crossing windows is established. No candidate fully meets every requested criterion: the safer rule has only **9** pre-Test <500 near windows, and even the Tool-only text-faithful rule has only 18. Hence do **not** start another type-wise RUL training run yet; obtain more genuine near-failure observations or change the separately justified prediction question first. Do not choose stride 500 solely because it reproduces the approximate 100k count.
'''
    (OUT/'analysis_report.md').write_text(report,encoding='utf-8')
    print(summary.to_string(index=False))
    print(cues.to_string(index=False))

if __name__=='__main__':main()
