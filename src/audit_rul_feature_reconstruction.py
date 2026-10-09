"""Build a source-linked feature inventory. This script never fits a model."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/experiments/rul_feature_reconstruction'
BASE=ROOT/'output/baseline_recovery/reproduction_report.json'
RAW=['IONGAUGEPRESSURE','ETCHBEAMVOLTAGE','ETCHBEAMCURRENT',
     'ETCHSUPPRESSORVOLTAGE','ETCHSUPPRESSORCURRENT','FLOWCOOLFLOWRATE',
     'FLOWCOOLPRESSURE','ETCHGASCHANNEL1READBACK','ETCHPBNGASREADBACK']
URLS={
 'Wu 2021':'https://www.researchgate.net/publication/348862878_Remaining_useful_life_prediction_for_ion_etching_machine_cooling_system_using_deep_recurrent_neural_network-based_approaches',
 'TV et al. 2018':'https://papers.phmsociety.org/index.php/phmconf/article/download/589/phmc_18_589',
 'Huang et al. 2018':'https://papers.phmsociety.org/index.php/phmconf/article/download/590/phmc_18_590'}


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    if OUT.exists():raise FileExistsError(f'Refusing to overwrite {OUT}')
    features=json.loads(BASE.read_text(encoding='utf-8'))['features']
    assert len(features)==85 and len(set(features))==85
    OUT.mkdir(parents=True)
    rows=[]
    for name in features:
        if name in RAW: group='paper raw sensor';status='A: directly available'
        elif name in ('stage','recipe','recipe_step','tool_number'):
            group='categorical/process/identity';status='C: causal but categorical encoding or identity policy needed'
        elif name=='FIXTURESHUTTERPOSITION':
            group='operating-state marker';status='C: available; constant after FSP=1 screening'
        elif name in ('time_diff_seconds','restart_after_stop','session_age_minutes','history_minutes'):
            group='past time/gap';status='C: causal; rebuild for chosen 4s or 1min grid'
        elif name.endswith('_changed'):
            group='process change';status='C: causal; rebuild for chosen grid'
        elif name in ('pressure_minus_flow','pressure_div_flow'):
            group='sensor arithmetic';status='C: causal, not paper MLP residual'
        elif name.endswith(('_std_1m','_min_1m','_max_1m')):
            group='within-minute aggregation';status='C: causal at minute end; 4s recomputation needed'
        elif any(token in name for token in ('_diff','_per_hour','_mean_','_std_','_slope_','_ewma_')):
            group='past-window derived';status='C: causal as implemented on 1min session; 4s recomputation needed'
        else:
            group='other contemporaneous sensor/process';status='C: available, inspect scaling'
        rows.append({'feature':name,'group':group,'reconstruction_class':status,
                     'existing_baseline_grid':'one-minute aggregate',
                     'model_input_future_label_leak':'none identified from source constructor',
                     'paper_exact_feature':'yes' if name in RAW else 'no/unknown'})
    write_csv(OUT/'existing_85_feature_inventory.csv',rows)
    papers=[
      {'paper':'Wu 2021','raw_sensor':'Specific final numeric list unconfirmed; FlowCool flow q and pressure p explicitly used',
       'removed_sensor':'missing/repeated rows removed; specific numeric sensors removed unconfirmed',
       'sample_screening':'FSP=1; missing/repeated removed; fault/normal training proxy RUL<500 samples vs >6250 samples',
       'downtime':'FSP !=1 excluded; detailed cross-stop sequence handling unconfirmed',
       'extra_feature':'pressure residual r=MLP(q)-p, 1 scalar',
       'construction':'Train MLP on normal data, input flow q, output pressure p; two hidden layers 10/50, tanh, MSE; exact training split/weights/initialization unconfirmed',
       'rolling_mean_std_slope_diff':'No explicit manual rolling mean/std/slope/diff as RUL inputs; RF probability curve smoothed with 150-sample window',
       'mlp_feature':'yes; q -> estimated p -> residual',
       'scaling':'preprocessing says standardization/normalization, exact fitted transform unconfirmed',
       'fault_type_difference':'Separate RUL models and sequence lengths F1/F2/F3=45/60/45; same residual formulation',
       'selection':'MI for categorical variables; stage/recipe/recipe-step removed, FSP retained; full numeric selection unconfirmed',
       'final_sequence_feature':'Selected numeric/process variables + one MLP residual; exact list/count unconfirmed',
       'source':URLS['Wu 2021']},
      {'paper':'TV et al. 2018','raw_sensor':'9 named numeric inputs, exact list in local raw_sensor_list.csv; FSP=1 screening',
       'removed_sensor':'Other Table 1 fields including stage, Lot, runnum, recipe, recipe step, tilt, rotation, timers',
       'sample_screening':'FSP=1; 2000-point windows, 500 overlap; far-RUL clipping and sampling',
       'downtime':'FSP!=1 removed; no explicit maximum inter-observation gap rule',
       'extra_feature':'No hand-crafted scalar; downsample d1=2 then T2V d2=10',
       'construction':'T2V concatenates 10 consecutive downsampled 9-vectors => 90 values per recurrent step; 100 steps/window',
       'rolling_mean_std_slope_diff':'No manual rolling mean/std/slope/diff described',
       'mlp_feature':'No','scaling':'Provided sensors observed Z-normalized; RUL clip ru=500 and divide by ru',
       'fault_type_difference':'Shared 3-output RNN, same inputs for all 3 fault modes',
       'selection':'9 named variables from Table 1; selection rationale physical function; other fields excluded',
       'final_sequence_feature':'9 raw channels; transformed to 100x90 by d1=2,d2=10',
       'source':URLS['TV et al. 2018']},
      {'paper':'Huang et al. 2018','raw_sensor':'Sensor measurements (individual numeric names unconfirmed) plus operating-time variables and one-hot FSP',
       'removed_sensor':'Recipe/recipe step and other categorical variables excluded; exact numeric exclusion list unconfirmed',
       'sample_screening':'missing rows removed; large-gap run-to-fault sequence truncated; remaining >=3000 historical observations',
       'downtime':'Large gap truncation; threshold in seconds unconfirmed; target converted to samples-to-fault',
       'extra_feature':'Detector residual r=q-f_hat(p), 1 scalar; not explicitly final RUL input',
       'construction':'Normal-data relation pressure p -> estimated flow q; model family/architecture and fitting details unconfirmed',
       'rolling_mean_std_slope_diff':'No explicit manual rolling mean/std/slope/diff inputs',
       'mlp_feature':'No; MLP is RUL comparator, not stated residual generator',
       'scaling':'Provided sensor variables already anonymized/normalized; no additional sensor normalization',
       'fault_type_difference':'Three separate LSTM RUL models; shared 21-parameter input description',
       'selection':'Sensor + operating-time + one-hot FSP; exact numeric list unconfirmed',
       'final_sequence_feature':'21 values/measurement after FSP one-hot; detector residual inclusion in RUL input unconfirmed',
       'source':URLS['Huang et al. 2018']}
    ]
    write_csv(OUT/'paper_feature_matrix.csv',papers)
    sets=[
      {'set':'A','channels_per_timestep':9,'components':'9 raw numeric sensors','paper_match':'TV 2018 numeric inputs','advantage':'fewest channels and closest published raw-sensor baseline','limitation':'no explicit flow-pressure relation; cross-stop sequence policy still necessary'},
      {'set':'B','channels_per_timestep':10,'components':'Set A + Wu residual MLP(FLOWCOOLFLOWRATE)-FLOWCOOLPRESSURE','paper_match':'Wu 2021 feature equation, implementation only approximate without original weights and full fitting protocol','advantage':'adds physics-inspired normal relation in one channel','limitation':'normal-only Train fitting and frozen validation transform required; not identical pretrained paper MLP'},
      {'set':'C','channels_per_timestep':85,'components':'Original Model 1 85 columns, including 9 raw; use one-minute semantics or causally recompute all on 4s grid','paper_match':'Project baseline, not a paper-exact sequence feature set','advantage':'tests existing process and history information','limitation':'high redundancy; 1min aggregated columns cannot be copied unchanged onto raw 4s samples; Tool identity and categoricals require explicit policy'}]
    write_csv(OUT/'proposed_feature_sets.csv',sets)
    write_csv(OUT/'raw_sensor_list.csv',[{'sensor':s} for s in RAW])
    (OUT/'source_urls.json').write_text(json.dumps(URLS,indent=2),encoding='utf-8')
    print('saved',OUT,'features',len(rows))


if __name__=='__main__':main()
