"""Rule B/C operating-observation RUL baseline on the frozen forward split.

Reads real pre-Test raw observations and audit window metadata. Never edits Model 1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/experiments/rul_rule_b_baseline'
MODEL = ROOT / 'output/models'
AUDIT = ROOT / 'output/experiments/rul_sequence_rule_audit/window_metadata.csv.gz'
PLAN = ROOT / 'output/experiments/forward_validation_ab_24h/split_plan.json'
RAW = ROOT / 'data/processed'
SENSORS = ['IONGAUGEPRESSURE', 'ETCHBEAMVOLTAGE', 'ETCHBEAMCURRENT',
           'ETCHSUPPRESSORVOLTAGE', 'ETCHSUPPRESSORCURRENT', 'FLOWCOOLFLOWRATE',
           'FLOWCOOLPRESSURE', 'ETCHGASCHANNEL1READBACK', 'ETCHPBNGASREADBACK']
TYPES = ['pressure_low', 'pressure_high', 'leak']
LENGTH, STRIDE, STEP, SEED = 2000, 1500, 20, 42
RULES = {'B_operating_axis': None, 'C_gap_1800s': 1800}


class GRU(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.gru = nn.GRU(features, 32, batch_first=True)
        self.head = nn.Linear(32, 3)

    def forward(self, x):
        return self.head(self.gru(x)[1][-1])


def construct(plan, audit):
    """Rebuild audited windows by index, verifying identity against the saved audit."""
    sequences = {r: [] for r in RULES}
    metadata = {r: [] for r in RULES}
    for item in plan['tools']:
        tool = item['tool_id']
        tab = pq.read_table(RAW/f'{tool}_sensor_ttf.parquet',
                            columns=['time', 'FIXTURESHUTTERPOSITION', *SENSORS],
                            filters=[('time', '<', item['validation_end_exclusive_second'])])
        frame = tab.to_pandas()
        if (frame.time >= item['validation_end_exclusive_second']).any():
            raise AssertionError('Reserved Test observation read')
        op = frame.FIXTURESHUTTERPOSITION.to_numpy() == 1
        t = frame.time.to_numpy(np.int64)[op]
        sensor = frame.loc[op, SENSORS].to_numpy(np.float32)
        del frame, tab
        phase = np.select([t < item['train_end_exclusive_second'],
                           t >= item['validation_start_second']], [0, 2], default=1)
        gap = np.diff(t, prepend=t[0]); gap[0] = 0
        # The audit's fault boundary and split boundary are reproduced before
        # extracting a sequence. Only genuine time reversal splits Rule B.
        events = audit.loc[audit.tool_id.eq(tool)]
        # Source log is used only for boundary construction, never as a feature.
        fault_log = pd.read_csv(RAW/'fault_events.csv', usecols=['tool_id', 'time'])
        ft = np.sort(fault_log.loc[(fault_log.tool_id == tool) &
            (fault_log.time < item['validation_end_exclusive_second']), 'time'].unique())
        del fault_log
        prior = np.r_[t[0]-1, t[:-1]]
        fault_cross = np.searchsorted(ft, t, side='right') > np.searchsorted(ft, prior, side='right')
        base_start = (phase != np.r_[-1, phase[:-1]]) | fault_cross | (gap < 0)
        base_start[0] = True
        for rule, cutoff in RULES.items():
            starts = base_start.copy()
            if cutoff is not None:
                starts |= gap > cutoff
            boundaries = np.r_[np.flatnonzero(starts), len(t)]
            expected = events.loc[events.rule.eq(rule)]
            lookup = {}
            for row in expected.itertuples(index=False):
                key = (int(row.start_time), int(row.time), row.phase)
                lookup.setdefault(key, []).append(row)
            count_all = 0
            matched = 0
            for a, b in zip(boundaries[:-1], boundaries[1:]):
                if phase[a] == 1 or b-a < LENGTH:
                    continue
                for left in range(a, b-LENGTH+1, STRIDE):
                    right = left+LENGTH-1
                    kind = 'train' if phase[a] == 0 else 'validation'
                    key = (int(t[left]), int(t[right]), kind)
                    bucket = lookup.get(key)
                    if not bucket:
                        raise AssertionError(f'Audit window mismatch: {tool} {rule} {key}')
                    record = bucket.pop()
                    matched += 1
                    count_all += 1
                    target = np.full(3, np.nan, dtype=np.float32)
                    ids = []
                    for j, typ in enumerate(TYPES):
                        eid = getattr(record, f'event_{typ}')
                        ids.append(eid if isinstance(eid, str) else None)
                        if ids[-1] is not None:
                            fault_time = int(eid.rsplit('|', 1)[-1])
                            target[j] = (fault_time-int(t[right]))/3600
                    if not np.isfinite(target).any():
                        continue
                    selected = sensor[left:right+1:STEP]
                    if selected.shape != (100, 9):
                        raise AssertionError('Sequence shape changed')
                    # No interpolation. Nonfinite raw sensor input is excluded
                    # and counted explicitly; the audit's time-only coverage is
                    # therefore an upper bound on actual training coverage.
                    if not np.isfinite(selected).all():
                        continue
                    internal = gap[left+1:right+1]
                    max_gap = int(internal.max())
                    if max_gap != int(record.max_operating_gap_s):
                        raise AssertionError('Maximum gap metadata mismatch')
                    sequences[rule].append(selected.copy())
                    metadata[rule].append({'tool_id': tool, 'phase': kind,
                        'start_time': int(t[left]), 'time': int(t[right]),
                        'elapsed_seconds': int(t[right]-t[left]),
                        'max_internal_gap_s': max_gap,
                        'large_gap_count_over_30min': int((internal > 1800).sum()),
                        **{f'event_id_{typ}': ids[j] for j, typ in enumerate(TYPES)},
                        **{f'ttf_h_{typ}': target[j] for j, typ in enumerate(TYPES)}})
            if matched != len(expected) or any(lookup.values()):
                raise AssertionError(f'Audit coverage mismatch {tool} {rule}: {matched}/{len(expected)}')
            print(tool, rule, 'audited', count_all, 'usable', len(metadata[rule]), flush=True)
        del sensor, t
    return {r: np.stack(sequences[r]).astype(np.float32) for r in RULES}, \
           {r: pd.DataFrame(metadata[r]) for r in RULES}


def scale_rule(x, meta, with_time):
    train = meta.phase.eq('train').to_numpy()
    # Fixed 100-step reduction of each 2,000-point observation window, as in
    # the existing paper baseline; metadata still describes all 2,000 points.
    flat = x[train].reshape(-1, 9)
    center = np.median(flat, axis=0).astype(np.float32)
    iqr = (np.percentile(flat, 75, axis=0)-np.percentile(flat, 25, axis=0)).astype(np.float32)
    iqr = np.where(iqr < 1e-6, 1, iqr).astype(np.float32)
    scaled = ((x-center)/iqr).astype(np.float32)
    definition = {'sensor_center': center.tolist(), 'sensor_iqr': iqr.tolist(),
                  'sensor_sample_step': STEP, 'time_features': []}
    if with_time:
        # Log1p keeps the 11-day tail finite. Both stats are Train-only.
        time = np.log1p(meta[['elapsed_seconds','max_internal_gap_s']].to_numpy(np.float32))
        tc = np.median(time[train], axis=0).astype(np.float32)
        ts = (np.percentile(time[train],75,axis=0)-np.percentile(time[train],25,axis=0)).astype(np.float32)
        ts = np.where(ts < 1e-6, 1, ts).astype(np.float32)
        extra = ((time-tc)/ts).astype(np.float32)
        scaled = np.concatenate([scaled, np.broadcast_to(extra[:,None,:],
                    (len(extra), scaled.shape[1], 2)).copy()], axis=2)
        definition['time_features'] = ['log1p_elapsed_seconds','log1p_max_internal_gap_s']
        definition['time_center'] = tc.tolist()
        definition['time_iqr'] = ts.tolist()
    return scaled, definition


def train_gru(x, meta, name):
    train = meta.phase.eq('train').to_numpy()
    yt = meta.loc[train, [f'ttf_h_{t}' for t in TYPES]].to_numpy(np.float32)
    if any(np.isfinite(yt[:,j]).sum() == 0 for j in range(3)):
        raise AssertionError('No training targets for a fault type')
    mask = np.isfinite(yt).astype(np.float32)
    target = np.log1p(np.nan_to_num(yt, nan=0)).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(x[train]), torch.from_numpy(target), torch.from_numpy(mask))
    loader = DataLoader(ds, batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    torch.manual_seed(SEED)
    model = GRU(x.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    for epoch in range(5):
        model.train(); losses=[]
        for xb, yb, mb in loader:
            opt.zero_grad()
            squared = (model(xb)-yb).square()*mb
            loss = (squared.sum(dim=0)/mb.sum(dim=0).clamp(min=1)).mean()
            loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        print(name, 'epoch', epoch+1, 'loss', np.mean(losses), flush=True)
    return model


def predict(model, x):
    model.eval(); chunks=[]
    with torch.no_grad():
        for a in range(0, len(x), 512):
            chunks.append(model(torch.from_numpy(x[a:a+512])).numpy())
    return np.maximum(0, np.expm1(np.vstack(chunks)))


def metric(y, p):
    err = np.abs(y-p)
    denom = np.abs(y)+np.abs(p)
    smape = np.where(denom > 1e-9, 2*err/np.maximum(denom, 1e-9), 0)
    return {'mae_h': float(np.mean(err)), 'rmse_h': float(np.sqrt(np.mean((y-p)**2))),
            'r2': float(r2_score(y,p)) if len(y)>1 else np.nan,
            'smape_pct': float(np.mean(smape)*100),
            'within_1h': float(np.mean(err<=1)), 'within_2h': float(np.mean(err<=2))}


def evaluate(name, meta, prediction, medians):
    valid = meta.phase.eq('validation').to_numpy()
    frame = meta.loc[valid].reset_index(drop=True)
    metrics=[]; event=[]; tool=[]; folds=[]; rows=[]
    for j, typ in enumerate(TYPES):
        known = frame[f'ttf_h_{typ}'].notna().to_numpy()
        if not known.any():
            continue
        part = frame.loc[known, ['tool_id','start_time','time','elapsed_seconds',
                'max_internal_gap_s', 'large_gap_count_over_30min',
                f'event_id_{typ}', f'ttf_h_{typ}']].copy()
        part.columns = ['tool_id','start_time','time','elapsed_seconds','max_internal_gap_s',
                        'large_gap_count_over_30min','event_id','actual_ttf_h']
        part['fault_type'] = typ
        part['condition'] = name
        part['predicted_ttf_h'] = prediction[known,j]
        part['dummy_ttf_h'] = medians[j]
        part['abs_error_h'] = np.abs(part.actual_ttf_h-part.predicted_ttf_h)
        rows.append(part)
        per_event = part.groupby('event_id').abs_error_h.mean()
        metrics.append({'condition': name, 'fault_type': typ,
            'validation_samples': len(part), 'validation_events': part.event_id.nunique(),
            'event_equal_mae_h': float(per_event.mean()),
            **metric(part.actual_ttf_h.to_numpy(), part.predicted_ttf_h.to_numpy())})
        event.extend({'condition':name,'fault_type':typ,'event_id':eid,'event_mae_h':float(v)}
                     for eid,v in per_event.items())
        for tool_id,g in part.groupby('tool_id'):
            tool.append({'condition':name,'fault_type':typ,'tool_id':tool_id,'events':g.event_id.nunique(),
                         'samples':len(g),'mae_h':float(g.abs_error_h.mean()),
                         'event_equal_mae_h':float(g.groupby('event_id').abs_error_h.mean().mean())})
        # Time-ordered validation thirds, used for stability reporting only.
        unique = np.sort(part.event_id.unique())
        when = {eid:int(eid.rsplit('|',1)[-1]) for eid in unique}
        bounds = np.quantile(list(when.values()), [1/3,2/3], method='nearest')
        part['fold'] = 1+np.searchsorted(bounds, part.event_id.map(when).to_numpy(), side='right')
        for fold,g in part.groupby('fold'):
            folds.append({'condition':name,'fault_type':typ,'fold':int(fold),
                          'events':g.event_id.nunique(),'samples':len(g),
                          'mae_h':float(g.abs_error_h.mean()),
                          'event_equal_mae_h':float(g.groupby('event_id').abs_error_h.mean().mean())})
    return metrics,event,tool,folds,rows


def main():
    model_names = ['rul_rule_b_raw9_gru.pt','rul_rule_b_raw9_time_gru.pt','rul_rule_c1800_raw9_gru.pt']
    if OUT.exists() or any((MODEL/n).exists() for n in model_names):
        raise FileExistsError('Refusing to overwrite an existing experiment or model')
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    if len(plan['tools']) != 20:
        raise AssertionError('Frozen 20-tool split changed')
    audit = pd.read_csv(AUDIT, compression='gzip', low_memory=False)
    audit = audit.loc[audit.rule.isin(RULES)].copy()
    x_by, meta_by = construct(plan, audit)
    torch.set_num_threads(4); np.random.seed(SEED); torch.manual_seed(SEED)
    OUT.mkdir(parents=True); MODEL.mkdir(parents=True, exist_ok=True)
    all_metrics=[]; all_event=[]; all_tool=[]; all_folds=[]; all_rows=[]
    config={'source':'real pre-Test raw sensor parquet and frozen RUL rule audit',
            'sensors':SENSORS,'length_observations':LENGTH,'overlap_observations':500,
            'stride_observations':STRIDE,'gru_sample_every_nth_observation':STEP,
            'rules':{'B':'FSP=1, split at Tool, any fault, frozen phase, time reversal',
                     'C':'B plus operating gap >1800 seconds'},
            'target':'three separate next-same-type-fault hours, masked if absent in same split',
            'mask_loss':'log1p hours, masked MSE normalized by valid count per type',
            'split_plan':str(PLAN),'seed':SEED,'architecture':'1-layer GRU 32 hidden, linear 3 head, Adam .001, 5 epochs',
            'reserved_test':'not read','model1':'unchanged'}
    conditions=[('B_raw9','B_operating_axis',False,model_names[0]),
                ('B_raw9_time','B_operating_axis',True,model_names[1]),
                ('C1800_raw9','C_gap_1800s',False,model_names[2])]
    for condition,rule,with_time,filename in conditions:
        meta=meta_by[rule]
        scaled,scaler=scale_rule(x_by[rule],meta,with_time)
        train=meta.phase.eq('train').to_numpy(); valid=meta.phase.eq('validation').to_numpy()
        medians=np.array([np.nanmedian(meta.loc[train,f'ttf_h_{typ}']) for typ in TYPES],dtype=np.float32)
        yv=meta.loc[valid,[f'ttf_h_{typ}' for typ in TYPES]].to_numpy(np.float32)
        dummy=np.broadcast_to(medians,(len(yv),3)).copy()
        model=train_gru(scaled,meta,condition)
        result=predict(model,scaled[valid])
        torch.save({'state_dict':model.state_dict(),'sensor_list':SENSORS,'scaler':scaler,
                    'condition':condition,'rule':rule,'sequence_length':LENGTH,
                    'sampling_step':STEP,'target_transform':'log1p hours'}, MODEL/filename)
        for label,pred in [(condition+'_Dummy',dummy),(condition,result)]:
            m,e,t,f,r=evaluate(label,meta,pred,medians)
            all_metrics+=m; all_event+=e; all_tool+=t; all_folds+=f; all_rows+=r
        meta.to_csv(OUT/f'{rule}_sequence_metadata.csv.gz',index=False,compression='gzip')
        config[condition]={'rule':rule,'model_file':str(MODEL/filename),
            'scaler':scaler,'train_sequences':int(train.sum()),
            'validation_sequences':int(valid.sum()),
            'train_target_median_hours':dict(zip(TYPES,medians.astype(float).tolist()))}
        print(condition,'finished',flush=True)
    pd.DataFrame(all_metrics).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(all_event).to_csv(OUT/'event_metrics.csv',index=False)
    pd.DataFrame(all_tool).to_csv(OUT/'tool_metrics.csv',index=False)
    pd.DataFrame(all_folds).to_csv(OUT/'fold_metrics.csv',index=False)
    pd.concat(all_rows,ignore_index=True).to_csv(OUT/'validation_predictions.csv.gz',index=False,compression='gzip')
    (OUT/'experiment_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    print(pd.DataFrame(all_metrics).to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
