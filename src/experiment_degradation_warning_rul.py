"""Out-of-sample degradation warning followed by type-specific RUL baselines.

The 24-hour pre-fault proxy is not an observed degradation-onset label.
Only the pre-existing forward Train/Validation partitions are read; Test is sealed.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_rul_paper_multioutput import (FAULTS, LENGTH, MODEL_DIR, PLAN, RAW,
    RESAMPLE_STEP, SEED, SENSORS, STEPS, STRIDE, TYPES, read_raw_before_boundary)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/experiments/degradation_warning_rul'
HORIZON = 86400


class GRURegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(len(SENSORS), 24, batch_first=True)
        self.head = nn.Linear(24, 1)

    def forward(self, x):
        return self.head(self.gru(x)[1][-1]).squeeze(1)


def summary(x):
    return np.concatenate((x[:, -1], x.mean(1), x.std(1),
                           x[:, -1] - x[:, 0]), axis=1)


def build_dataset(plan, faults):
    seq, rows = [], []
    for item in plan['tools']:
        tool = item['tool_id']
        frame = read_raw_before_boundary(RAW / f'{tool}_sensor_ttf.parquet',
                                         item['validation_end_exclusive_second'])
        frame = frame.sort_values('time', kind='stable')
        operating = frame.FIXTURESHUTTERPOSITION.to_numpy() == 1
        times = frame.time.to_numpy(np.int64)[operating]
        values = frame.loc[operating, SENSORS].to_numpy(np.float32)
        finite = np.isfinite(values).all(1)
        log = faults[(faults.tool_id == tool) &
                     (faults.time < item['validation_end_exclusive_second'])]
        ft = {typ: np.sort(log.loc[log.fault_type == typ, 'time'].to_numpy(np.int64))
              for typ in TYPES}
        train_end = int(item['train_end_exclusive_second'])
        val_start = int(item['validation_start_second'])
        val_end = int(item['validation_end_exclusive_second'])
        # Frozen, tool-local chronological division of Train. A 24-hour embargo
        # on both sides prevents the detector's 24h target crossing into RUL fit.
        cut = int(times[0] + .70 * (train_end - times[0])) if len(times) else 0
        for left in range(0, max(0, len(times)-LENGTH+1), STRIDE):
            right = left + LENGTH - 1
            start, anchor = int(times[left]), int(times[right])
            if anchor < cut-HORIZON:
                phase, end = 'detector_train', cut-HORIZON
            elif start >= cut+HORIZON and anchor < train_end:
                phase, end = 'rul_train', train_end
            elif start >= val_start and anchor < val_end:
                phase, end = 'validation', val_end
            else:
                continue
            if not finite[left:right+1].all():
                continue
            selected = values[left:right+1:RESAMPLE_STEP]
            if selected.shape != (STEPS, len(SENSORS)):
                raise AssertionError('Sequence shape mismatch')
            row = {'tool_id':tool, 'phase':phase, 'start_time':start,
                   'time':anchor, 'phase_end':end,
                   'max_operating_gap_s':int(np.diff(times[left:right+1]).max())}
            for typ in TYPES:
                event_times = ft[typ]
                pos = int(np.searchsorted(event_times, anchor, side='right'))
                future = int(event_times[pos]) if pos < len(event_times) else None
                previous = int(event_times[pos-1]) if pos else None
                eligible = (future is not None and future < end and
                            (previous is None or start > previous))
                row[f'event_id_{typ}'] = f'{tool}|{typ}|{future}' if eligible else None
                row[f'ttf_h_{typ}'] = (future-anchor)/3600 if eligible else np.nan
                row[f'label_{typ}'] = (int(future-anchor <= HORIZON)
                                      if anchor+HORIZON < end and eligible else
                                      (0 if anchor+HORIZON < end else np.nan))
            seq.append(selected)
            rows.append(row)
        print(tool, 'windows', len(rows), flush=True)
    x = np.stack(seq).astype(np.float32)
    meta = pd.DataFrame(rows)
    return x, meta


def regression_metrics(y, p, events):
    delta = np.abs(y-p)
    return {'rows':len(y), 'events':len(np.unique(events)),
            'mae_h':float(delta.mean()), 'rmse_h':float(np.sqrt(np.mean((y-p)**2))),
            'r2':float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)) if len(y)>1 and np.var(y)>0 else None,
            'smape_pct':float(np.mean(200*delta/np.maximum(np.abs(y)+np.abs(p),1e-6))),
            'within_1h':float(np.mean(delta<=1)), 'within_2h':float(np.mean(delta<=2)),
            'event_equal_mae_h':float(pd.DataFrame({'event':events,'err':delta}).groupby('event').err.mean().mean())}


def fit_gru(train_x, train_y, center, scale):
    torch.manual_seed(SEED)
    model = GRURegressor()
    optim = torch.optim.Adam(model.parameters(), lr=.001)
    scaled = np.clip((train_x-center)/scale, -20, 20).astype(np.float32)
    data = TensorDataset(torch.from_numpy(scaled),
                         torch.from_numpy(np.log1p(train_y).astype(np.float32)))
    loader = DataLoader(data, batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    for epoch in range(5):
        model.train()
        for a,b in loader:
            optim.zero_grad(); loss=((model(a)-b)**2).mean(); loss.backward(); optim.step()
    return model


def predict_gru(model, x, center, scale):
    model.eval(); result=[]
    with torch.no_grad():
        for i in range(0,len(x),512):
            a=np.clip((x[i:i+512]-center)/scale,-20,20).astype(np.float32)
            result.append(model(torch.from_numpy(a)).numpy())
    return np.maximum(0,np.expm1(np.concatenate(result)))


def main():
    if OUT.exists():
        raise FileExistsError(f'Refusing to overwrite {OUT}')
    np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    faults=pd.read_csv(FAULTS).drop_duplicates(['tool_id','time','fault_type'])
    x,meta=build_dataset(plan,faults)
    if not set(meta.phase).issuperset({'detector_train','rul_train','validation'}):
        raise AssertionError('Missing phase')
    if any(s in ' '.join(SENSORS).lower() for s in ('ttf','fault','time','event')):
        raise AssertionError('Target or identifier in sensor feature')
    OUT.mkdir(parents=True); MODEL_DIR.mkdir(parents=True,exist_ok=True)
    flat=summary(x)
    feature_names=[f'{s}_{stat}' for stat in ('last','mean','std','end_minus_start') for s in SENSORS]
    config={'degradation_proxy_label':'next same-type fault within 24h; only full 24h observable',
            'warning_threshold':.5,'train_internal_split':'70% of tool-local Train elapsed time with 24h embargo both sides',
            'frozen_external_split':str(PLAN),'sensor_columns':SENSORS,
            'feature_names':feature_names,'operating_filter':'FIXTURESHUTTERPOSITION == 1',
            'sequence_observations':LENGTH,'stride_observations':STRIDE,
            'sequence_downsample':RESAMPLE_STEP,'missing_handling':'no imputation, finite full windows only; stops create time gaps',
            'random_state':SEED,'reserved_test':'not read','model1':'untouched',
            'rul_ablation':'A all RUL-fit Train windows, B only after first out-of-sample detector warning; both scored on same B Validation windows'}
    (OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    detector_rows=[];warning_rows=[];reg_rows=[];detail=[];fold_rows=[];tool_rows=[]; event_rows=[]
    for typ in TYPES:
        key=f'label_{typ}'; y=meta[key].to_numpy()
        dtrain=(meta.phase.eq('detector_train').to_numpy() & np.isfinite(y))
        dval=(meta.phase.eq('validation').to_numpy() & np.isfinite(y))
        if np.unique(y[dtrain]).size<2:
            detector_rows.append({'fault_type':typ,'status':'one class in detector train'});continue
        detector=RandomForestClassifier(n_estimators=100,max_depth=8,min_samples_leaf=10,
                   class_weight='balanced_subsample',n_jobs=4,random_state=SEED)
        detector.fit(flat[dtrain],y[dtrain].astype(int))
        joblib.dump({'model':detector,'features':feature_names,'config':config},
                    MODEL_DIR/f'degradation_rf_{typ}.joblib')
        prob=detector.predict_proba(flat)[:,1]
        active=prob>=.5
        if dval.sum() and np.unique(y[dval]).size==2:
            actual=y[dval].astype(int); predicted=active[dval].astype(int)
            detector_rows.append({'fault_type':typ,'validation_rows':int(dval.sum()),
              'positive_rows':int(actual.sum()),'roc_auc':float(roc_auc_score(actual,prob[dval])),
              'ap':float(average_precision_score(actual,prob[dval])),
              'precision':float(precision_score(actual,predicted,zero_division=0)),
              'recall':float(recall_score(actual,predicted,zero_division=0)),
              'f1':float(f1_score(actual,predicted,zero_division=0)),
              'balanced_accuracy':float(balanced_accuracy_score(actual,predicted)),
              'normal_false_warning_rows':int(np.sum((actual==0)&(predicted==1)))})
        meta[f'prob_{typ}']=prob
        meta[f'warning_{typ}']=active
        # First warning in the 24h pre-fault proxy interval. Event assignment
        # uses fault timestamps only for retrospective target/evaluation linkage.
        event_key=f'event_id_{typ}'
        for phase in ('rul_train','validation'):
            subset=meta[(meta.phase==phase)&meta[event_key].notna()].copy()
            ids=subset[event_key].unique()
            for eid in ids:
                group=subset[subset[event_key]==eid]
                fault_time=int(eid.split('|')[-1])
                window=group[(group.time>=fault_time-HORIZON)&(group.time<fault_time)]
                hit=window[window[f'warning_{typ}']]
                first=int(hit.time.min()) if len(hit) else None
                warning_rows.append({'fault_type':typ,'phase':phase,'event_id':eid,
                    'tool_id':eid.split('|')[0],'first_warning':first,
                    'lead_time_h':(fault_time-first)/3600 if first is not None else np.nan,
                    'eligible_window_count':len(window),'warned':first is not None})
        warnings=pd.DataFrame(warning_rows)
        for phase in ('rul_train','validation'):
            map_first=warnings[(warnings.fault_type==typ)&(warnings.phase==phase)].set_index('event_id').first_warning
            meta.loc[meta.phase==phase,f'postwarning_{typ}']=(meta.loc[meta.phase==phase,event_key].map(map_first).le(meta.loc[meta.phase==phase,'time']) &
                meta.loc[meta.phase==phase,event_key].map(map_first).notna())
        train_base=(meta.phase.eq('rul_train').to_numpy() & np.isfinite(meta[f'ttf_h_{typ}'].to_numpy()))
        post_mask=meta[f'postwarning_{typ}'].fillna(False).astype(bool).to_numpy()
        eval_mask=(meta.phase.eq('validation').to_numpy() & post_mask)
        train_post=(train_base & post_mask)
        event_rows.append({'fault_type':typ,'train_all_events':meta.loc[train_base,event_key].nunique(),
            'train_postwarning_events':meta.loc[train_post,event_key].nunique(),
            'validation_all_events':meta.loc[(meta.phase=='validation')&meta[event_key].notna(),event_key].nunique(),
            'validation_warned_events':meta.loc[eval_mask,event_key].nunique(),
            'train_all_rows':int(train_base.sum()),'train_postwarning_rows':int(train_post.sum()),
            'validation_postwarning_rows':int(eval_mask.sum())})
        if not eval_mask.any():continue
        yval=meta.loc[eval_mask,f'ttf_h_{typ}'].to_numpy(np.float32)
        idsval=meta.loc[eval_mask,event_key].to_numpy()
        xval=x[eval_mask]
        for arm,mask in (('without_warning',train_base),('after_warning',train_post)):
            if mask.sum()<20:
                reg_rows.append({'fault_type':typ,'arm':arm,'status':'insufficient training windows'});continue
            yt=meta.loc[mask,f'ttf_h_{typ}'].to_numpy(np.float32)
            center=np.median(x[mask].reshape(-1,len(SENSORS)),axis=0).astype(np.float32)
            iqr=np.percentile(x[mask].reshape(-1,len(SENSORS)),75,axis=0)-np.percentile(x[mask].reshape(-1,len(SENSORS)),25,axis=0)
            scale=np.where(iqr<1e-6,1,iqr).astype(np.float32)
            models={}
            models['Dummy']=np.full(len(yval),np.median(yt),dtype=np.float32)
            tree=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=31,learning_rate=.05,
                       l2_regularization=1,early_stopping=False,random_state=SEED)
            tree.fit(flat[mask],np.log1p(yt))
            models['Tree_HGB']=np.maximum(0,np.expm1(tree.predict(flat[eval_mask])))
            joblib.dump({'model':tree,'features':feature_names,'config':config},
                        MODEL_DIR/f'rul_degradation_{typ}_{arm}_tree.joblib')
            gru=fit_gru(x[mask],yt,center,scale)
            models['Sequence_GRU']=predict_gru(gru,xval,center,scale)
            torch.save({'state_dict':gru.state_dict(),'center':center,'scale':scale,
                        'sensors':SENSORS,'config':config},
                       MODEL_DIR/f'rul_degradation_{typ}_{arm}_gru.pt')
            joblib.dump({'center':center,'scale':scale,'sensors':SENSORS},
                        MODEL_DIR/f'rul_degradation_{typ}_{arm}_scaler.joblib')
            valmeta=meta.loc[eval_mask,['tool_id','time',event_key]].copy()
            valmeta['fault_time']=valmeta[event_key].str.split('|').str[-1].astype(int)
            unique_times=np.sort(valmeta.fault_time.unique())
            boundaries=np.quantile(unique_times,[1/3,2/3]) if len(unique_times)>=3 else []
            folds=np.searchsorted(boundaries,valmeta.fault_time.to_numpy(),side='right')+1
            for model_name,p in models.items():
                m=regression_metrics(yval,p,idsval)
                reg_rows.append({'fault_type':typ,'arm':arm,'model':model_name,**m})
                err=np.abs(yval-p)
                for i in range(len(p)):
                    detail.append({'fault_type':typ,'arm':arm,'model':model_name,
                        'tool_id':valmeta.iloc[i].tool_id,'time':int(valmeta.iloc[i].time),
                        'event_id':idsval[i],'actual_ttf_h':float(yval[i]),
                        'predicted_ttf_h':float(p[i]),'abs_error_h':float(err[i]),'fold':int(folds[i])})
                for tool in np.unique(valmeta.tool_id):
                    sel=(valmeta.tool_id.to_numpy()==tool)
                    tool_rows.append({'fault_type':typ,'arm':arm,'model':model_name,'tool_id':tool,
                        **regression_metrics(yval[sel],p[sel],idsval[sel])})
                for fold in np.unique(folds):
                    sel=(folds==fold)
                    fold_rows.append({'fault_type':typ,'arm':arm,'model':model_name,'fold':int(fold),
                        **regression_metrics(yval[sel],p[sel],idsval[sel])})
        print(typ,'detector AP',detector_rows[-1].get('ap'),'post train',int(train_post.sum()),'val',int(eval_mask.sum()),flush=True)
    pd.DataFrame(detector_rows).to_csv(OUT/'detector_metrics.csv',index=False)
    pd.DataFrame(warning_rows).to_csv(OUT/'warning_events.csv',index=False)
    pd.DataFrame(event_rows).to_csv(OUT/'event_availability.csv',index=False)
    pd.DataFrame(reg_rows).to_csv(OUT/'rul_metrics.csv',index=False)
    pd.DataFrame(detail).to_csv(OUT/'rul_predictions.csv.gz',index=False,compression='gzip')
    pd.DataFrame(fold_rows).to_csv(OUT/'fold_metrics.csv',index=False)
    pd.DataFrame(tool_rows).to_csv(OUT/'tool_metrics.csv',index=False)
    meta[['tool_id','phase','start_time','time','max_operating_gap_s',
          *[f'label_{t}' for t in TYPES],*[f'prob_{t}' for t in TYPES],
          *[f'warning_{t}' for t in TYPES]]].to_csv(OUT/'detector_predictions.csv.gz',index=False,compression='gzip')
    print('Saved',OUT,flush=True)


if __name__=='__main__':
    main()
