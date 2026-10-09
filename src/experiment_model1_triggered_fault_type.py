"""Frozen Model 1 alert-time next-fault type: Dummy, HGB summaries, GRU raw sequence."""
from __future__ import annotations
import hashlib,json,pickle
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (balanced_accuracy_score,confusion_matrix,
                             f1_score,precision_recall_fscore_support,accuracy_score)
from evaluate_model1_triggered_rul import select_manager_notifications
from experiment_event_balanced_rul import ROOT,MODELS,SmallGRU
from train_simple_full_fault_rul import RAW,PLAN,SENSORS

OUT=ROOT/'output/experiments/model1_triggered_fault_type'
SCORES=ROOT/'output/experiments/forward_validation_ab_24h/validation_probabilities_a.npz'
SOURCE=ROOT/'output/experiments/forward_validation_ab_24h'
RUL_ALERT=ROOT/'output/experiments/model1_triggered_rul/manager_notifications.csv.gz'
BASE_PREPARED=ROOT/'output/baseline_recovery/prepared'
MODEL1=MODELS/'xgb_24h_forward_a.joblib'
TYPES=['pressure_low','pressure_high','leak']
SEED=42

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def create_data():
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    provenance=json.loads((SOURCE/'training_a.json').read_text(encoding='utf-8'))
    assert digest(MODEL1)==provenance['model_sha256']
    assert digest(SCORES)==provenance['validation_score_sha256']
    model1=joblib.load(MODEL1)
    assert len(model1['features'])==85
    faults=pd.read_csv(RAW/'fault_events.csv').drop_duplicates(['tool_id','time','fault_type'])
    with np.load(SCORES) as z:
        vt=z['time'].astype(np.int64);vc=z['tool_code'];vp=z['probability']
    result={p:{'sequences':[],'meta':[],'counts':[]} for p in ('train','validation')}
    for code,item in enumerate(plan['tools']):
        tool=item['tool_id'];vstart=int(item['validation_start_second'])
        vend=int(item['validation_end_exclusive_second']);tend=int(item['train_end_exclusive_second'])
        raw=pq.read_table(RAW/f'{tool}_sensor_ttf.parquet',columns=['time']+SENSORS,
                          filters=[('time','<',vend)]).to_pandas()
        sensor_t=raw.time.to_numpy(dtype=np.int64)
        sensor_x=raw[SENSORS].to_numpy(dtype=np.float32)
        faultlog=faults[faults.tool_id==tool].sort_values('time',kind='stable')
        fault_t=faultlog.time.to_numpy(dtype=np.int64)
        for phase in ('train','validation'):
            if phase=='train':
                prepared=pq.read_table(BASE_PREPARED/f'{tool}.parquet',
                    columns=['time','label']+model1['features'],filters=[('time','<',tend)]).to_pandas()
                prepared=prepared[(prepared.time<tend)&(prepared.label>=0)]
                phase_t=prepared.time.to_numpy(dtype=np.int64)
                phase_p=model1['model'].predict_proba(
                    prepared[model1['features']].to_numpy(dtype=np.float32))[:,1]
                phase_start=int(sensor_t[0]);phase_end=tend
            else:
                ii=np.flatnonzero(vc==code)
                phase_t=vt[ii];phase_p=vp[ii]
                phase_start=vstart;phase_end=vend
            local=np.arange(len(phase_t),dtype=np.int64)
            hits,starts,notifications=select_manager_notifications(phase_t,phase_p,local)
            result[phase]['counts'].append({'tool_id':tool,'phase':phase,
              'scored_rows':len(phase_t),'threshold_hits':len(hits),
              'raw_alarm_starts':len(starts),'manager_notifications':len(notifications)})
            phase_fault=faultlog[(faultlog.time>=phase_start)&(faultlog.time<phase_end)]
            next_t=phase_fault.time.to_numpy(dtype=np.int64)
            next_id=phase_fault.fault_event_id.to_numpy(dtype=np.int64)
            next_type=phase_fault.fault_type.to_numpy()
            for ni in notifications:
                now=int(phase_t[ni]);fj=int(np.searchsorted(next_t,now,side='right'))
                has_fault=fj<len(next_t)
                prev_idx=int(np.searchsorted(fault_t,now,side='right'))-1
                prev=int(fault_t[prev_idx]) if prev_idx>=0 else None
                last=int(np.searchsorted(sensor_t,now,side='right'))-1
                valid=False;nobs=0;span=None;last_t=None
                if last>=0:
                    last_t=int(sensor_t[last])
                    lo=max(int(np.searchsorted(sensor_t,now-3600,side='left')),
                           int(np.searchsorted(sensor_t,phase_start,side='left')),
                           0 if prev is None else int(np.searchsorted(sensor_t,prev,side='right')))
                    nobs=last-lo+1
                    if nobs>=60:
                        span=int(sensor_t[last]-sensor_t[lo])
                        valid=span>=2880
                status='A_evaluable' if has_fault and valid else (
                    'B_history_insufficient' if has_fault else 'C_no_phase_next_fault')
                row={'phase':phase,'tool_id':tool,'alert_time':now,
                  'model1_probability':float(phase_p[ni]),
                  'next_fault_event_id':int(next_id[fj]) if has_fault else None,
                  'fault_type':str(next_type[fj]) if has_fault else None,
                  'next_fault_time':int(next_t[fj]) if has_fault else None,
                  'status':status,'history_observations':max(0,nobs),
                  'history_span_seconds':span,'last_sensor_time':last_t,
                  'sensor_age_seconds':now-last_t if last_t is not None else None,
                  'sequence_index':None}
                if status=='A_evaluable':
                    idx=np.linspace(lo,last,60,dtype=np.int64)
                    assert sensor_t[idx[-1]]<=now and sensor_t[idx[0]]>=now-3600
                    assert prev is None or sensor_t[idx[0]]>prev
                    assert sensor_t[idx[0]]>=phase_start
                    row['sequence_index']=len(result[phase]['sequences'])
                    result[phase]['sequences'].append(sensor_x[idx].copy())
                result[phase]['meta'].append(row)
            print('alerts',phase,tool,len(notifications),flush=True)
    for phase in result:
        result[phase]['sequences']=np.stack(result[phase]['sequences']).astype(np.float32)
        result[phase]['meta']=pd.DataFrame(result[phase]['meta'])
    return result

def summary(x):
    return np.concatenate([x[:,-1,:],x.mean(axis=1),x.std(axis=1),
                           x.min(axis=1),x.max(axis=1),x[:,-1,:]-x[:,0,:]],axis=1)

class TypeGRU(nn.Module):
    def __init__(self):
        super().__init__();self.gru=nn.GRU(9,32,batch_first=True);self.head=nn.Linear(32,3)
    def forward(self,x):return self.head(self.gru(x)[1][-1])

def event_weight(ids):
    series=pd.Series(ids)
    count=series.value_counts()
    w=series.map(1/count).to_numpy(dtype=np.float32)
    return w*(len(w)/w.sum())

def train_gru(x,y,w,vx,scaler):
    torch.manual_seed(SEED);torch.set_num_threads(4)
    def z(arr):return np.clip(np.nan_to_num((arr-scaler['center'])/scaler['scale'],
                   nan=0,posinf=0,neginf=0),-20,20).astype(np.float32)
    ds=TensorDataset(torch.from_numpy(z(x)),torch.from_numpy(y.astype(np.int64)),
                     torch.from_numpy(w))
    dl=DataLoader(ds,batch_size=256,shuffle=True)
    model=TypeGRU();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    losses=[]
    for epoch in range(5):
        model.train();batch=[]
        for bx,by,bw in dl:
            optimizer.zero_grad();logit=model(bx)
            loss=(nn.functional.cross_entropy(logit,by,reduction='none')*bw).sum()/bw.sum()
            loss.backward();optimizer.step();batch.append(float(loss))
        losses.append(float(np.mean(batch)))
        print('type GRU epoch',epoch+1,losses[-1],flush=True)
    model.eval();out=[];val=z(vx)
    with torch.no_grad():
        for i in range(0,len(val),512):out.append(torch.softmax(model(torch.from_numpy(val[i:i+512])),dim=1).numpy())
    return model,np.concatenate(out),losses

def classification(truth,pred):
    m=precision_recall_fscore_support(truth,pred,labels=np.arange(3),zero_division=0)
    return {'rows':len(truth),'macro_f1':float(f1_score(truth,pred,labels=np.arange(3),average='macro',zero_division=0)),
      'balanced_accuracy':float(balanced_accuracy_score(truth,pred)),
      'accuracy':float(accuracy_score(truth,pred)),
      'per_class':{TYPES[i]:{'precision':float(m[0][i]),'recall':float(m[1][i]),
       'f1':float(m[2][i]),'support':int(m[3][i])} for i in range(3)},
      'confusion_matrix':confusion_matrix(truth,pred,labels=np.arange(3)).tolist()}

def evaluate_predictions(phase,probabilities):
    truth=phase.fault_type.map({x:i for i,x in enumerate(TYPES)}).to_numpy(dtype=np.int64)
    pred=np.argmax(probabilities,axis=1)
    row=classification(truth,pred)
    frame=phase[['next_fault_event_id','fault_type']].copy()
    for i,name in enumerate(TYPES):frame[name]=probabilities[:,i]
    event=frame.groupby('next_fault_event_id').agg(
       {'fault_type':'first',**{name:'mean' for name in TYPES}})
    ev_truth=event.fault_type.map({x:i for i,x in enumerate(TYPES)}).to_numpy(dtype=np.int64)
    ev_pred=np.argmax(event[TYPES].to_numpy(),axis=1)
    return row,classification(ev_truth,ev_pred)

def main():
    if OUT.exists():raise FileExistsError(f'Refusing to overwrite {OUT}')
    OUT.mkdir(parents=True)
    data=create_data()
    counts=pd.DataFrame(data['train']['counts']+data['validation']['counts'])
    counts.to_csv(OUT/'alert_counts_by_tool.csv',index=False)
    for phase in ('train','validation'):
        data[phase]['meta'].to_csv(OUT/f'{phase}_alert_population.csv.gz',index=False,compression='gzip')
    tr=data['train'];va=data['validation']
    old=pd.read_csv(RUL_ALERT)
    valid_meta=va['meta']
    assert len(valid_meta)==len(old)
    assert np.array_equal(valid_meta.alert_time,old.model1_time)
    assert np.array_equal(valid_meta.tool_id,old.tool_id)
    assert np.array_equal(valid_meta.status.to_numpy(dtype=str),
      old.status.replace({'C_no_validation_next_fault':'C_no_phase_next_fault'}).to_numpy(dtype=str))
    trmeta=tr['meta'][tr['meta'].status=='A_evaluable'].copy().reset_index(drop=True)
    vameta=va['meta'][va['meta'].status=='A_evaluable'].copy().reset_index(drop=True)
    assert len(trmeta)==len(tr['sequences']) and len(vameta)==len(va['sequences'])
    assert set(trmeta.next_fault_event_id).isdisjoint(set(vameta.next_fault_event_id))
    y=trmeta.fault_type.map({x:i for i,x in enumerate(TYPES)}).to_numpy(dtype=np.int64)
    w=event_weight(trmeta.next_fault_event_id.to_numpy(dtype=np.int64))
    event_classes=trmeta.drop_duplicates('next_fault_event_id').fault_type.value_counts()
    dummy_label=TYPES.index(event_classes.idxmax())
    dummy_p=np.zeros((len(vameta),3),dtype=np.float32);dummy_p[:,dummy_label]=1
    tree=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=31,
       learning_rate=.05,random_state=SEED,l2_regularization=1)
    tree.fit(summary(tr['sequences']),y,sample_weight=w)
    tree_p=tree.predict_proba(summary(va['sequences']))
    with open(MODELS/'simple_full_fault_rul_scaler.pkl','rb') as f:scaler=pickle.load(f)
    gru,gru_p,gru_loss=train_gru(tr['sequences'],y,w,va['sequences'],scaler)
    joblib.dump({'model':tree,'feature_set':'raw sensor 9, current/mean/std/min/max/slope over causal 1h',
                 'sensors':SENSORS,'classes':TYPES},MODELS/'model1_triggered_fault_type_tree.joblib')
    torch.save({'state_dict':gru.state_dict(),'architecture':'GRU9x32, linear3',
      'classes':TYPES,'scaler_path':str(MODELS/'simple_full_fault_rul_scaler.pkl'),
      'history_minutes':60,'sequence_points':60},MODELS/'model1_triggered_fault_type_gru.pt')
    predictions=vameta.copy()
    probs={'Dummy':dummy_p,'Tree':tree_p,'GRU':gru_p}
    for name,p in probs.items():
        predictions[name+'_class']=[TYPES[k] for k in np.argmax(p,axis=1)]
        predictions[name+'_confidence']=p.max(axis=1)
        for k,typ in enumerate(TYPES):predictions[name+'_p_'+typ]=p[:,k]
    predictions.to_csv(OUT/'validation_alert_predictions.csv.gz',index=False,compression='gzip')
    metrics=[];perclass=[];conf=[];event_predictions=[]
    for name,p in probs.items():
        row,event=evaluate_predictions(vameta,p)
        for scope,m in [('alert_row',row),('event_mean_probability',event)]:
            metrics.append({'model':name,'scope':scope,'rows_or_events':m['rows'],
              'macro_f1':m['macro_f1'],'balanced_accuracy':m['balanced_accuracy'],
              'accuracy':m['accuracy']})
            for typ,v in m['per_class'].items():
                perclass.append({'model':name,'scope':scope,'fault_type':typ,**v})
            for i,t in enumerate(TYPES):
                for j,predicted in enumerate(TYPES):
                    conf.append({'model':name,'scope':scope,'actual':t,'predicted':predicted,
                                 'count':m['confusion_matrix'][i][j]})
        temp=vameta[['next_fault_event_id','fault_type']].copy()
        for k,t in enumerate(TYPES):temp[t]=p[:,k]
        ev=temp.groupby('next_fault_event_id').agg({'fault_type':'first',**{t:'mean' for t in TYPES}})
        ev['model']=name;ev['predicted_type']=[TYPES[k] for k in np.argmax(ev[TYPES].to_numpy(),axis=1)]
        ev['confidence']=ev[TYPES].max(axis=1)
        event_predictions.append(ev.reset_index())
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(perclass).to_csv(OUT/'per_class_metrics.csv',index=False)
    pd.DataFrame(conf).to_csv(OUT/'confusion_matrix_long.csv',index=False)
    pd.concat(event_predictions).to_csv(OUT/'event_predictions.csv',index=False)
    first_actual=valid_meta[valid_meta.next_fault_event_id.notna()].sort_values('alert_time').drop_duplicates('next_fault_event_id',keep='first')
    last_actual=valid_meta[valid_meta.next_fault_event_id.notna()].sort_values('alert_time').drop_duplicates('next_fault_event_id',keep='last')
    first_rows=[]
    for scope,subset in [('true_first',first_actual),('true_last',last_actual)]:
        # Alert times may coincide across Tools; key by Tool and event ID as well.
        keys=subset[subset.status=='A_evaluable'][['tool_id','next_fault_event_id','alert_time']]
        selected=predictions.merge(keys,on=['tool_id','next_fault_event_id','alert_time'],how='inner')
        selected.to_csv(OUT/f'{scope}_evaluable.csv',index=False)
        for name,p in probs.items():
            from_file=selected[['fault_type',name+'_class']]
            if len(from_file):
                yy=from_file.fault_type.map({x:i for i,x in enumerate(TYPES)}).to_numpy()
                pp=from_file[name+'_class'].map({x:i for i,x in enumerate(TYPES)}).to_numpy()
                m=classification(yy,pp)
                first_rows.append({'scope':scope,'model':name,'all_next_fault_events_with_alert':len(subset),
                  'evaluable_events':len(selected),'macro_f1':m['macro_f1'],
                  'balanced_accuracy':m['balanced_accuracy'],
                  **{f'{t}_recall':m['per_class'][t]['recall'] for t in TYPES}})
    pd.DataFrame(first_rows).to_csv(OUT/'first_last_metrics.csv',index=False)
    stability=[]
    for name in ('Tree','GRU'):
        for event,part in predictions.sort_values('alert_time').groupby('next_fault_event_id'):
            if len(part)<2:continue
            classes=part[name+'_class'].to_numpy();same=np.mean(classes[1:]==classes[:-1])
            stability.append({'model':name,'fault_event_id':event,'alerts':len(part),
              'same_consecutive_pct':100*same,'changed_consecutive_pct':100*(1-same),
              'last_correct':bool(classes[-1]==part.fault_type.iloc[0])})
    pd.DataFrame(stability).to_csv(OUT/'prediction_stability_by_event.csv',index=False)
    confidence=[]
    for name in ('Tree','GRU'):
        for lo,hi,label in [(0,.5,'<0.50'),(.5,.7,'0.50-0.70'),(.7,.9,'0.70-0.90'),(.9,1.00001,'>=0.90')]:
            part=predictions[(predictions[name+'_confidence']>=lo)&(predictions[name+'_confidence']<hi)]
            if not len(part):continue
            yt=part.fault_type.map({x:i for i,x in enumerate(TYPES)}).to_numpy()
            yp=part[name+'_class'].map({x:i for i,x in enumerate(TYPES)}).to_numpy()
            confidence.append({'model':name,'confidence_bin':label,'alerts':len(part),
              'events':part.next_fault_event_id.nunique(),'correct_rate':accuracy_score(yt,yp),
              'macro_f1':f1_score(yt,yp,labels=np.arange(3),average='macro',zero_division=0)})
    pd.DataFrame(confidence).to_csv(OUT/'confidence_bins.csv',index=False)
    config={'model1_fitting':False,'model1_model_sha256':digest(MODEL1),
      'validation_model1_score_sha256':digest(SCORES),
      'train_model1_score_source':'frozen Model 1 applied in-sample to original Train minute rows; selection may be optimistic relative to out-of-sample Validation',
      'threshold':.34,'raw_alarm_gap_seconds':600,'manager_cooldown_seconds':1800,
      'train_alerts':len(tr['meta']),'validation_alerts':len(va['meta']),
      'train_evaluable_alerts':len(trmeta),'validation_evaluable_alerts':len(vameta),
      'train_events':trmeta.next_fault_event_id.nunique(),
      'validation_events':vameta.next_fault_event_id.nunique(),
      'train_fault_type_events':trmeta.drop_duplicates('next_fault_event_id').fault_type.value_counts().to_dict(),
      'validation_fault_type_events':vameta.drop_duplicates('next_fault_event_id').fault_type.value_counts().to_dict(),
      'history':'actual 1h raw observations before alert, >=60 observations covering >=48m; 60 evenly spaced observed indices; no fill; no fault boundary crossing',
      'features':'9 raw sensors; Tree current/mean/std/min/max/slope of same 60 points; GRU raw sequence',
      'event_weight':'each alert weight 1/alerts for its next fault event, common global scaling; no class weighting',
      'dummy':'Train unique event majority class',
      'tree':'HistGradientBoostingClassifier max_iter120 max_leaf_nodes31 lr0.05 l2=1 seed42',
      'gru':'one layer GRU32, linear3, weighted cross entropy, Adam lr0.001, 5 epochs batch256, legacy Train sensor scaler',
      'gru_loss_by_epoch':gru_loss,'test_access':'none',
      'target':'immediate next logged fault type within same phase; fault time/TTF never input'}
    (OUT/'experiment_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(json.dumps({k:config[k] for k in ('train_alerts','validation_alerts','train_evaluable_alerts','validation_evaluable_alerts','train_events','validation_events','train_fault_type_events','validation_fault_type_events')},indent=2),flush=True)
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)

if __name__=='__main__':main()
