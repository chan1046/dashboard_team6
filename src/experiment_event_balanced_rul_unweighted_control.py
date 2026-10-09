"""Unweighted cap64 control to isolate the weight effect from the legacy hybrid A."""
from __future__ import annotations
import json,pickle
import numpy as np
import pandas as pd
import torch
from experiment_event_balanced_rul import (ROOT,BASE,OUT,MODELS,build,train_tree,
                                          train_gru,metrics,BINS,NAMES)

def main():
    if (OUT/'unweighted_control_metrics.csv').exists():
        raise FileExistsError(OUT/'unweighted_control_metrics.csv')
    d=build();tr=d['train'];va=d['validation'];ix=tr['a']
    old=pd.read_csv(BASE/'validation_predictions.csv.gz')
    assert np.array_equal(old.anchor_time,va['meta'].anchor_time)
    assert np.array_equal(old.fault_event_id,va['meta'].fault_event_id)
    with open(MODELS/'simple_full_fault_rul_scaler.pkl','rb') as f:scaler=pickle.load(f)
    w=np.ones(len(ix),dtype=np.float32)
    tree_model,tree=train_tree(tr['x'][ix],tr['y'][ix],w,va['x'])
    with open(MODELS/'event_balanced_rul_tree_A_cap64_unweighted.pkl','wb') as f:
        pickle.dump(tree_model,f)
    gru_model,gru=train_gru(tr['x'][ix],tr['y'][ix],w,va['x'],scaler)
    torch.save({'state_dict':gru_model.state_dict(),'variant':'A_cap64_unweighted',
      'legacy_scaler_path':str(MODELS/'simple_full_fault_rul_scaler.pkl')},
      MODELS/'event_balanced_rul_gru_A_cap64_unweighted.pt')
    p=va['meta'].copy();p['Tree']=tree;p['GRU']=gru
    p.to_csv(OUT/'unweighted_control_predictions.csv.gz',index=False,compression='gzip')
    rows=[];binrows=[]
    y=va['y'];ids=va['meta'].fault_event_id.to_numpy()
    for name,pred in [('Tree',tree),('GRU',gru)]:
        for subset,mask in [('all',np.ones(len(y),dtype=bool))]+[
            (f'within_{h}h',y<=h) for h in (24,12,6,3)]:
            rows.append({'variant':'A_cap64_unweighted','model':name,'subset':subset,
                         **metrics(y[mask],pred[mask],ids[mask])})
        for lo,hi,label in zip(BINS[:-1],BINS[1:],NAMES):
            mask=(y>=lo)&(y<hi);actual=y[mask];forecast=pred[mask]
            binrows.append({'variant':'A_cap64_unweighted','model':name,'ttf_bin':label,
               'rows':len(actual),'events':len(np.unique(ids[mask])),
               'actual_median_h':float(np.median(actual)),
               'prediction_median_h':float(np.median(forecast)),
               'mae_h':float(np.mean(np.abs(forecast-actual))),
               'mean_bias_h':float(np.mean(forecast-actual))})
    pd.DataFrame(rows).to_csv(OUT/'unweighted_control_metrics.csv',index=False)
    pd.DataFrame(binrows).to_csv(OUT/'unweighted_control_bias.csv',index=False)
    (OUT/'unweighted_control_config.json').write_text(json.dumps({
      'purpose':'isolate sample weighting while holding cap64 samples, validation, scaler, architecture and hyperparameters fixed',
      'train_samples':len(ix),'train_events':tr['meta'].iloc[ix].fault_event_id.nunique(),
      'weights':'all 1.0','test_access':'none'},indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).query('subset=="all"').to_string(index=False),flush=True)

if __name__=='__main__':main()
