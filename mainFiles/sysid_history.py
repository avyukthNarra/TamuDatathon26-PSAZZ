"""Train-only temporal structure search and continuous random-effects diagnostics."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','2')
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import expit, logit
from scipy.linalg import solve
from sysid_study import (ROOT,NAMES,load_machines,fit_logistic,folds)
from slot_forecasting.io import read_labels, read_targets, write_json


def predict_laplace(F,b,H):
    X=np.column_stack([np.ones(len(F)), F])
    v=np.sum(X*solve(H,X.T,assume_a='pos',check_finite=False).T,axis=1)
    return expit((X@b)/np.sqrt(1+np.pi*v/8))


def evaluate_features(train, labels, names, half_life=1e9, frac=.65, precision=.5, details=False):
    inds=[NAMES.index(n) for n in names]
    ll, n, bs, pred_records = 0.,0,[],[]
    for m in train:
        F,payout,ticks=m.rows(True,labels)
        ok=np.all(np.isfinite(F),axis=1)
        F,payout,ticks=F[ok][:,inds],payout[ok],ticks[ok]
        if len(ticks) < 20:
            continue
        stop=max(8,int(len(ticks)*frac))
        weights=2**(-(ticks[stop-1]-ticks[:stop])/half_life)
        b,H=fit_logistic(F[:stop],(payout[:stop]>0).astype(float),precision,weights)
        p=predict_laplace(F[stop:],b,H)
        y=payout[stop:]>0
        ll+=np.sum(np.log(np.where(y,p,1-p)))
        n+=len(y)
        bs.append(b)
        if details:
            pred_records.append({'query_id':m.query.query_id,'loglik':float(np.sum(np.log(np.where(y,p,1-p)))), 'n':len(y)})
    return {'names':names,'half_life':half_life,'precision':precision,'loglik':float(ll),'n':n,
            'loglik_per_tick':float(ll/n),'coefficient_quantiles':np.quantile(bs,[.05,.25,.5,.75,.95],axis=0).tolist(),
            'records':pred_records}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=ROOT/'public')
    parser.add_argument('--out',type=Path,default=ROOT/'runs/sysid')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    train,tq,_=load_machines(args.data,'train');labels=read_labels(args.data/'train/labels.csv',tq)
    top=json.loads((args.out/'screening.json').read_text())
    candidates=[r['feature'] for r in top[:12]]
    result=[]
    selected=[]
    r=evaluate_features(train,labels,selected);result.append(r);print(r,flush=True)
    previous=r['loglik_per_tick']
    for step in range(6):
        trials=[]
        for name in candidates:
            if name in selected:continue
            r=evaluate_features(train,labels,selected+[name]);result.append(r);trials.append(r)
        best=max(trials,key=lambda r:r['loglik_per_tick'])
        if best['loglik_per_tick']<=previous:break
        previous=best['loglik_per_tick']
        selected=best['names'];print('STEP',step,best,flush=True)
    for names in [list(NAMES[:8]),list(NAMES[:36]),list(NAMES),selected[:4],selected[:3],selected[:2]]:
        r=evaluate_features(train,labels,names,details=True);result.append(r);print('COMPARISON', {k:v for k,v in r.items() if k not in ('records','coefficient_quantiles')},flush=True)
    for half in [32.,64.,128.]:
        r=evaluate_features(train,labels,selected[:4],half_life=half,details=True);result.append(r);print('DECAY',{k:v for k,v in r.items() if k not in ('records','coefficient_quantiles')},flush=True)
    write_json(args.out/'history_comparisons.json',result)


if __name__=='__main__':main()
