"""Episode-stratified cross-fitting, official scoring and bootstrap for sysid.

Predictions are supplied without answers. This module alone reads targets for
selection/scoring; target probabilities never enter history model features.
"""
from collections import defaultdict
import argparse
import json
from pathlib import Path

import numpy as np
from sysid_study import ROOT,folds,load_machines
from sysid_bayes import static_predict
from sysid_dynamic import dynamic_predict
from sysid_size import size_predict
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import read_labels,read_targets,read_submission,write_json,write_submission
from slot_forecasting.contest import contest_score


def probability_distribution(win,size):
    return np.r_[1-win,win*np.asarray(size)]


def official_score(pred,queries,targets,groups,reference,keep=None):
    ids={q.query_id for q in queries} if keep is None else set(keep)
    return contest_score({k:tuple(v) for k,v in pred.items() if k in ids},[t for t in targets if t.query_id in ids],
        {k:v for k,v in groups.items() if k in ids},{k:v for k,v in reference.items() if k in ids},max_points=60)


def bootstrap(pred,queries,targets,groups,reference,repeats=4000,seed=20261108):
    truth={t.query_id:np.array(t.probabilities) for t in targets}
    ep=defaultdict(list)
    for q in queries:ep[q.episode_id].append(q.query_id)
    bycon=defaultdict(list)
    for e,keys in sorted(ep.items()):
        err=np.mean([np.sum((np.asarray(pred[k])-truth[k])**2) for k in keys])
        ref=np.mean([np.sum((np.asarray(reference[k])-truth[k])**2) for k in keys])
        bycon[groups[keys[0]]].append((err,ref))
    rng=np.random.default_rng(seed);ratios=[]
    for con in sorted(bycon):
        a=np.array(bycon[con]);ix=rng.integers(0,len(a),size=(repeats,len(a)))
        avg=a[ix].mean(axis=1);ratios.append(avg[:,0]/np.maximum(avg[:,1],1e-9))
    points=60/(1+np.mean(ratios,axis=0))
    return np.quantile(points,[.025,.975]).tolist()


def select_cross_fitted(candidates,queries,fold,scores):
    selected=[max(scores,key=lambda n:scores[n][i]) for i in range(2)]
    pred={q.query_id:candidates[selected[1-int(f)]][q.query_id] for q,f in zip(queries,fold)}
    return pred,{'tune_A_predict_B':selected[0],'tune_B_predict_A':selected[1],'fold_scores':scores}


def crossfit(candidates,queries,targets,groups,reference,data=ROOT/'public'):
    _,fold=folds(data,queries)
    ids=[{q.query_id for q,f in zip(queries,fold) if f==i} for i in range(2)]
    scores={name:[official_score(p,queries,targets,groups,reference,s)['technical_points'] for s in ids] for name,p in candidates.items()}
    return select_cross_fitted(candidates,queries,fold,scores)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=ROOT/'public');parser.add_argument('--out',type=Path,default=ROOT/'runs/sysid');parser.add_argument('--A-only',action='store_true');args=parser.parse_args()
    out=args.out
    train,tq,th=load_machines(args.data,'train');val,vq,vh=load_machines(args.data,'validation')
    labels=read_labels(args.data/'train/labels.csv',tq);base=FrequencyForecaster('pooled').fit(tq,th,labels);pooled=base.predict(vq,vh)
    model=json.loads((out/'win_model.json').read_text());size_model=json.loads((out/'size_model.json').read_text())
    targets=read_targets(args.data/'validation/targets.csv');groups,fold=folds(args.data,vq);reference=read_submission(args.data/'validation/reference.csv')
    sizes={m.query.query_id:size_predict(size_model,m)[0] for m in val}
    candidates={};wins={}
    for half in [None,8,16,24,32,48,64]:
        name='static' if half is None else f'decay{half}'
        wins[name]={m.query.query_id:static_predict(model,m,half)[0] for m in val}
    win=json.loads((out/'dynamic_validation_0.json').read_text())['prediction'];wins['dynamic0']=win
    for name,w in wins.items():candidates[name]={k:probability_distribution(p,sizes[k]) for k,p in w.items()}
    ids={q.query_id for q,f in zip(vq,fold) if f==0} if args.A_only else None
    report={}
    for name,p in {'pooled':pooled,**candidates}.items():
        s=official_score(p,vq,targets,groups,reference,ids)
        report[name]={'points':s['technical_points'],'ratios':{k:v['loss_ratio'] for k,v in s['by_condition'].items()}}
    write_json(out/('scores_A.json' if args.A_only else 'scores_initial.json'),report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
