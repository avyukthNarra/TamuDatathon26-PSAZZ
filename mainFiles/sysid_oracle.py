"""Fold-A-only functional-form diagnostics; oracle modes never predict anything."""
import argparse
from pathlib import Path
import numpy as np
from scipy.special import logit,expit
from sysid_study import ROOT,NAMES,load_machines,folds
from slot_forecasting.io import read_targets,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'public');p.add_argument('--out',type=Path,default=ROOT/'runs/sysid');args=p.parse_args()
    machines,queries,_=load_machines(args.data,'validation');conditions,f=folds(args.data,queries);fold={q.query_id:i for q,i in zip(queries,f)}
    machines=[m for m in machines if fold[m.query.query_id]==0];q=[m.query for m in machines]
    targets={t.query_id:t.probabilities for t in read_targets(args.data/'validation/targets.csv')}
    win=np.array([1-targets[v.query_id][0] for v in q]);z=logit(win)
    F=np.array([m.features[m.query.cutoff] for m in machines]);mode=np.zeros(len(q));clusters={}
    for typ in sorted({v.machine_type for v in q}):
        mask=np.array([v.machine_type==typ for v in q]);zz=z[mask];centers=np.quantile(zz,[.25,.75])
        for _ in range(50):
            k=np.argmin(abs(zz[:,None]-centers[None,:]),axis=1)
            new=np.array([zz[k==j].mean() for j in range(2)])
            if np.max(abs(new-centers))<1e-12:break
            centers=new
        mode[mask]=2*k-1;clusters[typ]={'centers':centers.tolist(),'within_logit_rmse':float(np.sqrt(np.mean((zz-centers[k])**2)))}
    types=np.column_stack([[v.machine_type==t for v in q] for t in ('steady','balanced','jackpot')]).astype(float)
    signal=np.array([v.signal for v in q])[:,None]
    base=np.column_stack([types,signal]);bn=['steady','balanced','jackpot','signal']
    modes=types*mode[:,None]
    eps=sorted({v.episode_id for v in q});episode=np.column_stack([[v.episode_id==ep for v in q] for ep in eps[1:]])
    tests=[('type_signal',base,bn),
           ('oracle_mode_shared_amplitude',np.column_stack([base,mode]),bn+['oracle_mode']),
           ('oracle_mode_type_amplitude',np.column_stack([base,modes]),bn+['mode_steady','mode_balanced','mode_jackpot']),
           ('oracle_mode_plus_own_lags',np.column_stack([base,modes,F[:,:8]]),bn+['mode_steady','mode_balanced','mode_jackpot']+NAMES[:8]),
           ('oracle_mode_plus_peers',np.column_stack([base,modes,F[:,72:80]]),bn+['mode_steady','mode_balanced','mode_jackpot']+NAMES[72:80]),
           ('oracle_mode_plus_episode',np.column_stack([base,modes,episode]),bn+['mode_steady','mode_balanced','mode_jackpot']+eps[1:]),
           ('oracle_mode_all',np.column_stack([base,modes,F[:,:8],F[:,72:80],episode]),bn+['mode_steady','mode_balanced','mode_jackpot']+NAMES[:8]+NAMES[72:80]+eps[1:])]
    results={}
    for name,X,names in tests:
        b=np.linalg.lstsq(X,z,rcond=None)[0];r=z-X@b
        results[name]={'n':len(z),'n_parameters':len(b),'logit_rmse':float(np.sqrt(np.mean(r*r))),
            'win_probability_rmse':float(np.sqrt(np.mean((win-expit(X@b))**2))),
            'max_logit_error':float(max(abs(r))),'coefficients':dict(zip(names,b.tolist()))}
    sizes=np.array([targets[v.query_id][1:] for v in q])/win[:,None]
    size_odds={}
    for typ in ('steady','balanced','jackpot'):
        s=sizes[[v.machine_type==typ for v in q]]
        u=np.unique(np.round(s,9),axis=0);size_odds[typ]={'log_odds_gap_p1':float(np.diff(logit(u[:,0]))[0]),'non_one_ratios':(u[:,1:]/u[:,1,None]).tolist()}
    report={'fold':'A only','oracle_warning':'Mode labels are derived from targets for diagnostics only and never used in predictions.',
            'logit_clusters':clusters,'functional_form_fits':results,'size_odds':size_odds}
    write_json(args.out/'oracle_structure_A.json',report)
    print({k:{kk:vv for kk,vv in v.items() if kk!='coefficients'} for k,v in results.items()},flush=True)
    print('SHARED FIT',results['oracle_mode_shared_amplitude']['coefficients'],flush=True)
    print('TYPE FIT',results['oracle_mode_type_amplitude']['coefficients'],flush=True)


if __name__=='__main__':main()
