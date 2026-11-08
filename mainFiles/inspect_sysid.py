"""Train-history likelihood alternatives and fold-A quadrature convergence."""
from collections import defaultdict
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import betaln,gammaln
from sysid_study import ROOT,load_machines,folds
from sysid_bayes import condition_map,complete_data
from sysid_dynamic import dynamic_predict,machine_grid,combine_episode
from sysid_size import size_data,size_predict
from sysid_joint import joint_predict
from sysid_pipeline import WIN_FILES,SIZE_FILES,REGIMES
from slot_forecasting.io import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'public');p.add_argument('--out',type=Path,default=ROOT/'runs/sysid-final');args=p.parse_args()
    choice=json.loads((args.out/'train_choice.json').read_text())['candidate'];wn,rn,sn=choice.split('/')
    wm=json.loads((args.out/WIN_FILES[wn]).read_text());sm=json.loads((args.out/SIZE_FILES[sn]).read_text())
    _,width,shared_width=next(x for x in REGIMES if x[0]==rn)
    train,qs,_=load_machines(args.data,'train');c=condition_map(args.data,'train');ec={m.query.episode_id:c[m.query.query_id] for m in train}
    _,episodes=dynamic_predict(wm,train,width=width,shared_width=shared_width)
    report={'model':choice,'conditioning':'Observed train history rows with all eight own lags present; training query labels excluded from this comparison.', 'by_condition':{}}
    for condition in sorted(set(c.values())):
        win_beta=0.;size_dirichlet=0.;size_rule=0.;n=0;npay=0
        for m in train:
            if c[m.query.query_id]!=condition:continue
            F,pay,ticks=complete_data(m);ones=np.sum(pay>0);zeros=len(pay)-ones;n+=len(pay)
            win_beta+=betaln(ones+.5,zeros+.5)-betaln(.5,.5)
            _,y,_=size_data(m);counts=np.bincount(y,minlength=4);npay+=len(y)
            size_dirichlet+=gammaln(2.)-gammaln(2.+len(y))+np.sum(gammaln(counts+.5)-gammaln(.5))
            size_rule+=size_predict(sm,m)[1]
        ep=[r for e,r in episodes.items() if ec[e]==condition]
        report['by_condition'][condition]={'n_binary_outcomes':n,'n_positive_outcomes':npay,
            'win_beta_bernoulli_log_marginal':float(win_beta),
            'win_static_rule_log_marginal':float(sum(r['component_log_evidences'][0] for r in ep)),
            'win_reversal_mixture_log_marginal':float(sum(r['log_evidence'] for r in ep)),
            'size_dirichlet_constant_log_marginal':float(size_dirichlet),
            'size_temporal_rule_log_marginal':float(size_rule)}
    size_change_comparison={}
    for change in [False,True]:
        _,r=joint_predict(wm,sm,train,width=width,shared_width=shared_width,size_change=change)
        size_change_comparison[str(change)]={con:float(sum(v['log_evidence'] for ep,v in r.items() if ec[ep]==con)) for con in sorted(set(c.values()))}
    report['joint_size_reversal_log_marginals']=size_change_comparison
    # Quadrature convergence uses validation inputs and condition assignments,
    # exclusively fold A; no validation targets or labels are loaded here.
    val,vq,_=load_machines(args.data,'validation');_,fold=folds(args.data,vq);fa={q.query_id:int(f) for q,f in zip(vq,fold)}
    ep=defaultdict(list)
    for m in val:
        if fa[m.query.query_id]==0:ep[m.query.episode_id].append(m)
    delta=[]
    for e,ms in ep.items():
        predictions=[]
        for nq in [15,31]:
            g=[machine_grid(wm,m,width=width,nquad=nq) for m in ms]
            sg=None if width==shared_width else [machine_grid(wm,m,width=shared_width,nquad=nq) for m in ms]
            predictions.append(combine_episode(g,shared_grids=sg)[0])
        delta.extend(abs(predictions[0]-predictions[1]))
    report['quadrature_A']={'nodes':[15,31],'n_queries':len(delta),'max_abs_probability_change':float(max(delta)),'mean_abs_probability_change':float(np.mean(delta))}
    write_json(args.out/'generative_checks.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
