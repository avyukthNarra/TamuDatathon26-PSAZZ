"""Exact grid inference for individual and episode-shared sign reversals."""
from collections import defaultdict
from pathlib import Path
import argparse
import json
import time

import numpy as np
from scipy.special import expit,logsumexp
from sysid_bayes import ROOT,complete_data,win_states,load_machines,condition_map,physical_values
from slot_forecasting.io import write_json,read_labels


def change_grid(cutoff,width=0.,fractions=None):
    fractions=np.arange(.25,.951,.05) if fractions is None else np.asarray(fractions)
    cp=fractions*cutoff
    return cp,cutoff*width


def machine_grid(model,m,width=0.,fractions=None,nquad=15,labels=None,end_tick=None):
    F,pay,ticks=complete_data(m,labels)
    if end_tick is not None:
        ok=ticks<end_tick;F,pay,ticks=F[ok],pay[ok],ticks[ok]
    cutoff=m.query.cutoff if end_tick is None else end_tick
    rid,signs,intercept,prior=win_states(model,m,nquad)
    amplitude=physical_values(model,m)[3]
    X=F[:,np.array(model['ruleids'])[rid]]*signs[None,:]
    offset=np.nan_to_num(F[:,model['commonids']])@model['common_coefficients']
    cp,span=change_grid(cutoff,width,fractions)
    if span==0:
        gs=np.where(ticks[None,:]<cp[:,None],1.,-1.)
        gn=np.where(cutoff<cp,1.,-1.)
    else:
        gs=np.clip((cp[:,None]-ticks[None,:])/(span/2),-1,1)
        gn=np.clip((cp-cutoff)/(span/2),-1,1)
    # First branch is the unchanged rule. Remaining branches share a cp grid.
    gs=np.vstack([np.ones((1,len(ticks))),gs]);gn=np.r_[1.,gn]
    z=intercept[None,None,:]+amplitude*X[None,:,:]*gs[:,:,None]+offset[None,:,None]
    ll=np.sum((pay>0)[None,:,None]*z-np.logaddexp(0,z),axis=1)+np.log(prior)[None,:]
    evidence=logsumexp(ll,axis=1)
    f=m.features[cutoff]
    off=np.nan_to_num(f[model['commonids']])@model['common_coefficients']
    feature=f[np.array(model['ruleids'])[rid]]
    nz=intercept[None,:]+amplitude*signs[None,:]*np.nan_to_num(feature)[None,:]*gn[:,None]+off
    probabilities=expit(nz)
    unknown=~np.isfinite(feature)
    probabilities[:,unknown]=.5*(expit(intercept[None,unknown]+off+amplitude*gn[:,None])+expit(intercept[None,unknown]+off-amplitude*gn[:,None]))
    conditional=np.sum(np.exp(ll-evidence[:,None])*probabilities,axis=1)
    return evidence,conditional


def combine_episode(grids,prior=(.5,.25,.25),unchanged_in_shared=0.,shared_grids=None):
    """Three models: static, independent cp, shared episode cp.

    Individual intercepts, signs and rule IDs are independent given cp. This
    allows the exact joint shared-cp posterior via products of evidences.
    """
    evidence=np.array([g[0] for g in grids]);conditional=np.array([g[1] for g in grids]);n,K=evidence.shape
    lp=-np.log(K-1)
    static_e=evidence[:,0].sum();static_p=conditional[:,0]
    independent_machine_e=logsumexp(evidence[:,1:]+lp,axis=1)
    independent_p=np.sum(np.exp(evidence[:,1:]+lp-independent_machine_e[:,None])*conditional[:,1:],axis=1)
    independent_e=independent_machine_e.sum()
    if shared_grids is not None:
        shared_evidence=np.array([g[0] for g in shared_grids]);shared_conditional=np.array([g[1] for g in shared_grids])
    else:
        shared_evidence=evidence;shared_conditional=conditional
    if unchanged_in_shared:
        a=np.log(unchanged_in_shared)+evidence[:,0,None]
        b=np.log1p(-unchanged_in_shared)+shared_evidence[:,1:]
        each=logsumexp(np.stack([np.broadcast_to(a,b.shape),b]),axis=0)
        c=np.exp(a-each)*conditional[:,0,None]+np.exp(b-each)*shared_conditional[:,1:]
    else:each=shared_evidence[:,1:];c=shared_conditional[:,1:]
    total=each.sum(axis=0)+lp;shared_e=logsumexp(total)
    shared_p=np.sum(np.exp(total-shared_e)[None,:]*c,axis=1)
    lml=np.array([static_e,independent_e,shared_e])+np.log(np.maximum(prior,1e-300))
    posterior=np.exp(lml-logsumexp(lml))
    prediction=posterior@np.array([static_p,independent_p,shared_p])
    return prediction,{'model_posterior':posterior.tolist(),'log_evidence':float(logsumexp(lml)),
        'component_log_evidences':[float(static_e),float(independent_e),float(shared_e)],
        'static_predictions':static_p.tolist(),'independent_predictions':independent_p.tolist(),'shared_predictions':shared_p.tolist(),
        'shared_cp_posterior':np.exp(total-shared_e).tolist()}


def dynamic_predict(model,machines,width=0.,prior=(.5,.25,.25),unchanged_in_shared=0.,labels=None,end_offset=None,shared_width=None):
    by_episode=defaultdict(list)
    for m in machines:by_episode[m.query.episode_id].append(m)
    prediction={};reports={}
    for ep,ms in sorted(by_episode.items()):
        grids=[machine_grid(model,m,width=width,labels=labels,end_tick=None if end_offset is None else m.query.cutoff-end_offset) for m in ms]
        shared_grids=None if shared_width is None or shared_width==width else [machine_grid(model,m,width=shared_width,labels=labels,end_tick=None if end_offset is None else m.query.cutoff-end_offset) for m in ms]
        p,r=combine_episode(grids,prior,unchanged_in_shared,shared_grids)
        prediction.update({m.query.query_id:float(v) for m,v in zip(ms,p)})
        reports[ep]=r
    return prediction,reports


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=ROOT/'public');parser.add_argument('--out',type=Path,default=ROOT/'runs/sysid');parser.add_argument('--split',default='validation');parser.add_argument('--width',type=float,default=0.);args=parser.parse_args()
    model=json.loads((args.out/'win_model.json').read_text());ms,qs,_=load_machines(args.data,args.split)
    start=time.time();p,r=dynamic_predict(model,ms,width=args.width)
    write_json(args.out/f'dynamic_{args.split}_{args.width:g}.json',{'prediction':p,'episodes':r,'runtime':time.time()-start})
    print('runtime',time.time()-start,flush=True)


if __name__=='__main__':main()
