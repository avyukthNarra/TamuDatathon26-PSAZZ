"""Finite-rule simulator inference with train-only empirical Bayes parameters.

No validation answers are read in this module. The latent rule is learned anew
from training histories. Gaussian intercepts use deterministic quadrature;
finite rule/sign states are summed exactly conditional on that quadrature.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','2')
from itertools import product
from pathlib import Path
import argparse
import csv
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.special import roots_hermitenorm
from sysid_study import ROOT,NAMES,load_machines,fit_logistic
from slot_forecasting.io import read_labels, write_json

TYPES=('steady','balanced','jackpot')


def condition_map(public,split):
    with (Path(public)/split/'conditions.csv').open() as f:
        return {r['query_id']:r['condition'] for r in csv.DictReader(f)}


def complete_data(m, labels=None):
    F,payout,ticks=m.rows(labels is not None,labels)
    ok=np.all(np.isfinite(F[:,:36]),axis=1)
    return F[ok],payout[ok],ticks[ok]


def quadrature(n=15):
    nodes,weights=roots_hermitenorm(n)
    return nodes,weights/np.sqrt(2*np.pi)


def rule_evidence(F,pay,ruleids,mu,sigma,amp,offset=None,nquad=15):
    nodes,qw=quadrature(nquad)
    sign=np.tile(np.repeat([-1.,1.],nquad),len(ruleids))
    b=np.tile(mu+sigma*np.tile(nodes,2),len(ruleids))
    r=np.repeat(np.arange(len(ruleids)),2*nquad)
    X=F[:,np.array(ruleids)[r]]*sign[None,:]
    z=b[None,:]+amp*X
    if offset is not None:z+=offset[:,None]
    y=(pay>0).astype(float)
    ll=np.sum(y[:,None]*z-np.logaddexp(0,z),axis=0)
    logprior=np.log(np.tile(qw/2,2*len(ruleids)))
    evidence=logsumexp((ll+logprior).reshape(len(ruleids),2*nquad),axis=1)
    return evidence,(ll,logprior,X,b,sign,r,nodes,qw)


def infer_rules(machines,labels,conditions,out):
    stable=[m for m in machines if conditions[m.query.query_id]=='stable']
    ruleids=list(range(36))
    # Rough local logistic fits initialize intercepts from histories alone.
    intercepts={t:[] for t in TYPES}
    amps=[]
    ev=[]
    for m in stable:
        F,pay,ticks=complete_data(m,labels)
        fits=[]
        for j in range(36):
            b,H=fit_logistic(F[:,j:j+1],(pay>0).astype(float),.25)
            z=b[0]+F[:,j]*b[1]
            ll=np.sum((pay>0)*z-np.logaddexp(0,z))
            fits.append((ll,b))
        best=max(fits,key=lambda v:v[0])[1]
        intercepts[m.query.machine_type].append(best[0])
        amps.append(abs(best[1]))
    pooled_intercepts=[b for values in intercepts.values() for b in values]
    if not pooled_intercepts:
        raise ValueError('At least one stable training machine with observed history is required.')
    fallback=float(np.median(pooled_intercepts))
    mu=np.array([np.median(intercepts[t]) if intercepts[t] else fallback for t in TYPES])
    amp=np.median(amps)
    for m in stable:
        F,pay,ticks=complete_data(m,labels)
        evidence,_=rule_evidence(F,pay,ruleids,mu[TYPES.index(m.query.machine_type)],.6,amp)
        ev.append(evidence)
    ev=np.array(ev)
    weights=np.ones(36)/36
    for _ in range(100):
        post=np.exp(ev+np.log(weights)-logsumexp(ev+np.log(weights),axis=1)[:,None])
        weights=(post.sum(axis=0)+.05)/(len(stable)+36*.05)
    order=np.argsort(-weights)
    # Keep all rules with at least 2% posterior mass, at least the best rule.
    selected=[int(i) for i in order if weights[i]>.02] or [int(order[0])]
    report={'initial_mu':mu.tolist(),'initial_amplitude':float(amp),
            'rule_weights':{NAMES[i]:float(weights[i]) for i in order},
            'selected_rules':[NAMES[i] for i in selected]}
    write_json(out/'rule_identification.json',report)
    return selected,weights[selected]/weights[selected].sum(),mu,amp


def train_win(machines,labels,conditions,out,shared_features=True):
    ruleids,ruleweights,mu,amp=infer_rules(machines,labels,conditions,out)
    # Fit physical coefficients on stable TRAIN machines. Dynamic histories are
    # reserved for choosing the filtering model rather than diluting amplitudes.
    stable=[m for m in machines if conditions[m.query.query_id]=='stable']
    if shared_features=='strongest_peer':
        import json
        peers=json.loads((out/'win_model_peers_only.json').read_text())
        commonids=[peers['commonids'][int(np.argmax(np.abs(peers['common_coefficients'])))]]
    elif shared_features=='all_peers':commonids=list(range(8))+list(range(72,80))
    elif shared_features=='peers_only':commonids=list(range(72,80))
    else:commonids=list(range(8))+[72] if shared_features else []
    prepared=[]
    nquad=15
    nodes,qw=quadrature(nquad)
    rs=np.repeat(np.arange(len(ruleids)),2*nquad)
    signs=np.tile(np.repeat([-1.,1.],nquad),len(ruleids))
    ns=np.tile(nodes,2*len(ruleids))
    lp=np.log(np.repeat(ruleweights,2*nquad)*np.tile(qw/2,2*len(ruleids)))
    for m in stable:
        F,pay,ticks=complete_data(m,labels)
        prepared.append((F[:,np.array(ruleids)[rs]]*signs[None,:],
             np.nan_to_num(F[:,commonids]),(pay>0).astype(float),TYPES.index(m.query.machine_type),m.query.signal))
    # Parameters: three intercept means, signal, log sd, amplitude, common terms.
    initial=np.r_[mu,0.,np.log(.5),amp,np.zeros(len(commonids))]
    bounds=[(-4,4)]*3+[(-1,1),(-3,1),(1,5)]+[(-1,1)]*len(commonids)
    def objective(theta):
        mu=theta[:3];sig=theta[3];sd=np.exp(theta[4]);amp=theta[5];gamma=theta[6:]
        loss=0.;g=np.zeros(len(theta))
        for X,C,y,typ,signal in prepared:
            z=mu[typ]+sig*signal+sd*ns[None,:]+amp*X+(C@gamma)[:,None]
            ll=np.sum(y[:,None]*z-np.logaddexp(0,z),axis=0)+lp
            norm=logsumexp(ll);post=np.exp(ll-norm)
            loss-=norm
            residual=(expit(z)-y[:,None])*post[None,:]
            sums=residual.sum(axis=0)
            g[typ]+=sums.sum();g[3]+=sums.sum()*signal
            g[4]+=sums@(sd*ns);g[5]+=np.sum(residual*X)
            g[6:]+=C.T@residual.sum(axis=1)
        # Weak shrinkage of common effects and signal, avoiding nuisance overfit.
        loss+=.5*4*np.sum(theta[6:]**2)+.5*theta[3]**2
        g[6:]+=4*theta[6:];g[3]+=theta[3]
        return loss,g
    fit=minimize(objective,initial,jac=True,bounds=bounds,method='L-BFGS-B',options={'maxiter':120,'ftol':1e-10})
    theta=fit.x
    report={'ruleids':ruleids,'ruleweights':ruleweights.tolist(),'commonids':commonids,
            'mu':theta[:3].tolist(),'signal':float(theta[3]),'sigma':float(np.exp(theta[4])),
            'amplitude':float(theta[5]),'common_coefficients':theta[6:].tolist(),
            'optimizer_success':bool(fit.success),'optimizer_message':str(fit.message),
            'stable_train_log_marginal':float(-fit.fun),'nquad_fit':nquad}
    write_json(out/('win_model_strongest_peer.json' if shared_features=='strongest_peer' else 'win_model_all_peers.json' if shared_features=='all_peers' else 'win_model_peers_only.json' if shared_features=='peers_only' else 'win_model.json' if shared_features else 'win_model_no_common.json'),report)
    return report


def win_states(model,m,nquad=21):
    nodes,qw=quadrature(nquad)
    R=len(model['ruleids'])
    rid=np.repeat(np.arange(R),2*nquad)
    signs=np.tile(np.repeat([-1.,1.],nquad),R)
    intercept=np.tile(model['mu'][TYPES.index(m.query.machine_type)]+model['signal']*m.query.signal+model['sigma']*np.tile(nodes,2),R)
    prior=np.repeat(model['ruleweights'],2*nquad)*np.tile(qw/2,2*R)
    return rid,signs,intercept,prior


def static_predict(model,m,half_life=None,labels=None,nquad=21):
    F,pay,ticks=complete_data(m,labels)
    rid,signs,intercept,prior=win_states(model,m,nquad)
    X=F[:,np.array(model['ruleids'])[rid]]*signs[None,:]
    C=np.nan_to_num(F[:,model['commonids']])
    offset=C@model['common_coefficients']
    z=intercept[None,:]+model['amplitude']*X+offset[:,None]
    w=np.ones(len(ticks)) if half_life is None else 2**(-(m.query.cutoff-1-ticks)/half_life)
    ll=np.sum(w[:,None]*((pay>0)[:,None]*z-np.logaddexp(0,z)),axis=0)
    post=np.exp(ll+np.log(prior)-logsumexp(ll+np.log(prior)))
    f=m.features[m.query.cutoff]
    off=np.nan_to_num(f[model['commonids']])@model['common_coefficients']
    feature=f[np.array(model['ruleids'])[rid]]
    nextz=intercept+model['amplitude']*signs*np.nan_to_num(feature)+off
    probabilities=expit(nextz)
    unknown=~np.isfinite(feature)
    probabilities[unknown]=.5*(expit(intercept[unknown]+off+model['amplitude'])+expit(intercept[unknown]+off-model['amplitude']))
    return float(post@probabilities),float(logsumexp(ll+np.log(prior)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=ROOT/'public');parser.add_argument('--out',type=Path,default=ROOT/'runs/sysid');args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    start=time.time();machines,queries,_=load_machines(args.data,'train');labels=read_labels(args.data/'train/labels.csv',queries);conditions=condition_map(args.data,'train')
    model=train_win(machines,labels,conditions,args.out)
    print(model,flush=True);print('elapsed',time.time()-start,flush=True)


if __name__=='__main__':main()
