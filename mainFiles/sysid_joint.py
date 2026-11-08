"""Joint win/size inference conditional on shared or individual change time."""
from collections import defaultdict
import numpy as np
from scipy.special import logsumexp
from sysid_bayes import TYPES
from sysid_size import size_data,size_features
from sysid_dynamic import machine_grid,change_grid


def size_grid(model,m,change=True,width=0.,fractions=None,labels=None,end_tick=None):
    F,y,ticks=size_data(m,labels)
    cutoff=m.query.cutoff if end_tick is None else end_tick
    ok=ticks<cutoff;F,y,ticks=F[ok],y[ok],ticks[ok]
    rules=model['ruleids'];R=len(rules);mode=np.array(model['modes'])[TYPES.index(m.query.machine_type)]
    cp,span=change_grid(cutoff,width,fractions)
    gs=np.ones((len(cp)+1,len(ticks)));gn=np.ones(len(cp)+1)
    if change:
        if span==0:
            gs[1:]=np.where(ticks[None,:]<cp[:,None],1.,-1.)
            gn[1:]=np.where(cutoff<cp,1.,-1.)
        else:
            gs[1:]=np.clip((cp[:,None]-ticks[None,:])/(span/2),-1,1)
            gn[1:]=np.clip((cp-cutoff)/(span/2),-1,1)
    feat=np.repeat(F[:,rules],2,axis=1)*np.tile([1,-1],R)[None,:]
    high=(1+gs[:,:,None]*feat[None,:,:])/2
    p=mode[0,y][None,:,None]*(1-high)+mode[1,y][None,:,None]*high
    lp=np.log(np.repeat(np.array(model['ruleweights'])/2,2))
    ll=np.sum(np.log(p),axis=1)+lp[None,:];evidence=logsumexp(ll,axis=1)
    f=np.repeat(size_features(m.features[cutoff:cutoff+1])[0,rules],2)*np.tile([1,-1],R)
    nhigh=(1+gn[:,None]*np.nan_to_num(f)[None,:])/2
    nprob=mode[0][None,None,:]*(1-nhigh[:,:,None])+mode[1][None,None,:]*nhigh[:,:,None]
    conditional=np.sum(np.exp(ll-evidence[:,None])[:,:,None]*nprob,axis=1)
    return evidence,conditional


def combine_full(grids,prior=(.5,.25,.25),shared_grids=None):
    evidence=np.array([g[0] for g in grids]);conditional=np.array([g[1] for g in grids]);n,K=evidence.shape
    lp=-np.log(K-1)
    static_e=evidence[:,0].sum();static_p=conditional[:,0]
    independent_machine_e=logsumexp(evidence[:,1:]+lp,axis=1)
    independent_p=np.sum(np.exp(evidence[:,1:]+lp-independent_machine_e[:,None])[:,:,None]*conditional[:,1:],axis=1)
    independent_e=independent_machine_e.sum()
    if shared_grids is None:each=evidence[:,1:];c=conditional[:,1:]
    else:each=np.array([g[0][1:] for g in shared_grids]);c=np.array([g[1][1:] for g in shared_grids])
    total=each.sum(axis=0)+lp;shared_e=logsumexp(total)
    shared_p=np.sum(np.exp(total-shared_e)[None,:,None]*c,axis=1)
    lml=np.array([static_e,independent_e,shared_e])+np.log(np.maximum(prior,1e-300))
    posterior=np.exp(lml-logsumexp(lml));prediction=np.einsum('c,cnd->nd',posterior,np.array([static_p,independent_p,shared_p]))
    return prediction,{'model_posterior':posterior.tolist(),'log_evidence':float(logsumexp(lml)),
        'component_log_evidences':[float(static_e),float(independent_e),float(shared_e)],
        'shared_cp_posterior':np.exp(total-shared_e).tolist()}


def joint_machine_grid(win_model,size_model,m,width,change,labels=None,end_tick=None,fractions=None):
    we,wp=machine_grid(win_model,m,width=width,labels=labels,end_tick=end_tick,fractions=fractions)
    se,sp=size_grid(size_model,m,change=change,width=0,labels=labels,end_tick=end_tick,fractions=fractions)
    conditional=np.concatenate([(1-wp)[:,None],wp[:,None]*sp],axis=1)
    return we+se,conditional


def joint_predict(win_model,size_model,machines,width=.1,shared_width=0.,size_change=True,prior=(.5,.25,.25),labels=None,end_offset=None,fractions=None):
    by_episode=defaultdict(list)
    for m in machines:by_episode[m.query.episode_id].append(m)
    predictions={};reports={}
    for ep,ms in sorted(by_episode.items()):
        grids=[joint_machine_grid(win_model,size_model,m,width,size_change,labels,None if end_offset is None else m.query.cutoff-end_offset,fractions) for m in ms]
        shared_grids=None if shared_width==width else [joint_machine_grid(win_model,size_model,m,shared_width,size_change,labels,None if end_offset is None else m.query.cutoff-end_offset,fractions) for m in ms]
        p,r=combine_full(grids,prior,shared_grids)
        predictions.update({m.query.query_id:v.tolist() for m,v in zip(ms,p)})
        reports[ep]=r
    return predictions,reports
