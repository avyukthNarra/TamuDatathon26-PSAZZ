"""Learn the winning-size generator using categorical train histories only."""
from itertools import combinations
import argparse
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp, expit, logit
from scipy.optimize import minimize
from sysid_study import ROOT, NAMES, load_machines
from sysid_bayes import TYPES,condition_map,complete_data
from slot_forecasting.io import read_labels,write_json

SIZE_NAMES=list(NAMES[:72])+['*'.join(f'w{j}' for j in p) for p in combinations(range(1,9),3)]


def size_features(F):
    return np.column_stack([F[:,:72]]+[np.prod(F[:,np.array(p)-1],axis=1) for p in combinations(range(1,9),3)])


def size_data(m,labels=None):
    F,pay,ticks=complete_data(m,labels)
    win=pay>0
    features=size_features(F[win])
    y=np.searchsorted([1,2,5,20],pay[win])
    return features,y,ticks[win]


def size_screen(machines,labels,conditions,out):
    scores=np.zeros(len(SIZE_NAMES));global_scores=np.zeros(len(SIZE_NAMES));denoms=np.zeros(len(SIZE_NAMES))
    n_machines=0
    for m in machines:
        if conditions[m.query.query_id]!='stable':continue
        F,y,ticks=size_data(m,labels)
        y=(y==0).astype(float)
        if len(y) < 2:
            continue
        n_machines+=1
        X=F-F.mean(axis=0)[None,:]
        yy=y-y.mean()
        xy=X.T@yy
        var=(X*X).sum(axis=0)*np.var(yy)
        z2=xy**2/np.maximum(var,1e-8)
        scores+=z2-1
        global_scores+=xy
        denoms+=var
    order=np.argsort(-scores)
    result=[{'feature':SIZE_NAMES[i],'excess_z2':float(scores[i]/max(n_machines,1)),'pooled_z':float(global_scores[i]/np.sqrt(max(denoms[i],1e-8)))} for i in order]
    write_json(out/'size_screening.json',result)
    return result


def train_size(machines,labels,conditions,out,nrules=128,maxiter=200,screened=False,constrained=False):
    stable=[m for m in machines if conditions[m.query.query_id]=='stable']
    ruleids=list(range(len(SIZE_NAMES)))[:nrules]
    if screened:
        screen=size_screen(machines,labels,conditions,out)
        ruleids=[SIZE_NAMES.index(r['feature']) for r in screen if r['excess_z2']>1.]
        if not ruleids:ruleids=[SIZE_NAMES.index(screen[0]['feature'])]
    R=len(ruleids)
    # Counts are sufficient because each candidate temporal rule is binary.
    counts=[];types=[]
    for m in stable:
        F,y,ticks=size_data(m,labels)
        counts.append(np.array([[[np.sum((F[:,r]==s)&(y==k)) for k in range(4)] for s in (-1,1)] for r in ruleids]))
        types.append(TYPES.index(m.query.machine_type))
    counts=np.array(counts);types=np.array(types)
    totals=counts[:,0].sum(axis=1)
    modes=[]
    for typ in range(3):
        pooled=(totals[types==typ].sum(axis=0)+1)/(totals[types==typ].sum()+4)
        q=np.stack([pooled.copy(),pooled.copy()])
        lo=max(.02,pooled[0]-.13);hi=min(.95,pooled[0]+.13)
        for i,p in enumerate([lo,hi]):q[i]=np.r_[p,(1-p)*pooled[1:]/(1-pooled[0])]
        modes.append(q)
    modes=np.array(modes);weights=np.ones(R)/R
    ratios=np.array([(totals[types==typ,1:].sum(axis=0)+.5)/(totals[types==typ,1:].sum()+1.5) for typ in range(3)])
    history=[]
    for iteration in range(maxiter):
        # Direction + maps feature -1 to low-size1 mode and +1 to high.
        ll_plus=np.einsum('nrfk,nfk->nr',counts,np.log(modes[types]))
        ll_minus=np.einsum('nrfk,nfk->nr',counts,np.log(modes[types,::-1]))
        ll=np.stack([ll_plus,ll_minus],axis=2)+np.log(weights)[None,:,None]-np.log(2)
        norm=logsumexp(ll,axis=(1,2));post=np.exp(ll-norm[:,None,None]);history.append(float(norm.sum()))
        weights=(post.sum(axis=(0,2))+.02)/(len(stable)+R*.02)
        new=np.zeros_like(modes)+.05
        for typ in range(3):
            c=counts[types==typ];p=post[types==typ]
            new[typ]+=np.einsum('nrfk,nr->fk',c,p[:,:,0])+np.einsum('nrfk,nr->fk',c[:,:,::-1],p[:,:,1])
        modes=new/new.sum(axis=2,keepdims=True)
        if constrained:
            # The non-one payout ratios are shared by both binary modes.
            # Learn a common log-odds gap, rather than six unrelated extremes.
            success=new[:,:,0];total=new.sum(axis=2)
            signs=np.array([-1.,1.])
            initial=np.r_[np.mean(logit(modes[:,:,0]),axis=1),.65]
            def q_objective(theta):
                z=theta[:3,None]+theta[3]*signs[None,:]
                loss=np.sum(total*np.logaddexp(0,z)-success*z)
                residual=total*expit(z)-success
                gradient=np.r_[residual.sum(axis=1),np.sum(residual*signs)]
                return loss,gradient
            fit=minimize(q_objective,initial,jac=True,method='L-BFGS-B',bounds=[(-5,5)]*3+[(.01,2.)])
            p1=expit(fit.x[:3,None]+fit.x[3]*signs[None,:])
            modes[:,:,0]=p1
            modes[:,:,1:]=(1-p1[:,:,None])*ratios[:,None,:]
        if iteration>4 and abs(history[-1]-history[-2])<1e-6:break
    order=np.argsort(-weights)
    select=[int(i) for i in order if weights[i]>.02] or [int(order[0])]
    report={'ruleids':[ruleids[i] for i in select], 'ruleweights':(weights[select]/weights[select].sum()).tolist(),
            'rule_names':[SIZE_NAMES[ruleids[i]] for i in select], 'modes':modes.tolist(),
            'all_rule_weights':{SIZE_NAMES[ruleids[i]]:float(weights[i]) for i in order},
            'iterations':iteration+1,'marginal_loglik':history[-1], 'loglik_trace':history}
    filename='size_model'+('_screened' if screened else '')+('_constrained' if constrained else '')+'.json'
    write_json(out/filename,report)
    return report


def size_predict(model,m,half_life=None):
    F,y,ticks=size_data(m)
    rules=model['ruleids'];R=len(rules);mode=np.array(model['modes'])[TYPES.index(m.query.machine_type)]
    w=np.ones(len(ticks)) if half_life is None else 2**(-(m.query.cutoff-1-ticks)/half_life)
    feats=F[:,rules]>0
    ll=np.empty((R,2))
    for r in range(R):
        for s in range(2):
            ll[r,s]=np.sum(w*np.log(mode[feats[:,r].astype(int)^s,y]))
    lp=np.log(np.array(model['ruleweights'])[:,None]/2)
    post=np.exp(ll+lp-logsumexp(ll+lp))
    feature=size_features(m.features[m.query.cutoff:m.query.cutoff+1])[0,rules]
    f=feature>0
    probabilities=np.array([[mode[int(f[r])^s] if np.isfinite(feature[r]) else mode.mean(axis=0) for s in range(2)] for r in range(R)])
    return np.einsum('rs,rsk->k',post,probabilities),float(logsumexp(ll+lp))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=ROOT/'public');parser.add_argument('--out',type=Path,default=ROOT/'runs/sysid');args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    start=time.time();ms,qs,_=load_machines(args.data,'train');labels=read_labels(args.data/'train/labels.csv',qs);conditions=condition_map(args.data,'train')
    screen=size_screen(ms,labels,conditions,args.out);print('SCREEN',screen[:12],flush=True)
    for screened,constrained in [(False,False),(True,False),(True,True)]:
        model=train_size(ms,labels,conditions,args.out,screened=screened,constrained=constrained);print('MODEL',screened,constrained,model,flush=True)
    print('elapsed',time.time()-start,flush=True)


if __name__=='__main__':main()
