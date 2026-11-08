"""T2/T3 additions to the earlier ResearchPairMixerV4 batching approach.

Win/size heads below reproduce pinned v4 exactly at baseline. New population
statistics and detector thresholds are estimated on TRAIN histories only.
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.special import gammaln,logsumexp,expit

ROOT=Path(__file__).resolve().parents[4]
HERE=ROOT/'docs/sai';OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from harness import load_split

PAIRS=((1,3),(4,7),(2,5));TYPES=('steady','balanced','jackpot')
METADATA=json.loads((HERE/'combos/TM_MIXER4.json').read_text())
POP=np.asarray(METADATA['win_prior']);DEFAULT=POP[:,0]


def win_walk(s,component_weights=None,trace_heads=False):
    w=s.win;n=s.n;tmax=int(s.cutoff.max())
    decay=2.**(-1/np.array([12.,40.]))
    wins=np.zeros((n,3,2,2));counts=np.zeros_like(wins)
    rate=np.zeros((n,2));logw=np.zeros((n,3,2))
    a=(POP[:,1]*POP[:,2])[None,None,None,:]
    b=((1-POP[:,1])*POP[:,2])[None,None,None,:]
    prior=np.broadcast_to(DEFAULT,(n,3)) if component_weights is None else component_weights
    lprior=np.log(prior)[:,None,None,:]
    trace=np.zeros((n,tmax+1));heads=np.zeros((n,tmax+1,2));final_p=np.zeros((n,3,2));final_lw=np.zeros_like(final_p)
    ii,jj,kk=np.indices((n,3,2))
    for t in range(tmax+1):
        base=(rate[:,0]+1)/(rate[:,1]+2)
        valid=np.zeros((n,3),bool);agree=np.zeros((n,3),int)
        for j,(x,y) in enumerate(PAIRS):
            if t>=max(x,y):
                valid[:,j]=np.isfinite(w[:,t-x])&np.isfinite(w[:,t-y])
                agree[:,j]=w[:,t-x]==w[:,t-y]
        cell=np.broadcast_to(agree[:,:,None],(n,3,2))
        cw=wins[ii,jj,kk,cell][...,None];cn=counts[ii,jj,kk,cell][...,None]
        logs=lprior+gammaln(a+cw)+gammaln(b+cn-cw)-gammaln(a+b+cn)-gammaln(a)-gammaln(b)+gammaln(a+b)
        post=np.exp(logs-logsumexp(logs,axis=-1,keepdims=True))
        pred=(post*(a+cw)/(a+b+cn)).sum(-1)
        pred=np.where(valid[...,None],pred,base[:,None,None])
        weights=np.exp(logw-logw.max((1,2),keepdims=True))
        trace[:,t]=(weights*pred).sum((1,2))/weights.sum((1,2))
        heads[:,t]=(weights*pred).sum(1)/weights.sum(1)
        emit=s.cutoff==t;final_p[emit]=pred[emit];final_lw[emit]=logw[emit]
        if t==w.shape[1]:break
        obs=np.isfinite(w[:,t]);value=np.nan_to_num(w[:,t])
        logw-=8*(pred-value[:,None,None])**2*obs[:,None,None]
        wins*=decay[None,None,:,None];counts*=decay[None,None,:,None]
        rate*=decay[0];rate[:,0]+=value;rate[:,1]+=obs
        add=obs[:,None,None]*valid[...,None]
        wins[ii,jj,kk,cell]+=add*value[:,None,None];counts[ii,jj,kk,cell]+=add
    result=(trace[np.arange(n),s.cutoff],trace,final_p,final_lw)
    return result+(heads,) if trace_heads else result


def size_walk(s,mode_probability=None):
    n=s.n;tmax=int(s.cutoff.max());cls=np.full(s.grid.shape,-1,int)
    for k,payout in enumerate((0,1,2,5,20)):cls[s.grid==payout]=k
    modes=np.array([METADATA['size_modes'][k] for k in s.mtype])
    llr=np.log(np.maximum(modes[:,1],1e-3)/np.maximum(modes[:,0],1e-3))
    offset=np.zeros(n) if mode_probability is None else np.log(mode_probability/(1-mode_probability))
    counts=np.zeros((n,5,5,4));machine=np.zeros((n,4));logw=np.zeros((n,5))
    result=np.zeros((n,4));decay=2.**(-1/80);ii=np.arange(n)[:,None];jj=np.arange(5)[None,:]
    for t in range(tmax+1):
        cell=np.zeros((n,5),int)
        for lag in range(1,5):cell[:,lag]=cls[:,t-lag] if t>=lag else -1
        cc=counts[ii,jj,np.maximum(cell,0)].copy()
        empty=(cc.sum(2)==0)|(cell<0)
        cc=np.where(empty[...,None],machine[:,None,:],cc)
        big=expit(np.clip((cc*llr[:,None,:]).sum(2)+offset[:,None],-50,50))
        pred=modes[:,0,None,:]+big[...,None]*(modes[:,1,None,:]-modes[:,0,None,:])
        ww=np.exp(logw-logw.max(1,keepdims=True));mix=(ww[...,None]*pred).sum(1)/ww.sum(1)[:,None]
        emit=s.cutoff==t;result[emit]=mix[emit]
        if t==cls.shape[1]:break
        value=cls[:,t];won=value>0;target=np.eye(4)[np.maximum(value-1,0)]
        logw-=8*((pred-target[:,None,:])**2).sum(2)*won[:,None]
        counts*=decay;machine*=decay
        for k in range(4):
            use=won&(value==k+1);machine[use,k]+=1
            counts[ii,jj,np.maximum(cell,0),k]+=use[:,None]*(cell>=0)
    return result


def beta_cell_evidence(s):
    """Best-pair parity-cell Beta marginal likelihood, no target probabilities."""
    w=s.win;tab=np.zeros((s.n,3,2,2))
    for j,(a,b) in enumerate(PAIRS):
        for t in range(max(a,b),w.shape[1]):
            good=np.isfinite(w[:,t])&np.isfinite(w[:,t-a])&np.isfinite(w[:,t-b])
            for cell in (0,1):
                mask=good&((w[:,t-a]==w[:,t-b])==cell)
                tab[mask,j,cell,0]+=w[mask,t];tab[mask,j,cell,1]+=1
    cnt=tab[...,1];rt=(tab[...,0]+.5)/(cnt+1)
    mean=tab[...,0].sum(2)/np.maximum(cnt.sum(2),1)
    score=(cnt*(rt-mean[...,None])**2).sum(2)/np.maximum(cnt.sum(2),1)
    chosen=tab[np.arange(s.n),score.argmax(1)]
    cw=chosen[:,:,0].reshape(-1,1);cn=chosen[:,:,1].reshape(-1,1)
    a=POP[:,1]*POP[:,2];b=(1-POP[:,1])*POP[:,2]
    ll=gammaln(a+cw)+gammaln(b+cn-cw)-gammaln(a+b+cn)-gammaln(a)-gammaln(b)+gammaln(a+b)
    machine=np.repeat(np.arange(s.n),2);valid=cn[:,0]>=5
    return ll[valid],machine[valid]


def size_evidence(s):
    ll=[];owner=[]
    for i in range(s.n):
        modes=np.array(METADATA['size_modes'][s.mtype[i]])
        for k,p in enumerate((1,2,5,20)):
            count=int((s.grid[i]==p).sum())
            if count:
                ll.extend([np.log(modes[:,k])]*count);owner.extend([i]*count)
    return np.asarray(ll),np.asarray(owner)


def em(ll,anchor,strength=8.):
    if not len(ll):return np.asarray(anchor)
    weights=np.asarray(anchor).copy()
    for _ in range(100):
        post=np.exp(ll+np.log(weights)-logsumexp(ll+np.log(weights),axis=1,keepdims=True))
        nxt=(post.sum(0)+strength*np.asarray(anchor))/(len(ll)+strength)
        if abs(nxt-weights).max()<1e-9:break
        weights=nxt
    return weights


def fit_weights(s,ll,owner,rows=None):
    rows=np.arange(s.n) if rows is None else rows
    included=np.isin(owner,rows);out={};default=DEFAULT if ll.shape[1]==3 else np.array([.5,.5])
    for kind in TYPES:
        mask=included&(s.mtype[owner]==kind)
        typed=em(ll[mask],default)
        out[(kind,None)]=typed
        for signal in (-1,0,1):
            use=mask&(s.signal[owner]==signal)
            out[(kind,signal)]=em(ll[use],typed)
    return out


def signal_evidence(s,ll,owner):
    learned=fit_weights(s,ll,owner);episode_diff={}
    # Fixed TRAIN episode halves; all group/type parameters fitted opposite fold.
    for f in (0,1):
        train_rows=np.flatnonzero(s.ep_idx%2!=f);test_rows=np.flatnonzero(s.ep_idx%2==f)
        params=fit_weights(s,ll,owner,train_rows)
        for i in test_rows:
            use=owner==i
            type_ll=logsumexp(ll[use]+np.log(params[(s.mtype[i],None)]),axis=1)
            signal_ll=logsumexp(ll[use]+np.log(params[(s.mtype[i],int(s.signal[i]))]),axis=1)
            episode_diff.setdefault(str(s.episode[i]),[]).extend((signal_ll-type_ll).tolist())
    values=np.array([np.mean(x) for x in episode_diff.values() if len(x)])
    rng=np.random.RandomState(473);boots=np.array([rng.choice(values,len(values)).mean() for _ in range(1000)])
    evidence={'train_crossfit_signal_minus_type_loglik_per_observation':float(values.mean()),
              'episode_bootstrap_ci95':np.percentile(boots,[2.5,97.5]).tolist(),
              'weights':{f'{k}:{g}':v.tolist() for (k,g),v in learned.items()}}
    return learned,evidence


def weights_for(s,learned,strength=1.,signal=True):
    default=DEFAULT if len(next(iter(learned.values())))==3 else np.array([.5,.5])
    target=np.array([learned[(k,int(g) if signal else None)] for k,g in zip(s.mtype,s.signal)])
    logits=np.log(default)+strength*(np.log(target)-np.log(default))
    return np.exp(logits-logsumexp(logits,axis=1,keepdims=True))


def detector_score(s,trace):
    """EWMA excess Brier, computed AFTER each observed spin, used only later."""
    w=s.win;ewma=np.zeros(s.n);score=np.full(trace.shape,np.nan)
    for t in range(w.shape[1]):
        p=trace[:,t];obs=np.isfinite(w[:,t]);value=np.nan_to_num(w[:,t])
        surprise=(value-p)**2-p*(1-p)
        ewma*=.8;ewma[obs]+=.2*surprise[obs]
        if t>=40:score[obs,t+1]=ewma[obs]
    return score


def trigger(s,score,threshold,window,peers=0):
    flags=np.zeros(s.n,bool)
    for i in range(s.n):
        now=int(s.cutoff[i]);rows=s.peers(i) if peers else [i]
        count=0
        for j in rows:
            recent=score[j,max(0,now-window):now+1]
            count+=bool(np.any(recent>threshold[s.mtype[j]]))
        flags[i]=count >= (peers if peers else 1)
    return flags


def fast_boost(pred,logw,flags,multiplier):
    lw=logw.copy();lw[:,:,0]+=np.log(multiplier)*flags[:,None]
    weights=np.exp(lw-lw.max((1,2),keepdims=True))
    return (weights*pred).sum((1,2))/weights.sum((1,2))


def contrast_detector_score(s,heads):
    """EWMA slow-minus-fast observed Brier; positive means fast memory wins."""
    w=s.win;ewma=np.zeros(s.n);score=np.full(heads.shape[:2],np.nan)
    for t in range(w.shape[1]):
        obs=np.isfinite(w[:,t]);value=np.nan_to_num(w[:,t])
        contrast=(value-heads[:,t,1])**2-(value-heads[:,t,0])**2
        ewma*=.8;ewma[obs]+=.2*contrast[obs]
        if t>=40:score[obs,t+1]=ewma[obs]
    return score
