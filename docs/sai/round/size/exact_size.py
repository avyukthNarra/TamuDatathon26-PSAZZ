"""Bayesian machine-level choice over eight size switches and a static expert.

Forecast at tick t precedes absorbing tick t. Hypotheses are lag1+/-, ...,
lag4+/-, static50/50. Only observed WIN SIZES contribute likelihood. The
likelihood evidence is discounted in calendar ticks, retaining the fixed prior.
"""
import numpy as np
from scipy.special import logsumexp

HALF_LIVES=(12.,40.,np.inf)
HYPOTHESIS_NAMES=tuple(f'lag{k}{d}' for k in range(1,5) for d in ('+','-'))+('static',)
HYPOTHESIS_PRIOR=np.array([.11]*8+[.12])


def train_big_win_prior(train):
    out={}
    for kind in sorted(set(train.mtype)):
        x=train.grid[train.mtype==kind]
        observed=np.isfinite(x)
        # The running P(win)*P(big|win) is exactly P(big payout). Two initial
        # population pseudo-observations stabilize the earliest missing lags.
        out[str(kind)]=float(((x>=5)&observed).sum()+.5)/float(observed.sum()+1)
    return out


class ExactSizeSwitch:
    def __init__(self,size_modes,big_win_prior,half_lives=HALF_LIVES):
        self.size_modes={}
        for kind,spec in size_modes.items():
            modes=np.asarray(spec['modes'] if isinstance(spec,dict) else spec,float)
            if modes.shape!=(2,4):raise ValueError('This exact switch requires two four-class size modes per type')
            modes=modes/modes.sum(1,keepdims=True)
            modes=modes[np.argsort(modes@np.array([1.,2.,5.,20.]))]
            self.size_modes[kind]=modes
        self.big_win_prior=big_win_prior
        self.half_lives=np.asarray(half_lives,float)
        self.decays=np.exp2(-1/self.half_lives)

    def predict(self,split,return_details=False):
        n=split.n;c=len(self.half_lives)
        modes=np.array([self.size_modes[k] for k in split.mtype])
        small,big=modes[:,0],modes[:,1]
        log_prior=np.log(HYPOTHESIS_PRIOR)[None,None,:]
        log_evidence=np.zeros((c,n,9))
        total=np.full(n,2.)
        nbig=2*np.array([self.big_win_prior[k] for k in split.mtype])
        out=np.zeros((c,n,4));posteriors=np.zeros((c,n,9))
        final_missing=np.zeros(n,int)
        for t in range(int(split.cutoff.max())+1):
            fallback=nbig/total
            b=np.empty((n,4));missing=np.zeros(n,int)
            for k in range(1,5):
                previous=split.grid[:,t-k] if t>=k else np.full(n,np.nan)
                known=np.isfinite(previous)
                b[:,k-1]=np.where(known,previous>=5,fallback)
                missing+=~known
            probability_big=np.column_stack((np.stack((b,1-b),axis=2).reshape(n,8),np.full(n,.5)))
            q=small[:,None,:]+probability_big[...,None]*(big-small)[:,None,:]
            logs=log_prior+log_evidence
            posterior=np.exp(logs-logsumexp(logs,axis=2,keepdims=True))
            forecast=np.einsum('cnh,nhk->cnk',posterior,q)
            emit=split.cutoff==t
            out[:,emit]=forecast[:,emit];posteriors[:,emit]=posterior[:,emit];final_missing[emit]=missing[emit]
            if t==split.grid.shape[1]:break
            value=split.grid[:,t];observed=np.isfinite(value);won=observed&(value>0)
            cls=np.zeros(n,int)
            for i,payout in enumerate((1,2,5,20)):cls[value==payout]=i
            likelihood=q[np.arange(n)[:,None],np.arange(9)[None,:],cls[:,None]]
            increment=np.log(np.maximum(likelihood,1e-12))*won[:,None]
            log_evidence*=self.decays[:,None,None]
            log_evidence+=increment[None,:,:]
            total+=observed;nbig+=observed&(value>=5)
        if return_details:
            return out,{'posterior':posteriors,'final_missing_triggers':final_missing,
                        'half_lives':self.half_lives.copy()}
        return out
