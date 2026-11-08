"""Reference screened p-biased Walsh BMA; fixed two-pass JJ/PG approximation.

The 42 supports are enumerated exactly, using the reference's shared 7-feature
variational precision (not refitting PG scales for each support). Prior c0=2,
Occam window=3 nats, and multiplicity penalty log C(36,d) are fixed.
"""
import sys
from pathlib import Path
from itertools import combinations
from math import comb
import numpy as np
from scipy.special import expit

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'win'))
from logistic import feature_matrix

SUBSETS_K6 = [[0]+list(c) for d in range(4) for c in combinations(range(1,7),d)]
MULT_PENALTY_K6 = np.array([np.log(comb(36,len(s)-1)) for s in SUBSETS_K6])
PAIRS = list(combinations(range(8),2))
HALF_LIVES = (12.,40.,None)


def centered_history(sequence,half_life):
    """Feature row at tick t uses only observations strictly before t.

    The EW centering rate uses this horizon's forgetting clock, not future
    history. The query tick is len(sequence), matching harness cutoff semantics.
    """
    sequence=np.asarray(sequence,float)
    observed=np.isfinite(sequence)&(sequence>=0)
    clean=np.where(observed,sequence,np.nan)
    raw=feature_matrix(clean,np.arange(len(clean)+1),full=True)[:,1:9]
    mean=np.zeros(len(clean)+1)
    gamma=1. if half_life is None else 2.**(-1./half_life)
    total=count=0.
    for t in range(len(clean)+1):
        mean[t]=np.clip(total/count if count else 0.,-.85,.85)
        if t<len(clean):
            total*=gamma;count*=gamma
            if observed[t]:
                total+=1. if clean[t]>0 else -1.
                count+=1.
    known=raw!=0
    singles=np.where(known,(raw-mean[:,None])/np.sqrt(1-mean[:,None]**2),0.)
    features=np.column_stack([singles]+[singles[:,a]*singles[:,b] for a,b in PAIRS])
    missing=np.column_stack([~known]+[~(known[:,a]&known[:,b]) for a,b in PAIRS])
    y=np.where(observed,clean>0,0.).astype(float)
    return features[:-1],y,observed,features[-1],missing[-1]


def walsh_bma_predict_horizon(X_hist,y_hist,obs_mask,x_next,next_missing_mask,
                              gamma,c0=2.,c_slab=1.35,diagnostics=False):
    """Reference Experiment 1 implementation, stable ties and expit added."""
    T=len(y_hist)
    ticks_ago=np.arange(T-1,-1,-1,dtype=float)
    w=(gamma**ticks_ago)*obs_mask
    w_sum=np.sum(w)
    if w_sum<1e-6:
        return (.5,{'top_indices':[],'retained_models':42}) if diagnostics else .5
    y_bar=np.sum(w*y_hist)/w_sum
    num=np.abs(X_hist.T@(w*(y_hist-y_bar)))
    den=np.sqrt((X_hist**2).T@w+4.)
    # Stable index tie-break is deterministic and does not use targets.
    top_idx=np.argsort(-(num/den),kind='stable')[:6]
    X_sub=np.column_stack([np.ones(T),X_hist[:,top_idx]])
    P0_diag=np.array([1./c0**2]+[1./c_slab**2]*6)
    h_vec=X_sub.T@(w*(y_hist-.5))
    omega=.25*w
    for _ in range(2):
        P_mat=np.diag(P0_diag)+(X_sub.T*omega)@X_sub
        Sigma=np.linalg.inv(P_mat)
        mu=Sigma@h_vec
        proj_mean=X_sub@mu
        proj_var=np.sum((X_sub@Sigma)*X_sub,axis=1)
        xi=np.maximum(np.sqrt(np.maximum(proj_mean**2+proj_var,0.)),1e-6)
        omega=w*np.tanh(.5*xi)/(2.*xi)
    P_mat=np.diag(P0_diag)+(X_sub.T*omega)@X_sub
    x_q_full=np.r_[1.,x_next[top_idx]]
    miss_q_full=np.r_[False,next_missing_mask[top_idx]]
    log_evidences=np.empty(42);preds=np.empty(42)
    for idx,s in enumerate(SUBSETS_K6):
        P_s=P_mat[np.ix_(s,s)];h_s=h_vec[s]
        Sigma_s=np.linalg.inv(P_s);beta_s=Sigma_s@h_s
        sign,logdet=np.linalg.slogdet(P_s)
        if sign!=1:raise ArithmeticError('Nonpositive submodel precision')
        d=len(s)-1
        log_evidences[idx]=(-.5*logdet+.5*(h_s@beta_s)
                            -.5*d*np.log(c_slab**2)-MULT_PENALTY_K6[idx])
        query=x_q_full[s];missing=miss_q_full[s]
        mean=float(query@beta_s)
        variance=float(query@Sigma_s@query)
        if np.any(missing):
            variance+=float(np.sum(beta_s[missing]**2+np.diag(Sigma_s)[missing]))
        z=mean/np.sqrt(1.+np.pi/8.*max(variance,0.))
        preds[idx]=expit(np.clip(z,-20.,20.))
    maximum=np.max(log_evidences)
    keep=log_evidences>=maximum-3.
    weights=np.zeros(42);weights[keep]=np.exp(log_evidences[keep]-maximum)
    weights/=weights.sum()
    result=float(weights@preds)
    if diagnostics:
        return result,{'top_indices':top_idx.tolist(),'retained_models':int(keep.sum()),
                       'model_weights':weights.tolist(),'log_evidences':log_evidences.tolist()}
    return result


def predict(split,c_slab=1.35):
    """No labels, conditions, or target probabilities enter prediction."""
    bank=np.empty((split.n,3))
    retained=np.zeros((split.n,3),int)
    top_features=np.zeros((split.n,3,6),int)
    for i in range(split.n):
        sequence=np.asarray(split.grid[i,:int(split.cutoff[i])],float)
        for j,half_life in enumerate(HALF_LIVES):
            args=centered_history(sequence,half_life)
            gamma=1. if half_life is None else 2.**(-1./half_life)
            p,info=walsh_bma_predict_horizon(*args,gamma,c_slab=c_slab,diagnostics=True)
            bank[i,j]=p;retained[i,j]=info['retained_models']
            top_features[i,j]=info['top_indices'] if info['top_indices'] else [-1]*6
    return bank,{'retained_models':retained,'top_features':top_features}
