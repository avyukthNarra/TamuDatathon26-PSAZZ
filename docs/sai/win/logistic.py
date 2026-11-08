"""Per-machine Gaussian-prior Walsh logistic regression with Laplace prediction.

A lag is +/-1 for an observed win/loss and 0 for an unobserved tick. All
history rows and query features exclude tick >= cutoff. No validation targets
enter population fitting or next-outcome half-life selection.
"""
import itertools
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit

KNOWN_PAIRS = ((1,3),(2,5),(4,7))


def terms(full=False):
    return [()] + [(i,) for i in range(1,9)] + list(
        itertools.combinations(range(1,9),2) if full else KNOWN_PAIRS)


def feature_matrix(sequence, ticks, full=False):
    sequence=np.asarray(sequence,float)
    ticks=np.asarray(ticks,int)
    lags=np.zeros((len(ticks),8))
    for lag in range(1,9):
        positions=ticks-lag
        good=(positions>=0)&(positions<len(sequence))
        values=np.full(len(ticks),np.nan)
        values[good]=sequence[positions[good]]
        lags[:,lag-1]=np.where(np.isfinite(values),np.where(values>0,1.,-1.),0.)
    columns=[np.ones(len(ticks))]+[lags[:,i] for i in range(8)]
    columns += [lags[:,a-1]*lags[:,b-1] for a,b in terms(full)[9:]]
    return np.column_stack(columns)


def machine_rows(split, full=False):
    """Cached histories/features only; truth and condition labels are ignored."""
    result=[]
    for i in range(split.n):
        cutoff=int(split.cutoff[i])
        seq=np.array(split.grid[i,:cutoff],copy=True)
        ticks=np.flatnonzero(np.isfinite(seq))
        x=feature_matrix(seq,ticks,full)
        y=(seq[ticks]>0).astype(float)
        query=feature_matrix(seq,[cutoff],full)[0]
        result.append((x,y,ticks,query,cutoff,str(split.mtype[i])))
    return result


def objective(beta,x,y,weights,precision):
    z=x@beta
    return float(np.dot(weights,np.logaddexp(0,z)-y*z)+.5*np.dot(precision*beta,beta))


def fit_laplace(x,y,weights,variance,maxiter=60,tolerance=1e-8):
    """Damped Newton with analytic gradient/Hessian and positive prior precision."""
    precision=1/np.asarray(variance,float)
    if np.any(precision<=0) or not np.isfinite(precision).all():
        raise ValueError('Gaussian prior variances must be finite and positive')
    beta=np.zeros(x.shape[1]); converged=False
    for iteration in range(maxiter):
        p=expit(x@beta)
        gradient=x.T@(weights*(p-y))+precision*beta
        hessian=(x.T*(weights*p*(1-p)))@x+np.diag(precision)
        factor=cho_factor(hessian,lower=True,check_finite=False)
        if np.max(np.abs(gradient))<tolerance:
            converged=True;break
        step=cho_solve(factor,gradient,check_finite=False)
        old=objective(beta,x,y,weights,precision)
        scale=1.
        while scale>2.**-30:
            proposal=beta-scale*step
            if objective(proposal,x,y,weights,precision)<=old-1e-4*scale*np.dot(gradient,step)+1e-13:
                break
            scale*=.5
        beta=proposal
        if np.max(abs(scale*step))<tolerance:
            converged=True;break
    p=expit(x@beta)
    gradient=x.T@(weights*(p-y))+precision*beta
    hessian=(x.T*(weights*p*(1-p)))@x+np.diag(precision)
    factor=cho_factor(hessian,lower=True,check_finite=False)
    covariance=cho_solve(factor,np.eye(len(beta)),check_finite=False)
    if not converged and np.max(abs(gradient))>1e-6:
        raise RuntimeError(f'Newton did not converge: gradient {np.max(abs(gradient))}')
    return beta,covariance,{'iterations':iteration+1,'max_gradient':float(np.max(abs(gradient)))}


def weights_at(ticks,cutoff,half_life):
    return np.ones(len(ticks)) if half_life is None else np.exp2(-(cutoff-1-ticks)/half_life)


def moderated(beta,covariance,query):
    mean=float(query@beta)
    variance=max(float(query@covariance@query),0.)
    return float(expit(mean/np.sqrt(1+np.pi*variance/8)))


def empirical_bayes(machines,half_life,iterations=5):
    """Zero-mean type/term Gaussian EB via Laplace posterior second moments.

    Each update maximizes the expected log Gaussian prior with respect to its
    variance: tau_k^2 = mean(beta_k^2 + Sigma_kk). Histories only; signs are free.
    A fixed five-step approximate EM scheme avoids dev-dependent stopping.
    """
    types=sorted({r[-1] for r in machines});dimension=machines[0][0].shape[1]
    variance={kind:np.full(dimension,4.) for kind in types}
    history=[]
    for iteration in range(iterations):
        moments={kind:[] for kind in types}
        gradients=[]
        for x,y,ticks,query,cutoff,kind in machines:
            b,cov,info=fit_laplace(x,y,weights_at(ticks,cutoff,half_life),variance[kind])
            moments[kind].append(b*b+np.diag(cov));gradients.append(info['max_gradient'])
        variance={kind:np.clip(np.mean(moments[kind],axis=0),1e-3,100.) for kind in types}
        history.append({'iteration':iteration+1,'variance':{k:v.tolist() for k,v in variance.items()},
                        'max_newton_gradient':float(max(gradients))})
    return variance,history


def predict(machines,variance,half_life,moderate=True):
    out=[];diagnostics=[]
    for x,y,ticks,query,cutoff,kind in machines:
        b,cov,info=fit_laplace(x,y,weights_at(ticks,cutoff,half_life),variance[kind])
        out.append(moderated(b,cov,query) if moderate else float(expit(query@b)))
        diagnostics.append(info)
    return np.array(out),{'max_gradient':max(i['max_gradient'] for i in diagnostics),
                          'max_iterations':max(i['iterations'] for i in diagnostics)}
