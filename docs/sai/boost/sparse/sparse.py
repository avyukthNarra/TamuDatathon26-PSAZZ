"""Causal sparse-history win experts. Dense forecasts are passed through exactly.

T1a: distribute observed outcomes over uncertain lag cells using the current
running rate. T1b: use a lag tick's own earlier forecast if it was unobserved.
No missing outcome is ever added as a training target. T1c changes forgetting
only. T1d adds single/wider lag experts. Size forecasts remain the v4 baseline.
"""
import itertools
import numpy as np
from scipy.special import gammaln

PAIRS = ((1, 3), (4, 7), (2, 5))


def gapped(s):
    result = np.zeros(s.n, bool)
    for i in range(s.n):
        ticks = np.flatnonzero(np.isfinite(s.grid[i, :s.cutoff[i]]))
        result[i] = len(ticks) > 0 and len(ticks) < int(s.cutoff[i])-int(ticks[0])
    return result


def beta_rate(wins, counts, prior):
    prior = np.asarray(prior)
    a = prior[:, 1]*prior[:, 2]
    b = (1-prior[:, 1])*prior[:, 2]
    ww, nn = wins[..., None], counts[..., None]
    log = (np.log(prior[:, 0])+gammaln(a+ww)+gammaln(b+nn-ww)-gammaln(a+b+nn)
           -gammaln(a)-gammaln(b)+gammaln(a+b))
    post = np.exp(log-log.max(-1, keepdims=True))
    return (post*(a+ww)/(a+b+nn)).sum(-1)/post.sum(-1)


def win_head(s, prior, method='baseline', half_lives=(12., 40.), eta=8.,
             decay_missing=True, soft_weight=1., extra='none', extra_mass=.1,
             trace=False):
    """Run observed and missing ticks in order; emit before absorbing each tick.

    Fractional-cell updates assume independent lag uncertainty. They do not
    pretend a missing outcome was observed. trace is for synthetic causal tests.
    Accepts arbitrary query cutoffs; values at/after a query's cutoff are masked.
    """
    w = s.win.copy()
    w[np.arange(w.shape[1])[None, :] >= s.cutoff[:, None]] = np.nan
    n, maximum = w.shape
    pairs = list(PAIRS)
    if extra in ('wide', 'both'):
        pairs += [p for p in itertools.combinations(range(1, 9), 2) if p not in pairs]
    if extra in ('single', 'both'):
        pairs += [(i,) for i in range(1, 9)]
    ne, nh = len(pairs), len(half_lives)
    decay = np.broadcast_to(2.**(-1./np.array(half_lives)), (n, ne, nh))
    wins = np.zeros((n, ne, nh, 2))
    counts = np.zeros_like(wins)
    logw = np.zeros((n, ne, nh))
    if ne > 3:
        logw[:, :3] = np.log((1-extra_mass)/3)
        logw[:, 3:] = np.log(extra_mass/(ne-3))
    rate = np.zeros((n, 2))
    imputed = np.full_like(w, .5)
    result = np.zeros(n)
    traces = []
    base_decay = decay[:, 0, 0]
    for t in range(int(s.cutoff.max())+1):
        base = (rate[:, 0]+1)/(rate[:, 1]+2)
        soft_cell = np.zeros((n, ne))
        valid = np.zeros((n, ne), bool)
        complete = np.zeros_like(valid)
        for j, lags in enumerate(pairs):
            if t < max(lags):
                continue
            lag_values = [w[:, t-lag] for lag in lags]
            complete[:, j] = np.logical_and.reduce([np.isfinite(x) for x in lag_values])
            if method in ('expected', 'forward'):
                lp = [np.where(np.isfinite(x), x,
                               imputed[:, t-lag] if method == 'forward' else base)
                      for x, lag in zip(lag_values, lags)]
                valid[:, j] = True
            else:
                lp = [np.nan_to_num(x) for x in lag_values]
                valid[:, j] = complete[:, j]
            # Cell 1 = agreement for pairs, win for unary experts.
            soft_cell[:, j] = lp[0] if len(lp) == 1 else lp[0]*lp[1]+(1-lp[0])*(1-lp[1])
        posterior = beta_rate(wins, counts, prior)
        pred = ((1-soft_cell[:, :, None])*posterior[..., 0]
                +soft_cell[:, :, None]*posterior[..., 1])
        pred = np.where(valid[..., None], pred, base[:, None, None])
        weights = np.exp(logw-logw.max((1, 2), keepdims=True))
        pw = (weights*pred).sum((1, 2))/weights.sum((1, 2))
        result[s.cutoff == t] = pw[s.cutoff == t]
        if trace:
            traces.append(pw.copy())
        if t == maximum:
            break
        obs = np.isfinite(w[:, t]) & (t < s.cutoff)
        won = np.nan_to_num(w[:, t])
        # A missing tick's forecast is frozen now; no later observation revises it.
        imputed[:, t] = np.where(obs, won, pw)
        logw -= eta*(pred-won[:, None, None])**2*obs[:, None, None]
        fade = decay if decay_missing else np.where(obs[:, None, None], decay, 1.)
        wins *= fade[..., None]
        counts *= fade[..., None]
        rate *= (base_decay if decay_missing else np.where(obs, base_decay, 1.))[:, None]
        rate[:, 0] += won*obs
        rate[:, 1] += obs
        strength = np.where(complete, 1., soft_weight)*valid*obs[:, None]
        for cell, share in ((0, 1-soft_cell), (1, soft_cell)):
            add = strength[:, :, None]*share[:, :, None]
            wins[..., cell] += add*won[:, None, None]
            counts[..., cell] += add
    return (result, np.array(traces), counts) if trace else result


def forecast(s, baseline, sizes, prior, **params):
    """Exact copy of dense forecast bytes, sparse queries recomputed only."""
    mask = gapped(s)
    out = baseline.copy()
    if mask.any():
        indices = np.flatnonzero(mask)
        pw = win_head(s.subset(indices), prior, **params)
        p = np.maximum(np.column_stack((1-pw, pw[:, None]*sizes[indices])), 1e-4)
        out[indices] = p/p.sum(1, keepdims=True)
    return out
