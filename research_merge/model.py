"""Exact level/pair posterior with independent and shared smooth reversals."""
from __future__ import annotations

from collections import defaultdict
import numpy as np
from scipy.special import logsumexp, expit

from research_exact.model import ExactModel, PAIRS, pair_alpha, softmax, stable_masked_exact


def reversal_grid(model, m, width=.1, fractions=None, hypothesis_eps=0.):
    """Return evidence and next-win probability for each smooth reversal center."""
    z, p, lw = model.grids[m.kind, m.signal]
    if hypothesis_eps:
        ph = p.copy()
        ph[:,0] = (1-hypothesis_eps)*ph[:,0] + hypothesis_eps*.5
        zh = z.copy()
        zh[:,0] = np.log(ph[:,0]/(1-ph[:,0]))
    else:
        zh = z
    alpha = pair_alpha(m)
    y = m.seq[:m.cutoff]
    centers = np.asarray(fractions if fractions is not None else np.arange(.25, .951, .05)) * m.cutoff
    span = width * m.cutoff
    out_e, out_p = [], []
    for cp in centers:
        g = np.clip((cp - np.arange(m.cutoff)) / max(span / 2, 1e-9), -1, 1)
        lp = np.zeros((len(PAIRS), 2, len(p)))
        for k in range(len(PAIRS)):
            for d in range(2):
                parity = 2*alpha[k]-1
                if d: parity = -parity
                parity = np.where(alpha[k] < 0, 0., parity)
                zz = zh.mean(1)[:, None] + ((zh[:, 1]-zh[:, 0])/2)[:, None] * parity[None, :] * g[None, :]
                pp = np.clip(expit(zz), 1e-12, 1-1e-12)
                observed = y >= 0
                yy = y[observed] > 0
                lp[k, d] = (yy[None, :] * np.log(pp[:, observed]) +
                            (~yy)[None, :] * np.log1p(-pp[:, observed])).sum(axis=1)
        lp += lw[None, None, :] + np.log(model.pair_prior)[:, None, None] - np.log(2)
        lp = lp.reshape(-1)
        evidence = logsumexp(lp)
        # Current alpha at the next tick, smoothly continued to cutoff.
        aa = np.array([float((m.seq[m.cutoff-a] > 0) == (m.seq[m.cutoff-b] > 0))
                       if m.seq[m.cutoff-a] >= 0 and m.seq[m.cutoff-b] >= 0 else .5
                       for a,b in PAIRS])
        gn = np.clip((cp-m.cutoff)/(span/2), -1, 1)
        pn = np.empty((len(PAIRS), 2, len(p)))
        for k in range(len(PAIRS)):
            for d in range(2):
                parity = 2*aa[k]-1
                if d: parity = -parity
                if aa[k] < 0: parity = 0.
                pn[k,d] = expit(z.mean(1) + ((z[:,1]-z[:,0])/2)*parity*gn)
        pred = pn.reshape(-1)
        out_e.append(evidence)
        out_p.append(float(np.exp(lp-evidence) @ pred))
    return np.asarray(out_e), np.asarray(out_p)


def merged_predict(model, machines, mode='merged', width=.1, hypothesis_eps=0.):
    """Modes: static, independent, shared, all, or posterior mixture over all three."""
    by_episode = defaultdict(list)
    for m in machines:
        by_episode[m.episode_id].append(m)
    predictions, diag = {k:{} for k in ('merged','static','independent','shared')}, {}
    centers = np.arange(.25, .951, .05)
    for eid, ms in sorted(by_episode.items()):
        if ms[0].short:
            for m in ms:
                value = stable_masked_exact(model, m, hypothesis_eps)[1]
                for k in predictions: predictions[k][m.query_id] = value
            diag[eid] = {'short': True}
            continue
        stat = [model.stable(m, hypothesis_eps)[:2] for m in ms]
        rev = [reversal_grid(model, m, width, centers, hypothesis_eps) for m in ms]
        # Per-machine reversal integrates its own center and latent pair/direction/level.
        inde = [logsumexp(e)-np.log(len(e)) for e, _ in rev]
        indep = [float(softmax(e) @ p) for e, p in rev]
        # Episode-shared reversal integrates a single center after multiplying evidence.
        shared_joint = np.sum(np.stack([r[0] for r in rev]), axis=0) - np.log(len(centers))
        shared_center = softmax(shared_joint)
        shared_ev = logsumexp(shared_joint)
        sharedp = [float(shared_center @ p) for _, p in rev]
        # Equal prior over static, independent reversals, and shared reversal.
        comp = np.array([sum(v[0] for v in stat), sum(inde), shared_ev])
        post = softmax(comp)
        options = {'static': [v[1] for v in stat], 'independent': indep, 'shared': sharedp}
        for j, m in enumerate(ms):
            predictions['merged'][m.query_id] = float(post @ np.asarray([options[k][j] for k in options]))
            for k in ('static','independent','shared'):
                predictions[k][m.query_id] = options[k][j]
        diag[eid] = {'short': False, 'component_posterior': post.tolist(),
                     'component_log_evidence': comp.tolist(),
                     'shared_center_fraction': float(shared_center @ centers)}
    return (predictions if mode == 'all' else predictions[mode]), diag

