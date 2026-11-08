"""Train-only empirical Bayes parity model with joint episode change inference.

No validation answers are used by this module. Missing lags are explicitly unknown.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

# Every lag pair with 1 <= a < b <= 8. fit() replaces PAIRS from train votes.
# If no pair clears MIN_SHARE of machines, this full set is the fallback.
ALL_PAIRS = tuple(combinations(range(1, 9), 2))
PAIRS = np.array(ALL_PAIRS, dtype=int)
MIN_SHARE = 0.05
MIN_MARGIN = 2.0
MIN_CELL = 5
TYPES = ('steady', 'balanced', 'jackpot')
POSITIVE = (1, 2, 5, 20)


def _bern_ll(wins, n):
    if n <= 0:
        return 0.0
    p = np.clip(wins / n, 1e-6, 1.0 - 1e-6)
    return float(wins * np.log(p) + (n - wins) * np.log(1.0 - p))


def learn_pairs(machines, min_share=MIN_SHARE, min_margin=MIN_MARGIN, min_cell=MIN_CELL):
    """Best pair per machine by two-cell likelihood gain, then a share cut."""
    votes = {}
    n = len(machines)
    for machine in machines:
        scored = []
        known = machine.seq[:machine.cutoff] >= 0
        for lag_a, lag_b in ALL_PAIRS:
            idx = np.arange(lag_b, machine.cutoff)
            if len(idx) == 0:
                continue
            ok = known[idx] & (machine.seq[idx - lag_a] >= 0) & (machine.seq[idx - lag_b] >= 0)
            ticks = idx[ok]
            if len(ticks) < 2 * min_cell:
                continue
            agree = (machine.seq[ticks - lag_a] > 0) == (machine.seq[ticks - lag_b] > 0)
            won = machine.seq[ticks] > 0
            n_agree = int(agree.sum())
            n_differ = len(ticks) - n_agree
            if n_agree < min_cell or n_differ < min_cell:
                continue
            w_agree = int(won[agree].sum())
            w_differ = int(won[~agree].sum())
            gain = (_bern_ll(w_differ, n_differ) + _bern_ll(w_agree, n_agree)
                    - _bern_ll(w_differ + w_agree, n_differ + n_agree))
            scored.append((gain, (int(lag_a), int(lag_b))))
        if len(scored) < 2:
            continue
        scored.sort(key=lambda row: -row[0])
        if scored[0][0] - scored[1][0] < min_margin:
            continue
        votes[scored[0][1]] = votes.get(scored[0][1], 0) + 1
    kept = [pair for pair, count in votes.items() if n and count / n >= min_share]
    if not kept:
        kept = list(ALL_PAIRS)
    kept = sorted(kept)
    top = sorted(votes.items(), key=lambda item: -item[1])[:6]
    print('pair support', kept, 'top votes', top, 'of', n, flush=True)
    return np.array(kept, dtype=int)


def softmax(x, axis=None):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


@dataclass
class Machine:
    query_id: str
    episode_id: str
    kind: str
    signal: int
    cutoff: int
    seq: np.ndarray

    @property
    def short(self):
        return np.sum(self.seq[:self.cutoff] >= 0) < self.cutoff * .75


def load_machines(public, split, labels=False):
    from slot_forecasting.io import load_inputs, read_labels, read_rows
    qs, hist = load_inputs(Path(public) / split)
    by_key = {(q.episode_id, q.machine_id): np.full(q.cutoff + 1, -1, dtype=np.int8) for q in qs}
    for p in hist:
        by_key[p.episode_id, p.machine_id][p.tick] = POSITIVE.index(p.payout) + 1 if p.payout else 0
    ys = read_labels(Path(public) / split / 'labels.csv', qs) if labels else {}
    if labels:
        for q in qs:
            y = ys[q.query_id]
            by_key[q.episode_id, q.machine_id][q.cutoff] = POSITIVE.index(y) + 1 if y else 0
    machines = [Machine(q.query_id, q.episode_id, q.machine_type, q.signal, q.cutoff,
                        by_key[q.episode_id, q.machine_id]) for q in qs]
    conditions = {}
    path = Path(public) / split / 'conditions.csv'
    if path.exists():
        conditions = {r['query_id']: r['condition'] for r in read_rows(path, ('query_id', 'condition'))}
    return machines, conditions, qs, hist, ys


def pair_alpha(machine, include_label=False):
    """Early-live indicator (direction=agree); missing context marked -1."""
    n = machine.cutoff + int(include_label)
    t = np.arange(n)
    out = np.full((len(PAIRS), n), -1., dtype=float)
    for j, (a, b) in enumerate(PAIRS):
        out[j, t < b] = .5
        valid = t >= b
        tt = t[valid]
        ok = (machine.seq[tt-a] >= 0) & (machine.seq[tt-b] >= 0)
        out[j, tt[ok]] = ((machine.seq[tt[ok]-a] > 0) == (machine.seq[tt[ok]-b] > 0))
    return out


def cell_counts(machine, include_label=False, flip=None):
    """Counts indexed pair, outcome, alpha in {0,1,1/2}."""
    alpha = pair_alpha(machine, include_label)
    if flip is not None:
        alpha = np.where(alpha < 0, -1, np.where(np.arange(alpha.shape[1]) >= flip, 1-alpha, alpha))
    y = machine.seq[:alpha.shape[1]]
    counts = np.zeros((len(PAIRS), 2, 3))
    for outcome in (0, 1):
        mask = (y >= 0) & ((y > 0) == outcome)
        for k, value in enumerate((0., 1., .5)):
            counts[:, outcome, k] = np.sum((alpha == value) & mask, axis=1)
    return counts


def initial_levels(machines, conditions):
    """Initialize from train stable MLEs; EM removes sampling variance next."""
    rows = {k: [] for k in TYPES}
    pair_counts = np.ones(len(PAIRS)) * .05
    for m in machines:
        if conditions[m.query_id] != 'stable':
            continue
        counts = cell_counts(m, True)[:, :, :2]
        n = counts.sum(1)
        rates = (counts[:, 1] + .5) / (n + 1)
        ll = np.sum(counts[:, 1] * np.log(rates) + counts[:, 0] * np.log1p(-rates), axis=1)
        j = int(np.argmax(ll))
        pair_counts[j] += 1
        if np.min(n[j]) < 10 or np.ptp(rates[j]) < .2:
            continue
        z = np.log(rates[j] / (1-rates[j]))
        z.sort()
        rows[m.kind].append((m.signal, *z))
    pars = {}
    for kind, values in rows.items():
        values = np.atleast_2d(np.array(values, dtype=float))
        if values.shape[1] != 3 or len(values) < 2:
            # Too little stable evidence for this type. A flat logit prior keeps
            # the filter runnable when the pair support is wrong for the kit.
            pars[kind] = {'mean': [-1.0, 1.0], 'slope': 0.0, 'cov': (np.eye(2) * 0.25).tolist()}
            continue
        x = np.column_stack([np.ones(len(values)), values[:, 0]])
        beta = np.linalg.lstsq(x, values[:, 1:], rcond=None)[0]
        residual = values[:, 1:] - x @ beta
        cov = np.cov(residual.T)
        cov = .5 * cov + np.eye(2) * .12
        pars[kind] = {'mean': beta[0].tolist(), 'slope': float(beta[1].mean()), 'cov': cov.tolist()}
    return pars, pair_counts / pair_counts.sum()


def level_grid(par, signal, size=25):
    """Fixed physical logit grid; normal density integrated by rectangle rule."""
    slope = np.asarray(par['slope'])
    mean = np.array(par['mean']) + (slope if slope.ndim else np.repeat(slope, 2)) * signal
    cov = np.array(par['cov'])
    sd = np.sqrt(np.diag(cov))
    axes = [mean[k] + np.linspace(-4.5, 4.5, size) * max(sd[k], .2) for k in range(2)]
    z = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 2)
    delta = z-mean
    lw = -.5 * np.einsum('ni,ij,nj->n', delta, np.linalg.inv(cov), delta)
    lw[z[:, 0] >= z[:, 1]] = -np.inf
    lw -= logsumexp(lw)
    return z, expit(z), lw


def count_likelihood(counts, p):
    """Both directions. Return [pair, direction, level-grid]."""
    pc = np.column_stack([p[:, 0], p[:, 1], p.mean(1)])
    logp, logq = np.log(pc), np.log1p(-pc)
    forward = counts[:, 1] @ logp.T + counts[:, 0] @ logq.T
    reverse = counts[:, 1, [1, 0, 2]] @ logp.T + counts[:, 0, [1, 0, 2]] @ logq.T
    return np.stack([forward, reverse], axis=1)


def fit_levels(machines, conditions, iterations=12, size=25):
    pars, pair_prior = initial_levels(machines, conditions)
    selected = [m for m in machines if conditions[m.query_id] == 'stable']
    counts = [cell_counts(m, True) for m in selected]
    for iteration in range(iterations):
        moments = {k: [] for k in TYPES}
        pair_sum = np.full(len(PAIRS), .025)
        total_ll = 0.
        grids = {(kind, sig): level_grid(pars[kind], sig, size) for kind in TYPES for sig in (-1, 0, 1)}
        for m, count in zip(selected, counts):
            z, p, lw = grids[m.kind, m.signal]
            lp = count_likelihood(count, p) + lw + np.log(pair_prior)[:, None, None] - np.log(2)
            total_ll += logsumexp(lp)
            post = softmax(lp)
            pair_sum += post.sum((1, 2))
            weights = post.sum((0, 1))
            ez = weights @ z
            ezz = np.einsum('n,ni,nj->ij', weights, z, z)
            moments[m.kind].append((m.signal, ez, ezz))
        for kind, rows in moments.items():
            sig = np.array([r[0] for r in rows])
            ez = np.array([r[1] for r in rows])
            # Shared signal coefficient and distinct dead/live intercepts.
            centered_sig = sig - sig.mean()
            slope = np.sum(centered_sig[:, None] * (ez-ez.mean(0))) / (2 * np.sum(centered_sig**2))
            intercept = ez.mean(0) - slope * sig.mean()
            mu = intercept + sig[:, None] * slope
            cov = sum(r[2] - np.outer(r[1], v) - np.outer(v, r[1]) + np.outer(v, v)
                      for r, v in zip(rows, mu)) / len(rows)
            eig, vec = np.linalg.eigh(cov)
            cov = (vec * np.maximum(eig, .025)) @ vec.T
            pars[kind] = {'mean': intercept.tolist(), 'slope': float(slope), 'cov': cov.tolist()}
        pair_prior = pair_sum / pair_sum.sum()
        print(f'EB {iteration+1}: stable log evidence {total_ll:.3f}', flush=True)
    return pars, pair_prior


def fit_schedule(machines, conditions):
    """Train ML timing estimates using train-identified early rule and levels."""
    drift_rows, change_rows = [], []
    for eid in sorted(set(m.episode_id for m in machines)):
        ms = [m for m in machines if m.episode_id == eid]
        condition = conditions[ms[0].query_id]
        if condition not in ('drift', 'shared_change'):
            continue
        prepared = []
        for m in ms:
            alpha = pair_alpha(m, True)
            n = int(.4*m.cutoff)
            y = m.seq[:alpha.shape[1]] > 0
            nn = np.stack([np.sum(alpha[:, :n] == a, axis=1) for a in (0., 1.)], axis=1)
            ww = np.stack([np.sum((alpha[:, :n] == a) & y[:n], axis=1) for a in (0., 1.)], axis=1)
            rates = (ww+.5)/(nn+1)
            ll = np.sum(ww*np.log(rates)+(nn-ww)*np.log1p(-rates), axis=1)
            j = int(np.argmax(ll))
            prepared.append((m, alpha[j], y, rates[j]))
        if condition == 'drift':
            def loss(x):
                anchor, length = x
                result = 0.
                for m, a, y, rates in prepared:
                    u = np.clip((np.arange(len(y))-(m.cutoff-anchor))/length, 0, 1)
                    w = a*(1-u)+(1-a)*u
                    p = rates[0]*(1-w)+rates[1]*w
                    p[a == .5] = rates.mean()
                    result -= np.sum(np.where(y, np.log(p), np.log1p(-p)))
                return result
            res = minimize(loss, [100., 60.], method='Nelder-Mead',
                           bounds=((20, 180), (5, 150)), options={'maxiter': 250})
            drift_rows.append((eid, *res.x, float(res.fun)))
        else:
            cs = np.arange(8, ms[0].cutoff)
            scores = []
            for c in cs:
                ll = 0.
                for m, a, y, rates in prepared:
                    aa = np.where(np.arange(len(y)) >= c, 1-a, a)
                    p = rates[0]*(1-aa)+rates[1]*aa
                    ll += np.sum(np.where(y, np.log(p), np.log1p(-p)))
                scores.append(ll)
            change_rows.append((eid, ms[0].cutoff, int(cs[np.argmax(scores)])))
    dr = np.array([r[1:3] for r in drift_rows])
    cr = np.array([r[1:3] for r in change_rows])
    # A fraction prior is better identified than a noisy intercept with 24 episodes.
    fraction = float(np.mean(cr[:, 1]/cr[:, 0]))
    sd = float(np.std(cr[:, 1]-fraction*cr[:, 0]))
    return {'anchor': float(np.median(dr[:, 0])), 'duration': float(np.median(dr[:, 1])),
            'flip_fraction': fraction, 'flip_sd': sd,
            'drift_train_fits': drift_rows, 'shared_train_fits': change_rows}


def size_states(max_lag=8):
    # Static states first, then every threshold/lag/direction: no canonical lag assumption.
    return [('static', 0, 0), ('static', 1, 0)] + [
        (threshold, lag, direction) for threshold in (1, 2, 5) for lag in range(1, max_lag+1)
        for direction in (0, 1)]


def size_assignments(m, include_label=False):
    n = m.cutoff + int(include_label)
    # Decode ordinal classes back to payouts for threshold rules.
    payout = np.array([-1, 0, 1, 2, 5, 20])[m.seq+1]
    times = np.flatnonzero(m.seq[:n] > 0)
    cats = m.seq[times]-1
    states = size_states()
    alpha = np.full((len(states), len(times)), .5)
    cutoff_alpha = np.full(len(states), .5)
    for s, (threshold, lag, direction) in enumerate(states):
        if threshold == 'static':
            alpha[s] = lag
            cutoff_alpha[s] = lag
        else:
            ok = times >= lag
            ts = times[ok]
            known = payout[ts-lag] >= 0
            selected = np.flatnonzero(ok)[known]
            alpha[s, selected] = ((payout[ts[known]-lag] >= threshold) != bool(direction))
            if m.cutoff >= lag and payout[m.cutoff-lag] >= 0:
                cutoff_alpha[s] = ((payout[m.cutoff-lag] >= threshold) != bool(direction))
    return cats, alpha, cutoff_alpha


def fit_size(machines, iterations=50):
    """Mixture EM learns two atoms and a prior over static and lag-rule models."""
    data = [(m, *size_assignments(m, True)) for m in machines]
    atoms = {}
    for kind in TYPES:
        counts = np.ones(4)
        for m, cats, *_ in data:
            if m.kind == kind:
                counts += np.bincount(cats, minlength=4)
        pooled = counts/counts.sum()
        a = pooled.copy(); b = pooled.copy()
        a[0] *= 1.4; b[0] *= .6
        atoms[kind] = np.array([a/a.sum(), b/b.sum()])
    priors = np.ones(len(size_states()))/len(size_states())
    family_history = []
    for iteration in range(iterations):
        expected = {k: np.full((2, 4), .02) for k in TYPES}
        state_counts = np.full(len(priors), .1)
        total_ll = 0.
        for m, cats, alpha, _ in data:
            aa = atoms[m.kind]
            like0 = aa[0, cats][None, :]
            like1 = aa[1, cats][None, :]
            like = (1-alpha)*like0+alpha*like1
            lp = np.log(like).sum(1) + np.log(priors)
            total_ll += logsumexp(lp)
            post = softmax(lp)
            state_counts += post
            r1 = alpha*like1/like
            for c in range(4):
                mask = cats == c
                expected[m.kind][1, c] += np.sum(post[:, None]*r1[:, mask])
                expected[m.kind][0, c] += np.sum(post[:, None]*(1-r1[:, mask]))
        for kind in TYPES:
            atoms[kind] = expected[kind]/expected[kind].sum(1, keepdims=True)
        priors = state_counts/state_counts.sum()
        if iteration % 10 == 9:
            family_history.append((iteration+1, total_ll, float(priors[:2].sum())))
            print('size EM', family_history[-1], flush=True)
    return {k: v.tolist() for k, v in atoms.items()}, priors.tolist(), family_history


def size_predict(machines, atoms, prior, family='all'):
    prior = np.array(prior, dtype=float).copy()
    if family == 'static':
        prior[2:] = 0; prior[:2] = .5
    elif family == 'lag':
        prior[:2] = 0
    prior /= prior.sum()
    logprior = np.where(prior > 0, np.log(np.maximum(prior, 1e-300)), -np.inf)
    result, evidence = {}, []
    for m in machines:
        cats, alpha, wanted = size_assignments(m)
        a = np.array(atoms[m.kind])
        like = (1-alpha)*a[0, cats] + alpha*a[1, cats]
        lp = np.log(like).sum(1) + logprior
        post = softmax(lp)
        big = post@wanted
        result[m.query_id] = (1-big)*a[0]+big*a[1]
        evidence.append(float(logsumexp(lp)))
    return result, evidence


class ExactModel:
    def __init__(self, params, size=25):
        global PAIRS
        PAIRS = np.array(params['pairs'], dtype=int)
        self.params = params
        self.size = size
        self.grids = {(kind, sig): level_grid(params['levels'][kind], sig, size)
                      for kind in TYPES for sig in (-1, 0, 1)}
        self.pair_prior = np.array(params['pair_prior'])

    def stable(self, m):
        z, p, lw = self.grids[m.kind, m.signal]
        lp = count_likelihood(cell_counts(m), p) + lw + np.log(self.pair_prior)[:, None, None] - np.log(2)
        # Unknown contexts contribute a marginal likelihood, never a fabricated loss.
        alpha = pair_alpha(m)
        y = m.seq[:m.cutoff]
        unknown = np.stack([np.sum((alpha < 0) & ((y > 0) == k) & (y >= 0), axis=1) for k in (0, 1)], axis=1)
        pmid = p.mean(1)
        lp += (unknown[:, 1, None]*np.log(pmid)+unknown[:, 0, None]*np.log1p(-pmid))[:, None, :]
        pred = self.pred_for(m, p, flip=False)
        return float(logsumexp(lp)), np.sum(softmax(lp)*pred), lp

    def pred_for(self, m, p, flip=False):
        out = np.empty((len(PAIRS), 2, len(p)))
        for j, (a, b) in enumerate(PAIRS):
            va, vb = m.seq[m.cutoff-a], m.seq[m.cutoff-b]
            alpha = .5 if va < 0 or vb < 0 else float((va > 0) == (vb > 0))
            if flip:
                alpha = 1-alpha
            out[j, 0] = p[:, 0]*(1-alpha)+p[:, 1]*alpha
            out[j, 1] = p[:, 0]*alpha+p[:, 1]*(1-alpha)
        return out

    def shared(self, m, cs):
        z, p, lw = self.grids[m.kind, m.signal]
        a = pair_alpha(m)
        y = m.seq[:m.cutoff]
        # Prefix counts permit exact Bernoulli likelihood for every candidate c.
        masks = np.stack([(a == value) & ((y > 0) == outcome) & (y >= 0)
                          for outcome in (0, 1) for value in (0., 1., .5)], axis=2)
        pref = np.concatenate([np.zeros((len(PAIRS), 1, 6)), np.cumsum(masks, axis=1)], axis=1)
        before = pref[:, cs].transpose(1, 0, 2).reshape(len(cs), len(PAIRS), 2, 3)
        after = pref[:, -1].reshape(1, len(PAIRS), 2, 3)-before
        counts = before + after[:, :, :, [1, 0, 2]]
        lp = count_likelihood(counts.reshape(-1, 2, 3), p).reshape(len(cs), len(PAIRS), 2, -1)
        lp += lw + np.log(self.pair_prior)[None, :, None, None]-np.log(2)
        evidence = logsumexp(lp, axis=(1, 2, 3))
        pred = self.pred_for(m, p, flip=True)
        win = np.sum(np.exp(lp-evidence[:, None, None, None])*pred, axis=(1, 2, 3))
        return evidence, win

    def drift(self, m):
        z, p, lw = self.grids[m.kind, m.signal]
        a = pair_alpha(m)
        schedule = self.params['schedule']
        u = np.clip((np.arange(m.cutoff)-(m.cutoff-schedule['anchor']))/schedule['duration'], 0, 1)
        w = np.where(a >= 0, a*(1-u)+(1-a)*u, .5)
        y = m.seq[:m.cutoff]
        lp = np.zeros((len(PAIRS), 2, len(p)))
        # Group exact identical alpha values: endpoints use counts, ramp uses BLAS.
        for d in (0, 1):
            ww = w if d == 0 else 1-w
            for value in np.unique(ww):
                ns = np.sum((ww == value) & (y >= 0), axis=1)
                wins = np.sum((ww == value) & (y > 0), axis=1)
                pp = p[:, 0]*(1-value)+p[:, 1]*value
                lp[:, d] += wins[:, None]*np.log(pp)+(ns-wins)[:, None]*np.log1p(-pp)
        lp += lw + np.log(self.pair_prior)[:, None, None]-np.log(2)
        at_cutoff = np.clip(schedule['anchor']/schedule['duration'], 0, 1)
        pred0 = self.pred_for(m, p)
        pred = pred0*(1-at_cutoff)+self.pred_for(m, p, True)*at_cutoff
        return float(logsumexp(lp)), np.sum(softmax(lp)*pred)

    def predict_win(self, machines, verbose=True):
        result, diagnostics = {}, []
        schedule = self.params['schedule']
        for eid in sorted(set(m.episode_id for m in machines)):
            ms = [m for m in machines if m.episode_id == eid]
            start = time.perf_counter()
            if ms[0].short:
                rows = [stable_masked_exact(self, m) for m in ms]
                for m, row in zip(ms, rows):
                    result[m.query_id] = float(row[1])
                diagnostics.append({'episode': eid, 'short': True, 'posterior': [1., 0., 0.]})
            else:
                stable = [self.stable(m)[:2] for m in ms]
                drift = [self.drift(m) for m in ms]
                cutoff = ms[0].cutoff
                cs = np.arange(8, cutoff)
                logc = -.5*((cs-schedule['flip_fraction']*cutoff)/schedule['flip_sd'])**2
                logc -= logsumexp(logc)
                shared = [self.shared(m, cs) for m in ms]
                jointc = logc + sum(row[0] for row in shared)
                condition_ll = [sum(row[0] for row in stable), sum(row[0] for row in drift), logsumexp(jointc)]
                condition_post = softmax(np.array(condition_ll)+np.log(np.array(self.params.get('condition_prior', [1/3]*3))))
                cp = softmax(jointc)
                for j, m in enumerate(ms):
                    result[m.query_id] = float(condition_post @ [stable[j][1], drift[j][1], cp@shared[j][1]])
                diagnostics.append({'episode': eid, 'short': False, 'posterior': condition_post.tolist(),
                                    'change_tick': float(cp@cs), 'evidence': [float(v) for v in condition_ll]})
            if verbose:
                print(f'{eid}: {time.perf_counter()-start:.2f}s {diagnostics[-1]["posterior"]}', flush=True)
        return result, diagnostics


def fit(public, iterations=12, size=25):
    global PAIRS
    machines, conditions, *_ = load_machines(public, 'train', labels=True)
    PAIRS = learn_pairs(machines)
    levels, pair_prior = fit_levels(machines, conditions, iterations, size)
    schedule = fit_schedule(machines, conditions)
    atoms, size_prior, size_history = fit_size(machines)
    return {'levels': levels, 'pair_prior': pair_prior.tolist(), 'schedule': schedule,
            'atoms': atoms, 'size_prior': size_prior, 'size_history': size_history,
            'condition_prior': [1/3]*3, 'pairs': PAIRS.tolist(),
            'pair_rule': {'min_share': MIN_SHARE, 'min_margin': MIN_MARGIN, 'min_cell': MIN_CELL}}


def masked_pair_likelihood(m, pair, p):
    """Exact finite-memory sum over every unobserved win bit for a static rule.

    The 2**max_lag states represent the generator's missing outcomes, not an
    additional latent regime. Both directions and all level nodes run together.
    Initial spins use the reconstruction's arithmetic mean of the two levels.
    """
    a, b = pair
    ngrid = len(p)
    states = np.arange(1 << b)
    agree = ((states >> (a-1)) & 1) == ((states >> (b-1)) & 1)
    live = np.r_[p[:, 1], p[:, 0]]
    dead = np.r_[p[:, 0], p[:, 1]]
    prob = np.where(agree[None, :], live[:, None], dead[:, None])
    init = np.tile(p.mean(1), 2)
    f = np.zeros((ngrid*2, 1 << b))
    f[:, 0] = 1.
    logscale = np.zeros(ngrid*2)
    # Once a trailing run spans b ticks, the past state is known exactly.
    known = 0
    stop = m.cutoff
    half = 1 << (b-1)
    for t in range(m.cutoff):
        pp = prob if t >= b else init[:, None]
        win_mass = f*pp
        loss_mass = f-win_mass
        new = np.zeros_like(f)
        observed = m.seq[t]
        if observed != 0:
            new[:, 1::2] = win_mass[:, :half]+win_mass[:, half:]
        if observed <= 0:
            new[:, ::2] = loss_mass[:, :half]+loss_mass[:, half:]
        f = new
        if observed >= 0:
            scale = f.sum(1)
            logscale += np.log(np.maximum(scale, 1e-300))
            f /= np.maximum(scale[:, None], 1e-300)
            known += 1
        else:
            known = 0
        if known >= b and np.all(m.seq[t+1:m.cutoff] >= 0):
            stop = t+1
            break
    ll = logscale + np.log(np.maximum(f.sum(1), 1e-300))
    for t in range(stop, m.cutoff):
        agree_t = (m.seq[t-a] > 0) == (m.seq[t-b] > 0)
        pp = live if agree_t else dead
        ll += np.log(pp) if m.seq[t] > 0 else np.log1p(-pp)
    return ll.reshape(2, ngrid)


def stable_masked_exact(model, m):
    z, p, lw = model.grids[m.kind, m.signal]
    lp = np.stack([masked_pair_likelihood(m, pair, p) for pair in PAIRS])
    lp += lw + np.log(model.pair_prior)[:, None, None]-np.log(2)
    pred = model.pred_for(m, p)
    return float(logsumexp(lp)), float(np.sum(softmax(lp)*pred)), lp
