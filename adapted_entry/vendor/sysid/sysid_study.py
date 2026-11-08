"""Independent simulator identification; only fold A targets enter diagnostics.

Run from the repository root: python3 mainFiles/sysid_study.py
Artifacts are kept under the ignored mainFiles/runs/sysid directory.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '2')
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import expit, logit
from scipy.linalg import solve
from slot_forecasting.io import load_inputs, read_labels, read_targets, write_json

ROOT = Path(__file__).resolve().parent
PAIRS = tuple(combinations(range(1, 9), 2))
NAMES = ([f'w{j}' for j in range(1, 9)] +
         [f'w{i}*w{j}' for i, j in PAIRS] +
         [f'l{j}' for j in range(1, 9)] +
         [f'l{i}*l{j}' for i, j in PAIRS] +
         [f'{p}{j}' for p in ('peer', 'prev', 'next') for j in range(1, 9)])


@dataclass
class Machine:
    query: object
    ticks: np.ndarray
    payouts: np.ndarray
    features: np.ndarray

    def rows(self, include_label=False, labels=None):
        ticks = self.ticks[self.ticks >= 8]
        if include_label:
            ticks = np.r_[ticks, self.query.cutoff]
        values = self.payouts.copy()
        if include_label:
            values[-1] = labels[self.query.query_id]
        return self.features[ticks], values[ticks], ticks


def load_machines(public, split):
    queries, history = load_inputs(Path(public) / split)
    by_episode = defaultdict(list)
    rows = defaultdict(list)
    for q in queries:
        by_episode[q.episode_id].append(q)
    for r in history:
        rows[r.episode_id, r.machine_id].append((r.tick, r.payout))
    machines = []
    for episode in sorted(by_episode):
        qs = sorted(by_episode[episode], key=lambda q: q.machine_id)
        cutoff = qs[0].cutoff
        a = np.full((len(qs), cutoff + 1), np.nan)
        for i, q in enumerate(qs):
            for tick, payout in rows[episode, q.machine_id]:
                a[i, tick] = payout
        w = np.where(np.isnan(a), np.nan, 2. * (a > 0) - 1.)
        large = np.where(np.isnan(a), np.nan, 2. * (a > 1) - 1.)
        for i, q in enumerate(qs):
            lag_w = np.full((cutoff + 1, 8), np.nan)
            lag_l = lag_w.copy()
            peer = lag_w.copy()
            prev = lag_w.copy()
            nxt = lag_w.copy()
            others = np.delete(w, i, axis=0)
            count = np.sum(np.isfinite(others), axis=0)
            mean = np.nansum(others, axis=0) / np.maximum(count, 1)
            mean[count == 0] = np.nan
            for lag in range(1, 9):
                lag_w[lag:, lag-1] = w[i, :-lag]
                lag_l[lag:, lag-1] = large[i, :-lag]
                peer[lag:, lag-1] = mean[:-lag]
                prev[lag:, lag-1] = w[(i-1) % len(qs), :-lag]
                nxt[lag:, lag-1] = w[(i+1) % len(qs), :-lag]
            F = np.column_stack([lag_w] +
                 [lag_w[:, x-1] * lag_w[:, y-1] for x, y in PAIRS] + [lag_l] +
                 [lag_l[:, x-1] * lag_l[:, y-1] for x, y in PAIRS] + [peer, prev, nxt])
            machines.append(Machine(q, np.flatnonzero(np.isfinite(a[i])), a[i], F))
    return machines, queries, history


def folds(public, queries):
    with (Path(public) / 'validation/conditions.csv').open() as f:
        c = {r['query_id']: r['condition'] for r in csv.DictReader(f)}
    ef = {}
    for condition in sorted(set(c.values())):
        episodes = sorted({q.episode_id for q in queries if c[q.query_id] == condition})
        ef.update({ep: i % 2 for i, ep in enumerate(episodes)})
    return c, np.array([ef[q.episode_id] for q in queries])


def fit_logistic(X, y, precision=1., weights=None, prior=None, maxiter=35):
    """Gaussian-prior logistic MAP and Laplace precision (intercept included)."""
    X = np.column_stack([np.ones(len(X)), X])
    d = X.shape[1]
    weights = np.ones(len(y)) if weights is None else weights
    penalty = np.broadcast_to(np.asarray(precision), (d,)).copy()
    penalty[0] = .25
    prior = np.zeros(d) if prior is None else prior
    b = prior.copy()
    for iteration in range(maxiter):
        z = X @ b
        p = expit(z)
        H = X.T @ ((weights * p * (1-p))[:, None] * X) + np.diag(penalty)
        gradient = X.T @ (weights * (p-y)) + penalty * (b-prior)
        step = solve(H, gradient, assume_a='pos', check_finite=False)
        old = np.sum(weights * (np.logaddexp(0, z)-y*z)) + .5*np.sum(penalty*(b-prior)**2)
        rate = 1.
        for _ in range(16):
            bb = b-rate*step
            zz = X@bb
            new = np.sum(weights * (np.logaddexp(0,zz)-y*zz)) + .5*np.sum(penalty*(bb-prior)**2)
            if new <= old+1e-8:
                break
            rate *= .5
        b = bb
        if np.max(np.abs(rate*step)) < 1e-6:
            break
    return b, H


def screening(machines, labels, output):
    """Random-effect association screen, plus local chronological likelihood checks."""
    numerator = np.zeros(len(NAMES))
    count = np.zeros(len(NAMES))
    by_machine = []
    for m in machines:
        F, payout, ticks = m.rows(True, labels)
        y = (payout > 0).astype(float)
        score = []
        for j in range(len(NAMES)):
            ok = np.isfinite(F[:,j])
            x, yy = F[ok,j], y[ok]
            x = x-x.mean() if len(x) else x
            yy = yy-yy.mean() if len(yy) else yy
            sxx = x@x
            z2 = (x@yy)**2 / max(sxx * (yy@yy)/max(len(yy),1), 1e-9)
            numerator[j] += z2-1
            count[j] += 1
            score.append(float(z2))
        by_machine.append(score)
    order = np.argsort(-numerator/count)
    result = [{'feature':NAMES[j], 'excess_z2': float(numerator[j]/count[j])} for j in order]
    write_json(output/'screening.json', result)
    return result


def target_structure(public, machines, queries, output):
    _, fold = folds(public, queries)
    targets = {t.query_id: t.probabilities for t in read_targets(Path(public)/'validation/targets.csv')}
    qa = [q for q, f in zip(queries,fold) if f == 0]
    result = {'n_queries_A':len(qa), 'types':{}}
    for typ in sorted({q.machine_type for q in qa}):
        subset = [q for q in qa if q.machine_type == typ]
        a = np.array([targets[q.query_id] for q in subset])
        win = 1-a[:,0]
        sizes = a[:,1:] / win[:,None]
        modes, counts = np.unique(np.round(sizes,9),axis=0,return_counts=True)
        result['types'][typ] = {'n':len(win), 'win_quantiles':np.quantile(win,[0,.1,.25,.5,.75,.9,1]).tolist(),
            'size_modes':modes.tolist(), 'size_mode_counts':counts.tolist(),
            'max_size_rounding_error':float(np.max(np.abs(sizes-np.round(sizes,9))))}
    # Global (shared coefficient) regression is a diagnostic, not a deployable model.
    # Per-machine random coefficients are considered in subsequent history analysis.
    lookup = {m.query.query_id:m for m in machines}
    F = np.array([lookup[q.query_id].features[q.cutoff] for q in qa])
    base = np.column_stack([np.ones(len(qa)), [q.signal for q in qa],
         *[[float(q.machine_type == t) for q in qa] for t in ('balanced','jackpot')]])
    z = logit(np.array([1-targets[q.query_id][0] for q in qa]))
    result['global_logit_fits'] = {}
    for name, inds in [('type_signal',[]),('own_win_lags',list(range(8))),
            ('own_win_lags_pairs',list(range(36))),('all_lags_pairs_peers',list(range(len(NAMES))))]:
        X = np.column_stack([base, np.nan_to_num(F[:,inds])])
        b = np.linalg.lstsq(X,z,rcond=None)[0]
        r = z-X@b
        result['global_logit_fits'][name] = {'rmse':float(np.sqrt(np.mean(r*r))),
            'max_abs':float(np.max(np.abs(r))), 'r2':float(1-np.mean(r*r)/np.var(z)),
            'coefficient_names':['intercept','signal','balanced','jackpot']+[NAMES[i] for i in inds],
            'coefficients':b.tolist()}
    write_json(output/'structure_A.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=ROOT/'public')
    parser.add_argument('--out',type=Path,default=ROOT/'runs/sysid')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    start=time.time()
    train,tq,th=load_machines(args.data,'train')
    labels=read_labels(args.data/'train/labels.csv',tq)
    screen=screening(train,labels,args.out)
    val,vq,vh=load_machines(args.data,'validation')
    structure=target_structure(args.data,val,vq,args.out)
    print(json.dumps({'runtime':time.time()-start,'top_screen':screen[:20],
                      'target_structure':structure},indent=2),flush=True)


if __name__ == '__main__':
    main()
