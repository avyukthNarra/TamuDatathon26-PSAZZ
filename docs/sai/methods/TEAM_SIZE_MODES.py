"""Severity-only causal mode mixture, with frozen TM_COMBINED win probability.
Mode centers are learned from outer-training DEV conditional q only. Candidate
severity lags are TRAIN-screened over 1..8 against null lags9..16 (A9).
Predeclared 12 cells: HL20/80/infinity x temperature .25/1/4 at eta8,
plus the same three HL at temperature1/eta0. No win model is refitted.
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[1] / 'mainFiles'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'methods'))
from harness import load_split, evaluate, diff_ci, blend_gain, read_pred_csv, smoke_split
from harness.scoring import scope_episodes
from harness.results import to_csv
from A9 import learned_lags

CELLS = [dict(hl=h, temp=t, eta=8.) for h in (20., 80., 1e9) for t in (.25, 1., 4.)]
CELLS += [dict(hl=h, temp=1., eta=0.) for h in (20., 80., 1e9)]
PAY = (0, 1, 2, 5, 20)


def learn_modes(v, rows):
    """Deterministic two-center fit; accesses truth only at supplied training rows."""
    out = {}
    for ty in sorted(set(v.mtype[rows])):
        ids = np.asarray([i for i in rows if v.mtype[i] == ty])
        x = v.truth[ids, 1:].copy()
        x /= x.sum(axis=1, keepdims=True)
        # Farthest-point seeds and deterministic Lloyd updates; no supplied constants.
        a = x[np.lexsort(x.T[::-1])[0]]
        b = x[np.argmax(((x-a)**2).sum(1))]
        centers = np.stack([a, b])
        for _ in range(20):
            group = ((x[:, None] - centers)**2).sum(2).argmin(1)
            nxt = np.stack([x[group == j].mean(0) if (group == j).any() else centers[j] for j in range(2)])
            if np.max(abs(nxt-centers)) < 1e-14:
                break
            centers = nxt
        # Quantize learned parameters for equivalent-fold cache identity, not fixed modes.
        centers = np.round(nxt[np.argsort(nxt @ np.asarray(PAY[1:]))], 12)
        centers /= centers.sum(axis=1, keepdims=True)
        out[str(ty)] = centers
    return out


def size_walk(grid, cutoff, modes, lags, cells=CELLS):
    """Vectorize settings; exact-class cells. Forecast before absorbing target tick."""
    c = int(cutoff)
    own = np.full(c, -1, dtype=int)
    for k, pay in enumerate(PAY):
        own[np.flatnonzero(grid[:c] == pay)] = k
    keys = (0,) + tuple(lags)  # 0 is the always-available machine-level expert.
    ncfg, nk = len(cells), len(keys)
    counts = np.zeros((ncfg, nk, 5, 4))
    machine = np.zeros((ncfg, 4))
    logw = np.zeros((ncfg, nk))
    decay = np.exp2(-1 / np.array([x['hl'] for x in cells]))
    temp = np.array([x['temp'] for x in cells])
    eta = np.array([x['eta'] for x in cells])
    llr = np.log(np.maximum(modes[1], 1e-3) / np.maximum(modes[0], 1e-3))
    for t in range(c+1):
        cell = np.array([0 if lag == 0 else own[t-lag] if t >= lag else -1 for lag in keys])
        valid = cell >= 0
        cc = counts[:, np.arange(nk), np.maximum(cell, 0), :].copy()
        # Vendor fallback uses machine-size evidence if a cell is absent, not zero evidence.
        empty = (cc.sum(2) == 0) | (~valid)[None, :]
        cc = np.where(empty[:, :, None], machine[:, None, :], cc)
        z = np.clip(temp[:, None] * (cc @ llr), -50, 50)
        prob = 1/(1+np.exp(-z))
        pred = modes[0] + prob[:, :, None] * (modes[1]-modes[0])
        w = np.exp(logw-logw.max(1, keepdims=True))
        share = (w[:, :, None]*pred).sum(1)/w.sum(1)[:, None]
        if t == c:
            return share
        value = own[t]
        if value > 0:
            target = np.eye(4)[value-1]
            logw -= eta[:, None] * ((pred-target)**2).sum(2)
        # Fixed calendar-tick forgetting also across unobserved ticks, matching vendor.
        counts *= decay[:, None, None, None]
        machine *= decay[:, None]
        if value > 0:
            machine[:, value-1] += 1
            counts[:, np.flatnonzero(valid), cell[valid], value-1] += 1
    raise AssertionError('unreachable')


def predict_sizes(v, modes, lags, cells=CELLS, use_cache=True):
    blob = hashlib.sha256()
    blob.update(Path(__file__).read_bytes())
    for a in (v.qid, v.mtype, v.cutoff, v.grid):
        blob.update(np.ascontiguousarray(a).tobytes())
    blob.update(json.dumps({k:a.tolist() for k,a in modes.items()},sort_keys=True).encode())
    blob.update(repr((tuple(lags), cells)).encode())
    path = HERE / 'cache' / ('team_size_modes_'+blob.hexdigest()[:24]+'.npz')
    if use_cache and path.exists():
        return np.load(path)['shares']
    shares = np.empty((len(cells), v.n, 4))
    for i in range(v.n):
        shares[:, i] = size_walk(v.grid[i], v.cutoff[i], modes[str(v.mtype[i])], lags, cells)
    if use_cache:
        np.savez(path, shares=shares)
    return shares


def folds(v, dev):
    out = {}
    for cond in sorted(set(v.condition)):
        eps = [e for e in dev if v.condition[v.episode_rows(e)[0]] == cond]
        eps.sort(key=lambda e: hashlib.sha256(('team-outer-v1|'+str(e)).encode()).hexdigest())
        out.update({e:i%6 for i,e in enumerate(eps)})
    return out


def compose(base, shares):
    return np.column_stack([base[:,0], (1-base[:,0,None])*shares])


def oof(v, base, dev, lags, cells=CELLS, use_cache=True):
    fm = folds(v, dev)
    result = np.tile(base, (len(cells),1,1))
    centers = []
    memory_cache = {}
    for f in range(6):
        tr = np.concatenate([v.episode_rows(e) for e in dev if fm[e] != f])
        te = np.concatenate([v.episode_rows(e) for e in dev if fm[e] == f])
        modes = learn_modes(v, tr)
        centers.append({k:a.tolist() for k,a in modes.items()})
        # Cache entire validation batch only when learned parameters coincide, no audit q used.
        mk = json.dumps({k:a.tolist() for k,a in modes.items()},sort_keys=True)
        if mk not in memory_cache:
            memory_cache[mk] = predict_sizes(v, modes, lags, cells, use_cache=use_cache)
        shares = memory_cache[mk]
        for j in range(len(cells)):
            result[j,te] = compose(base[te],shares[j,te])
    return result, centers, fm


def estimate_modes(v, trainrows):
    return learn_modes(v, trainrows)


def predict_shapes(split, learnedmodes, params):
    # Caller supplies frozen TRAIN-screened lags for a self-contained outer fold.
    cfg = {k: params[k] for k in ('hl', 'temp', 'eta')}
    return predict_sizes(split, learnedmodes, tuple(params['lags']), [cfg])[0]


def main():
    started = time.perf_counter()
    v,t = load_split('validation'),load_split('train')
    base = read_pred_csv(HERE/'preds/TM_COMBINED_validation.csv',v)
    lags = learned_lags(t,1.5,.3)
    if not lags:
        raise ValueError('TRAIN severity screen returned no lags')
    if '--smoke' in sys.argv:
        sm=smoke_split(v)
        pos={q:i for i,q in enumerate(v.qid)}
        b=base[[pos[q] for q in sm.qid]]
        modes=learn_modes(v,np.concatenate([v.episode_rows(e) for e in scope_episodes(v,'dev')]))
        sh=predict_sizes(sm,modes,lags,cells=CELLS[:1],use_cache=False)
        # Compare severity-only rewrite with the pinned mixer on one real machine.
        from _team_mixer_source import PairMixer
        row=0; c=int(sm.cutoff[row]); own=[None if np.isnan(x) else PAY.index(int(x)) for x in sm.grid[row,:c]]
        model=PairMixer(pairs=((1,2),),size_modes={k:tuple(map(tuple,a)) for k,a in modes.items()},size_keys=((),)+tuple((l,) for l in lags),size_half_life=CELLS[0]['hl'],mode_temp=CELLS[0]['temp'],size_eta=CELLS[0]['eta'])
        got=[]; model.walk(own,str(sm.mtype[row]),c,lambda tt,p:got.append(p) if tt==c else None)
        vendor=np.array(got[0][1:]);vendor/=vendor.sum()
        delta=float(abs(vendor-sh[0,row]).max());assert delta<1e-10,delta
        # Terminal/future values must be irrelevant.
        g=sm.grid[row].copy();g[c:]=20
        assert np.array_equal(size_walk(g,c,modes[str(sm.mtype[row])],lags,CELLS[:1]),sh[:,row])
        print(json.dumps({'smoke_seconds':time.perf_counter()-started,'episodes':len(sm.episodes),'queries':sm.n,'train_severity_lags':lags,'vendor_max_diff':delta,'finite':bool(np.isfinite(sh).all()),'estimated_full_12cell_seconds':(time.perf_counter()-started)*12*4}),flush=True)
        return
    dev=scope_episodes(v,'dev')
    pp,centers,fm=oof(v,base,dev,lags)
    scores=[evaluate(p,v,'dev',bootstrap=0) for p in pp]
    best=int(np.argmax([e['points'] for e in scores])); cfg=CELLS[best]
    grid=[dict(params=c,points=e['points'],condition_ratios=e['condition_ratios']) for c,e in zip(CELLS,scores)]
    (HERE/'grids/TEAM_SIZE_MODES.json').write_text(json.dumps(grid,indent=1)+'\n')
    # Actual selected-config OOF recomputation bypasses disk cache.
    repeat,_,_=oof(v,base,dev,lags,[cfg],use_cache=False)
    det=float(abs(repeat[0]-pp[best]).max());assert det<1e-12,det
    modes=learn_modes(v,np.concatenate([v.episode_rows(e) for e in dev]))
    finalshare=predict_sizes(v,modes,lags,[cfg])[0]
    final=compose(base,finalshare)
    devrows=np.concatenate([v.episode_rows(e) for e in dev])
    final[devrows]=pp[best,devrows]  # canonical DEV OOF / audit inference hybrid.
    assert np.isfinite(final).all() and np.min(final)>=0 and np.max(abs(final.sum(1)-1))<1e-10
    to_csv(HERE/'preds/TEAM_SIZE_MODES_validation.csv',v,final)
    ev=evaluate(final,v,'dev',bootstrap=1000)
    d=diff_ci(final,base,v,'dev')
    bg=blend_gain(final,base,v)
    doc=dict(id='TEAM_SIZE_MODES',params=cfg,train_severity_lags=lags,n_grid_cells=len(CELLS),**ev,
        vs_TM_COMBINED=d,blend_gain_vs_TM_COMBINED={k:bg[k] for k in ('points','ci95','vs_base','w_mean')},
        runtime_seconds=time.perf_counter()-started,determinism_max_diff=det,folds=fm,fold_centers=centers,
        inference_centers={k:a.tolist() for k,a in modes.items()},
        caveat='Centers fit outer-training DEV only; hyperparameter globally selected on OOF DEV (not nested), so selected score and blend gain retain selection optimism. Audit predictions inferred with full DEV centers and never scored.',
        provenance={'base':'TM_COMBINED cached win frozen','algorithm':'severity walk matched pinned PairMixer 6a8098c','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'vendor_sha256':hashlib.sha256((HERE/'methods/_team_mixer_source.py').read_bytes()).hexdigest()})
    (HERE/'results/TEAM_SIZE_MODES.json').write_text(json.dumps(doc,indent=1)+'\n')
    lines=['# TEAM_SIZE_MODES',f"12 predeclared cells; TRAIN severity lags {lags}; six condition-stratified episode folds.",f"Best {cfg}; DEV OOF {ev['points']:.3f}, CI {ev['ci95']}.",f"vs TM_COMBINED {d}; blend gain {doc['blend_gain_vs_TM_COMBINED']}.",f"Runtime {doc['runtime_seconds']:.1f}s; actual selected OOF repeat max diff {det}.",'Two mode centers per type learned only from each outer-training DEV fold; no audit truth fitting.', 'Win probability frozen from TM_COMBINED; severity inference uses only observed payouts strictly before cutoff.', 'Centers rounded to 12 digits for learned-parameter cache identity; no hardcoded modes or selected lags.', 'Canonical CSV contains DEV OOF predictions and unscored audit inference from full-DEV centers.',doc['caveat'],'No overall verdict.']
    (HERE/'reports/TEAM_SIZE_MODES.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:doc[k] for k in ('points','ci95','params','vs_TM_COMBINED','runtime_seconds','determinism_max_diff')}),flush=True)

if __name__=='__main__':
    main()
