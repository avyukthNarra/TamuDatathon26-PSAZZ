"""Win-head model experiments. No new structure or priors read validation truth.

The vectorized win head reproduces PairMixer v3 (6a8098c); conditional payout
sizes are its saved forecasts, independent of the win head. Only dev scores
select settings. Existing validation-derived size modes remain inherited.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[4]
HERE = ROOT / 'docs/sai'
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT/'mainFiles'))
from harness import load_split, read_pred_csv, evaluate, diff_ci, to_csv
from harness.scoring import scope_episodes

OUT = Path(__file__).resolve().parent
PAIRS = ((1, 3), (4, 7), (2, 5))
TYPES = ('steady', 'balanced', 'jackpot')


def train_structure(s):
    """Estimate pair frequencies, parity-cell means and bimodal rates on TRAIN."""
    w = s.win
    tables = np.zeros((s.n, 3, 2, 2))
    for j, (a, b) in enumerate(PAIRS):
        for t in range(max(a, b), w.shape[1]):
            good = np.isfinite(w[:, t]) & np.isfinite(w[:, t-a]) & np.isfinite(w[:, t-b])
            for cell in (0, 1):
                use = good & ((w[:, t-a] == w[:, t-b]) == cell)
                tables[use, j, cell, 0] += w[use, t]
                tables[use, j, cell, 1] += 1
    rates = (tables[..., 0] + .5) / (tables[..., 1] + 1)
    count = tables[..., 1]
    total = count.sum(2)
    mean = tables[..., 0].sum(2) / np.maximum(total, 1)
    strength = (count * (rates - mean[..., None])**2).sum(2) / np.maximum(total, 1)
    winners = strength.argmax(1)
    priors, cells, beta = {}, {}, {}
    for kind in TYPES:
        rows = np.flatnonzero(s.mtype == kind)
        freq = np.bincount(winners[rows], minlength=3) + 1.
        priors[kind] = freq / freq.sum()
        pooled = tables[rows].sum(0)
        cells[kind] = (pooled[..., 0] + .5) / (pooled[..., 1] + 1)
        best = rates[rows, winners[rows]].ravel()
        low, high = best[best < .5], best[best >= .5]
        beta[kind] = [float(np.clip(np.median(low), .01, .49)), float(np.clip(np.median(high), .51, .99))]
    # Peer checks: compare actual peers with type-stratified shuffled machines.
    episodes = [np.flatnonzero(s.episode == ep) for ep in s.episodes]
    pairs = np.array([(a, b) for rows in episodes for a in rows for b in rows if a < b])
    aa, bb = pairs.T
    rate = np.nanmean(w, axis=1)
    rate -= np.array([np.nanmean(rate[s.mtype == k]) for k in s.mtype])
    # Change location = strongest difference between pre/post 24-tick means.
    cp = np.full(s.n, np.nan)
    for i in range(s.n):
        x = w[i, :s.cutoff[i]]
        choices = []
        for t in range(max(24, len(x)-100), len(x)-24):
            l, r = x[t-24:t], x[t:t+24]
            if np.isfinite(l).sum() >= 16 and np.isfinite(r).sum() >= 16:
                choices.append((abs(np.nanmean(l)-np.nanmean(r)), t / len(x)))
        if choices:
            cp[i] = max(choices)[1]
    def metrics(index):
        a, b = index[aa], index[bb]
        valid = np.isfinite(cp[a]) & np.isfinite(cp[b])
        return np.array([(winners[a] == winners[b]).mean(), (rate[a]*rate[b]).mean(),
                         (abs(cp[a[valid]]-cp[b[valid]]) < .05).mean()])
    observed = metrics(np.arange(s.n))
    rng = np.random.RandomState(7201)
    null = []
    for _ in range(300):
        order = np.arange(s.n)
        for k in TYPES:
            rows = np.flatnonzero(s.mtype == k)
            order[rows] = rng.permutation(rows)
        null.append(metrics(order))
    null = np.array(null)
    check = {k: {'actual': float(observed[j]), 'null_mean': float(null[:, j].mean()),
                 'null_ci95': np.percentile(null[:, j], [2.5, 97.5]).tolist(),
                 'upper_tail_p': float((1+(null[:, j] >= observed[j]).sum())/301)}
             for j, k in enumerate(('pair_agreement', 'type_centered_rate_covariance', 'changepoint_alignment'))}
    return priors, cells, beta, check


def win_head(s, priors, cells, beta, prior_power=0., eta_short=16.,
             k_short=0., k_long=0., jackpot_hl=None, beta_strength=0.,
             beta_eta=16., beta_types=('jackpot',), peer_strength=0.,
             population_prior=None, eta_long=16.):
    """Batch equivalent of v3 walk's parity-cell win experts, plus requested priors."""
    w = s.win
    n = s.n
    nobs = np.isfinite(w).sum(1)
    hls = np.tile(np.array([12., 40.]), (n, 3, 1))
    if jackpot_hl is not None:
        hls[s.mtype == 'jackpot'] = jackpot_hl
    decay = 2.**(-1./hls)
    base_decay = decay[:, 0, 0]
    k = np.where(nobs == 64, k_short, k_long)[:, None, None]
    eta = np.where(nobs == 64, eta_short, eta_long)
    beta_mask = np.isin(s.mtype, beta_types) if beta_strength else np.zeros(n, bool)
    if population_prior is None:
        eta[beta_mask] = beta_eta
    else:
        beta_mask = np.ones(n,bool)
    pprior = np.array([priors[t] for t in s.mtype])
    means = np.array([cells[t] for t in s.mtype])
    if peer_strength:
        # Train showed type-centered rates covary within episodes. Center peers
        # by TRAIN type rates; shrink noisy episode residuals by 32 observations.
        type_rates = {kind: float(cells[kind].mean()) for kind in TYPES}
        observed_rate = np.nanmean(w,axis=1)
        for i in np.flatnonzero(nobs == 64):
            peers = s.peers(i)
            residual = np.mean([observed_rate[j]-type_rates[s.mtype[j]] for j in peers])
            weight = sum(nobs[j] for j in peers)/(sum(nobs[j] for j in peers)+32.)
            means[i] = np.clip(means[i]+weight*residual,.01,.99)
        k = k + np.where(nobs == 64,peer_strength,0.)[:,None,None]
    logw = np.broadcast_to(prior_power*np.log(pprior)[:, :, None], (n, 3, 2)).copy()
    counts = np.zeros((n, 3, 2, 2))
    wins = np.zeros_like(counts)
    rate = np.zeros((n, 2))
    result = np.zeros(n)
    ii, jj, kk = np.indices((n, 3, 2))
    for t in range(int(s.cutoff.max())+1):
        base = (rate[:, 0]+1)/(rate[:, 1]+2)
        valid = np.zeros((n, 3), bool)
        agree = np.zeros((n, 3), int)
        for j, (a, b) in enumerate(PAIRS):
            if t >= max(a, b):
                valid[:, j] = np.isfinite(w[:, t-a]) & np.isfinite(w[:, t-b])
                agree[:, j] = w[:, t-a] == w[:, t-b]
        cell = np.broadcast_to(agree[:, :, None], (n, 3, 2))
        cw, cn = wins[ii, jj, kk, cell], counts[ii, jj, kk, cell]
        cm = np.take_along_axis(means, agree[..., None], axis=2)[..., 0, None]
        pred = (cw + .25*base[:, None, None] + k*cm)/(cn+.25+k)
        if beta_mask.any():
            if population_prior is None:
                br = np.array([beta[kind] for kind in s.mtype[beta_mask]])
                bm = np.column_stack((br, np.full(len(br), .5)))
                strengths = np.array([beta_strength, beta_strength, 2.])
                prior_weights = np.array([.45,.45,.1])
            else:
                prior_weights = np.array([p[0] for p in population_prior])
                bm = np.broadcast_to([p[1] for p in population_prior],(n,len(population_prior)))
                strengths = np.array([p[2] for p in population_prior])
            a = bm[:, None, None, :]*strengths
            b = (1-bm[:, None, None, :])*strengths
            # M2/M3 add train-learned pseudo-observations inside each v4
            # Beta posterior, before both component means and weights update.
            ww = (cw+k*cm)[beta_mask,...,None]
            nn = (cn+k)[beta_mask,...,None]
            logs = (np.log(prior_weights)+gammaln(a+ww)+gammaln(b+nn-ww)-gammaln(a+b+nn)
                    -gammaln(a)-gammaln(b)+gammaln(a+b))
            post = np.exp(logs-logs.max(-1, keepdims=True))
            pred[beta_mask] = (post*((a+ww)/(a+b+nn))).sum(-1)/post.sum(-1)
        pred = np.where(valid[..., None], pred, base[:, None, None])
        weights = np.exp(logw-logw.max((1,2), keepdims=True))
        pw = (weights*pred).sum((1,2))/weights.sum((1,2))
        emit = s.cutoff == t
        result[emit] = pw[emit]
        if t == w.shape[1]:
            break
        obs = np.isfinite(w[:, t])
        value = np.nan_to_num(w[:, t])
        logw -= eta[:, None, None]*(pred-value[:,None,None])**2*obs[:,None,None]
        wins *= decay[..., None]
        counts *= decay[..., None]
        rate *= base_decay[:, None]
        rate[:,0] += value
        rate[:,1] += obs
        add = obs[:, None, None]*valid[..., None]
        wins[ii,jj,kk,cell] += add*value[:,None,None]
        counts[ii,jj,kk,cell] += add
    return result


def main():
    started = time.time()
    train, v = load_split('train'), load_split('validation')
    priors, cells, beta, peers = train_structure(train)
    structure = {'pair_order': PAIRS, 'type_pair_prior': {k:a.tolist() for k,a in priors.items()},
                 'type_cell_mean': {k:a.tolist() for k,a in cells.items()},
                 'train_beta_levels': beta, 'train_peer_checks': peers}
    (OUT/'train_structure.json').write_text(json.dumps(structure, indent=2)+'\n')
    print('TRAIN', json.dumps(structure), flush=True)
    mixer = read_pred_csv(HERE/'preds/TM_MIXER3_validation.csv', v)
    rule = read_pred_csv(HERE/'preds/TM_RULE_validation.csv', v)
    ref = read_pred_csv(HERE/'preds/TM_COMBINED3_validation.csv', v)
    # Recover unrounded, unclipped conditional sizes from the ORIGINAL v3 walk.
    # This baseline-only step retains its documented validation-target dependency.
    raw_path = OUT/'v3_raw.npy'
    if raw_path.exists():
        raw = np.load(raw_path)
    else:
        from pranav_v3 import PairMixer, size_modes_from_targets, CLASS_INDEX
        from slot_forecasting.io import load_inputs, read_targets
        tq,th = load_inputs(ROOT/'mainFiles/public/train')
        vq,_ = load_inputs(ROOT/'mainFiles/public/validation')
        modes = size_modes_from_targets(read_targets(ROOT/'mainFiles/public/validation/targets.csv'),vq)
        model = PairMixer(size_modes=modes).fit(tq,th)
        raw = np.zeros((v.n,5))
        for i in range(v.n):
            own = [None if not np.isfinite(x) else CLASS_INDEX[int(x)] for x in v.grid[i,:v.cutoff[i]]]
            model.walk(own,str(v.mtype[i]),int(v.cutoff[i]),lambda t,p,i=i: raw.__setitem__(i,p) if t == v.cutoff[i] else None)
        np.save(raw_path,raw)
    sizes = raw[:,1:]/raw[:,1:].sum(1,keepdims=True)
    def predict(**settings):
        pw = win_head(v, priors, cells, beta, **settings)
        p = np.column_stack((1-pw, pw[:,None]*sizes))
        p = np.maximum(p, 1e-4)
        p /= p.sum(1,keepdims=True)
        return p
    p = predict()
    difference = float(abs(p-mixer).max())
    print('BASELINE_REPRODUCTION', difference, flush=True)
    if difference > 1e-10:
        raise AssertionError('win-head must reproduce v3 before experiments')
    dev_eps = scope_episodes(v, 'dev')
    # Two fixed, condition-balanced dev folds. No audit truth is selected.
    folds = [[], []]
    for c in sorted(set(v.condition)):
        eps = [e for e in dev_eps if v.condition[v.episode_rows(e)[0]] == c]
        for f in (0,1):
            folds[f] += [i for e in eps[f::2] for i in v.episode_rows(e)]
    folds = [np.array(sorted(f)) for f in folds]
    def score(pp):
        return evaluate(pp,v,'dev',bootstrap=0)['points']
    def fold_score(pp, f):
        rows = folds[f]
        return evaluate(pp[rows],v.subset(rows),'dev',bootstrap=0)['points']
    grid, results = [], {}
    def experiment(mid, settings):
        candidates = []
        for params in settings:
            pp = predict(**params)
            for rw in (.1,.2,.3):
                combined = rw*rule+(1-rw)*pp
                row = {'id':mid, 'params':params, 'rule_weight':rw, 'dev':score(combined),
                       'fold0':fold_score(combined,0), 'fold1':fold_score(combined,1)}
                candidates.append((row,combined))
            print('GRID',mid, params, candidates[-2][0]['dev'], flush=True)
        grid.extend(r for r,_ in candidates)
        best, pp = max(candidates,key=lambda pair:pair[0]['dev'])
        to_csv(HERE/f'preds/{mid}_validation.csv', v, pp)
        results[mid] = {**best, 'evaluation':evaluate(pp,v,'dev'), 'difference':diff_ci(pp,ref,v,'dev')}
        # Fold-specific selection and honest cross-fit dev scores limit tuning optimism.
        oof = ref.copy()
        choices = []
        for f in (0,1):
            selected, pred = max(candidates,key=lambda pair:pair[0][f'fold{1-f}'])
            oof[folds[f]] = pred[folds[f]]
            choices.append(selected)
        results[mid]['crossfit'] = {'evaluation': evaluate(oof,v,'dev'),
                                   'difference':diff_ci(oof,ref,v,'dev'), 'choices':choices}
        (OUT/'results.json').write_text(json.dumps({'reference':evaluate(ref,v,'dev'),
            'baseline_max_abs_diff':difference, 'experiments':results, 'grid':grid,
            'seconds':time.time()-started},indent=2)+'\n')
        print('RESULT',mid,json.dumps(results[mid]),flush=True)
    experiment('IMP_M1', [dict(prior_power=a,eta_short=e) for a in (0.,1.,4.) for e in (4.,8.,16.)])
    experiment('IMP_M2', [dict(k_short=k,k_long=l) for k in (0.,.25,1.,4.) for l in (0.,.25)])
    # M3: train establishes rate covariance, so pool a small peer residual into short-history priors.
    experiment('IMP_M3', [dict(peer_strength=k) for k in (0.,.25,1.,4.)])
    experiment('IMP_M4', [dict(jackpot_hl=h) for h in ((6.,24.),(12.,80.),(24.,80.))]
               +[dict(beta_strength=k,beta_eta=e) for k in (2.,6.,16.) for e in (8.,16.)])
    # Combined model-level changes, selected exclusively on dev; fixed .2 rule is included.
    m1 = results['IMP_M1']['params']; m2 = results['IMP_M2']['params']; m4 = results['IMP_M4']['params']
    experiment('IMP_M5',[{**m1,**m2,**m4}])
    # M5 parameters above were chosen on full dev: reselect EVERY component's
    # parameters on the opposite fold before describing a cross-fit result.
    oof = ref.copy()
    selections = []
    for f in (0,1):
        params = {}
        for mid in ('IMP_M1','IMP_M2','IMP_M4'):
            selected = max((r for r in grid if r['id'] == mid), key=lambda r:r[f'fold{1-f}'])
            params.update(selected['params'])
        pp = predict(**params)
        weight = max((.1,.2,.3),key=lambda rw:fold_score(rw*rule+(1-rw)*pp,1-f))
        combined = weight*rule+(1-weight)*pp
        oof[folds[f]] = combined[folds[f]]
        selections.append({'test_fold':f,'params':params,'rule_weight':weight})
    results['IMP_M5']['crossfit'] = {'evaluation':evaluate(oof,v,'dev'),
        'difference':diff_ci(oof,ref,v,'dev'),'choices':selections}
    peer = read_pred_csv(HERE/'preds/IMP_M3_validation.csv',v)
    pk = results['IMP_M3']['params']['peer_strength']
    controlled = .2*rule+.8*predict(k_short=pk)
    results['IMP_M3']['peer_increment_vs_matched_type_shrink'] = diff_ci(peer,controlled,v,'dev')
    rows = np.concatenate([v.episode_rows(e) for e in dev_eps])
    jackpot = {}
    for kind in TYPES:
        ids = rows[v.mtype[rows] == kind]
        improved = read_pred_csv(HERE/'preds/IMP_M4_validation.csv',v)
        jackpot[kind] = {'train_win_rate':float(np.nanmean(train.win[train.mtype == kind])),
            'baseline_p0_mse':float(((ref[ids,0]-v.truth[ids,0])**2).mean()),
            'M4_p0_mse':float(((improved[ids,0]-v.truth[ids,0])**2).mean())}
    (OUT/'results.json').write_text(json.dumps({'reference':evaluate(ref,v,'dev'),
        'baseline_max_abs_diff':difference,'experiments':results,'grid':grid,
        'type_diagnosis':jackpot,'seconds':time.time()-started},indent=2)+'\n')
    print('FINISHED', time.time()-started, flush=True)

if __name__ == '__main__':
    main()
