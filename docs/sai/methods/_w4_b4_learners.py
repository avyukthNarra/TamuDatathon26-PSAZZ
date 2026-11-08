"""B4 residual learners. Importing this module never loads data or runs fits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import (CONDITIONS, PAY, blend_gain, check_determinism, diff_ci,
                     evaluate, smoke_split, to_csv, write_result)
from harness.scoring import scope_episodes

SEED = 20261107
STRENGTHS = [0, .1, .25, .5, 1]
OPTIONAL = '/private/tmp/house-edge-w3w4-optional'


def model_grid(mid):
    if mid == 'S1_B4':
        return [dict(family='hist', iterations=i, leaves=l, l2=r)
                for i, l, r in [(20, 7, 10), (20, 15, 100),
                                (60, 7, 100), (20, 3, 100)]]
    if mid == 'S2_B4':
        return ([dict(family='cat', iterations=i, depth=d, l2=10)
                 for i in (25, 75) for d in (3, 5)] +
                [dict(family='light', iterations=i, leaves=l, l2=10)
                 for i in (25, 75) for l in (7, 15)])
    return [dict(family='ridge', alpha=a) for a in (10, 100, 1000, 10000)]


def fit_predict(X, Y, Z, params):
    """One thread, fixed seed, five residual outputs (not classification)."""
    family = params['family']
    if family == 'ridge':
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        return make_pipeline(StandardScaler(), Ridge(alpha=params['alpha'])).fit(X, Y).predict(Z)
    if family == 'cat':
        if OPTIONAL not in sys.path:
            sys.path.insert(0, OPTIONAL)
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(iterations=params['iterations'], depth=params['depth'],
                                  l2_leaf_reg=params['l2'], loss_function='MultiRMSE',
                                  random_seed=SEED, thread_count=1,
                                  allow_writing_files=False, verbose=False)
        return model.fit(X, Y).predict(Z)
    if family == 'hist':
        from sklearn.ensemble import HistGradientBoostingRegressor
        def factory():
            return HistGradientBoostingRegressor(max_iter=params['iterations'],
                max_leaf_nodes=params['leaves'], l2_regularization=params['l2'],
                min_samples_leaf=32, early_stopping=False, random_state=SEED)
    else:
        if OPTIONAL not in sys.path:
            sys.path.insert(0, OPTIONAL)
        from lightgbm import LGBMRegressor
        def factory():
            return LGBMRegressor(n_estimators=params['iterations'],
                num_leaves=params['leaves'], reg_lambda=params['l2'], n_jobs=1,
                random_state=SEED, verbosity=-1, deterministic=True, force_col_wise=True)
    return np.column_stack([factory().fit(X, Y[:, j]).predict(Z) for j in range(5)])


def normalized(base, residual, strength):
    P = np.clip(base + strength * residual, 1e-12, 1)
    return P / P.sum(1, keepdims=True)


def folds_for(v):
    eps = scope_episodes(v, 'dev')
    ordered = sorted(eps, key=lambda e: hashlib.sha256(f'fold-v1|{e}'.encode()).hexdigest())
    mapping = {e: i % 6 for i, e in enumerate(ordered)}
    return np.array([mapping.get(e, -1) for e in v.episode])


def sliced(block, rows):
    n = len(block['base'])
    return {k: (a[rows] if isinstance(a, np.ndarray) and a.ndim and len(a) == n else a)
            for k, a in block.items()}


def residuals(mid, D, cfg, params):
    from _w4_b4_features import cell_X
    V = cell_X(D['va'], cfg)
    if mid != 'S5_B4':
        Y = np.column_stack([D['tr']['outcome'] == pay for pay in PAY]).astype(float)
        Y -= D['tr']['base']
        return fit_predict(cell_X(D['tr'], cfg), Y, V, params)
    fold = folds_for(D['v'])
    dev = fold >= 0
    # Audit labels are never read: all target indexing is restricted to DEV.
    out = np.zeros_like(D['baseline'])
    for f in range(6):
        tr, te = np.flatnonzero(dev & (fold != f)), np.flatnonzero(fold == f)
        Y = D['v'].truth[tr] - D['baseline'][tr]
        out[te] = fit_predict(V[tr], Y, V[te], params)
    audit = np.flatnonzero(~dev)
    if len(audit):
        tr = np.flatnonzero(dev)
        out[audit] = fit_predict(V[tr], D['v'].truth[tr] - D['baseline'][tr], V[audit], params)
    return out


def nested_blend(D, cfg, params, strength):
    """Fixed hyperparameters; member and blend weights exclude each outer fold."""
    from _w4_b4_features import cell_X
    v, B = D['v'], D['baseline']
    X, fold = cell_X(D['va'], cfg), folds_for(v)
    out, weights = B.copy(), []
    for outer in range(6):
        member = B.copy()
        for inner in range(6):
            if inner == outer:
                continue
            tr = np.flatnonzero((fold >= 0) & (fold != outer) & (fold != inner))
            te = np.flatnonzero(fold == inner)
            R = fit_predict(X[tr], v.truth[tr] - B[tr], X[te], params)
            member[te] = normalized(B[te], R, strength)
        tr, te = np.flatnonzero((fold >= 0) & (fold != outer)), np.flatnonzero(fold == outer)
        R = fit_predict(X[tr], v.truth[tr] - B[tr], X[te], params)
        member[te] = normalized(B[te], R, strength)
        # Only this outer fold is retained; other outputs of blend_gain are discarded.
        blend = blend_gain(member, B, v, bootstrap=1)
        out[te] = blend['oof'][te]
        weights.append(blend['w_per_fold'][outer])
    audit, tr = np.flatnonzero(fold < 0), np.flatnonzero(fold >= 0)
    if len(audit):
        R = fit_predict(X[tr], v.truth[tr] - B[tr], X[audit], params)
        member = normalized(B[audit], R, strength)
        out[audit] = np.mean(weights) * member + (1 - np.mean(weights)) * B[audit]
    return out, weights


def smoke(mid, D, configs):
    v = D['v']
    sm = smoke_split(v)
    pos = {q: i for i, q in enumerate(v.qid)}
    ix = np.array([pos[q] for q in sm.qid])
    eps = sorted(set(D['tr']['episode']), key=lambda e: hashlib.sha256(str(e).encode()).hexdigest())[:24]
    ti = np.flatnonzero(np.isin(D['tr']['episode'], eps))
    SD = dict(D, v=sm, va=sliced(D['va'], ix), tr=sliced(D['tr'], ti), baseline=D['baseline'][ix])
    families = {}
    for p in model_grid(mid):
        families.setdefault(p['family'], p)
    rows = []
    for p in families.values():
        t = time.perf_counter()
        R = residuals(mid, SD, configs[0], p)
        for s in STRENGTHS:
            P = normalized(SD['baseline'], R, s)
            ev = evaluate(P, sm, scope='smoke', bootstrap=0)
            rows.append(dict(params=p, strength=s, points=ev['points']))
        print(f'[smoke] {mid} {p["family"]} ok {time.perf_counter()-t:.2f}s', flush=True)
    return rows


def run(mid):
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--grid-seconds', type=float, default=480)
    args = parser.parse_args()
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        if os.environ.get(k) != '1':
            raise RuntimeError(f'export {k}=1 before running')
    from _w4_b4_features import CONFIGS, load_data
    started = time.perf_counter()
    D = load_data()
    configs = list(CONFIGS)
    for name in ('results', 'grids', 'reports', 'preds'):
        (ROOT / name).mkdir(exist_ok=True)
    # Smoke failures propagate, leaving no false negative performance verdict.
    smoke_log = smoke(mid, D, configs)
    if args.smoke_only:
        (ROOT / 'grids' / f'{mid}_smoke.json').write_text(json.dumps(smoke_log, indent=1))
        return
    grid, best = [], None
    deadline = time.perf_counter() + min(args.grid_seconds, 480)
    for ci, cfg in enumerate(configs):
        for params in model_grid(mid):
            t = time.perf_counter()
            R, error = None, None
            if t >= deadline:
                error = 'not run: internal grid time budget reached'
            else:
                try:
                    R = residuals(mid, D, cfg, params)
                except Exception as exc:
                    error = repr(exc)
            for strength in STRENGTHS:
                row = dict(config_index=ci, config=cfg, model=params, strength=strength)
                if error:
                    row['error'] = error
                else:
                    P = normalized(D['baseline'], R, strength)
                    ev = evaluate(P, D['v'], 'dev', bootstrap=0)
                    row.update(points=ev['points'], condition_ratios=ev['condition_ratios'])
                    if best is None or ev['points'] > best['points']:
                        best = dict(row)
                row['fit_seconds'] = time.perf_counter() - t
                grid.append(row)
            (ROOT / 'grids' / f'{mid}.json').write_text(json.dumps(grid, indent=1))
            print(f'{mid} config={ci} model={params} {error or "completed"}', flush=True)
    if best is None:
        raise RuntimeError('No successful grid cells; inspect grids artifact')
    cfg, params, strength = best['config'], best['model'], best['strength']
    def refit(_split):
        return normalized(D['baseline'], residuals(mid, D, cfg, params), strength)
    det, P = check_determinism(refit, D['v'])
    if det > 1e-8:
        raise RuntimeError(f'refit determinism failed: {det}')
    v, B = D['v'], D['baseline']
    to_csv(ROOT / 'preds' / f'{mid}_validation.csv', v, P)
    base_ev = evaluate(B, v)
    paired = diff_ci(P, B, v)
    bg = blend_gain(P, B, v)
    bg.pop('oof')
    conditional = {}
    for c in CONDITIONS:
        hybrid = B.copy()
        ix = np.flatnonzero(v.condition == c)
        hybrid[ix] = P[ix]
        conditional[c] = diff_ci(hybrid, B, v)
    extra = dict(smoke=smoke_log, declared_cells=len(configs)*len(model_grid(mid))*5,
        completed_cells=sum('points' in r for r in grid), standalone_vs_B4=paired,
        episode_oof_blend=bg, B4_condition_ratios=base_ev['condition_ratios'],
        conditional_hybrid_vs_B4=conditional, feature_metadata=D['metadata'],
        selection_caveat='Global DEV hyperparameter/strength selection is optimistic; audit truth never fit or tuned.',
        time_cap_seconds=600, grid_budget_seconds=min(args.grid_seconds, 480))
    if mid == 'S5_B4':
        nested, weights = nested_blend(D, cfg, params, strength)
        to_csv(ROOT / 'preds' / f'{mid}_NESTED_validation.csv', v, nested)
        extra['nested_fixed_selected_hyperparameters'] = dict(
            evaluation=evaluate(nested, v), vs_B4=diff_ci(nested, B, v), outer_weights=weights,
            caveat='Learner/weight fitting nested; global hyperparameter selection remains optimistic.')
        extra['non_nested_blend_caveat'] = 'OOF member labels cross blend-training folds; use nested diagnostic.'
    doc = write_result(mid, 'codex', 'B4 residual '+params['family'],
        dict(config=cfg, model=params, strength=strength), v, P,
        time.perf_counter()-started, SEED, determinism=det, extra=extra)
    lines = [mid, f'Grid cells completed: {extra["completed_cells"]}/{extra["declared_cells"]}; failures/skips retained in grids/{mid}.json',
        f'Selected: {json.dumps(doc["params"])}',
        f'DEV points {doc["points"]:.6f}; CI95 {doc["ci95"]}; B4 {base_ev["points"]:.6f}',
        f'Condition ratios: {json.dumps(doc["condition_ratios"])}',
        f'B4 condition ratios: {json.dumps(base_ev["condition_ratios"])}',
        f'Class shares: {json.dumps(doc["class_error_shares"])}',
        f'Standalone paired vs B4: {json.dumps(paired)}',
        f'Episode OOF blend vs B4: {json.dumps(bg)}',
        f'Conditional hybrid paired CIs: {json.dumps(conditional)}',
        f'Runtime {doc["runtime_seconds"]:.2f}s; repeated selected refit max diff {det}; stdlib-only: no',
        extra['selection_caveat']]
    if mid == 'S5_B4':
        lines += [f'Nested fixed-selected diagnostic: {json.dumps(extra["nested_fixed_selected_hyperparameters"])}',
                  extra['non_nested_blend_caveat']]
    (ROOT / 'reports' / f'{mid}.md').write_text('\n'.join(lines)+'\n')
    print(f'{mid}: {doc["points"]:.6f}; paired B4 {paired}', flush=True)
    return doc
