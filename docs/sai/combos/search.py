"""Bounded team-combination search; run dev, then audit exactly once.

All weight fits, objective denominators, folds and structure selection use dev.
OOF metrics evaluate fixed structures; a separate nested selector measures search
optimism with 3 inner folds within 6 outer episode folds. Frozen member forecasts
may themselves be validation-tuned: this procedure cannot undo that leakage.
"""
import argparse
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from harness import load_split, read_pred_csv, to_csv, evaluate, diff_ci
from harness.data import CONDITIONS
from harness.scoring import scope_episodes, _prep
from harness.stack_w5 import parts, combine

OUT = HERE / 'combos'
PREDOUT = HERE / 'preds'
POOL = ['TM_RULE', 'TM_MIXER3', 'TM_MIXER4', 'TM_ZPOOL', 'TM_AVYCOMBO',
        'C1', 'C2', 'M9b_B4', 'S5_B4', 'TM_E67', 'TM_COMBINED3']
STDLIB = {'TM_RULE', 'TM_MIXER3', 'TM_MIXER4', 'TM_ZPOOL', 'TM_COMBINED3'}
FAMILIES = ('linear', 'factorised', 'geometric')
SEED = 'combo-top10-v1'


def folds(v, episodes, count, salt):
    """Condition-stratified episode folds, with no machine crossing a fold."""
    out = {}
    for c in CONDITIONS:
        eps = [e for e in episodes if v.condition[v.episode_rows(e)[0]] == c]
        eps.sort(key=lambda e: hashlib.sha256(f'{SEED}|{salt}|{e}'.encode()).hexdigest())
        out.update({e: i % count for i, e in enumerate(eps)})
    return out


def rows_for(v, episodes):
    return np.concatenate([v.episode_rows(e) for e in episodes])


def row_weights(v, rows):
    """Condition reference losses estimated ONLY from this fit's episodes."""
    w = np.zeros(v.n)
    for c in CONDITIONS:
        rr = rows[v.condition[rows] == c]
        eps = sorted(set(v.episode[rr]))
        base = np.mean([((v.ref[v.episode_rows(e)] - v.truth[v.episode_rows(e)]) ** 2).sum(1).mean() for e in eps])
        for e in eps:
            er = v.episode_rows(e)
            w[er] = 1 / (len(eps) * len(er) * max(base, 1e-9))
    return w[rows]


def forecast(A, family, x):
    k = len(A)
    if family == 'linear':
        return np.einsum('k,knc->nc', x, A)
    if family == 'geometric':
        z = np.einsum('k,knc->nc', x, np.log(np.maximum(A, 1e-12)))
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(1, keepdims=True)
    pw, sh = zip(*(parts(p) for p in A))
    return combine(pw, sh, x[:k], x[k:])


def fit(A, family, q, w):
    k = len(A)
    if family == 'factorised':
        pw, sh = zip(*(parts(p) for p in A))
        pw, sh = np.array(pw), np.array(sh)
        def objective(x):
            a, b = x[:k], x[k:]
            win = a @ pw
            size = np.einsum('k,knc->nc', b, sh)
            p = np.column_stack([1-win, win[:, None]*size])
            r = 2*w[:, None]*(p-q)
            da = np.einsum('kn,n->k', pw, -r[:, 0]+np.sum(r[:, 1:]*size, axis=1))
            db = np.einsum('knc,nc->k', sh, r[:, 1:]*win[:, None])
            return float(np.sum(w[:, None]*(p-q)**2)), np.r_[da, db]
        constraints = [{'type': 'eq', 'fun': lambda x: x[:k].sum()-1,
                        'jac': lambda x: np.r_[np.ones(k), np.zeros(k)]},
                       {'type': 'eq', 'fun': lambda x: x[k:].sum()-1,
                        'jac': lambda x: np.r_[np.zeros(k), np.ones(k)]}]
        x0 = np.full(2*k, 1/k)
    else:
        logs = np.log(np.maximum(A, 1e-12))
        def objective(x):
            p = forecast(A, family, x)
            r = 2*w[:, None]*(p-q)
            if family == 'linear':
                grad = np.einsum('knc,nc->k', A, r)
            else:
                centered = logs - np.einsum('knc,nc->kn', logs, p)[:, :, None]
                grad = np.einsum('knc,nc->k', centered, r*p)
            return float(np.sum(w[:, None]*(p-q)**2)), grad
        constraints = [{'type': 'eq', 'fun': lambda x: x.sum()-1,
                        'jac': lambda x: np.ones(k)}]
        x0 = np.full(k, 1/k)
    result = minimize(objective, x0, jac=True, method='SLSQP', bounds=[(0, 1)]*len(x0),
                      constraints=constraints, options={'maxiter': 200, 'ftol': 1e-11})
    if not result.success:
        raise RuntimeError(f'{family}: optimizer failed: {result.message}')
    x = np.clip(result.x, 0, 1)
    if family == 'factorised':
        x[:k] /= x[:k].sum(); x[k:] /= x[k:].sum()
    else:
        x /= x.sum()
    return x


def cv(v, A, family, episodes, count, salt):
    fold = folds(v, episodes, count, salt)
    p = np.full((v.n, 5), .2)
    weights = []
    for f in range(count):
        tr = rows_for(v, [e for e in episodes if fold[e] != f])
        te = rows_for(v, [e for e in episodes if fold[e] == f])
        x = fit(A[:, tr], family, v.truth[tr], row_weights(v, tr))
        p[te] = forecast(A[:, te], family, x)
        weights.append(x.tolist())
    return p, weights


def weights_dict(names, family, x):
    if family == 'factorised':
        return {'win': dict(zip(names, x[:len(names)].tolist())),
                'size': dict(zip(names, x[len(names):].tolist()))}
    return dict(zip(names, x.tolist()))


def load():
    v = load_split('validation')
    names = [n for n in POOL if (HERE / 'preds' / f'{n}_validation.csv').exists()]
    A = np.array([_prep(read_pred_csv(HERE / 'preds' / f'{n}_validation.csv', v), v) for n in names])
    return v, names, A


def dev():
    if (OUT / 'audit.once.json').exists():
        raise RuntimeError('Audit already attempted: dev ranking must remain frozen.')
    v, names, A = load()
    eps = scope_episodes(v, 'dev')
    rr = rows_for(v, eps)
    reference = A[names.index('TM_COMBINED3')]
    specs = [(family, sub) for size in (2, 3) for sub in itertools.combinations(range(len(names)), size) for family in FAMILIES]
    results, oofs = [], []
    for i, (family, sub) in enumerate(specs):
        aa = A[list(sub)]
        p, per_fold = cv(v, aa, family, eps, 6, 'outer')
        x = fit(aa[:, rr], family, v.truth[rr], row_weights(v, rr))
        members = [names[j] for j in sub]
        results.append({'id': f'{family}:' + '+'.join(members), 'family': family, 'members': members,
                        'weights': weights_dict(members, family, x), 'weight_vector': x.tolist(),
                        'fold_weights': per_fold, 'dev': evaluate(p, v, 'dev', bootstrap=0),
                        'stdlib_only': {n for j, n in enumerate(members) if x[j] > 1e-8 or (family == 'factorised' and x[j+len(members)] > 1e-8)} <= STDLIB})
        oofs.append(p)
        if i % 30 == 0:
            print('OOF', i, '/', len(specs), flush=True)
    order = sorted(range(len(results)), key=lambda i: (-results[i]['dev']['points'], results[i]['id']))
    best = oofs[order[0]]
    # CIs from the shared harness, not uncertainty across machine rows.
    for position, i in enumerate(order):
        if position % 100 == 0:
            print('bootstrap', position, '/', len(order), flush=True)
        results[i]['dev'] = evaluate(oofs[i], v, 'dev')
        results[i]['vs_reference'] = diff_ci(oofs[i], reference, v, 'dev')
        results[i]['vs_best'] = diff_ci(oofs[i], best, v, 'dev')
        lo, hi = results[i]['vs_best']['ci95']
        results[i]['noise_band'] = lo <= 0 <= hi
    # Nested structure selector: inner OOF chooses ONE structure on each outer
    # training set; its outer holdout is untouched by either weights or choice.
    fold = folds(v, eps, 6, 'outer')
    nested = np.full((v.n, 5), .2)
    selections = []
    for f in range(6):
        train_eps = [e for e in eps if fold[e] != f]
        tr = rows_for(v, train_eps)
        te = rows_for(v, [e for e in eps if fold[e] == f])
        w = row_weights(v, tr)
        losses = []
        for i, (family, sub) in enumerate(specs):
            inner, _ = cv(v, A[list(sub)], family, train_eps, 3, f'inner-{f}')
            losses.append(float(np.sum(w[:, None]*(inner[tr]-v.truth[tr])**2)))
        chosen = int(np.argmin(losses))
        family, sub = specs[chosen]
        aa = A[list(sub)]
        x = fit(aa[:, tr], family, v.truth[tr], w)
        nested[te] = forecast(aa[:, te], family, x)
        selections.append({'outer_fold': f, 'structure': results[chosen]['id'], 'weights': x.tolist()})
        print('nested fold', f, 'selected', results[chosen]['id'], flush=True)
    top = []
    # Suppress numerical copies caused by zero-weight extra members. Keep
    # distinct deployed forecasts, retaining every structure in the record.
    finals = []
    for i in order:
        family, sub = specs[i]
        full = forecast(A[list(sub)], family, np.array(results[i]['weight_vector']))
        duplicate = any(np.max(np.abs(full[rr]-p[rr])) < 1e-5 for j, p in finals)
        if duplicate:
            continue
        finals.append((i, full))
        top.append(i)
        if len(top) == 10:
            break
    for rank, (i, p) in enumerate(finals, 1):
        results[i]['rank'] = rank
        to_csv(PREDOUT / f'COMBO_{rank}_validation.csv', v, p)
        to_csv(OUT / f'COMBO_{rank}_oof_validation.csv', v, oofs[i])
    source_hashes = {n: hashlib.sha256((HERE / 'preds' / f'{n}_validation.csv').read_bytes()).hexdigest() for n in names}
    hashes = {str(rank): hashlib.sha256((PREDOUT / f'COMBO_{rank}_validation.csv').read_bytes()).hexdigest() for rank in range(1, 11)}
    doc = {'seed': SEED, 'folds': fold, 'members': names, 'missing_members': sorted(set(POOL)-set(names)),
           'source_sha256': source_hashes, 'prediction_sha256': hashes,
           'reference_dev': evaluate(reference, v, 'dev'),
           'selection': {'dev': evaluate(nested, v, 'dev'), 'vs_reference': diff_ci(nested, reference, v, 'dev'), 'outer_selections': selections},
           'top10': [results[i]['id'] for i in top], 'noise_band_count': sum(r['noise_band'] for r in results),
           'duplicate_tolerance': 1e-5, 'results': [results[i] for i in order]}
    (OUT / 'results.json').write_text(json.dumps(doc, indent=2) + '\n')
    print('FROZEN', doc['top10'], flush=True)


def audit():
    path = OUT / 'results.json'
    doc = json.loads(path.read_text())
    if 'audit' in doc:
        raise RuntimeError('Audit results already present.')
    # Check immutable inputs BEFORE consuming the one-shot guard.
    for n, h in doc['source_sha256'].items():
        assert hashlib.sha256((HERE / 'preds' / f'{n}_validation.csv').read_bytes()).hexdigest() == h
    for rank, h in doc['prediction_sha256'].items():
        assert hashlib.sha256((PREDOUT / f'COMBO_{rank}_validation.csv').read_bytes()).hexdigest() == h
    log = HERE / 'results/audit_access.log'
    if log.exists() and any(line.rstrip().endswith('\tCOMBO_TOP10') for line in log.read_text().splitlines()):
        raise RuntimeError('COMBO_TOP10 audit already appears in access log.')
    with (OUT / 'audit.once.json').open('x') as f:
        json.dump({'frozen_results_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'prediction_sha256': doc['prediction_sha256'], 'top10': doc['top10']}, f, indent=2)
    v = load_split('validation')
    scores = {}
    for name in ['TM_COMBINED3'] + [f'COMBO_{i}' for i in range(1, 11)]:
        p = read_pred_csv((PREDOUT if name.startswith('COMBO_') else HERE / 'preds') / f'{name}_validation.csv', v)
        scores[name] = evaluate(p, v, 'audit', who='COMBO_TOP10')
        print(name, scores[name]['points'], flush=True)
        (OUT / 'audit.partial.json').write_text(json.dumps(scores, indent=2) + '\n')
    doc['audit'] = scores
    path.write_text(json.dumps(doc, indent=2) + '\n')


if __name__ == '__main__':
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        if os.environ.get(key) != '1':
            raise RuntimeError(f'Set {key}=1 for shared-machine jobs.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['dev', 'audit'])
    parser.add_argument('--out', type=Path, default=OUT, help='New directory for dev replay; frozen default refuses changes after audit.')
    args = parser.parse_args()
    OUT = args.out.resolve()
    PREDOUT = HERE / 'preds' if OUT == HERE / 'combos' else OUT / 'preds'
    OUT.mkdir(parents=True, exist_ok=True)
    PREDOUT.mkdir(parents=True, exist_ok=True)
    (dev if args.mode == 'dev' else audit)()
