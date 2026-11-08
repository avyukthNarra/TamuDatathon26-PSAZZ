#!/usr/bin/env python3
"""Independent check of the g5-size lag/polarity atom claim (stdlib only)."""
import csv, json, math, random, sys
from collections import defaultdict
from pathlib import Path

MAIN = Path('/Users/avynarra/Documents/Datathon/worktrees/pranav-5629/mainFiles')
sys.path.insert(0, str(MAIN))
from slot_forecasting.contest import score_contest_files  # noqa: E402

VD, TD = MAIN / 'public/validation', MAIN / 'public/train'
V6 = MAIN / 'runs/combined-v6/validation_predictions.csv'
OUT = Path('/Users/avynarra/Documents/Datathon/research_pf/h2-size-verify')
WORK = OUT / 'work'
WORK.mkdir(parents=True, exist_ok=True)
SIDX = {1: 0, 2: 1, 5: 2, 20: 3}
FLOOR = 1e-4
LAGS_CLAIM = (1, 2, 4)
LAGS_WIDE = tuple(range(1, 9))


def read_csv(p):
    with open(p, newline='') as fh:
        return list(csv.DictReader(fh))


def load_split(d):
    queries = {r['query_id']: r for r in read_csv(d / 'queries.csv')}
    hist = defaultdict(dict)
    for r in read_csv(d / 'history.csv'):
        hist[r['machine_id']][int(r['tick'])] = int(r['payout'])
    machines = {}
    for q in queries.values():
        cut = int(q['cutoff'])
        h = hist[q['machine_id']]
        # history is a window: ticks before the window are missing (None)
        machines[q['query_id']] = (q['machine_type'], [h.get(t) for t in range(cut)])
    return queries, machines


# ---------- atoms from validation targets (reproduction only) ----------
def mean_vec(vs):
    n = len(vs)
    return [sum(v[i] for v in vs) / n for i in range(4)]


def dist2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def target_atoms(queries, targets):
    by_type = defaultdict(list)
    for r in targets:
        p = [float(r[c]) for c in ('p_1', 'p_2', 'p_5', 'p_20')]
        s = sum(p)
        by_type[queries[r['query_id']]['machine_type']].append((r['query_id'], [x / s for x in p]))
    atoms, report = {}, {}
    for typ, items in by_type.items():
        ws = [w for _, w in items]
        c = [min(ws, key=lambda w: w[0]), max(ws, key=lambda w: w[0])]
        for _ in range(50):
            g = [[], []]
            for w in ws:
                g[0 if dist2(w, c[0]) <= dist2(w, c[1]) else 1].append(w)
            c = [mean_vec(x) for x in g]
        dev = max(math.sqrt(min(dist2(w, c[0]), dist2(w, c[1]))) for w in ws)
        c.sort(key=lambda v: -v[0])  # atom 0 = higher p(1)
        atoms[typ] = c
        report[typ] = {'sizes': [len(x) for x in g], 'max_dev': dev}
    return atoms, report


# ---------- posterior over (L, s) ----------
def fval(pay, t, L):
    if t - L < 0 or pay[t - L] is None:
        return None
    return 1 if pay[t - L] >= 2 else 0


def wins_before(pay, cut):
    return [(t, SIDX[pay[t]]) for t in range(cut) if pay[t] in SIDX]


def posterior_predict(pay, cut, A, lags):
    """Uniform prior over (L,s). Returns size mix (4) at tick cut and posterior dict."""
    wins = wins_before(pay, cut)
    states = [(L, s) for L in lags for s in (0, 1)]
    ll = {}
    for L, s in states:
        tot = 0.0
        for t, y in wins:
            f = fval(pay, t, L)
            p = 0.5 * (A[0][y] + A[1][y]) if f is None else A[s ^ f][y]
            tot += math.log(p)
        ll[(L, s)] = tot
    m = max(ll.values())
    w = {k: math.exp(v - m) for k, v in ll.items()}
    Z = sum(w.values())
    post = {k: v / Z for k, v in w.items()}
    mix = [0.0] * 4
    for (L, s), pr in post.items():
        f = fval(pay, cut, L)
        dist = [0.5 * (A[0][i] + A[1][i]) for i in range(4)] if f is None else A[s ^ f]
        for i in range(4):
            mix[i] += pr * dist[i]
    return mix, post


def predict_all(queries, machines, atoms, lags):
    mixes, posts = {}, {}
    for qid, (typ, pay) in machines.items():
        mixes[qid], posts[qid] = posterior_predict(pay, len(pay), atoms[typ], lags)
    return mixes, posts


# ---------- submission + scoring ----------
def write_sub(path, qids, v6, mixes):
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['query_id', 'p_0', 'p_1', 'p_2', 'p_5', 'p_20'])
        for qid in qids:
            p0 = float(v6[qid]['p_0'])
            p = [p0] + [(1 - p0) * m for m in mixes[qid]]
            p = [max(x, FLOOR) for x in p]
            s = sum(p)
            p = [x / s for x in p]
            w.writerow([qid] + [repr(x) for x in p])


def subset_csv(src, dst, keep):
    rows = read_csv(src)
    with open(dst, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            if keep(r):
                w.writerow(r)


def score(sub_path, tag, parity=None):
    """Full or half (parity: 'even'/'odd' on episode index) validation scoring."""
    if parity is None:
        files = [VD / n for n in ('targets.csv', 'queries.csv', 'conditions.csv', 'reference.csv')]
        sub = sub_path
    else:
        def ep_ok(r):
            k = int(r['episode_id'].rsplit('e', 1)[1])
            return (k % 2 == 0) == (parity == 'even')
        d = WORK / f'half_{parity}_{tag}'
        d.mkdir(exist_ok=True)
        keep_q = {r['query_id'] for r in read_csv(VD / 'queries.csv') if ep_ok(r)}
        keep = lambda r: r['query_id'] in keep_q  # noqa: E731
        names = ('targets.csv', 'queries.csv', 'conditions.csv', 'reference.csv')
        files = []
        for n in names:
            dst = d / n
            subset_csv(VD / n, dst, keep)
            files.append(dst)
        sub = d / 'submission.csv'
        subset_csv(sub_path, sub, keep)
    res = score_contest_files(str(sub), str(files[0]), str(files[1]), str(files[2]), str(files[3]), max_points=60)
    return {'points': res['technical_points'], 'mean_loss_ratio': res['mean_loss_ratio'],
            'by_condition': {c: v['loss_ratio'] for c, v in res['by_condition'].items()}}


# ---------- train-only EM for atoms + lag marginal ----------
def suff_stats(machines, queries, lags):
    stats = []
    for qid, (typ, pay) in machines.items():
        cut = len(pay)
        n = {L: [[0] * 4, [0] * 4] for L in lags}
        nu = {L: [0] * 4 for L in lags}
        for t, y in wins_before(pay, cut):
            for L in lags:
                f = fval(pay, t, L)
                if f is None:
                    nu[L][y] += 1
                else:
                    n[L][f][y] += 1
        stats.append((typ, n, nu))
    return stats


def em_fit(stats, lags, seed, iters=200):
    rng = random.Random(seed)
    types = sorted({t for t, _, _ in stats})

    def simplex():
        g = [rng.gammavariate(1.0, 1.0) for _ in range(4)]
        s = sum(g)
        return [x / s for x in g]

    A = {T: [simplex(), simplex()] for T in types}
    piL = {L: 1.0 / len(lags) for L in lags}
    prev = None
    for _ in range(iters):
        N = {T: [[1e-9] * 4, [1e-9] * 4] for T in types}
        piacc = {L: 0.0 for L in lags}
        total = 0.0
        for typ, n, nu in stats:
            a0, a1 = A[typ]
            both = [0.5 * (a0[y] + a1[y]) for y in range(4)]
            ll = {}
            for L in lags:
                for s in (0, 1):
                    v = 0.0
                    for y in range(4):
                        for f in (0, 1):
                            c = n[L][f][y]
                            if c:
                                v += c * math.log(A[typ][s ^ f][y])
                        if nu[L][y]:
                            v += nu[L][y] * math.log(both[y])
                    ll[(L, s)] = v
            m = max(ll.values())
            w = {k: piL[k[0]] * 0.5 * math.exp(v - m) for k, v in ll.items()}
            Z = sum(w.values())
            total += m + math.log(Z)
            for (L, s), v in w.items():
                g = v / Z
                piacc[L] += g
                for y in range(4):
                    for f in (0, 1):
                        c = n[L][f][y]
                        if c:
                            N[typ][s ^ f][y] += g * c
                    c = nu[L][y]
                    if c:
                        # unknown f: posterior over atom a is A[a][y] / (A0[y]+A1[y])
                        N[typ][0][y] += g * c * A[typ][0][y] / (both[y] * 2)
                        N[typ][1][y] += g * c * A[typ][1][y] / (both[y] * 2)
        for T in types:
            for a in (0, 1):
                tot = sum(N[T][a])
                A[T][a] = [x / tot for x in N[T][a]]
        zl = sum(piacc.values())
        piL = {L: piacc[L] / zl for L in lags}
        if prev is not None and abs(total - prev) < 1e-7 * abs(total):
            break
        prev = total
    return A, piL, total


def canon(A):
    out = {}
    for T, (a0, a1) in A.items():
        out[T] = sorted([a0, a1], key=lambda v: -v[0])
    return out


def main():
    res = {}
    queries_v, machines_v = load_split(VD)
    queries_t, machines_t = load_split(TD)
    targets = read_csv(VD / 'targets.csv')
    v6 = {r['query_id']: r for r in read_csv(V6)}
    qids = [r['query_id'] for r in read_csv(VD / 'queries.csv')]
    assert set(v6) == set(qids)

    # baseline check: v6 as-is through the scorer
    base_sub = WORK / 'v6_as_is.csv'
    with open(base_sub, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['query_id', 'p_0', 'p_1', 'p_2', 'p_5', 'p_20'])
        for qid in qids:
            r = v6[qid]
            w.writerow([qid] + [r[c] for c in ('p_0', 'p_1', 'p_2', 'p_5', 'p_20')])
    res['v6'] = score(base_sub, 'v6')

    atoms, rep = target_atoms(queries_v, targets)
    res['target_atom_clusters'] = rep
    res['target_atoms'] = atoms

    # Reproduction of the claim
    mix_c, post_c = predict_all(queries_v, machines_v, atoms, LAGS_CLAIM)
    f = WORK / 'claim.csv'
    write_sub(f, qids, v6, mix_c)
    res['claim_repro'] = score(f, 'claim')
    # 2(a) uniform prior over 1..8, target atoms
    mix_a, post_a = predict_all(queries_v, machines_v, atoms, LAGS_WIDE)
    f = WORK / '2a_wide_lags.csv'
    write_sub(f, qids, v6, mix_a)
    res['2a_wide_lags'] = score(f, '2a')

    # 2(b) atoms from TRAIN-only EM, same {1,2,4} posterior
    stats_t = suff_stats(machines_t, queries_t, LAGS_WIDE)
    best = None
    fits = []
    for seed in range(6):
        A, piL, total = em_fit(stats_t, LAGS_WIDE, seed)
        fits.append({'seed': seed, 'train_loglik': total})
        if best is None or total > best[2]:
            best = (A, piL, total)
    A_em, piL, total = best
    A_em = canon(A_em)
    res['2b_em'] = {'restarts': fits, 'best_loglik': total,
                    'atoms': A_em, 'lag_marginal_pi': piL}
    mix_b, post_b = predict_all(queries_v, machines_v, A_em, LAGS_CLAIM)
    f = WORK / '2b_em_atoms_124.csv'
    write_sub(f, qids, v6, mix_b)
    res['2b_em_atoms_124'] = score(f, '2b')
    mix_b2, _ = predict_all(queries_v, machines_v, A_em, LAGS_WIDE)
    f = WORK / '2b_em_atoms_wide.csv'
    write_sub(f, qids, v6, mix_b2)
    res['2b_em_atoms_wide'] = score(f, '2b2')

    # 2(c) lag usage from TRAIN only: posterior mass per lag over train machines (EM atoms, uniform prior 1..8)
    mass = {L: 0.0 for L in LAGS_WIDE}
    argmax_count = {L: 0 for L in LAGS_WIDE}
    for qid, (typ, pay) in machines_t.items():
        _, post = posterior_predict(pay, len(pay), A_em[typ], LAGS_WIDE)
        pl = {L: post[(L, 0)] + post[(L, 1)] for L in LAGS_WIDE}
        for L in LAGS_WIDE:
            mass[L] += pl[L]
        argmax_count[max(pl, key=pl.get)] += 1
    n_tr = len(machines_t)
    res['2c_lag_mass_train'] = {str(L): mass[L] / n_tr for L in LAGS_WIDE}
    res['2c_argmax_counts_train'] = {str(L): argmax_count[L] for L in LAGS_WIDE}
    res['2c_em_pi_L'] = {str(L): piL[L] for L in LAGS_WIDE}

    # Honest split: even/odd episode indices
    for name, f in (('claim', WORK / 'claim.csv'), ('2a', WORK / '2a_wide_lags.csv')):
        res[f'{name}_halves'] = {p: score(f, name, parity=p) for p in ('even', 'odd')}
    res['v6_halves'] = {p: score(base_sub, 'v6', parity=p) for p in ('even', 'odd')}

    # Summary of atom-to-query agreement and posterior lag stats for the claim (reproduction)
    with open(OUT / 'results.json', 'w') as fh:
        json.dump(res, fh, indent=2, default=float)
    print(json.dumps({k: v for k, v in res.items() if k not in ('target_atoms',)}, indent=1, default=float))


if __name__ == '__main__':
    main()
