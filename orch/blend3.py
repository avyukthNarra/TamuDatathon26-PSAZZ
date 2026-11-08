"""Cross-fitted win-probability blend of v13 with other members (log-odds pool), v13 sizes kept.
Weights on a simplex grid, chosen on fold A, scored on B, swapped."""
import csv, sys, itertools, numpy as np
from common import *
v = load("validation"); T = v["targets"]; Qd = {q["query_id"]: q for q in v["Q"]}
rd = lambda f: {r["query_id"]: np.array([float(r[c]) for c in PCOLS]) for r in csv.DictReader(open(f))}
files = sys.argv[1:]; M = [rd(f) for f in files]
A, B = folds(v); ep = {q: Qd[q]["episode_id"] for q in T}
lg = lambda p: np.log(p / (1 - p))
W = {q: np.array([lg(np.clip(1 - m[q][0], 1e-5, 1 - 1e-5)) for m in M]) for q in T}
def pred(q, w, mode):
    z = W[q]
    p = 1 / (1 + np.exp(-(w @ z))) if mode == "logodds" else w @ (1 / (1 + np.exp(-z)))
    s = M[0][q][1:] / M[0][q][1:].sum()
    return np.concatenate([[1 - p], p * s])
step = 0.1; grid = [np.array(w) for w in itertools.product(np.arange(0, 1 + 1e-9, step), repeat=len(M)) if abs(sum(w) - 1) < 1e-9]
for mode in ("logodds", "linear"):
    P, ch = {}, []
    for tr, te in ((A, B), (B, A)):
        qs = [q for q in T if ep[q] in tr]
        w = max(grid, key=lambda w: points({q: pred(q, w, mode) for q in qs}, v, qs)[0]); ch.append(np.round(w, 2).tolist())
        for q in T:
            if ep[q] in te: P[q] = pred(q, w, mode)
    eq = np.ones(len(M)) / len(M)
    qa = [q for q in T if ep[q] in A]; qb = [q for q in T if ep[q] in B]
    print(f"{mode:8s} crossfit {points(P, v)[0]:.3f} (A {points(P, v, qa)[0]:.2f} B {points(P, v, qb)[0]:.2f}) weights {ch} | equal weights {points({q: pred(q, eq, mode) for q in T}, v)[0]:.3f}")
for f, m in zip(files, M): print(f"member {points(m, v)[0]:.3f} {f}")
