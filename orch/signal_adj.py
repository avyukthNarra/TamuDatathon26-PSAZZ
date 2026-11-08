"""Cross-fitted logit shift of a model's win probability by (machine_type, signal); size mix kept."""
import csv, sys, numpy as np
from scipy.optimize import minimize_scalar
from common import *
v = load("validation"); T = v["targets"]; Qd = {q["query_id"]: q for q in v["Q"]}
M = {r["query_id"]: np.array([float(r[c]) for c in PCOLS]) for r in csv.DictReader(open(sys.argv[1]))}
A, B = folds(v)
ep = {q: Qd[q]["episode_id"] for q in T}
lg = lambda p: np.log(p / (1 - p)); sg = lambda x: 1 / (1 + np.exp(-x))
def adjust(q, b):
    p = M[q]; w = np.clip(1 - p[0], 1e-4, 1 - 1e-4); w2 = sg(lg(w) + b)
    return np.concatenate([[1 - w2], p[1:] / w * w2])
def fit(eps, key):
    groups = {}
    for q in T:
        if ep[q] in eps: groups.setdefault(key(q), []).append(q)
    out = {}
    for k, qs in groups.items():
        f = lambda b: sum(((adjust(q, b) - T[q]) ** 2).sum() for q in qs)
        out[k] = minimize_scalar(f, bounds=(-3, 3), method="bounded").x
    return out
for name, key in [("type", lambda q: Qd[q]["machine_type"]), ("signal", lambda q: Qd[q]["signal"]),
                  ("type x signal", lambda q: (Qd[q]["machine_type"], Qd[q]["signal"]))]:
    P = {}
    for tr, te in ((A, B), (B, A)):
        bs = fit(tr, key)
        for q in T:
            if ep[q] in te: P[q] = adjust(q, bs.get(key(q), 0.0))
    qa = [q for q in T if ep[q] in A]; qb = [q for q in T if ep[q] in B]
    print(f"{name:14s} crossfit {points(P, v)[0]:.3f}  (halves A {points(P, v, qa)[0]:.2f} B {points(P, v, qb)[0]:.2f})  shifts(full) { {k: round(x, 2) for k, x in fit(A | B, key).items()} }")
qa = [q for q in T if ep[q] in A]; qb = [q for q in T if ep[q] in B]
print(f"base           {points(M, v)[0]:.3f}  (halves A {points(M, v, qa)[0]:.2f} B {points(M, v, qb)[0]:.2f})")
