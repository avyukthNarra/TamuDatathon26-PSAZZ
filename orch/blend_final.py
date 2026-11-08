"""Reproduce the 57.91 blend: equal log-odds average of three win models, v13's size split.

python blend_final.py   (from orch/; reads results/*.csv and ../mainFiles/public/validation)
"""
import csv, numpy as np
from common import *
v = load("validation"); T = v["targets"]; Qd = {q["query_id"]: q for q in v["Q"]}; A, B = folds(v)
rd = lambda f: {r["query_id"]: np.array([float(r[c]) for c in PCOLS]) for r in csv.DictReader(open(f))}
lg = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))
def blend(win_members, size_member):
    out = {}
    for q in T:
        p = 1 / (1 + np.exp(-np.mean([lg(1 - m[q][0]) for m in win_members])))
        s = size_member[q][1:] / size_member[q][1:].sum()
        out[q] = np.concatenate([[1 - p], p * s])
    return out
V, VS = rd("results/v13_crossfit.csv"), rd("results/v13_shortpatch_crossfit.csv")
E, S = rd("results/exact_crossfit.csv"), rd("results/sysid_train_only.csv")
qa = [q for q in T if Qd[q]["episode_id"] in A]; qb = [q for q in T if Qd[q]["episode_id"] in B]
rows = [("v13", V), ("v13 + short patch", VS), ("exact filter", E), ("sysid rule library", S),
        ("blend v13, exact, sysid", blend([V, E, S], V)), ("blend v13+short, exact, sysid", blend([VS, E, S], V))]
for name, P in rows:
    pts, r = points(P, v)
    print(f"{name:32s} {pts:.3f}  halves A {points(P, v, qa)[0]:.2f} B {points(P, v, qb)[0]:.2f}  "
          + " ".join(f"{k} {x:.4f}" for k, x in sorted(r.items())))
