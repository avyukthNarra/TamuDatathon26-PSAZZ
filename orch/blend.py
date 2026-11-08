"""Cross-fitted blend of a base model with another: weight on win prob and on size mix fitted on fold A, scored on B, swapped."""
import csv, sys, itertools, numpy as np
from common import *
v = load("validation"); T = v["targets"]; Qd = {q["query_id"]: q for q in v["Q"]}
rd = lambda f: {r["query_id"]: np.array([float(r[c]) for c in PCOLS]) for r in csv.DictReader(open(f))}
base, other = rd(sys.argv[1]), rd(sys.argv[2])
A, B = folds(v); ep = {q: Qd[q]["episode_id"] for q in T}
def mix(q, a, b):
    """a: weight of other on win prob; b: weight of other on size-given-win."""
    w = (1 - a) * (1 - base[q][0]) + a * (1 - other[q][0])
    s1 = base[q][1:] / base[q][1:].sum(); s2 = other[q][1:] / max(other[q][1:].sum(), 1e-12)
    s = (1 - b) * s1 + b * s2
    return np.concatenate([[1 - w], w * s])
grid = np.round(np.arange(0, 1.01, 0.05), 2)
def best(eps):
    qs = [q for q in T if ep[q] in eps]
    return max(((a, b) for a in grid for b in grid), key=lambda ab: points({q: mix(q, *ab) for q in qs}, v, qs)[0])
P, chosen = {}, []
for tr, te in ((A, B), (B, A)):
    ab = best(tr); chosen.append(ab)
    for q in T:
        if ep[q] in te: P[q] = mix(q, *ab)
pts, rat = points(P, v)
print("base", round(points(base, v)[0], 3), "other", round(points(other, v)[0], 3))
print("crossfit blend", round(pts, 3), "weights chosen (win, size) per fold", chosen, {k: round(x, 4) for k, x in rat.items()})
for a, b in [(0.1, 0), (0.2, 0), (0.3, 0), (0.2, 0.2), (0.3, 0.3)]:
    print(f"fixed win {a} size {b}: {points({q: mix(q, a, b) for q in T}, v)[0]:.3f}")
