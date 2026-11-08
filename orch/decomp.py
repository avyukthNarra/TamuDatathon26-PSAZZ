import csv, sys, numpy as np
from common import *
v = load("validation")
def readp(f): return {r["query_id"]: np.array([float(r[c]) for c in PCOLS]) for r in csv.DictReader(open(f))}
M = readp(sys.argv[1])
T = v["targets"]
def recompose(win, size): return np.concatenate([[1 - win], win * size])
w = {q: 1 - M[q][0] for q in M}; s = {q: M[q][1:] / max(1e-12, 1 - M[q][0]) for q in M}
tw = {q: 1 - T[q][0] for q in T}; ts = {q: T[q][1:] / (1 - T[q][0]) for q in T}
print("model           ", round(points(M, v)[0], 3), {k: round(x, 4) for k, x in points(M, v)[1].items()})
P = {q: recompose(tw[q], s[q]) for q in T}; print("true win        ", round(points(P, v)[0], 3), {k: round(x, 4) for k, x in points(P, v)[1].items()})
P = {q: recompose(w[q], ts[q]) for q in T}; print("true size       ", round(points(P, v)[0], 3), {k: round(x, 4) for k, x in points(P, v)[1].items()})
