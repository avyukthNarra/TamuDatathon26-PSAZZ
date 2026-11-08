"""Exact posterior over the binary payout-size mode: fixed vs switching (HMM) and size-of-switch hazard."""
import csv, sys, collections, numpy as np
from common import *
v = load("validation"); T = v["targets"]
def readp(f): return {r["query_id"]: np.array([float(r[c]) for c in PCOLS]) for r in csv.DictReader(open(f))}
M = readp(sys.argv[1])
# the 6 mode distributions, from validation targets (2 per type)
typ = {q["query_id"]: q["machine_type"] for q in v["Q"]}
modes = collections.defaultdict(list)
for q, t in T.items():
    s = tuple(np.round(t[1:] / (1 - t[0]), 3))
    modes[typ[q]].append(s)
MODES = {}
for ty, L in modes.items():
    c = collections.Counter(tuple(np.round(x, 2)) for x in L)
    keys = [k for k, _ in c.most_common(2)]
    MODES[ty] = np.array([np.mean([x for x in L if tuple(np.round(x, 2)) == k], axis=0) for k in keys])
    print(ty, MODES[ty].round(4))
IDX = {1: 0, 2: 1, 5: 2, 20: 3}
def true_mode(q):
    s = T[q][1:] / (1 - T[q][0]); return int(np.argmin(((MODES[typ[q]] - s) ** 2).sum(1)))

def posterior(hist, cutoff, E, hazard, prior=0.5):
    """Forward filter over 2 modes. hazard = per-tick switch prob (applied per elapsed tick, also over gaps)."""
    p = np.array([1 - prior, prior]); last = None
    for t in sorted(k for k in hist if k < cutoff):
        if last is not None and hazard > 0:
            n = t - last; stay = 0.5 + 0.5 * (1 - 2 * hazard) ** n
            p = np.array([p[0] * stay + p[1] * (1 - stay), p[1] * stay + p[0] * (1 - stay)])
        last = t
        y = hist[t]
        if y > 0:
            p = p * E[:, IDX[y]]; p /= p.sum()
    n = cutoff - last
    stay = 0.5 + 0.5 * (1 - 2 * hazard) ** n
    return np.array([p[0] * stay + p[1] * (1 - stay), p[1] * stay + p[0] * (1 - stay)])

tw = {q: 1 - T[q][0] for q in T}
mw = {q: 1 - M[q][0] for q in M}
for hz in [0, 0.002, 0.005, 0.01, 0.02, 0.04]:
    P1, P2, acc = {}, {}, []
    for q in v["Q"]:
        qid = q["query_id"]; E = MODES[q["machine_type"]]
        post = posterior(v["H"][q["machine_id"]], int(q["cutoff"]), E, hz)
        size = post @ E
        acc.append(post[true_mode(qid)])
        P1[qid] = np.concatenate([[1 - tw[qid]], tw[qid] * size])
        P2[qid] = np.concatenate([[1 - mw[qid]], mw[qid] * size])
    print(f"hazard {hz}: mean post on true mode {np.mean(acc):.3f} | true win + this size {points(P1, v)[0]:.3f} | mixer6 win + this size {points(P2, v)[0]:.3f}")
