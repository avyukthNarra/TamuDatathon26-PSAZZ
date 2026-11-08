"""Paired episode comparison of two forecasts (Diebold-Mariano style), in contest points.

Per episode, the difference in mean squared error vs truth; contest weighting (each condition's errors
divided by its reference error, conditions equally weighted); bootstrap resamples episodes within
condition with BOTH forecasts kept together, so shared episode difficulty cancels.
Usage: python paired.py base.csv candidate.csv
"""
import csv, sys, collections, numpy as np
from common import *
v = load("validation"); T, R, C = v["targets"], v["reference"], v["cond"]
ep = {q["query_id"]: q["episode_id"] for q in v["Q"]}
rd = lambda f: {r["query_id"]: np.array([float(r[c]) for c in PCOLS]) for r in csv.DictReader(open(f))}
def paired(a, b, reps=4000, seed=1):
    E = collections.defaultdict(lambda: collections.defaultdict(lambda: [[], [], []]))
    for q in T:
        cell = E[C[q]][ep[q]]
        cell[0].append(((a[q] - T[q]) ** 2).sum()); cell[1].append(((b[q] - T[q]) ** 2).sum()); cell[2].append(((R[q] - T[q]) ** 2).sum())
    conds = sorted(E); arr = {c: np.array([[np.mean(x) for x in E[c][e]] for e in sorted(E[c])]) for c in conds}
    def pts(sample):
        ra = np.mean([s[:, 0].mean() / s[:, 2].mean() for s in sample]); rb = np.mean([s[:, 1].mean() / s[:, 2].mean() for s in sample])
        return 60 / (1 + rb) - 60 / (1 + ra)
    d0 = pts([arr[c] for c in conds]); rng = np.random.default_rng(seed)
    boots = np.array([pts([arr[c][rng.integers(0, len(arr[c]), len(arr[c]))] for c in conds]) for _ in range(reps)])
    return d0, np.quantile(boots, [.025, .975]), (boots > 0).mean()
if __name__ == "__main__":
    d, ci, pr = paired(rd(sys.argv[1]), rd(sys.argv[2]))
    print(f"candidate - base: {d:+.3f} points, paired 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}], P(gain>0) {pr:.3f}")
