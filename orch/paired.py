"""Paired episode comparison of two forecasts (Diebold-Mariano style), in contest points.

Per episode, the difference in mean squared error vs truth; contest weighting (each condition's errors
divided by its reference error, conditions equally weighted); bootstrap resamples episodes within
condition with BOTH forecasts kept together, so shared episode difficulty cancels.
Usage: python paired.py base.csv candidate.csv [KIT_PUBLIC_DIR]
(no kit argument: main-round public/; with a kit: its validation answers and metadata max_points.)
"""
import csv, json, sys, collections, numpy as np
from pathlib import Path
from common import *


def read_preds(path):
    return {r["query_id"]: np.array([float(r[c]) for c in PCOLS]) for r in csv.DictReader(open(path))}


def paired(a, b, kit=None, max_points=60, reps=4000, seed=1):
    """a, b: {query_id: 5-vector} forecasts on validation. Returns (delta, CI, P(delta>0)) for b - a."""
    v = load("validation", pub=Path(kit) if kit else PUB)
    T, R, C = v["targets"], v["reference"], v["cond"]
    ep = {q["query_id"]: q["episode_id"] for q in v["Q"]}
    E = collections.defaultdict(lambda: collections.defaultdict(lambda: [[], [], []]))
    for q in T:
        cell = E[C[q]][ep[q]]
        cell[0].append(((a[q] - T[q]) ** 2).sum()); cell[1].append(((b[q] - T[q]) ** 2).sum()); cell[2].append(((R[q] - T[q]) ** 2).sum())
    conds = sorted(E); arr = {c: np.array([[np.mean(x) for x in E[c][e]] for e in sorted(E[c])]) for c in conds}
    def pts(sample):
        ra = np.mean([s[:, 0].mean() / s[:, 2].mean() for s in sample]); rb = np.mean([s[:, 1].mean() / s[:, 2].mean() for s in sample])
        return max_points / (1 + rb) - max_points / (1 + ra)
    d0 = pts([arr[c] for c in conds]); rng = np.random.default_rng(seed)
    boots = np.array([pts([arr[c][rng.integers(0, len(arr[c]), len(arr[c]))] for c in conds]) for _ in range(reps)])
    return d0, np.quantile(boots, [.025, .975]), (boots > 0).mean()


if __name__ == "__main__":
    kit = sys.argv[3] if len(sys.argv) > 3 else None
    mp = json.loads((Path(kit) / "metadata.json").read_text()).get("max_points", 40) if kit else 60
    d, ci, pr = paired(read_preds(sys.argv[1]), read_preds(sys.argv[2]), kit=kit, max_points=mp)
    print(f"candidate - base: {d:+.3f} points, paired 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}], P(gain>0) {pr:.3f}")
