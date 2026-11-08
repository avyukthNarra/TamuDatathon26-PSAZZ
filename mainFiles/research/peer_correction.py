"""Peer correction: logit p = logit(f) + sum_j b_j x_j, fitted on TRAIN outcomes only (run from mainFiles/).

x = centred peer / own features; f = the 56.54 model's win chance. Validation is never used for
fitting: its score (all, halves) is an untouched check. Train skill: 4 episode folds.
"""
import math, sys, random
from collections import defaultdict
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
sys.argv = sys.argv[:1] + ["--methods", "none"]
import stack
import residuals as R  # computes r["f"], r["peers"], r["seq"] on stack-like rows

VAL, TRAIN = R.VAL, R.TRAIN
for r in TRAIN: r["won"] = 1.0 if r["outcome"] else 0.0

def share(r, lags):
    v = [s[r["cutoff"] - k] > 0 for k in lags for s in r["peers"] if r["cutoff"] - k in s]
    return sum(v) / len(v) if v else None

def own_rate(r, a, b):
    ts = sorted(t for t in r["seq"] if t < r["cutoff"])
    seg = ts[-b:-a] if a else ts[-b:]
    return sum(r["seq"][t] > 0 for t in seg) / len(seg) if seg else None

FEATURES = {
    "peers 1-5": lambda r: share(r, range(1, 6)),
    "peers 1-8": lambda r: share(r, range(1, 9)),
    "peers 6-20": lambda r: share(r, range(6, 21)),
    "own last 10": lambda r: own_rate(r, 0, 10),
    "own last 40": lambda r: own_rate(r, 0, 40),
}
for rows in (VAL, TRAIN):
    for r in rows:
        r["x"] = {k: f(r) for k, f in FEATURES.items()}
MEANS = {k: sum(r["x"][k] for r in TRAIN if r["x"][k] is not None) / sum(1 for r in TRAIN if r["x"][k] is not None) for k in FEATURES}

def design(r, names):
    return [(r["x"][k] if r["x"][k] is not None else MEANS[k]) - MEANS[k] for k in names]

def fit(rows, names, l2=1e-3, steps=60):
    """Newton steps on log-loss with offset logit(f) (weights 1/condition reference error)."""
    b = [0.0] * len(names)
    data = [(design(r, names), stack.logit(r["f"]), r["won"], 1 / stack.BASE_ERR[r["condition"]]) for r in rows]
    for _ in range(steps):
        g = [l2 * x for x in b]; H = [[l2 * (i == j) for j in range(len(b))] for i in range(len(b))]
        W = 0.0
        for x, off, y, w in data:
            p = 1 / (1 + math.exp(-(off + sum(bi * xi for bi, xi in zip(b, x)))))
            W += w
            for i in range(len(b)):
                g[i] += w * (p - y) * x[i]
                for j in range(len(b)): H[i][j] += w * p * (1 - p) * x[i] * x[j]
        n = len(b)  # solve H d = g (small, Gaussian elimination)
        A = [H[i][:] + [g[i]] for i in range(n)]
        for c in range(n):
            piv = max(range(c, n), key=lambda i: abs(A[i][c])); A[c], A[piv] = A[piv], A[c]
            for i in range(n):
                if i != c:
                    f = A[i][c] / A[c][c]; A[i] = [a - f * bb for a, bb in zip(A[i], A[c])]
        b = [bi - A[i][n] / A[i][i] for i, bi in enumerate(b)]
    return b

def corrected(r, names, b):
    return 1 / (1 + math.exp(-(stack.logit(r["f"]) + sum(bi * xi for bi, xi in zip(b, design(r, names))))))

def val_score(names, b):
    preds = {r["qid"]: stack.as_probs(r, corrected(r, names, b)) for r in VAL}
    return [stack.subset_score(preds, stack.vt, stack.vcond, stack.ref, e, 60.0) for e in (stack.ALL_EP, *stack.HALVES)]

def train_cv(names):
    eps = sorted({r["episode"] for r in TRAIN}); fold = {e: i % 4 for i, e in enumerate(eps)}
    diff = defaultdict(lambda: defaultdict(list))
    for which in ("cutoff", "window"):
        per_q = defaultdict(list)
        for k in range(4):
            b = fit([r for r in TRAIN if fold[r["episode"]] != k], names)
            for r in TRAIN:
                if fold[r["episode"]] != k or (which == "cutoff" and r["back"]): continue
                d = stack.brier(stack.as_probs(r, corrected(r, names, b)), r["outcome"]) - stack.brier(stack.as_probs(r, r["f"]), r["outcome"])
                per_q[(r["condition"], r["episode"], r["qid"])].append(d)
        dd = defaultdict(lambda: defaultdict(list))
        for (c, e, _), v in per_q.items(): dd[c][e].append(sum(v) / len(v))
        def score(sel=None):
            return sum(sum(sum(dd[c][x]) / len(dd[c][x]) for x in (dd[c] if sel is None else sel[c])) / len(dd[c] if sel is None else sel[c]) / stack.BASE_ERR[c] for c in dd) / len(dd)
        rng = random.Random(0)
        boot = sorted(score({c: [rng.choice(list(e)) for _ in e] for c, e in dd.items()}) for _ in range(500))
        diff[which] = (score(), boot[12], boot[487])
    return diff

base = [stack.subset_score({r["qid"]: stack.as_probs(r, r["f"]) for r in VAL}, stack.vt, stack.vcond, stack.ref, e, 60.0)["technical_points"] for e in (stack.ALL_EP, *stack.HALVES)]
print(f"56.54 model: validation {base[0]:.2f} ({base[1]:.2f} / {base[2]:.2f})")
for names in (["peers 1-5"], ["peers 1-8"], ["peers 1-5", "peers 6-20"], ["peers 1-5", "own last 10", "own last 40"], ["peers 1-8", "peers 6-20", "own last 10", "own last 40"]):
    b = fit(TRAIN, names)
    s = val_score(names, b); cv = train_cv(names)
    bc = s[0]["by_condition"]
    print(f"{' + '.join(names):45s} coef {[round(x, 2) for x in b]} | validation {s[0]['technical_points']:.2f} ({s[1]['technical_points']:.2f} / {s[2]['technical_points']:.2f}) "
          + " ".join(f"{c[:5]} {bc[c]['loss_ratio']:.3f}" for c in sorted(bc))
          + f" | train CV cutoff {cv['cutoff'][0]:+.4f} [{cv['cutoff'][1]:+.4f},{cv['cutoff'][2]:+.4f}] window {cv['window'][0]:+.4f} [{cv['window'][1]:+.4f},{cv['window'][2]:+.4f}]", flush=True)
