"""Stacking research: learn WHEN to trust each base forecaster (run from mainFiles/).

  python research/stack.py --cache runs/research-cache

Methods are fitted on validation TRUE probabilities and checked two honest ways:
  cross-fit   fit on one validation half, score the other (official contest metric, sizes
              from the pair mixer), both halves reported
  train       fit on all validation, score train labels it never saw (out-of-fold base
              forecasts; dR vs the 56.51 model at the cutoff and over the last-10-tick window)
"""
import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slot_forecasting.experiments import episode_folds, subset_score  # noqa: E402
from slot_forecasting.io import load_inputs, read_submission, read_targets  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--cache", default="runs/research-cache")
parser.add_argument("--methods", default="all")
args = parser.parse_args()
cache = Path(args.cache)
pub = Path("public")

VAL = json.load(open(cache / "validation.json"))
TRAIN = json.load(open(cache / "train.json"))
vt = read_targets(pub / "validation/targets.csv")
ref = read_submission(pub / "validation/reference.csv")
vcond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(pub / "validation/conditions.csv"))}
HALVES = episode_folds(vt, vcond)
ALL_EP = {t.episode_id for t in vt}
# frozen reference error per condition (metric denominator), from validation
_be = defaultdict(lambda: defaultdict(list))
for t in vt:
    _be[vcond[t.query_id]][t.episode_id].append(sum((a - b) ** 2 for a, b in zip(ref[t.query_id], t.probabilities)))
BASE_ERR = {c: sum(sum(v) / len(v) for v in e.values()) / len(e) for c, e in _be.items()}


# ---------------- situation features (history before the row's cutoff only) ----------------
def load_hist(split):
    h = defaultdict(dict)
    for p in load_inputs(pub / split)[1]:
        h[(p.episode_id, p.machine_id)][p.tick] = p.payout
    return h


HIST = {"validation": load_hist("validation"), "train": load_hist("train")}


def rates(seq, cutoff):
    ticks = sorted(t for t in seq if t < cutoff)
    wins = [1.0 if seq[t] > 0 else 0.0 for t in ticks]
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.5  # noqa: E731
    recent, old = wins[-40:], wins[:-40]
    return len(ticks), mean(wins[-10:]), mean(recent), mean(old) if old else mean(recent)


def add_features(rows, split):
    by_episode = defaultdict(list)
    for r in rows:
        by_episode[(r["episode"], r["cutoff"])].append(r)
    hist = HIST[split]
    for r in rows:
        n, r10, r40, rold = rates(hist[(r["episode"], r["machine"])], r["cutoff"])
        r["n"], r["short"], r["r10"], r["r40"], r["shift"] = n, 1.0 if n < 120 else 0.0, r10, r40, abs(r40 - rold)
    for r in rows:  # episode-wide shift among the OTHER machines at the same cutoff (shared change)
        others = [o["shift"] for o in by_episode[(r["episode"], r["cutoff"])] if o is not r]
        r["ep_shift"] = sum(others) / len(others) if others else 0.0
    for r in rows:
        wins = (1 - r["mixer"][0], r["hmm"], r["regime"])
        r["wins"] = wins
        r["spread"] = max(wins) - min(wins)


add_features(VAL, "validation")
add_features(TRAIN, "train")


def gate_x(r):
    """Inputs of the gate (what situation are we in)."""
    return [1.0, r["short"], r["shift"], r["ep_shift"], r["spread"], r["r10"] - r["r40"],
            1.0 if r["type"] == "jackpot" else 0.0, 1.0 if r["type"] == "steady" else 0.0, float(r["signal"])]


# ---------------- methods: f(train_rows) -> predict(row) -> p_win ----------------
def fixed(weights):
    def fit(_rows):
        return lambda r: sum(w * p for w, p in zip(weights, r["wins"]))
    return fit


def logit_pool(weights):
    """Fixed log-odds pooling: p = sigmoid(sum_k w_k logit p_k)."""
    def fit(_rows):
        def predict(r):
            z = sum(w * logit(p) for w, p in zip(weights, r["wins"]))
            return 1 / (1 + math.exp(-z))
        return predict
    return fit


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def target_weight(r):
    return 1.0 / BASE_ERR[r["condition"]]


def gradient_fit(rows, predict, grad, n_params, l2, steps=300, lr=0.05, init=None):
    """Minimise metric-weighted squared error of p_win vs the true p_win (Adam, full batch)."""
    theta = list(init) if init else [0.0] * n_params
    w = [target_weight(r) for r in rows]
    wsum = sum(w)
    ys = [1 - r["truth"][0] for r in rows]
    m, v = [0.0] * n_params, [0.0] * n_params
    for step in range(1, steps + 1):
        g = [l2 * t for t in theta]
        for r, wi, y in zip(rows, w, ys):
            p, dp = predict(theta, r), grad(theta, r)
            c = wi / wsum * 2 * (p - y)
            for j in range(n_params):
                g[j] += c * dp[j]
        for j in range(n_params):
            m[j] = 0.9 * m[j] + 0.1 * g[j]
            v[j] = 0.999 * v[j] + 0.001 * g[j] * g[j]
            theta[j] -= lr * (m[j] / (1 - 0.9 ** step)) / (math.sqrt(v[j] / (1 - 0.999 ** step)) + 1e-8)
    return theta


def gated(l2=1e-3, features=gate_x):
    """Mixture of experts: p = sum_k softmax(W_k . x)_k p_k, the gate learns who to trust when."""
    F = len(features(VAL[0]))

    def weights(theta, r):
        x = features(r)
        z = [sum(theta[k * F + j] * x[j] for j in range(F)) for k in range(3)]
        top = max(z)
        e = [math.exp(v - top) for v in z]
        s = sum(e)
        return [v / s for v in e], x

    def predict(theta, r):
        pi, _ = weights(theta, r)
        return sum(a * b for a, b in zip(pi, r["wins"]))

    def grad(theta, r):
        pi, x = weights(theta, r)
        p = sum(a * b for a, b in zip(pi, r["wins"]))
        g = [0.0] * (3 * F)
        for k in range(3):
            d = pi[k] * (r["wins"][k] - p)
            for j in range(F):
                g[k * F + j] = d * x[j]
        return g

    def fit(rows):
        init = [0.0] * (3 * F)
        init[0], init[F], init[2 * F] = math.log(0.7), math.log(0.3), math.log(1e-3)  # start at the 56.51 blend
        theta = gradient_fit(rows, predict, grad, 3 * F, l2, init=init)
        return lambda r: predict(theta, r)
    return fit


def logistic(l2=1e-3, interact=True):
    """p = sigmoid(b0 + sum_k a_k logit p_k [+ situation x logit interactions]): recalibrates as well as blends."""
    def x(r):
        ls = [logit(p) for p in r["wins"]]
        base = [1.0] + ls
        if interact:
            g = gate_x(r)[1:]
            base += [gi * l for gi in g for l in ls] + g
        return base
    n = len(x(VAL[0]))

    def predict(theta, r):
        z = sum(t * v for t, v in zip(theta, x(r)))
        return 1 / (1 + math.exp(-max(min(z, 30), -30)))

    def grad(theta, r):
        p = predict(theta, r)
        return [p * (1 - p) * v for v in x(r)]

    def fit(rows):
        init = [0.0] * n
        init[1], init[2] = 0.7, 0.3
        theta = gradient_fit(rows, predict, grad, n, l2, init=init)
        return lambda r: predict(theta, r)
    return fit


METHODS = {
    "56.51 model (0.7 mixer + 0.3 HMM)": fixed((0.7, 0.3, 0.0)),
    "equal 3-way": fixed((1 / 3, 1 / 3, 1 / 3)),
    "mixer alone": fixed((1.0, 0.0, 0.0)),
    "log-odds pool mixer 0.7 / HMM 0.3": logit_pool((0.7, 0.3, 0.0)),
    "log-odds pool mixer 0.6 / HMM 0.4": logit_pool((0.6, 0.4, 0.0)),
    "log-odds pool mixer 0.5 / HMM 0.5": logit_pool((0.5, 0.5, 0.0)),
    "log-odds pool mixer 0.4 / HMM 0.6": logit_pool((0.4, 0.6, 0.0)),
    "log-odds pool 1/3 each": logit_pool((1 / 3, 1 / 3, 1 / 3)),
    "HMM alone": fixed((0.0, 1.0, 0.0)),
    "regime alone": fixed((0.0, 0.0, 1.0)),
    "mixer 0.5 + regime 0.5": fixed((0.5, 0.0, 0.5)),
    "mixer 0.5 + HMM 0.5": fixed((0.5, 0.5, 0.0)),
    "gated mixture of experts": gated(),
    "gated, stronger L2": gated(l2=1e-2),
    "gated, L2 1e-1": gated(l2=1e-1),
    "logistic stack (no interactions)": logistic(interact=False),
    "logistic stack + situation interactions": logistic(),
    "logistic stack + interactions, L2 1e-2": logistic(l2=1e-2),
}


# ---------------- evaluation ----------------
def as_probs(r, pw):
    m = r["mixer"]
    ow = 1 - m[0]
    raw = [max(x, 1e-4) for x in [1 - pw] + [pw * x / ow for x in m[1:]]]
    s = sum(raw)
    return [x / s for x in raw]


def crossfit(method):
    preds = {}
    for scored, other in zip(HALVES, reversed(HALVES)):
        model = method([r for r in VAL if r["episode"] in other])
        for r in VAL:
            if r["episode"] in scored:
                preds[r["qid"]] = as_probs(r, model(r))
    return [subset_score(preds, vt, vcond, ref, e, 60.0)["technical_points"] for e in (ALL_EP, *HALVES)]


def brier(p, k):
    return sum((x - (1.0 if i == k else 0.0)) ** 2 for i, x in enumerate(p))


def train_check(method, base_method):
    model, base = method(VAL), base_method(VAL)
    out = {}
    for which in ("cutoff", "window"):
        diff = defaultdict(lambda: defaultdict(list))
        per_q = defaultdict(list)
        for r in TRAIN:
            if which == "cutoff" and r["back"] != 0:
                continue
            d = brier(as_probs(r, model(r)), r["outcome"]) - brier(as_probs(r, base(r)), r["outcome"])
            per_q[(r["condition"], r["episode"], r["qid"])].append(d)
        for (c, e, _), v in per_q.items():
            diff[c][e].append(sum(v) / len(v))

        def dr(sel=None):
            vals = []
            for c, eps in diff.items():
                chosen = eps if sel is None else sel[c]
                vals.append(sum(sum(eps[x]) / len(eps[x]) for x in chosen) / len(chosen) / BASE_ERR[c])
            return sum(vals) / len(vals)
        rng = random.Random(0)
        boot = sorted(dr({c: [rng.choice(list(e)) for _ in e] for c, e in diff.items()}) for _ in range(1000))
        out[which] = (dr(), boot[25], boot[974])
    return out


base = METHODS["56.51 model (0.7 mixer + 0.3 HMM)"]
for name, method in (METHODS.items() if __name__ == "__main__" else ()):
    if args.methods != "all" and not any(s in name for s in args.methods.split(",")):
        continue
    cf = crossfit(method)
    tc = train_check(method, base)
    print(f"{name:42s} cross-fit {cf[0]:.2f} ({cf[1]:.2f} / {cf[2]:.2f}) | train dR cutoff "
          f"{tc['cutoff'][0]:+.4f} [{tc['cutoff'][1]:+.4f},{tc['cutoff'][2]:+.4f}] window "
          f"{tc['window'][0]:+.4f} [{tc['window'][1]:+.4f},{tc['window'][2]:+.4f}]", flush=True)
