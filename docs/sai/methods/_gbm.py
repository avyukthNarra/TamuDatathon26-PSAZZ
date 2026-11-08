"""Shared GBM-style (Gaussian random walk on log-odds) grid filters for Q1/Q2/Q3. Observations: win flag = payout>0 at observed ticks < cutoff
(NaN = unobserved, propagated over). Everything (prior mean, sigma, pairs) is learned from TRAIN histories or the machine's own history."""
import sys, itertools
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "scripts"))
from harness import load_split, cached

K = 61
X = np.linspace(-6, 6, K)
SIG = np.array([0.0, 0.03, 0.06, 0.12, 0.25, 0.5])           # per-tick process sd on log-odds
PX = 1 / (1 + np.exp(-X))
_T = {}


def trans(sig, dt):
    key = (round(float(sig), 5), int(dt))
    m = _T.get(key)
    if m is None:
        s = sig * np.sqrt(dt)
        if s < 1e-9:
            m = np.eye(K)
        else:
            m = np.exp(-0.5 * ((X[:, None] - X[None, :]) / s) ** 2)
            m /= m.sum(0, keepdims=True)
        _T[key] = m
    return m


def prior(mu, sd=1.5):
    p = np.exp(-0.5 * ((X - mu) / sd) ** 2)
    return p / p.sum()


def obs_list(g):
    t = np.flatnonzero(~np.isnan(g))
    return t, (g[t] > 0).astype(float)


def run_filter(t, w, cutoff, sig, mu, sig_fn=None):
    """Return (loglik_prequential, predictive pw at cutoff). sig scalar, or sig_fn(history_w)->sigma for the step."""
    post = prior(mu); ll = 0.0; last = None
    for j in range(len(t)):
        dt = 1 if last is None else int(t[j] - last)
        s = sig if sig_fn is None else sig_fn(j)
        if last is not None:
            post = trans(s, dt) @ post
        p = float(post @ PX)
        ll += np.log(p if w[j] > 0 else 1 - p)
        post = post * (PX if w[j] > 0 else 1 - PX); post /= post.sum()
        last = t[j]
    dt = 1 if last is None else int(cutoff - last)
    s = sig if sig_fn is None else sig_fn(len(t))
    post = trans(s, dt) @ post
    return ll, float(post @ PX)


def global_mu(train):
    r = np.nanmean(np.where(np.isnan(train.grid), np.nan, train.grid > 0))
    return float(np.log(r / (1 - r)))


def multi_sigma(split, mu, rows=None):
    """per machine: LL[i,k], PW[i,k] over SIG"""
    rows = range(split.n) if rows is None else rows
    LL = np.zeros((split.n, len(SIG))); PW = np.zeros((split.n, len(SIG)))
    for i in rows:
        t, w = obs_list(split.grid[i])
        for k, s in enumerate(SIG):
            LL[i, k], PW[i, k] = run_filter(t, w, int(split.cutoff[i]), s, mu)
    return LL, PW


# ---- online volatility features (same recursions as volfeat.py, per observed tick) ----
def online_vol(w, kind):
    n = len(w); f = np.zeros(n + 1)
    lw = 0.3; m = w[0] if n else 0.3; h = lw * (1 - lw); a = 1 - 0.5 ** (1 / 16); mm = w[0] if n else 0.3; s2 = 0.0
    f[0] = lw * (1 - lw)
    run = 0.0
    for j in range(n):
        run += w[j]; lw = run / (j + 1); base = max(lw * (1 - lw), 1e-2)
        z = w[j]
        if kind == "garch":
            e = z - m
            h = 0.05 * lw * (1 - lw) + 0.1 * e * e + 0.85 * h
            m += 0.1 * e; v = h
        else:   # ewv16 of volfeat
            d = z - mm; mm += a * d; s2 = (1 - a) * (s2 + a * d * d); v = s2
        f[j + 1] = np.log(max(v, 1e-3) / base)      # excess volatility vs Bernoulli noise
    return f
