"""Shared W1 (lag structure) engine: B3-style per-machine win model with switchable pieces.
Not a method itself; every methods/L*.py imports it. numpy only. Leakage: only ticks < cutoff.

predict(split, cfg) -> (n,5). Win probability from fit_machine; payout-size split is B3's
(recency-weighted, type-shrunk shares, HL 64, prior 8), cached once per split.
cfg keys (defaults reproduce B3 when lags/pairs are set to B3's):
  lags, pairs, triples, lam, hl_grid, holdout, coding ('flag','flag_big','flag_log','onehot','log'),
  big_thr, link ('lin','logit'), pool (None,'global','type','signal','type_signal','type_signal_lin'),
  pool_lam, sel (None or dict(thr=z, cand_lags, cand_pairs)), min_row_tick, gappy
"""
import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from harness import cached  # noqa: E402

PAYV = (1, 2, 5, 20)
B3 = dict(lags=(1, 2, 3, 4, 5, 7), pairs=((1, 3), (2, 5), (4, 7)), triples=(), lam=20.0,
          hl_grid=(1e9, 128, 64, 32, 16), holdout=40, coding="flag", big_thr=5, link="lin", pool=None,
          pool_lam=20.0, sel=None, min_row_tick=8, gappy=0.6)


def C(**kw):
    c = dict(B3)
    c.update(kw)
    return c


# ---------------------------------------------------------------- features
def _lag(a, tt, l):
    idx = tt - l
    return np.where(idx >= 0, a[np.clip(idx, 0, len(a) - 1)], 0.0)


def machine_arrays(g, cutoff, cfg):
    """Per-tick arrays of length cutoff+1 (index = tick; cutoff slot = unobserved)."""
    obs = ~np.isnan(g[:cutoff])
    pay = np.where(obs, g[:cutoff], 0.0)
    ext = lambda x: np.concatenate([x, [0.0]])
    f = ext(np.where(obs, np.where(pay > 0, 1.0, -1.0), 0.0))
    win = ext(np.where(obs, (pay > 0).astype(float), 0.0))
    miss = ext((~obs).astype(float))
    m = obs.sum()
    big = np.where(obs, (pay >= cfg["big_thr"]).astype(float), 0.0)
    big = ext(np.where(obs, big - (big.sum() / max(m, 1)), 0.0))
    lg = np.where(obs, np.log1p(pay), 0.0)
    lg = ext(np.where(obs, lg - lg.sum() / max(m, 1), 0.0))
    return dict(f=f, win=win, miss=miss, big=big, lg=lg)


def lag_cols(A, tt, l, coding):
    f = _lag(A["f"], tt, l)
    if coding == "flag":
        return [f]
    if coding == "flag_big":
        return [f, _lag(A["big"], tt, l)]
    if coding == "flag_log":
        return [f, _lag(A["lg"], tt, l)]
    if coding == "log":
        return [_lag(A["lg"], tt, l)]
    if coding == "onehot":
        return [_lag(A["win"], tt, l), _lag(A["miss"], tt, l)]
    raise ValueError(coding)


def design(A, tt, cfg, lags=None, pairs=None, triples=None):
    lags = cfg["lags"] if lags is None else lags
    pairs = cfg["pairs"] if pairs is None else pairs
    triples = cfg["triples"] if triples is None else triples
    cols = [np.ones(len(tt))]
    for l in lags:
        cols += lag_cols(A, tt, l, cfg["coding"])
    pc = cfg.get("pair_coding", "flag")
    for a, b in pairs:
        cols.append(_lag(A["f"], tt, a) * _lag(A["f"], tt, b))
        if pc == "flag+big":
            cols.append(_lag(A["big"], tt, a) * _lag(A["big"], tt, b))
        elif pc == "flag+log":
            cols.append(_lag(A["lg"], tt, a) * _lag(A["lg"], tt, b))
    for a, b, c in triples:
        cols.append(_lag(A["f"], tt, a) * _lag(A["f"], tt, b) * _lag(A["f"], tt, c))
    return np.column_stack(cols)


# ---------------------------------------------------------------- fitting
def ridge(X, y, ticks, ref, hl, lam, mu=None, link="lin", pen_mask=None):
    w = 0.5 ** ((ref - 1 - ticks) / hl)
    k = X.shape[1]
    pen = np.eye(k) * lam
    if pen_mask is not None:
        pen = np.diag(pen_mask * lam)
    pen[0, 0] = 1e-6
    m0 = np.zeros(k) if mu is None else np.asarray(mu, float).copy()
    m0[0] = 0.0
    if link == "lin":
        return np.linalg.solve(X.T @ (w[:, None] * X) + pen, X.T @ (w * y) + pen @ m0)
    # penalised weighted logistic, Newton, fixed 12 iterations (deterministic)
    pbar = float(np.clip((w * y).sum() / w.sum(), 0.02, 0.98))
    b = np.zeros(k)
    b[0] = np.log(pbar / (1 - pbar))
    for _ in range(12):
        eta = np.clip(X @ b, -12, 12)
        p = 1 / (1 + np.exp(-eta))
        gr = X.T @ (w * (y - p)) - pen @ (b - m0)
        H = X.T @ ((w * p * (1 - p))[:, None] * X) + pen
        b = b + np.linalg.solve(H, gr)
    return b


def predict_lin(x, b, link):
    z = x @ b
    return np.clip(z, 0, 1) if link == "lin" else 1 / (1 + np.exp(-np.clip(z, -12, 12)))


def select_cols(X, y, thr, k=None):
    """Univariate z screen of non-intercept columns on this machine's own training rows."""
    n = len(y)
    keep = [0]
    yc = y - y.mean()
    sy = yc.std()
    zs = []
    for j in range(1, X.shape[1]):
        x = X[:, j]
        sx = x.std()
        if sx < 1e-9 or sy < 1e-9:
            zs.append(0.0)
            continue
        zs.append(abs(np.corrcoef(x, y)[0, 1]) * np.sqrt(n - 2))
    zs = np.array(zs)
    if k is not None:
        order = np.argsort(-zs, kind="stable")[:k]
        keep += [int(j) + 1 for j in order]
    else:
        keep += [j + 1 for j in range(len(zs)) if zs[j] >= thr]
    return sorted(keep)


def fit_machine(g, cutoff, cfg, mu=None):
    """Returns (pw, beta, hl). g = grid row (NaN unobserved). mu = prior mean for beta (pooling)."""
    A = machine_arrays(g, cutoff, cfg)
    obs_t = np.flatnonzero(~np.isnan(g[:cutoff]))
    ticks = obs_t[obs_t >= cfg["min_row_tick"]]
    sel = cfg["sel"]
    if sel:
        X = design(A, ticks, cfg, sel["cand_lags"], sel["cand_pairs"], sel.get("cand_triples", ()))
        xq = design(A, np.array([cutoff]), cfg, sel["cand_lags"], sel["cand_pairs"], sel.get("cand_triples", ()))[0]
    else:
        X = design(A, ticks, cfg)
        xq = design(A, np.array([cutoff]), cfg)[0]
    y = (g[ticks] > 0).astype(float)
    link, lam = cfg["link"], cfg["lam"]
    keep = None
    if sel:
        keep = select_cols(X, y, sel.get("thr", 2.0), sel.get("k"))
        X, xq = X[:, keep], xq[keep]
    mu_k = None if mu is None else (np.asarray(mu)[keep] if keep is not None else mu)
    hl = 1e9
    if len(obs_t) >= cfg["gappy"] * cutoff:
        cut = cutoff - cfg["holdout"]
        tr, te = ticks < cut, ticks >= cut
        errs = []
        for c in cfg["hl_grid"]:
            if tr.sum() < 30 or te.sum() < 10:
                errs.append(0.0 if c == 1e9 else 1.0)
                continue
            b = ridge(X[tr], y[tr], ticks[tr], cut, c, lam, mu_k, link)
            errs.append(np.mean((predict_lin(X[te], b, link) - y[te]) ** 2))
        hl = cfg["hl_grid"][int(np.argmin(errs))]
    b = ridge(X, y, ticks, cutoff, hl, lam, mu_k, link)
    pw = float(np.clip(predict_lin(xq, b, link), 0.005, 0.995))
    return pw, b, hl


# ---------------------------------------------------------------- sizes (B3)
def size_shares(split, hl=64.0, prior=8.0):
    def build():
        n = split.n
        cnt = np.zeros((n, 4))
        for i in range(n):
            g = split.grid[i, :split.cutoff[i]]
            for k, p in enumerate(PAYV):
                cnt[i, k] = ((g == p) * 0.5 ** ((split.cutoff[i] - 1 - np.arange(len(g))) / hl)).sum()
        raw = np.zeros((n, 4))
        for i in range(n):
            g = split.grid[i, :split.cutoff[i]]
            for k, p in enumerate(PAYV):
                raw[i, k] = (g == p).sum()
        ts = {}
        for ty in sorted(set(split.mtype)):
            c = raw[split.mtype == ty].sum(0)
            ts[ty] = c / c.sum()
        out = np.zeros((n, 4))
        for i in range(n):
            c = prior * ts[split.mtype[i]] + cnt[i]
            out[i] = c / c.sum()
        return out
    key = f"w1_sizes_{split.name}_{split.n}_{int(split.cutoff.sum())}_{int(np.nansum(split.grid))}"
    return cached(key, build)


def finish(pw, split):
    sh = size_shares(split)
    P = np.column_stack([1 - pw, pw[:, None] * sh])
    return P / P.sum(1, keepdims=True)


# ---------------------------------------------------------------- pooling / predict
def _cells(split, pool):
    if pool == "global":
        return [0] * split.n
    if pool == "type":
        return [str(m) for m in split.mtype]
    if pool == "signal":
        return [int(s) for s in split.signal]
    return [(str(m), int(s)) for m, s in zip(split.mtype, split.signal)]


def predict_pw(split, cfg, return_extra=False):
    n = split.n
    res = [fit_machine(split.grid[i], int(split.cutoff[i]), cfg) for i in range(n)]
    pool = cfg["pool"]
    if pool and not cfg["sel"]:
        B = np.array([r[1] for r in res])
        if pool == "type_signal_lin":
            mus = np.zeros_like(B)
            for ty in sorted(set(split.mtype)):
                m = split.mtype == ty
                S = np.column_stack([np.ones(m.sum()), split.signal[m].astype(float)])
                coef = np.linalg.lstsq(S, B[m], rcond=None)[0]
                mus[m] = S @ coef
        else:
            cells = _cells(split, pool)
            mus = np.zeros_like(B)
            for c in set(cells):
                m = np.array([x == c for x in cells])
                mus[m] = B[m].mean(0)
        c2 = dict(cfg, lam=cfg["pool_lam"])
        res = [fit_machine(split.grid[i], int(split.cutoff[i]), c2, mus[i]) for i in range(n)]
    pw = np.array([r[0] for r in res])
    return (pw, res) if return_extra else pw


def predict(split, cfg):
    return finish(predict_pw(split, cfg), split)


# ---------------------------------------------------------------- screening from TRAIN histories
def _flags(row, cutoff):
    w = np.full(cutoff, np.nan)
    ok = ~np.isnan(row[:cutoff])
    w[ok] = np.where(row[:cutoff][ok] > 0, 1.0, -1.0)
    return w


def screen(train, max_lag=16):
    """lag_screen-style excess z^2 on train histories only. Returns dict with per-type single-lag and pair
    excess (mean z^2 - 1) and the data-chosen lags/pairs (above the null-control max of lags/pairs 9-16)."""
    def build():
        spec = importlib.util.spec_from_file_location("lag_screen", HERE.parent / "scripts" / "lag_screen.py")
        ls = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ls)
        W = [(str(train.mtype[i]), _flags(train.grid[i], int(train.cutoff[i]))) for i in range(train.n)]
        out = {"single": {}, "pair": {}}
        for ty in sorted(set(train.mtype)):
            ws = [w for t, w in W if t == ty]
            for l in range(1, max_lag + 1):
                z = np.array([ls.z_single(w, l) for w in ws])
                out["single"][(ty, l)] = float(np.nanmean(z ** 2) - 1)
            for lo, hi in itertools.combinations(range(1, max_lag + 1), 2):
                if not (hi <= 8 or lo >= 9):
                    continue
                z = np.array([ls.z_pair(w, lo, hi) for w in ws])
                z = z[~np.isnan(z)]
                if len(z) >= 30:
                    out["pair"][(ty, lo, hi)] = float(np.mean(z ** 2) - 1)
        return out
    return cached(f"w1_screen_{train.n}_{int(np.nansum(train.grid))}", build)


def screened_sets(train, lag_margin=1.0, pair_margin=1.5):
    """Lags/pairs (within 1-8) whose excess beats the null-control max (lags / pairs in 9-16) by a margin
    in at least one type. Margins are fixed multiples; the lag/pair identities come from the data."""
    s = screen(train)
    types = sorted({k[0] for k in s["single"]})
    lags, pairs = set(), set()
    for ty in types:
        nl = max(s["single"][(ty, l)] for l in range(9, 17))
        for l in range(1, 9):
            if s["single"][(ty, l)] > max(nl, 0.0) * lag_margin + 0.5:
                lags.add(l)
        npairs = [v for (t, a, b), v in s["pair"].items() if t == ty and a >= 9]
        nm = max(npairs)
        for (t, a, b), v in s["pair"].items():
            if t == ty and b <= 8 and v > max(nm, 0.0) * pair_margin + 0.5:
                pairs.add((a, b))
    return tuple(sorted(lags)), tuple(sorted(pairs))


def learned(train):
    """Data-learned structure from train histories (same rules as L2): strict lags (excess > 3x null max in
    some type), top3 pairs (largest pooled excess), strict pairs, screened lags/pairs, all within lags 1-8."""
    s = screen(train)
    ls, ps = [], []
    for (ty, l), val in s["single"].items():
        nm = max(v for (t2, l2), v in s["single"].items() if t2 == ty and l2 >= 9)
        if l <= 8 and val > 3 * max(nm, 1.0) and l not in ls:
            ls.append(l)
    agg = {}
    for (ty, a, b), val in s["pair"].items():
        if b <= 8:
            agg[(a, b)] = agg.get((a, b), 0) + val
            nm = max(v for (t2, a2, b2), v in s["pair"].items() if t2 == ty and a2 >= 9)
            if val > 3 * max(nm, 1.0) and (a, b) not in ps:
                ps.append((a, b))
    top = lambda k: tuple(sorted(sorted(agg, key=lambda x: -agg[x])[:k]))
    scl, scp = screened_sets(train)
    return dict(lag_strict=tuple(sorted(ls)), pair_strict=tuple(sorted(ps)), pair_top3=top(3), pair_top6=top(6),
                lag_scr=scl, pair_scr=scp, lag_all=tuple(range(1, 9)),
                pair_all=tuple(itertools.combinations(range(1, 9), 2)))
