"""L8: mixture transition distribution (MTD) win model per machine.
p(win_t) = sum_c lambda_c * P_c(win | state_c) over components c = {base rate} + single-lag components (state = win/loss at lag l)
+ optional pair components (state = joint win/loss of lags a,b). Conditional rates P_c are recency-weighted, Beta-shrunk
(strength k) to the machine base rate; lambda fit by 40 fixed EM iterations on leave-one-out rates (so a cell does not
explain its own row). Components with unobserved state at a tick are dropped and lambda renormalised over the rest.
Half-life by the B3 holdout rule. Payout split = B3's. Sets learned from train (_core.learned).
PRE-DECLARED GRID: (lags, pairs) in {(1-8,-), (B3,-), (strict,-), (strict,top3), (B3,top3), (1-8,top3), (scr,scr)} x k {1,4} = 14 cells.
(Pure single-lag MTD is additive in probability, so it cannot represent pure pair interactions; pair components test that.)
Run: python3 methods/L8.py [--smoke]
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _core as K
from _run import run_grid
from harness import load_split
from L4 import _terms  # state codes (-1 unobserved), shared helper

LN = K.learned(load_split("train"))
HLG, HOLD, GAPPY, MINT, ITERS = K.B3["hl_grid"], 40, 0.6, 8, 40


def comp_probs(codes, y, w, k, pbar, loo):
    """per-row component probabilities (rows x C) and mask. loo: leave-one-out rates for the training rows."""
    C = len(codes)
    P = np.zeros((len(y), C))
    M = np.zeros((len(y), C))
    for c, ct in enumerate(codes):
        ok = ct >= 0
        M[:, c] = ok
        for s in np.unique(ct[ok]):
            m = ct == s
            S, W = (w[m] * y[m]).sum(), w[m].sum()
            if loo:
                P[m, c] = (S - w[m] * y[m] + k * pbar) / (W - w[m] + k)
            else:
                P[m, c] = (S + k * pbar) / (W + k)
    return np.clip(P, 0.01, 0.99), M


def mtd_fit(codes, y, w, k):
    pbar = float(np.clip((w * y).sum() / w.sum(), 0.01, 0.99))
    P, M = comp_probs(codes, y, w, k, pbar, True)
    P = np.column_stack([np.full(len(y), pbar), P])
    M = np.column_stack([np.ones(len(y)), M])
    lam = np.full(P.shape[1], 1.0 / P.shape[1])
    lik1 = np.where(y[:, None] > 0.5, P, 1 - P)
    for _ in range(ITERS):
        num = lam[None, :] * M * lik1
        r = num / np.maximum(num.sum(1, keepdims=True), 1e-12)
        lam = (w[:, None] * r).sum(0) / w.sum()
    return pbar, lam


def mtd_predict(codes_tr, y, w, codes_q, k, pbar, lam):
    """probability at query states."""
    num, den = lam[0] * pbar, lam[0]
    for c, (ct, cq) in enumerate(zip(codes_tr, codes_q)):
        if cq < 0:
            continue
        m = ct == cq
        pc = ((w[m] * y[m]).sum() + k * pbar) / (w[m].sum() + k)
        num += lam[c + 1] * float(np.clip(pc, 0.01, 0.99))
        den += lam[c + 1]
    return num / den


def fit_one(g, cutoff, p):
    A = K.machine_arrays(g, cutoff, K.B3)
    obs = np.flatnonzero(~np.isnan(g[:cutoff]))
    ticks = obs[obs >= MINT]
    y = (g[ticks] > 0).astype(float)
    lags, pairs, k = p["lags"], p["pairs"], p["k"]
    ctr = _terms(A["f"], ticks, lags, pairs)
    cq = [int(c[0]) for c in _terms(A["f"], np.array([cutoff]), lags, pairs)]
    hl = 1e9

    def run(sel, wref, c, qcodes=None):
        w = 0.5 ** ((wref - 1 - ticks[sel]) / c)
        ct = [x[sel] for x in ctr]
        pbar, lam = mtd_fit(ct, y[sel], w, k)
        return ct, w, pbar, lam

    if len(obs) >= GAPPY * cutoff:
        cut = cutoff - HOLD
        tr, te = ticks < cut, ticks >= cut
        errs = []
        for c in HLG:
            if tr.sum() < 30 or te.sum() < 10:
                errs.append(0.0 if c == 1e9 else 1.0)
                continue
            ct, w, pbar, lam = run(tr, cut, c)
            pr = [mtd_predict(ct, y[tr], w, [int(x[j]) for x in ctr], k, pbar, lam) for j in np.flatnonzero(te)]
            errs.append(np.mean((np.array(pr) - y[te]) ** 2))
        hl = HLG[int(np.argmin(errs))]
    ct, w, pbar, lam = run(np.ones(len(ticks), bool), cutoff, hl)
    return float(np.clip(mtd_predict(ct, y, w, cq, k, pbar, lam), 0.005, 0.995))


def predict(split, p):
    pw = np.array([fit_one(split.grid[i], int(split.cutoff[i]), p) for i in range(split.n)])
    return K.finish(pw, split)


SETS = {"1-8_nopair": (LN["lag_all"], ()), "B3_nopair": (K.B3["lags"], ()), "strict_nopair": (LN["lag_strict"], ()),
        "strict_top3": (LN["lag_strict"], LN["pair_top3"]), "B3_top3": (K.B3["lags"], LN["pair_top3"]),
        "1-8_top3": (LN["lag_all"], LN["pair_top3"]), "scr_scr": (LN["lag_scr"], LN["pair_scr"])}
cells = [(f"{n}_k{k}", dict(lags=l, pairs=pr, k=k)) for n, (l, pr) in SETS.items() for k in (1, 4)]
if __name__ == "__main__":
    run_grid("L8", "mixture-transition-distribution win model (single-lag and pair components, EM lambda)", cells, predict)
