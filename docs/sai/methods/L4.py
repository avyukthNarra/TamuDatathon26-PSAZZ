"""L4: per-machine conditional-lift model (teammate-guide style) instead of product features.
logit p = logit pbar + gamma * sum_terms [logit(p_cell) - logit(pbar)], where each term is the shrunken (Beta, strength k)
recency-weighted win rate of the machine given the state of one lag (mode 'single') or the joint 4-cell state of a lag pair
(mode 'pairjoint': joint cell for each pair + single-lag lifts only for lags not already in a pair). Terms whose state is
unobserved at the cutoff are skipped. Half-life per machine from the same holdout rule as B3 (grid 1e9,128,64,32,16).
Payout split = B3's. Lag/pair sets learned from train (_core.learned).
PRE-DECLARED GRID (second run; the first run completed 54 cells: single 36 [all 30.9-31.1 pts] + pairjoint 18, best pairjoint_top3_k2_g1.0 = 46.27 on the k=2 / gamma=1 edge, so the pairjoint grid was extended and the single grid trimmed for runtime): single: lag set {strict, B3, 1-8, scr} x k {8} x gamma {0.75,1} = 8 cells;
 pairjoint: top3 x k {0.25,0.5,1,2,8,32} x gamma {0.75,1,1.25} = 18 cells; top6, scr x k {0.25,1,8} x gamma 1 = 6 cells. Single-lag-only lags in pairjoint = strict lags not in any pair.
Run: python3 methods/L4.py [--smoke]
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _core as K
from _run import run_grid
from harness import load_split

LN = K.learned(load_split("train"))
HLG, HOLD, GAPPY, MINT = K.B3["hl_grid"], 40, 0.6, 8
lg = lambda p: np.log(p / (1 - p))


def _terms(f, tt, lags, pairs):
    """integer state codes per term at ticks tt; -1 = unobserved."""
    out = []
    for l in lags:
        s = K._lag(f, tt, l)
        out.append(np.where(s == 0, -1, (s > 0).astype(int)))
    for a, b in pairs:
        sa, sb = K._lag(f, tt, a), K._lag(f, tt, b)
        out.append(np.where((sa == 0) | (sb == 0), -1, 2 * (sa > 0) + (sb > 0)).astype(int))
    return out


def _pw(codes_tr, y, w, codes_q, k, gamma, ncells):
    pbar = float(np.clip((w * y).sum() / w.sum(), 0.01, 0.99))
    z = lg(pbar)
    for ct, cq, nc in zip(codes_tr, codes_q, ncells):
        if cq < 0:
            continue
        m = ct == cq
        pc = ((w[m] * y[m]).sum() + k * pbar) / (w[m].sum() + k)
        z += gamma * (lg(float(np.clip(pc, 0.01, 0.99))) - lg(pbar))
    return 1 / (1 + np.exp(-z))


def fit_one(g, cutoff, p):
    A = K.machine_arrays(g, cutoff, K.B3)
    obs = np.flatnonzero(~np.isnan(g[:cutoff]))
    ticks = obs[obs >= MINT]
    y = (g[ticks] > 0).astype(float)
    lags, pairs = p["lags"], p["pairs"]
    ctr = _terms(A["f"], ticks, lags, pairs)
    cq = [int(c[0]) for c in _terms(A["f"], np.array([cutoff]), lags, pairs)]
    nc = [2] * len(lags) + [4] * len(pairs)
    hl = 1e9
    if len(obs) >= GAPPY * cutoff:
        cut = cutoff - HOLD
        tr, te = ticks < cut, ticks >= cut
        errs = []
        for c in HLG:
            if tr.sum() < 30 or te.sum() < 10:
                errs.append(0.0 if c == 1e9 else 1.0)
                continue
            w = 0.5 ** ((cut - 1 - ticks[tr]) / c)
            e = 0.0
            ctrt = [x[tr] for x in ctr]
            pr = []
            for j in np.flatnonzero(te):
                pr.append(_pw(ctrt, y[tr], w, [int(x[j]) for x in ctr], p["k"], p["gamma"], nc))
            errs.append(np.mean((np.array(pr) - y[te]) ** 2))
        hl = HLG[int(np.argmin(errs))]
    w = 0.5 ** ((cutoff - 1 - ticks) / hl)
    return float(np.clip(_pw(ctr, y, w, cq, p["k"], p["gamma"], nc), 0.005, 0.995))


def predict(split, p):
    pw = np.array([fit_one(split.grid[i], int(split.cutoff[i]), p) for i in range(split.n)])
    return K.finish(pw, split)


LS = {"strict": LN["lag_strict"], "B3": K.B3["lags"], "1-8": LN["lag_all"], "scr": LN["lag_scr"]}
PS = {"top3": LN["pair_top3"], "top6": LN["pair_top6"], "scr": LN["pair_scr"]}
cells = []
for ln, l in LS.items():
    for k in (8,):
        for gm in (0.75, 1.0):
            cells.append((f"single_{ln}_k{k}_g{gm}", dict(lags=l, pairs=(), k=k, gamma=gm)))
for pn, pr in PS.items():
    inpair = {x for ab in pr for x in ab}
    single = tuple(x for x in LN["lag_strict"] if x not in inpair)  # lag 7 is in pair (4,7): then it appears only via the joint cell
    for k in ((0.25, 0.5, 1, 2, 8, 32) if pn == "top3" else (0.25, 1, 8)):
        for gm in ((0.75, 1.0, 1.25) if pn == "top3" else (1.0,)):
            cells.append((f"pairjoint_{pn}_k{k}_g{gm}", dict(lags=single, pairs=pr, k=k, gamma=gm)))
if __name__ == "__main__":
    run_grid("L4", "conditional-lift naive-Bayes per machine (single-lag or joint pair cells) instead of product features", cells, predict)
