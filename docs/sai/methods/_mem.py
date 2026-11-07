"""Shared core for W2 memory/nonstationarity methods (M1-M10).

Keeps B3's pair-ridge win model (lags 1-5,7 + pairs (1,3),(2,5),(4,7), ridge 20, missing lag -> 0) and its
size split (64-tick recency, prior 8 x type shares). A memory method only changes the ROW WEIGHTS of the ridge
fit (or mixes several fits). `b3_pw` reproduces B3's holdout selection exactly (checked against B3 preds).
Only ticks < cutoff are used. Peers (M9) use their own ticks < their cutoff.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SAI = HERE.parent
sys.path.insert(0, str(SAI))
from harness import (load_split, smoke_split, evaluate, diff_ci, blend_gain, run_method,  # noqa: E402
                     read_pred_csv, PAY)

LAGS = (1, 2, 3, 4, 5, 7)
PAIRS = ((1, 3), (2, 5), (4, 7))
LAM = 20.0
SIZE_HL, SIZE_PRIOR = 64.0, 8.0
MIN_ROW_TICK = 8
GAPPY_FRAC = 0.6
PAYV = (1, 2, 5, 20)
B3_GRID = (1e9, 128, 64, 32, 16)
INF = 1e9

_QD = {}


class Q:
    """Per-query precomputed data."""
    pass


def _flag(g, t, k):
    """+-1 win flag at tick t-k, 0 if missing/out of range (B3 semantics)."""
    j = t - k
    if j < 0:
        return 0.0
    v = g[j]
    return 0.0 if np.isnan(v) else (1.0 if v > 0 else -1.0)


def prep(split):
    key = (split.name, split.n, str(split.qid[0]))
    if key in _QD:
        return _QD[key]
    out = []
    # type shares as B3: per machine type, counts over all queries' pre-cutoff payouts
    cnt = {}
    for i in range(split.n):
        g = split.grid[i, :split.cutoff[i]]
        for k, p in enumerate(PAYV):
            cnt.setdefault(split.mtype[i], np.zeros(4))[k] += np.sum(g == p)
    tshare = {k: c / c.sum() for k, c in cnt.items()}
    for i in range(split.n):
        q = Q()
        c = int(split.cutoff[i])
        g = split.grid[i, :c]
        q.i, q.c, q.g = i, c, g
        obs = ~np.isnan(g)
        q.nobs = int(obs.sum())
        q.ticks = np.array([t for t in np.flatnonzero(obs) if t >= MIN_ROW_TICK])
        d = lambda t: [1.0] + [_flag(g, t, k) for k in LAGS] + [_flag(g, t, a) * _flag(g, t, b) for a, b in PAIRS]
        q.X = np.array([d(t) for t in q.ticks]) if len(q.ticks) else np.zeros((0, 1 + len(LAGS) + len(PAIRS)))
        q.y = (g[q.ticks] > 0).astype(float) if len(q.ticks) else np.zeros(0)
        q.xq = np.array(d(c))
        q.gappy = q.nobs >= GAPPY_FRAC * c
        q.win = np.where(obs, (np.nan_to_num(g) > 0).astype(float), np.nan)   # per tick 1/0/nan
        # size counts
        q.tshare = tshare[split.mtype[i]]
        pay = [(t, g[t]) for t in np.flatnonzero(obs) if g[t] > 0]
        q.pay_t = np.array([t for t, _ in pay], int)
        q.pay_k = np.array([PAYV.index(int(p)) for _, p in pay], int)
        out.append(q)
    _QD[key] = out
    return out


def sizes(q):
    cnt = SIZE_PRIOR * q.tshare.copy()
    if len(q.pay_t):
        w = 0.5 ** ((q.c - 1 - q.pay_t) / SIZE_HL)
        np.add.at(cnt, q.pay_k, w)
    return cnt / cnt.sum()


def ridge_w(X, y, w):
    pen = np.eye(X.shape[1]) * LAM
    pen[0, 0] = 1e-6
    return np.linalg.solve(X.T @ (w[:, None] * X) + pen, X.T @ (w * y))


def hl_w(ticks, ref, hl, floor=0.0):
    w = 0.5 ** ((ref - 1 - ticks) / hl)
    return np.maximum(w, floor) if floor > 0 else w


def pw_from_w(q, w):
    b = ridge_w(q.X, q.y, w)
    return float(np.clip(q.xq @ b, 0.005, 0.995))


def assemble(split, pws):
    """pws: list of win prob per query -> (n,5) with B3 size split."""
    qs = prep(split)
    P = np.zeros((split.n, 5))
    for q, pw in zip(qs, pws):
        pw = float(np.clip(pw, 0.005, 0.995))
        P[q.i, 0] = 1 - pw
        P[q.i, 1:] = pw * sizes(q)
    return P / P.sum(1, keepdims=True)


# ---------------- B3-style holdout selection (M1) ----------------
def holdout_errs(q, grid, holdout, floor=0.0):
    cut = q.c - holdout
    tr, te = q.ticks < cut, q.ticks >= cut
    errs = []
    for c in grid:
        if tr.sum() < 30 or te.sum() < 10:
            errs.append(0.0 if c == INF else 1.0)
            continue
        b = ridge_w(q.X[tr], q.y[tr], hl_w(q.ticks[tr], cut, c, floor))
        errs.append(np.mean((np.clip(q.X[te] @ b, 0, 1) - q.y[te]) ** 2))
    return errs


def b3_hl(q, grid=B3_GRID, holdout=40, floor=0.0):
    if not q.gappy:
        return INF
    return grid[int(np.argmin(holdout_errs(q, grid, holdout, floor)))]


def pw_hl(q, hl, floor=0.0):
    return pw_from_w(q, hl_w(q.ticks, q.c, hl, floor))


# ---------------- sequential predictions over a recent window (M2/M3) ----------------
_SEQ = {}


def seq_preds(q, grid, window, block, floor=0.0):
    """Block-refit one-step-ahead predictions of the win flag over the last `window` ticks.
    Returns (ticks_eval, y_eval, p[len(grid), n_eval]). Fit for block starting at s uses ticks < s only."""
    key = (q.i, q.c, tuple(grid), window, block, floor, id(q.X))
    if key in _SEQ:
        return _SEQ[key]
    start = q.c - window
    te_all = q.ticks[q.ticks >= start]
    P = np.zeros((len(grid), len(te_all)))
    for s0 in range(start, q.c, block):
        m = (te_all >= s0) & (te_all < s0 + block)
        if not m.any():
            continue
        tr = q.ticks < s0
        for gi, h in enumerate(grid):
            if tr.sum() < 30:
                P[gi, m] = float(np.clip(q.y[tr].mean(), 0.02, 0.98)) if tr.any() else 0.3
                continue
            b = ridge_w(q.X[tr], q.y[tr], hl_w(q.ticks[tr], s0, h, floor))
            P[gi, m] = np.clip(q.X[m_ticks(q, m, te_all)] @ b, 0.02, 0.98)
    res = (te_all, (q.win[te_all] if len(te_all) else np.zeros(0)), P)
    if len(_SEQ) > 40000:
        _SEQ.clear()
    _SEQ[key] = res
    return res


def m_ticks(q, m, te_all):
    """row indices into q.X for the evaluation ticks selected by mask m."""
    idx = np.searchsorted(q.ticks, te_all[m])
    return idx


# ---------------- scoring a cell ----------------
def cell_eval(P, split, scope):
    ev = evaluate(P, split, scope, bootstrap=0)
    return ev["points"], ev["condition_ratios"]


def run_grid(mid, doc, cells, make_predict, v, desc, smoke_only=False, cap=600, fixed_extra=None):
    """cells: list of param dicts. make_predict(params) -> predict_fn(split)->(n,5).
    Runs smoke first (24 eps), then full grid on dev. Writes grids/<ID>.json and best via run_method."""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        assert os.environ.get(k) == "1", k
    sm = smoke_split(v)
    t0 = time.perf_counter()
    P = make_predict(cells[0])(sm)
    pts, _ = cell_eval(P, sm, "smoke")
    dt = time.perf_counter() - t0
    est = dt * len(cells) * (v.n / sm.n)
    print(f"[{mid}] smoke cell0 {pts:.2f} pts in {dt:.1f}s; est full grid {est/60:.1f} min for {len(cells)} cells", flush=True)
    if smoke_only:
        return None
    b3 = read_pred_csv(str(SAI / "preds/B3_validation.csv"), v)
    grid_out, best = [], None
    for params in cells:
        t0 = time.perf_counter()
        P = make_predict(params)(v)
        dt = time.perf_counter() - t0
        pts, cr = cell_eval(P, v, "dev")
        rec = {"params": params, "points": pts, "condition_ratios": cr, "seconds": round(dt, 1)}
        grid_out.append(rec)
        print(f"[{mid}] {pts:.3f} {json.dumps(params)} {dt:.1f}s", flush=True)
        if best is None or pts > best[0]:
            best = (pts, params)
        (SAI / "grids" / f"{mid}.json").write_text(json.dumps(grid_out, indent=1, default=float))
    params = best[1]
    d, P = run_method(mid, "claude-w2", desc, params, make_predict(params), v)
    dd = diff_ci(P, b3, v, "dev")
    bg = blend_gain(P, b3, v)
    cr0 = json.load(open(SAI / "results/B3.json"))["condition_ratios"]
    d["extra"] = {"vs_B3": dd, "blend_gain": {"points": bg["points"], "ci95": bg["ci95"], "vs_base": bg["vs_base"],
                                              "w_mean": bg["w_mean"]},
                  "cond_ratio_diff_vs_B3": {k: d["condition_ratios"][k] - cr0[k] for k in cr0},
                  "grid_cells_completed": len(grid_out)}
    (SAI / "results" / f"{mid}.json").write_text(json.dumps(d, indent=1, default=float))
    print(f"[{mid}] BEST {d['points']:.3f} CI {d['ci95']} vsB3 {dd} blend {bg['points']:.3f} vs_base {bg['vs_base']} w={bg['w_mean']:.2f}")
    print("  cond diff vs B3:", {k: round(x, 3) for k, x in d["extra"]["cond_ratio_diff_vs_B3"].items()})
    return d, grid_out


def main_runner(mid, doc, cells, make_predict, desc):
    v = load_split("validation")
    smoke = "--smoke" in sys.argv
    return run_grid(mid, doc, cells, make_predict, v, desc, smoke_only=smoke)


_PWH = {}


def pw_hl_cached(q, hl):
    k = (q.i, q.c, hl, id(q.X))
    if k not in _PWH:
        _PWH[k] = pw_hl(q, hl)
    return _PWH[k]


POOLS = {"B3": B3_GRID, "B3+8": B3_GRID + (8,), "wide": (INF, 128, 64, 32, 16, 12, 8, 4),
         "fine": (INF, 192, 128, 96, 64, 48, 32, 24, 16)}
