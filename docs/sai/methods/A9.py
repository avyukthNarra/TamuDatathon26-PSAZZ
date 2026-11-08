"""A9: severity-given-win model atop B4. numpy only. Only ticks < cutoff.

B4 = _core win model (lag 7 + pairs, lam .05) x B3 size split. A9 keeps pw and replaces the fixed size shares by a
hierarchical split with per-machine ridge adjustments: c1=P(pay>=2|win), c2=P(pay>=5|>=2), c3=P(20|>=5) start at the B3
shares' implied values and move by (x_q - xbar)·beta_k, beta_k = recency-weighted ridge on the machine's own win rows
(centred target/features, so the intercept stays at the shrunk share). Features: lagged payout levels at lags learned from
TRAIN severity screen (sev_screen: excess z^2-1 of corr(log1p pay_t | win, lag-l feature), null control lags 9-16).
Codings: 'win' (centred log1p payout if win at the lag else 0 = wins-only), 'lvl' (centred log1p payout, loss=0),
'both' (win + lvl). pair=True adds products of the first-two-learned lag win-only features.
Modes: own (beta shrunk to 0), type_prior (beta shrunk toward train type pooled beta), type_only (beta=pooled, no own fit),
global_prior (pooled over types).
PRE-DECLARED GRID (round 1 below; round 2 extension is in cell_list) (stage 1, own mode): lagset {strict, loose, 1-8} x coding {win, lvl, both} x steps {1,3} x lam {10,40,160,640}
 x hl {64, inf} = 144 cells.  Stage 2 (on best stage-1 cell): mode {type_prior, type_only, global_prior} x lam {40,160,640} = 9 cells,
 and pair on/off = 2 cells (+ 3 'null' cells with lags 9-16 features, same best settings, as a control).
Run: python3 methods/A9.py [--smoke]
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _core as K
from harness import (load_split, smoke_split, evaluate, run_method, blend_gain, diff_ci, read_pred_csv, class_shares,
                     cached)

SAI = Path(__file__).resolve().parent.parent
B4CFG = K.C(lam=0.05, lags=(7,), pairs=((1, 3), (2, 5), (4, 7)))
PAYV = K.PAYV


# ---------------------------------------------------------------- screening (train only)
def _zc(x, y):
    ok = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() < 40 or x[ok].std() == 0 or y[ok].std() == 0:
        return np.nan
    return np.corrcoef(x[ok], y[ok])[0, 1] * np.sqrt(ok.sum() - 2)


def sev_screen(t):
    def build():
        out = {}
        for ty in sorted(set(t.mtype)):
            zs = {"lvl": [], "win": []}
            for i in range(t.n):
                if t.mtype[i] != ty:
                    continue
                g = t.grid[i, :t.cutoff[i]]
                lg = np.log1p(g)
                y = np.where(g > 0, lg, np.nan)
                zl, zw = [], []
                for l in range(1, 17):
                    zl.append(_zc(lg[:-l], y[l:]))
                    zw.append(_zc(np.where(g[:-l] > 0, lg[:-l], np.nan), y[l:]))
                zs["lvl"].append(zl)
                zs["win"].append(zw)
            for k in zs:
                a = np.array(zs[k])
                out[(ty, k)] = (np.nanmean(a ** 2, 0) - 1).tolist()
        return out
    return cached(f"a9_sevscreen_{t.n}_{int(np.nansum(t.grid))}", build)


def learned_lags(t, margin, floor):
    s = sev_screen(t)
    keep = set()
    for (ty, k), e in s.items():
        if k != "lvl":
            continue
        thr = max(margin * max(max(e[8:]), 0), floor)
        keep |= {l for l in range(1, 9) if e[l - 1] > thr}
    return tuple(sorted(keep))


# ---------------------------------------------------------------- features
def feat_arrays(g, c):
    obs = ~np.isnan(g[:c])
    pay = np.where(obs, g[:c], 0.0)
    lg = np.where(obs, np.log1p(pay), 0.0)
    win = obs & (pay > 0)
    lvl = np.where(obs, lg - lg.sum() / max(obs.sum(), 1), 0.0)
    wl = np.where(win, lg - lg[win].mean() if win.any() else 0.0, 0.0)
    ext = lambda a: np.concatenate([a, [0.0]])
    return dict(lvl=ext(lvl), win=ext(wl), pay=ext(pay), obs=obs)


def lagv(a, tt, l):
    idx = tt - l
    return np.where(idx >= 0, a[np.clip(idx, 0, len(a) - 1)], 0.0)


def build_X(A, tt, lags, coding, pair):
    cols = []
    for l in lags:
        if coding in ("win", "both"):
            cols.append(lagv(A["win"], tt, l))
        if coding in ("lvl", "both"):
            cols.append(lagv(A["lvl"], tt, l))
    if pair and len(lags) >= 2:
        cols.append(lagv(A["win"], tt, lags[0]) * lagv(A["win"], tt, lags[1]))
    return np.column_stack(cols) if cols else np.zeros((len(tt), 0))


def step_sets(pay):
    """(row mask, target) for the 3 hierarchical steps."""
    return [(pay > 0, (pay >= 2).astype(float)), (pay >= 2, (pay >= 5).astype(float)), (pay >= 5, (pay >= 20).astype(float))]


# ---------------------------------------------------------------- pooled train betas
def pooled_betas(t, lags, coding, pair, hl, lam_pool, nsteps):
    def build():
        acc = {}
        for i in range(t.n):
            g = t.grid[i]
            c = int(t.cutoff[i])
            A = feat_arrays(g, c)
            ticks = np.flatnonzero(A["obs"])
            ticks = ticks[ticks >= 8]
            pay = g[ticks]
            X = build_X(A, ticks, lags, coding, pair)
            w = 0.5 ** ((c - 1 - ticks) / hl)
            for ty in (str(t.mtype[i]), "ALL"):
                for k, (m, y) in enumerate(step_sets(pay)[:nsteps]):
                    if m.sum() < 5:
                        continue
                    ww = w[m] / w[m].sum()           # per-machine weight sums to 1: every machine counts equally
                    Xm, ym = X[m], y[m]
                    xb, yb = ww @ Xm, ww @ ym
                    Xc = Xm - xb
                    s = acc.setdefault((ty, k), [np.zeros((X.shape[1],) * 2), np.zeros(X.shape[1])])
                    s[0] += Xc.T @ (ww[:, None] * Xc)
                    s[1] += Xc.T @ (ww * (ym - yb))
        return {k: np.linalg.solve(s[0] + lam_pool * 1e-3 * np.eye(len(s[1])), s[1]) for k, s in acc.items()}
    key = f"a9_pooled_{t.n}_{int(np.nansum(t.grid))}_{lags}_{coding}_{pair}_{hl}_{nsteps}"
    return cached(key, build)


# ---------------------------------------------------------------- prediction
def severity_shares(g, c, sh, p, pool):
    """sh = B3 size shares (4). Returns adjusted 4 shares."""
    lags, coding, pair, hl, lam, nst = p["lags"], p["coding"], p.get("pair", False), p.get("hl", 64.0), p["lam"], p.get("steps", 3)
    if not lags:
        return sh
    A = feat_arrays(g, c)
    ticks = np.flatnonzero(A["obs"])
    ticks = ticks[ticks >= 8]
    pay = g[ticks]
    X = build_X(A, ticks, lags, coding, pair)
    xq = build_X(A, np.array([c]), lags, coding, pair)[0]
    w = 0.5 ** ((c - 1 - ticks) / hl)
    lam2 = p.get("lam2", lam)
    c1 = sh[1:].sum() / sh.sum()
    c2 = (sh[2] + sh[3]) / max(sh[1:].sum(), 1e-12)
    c3 = sh[3] / max(sh[2] + sh[3], 1e-12)
    cs = [c1, c2, c3]
    mode = p.get("mode", "own")
    ty = pool[0] if mode in ("type_prior", "type_only") else "ALL"
    sets = step_sets(pay)
    for k in range(nst):
        m, y = sets[k]
        mu = pool[1].get((ty, k)) if mode != "own" else None
        if mu is None:
            mu = np.zeros(X.shape[1])
        if mode == "type_only":
            beta = mu
            if m.sum() < 5:
                continue
            ww = w[m] / w[m].sum()
            xb = ww @ X[m]
        else:
            if m.sum() < 15:
                continue
            ww = w[m]
            sw = ww.sum()
            xb = (ww / sw) @ X[m]
            yb = (ww / sw) @ y[m]
            Xc = X[m] - xb
            r = (y[m] - yb) - Xc @ mu
            d = np.linalg.solve(Xc.T @ (ww[:, None] * Xc) + (lam if k == 0 else lam2) * np.eye(X.shape[1]), Xc.T @ (ww * r))
            beta = mu + d
        cs[k] = float(np.clip(cs[k] + (xq - xb) @ beta, 0.003, 0.997))
    c1, c2, c3 = cs
    return np.array([1 - c1, c1 * (1 - c2), c1 * c2 * (1 - c3), c1 * c2 * c3])


def predict(split, p):
    P0 = K.predict(split, B4CFG)  # B4 (also cached sizes)
    if not p.get("lags"):
        return P0
    sh = K.size_shares(split)
    pw = 1 - P0[:, 0]
    t = load_split("train")
    pools = None
    if p.get("mode", "own") != "own":
        pools = pooled_betas(t, tuple(p["lags"]), p["coding"], p.get("pair", False), p.get("hl", 64.0), p["lam"], p.get("steps", 3))
    P = np.zeros((split.n, 5))
    for i in range(split.n):
        s4 = severity_shares(split.grid[i], int(split.cutoff[i]), sh[i], p, (str(split.mtype[i]), pools))
        P[i, 0] = P0[i, 0]
        P[i, 1:] = pw[i] * s4
    return P / P.sum(1, keepdims=True)


def cell_list(t):
    strict, loose = learned_lags(t, 1.5, 0.3), learned_lags(t, 1.0, 0.1)
    sets = {"strict": strict, "loose": loose, "1-8": tuple(range(1, 9))}
    cells = []
    for (ln, ls), cd, st, lam, hl in itertools.product(sets.items(), ("win", "lvl", "both"), (1, 3), (10, 40, 160, 640), (64.0, 1e9)):
        cells.append((f"{ln}_{cd}_s{st}_lam{lam}_hl{int(min(hl, 999))}", dict(lags=ls, coding=cd, steps=st, lam=float(lam), hl=hl)))
    # round 2 (added because the best cell sat on the lam=10 edge and steps=1 beat steps=3): lam {1,2.5,5,10,20} x
    # lagset {strict, loose, 1-8} x coding {both, lvl} x hl {64, inf} x (steps 1 | steps 3 with lam2=640 for steps 2-3) = 120 cells
    for (ln, ls), cd, lam, hl, st in itertools.product(sets.items(), ("both", "lvl"), (1, 2.5, 5, 10, 20), (64.0, 1e9), (1, 3)):
        cells.append((f"R2_{ln}_{cd}_s{st}_lam{lam}_hl{int(min(hl, 999))}", dict(lags=ls, coding=cd, steps=st, lam=float(lam), lam2=640.0, hl=hl)))
    return cells, sets


def ev(P, v):
    return evaluate(P, v, "dev", bootstrap=0)


def main():
    v = load_split("validation")
    t = load_split("train")
    PB = read_pred_csv(str(SAI / "preds/B4_validation.csv"), v)
    cells, sets = cell_list(t)
    print("learned lag sets", sets, flush=True)
    if "--smoke" in sys.argv:
        sm = smoke_split(v)
        for lab, p in cells[:3]:
            P = predict(sm, p)
            assert np.isfinite(P).all()
            print("smoke", lab, ev(P, sm)["points"] if False else evaluate(P, sm, "smoke", bootstrap=0)["points"], flush=True)
        print("smoke B4", evaluate(predict(sm, dict(lags=())), sm, "smoke", bootstrap=0)["points"])
        return
    b4 = ev(PB, v)
    print("B4 ref", b4["points"], "repro", ev(predict(v, dict(lags=())), v)["points"], class_shares(PB, v, "dev"), flush=True)
    log = []
    gp = SAI / "grids/A9.json"

    def run(lab, p):
        t0 = time.time()
        P = predict(v, p)
        e = ev(P, v)
        log.append({"label": lab, "params": json.loads(json.dumps(p, default=lambda o: float(o))), "points": e["points"],
                    "condition_ratios": e["condition_ratios"], "seconds": round(time.time() - t0, 1)})
        gp.write_text(json.dumps(log, indent=1))
        print(f"A9 {lab}: {e['points']:.3f} ({e['points'] - b4['points']:+.3f}) {time.time() - t0:.1f}s", flush=True)
        return e["points"]

    best = (-1, None, None)
    for lab, p in cells:
        pt = run(lab, p)
        if pt > best[0]:
            best = (pt, lab, p)
    print("stage1 best", best[1], best[0], flush=True)
    bp = dict(best[2])
    for md, lam in itertools.product(("type_prior", "type_only", "global_prior"), (5.0, 20.0, 40.0, 160.0)):
        p = dict(bp, mode=md, lam=lam)
        pt = run(f"S2_{md}_lam{lam}", p)
        if pt > best[0]:
            best = (pt, f"S2_{md}_lam{lam}", p)
    run("S2_pair", dict(bp, pair=True))
    pb = max(log[-1:], key=lambda r: r["points"])
    if pb["points"] > best[0]:
        best = (pb["points"], "S2_pair", dict(bp, pair=True))
    for nl in ((9, 10, 11), (9, 10, 12), (10, 11, 12)):   # null control: same settings, lags beyond the simulator's range
        run(f"NULL_{nl}", dict(bp, lags=nl))
    p = best[2]
    doc, P = run_method("A9", "claude-w1", f"severity-given-win hierarchical ridge atop B4; best {best[1]}",
                        json.loads(json.dumps(p, default=lambda o: float(o))), lambda s: predict(s, p), v)
    d = diff_ci(P, PB, v, "dev")
    bg = blend_gain(P, PB, v)
    res = json.loads((SAI / "results/A9.json").read_text())
    res.update(best_label=best[1], n_grid_cells=len(log), vs_B4=d,
               blend_gain_vs_B4=dict(points=bg["points"], ci95=bg["ci95"], vs_base=bg["vs_base"], w_mean=bg["w_mean"]),
               class_shares_B4=class_shares(PB, v, "dev"))
    (SAI / "results/A9.json").write_text(json.dumps(res, indent=1))
    print("BEST", best[1], doc["points"], doc["ci95"], "vsB4", d, "blend", bg["vs_base"], bg["w_mean"])
    print("shares", res["class_error_shares"], "B4", res["class_shares_B4"])
    print("ratios", res["condition_ratios"])


if __name__ == "__main__":
    main()
