"""L10: payout-level features in the B4 WIN model. numpy only. Only ticks < cutoff.
B4 = _core win model (lag 7 flag + pairs (1,3),(2,5),(4,7), ridge lam .05, B3 holdout half-life memory) x B3 size split.
Only the win design changes: extra columns appended through a wrapper around K.design:
  plags  : centred log1p payout (flag-0 for loss gives -mean) at these lags (added to flags)
  lvl7   : replace lag-7 flag by L5 'log' coding (coding='log')
  lpair  : lg_a*lg_b product for each B4 pair
  xprod  : cross products f_a*lg_b + lg_a*f_b as two columns for listed (a,b)
TRAIN screen (lvl_screen, train histories only): per machine z of corr(win_t, lg_{t-l}) lags 1-16 (null = 9-16); products
lg_a*f_b / f_a*lg_b / lg_a*lg_b (partial z of the product term controlling both mains) for pairs a<b<=8, null pairs a,b in 9-16.
Features picked: lags whose mean z^2-1 in some type exceeds null max + 0.5; products likewise (margin 0.5 over null max).
PRE-DECLARED GRID: feature set {B4, +lg@screened, +lg@(1,2,4), +lg@(1,2,4,7), L5 lag7 flag+lg, L5 lag7 lg-only,
 +lg@screened & lpair, +lg@screened & xprod(screened)} x lam {0.05,0.5,2} = 24 cells. Then compose win prob with C1 shares.
Run: python3 methods/L10.py [--smoke]
"""
import itertools, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _core as K
from harness import (load_split, smoke_split, evaluate, run_method, blend_gain, diff_ci, read_pred_csv, cached, class_shares)

SAI = Path(__file__).resolve().parent.parent
B4CFG = K.C(lam=0.05, lags=(7,), pairs=((1, 3), (2, 5), (4, 7)))
_orig_design = K.design


def design(A, tt, cfg, lags=None, pairs=None, triples=None):
    X = _orig_design(A, tt, cfg, lags, pairs, triples)
    ex = []
    for l in cfg.get("plags", ()):
        ex.append(K._lag(A["lg"], tt, l))
    if cfg.get("lpair"):
        for a, b in cfg["pairs"]:
            ex.append(K._lag(A["lg"], tt, a) * K._lag(A["lg"], tt, b))
    for a, b in cfg.get("xprod", ()):
        ex.append(K._lag(A["f"], tt, a) * K._lag(A["lg"], tt, b))
        ex.append(K._lag(A["lg"], tt, a) * K._lag(A["f"], tt, b))
    return np.column_stack([X] + ex) if ex else X


K.design = design


# ------------------------------------------------------------ train screen
def _zc(x, y):
    ok = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() < 40 or x[ok].std() == 0 or y[ok].std() == 0:
        return np.nan
    return np.corrcoef(x[ok], y[ok])[0, 1] * np.sqrt(ok.sum() - 2)


def _zprod(y, xa, xb):
    ok = ~np.isnan(y) & ~np.isnan(xa) & ~np.isnan(xb)
    y, xa, xb = y[ok], xa[ok], xb[ok]
    n = len(y)
    if n < 60 or y.std() == 0 or xa.std() == 0 or xb.std() == 0:
        return np.nan
    X = np.column_stack([np.ones(n), xa, xb, xa * xb])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ beta
    try:
        var = (r @ r / (n - 4)) * np.linalg.inv(X.T @ X)[3, 3]
    except np.linalg.LinAlgError:
        return np.nan
    return beta[3] / np.sqrt(var) if var > 0 else np.nan


def lvl_screen(t):
    def build():
        out = {"single": {}, "lg_f": {}, "f_lg": {}, "lg_lg": {}}
        pairs = [(a, b) for a, b in itertools.combinations(range(1, 17), 2) if b <= 8 or a >= 9]
        for ty in sorted(set(t.mtype)):
            S = []; P = {k: [] for k in ("lg_f", "f_lg", "lg_lg")}
            for i in range(t.n):
                if t.mtype[i] != ty:
                    continue
                g = t.grid[i, :t.cutoff[i]]
                lg = np.log1p(g); lg = lg - np.nanmean(lg)
                f = np.where(np.isnan(g), np.nan, np.where(g > 0, 1.0, -1.0))
                y = f
                S.append([_zc(lg[:-l], y[l:]) for l in range(1, 17)])
                def sh(x, l):  # value at t-l aligned to t >= 16
                    return x[16 - l:len(x) - l]
                yy = y[16:]
                for k, fa, fb in (("lg_f", lg, f), ("f_lg", f, lg), ("lg_lg", lg, lg)):
                    P[k].append([_zprod(yy, sh(fa, a), sh(fb, b)) for a, b in pairs])
            out["single"][ty] = (np.nanmean(np.array(S) ** 2, 0) - 1).tolist()
            for k in P:
                out[k][ty] = (np.nanmean(np.array(P[k]) ** 2, 0) - 1).tolist()
        out["pairs"] = pairs
        return out
    return cached(f"l10_lvlscreen_{t.n}_{int(np.nansum(t.grid))}", build)


def screened(t, lag_add=0.5, prod_add=0.5, top=2):
    s = lvl_screen(t)
    pairs = s["pairs"]
    lags = set(); prods = {}
    for ty, e in s["single"].items():
        thr = max(max(e[8:]), 0) + lag_add
        lags |= {l for l in range(1, 9) if e[l - 1] > thr}
    for k in ("lg_f", "f_lg", "lg_lg"):
        for ty, e in s[k].items():
            nm = max(max(e[j] for j, (a, b) in enumerate(pairs) if a >= 9), 0)
            for j, (a, b) in enumerate(pairs):
                if b <= 8 and e[j] > nm + prod_add:
                    prods[(a, b)] = max(prods.get((a, b), 0), e[j] - nm)
    best = tuple(sorted(sorted(prods, key=lambda x: -prods[x])[:top]))
    return tuple(sorted(lags)), best, s


def cell_list(t):
    lags, prods, _ = screened(t)
    sets = {
        "B4": {},
        "lg_scr": dict(plags=lags),
        "lg_124": dict(plags=(1, 2, 4)),
        "lg_1247": dict(plags=(1, 2, 4, 7)),
        "L5_flaglog7": dict(lags=(7,), coding="flag_log"),
        "L5_log7": dict(lags=(7,), coding="log"),
        "lg_scr+lpair": dict(plags=lags, lpair=True),
        "lg_scr+xprod": dict(plags=lags, xprod=prods),
    }
    return [(f"{n}_lam{lam}", dict(set=n, lam=lam, **kw)) for n, kw in sets.items() for lam in (0.05, 0.5, 2.0)]


def predict_pw(split, p):
    q = {k: v for k, v in p.items() if k != "set"}
    return K.predict_pw(split, K.C(**dict(dict(lags=(7,), pairs=((1, 3), (2, 5), (4, 7))), **q)))


def predict(split, p, shares="B3"):
    pw = predict_pw(split, p)
    if shares == "B3":
        return K.finish(pw, split)
    c1 = read_pred_csv(SAI / "preds/C1_validation.csv", split)
    sh = c1[:, 1:] / c1[:, 1:].sum(1, keepdims=True)
    return np.column_stack([1 - pw, pw[:, None] * sh])


def _j(o):
    if isinstance(o, dict): return {str(k): _j(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_j(x) for x in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o


if __name__ == "__main__":
    v = load_split("validation"); t = load_split("train")
    lags, prods, s = screened(t)
    print("screened lags", lags, "prods", prods)
    for ty in s["single"]:
        print(ty, "single", np.round(s["single"][ty], 2).tolist())
    cells = cell_list(t)
    if "--smoke" in sys.argv:
        sm = smoke_split(v)
        for lab, p in cells[:3] + cells[-3:]:
            P = predict(sm, p); assert np.isfinite(P).all()
            print("[smoke]", lab, evaluate(P, sm, "smoke", bootstrap=0)["points"], flush=True)
        sys.exit()
    log = []
    for lab, p in cells:
        t0 = time.time()
        ev = evaluate(predict(v, p), v, "dev", bootstrap=0)
        log.append(dict(label=lab, params=_j(p), points=ev["points"], condition_ratios=ev["condition_ratios"]))
        print(lab, round(ev["points"], 3), round(time.time() - t0, 1), flush=True)
    (SAI / "grids/L10.json").write_text(json.dumps(log, indent=1))
    best = max(log, key=lambda r: r["points"]); bp = dict(cells)[best["label"]]
    doc, P = run_method("L10", "claude", f"B4 win model + payout-level features | best {best['label']}", _j(bp),
                        lambda sp: predict(sp, bp), v)
    PB4 = read_pred_csv(SAI / "preds/B4_validation.csv", v); PC2 = read_pred_csv(SAI / "preds/C2_validation.csv", v)
    out = dict(best=best["label"], screened_lags=lags, screened_prods=prods)
    out["vs_B4"] = diff_ci(P, PB4, v, "dev"); out["vs_C2"] = diff_ci(P, PC2, v, "dev")
    bg = blend_gain(P, PB4, v); out["blend_gain_vs_B4"] = dict(points=bg["points"], vs_base=bg["vs_base"], w=bg["w_mean"])
    out["class_error_shares"] = class_shares(P, v, "dev")
    PCo = predict(v, bp, shares="C1")
    out["composed_C1"] = evaluate(PCo, v, "dev", bootstrap=0)
    out["composed_vs_C2"] = diff_ci(PCo, PC2, v, "dev")
    out["composed_vs_B4"] = diff_ci(PCo, PB4, v, "dev")
    bg2 = blend_gain(PCo, PC2, v); out["composed_blend_vs_C2"] = dict(points=bg2["points"], vs_base=bg2["vs_base"], w=bg2["w_mean"])
    out["cells"] = len(log)
    (SAI / "results/L10_extra.json").write_text(json.dumps(_j(out), indent=1, default=str))
    print(json.dumps(_j(out), indent=1, default=str))
