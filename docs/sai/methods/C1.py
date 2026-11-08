"""C1: B4 win model + A9 severity ridge applied to alpha-shrunk size shares (A1_B4 type prior). numpy only. Ticks < cutoff.
pw = B4 pw (unchanged). Size shares sh = _freq.size_shares with prior=train type shares, alpha a, hl 64 (alpha=8 'self' = B3 shares,
i.e. A9 alone). Then A9.severity_shares (per-machine ridge on c1=P(>=2|win), lags 1,2,4 learned in A9 from TRAIN severity screen,
coding both, steps 1, lam2 640, hl inf, pair on) adjusts them.
PRE-DECLARED GRID. Stage 1: alpha {24,64,128} x A9 lam {10,20,40} = 9 cells (+2 controls: alpha 8-self lam20 = A9 repro,
alpha 64 with no severity = A1_B4 repro). Stage 2 (best stage-1 cell, ONE extra idea): severity features re-screened with win lag 7
and pair terms: lags {1,2,4,7}; {1,2,4}+pairs (1,4),(2,4); {1,2,4,7}+pair (1,7); {1,2,4,7}+pairs (1,7),(2,7),(4,7) = 4 cells.
Run: python3 methods/C1.py [--smoke]
"""
import json, sys, time
import itertools
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _b4size as b          # patches _freq.pw_ridge to B4 pw
import _freq as f
import _core as K
import A9
from harness import (load_split, smoke_split, evaluate, run_method, blend_gain, diff_ci, read_pred_csv, class_shares)

SAI = Path(__file__).resolve().parent.parent
_orig_build_X = A9.build_X
_EXTRA = []


def _bx(A, tt, lags, coding, pair):
    X = _orig_build_X(A, tt, lags, coding, pair)
    if _EXTRA:
        cols = [A9.lagv(A["win"], tt, a) * A9.lagv(A["win"], tt, c) for a, c in _EXTRA]
        X = np.column_stack([X] + cols)
    return X


A9.build_X = _bx
_SH = {}


def shares(split, alpha):
    key = (split.name, split.n, alpha)
    if key not in _SH:
        if alpha == "self":
            _SH[key] = K.size_shares(split)
        else:
            P = f.make(dict(mode="ridge", prior="type", alpha=alpha, hl=64))(split)
            s = P[:, 1:]
            _SH[key] = s / s.sum(1, keepdims=True)
    return _SH[key]


def predict(split, p):
    global _EXTRA
    P0 = K.predict(split, A9.B4CFG)
    sh = shares(split, p["alpha"])
    pw = 1 - P0[:, 0]
    if not p.get("lags"):
        P = np.zeros((split.n, 5)); P[:, 0] = P0[:, 0]; P[:, 1:] = pw[:, None] * sh
        return P / P.sum(1, keepdims=True)
    _EXTRA = [tuple(x) for x in p.get("pairs", [])]
    sp = dict(lags=tuple(p["lags"]), coding="both", steps=1, lam=p["lam"], lam2=640.0, hl=1e9, pair=True, mode="own")
    P = np.zeros((split.n, 5))
    for i in range(split.n):
        s4 = A9.severity_shares(split.grid[i], int(split.cutoff[i]), sh[i], sp, None)
        P[i, 0] = P0[i, 0]
        P[i, 1:] = pw[i] * s4
    _EXTRA = []
    return P / P.sum(1, keepdims=True)


def main():
    v = load_split("validation")
    t = load_split("train")
    PB = read_pred_csv(str(SAI / "preds/B4_validation.csv"), v)
    lg = [1, 2, 4]
    if "--smoke" in sys.argv:
        sm = smoke_split(v)
        for p in (dict(alpha=64, lags=lg, lam=20.0), dict(alpha="self", lags=lg, lam=20.0), dict(alpha=64, lags=[])):
            print("smoke", p, evaluate(predict(sm, p), sm, "smoke", bootstrap=0)["points"], flush=True)
        return
    ev = lambda P: evaluate(P, v, "dev", bootstrap=0)
    print("B4", ev(PB)["points"], "learned lags", A9.learned_lags(t, 1.5, 0.3), flush=True)
    log = []
    gp = SAI / "grids/C1.json"

    def run(lab, p):
        t0 = time.time()
        P = predict(v, p)
        e = ev(P)
        log.append(dict(label=lab, params=p, points=e["points"], condition_ratios=e["condition_ratios"], seconds=round(time.time() - t0, 1)))
        gp.write_text(json.dumps(log, indent=1, default=float))
        print(f"C1 {lab}: {e['points']:.3f} ({e['points'] - ev(PB)['points']:+.3f})", flush=True)
        return e["points"]
    run("ctl_A9_self_lam20", dict(alpha="self", lags=lg, lam=20.0))
    run("ctl_A1_alpha64_nosev", dict(alpha=64, lags=[]))
    best = (-1, None, None)
    for a, lam in itertools.product((24, 64, 128), (10.0, 20.0, 40.0)):
        p = dict(alpha=a, lags=lg, lam=lam)
        pt = run(f"a{a}_lam{lam:g}", p)
        if pt > best[0]:
            best = (pt, f"a{a}_lam{lam:g}", p)
    bp = best[2]
    for lab, lags, pairs in (("L1247", [1, 2, 4, 7], []), ("pairs14_24", lg, [(1, 4), (2, 4)]), ("L1247_p17", [1, 2, 4, 7], [(1, 7)]),
                             ("L1247_p17_27_47", [1, 2, 4, 7], [(1, 7), (2, 7), (4, 7)])):
        p = dict(bp, lags=lags, pairs=pairs)
        pt = run("S2_" + lab, p)
        if pt > best[0]:
            best = (pt, "S2_" + lab, p)
    p = best[2]
    doc, P = run_method("C1", "claude-w1", f"B4 + A9 severity on alpha-shrunk shares; best {best[1]}", json.loads(json.dumps(p, default=float)),
                        lambda s: predict(s, p), v)
    A9P = read_pred_csv(str(SAI / "preds/A9_validation.csv"), v)
    res = json.loads((SAI / "results/C1.json").read_text())
    bg = blend_gain(P, PB, v)
    res.update(best_label=best[1], n_cells=len(log), vs_B4=diff_ci(P, PB, v, "dev"), vs_A9=diff_ci(P, A9P, v, "dev"),
               blend_gain_vs_B4=dict(points=bg["points"], vs_base=bg["vs_base"], w_mean=bg["w_mean"]), class_shares_B4=class_shares(PB, v, "dev"))
    (SAI / "results/C1.json").write_text(json.dumps(res, indent=1, default=float))
    print("BEST", best[1], doc["points"], doc["ci95"], "vsB4", res["vs_B4"], "vsA9", res["vs_A9"], "blend", bg["vs_base"])
    print("shares", res["class_error_shares"], "B4", res["class_shares_B4"]); print("ratios", res["condition_ratios"])
    # also the pre-declared plain combo (alpha64, lam20) for the report
    Pp = predict(v, dict(alpha=64, lags=lg, lam=20.0))
    print("PLAIN a64 lam20", ev(Pp)["points"], "vsB4", diff_ci(Pp, PB, v, "dev"), "vsA9", diff_ci(Pp, A9P, v, "dev"))


if __name__ == "__main__":
    main()
