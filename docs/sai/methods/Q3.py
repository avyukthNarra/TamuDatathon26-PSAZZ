"""Q3 (Q1 only): use Q1 filter win prob as a member of BEST (TM_COMBINED3). Shares from BEST.
(a) 'blend': pw = (1-a) pw_BEST + a pw_Q1, a in [0,1] (also 'blend2' in logit space: logit pw = logit BEST + a (logit Q - logit BEST), a free in [-1,1]).
(b) 'gate': pw' = pw_BEST + s(z)(pw_Q1 - pw_BEST), s=sigmoid(b0+z.beta), z = 6 vol feats (V3's set, volfeat.py), ridge lam {3e-3,3e-2,3e-1}.
Q1 sources: machine / pooled / bma sigma. Fit on true q, R-weighted rows (row_weights), episode-OOF 6 folds; audit rows fit on all dev."""
import sys, json, time
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, __import__("os").path.dirname(__file__))
import volfeat as V
from harness import to_csv, write_result, blend_gain
HERE = V.HERE
FE = ["ewv_w16", "vol_of_vol", "garch_h", "abs_gap_z", "obs_frac", "peer_mean_ewv_w16"]
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
sg = lambda x: 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    t0 = time.time()
    v, names, F, dev, rows, arows, w, fold = V.setup()
    base = V.read_pred_csv(HERE / "preds" / "TM_COMBINED3_validation.csv", v)
    pb, sh = V.parts(base)
    Z = np.column_stack([(F[:, names.index(f)] - F[rows, names.index(f)].mean()) / F[rows, names.index(f)].std() for f in FE])
    grid = []; best = None
    for src in ("machine", "pooled", "bma"):
        pq = np.load(HERE / "cache" / f"Q1_pw_{src}.npy")
        def mkP(p, idx): return np.column_stack([1 - p, p[:, None] * sh[idx]])
        cfgs = {"blend": (lambda x, idx: pb[idx] + x[0] * (pq[idx] - pb[idx]), np.array([0.1]), [(0, 1)], 0.0),
                "blend2": (lambda x, idx: sg(lg(pb[idx]) + x[0] * (lg(pq[idx]) - lg(pb[idx]))), np.array([0.1]), [(-1, 1)], 0.0)}
        for lam in (3e-3, 3e-2, 3e-1):
            cfgs[f"gate_{lam}"] = (lambda x, idx: pb[idx] + sg(x[0] + Z[idx] @ x[1:]) * (pq[idx] - pb[idx]), np.r_[-3.0, np.zeros(6)], None, lam)
        for cn, (fn, x0, bnd, lam) in cfgs.items():
            def fitfn(tr):
                ww = w[tr] / w[tr].sum()
                f = lambda x: float((ww * ((mkP(fn(x, tr), tr) - v.truth[tr]) ** 2).sum(1)).sum() + lam * (x[1:] ** 2).sum())
                return minimize(f, x0, method="L-BFGS-B", bounds=bnd, options={"maxiter": 200}).x
            out, m = V.oof_run(v, dev, fold, rows, arows, fitfn, lambda x, idx: mkP(fn(x, idx), idx))
            ev, d = V.report(out, v, base, f"Q3 {src} {cn} x={np.round(m[:3],3)}")
            bg = blend_gain(out, base, v)
            grid.append({"params": {"src": src, "cfg": cn, "x": m.tolist()}, "points": ev["points"], "ci95": ev["ci95"], "condition_ratios": ev["condition_ratios"], "diff_vs_BEST": d["diff"], "diff_ci": d["ci95"]})
            if best is None or ev["points"] > best[0]: best = (ev["points"], out, src, cn, m)
    _, out, src, cn, m = best
    to_csv(HERE / "preds" / "Q3_validation.csv", v, out)
    write_result("Q3", "sai-worker", "BEST win prob composed/gated with Q1 log-odds filter win prob, OOF fit", {"src": src, "cfg": cn, "x": m.tolist(), "settings_tried": len(grid)}, v, out, time.time() - t0)
    json.dump(grid, open(HERE / "grids" / "Q3.json", "w"), indent=1)
    print("best", src, cn, np.round(m, 3))
