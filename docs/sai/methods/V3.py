"""V3 vol-scaled shrinkage of p_win=1-P0 of TM_COMBINED toward a longer-memory baseline; shares unchanged.
p' = p + s(z)(b - p), s = sigmoid(b0 + z.beta) in (0,1), z = standardised vol feats (ewv_w16, vol_of_vol, garch_h, abs_gap_z, obs_frac,
peer_mean_ewv_w16); params = 1+6 = 7, ridge lam on beta. Baselines b: 'long' = machine's whole-history win rate (obs ticks only),
'peer' = mean of peers' whole-history win rates (blend 50/50 'mix'), 'L7' = ours L7 p_win. Grid: base x lam{3e-3,3e-2,3e-1} = 9 settings.
Episode-OOF; audit rows fit on all dev."""
import json, numpy as np
from scipy.optimize import minimize
import volfeat as V
FE = ["ewv_w16", "vol_of_vol", "garch_h", "abs_gap_z", "obs_frac", "peer_mean_ewv_w16"]
if __name__ == "__main__":
    sc = V.setup(); v, names, F, dev, rows, arows, w, fold = sc
    base = V.read_pred_csv(V.HERE / "preds" / "TM_COMBINED_validation.csv", v)
    pw, sh = V.parts(base)
    Z = np.column_stack([(F[:, names.index(f)] - F[rows, names.index(f)].mean()) / F[rows, names.index(f)].std() for f in FE])
    lr = F[:, names.index("long_rate")]
    peer = np.array([np.mean([lr[j] for j in v.peers(i)]) for i in range(v.n)])
    B = {"long": lr, "peer": peer, "mix": 0.5 * (lr + peer), "L7": V.parts(V.read_pred_csv(V.HERE / "preds" / "L7_validation.csv", v))[0]}
    grid = []; best = None
    for bn, b in B.items():
        for lam in (3e-3, 3e-2, 3e-1):
            def mk(x, idx):
                s = 1 / (1 + np.exp(-(x[0] + Z[idx] @ x[1:])))
                p = pw[idx] + s * (b[idx] - pw[idx])
                return np.column_stack([1 - p, p[:, None] * sh[idx]])
            def fitfn(tr):
                ww = w[tr] / w[tr].sum()
                f = lambda x: float((ww * ((mk(x, tr) - v.truth[tr]) ** 2).sum(1)).sum() + lam * (x[1:] ** 2).sum())
                x0 = np.r_[-3.0, np.zeros(Z.shape[1])]
                return minimize(f, x0, method="L-BFGS-B", options={"maxiter": 200}).x
            out, m = V.oof_run(v, dev, fold, rows, arows, fitfn, mk)
            ev, d = V.report(out, v, base, f"V3 base={bn} lam={lam}")
            grid.append({"params": {"base": bn, "lam": lam, "b0": m[0], "beta": m[1:].tolist()}, "points": ev["points"], "ci95": ev["ci95"],
                         "condition_ratios": ev["condition_ratios"], "diff_vs_TMC": d["diff"], "diff_ci": d["ci95"]})
            if best is None or ev["points"] > best[0]: best = (ev["points"], out, bn, lam, m)
    from harness import to_csv, write_result
    _, out, bn, lam, m = best
    to_csv(V.HERE / "preds" / "V3_validation.csv", v, out)
    write_result("V3", "sai-worker", "vol-scaled shrinkage of TM_COMBINED p_win toward long-memory baseline, shares unchanged",
                 {"base": bn, "lam": lam, "feats": FE, "n_params": len(m), "b0": m[0], "beta": m[1:].tolist(), "settings_tried": len(grid)}, v, out, 0.0)
    (V.HERE / "grids" / "V3.json").write_text(json.dumps(grid, indent=1))
    print("final b0,beta", np.round(m, 2))
