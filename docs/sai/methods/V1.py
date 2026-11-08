"""V1 vol screen + binned gate. The screen table (per-member error vs vol tercile, spearman) is printed by scr1.py (see report).
Prediction saved as V1 = binned gate: one-hot terciles (dev quantiles) of ewv_w16 and of obs_frac (4 indicator cols, 2 per feature,
third tercile = baseline), softmax-linear weights over TM_COMBINED,C2,TM_E7,TM_MIXER,TM_RULE, ridge lam on slopes.
Params = 4*(4+1)=20. Grid: lam in {3e-3,3e-2,3e-1}. Episode-OOF; audit rows fit on all dev."""
import json, numpy as np
import volfeat as V
import V2 as G
if __name__ == "__main__":
    sc = V.setup(); v, names, F, dev, rows, arows, w, fold = sc
    base = V.read_pred_csv(V.HERE / "preds" / "TM_COMBINED_validation.csv", v)
    Pm = [V.read_pred_csv(V.HERE / "preds" / f"{m}_validation.csv", v) for m in G.MEM]
    cols = []
    for f in ("ewv_w16", "obs_frac"):
        x = F[:, names.index(f)]; q = np.quantile(x[rows], [1 / 3, 2 / 3]); b = np.digitize(x, q)
        cols += [(b == 0).astype(float), (b == 1).astype(float)]
    Z = np.column_stack(cols); K = len(Pm)
    grid = []; best = None
    for lam in (3e-3, 3e-2, 3e-1):
        fitfn = lambda tr: G.fit_gate([p[tr] for p in Pm], Z[tr], v.truth[tr], w[tr], lam)
        predfn = lambda x, te: np.einsum("nk,knc->nc", G.gates(x, Z[te], K), np.stack([p[te] for p in Pm]))
        out, m = V.oof_run(v, dev, fold, rows, arows, fitfn, predfn)
        ev, d = V.report(out, v, base, f"V1 lam={lam}")
        grid.append({"params": {"lam": lam, "n_params": len(m)}, "points": ev["points"], "ci95": ev["ci95"],
                     "condition_ratios": ev["condition_ratios"], "diff_vs_TMC": d["diff"], "diff_ci": d["ci95"]})
        if best is None or ev["points"] > best[0]: best = (ev["points"], out, lam, m)
    from harness import to_csv, write_result
    _, out, lam, m = best
    to_csv(V.HERE / "preds" / "V1_validation.csv", v, out)
    write_result("V1", "sai-worker", "vol-tercile binned gate (ewv_w16 x obs_frac) over 5 members, episode-OOF",
                 {"lam": lam, "n_params": len(m), "settings_tried": len(grid)}, v, out, 0.0)
    (V.HERE / "grids" / "V1.json").write_text(json.dumps(grid, indent=1))
