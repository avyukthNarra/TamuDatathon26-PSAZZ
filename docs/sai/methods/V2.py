"""V2 volatility-gated blend. P = sum_k g_k(z) P_k, g = softmax(b_k + z.theta_k), reference member (TM_COMBINED) logits fixed 0,
z = standardised vol features (dev mean/sd). Params = (K-1)*(nfeat+1). Ridge on theta only (lam), loss = R-weighted true-q squared error
(row_weights from stack_w5, renormalised to sum 1). Episode-OOF 6 folds (w5fold hash). Audit rows: fit on all dev.
Grid (6 settings): feature sets {A,B} x lam {3e-3,3e-2,3e-1}.
  A = ewv_w16, vol_of_vol, peer_mean_ewv_w16, obs_frac    (4 feats, 20 params)
  B = A + garch_h, abs_gap_z, since_shift, log n_obs      (8 feats, 36 params)
Best OOF cell saved as V2."""
import json, time, sys, os
import numpy as np
from scipy.optimize import minimize
import volfeat as V
MEM = ["TM_COMBINED", "C2", "TM_E7", "TM_MIXER", "TM_RULE"]
FS = {"A": ["ewv_w16", "vol_of_vol", "peer_mean_ewv_w16", "obs_frac"],
      "B": ["ewv_w16", "vol_of_vol", "peer_mean_ewv_w16", "obs_frac", "garch_h", "abs_gap_z", "since_shift", "log_n_obs"]}


def make_Z(names, F, fs, dev_rows):
    cols = []
    for f in FS[fs]:
        x = np.log(F[:, names.index("n_obs")]) if f == "log_n_obs" else F[:, names.index(f)]
        cols.append((x - x[dev_rows].mean()) / (x[dev_rows].std() + 1e-12))
    return np.column_stack(cols)


def gates(x, Z, K):
    d = Z.shape[1]
    th = x.reshape(K - 1, d + 1)
    L = np.column_stack([np.zeros(len(Z))] + [th[k, 0] + Z @ th[k, 1:] for k in range(K - 1)])
    L -= L.max(1, keepdims=True)
    G = np.exp(L); return G / G.sum(1, keepdims=True)


def fit_gate(Pm, Z, q, ww, lam):
    K, d = len(Pm), Z.shape[1]
    A = np.stack(Pm)                      # K,n,5
    ww = ww / ww.sum()
    def loss(x):
        G = gates(x, Z, K)
        P = np.einsum("nk,knc->nc", G, A)
        r = P - q
        th = x.reshape(K - 1, d + 1)
        return float((ww * (r ** 2).sum(1)).sum() + lam * (th[:, 1:] ** 2).sum())
    x0 = np.zeros((K - 1) * (d + 1))
    # start at plain simplex-ish: small prior toward reference
    r = minimize(loss, x0, method="L-BFGS-B", options={"maxiter": 200})
    return r.x


def run(fs, lam, sc=None):
    v, names, F, dev, rows, arows, w, fold = V.setup() if sc is None else sc
    Pm = [V.read_pred_csv(V.HERE / "preds" / f"{m}_validation.csv", v) for m in MEM]
    Z = make_Z(names, F, fs, rows)
    K = len(MEM)
    fitfn = lambda tr: fit_gate([p[tr] for p in Pm], Z[tr], v.truth[tr], w[tr], lam)
    def predfn(x, te):
        G = gates(x, Z[te], K)
        return np.einsum("nk,knc->nc", G, np.stack([p[te] for p in Pm]))
    out, m = V.oof_run(v, dev, fold, rows, arows, fitfn, predfn)
    return out, m, Z


if __name__ == "__main__":
    sc = V.setup(); v = sc[0]
    base = V.read_pred_csv(V.HERE / "preds" / "TM_COMBINED_validation.csv", v)
    grid = []; best = None
    for fs in "AB":
        for lam in (3e-3, 3e-2, 3e-1):
            t0 = time.time()
            out, m, Z = run(fs, lam, sc)
            ev, d = V.report(out, v, base, f"V2 fs={fs} lam={lam}")
            grid.append({"params": {"feat": fs, "lam": lam, "n_params": (len(MEM) - 1) * (Z.shape[1] + 1)},
                         "points": ev["points"], "ci95": ev["ci95"], "condition_ratios": ev["condition_ratios"],
                         "diff_vs_TMC": d["diff"], "diff_ci": d["ci95"]})
            if best is None or ev["points"] > best[0]: best = (ev["points"], out, m, fs, lam, Z)
    _, out, m, fs, lam, Z = best
    from harness import to_csv, write_result
    to_csv(V.HERE / "preds" / "V2_validation.csv", v, out)
    prm = {"feat": fs, "lam": lam, "members": MEM, "feats": FS[fs], "n_params": len(m), "settings_tried": len(grid)}
    write_result("V2", "sai-worker", "volatility-gated blend of TM_COMBINED,C2,TM_E7,TM_MIXER,TM_RULE (episode-OOF)", prm, v, out, 0.0)
    (V.HERE / "grids" / "V2.json").write_text(json.dumps(grid, indent=1))
    print("best", fs, lam, "theta(final fit):", np.round(m.reshape(len(MEM) - 1, -1), 2).tolist())
