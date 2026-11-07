"""G1 = observable-evidence gated linear pool (no volatility features).
P_i = sum_m w_m(x_i) P_m(i), w = softmax_m(b_m + g_m . z_i), z = observable features only: observed-tick count, fraction of
missing ticks, cutoff, machine_type (one-hot), signal. Fit on true q (R-weighted 5-class squared error), ridge on g, episode-OOF
(6 folds, same hash as W5) on dev; audit rows = fit on all dev. Members (7, chosen from the error-correlation matrix, see
reports/G1.md): TM_MIXER TM_RULE TM_E7 TM_E6 C2 L7 L5.
Grid: gate in {none(global), binned(nobs tercile x type, shrunk), smooth(all 7 features), smooth_small(nobs,fm,cut)}
      x lam in {1e-4,1e-3,1e-2,1e-1}. Best cell chosen by dev OOF points (mild selection optimism)."""
import json, sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import numpy as np
from G_common import *
from G_gate import fit_gate, pred_gate

MEMBERS = ["TM_MIXER", "TM_RULE", "TM_E7", "TM_E6", "C2", "L7", "L5"]


def zmat(v, kind, rows_ref):
    f = obs_features(v)
    cols = {"nobs": f["nobs"], "fm": f["fm"], "cut": f["cut"], "sig": f["sig"]}
    if kind == "none":
        return np.zeros((v.n, 0))
    if kind == "binned":
        t = np.percentile(f["nobs"][rows_ref], [33.3, 66.7])          # edges learned on dev only
        b = np.digitize(f["nobs"], t)
        return np.column_stack([(b == 1), (b == 2)] + [f["oh"][:, j] for j in (1, 2)]).astype(float)
    names = ["nobs", "fm", "cut", "sig"] if kind == "smooth" else ["nobs", "fm", "cut"]
    Z = np.column_stack([cols[k] for k in names] + ([f["oh"][:, 1:]] if kind == "smooth" else []))
    mu, sd = Z[rows_ref].mean(0), Z[rows_ref].std(0) + 1e-9
    return (Z - mu) / sd


def run(v, rows, arows, w, folds, kind, lam, members=MEMBERS):
    Pm = np.stack([load(v, n) for n in members])
    Z = zmat(v, kind, rows)
    fitfn = lambda tr: fit_gate(Pm[:, tr], Z[tr], v.truth[tr], w[tr], lam)
    predfn = lambda B, te: pred_gate(B, Pm[:, te], Z[te])
    P, B = run_cv(v, rows, arows, folds, fitfn, predfn)
    train_err = None
    return P, B, Z.shape[1]


def predict_all():
    v, rows, arows, w, folds = setup()
    TMC = load(v, "TM_COMBINED"); TS = team_stack_oof(v, rows, w, folds)
    grid = []; best = None
    for kind in ["none", "binned", "smooth_small", "smooth"]:
        for lam in ([0.0] if kind == "none" else [1e-4, 1e-3, 1e-2, 1e-1]):
            P, B, k = run(v, rows, arows, w, folds, kind, lam)
            r = report(P, v, TMC, TS, f"{kind}_{lam}")
            npar = len(MEMBERS) * (1 + k) - 1
            print(kind, lam, npar, fmt(r), flush=True)
            grid.append({"params": {"gate": kind, "lam": lam, "n_params": npar}, "points": r["points"],
                         "ci95": r["ci95"], "condition_ratios": r["ratios"], "vs_TMC": r["vs_TMC"], "vs_TEAM": r["vs_TEAM"]})
            if best is None or r["points"] > best[0]:
                best = (r["points"], kind, lam, P, B, r, npar)
    return v, grid, best


if __name__ == "__main__":
    t0 = time.perf_counter()
    v, grid, (pts, kind, lam, P, B, r, npar) = predict_all()
    rt = time.perf_counter() - t0
    print("BEST", kind, lam, npar, fmt(r))
    to_csv(HERE / "preds/G1_validation.csv", v, P)
    (HERE / "grids/G1.json").write_text(json.dumps(grid, indent=1))
    from harness import write_result
    write_result("G1", "claude", f"observable gated linear pool ({kind}, lam={lam})", {"gate": kind, "lam": lam, "members": MEMBERS,
                 "n_params": npar}, v, P, rt, extra={"vs_TM_COMBINED": r["vs_TMC"], "vs_TEAM_stack": r["vs_TEAM"]})
    print("coef (members x [b, g...]):\n", np.round(B, 2))
