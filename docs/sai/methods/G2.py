"""G2 = per-class stacking and log-linear pool of the same 7 members as G1 (TM_MIXER TM_RULE TM_E7 TM_E6 C2 L7 L5).
Variants, all fit on true q (R-weighted 5-class squared error), episode-OOF on dev, audit rows = fit on all dev:
  pwsh   : W5 factorisation (pw simplex x shares simplex), 7+7-2 params      [reference]
  pc     : per-class weights, a_{m,c}=softmax_m(s_mc), P_c=sum_m a_mc P_mc then row-renormalise, 35-5 params
  pc_win : per-class weights only for the 4 win classes + shared weights for the loss class (kept simple), 
  log    : geometric pool P_c ~ prod_m P_mc^{a_m}, a_m>=0 (softplus), renormalised, 7 params
  log_pc : geometric pool with per-class exponents, 35 params
Optimiser: L-BFGS-B with finite-difference gradient (cheap: 7 x 576 rows), several ridge strengths toward uniform weights."""
import json, sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import numpy as np
from scipy.optimize import minimize
from G_common import *
from G1 import MEMBERS

EPS = 1e-6


def sm(S, axis=0):
    S = S - S.max(axis, keepdims=True); E = np.exp(S); return E / E.sum(axis, keepdims=True)


def make(kind, Pm, q, ww, lam):
    M = Pm.shape[0]
    L = np.log(np.clip(Pm, EPS, 1))
    if kind == "pc":
        shape = (M, 5)
        def pred(x, Pm=Pm):
            A = sm(x.reshape(shape), 0)
            P = np.einsum("mc,mnc->nc", A, Pm); return P / P.sum(1, keepdims=True)
    elif kind == "log":
        shape = (M,)
        def pred(x, Pm=Pm):
            a = np.logaddexp(0, x)
            Lg = np.einsum("m,mnc->nc", a, np.log(np.clip(Pm, EPS, 1))); return sm(Lg, 1)
    elif kind == "log_pc":
        shape = (M, 5)
        def pred(x, Pm=Pm):
            a = np.logaddexp(0, x.reshape(shape))
            Lg = np.einsum("mc,mnc->nc", a, np.log(np.clip(Pm, EPS, 1))); return sm(Lg, 1)
    return shape, pred


def fit_kind(kind, Pm, q, ww, lam):
    ww = ww / ww.sum()
    shape, pred = make(kind, Pm, q, ww, lam)
    x0 = np.zeros(int(np.prod(shape)))
    if kind.startswith("log"):
        x0 += np.log(np.expm1(1.0 / Pm.shape[0]))          # start at the plain geometric mean
    def f(x):
        P = pred(x)
        return (ww * ((P - q) ** 2).sum(1)).sum() + lam * ((x - x0) ** 2).sum()
    r = minimize(f, x0, method="L-BFGS-B", options={"maxiter": 200, "ftol": 1e-12})
    return r.x


def predict_kind(kind, x, Pm):
    return make(kind, Pm, None, None, 0)[1](x, Pm)


def main():
    v, rows, arows, w, folds = setup()
    Pm = np.stack([load(v, n) for n in MEMBERS])
    TMC = load(v, "TM_COMBINED"); TS = team_stack_oof(v, rows, w, folds)
    grid = []; best = None
    # reference: W5 factorisation on same 7 members
    PWm = [parts(m)[0] for m in Pm]; SHm = [parts(m)[1] for m in Pm]
    P = np.tile(np.array([0.5, 0.2, 0.15, 0.1, 0.05]), (v.n, 1))
    for tr, te in folds:
        a, b = w5fit(PWm, SHm, v, tr, w); P[te] = w5combine([m[te] for m in PWm], [m[te] for m in SHm], a, b)
    a, b = w5fit(PWm, SHm, v, rows, w); P[arows] = w5combine([m[arows] for m in PWm], [m[arows] for m in SHm], a, b)
    r = report(P, v, TMC, TS, "pwsh"); print("pwsh", 12, fmt(r), flush=True)
    grid.append({"params": {"kind": "pwsh", "lam": 0, "n_params": 12}, "points": r["points"], "ci95": r["ci95"],
                 "condition_ratios": r["ratios"], "vs_TMC": r["vs_TMC"], "vs_TEAM": r["vs_TEAM"]})
    for kind in ["pc", "log", "log_pc"]:
        npar = {"pc": 28, "log": 7, "log_pc": 35}[kind]
        for lam in [0.0, 1e-4, 1e-3, 1e-2]:
            fitfn = lambda tr: fit_kind(kind, Pm[:, tr], v.truth[tr], w[tr], lam)
            predfn = lambda x, te: predict_kind(kind, x, Pm[:, te])
            P, x = run_cv(v, rows, arows, folds, fitfn, predfn)
            r = report(P, v, TMC, TS, kind); print(kind, lam, npar, fmt(r), flush=True)
            grid.append({"params": {"kind": kind, "lam": lam, "n_params": npar}, "points": r["points"], "ci95": r["ci95"],
                         "condition_ratios": r["ratios"], "vs_TMC": r["vs_TMC"], "vs_TEAM": r["vs_TEAM"]})
            if best is None or r["points"] > best[0]:
                best = (r["points"], kind, lam, P, x, r, npar)
    return v, grid, best


if __name__ == "__main__":
    t0 = time.perf_counter()
    v, grid, (pts, kind, lam, P, x, r, npar) = main()
    rt = time.perf_counter() - t0
    print("BEST", kind, lam, npar, fmt(r))
    if kind == "pc": print(np.round(sm(x.reshape(len(MEMBERS), 5), 0), 2))
    elif kind == "log": print(np.round(np.logaddexp(0, x), 2))
    else: print(np.round(np.logaddexp(0, x.reshape(len(MEMBERS), 5)), 2))
    to_csv(HERE / "preds/G2_validation.csv", v, P)
    (HERE / "grids/G2.json").write_text(json.dumps(grid, indent=1))
    from harness import write_result
    write_result("G2", "claude", f"per-class / log-linear pool ({kind}, lam={lam})", {"kind": kind, "lam": lam,
                 "members": MEMBERS, "n_params": npar}, v, P, rt, extra={"vs_TM_COMBINED": r["vs_TMC"], "vs_TEAM_stack": r["vs_TEAM"]})
