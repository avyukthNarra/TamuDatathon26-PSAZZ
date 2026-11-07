"""Post-hoc experiments H1-H5 on saved validation predictions.

Every fitted parameter is learned inside a 6-fold CV over DEV episodes (sorted episode id, i % 6),
so every reported score is out-of-fold. Scope 'dev' only (never audit/all). Gates use only
nobs (observable) and mtype; v.condition is used for reporting (via harness scoring) only.

Scored array: dev rows = OOF predictions; non-dev rows = full-dev-fit predictions (not scored).
Run: OMP_NUM_THREADS=1 nice -n 10 python3 improve/gated.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

HERE = Path(__file__).resolve().parent
SAI = HERE.parent
sys.path.insert(0, str(SAI))
from harness.data import load_split  # noqa: E402
from harness.results import read_pred_csv, to_csv  # noqa: E402
from harness.scoring import diff_ci, evaluate, scope_episodes  # noqa: E402

OUT = HERE / "gated_results.json"
v = load_split("validation")
T = v.truth
N = v.n
MEMBERS7 = ["TM_COMBINED3", "TM_RULE", "TM_MIXER3", "C2", "M9b_B4", "TM_E67", "TM_MIXER4"]
MEMBERS3 = ["TM_COMBINED3", "C2", "TM_RULE"]
P = {n: read_pred_csv(SAI / "preds" / f"{n}_validation.csv", v) for n in set(MEMBERS7) | set(MEMBERS3)}
C3 = P["TM_COMBINED3"]

nobs = (~np.isnan(v.grid)).sum(1)
short = (nobs == 64).astype(int)
MT = {"balanced": 0, "jackpot": 1, "steady": 2}
mt = np.array([MT[m] for m in v.mtype])
cell6 = mt * 2 + short
cell2 = short
GROUPS = {"global": np.zeros(N, int), "mtype": mt, "mtype_short": cell6}

dev_eps = scope_episodes(v, "dev")
dev_rows = np.flatnonzero(np.isin(v.episode, dev_eps))
fold_of = {e: i % 6 for i, e in enumerate(sorted(dev_eps))}
fold = np.array([fold_of.get(e, -1) for e in v.episode.tolist()])
REST = np.flatnonzero(fold < 0)


def inner_oof(base_fit, tr, k=5):
    """Out-of-fold base predictions on training rows tr (for nested recalibration)."""
    eps = v.episode[tr].tolist()
    ue = sorted(set(eps))
    fo = {e: i % k for i, e in enumerate(ue)}
    f = np.array([fo[e] for e in eps])
    out = np.zeros((len(tr), 5))
    for j in range(k):
        out[f == j] = base_fit(tr[f != j])(tr[f == j])
    return out


# ---------- H1: win-prob logit recalibration ----------
def _h1_transform(p, a, b):
    pw = np.clip(1 - p[:, 0], 1e-4, 1 - 1e-4)
    pw2 = 1 / (1 + np.exp(-(a + b * np.log(pw / (1 - pw)))))
    s = p[:, 1:] + 1e-12
    s = s / s.sum(1, keepdims=True)
    return np.column_stack([1 - pw2, pw2[:, None] * s])


def _h1_loss(ab, p, t):
    return ((_h1_transform(p, ab[0], ab[1]) - t) ** 2).sum()


def h1_method(base_fit, gname):
    g = GROUPS[gname]
    ng = int(g.max()) + 1

    def fit(tr):
        base_pred = base_fit(tr)
        pb = inner_oof(base_fit, tr)  # nested: base preds on train rows are out-of-fold
        params = {}
        for k in range(ng):
            m = g[tr] == k
            if not m.any():
                params[k] = (0.0, 1.0)
                continue
            res = minimize(_h1_loss, x0=[0.0, 1.0], args=(pb[m], T[tr][m]), method="Nelder-Mead",
                           options={"xatol": 1e-6, "fatol": 1e-12, "maxiter": 4000})
            params[k] = (float(res.x[0]), float(res.x[1]))

        def pred(idx):
            out = base_pred(idx).copy()
            for k, (a, b) in params.items():
                m = g[idx] == k
                if m.any():
                    out[m] = _h1_transform(out[m], a, b)
            return out

        pred.params = {int(k): v_ for k, v_ in params.items()}
        return pred

    return fit


# ---------- H2: cell-gated simplex blends ----------
def simplex_fit(Ms, Tr):
    K = Ms.shape[0]
    if Ms.shape[1] == 0:
        return np.ones(K) / K

    def f(w):
        return ((np.tensordot(w, Ms, 1) - Tr) ** 2).sum()

    def g(w):
        return 2 * np.einsum("kij,ij->k", Ms, np.tensordot(w, Ms, 1) - Tr)

    res = minimize(f, np.ones(K) / K, jac=g, method="SLSQP", bounds=[(0, 1)] * K,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1,
                                 "jac": lambda w: np.ones(K)}], options={"maxiter": 500})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def blend_method(members, cellarr, ncell):
    def fit(tr):
        W = []
        for c in range(ncell):
            r = tr[cellarr[tr] == c]
            W.append(simplex_fit(np.stack([P[n][r] for n in members]), T[r]))

        def pred(idx):
            out = np.zeros((len(idx), 5))
            for c in range(ncell):
                m = cellarr[idx] == c
                if m.any():
                    out[m] = np.tensordot(W[c], np.stack([P[n][idx[m]] for n in members]), 1)
            return out

        pred.W = [w.round(4).tolist() for w in W]
        return pred

    return fit


# ---------- H3: factorised (win-prob blend and size-share blend, separate simplex weights) ----------
def factor_fit(Ms, Tr):
    K = Ms.shape[0]
    if Ms.shape[1] == 0:
        return np.ones(K) / K, np.ones(K) / K
    PW = 1 - Ms[:, :, 0]
    S = Ms[:, :, 1:] / np.maximum(Ms[:, :, 1:].sum(2, keepdims=True), 1e-12)

    def pof(x):
        pw = x[:K] @ PW
        s = np.tensordot(x[K:], S, 1)
        return np.column_stack([1 - pw, pw[:, None] * s])

    cons = [{"type": "eq", "fun": lambda x: x[:K].sum() - 1},
            {"type": "eq", "fun": lambda x: x[K:].sum() - 1}]
    res = minimize(lambda x: ((pof(x) - Tr) ** 2).sum(), np.r_[np.ones(K) / K, np.ones(K) / K],
                   method="SLSQP", bounds=[(0, 1)] * (2 * K), constraints=cons, options={"maxiter": 500})
    a = np.clip(res.x[:K], 0, None)
    b = np.clip(res.x[K:], 0, None)
    return a / a.sum(), b / b.sum()


def factor_method(members, cellarr, ncell):
    def fit(tr):
        W = []
        for c in range(ncell):
            r = tr[cellarr[tr] == c]
            W.append(factor_fit(np.stack([P[n][r] for n in members]), T[r]))

        def pred(idx):
            out = np.zeros((len(idx), 5))
            for c in range(ncell):
                m = cellarr[idx] == c
                if not m.any():
                    continue
                Ms = np.stack([P[n][idx[m]] for n in members])
                a, b = W[c]
                pw = np.tensordot(a, 1 - Ms[:, :, 0], 1)
                S = Ms[:, :, 1:] / np.maximum(Ms[:, :, 1:].sum(2, keepdims=True), 1e-12)
                s = np.tensordot(b, S, 1)
                out[m] = np.column_stack([1 - pw, pw[:, None] * s])
            return out

        pred.W = [[x.round(4).tolist() for x in w] for w in W]
        return pred

    return fit


# ---------- H4: short-history shrink toward per-mtype prior ----------
def _shrink_loss(w, c3, q, t):
    return (((1 - w) * c3 + w * q - t) ** 2).sum()


def shrink_fit(tr):
    wq = {}
    for m in range(3):
        lt = tr[(mt[tr] == m) & (short[tr] == 0)]
        q = C3[lt].mean(0)
        st = tr[(mt[tr] == m) & (short[tr] == 1)]
        w = minimize_scalar(_shrink_loss, bounds=(0, 1), method="bounded",
                            args=(C3[st], q, T[st])).x if len(st) else 0.0
        wq[m] = (float(w), q)

    def pred(idx):
        out = C3[idx].copy()
        for m in range(3):
            sel = (mt[idx] == m) & (short[idx] == 1)
            if sel.any():
                w, q = wq[m]
                out[sel] = (1 - w) * C3[idx[sel]] + w * q
        return out

    pred.w = {k: w for k, (w, _) in wq.items()}
    return pred


def fixed_fit(tr):
    return lambda idx: C3[idx]


# ---------- driver ----------
def run(fit):
    """OOF over dev episodes; full-dev fit fills non-dev rows (never scored)."""
    Pout = np.zeros((N, 5))
    for k in range(6):
        tr = dev_rows[fold[dev_rows] != k]
        te = dev_rows[fold[dev_rows] == k]
        Pout[te] = fit(tr)(te)
    full = fit(dev_rows)
    Pout[REST] = full(REST)
    return Pout / Pout.sum(1, keepdims=True), full


def score(Pout):
    ev = evaluate(Pout, v, "dev", bootstrap=1000)
    d = diff_ci(Pout, C3, v, "dev", bootstrap=1000)
    return {"points": ev["points"], "ci95": ev["ci95"], "R": ev["R"],
            "condition_ratios": ev["condition_ratios"], "n_dev_episodes": ev["n_episodes"],
            "diff_vs_C3": d["diff"], "diff_ci95": d["ci95"]}


def main():
    results = {"reference_TM_COMBINED3": score(C3), "experiments": {}}
    ref = results["reference_TM_COMBINED3"]
    print(f"REF TM_COMBINED3 dev {ref['points']:.3f} CI {np.round(ref['ci95'], 2)}")

    def record(exp, name, fit):
        Pout, full = run(fit)
        s = score(Pout)
        s["full_fit_params"] = getattr(full, "W", None) or getattr(full, "params", None) or getattr(full, "w", None)
        results["experiments"].setdefault(exp, {})[name] = s
        print(f"{exp:4s} {name:28s} {s['points']:.3f} CI {np.round(s['ci95'], 2)} "
              f"diff {s['diff_vs_C3']:+.3f} [{s['diff_ci95'][0]:+.2f},{s['diff_ci95'][1]:+.2f}]",
              flush=True)
        return Pout, fit, s

    store = {}  # (exp, name) -> (Pout, fit, score)
    # H1 on the fixed reference
    for g in GROUPS:
        store[("H1", g)] = record("H1", g, h1_method(fixed_fit, g))
    # H2 cell-gated blends
    store[("H2", "6cell_7mem")] = record("H2", "6cell_7mem", blend_method(MEMBERS7, cell6, 6))
    store[("H2", "6cell_3mem")] = record("H2", "6cell_3mem", blend_method(MEMBERS3, cell6, 6))
    store[("H2", "2cell_7mem")] = record("H2", "2cell_7mem", blend_method(MEMBERS7, cell2, 2))
    store[("H2", "2cell_3mem")] = record("H2", "2cell_3mem", blend_method(MEMBERS3, cell2, 2))
    # H3 factorised
    store[("H3", "6cell_7mem")] = record("H3", "6cell_7mem", factor_method(MEMBERS7, cell6, 6))
    store[("H3", "6cell_3mem")] = record("H3", "6cell_3mem", factor_method(MEMBERS3, cell6, 6))
    # H4 short-history shrink
    store[("H4", "shrink_per_mtype")] = record("H4", "shrink_per_mtype", shrink_fit)
    # H5: H1 recalibration nested on top of the best H2/H3 variant
    base_key = max([k for k in store if k[0] in ("H2", "H3")], key=lambda k: store[k][2]["points"])
    base_fit = store[base_key][1]
    results["H5_base"] = f"{base_key[0]}/{base_key[1]}"
    print("H5 base:", results["H5_base"])
    for g in GROUPS:
        store[("H5", g)] = record("H5", g, h1_method(base_fit, g))

    # best variant per experiment -> preds/IMP_H<k>_validation.csv
    results["best"] = {}
    for exp in ["H1", "H2", "H3", "H4", "H5"]:
        keys = [k for k in store if k[0] == exp]
        bk = max(keys, key=lambda k: store[k][2]["points"])
        results["best"][exp] = {"variant": bk[1], "points": store[bk][2]["points"],
                                "diff_vs_C3": store[bk][2]["diff_vs_C3"],
                                "diff_ci95": store[bk][2]["diff_ci95"]}
        to_csv(SAI / "preds" / f"IMP_{exp}_validation.csv", v, store[bk][0])
    results["meta"] = {"members7": MEMBERS7, "members3": MEMBERS3, "dev_episodes": len(dev_eps),
                       "dev_rows": int(len(dev_rows)), "folds": 6,
                       "note": "dev rows OOF; non-dev rows full-dev-fit (never scored). "
                               "Best-variant choice within each experiment and H5 base are selected on "
                               "the same dev OOF, so their points are mildly optimistic."}
    OUT.write_text(json.dumps(results, indent=1, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
