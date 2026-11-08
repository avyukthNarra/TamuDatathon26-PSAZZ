"""W5 stack. P = [1-pw, pw*shares] with pw = sum a_i pw_i (win-probability members) and shares = sum b_j shares_j
(size-share members), a and b on simplices, fit to the TRUE validation q (5-class squared error, R-weighted rows).
Episode-OOF (6 folds) on dev gives the honest dev number; final weights are fit on all dev and scored ONCE on audit.
  python3 -m harness.stack_w5 dev      # OOF on dev only, audit untouched
  python3 -m harness.stack_w5 audit    # final fit on dev, scores audit once (logged)
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import load_split, read_pred_csv, evaluate, diff_ci
from harness.data import CONDITIONS
from harness.scoring import scope_episodes

HERE = Path(__file__).resolve().parents[1]
PW = ["M9b_B4", "M2_B4", "M3_B4", "L5", "L7", "B4", "S2", "S1"]          # win-probability members
SH = ["C1", "A9", "A1_B4", "B4", "L5", "S2", "S1"]                        # size-share members
FOLDS = 6
import os
TEAM = os.environ.get("TEAM") == "1"          # add the teammates' models as extra members
if TEAM:
    PW += ["TM_MIXER", "TM_COMBINED", "TM_RULE", "TM_E6"]
    SH += ["TM_MIXER", "TM_COMBINED", "TM_RULE"]
TAG = "team_" if TEAM else ""


def load(v, name):
    p = HERE / "preds" / f"{name}_validation.csv"
    if not p.exists():
        p = HERE / "results" / "preds" / f"{name}_validation.csv"        # Codex layout
    return read_pred_csv(p, v)


def parts(P):
    pw = 1 - P[:, 0]
    return pw, P[:, 1:] / P[:, 1:].sum(1, keepdims=True)


def row_weights(v, rows):
    """R-objective weights: 1 / (episodes in condition * rows in episode * reference MSE of the condition)."""
    w = np.zeros(v.n)
    refmse = {c: ((v.ref - v.truth) ** 2).sum(1)[[i for i in rows if v.condition[i] == c]].mean() for c in CONDITIONS}
    for c in CONDITIONS:
        eps = sorted({v.episode[i] for i in rows if v.condition[i] == c})
        for e in eps:
            r = [i for i in v.episode_rows(e) if i in set(rows)]
            for i in r:
                w[i] = 1.0 / (len(eps) * len(r) * refmse[c])
    return w


def fit(PWm, SHm, v, rows, w):
    npw, nsh = len(PWm), len(SHm)
    pwA = np.stack([m[rows] for m in PWm])            # (npw, n)
    shA = np.stack([m[rows] for m in SHm])            # (nsh, n, 4)
    q, ww = v.truth[rows], w[rows]

    def loss(x):
        a, b = x[:npw], x[npw:]
        pw = a @ pwA
        sh = np.tensordot(b, shA, axes=1)
        P = np.column_stack([1 - pw, pw[:, None] * sh])
        return float((ww * ((P - q) ** 2).sum(1)).sum())

    x0 = np.r_[np.ones(npw) / npw, np.ones(nsh) / nsh]
    cons = [{"type": "eq", "fun": lambda x: x[:npw].sum() - 1}, {"type": "eq", "fun": lambda x: x[npw:].sum() - 1}]
    r = minimize(loss, x0, method="SLSQP", bounds=[(0, 1)] * (npw + nsh), constraints=cons,
                 options={"maxiter": 300, "ftol": 1e-12})
    return r.x[:npw], r.x[npw:]


def combine(PWm, SHm, a, b, rows=None):
    pw = a @ np.stack(PWm)
    sh = np.tensordot(b, np.stack(SHm), axes=1)
    return np.column_stack([1 - pw, pw[:, None] * sh])


def main(mode):
    v = load_split("validation")
    PWm = [parts(load(v, n))[0] for n in PW]
    SHm = [parts(load(v, n))[1] for n in SH]
    dev = scope_episodes(v, "dev")
    rows = np.concatenate([v.episode_rows(e) for e in dev])
    w = row_weights(v, set(rows.tolist()))
    order = sorted(dev, key=lambda e: hashlib.sha256(f"w5fold|{e}".encode()).hexdigest())
    fold = {e: i % FOLDS for i, e in enumerate(order)}
    oof = np.tile(np.array([0.5, 0.2, 0.15, 0.1, 0.05]), (v.n, 1))
    wa, wb = [], []
    for f in range(FOLDS):
        tr = np.concatenate([v.episode_rows(e) for e in dev if fold[e] != f])
        te = np.concatenate([v.episode_rows(e) for e in dev if fold[e] == f])
        a, b = fit(PWm, SHm, v, tr, w)
        wa.append(a); wb.append(b)
        oof[te] = combine([m[te] for m in PWm], [m[te] for m in SHm], a, b)
    ev = evaluate(oof, v, "dev", bootstrap=1000)
    C2 = load(v, "C2"); B4 = load(v, "B4"); TMC = load(v, "TM_COMBINED") if TEAM else C2
    print(f"STACK dev OOF: {ev['points']:.3f} CI {[round(x,2) for x in ev['ci95']]} ratios",
          {k[:5]: round(x, 3) for k, x in ev["condition_ratios"].items()})
    print("  vs C2:", {k: (round(x, 3) if not isinstance(x, list) else [round(y, 2) for y in x]) for k, x in diff_ci(oof, C2, v, "dev").items()},
          " vs B4:", {k: (round(x, 3) if not isinstance(x, list) else [round(y, 2) for y in x]) for k, x in diff_ci(oof, B4, v, "dev").items()})
    print("  vs TM_COMBINED:" if TEAM else "  (vs C2 again):", {k: (round(x, 3) if not isinstance(x, list) else [round(y, 2) for y in x]) for k, x in diff_ci(oof, TMC, v, "dev").items()})
    print("  mean pw weights :", {n: round(float(np.mean([x[i] for x in wa])), 2) for i, n in enumerate(PW)})
    print("  mean share wts  :", {n: round(float(np.mean([x[i] for x in wb])), 2) for i, n in enumerate(SH)})
    a, b = fit(PWm, SHm, v, rows, w)
    final = combine(PWm, SHm, a, b)
    out = {"pw_members": PW, "share_members": SH, "pw_weights": a.tolist(), "share_weights": b.tolist(),
           "dev_oof_points": ev["points"], "dev_oof_ci95": ev["ci95"]}
    if mode == "audit":
        ea = evaluate(final, v, "audit", who="W5_final", bootstrap=1000)
        out["audit"] = {"stack": ea}
        for name in ("B4", "C2", "B3"):
            out["audit"][name] = evaluate(load(v, name), v, "audit", who="W5_final_ref", bootstrap=1000)
        out["audit"]["stack_vs_B4"] = diff_ci(final, B4, v, "audit", who="W5_final")
        out["audit"]["stack_vs_C2"] = diff_ci(final, C2, v, "audit", who="W5_final")
        out["audit"]["C2_vs_B4"] = diff_ci(C2, B4, v, "audit", who="W5_final")
        print("AUDIT stack:", round(ea["points"], 3), [round(x, 2) for x in ea["ci95"]], ea["condition_ratios"])
        for k in ("B3", "B4", "C2"):
            print(f"AUDIT {k}:", round(out["audit"][k]["points"], 3), [round(x, 2) for x in out["audit"][k]["ci95"]])
        print("AUDIT stack vs B4:", out["audit"]["stack_vs_B4"], " vs C2:", out["audit"]["stack_vs_C2"])
    (HERE / "results" / f"W5_{TAG}{mode}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
