"""K2: small-subset two-stage blends + observable gates (dev only, episode-OOF 6 folds, R-weighted rows).

Pre-declared grid (written before the full run):
  Step 1: win set W (1-3 of TM_COMBINED3, TM_MIXER3, TM_RULE, TM_E7, M9b_B4, L7)  -> 41 subsets
          size set S (1-3 of TM_COMBINED3, TM_MIXER3, C1, A9, C2, TM_RULE)        -> 41 subsets
          41 x 41 = 1681 structures; each fit with stack_w5.fit on 6 training folds, OOF on dev,
          scored with evaluate(oof, v, "dev", bootstrap=0). Top 10 -> grids/K2.json.
          Top 3 -> diff_ci vs BEST (TM_COMBINED3), bootstrap 500.
  Step 2: gated convex pairs w(z)=sigmoid(a+b*z) between two full prediction vectors, pairs among
          TM_COMBINED3, C2, TM_E7, A9 (6 pairs). Gate variants: ungated (1 weight), observed-tick fraction (a,b),
          cutoff (a,b), machine type (3 weights). Fit by R-weighted squared error (BFGS, 3 starts), OOF on dev.
          Step-1 structures are NOT used as gate inputs: their dev OOF predictions were fit on rows that the
          gate would train on, which would leak.
  Audit rows of the saved preds: the chosen structure fit on ALL dev rows (never scored).
  Run: python3 methods/K2.py            (add --smoke for a few-combo check, writes nothing)
"""
import hashlib, itertools, json, sys, time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from harness import load_split, evaluate, diff_ci, write_result, to_csv
from harness.scoring import scope_episodes
from harness.stack_w5 import load, parts, row_weights, fit, combine

SMOKE = "--smoke" in sys.argv
W = ["TM_COMBINED3", "TM_MIXER3", "TM_RULE", "TM_E7", "M9b_B4", "L7"]
S = ["TM_COMBINED3", "TM_MIXER3", "C1", "A9", "C2", "TM_RULE"]
GATE_MEMBERS = ["TM_COMBINED3", "C2", "TM_E7", "A9"]
BEST_NAME = "TM_COMBINED3"
DEFAULT = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
t0 = time.time()

v = load_split("validation")
M = {n: load(v, n) for n in set(W) | set(S) | set(GATE_MEMBERS)}
PWv = {n: parts(M[n])[0] for n in M}
SHv = {n: parts(M[n])[1] for n in M}
BEST = M[BEST_NAME]

dev = scope_episodes(v, "dev")
aud = scope_episodes(v, "audit")
rows = np.concatenate([v.episode_rows(e) for e in dev])
arows = np.concatenate([v.episode_rows(e) for e in aud])
w = row_weights(v, set(rows.tolist()))
order = sorted(dev, key=lambda e: hashlib.sha256(f"w5fold|{e}".encode()).hexdigest())
fold = {e: i % 6 for i, e in enumerate(order)}
FOLDS = [(np.concatenate([v.episode_rows(e) for e in dev if fold[e] != f]),
          np.concatenate([v.episode_rows(e) for e in dev if fold[e] == f])) for f in range(6)]
Q = v.truth


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------- step 1: two-stage structures ----------------
def struct_oof(pw_names, sh_names):
    PWm = [PWv[n] for n in pw_names]
    SHm = [SHv[n] for n in sh_names]
    oof = np.tile(DEFAULT, (v.n, 1))
    for tr, te in FOLDS:
        a, b = fit(PWm, SHm, v, tr, w)
        oof[te] = combine([m[te] for m in PWm], [m[te] for m in SHm], a, b)
    return oof


def struct_full(pw_names, sh_names, idx):
    PWm = [PWv[n] for n in pw_names]
    SHm = [SHv[n] for n in sh_names]
    a, b = fit(PWm, SHm, v, rows, w)
    return combine([m[idx] for m in PWm], [m[idx] for m in SHm], a, b)


wsets = [c for k in (1, 2, 3) for c in itertools.combinations(W, k)]
ssets = [c for k in (1, 2, 3) for c in itertools.combinations(S, k)]
if SMOKE:
    wsets, ssets = wsets[:3], ssets[:3]

step1 = []
for pwn in wsets:
    for shn in ssets:
        oof = struct_oof(list(pwn), list(shn))
        pts = evaluate(oof, v, "dev", bootstrap=0)["points"]
        step1.append({"win": list(pwn), "shares": list(shn), "points": pts, "oof": oof})
print(f"step1 combos searched: {len(step1)}  ({time.time() - t0:.0f}s)", flush=True)
step1.sort(key=lambda d: -d["points"])
top10 = [{k: d[k] for k in ("win", "shares", "points")} for d in step1[:10]]
for d in top10:
    print(f"  {d['points']:.3f}  win={d['win']}  shares={d['shares']}")

top3_diff = []
for d in step1[:3]:
    dc = diff_ci(d["oof"], BEST, v, "dev", bootstrap=500)
    top3_diff.append({"win": d["win"], "shares": d["shares"], "points": d["points"], "diff": dc["diff"], "ci95": dc["ci95"]})
    print(f"  vs BEST {d['win']}/{d['shares']}: diff {dc['diff']:.3f} CI [{dc['ci95'][0]:.2f},{dc['ci95'][1]:.2f}]")

# ---------------- step 2: gated pairs ----------------
TYPES = sorted(set(v.mtype.tolist()))
TCODE = np.array([TYPES.index(t) for t in v.mtype.tolist()])
nobs = np.array([np.isfinite(v.grid[i, :v.cutoff[i]]).sum() for i in range(v.n)], float)
OBS = nobs / v.cutoff
CUT = v.cutoff.astype(float)
# standardise on dev rows (a reparametrisation of the same sigmoid family, so it does not change the fitted gate)
Z = {"obs": (OBS - OBS[rows].mean()) / OBS[rows].std(), "cutoff": (CUT - CUT[rows].mean()) / CUT[rows].std()}


def gate_scale(kind, g, idx):
    if kind == "ungated":
        return np.full(len(idx), sig(g[0]))
    if kind in Z:
        return sig(g[0] + g[1] * Z[kind][idx])
    return sig(g[TCODE[idx]])          # machine type: 3 weights


def gate_fit(A, B, kind, tr):
    q, ww = Q[tr], w[tr]
    At, Bt = A[tr], B[tr]

    def f(g):
        s = gate_scale(kind, g, tr)[:, None]
        P = s * At + (1 - s) * Bt
        return float((ww * ((P - q) ** 2).sum(1)).sum())

    npar = 3 if kind == "type" else (1 if kind == "ungated" else 2)
    best = None
    for g0 in (-1.0, 0.0, 1.0):
        r = minimize(f, np.full(npar, g0), method="BFGS", options={"gtol": 1e-9, "maxiter": 200})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def gate_pred(A, B, kind, g, idx):
    s = gate_scale(kind, g, idx)[:, None]
    return s * A[idx] + (1 - s) * B[idx]


def gate_oof(A, B, kind):
    oof = np.tile(DEFAULT, (v.n, 1))
    for tr, te in FOLDS:
        g = gate_fit(A, B, kind, tr)
        oof[te] = gate_pred(A, B, kind, g, te)
    return oof


step2 = []
KINDS = ["ungated", "obs", "cutoff", "type"]
pairs = list(itertools.combinations(GATE_MEMBERS, 2))
if SMOKE:
    pairs = pairs[:1]
for na, nb in pairs:
    A, B = M[na], M[nb]
    row = {}
    for kind in KINDS:
        oof = gate_oof(A, B, kind)
        row[kind] = {"points": evaluate(oof, v, "dev", bootstrap=0)["points"], "oof": oof}
    for kind in KINDS:
        g_all = gate_fit(A, B, kind, rows)
        step2.append({"A": na, "B": nb, "gate": kind, "points": row[kind]["points"],
                      "gain_vs_ungated": row[kind]["points"] - row["ungated"]["points"],
                      "params_all_dev": [round(float(x), 3) for x in g_all], "oof": row[kind]["oof"]})
        print(f"  gate {na}+{nb} {kind:8s} OOF {row[kind]['points']:.3f} "
              f"gain {row[kind]['points'] - row['ungated']['points']:+.3f}  params {np.round(g_all, 2).tolist()}", flush=True)
print(f"step2 gate configs: {len(step2)}  ({time.time() - t0:.0f}s)", flush=True)

best1 = step1[0]
best2 = max(step2, key=lambda d: d["points"])
gated_helped = [d for d in step2 if d["gate"] != "ungated" and d["gain_vs_ungated"] > 0.1]
print(f"best structure: step1 {best1['points']:.3f} vs best gate {best2['points']:.3f} ({best2['A']}+{best2['B']} {best2['gate']})")

if SMOKE:
    print("SMOKE OK, nothing written")
    sys.exit(0)

# ---------------- choose final structure, build full output ----------------
if best1["points"] >= best2["points"]:
    final_desc = f"two-stage win={best1['win']} shares={best1['shares']} (fit on all dev for audit rows)"
    P = best1["oof"].copy()
    P[arows] = struct_full(best1["win"], best1["shares"], arows)
    params = {"type": "two_stage", "win": best1["win"], "shares": best1["shares"]}
else:
    A, B = M[best2["A"]], M[best2["B"]]
    P = best2["oof"].copy()
    g_all = gate_fit(A, B, best2["gate"], rows)
    P[arows] = gate_pred(A, B, best2["gate"], g_all, arows)
    final_desc = f"gated pair {best2['A']}+{best2['B']} gate={best2['gate']}"
    params = {"type": "gated_pair", "A": best2["A"], "B": best2["B"], "gate": best2["gate"],
              "params_all_dev": best2["params_all_dev"] if "params_all_dev" in best2 else None}

n_combos = len(step1) + len(step2)
extra = {"step1_combos": len(step1), "step2_gate_configs": len(step2),
         "top3_vs_BEST_bootstrap500": top3_diff, "gated_gt_0.1_over_ungated": [
             {"A": d["A"], "B": d["B"], "gate": d["gate"], "gain": round(d["gain_vs_ungated"], 3)} for d in gated_helped],
         "gate_table": [{k: d[k] for k in ("A", "B", "gate", "points", "gain_vs_ungated")} for d in step2]}
to_csv(HERE / "preds" / "K2_validation.csv", v, P)
doc = write_result("K2", "claude", f"K2 small-subset stack and observable gates; best dev OOF: {final_desc}",
                   params, v, P, round(time.time() - t0, 1), extra=extra)
(HERE / "grids" / "K2.json").write_text(json.dumps(top10, indent=1))

best_pts = doc["points"]
lines = ["# K2 report (dev OOF, 6 folds, R-weighted; audit untouched)",
         f"- Combos searched: step1 {len(step1)} two-stage (41 win x 41 size subsets), step2 {len(step2)} gate configs (6 pairs x 4 gate forms x 1 ungated baseline)",
         "- Top 10 two-stage (dev OOF points):"]
for d in top10[:10]:
    lines.append(f"  {d['points']:.3f} win={'+'.join(d['win'])} | shares={'+'.join(d['shares'])}")
for d in top3_diff:
    lines.append(f"- vs BEST (TM_COMBINED3, {BEST_NAME} dev points): {d['win']}/{d['shares']} diff {d['diff']:+.3f} CI [{d['ci95'][0]:.2f},{d['ci95'][1]:.2f}]")
lines.append(f"- Gates: {'helped (>0.1 over ungated) in ' + str(len(gated_helped)) + ' configs' if gated_helped else 'no gate beat its ungated pair by >0.1 points'}; best gate OOF {best2['points']:.3f} ({best2['A']}+{best2['B']} {best2['gate']}, gain {best2['gain_vs_ungated']:+.3f})")
lines.append(f"- Saved structure: {final_desc}, dev OOF points {best_pts:.3f}")
lines.append(f"- Runtime {time.time() - t0:.0f}s")
(HERE / "reports" / "K2.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
