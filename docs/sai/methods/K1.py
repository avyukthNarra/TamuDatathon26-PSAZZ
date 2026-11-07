"""K1: extensive combination search around BEST = TM_COMBINED3 (dev, 72 episodes, episode-OOF 6 folds, R-weighted rows).
Stages (counts are recorded in grids/K1.json):
 1. compose: P=[1-pw_i, pw_i*shares_j] for all 18x18 member pairs (parameter-free, no fit).
 2. convex blends of 2-4 of 8 selected members (5-class vectors), SLSQP on the simplex, OOF.
 (3,4 moved to worker K2) 3. two-stage: pw blend of <=3 and shares blend of <=3 of the 8 members; in-sample screen on all dev, OOF for the top 40.
 4. gated pairs: w(z)=sigmoid(g0+g1*z), z in {obs fraction, cutoff, machine-type indicators}; 2 gate params per pair, OOF.
 5. nested check: select on a random half of dev episodes (stratified by condition, 10 splits), score the other half.
Audit rows: same structure fit on ALL dev rows. Audit truth/scoring never touched.
  python3 methods/K1.py
"""
import hashlib, itertools, json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from harness import load_split, read_pred_csv, evaluate, diff_ci, to_csv, write_result
from harness.data import CONDITIONS
from harness.scoring import scope_episodes

ALL = ["TM_COMBINED3", "TM_MIXER3", "TM_RULE", "TM_COMBINED", "TM_MIXER", "TM_E6", "TM_E7", "TM_E67", "C2", "C1", "A9",
       "M9b_B4", "M2_B4", "M3_B4", "L5", "L7", "B4", "A1_B4"]
v = load_split("validation")
M = {n: read_pred_csv(HERE / "preds" / f"{n}_validation.csv", v) for n in ALL}
BEST = M["TM_COMBINED3"]
dev = scope_episodes(v, "dev")
aud = scope_episodes(v, "audit")
rows = np.concatenate([v.episode_rows(e) for e in dev])
arows = np.concatenate([v.episode_rows(e) for e in aud])
ep_of = {e: i for i, e in enumerate(dev)}
rowep = np.full(v.n, -1)
for e in dev:
    rowep[v.episode_rows(e)] = ep_of[e]
epcond = np.array([CONDITIONS.index(v.condition[v.episode_rows(e)[0]]) for e in dev])
base_err = ((v.ref - v.truth) ** 2).sum(1)
# R-weights (as stack_w5)
w = np.zeros(v.n)
refmse = {c: base_err[[i for i in rows if v.condition[i] == c]].mean() for c in CONDITIONS}
for c in CONDITIONS:
    eps = [e for e in dev if v.condition[v.episode_rows(e)[0]] == c]
    for e in eps:
        r = v.episode_rows(e)
        w[r] = 1.0 / (len(eps) * len(r) * refmse[c])
order = sorted(dev, key=lambda e: hashlib.sha256(f"w5fold|{e}".encode()).hexdigest())
fold = {e: i % 6 for i, e in enumerate(order)}
FOLDROWS = [(np.concatenate([v.episode_rows(e) for e in dev if fold[e] != f]),
             np.concatenate([v.episode_rows(e) for e in dev if fold[e] == f])) for f in range(6)]
dev_idx = np.array(sorted(set(rows.tolist())))
Q = v.truth
BASE_EP = np.bincount(rowep[rows], weights=base_err[rows], minlength=72) / np.bincount(rowep[rows], minlength=72)


def pts(P, epmask=None):
    """fast dev points (same formula as harness), optionally on a subset of dev episodes (bool mask)."""
    err = ((P[rows] - Q[rows]) ** 2).sum(1)
    e = np.bincount(rowep[rows], weights=err, minlength=72) / np.bincount(rowep[rows], minlength=72)
    m = np.ones(72, bool) if epmask is None else epmask
    rat = [e[m & (epcond == c)].mean() / BASE_EP[m & (epcond == c)].mean() for c in range(4)]
    return v.max_points / (1 + np.mean(rat))


def parts(P):
    pw = 1 - P[:, 0]
    return pw, P[:, 1:] / P[:, 1:].sum(1, keepdims=True)


# ---------- fitters ----------
def simplex_fit(Ms, tr):
    """Ms list of (n,5); convex weights minimising weighted sq error on rows tr."""
    K = len(Ms)
    A = np.stack([m[tr] for m in Ms])  # K,n,5
    q, ww = Q[tr], w[tr]
    G = np.einsum("n,knc,lnc->kl", ww, A, A)
    b = np.einsum("n,knc,nc->k", ww, A, q)
    f = lambda x: x @ G @ x - 2 * b @ x
    g = lambda x: 2 * (G @ x - b)
    r = minimize(f, np.ones(K) / K, jac=g, method="SLSQP", bounds=[(0, 1)] * K,
                 constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1, "jac": lambda x: np.ones(K)}],
                 options={"maxiter": 200, "ftol": 1e-13})
    return r.x


def blend(Ms, a):
    return np.tensordot(a, np.stack(Ms), axes=1)


def two_stage_fit(PWm, SHm, tr, x0=None):
    npw, nsh = len(PWm), len(SHm)
    pwA = np.stack([m[tr] for m in PWm]); shA = np.stack([m[tr] for m in SHm]); q, ww = Q[tr], w[tr]

    def fg(x):
        a, b = x[:npw], x[npw:]
        pw = a @ pwA; sh = np.tensordot(b, shA, axes=1)
        P = np.column_stack([1 - pw, pw[:, None] * sh]); r = P - q
        L = (ww * (r ** 2).sum(1)).sum()
        dpw = 2 * ww * (-r[:, 0] + (r[:, 1:] * sh).sum(1))
        dsh = 2 * ww[:, None] * r[:, 1:] * pw[:, None]
        return L, np.r_[pwA @ dpw, np.einsum("jnk,nk->j", shA, dsh)]
    x0 = np.r_[np.ones(npw) / npw, np.ones(nsh) / nsh] if x0 is None else x0
    cons = [{"type": "eq", "fun": lambda x: x[:npw].sum() - 1}, {"type": "eq", "fun": lambda x: x[npw:].sum() - 1}]
    r = minimize(fg, x0, jac=True, method="SLSQP", bounds=[(0, 1)] * (npw + nsh), constraints=cons,
                 options={"maxiter": 200, "ftol": 1e-13})
    return r.x[:npw], r.x[npw:]


def two_stage_pred(PWm, SHm, a, b, idx):
    pw = a @ np.stack([m[idx] for m in PWm]); sh = np.tensordot(b, np.stack([m[idx] for m in SHm]), axes=1)
    return np.column_stack([1 - pw, pw[:, None] * sh])


def oof_generic(fit_fn, pred_fn):
    """fit_fn(tr)->params ; pred_fn(params, idx)->(len(idx),5). Returns full (n,5): dev rows OOF, audit rows full-fit, rest BEST."""
    out = BEST.copy()
    for tr, te in FOLDROWS:
        out[te] = pred_fn(fit_fn(tr), te)
    return out


def full_audit(out, fit_fn, pred_fn):
    out = out.copy(); out[arows] = pred_fn(fit_fn(rows), arows); return out


# ---------- features for gating ----------
nobs = np.array([np.isfinite(v.grid[i, :v.cutoff[i]]).sum() for i in range(v.n)], float)
FEATS = {"obsfrac": nobs / v.cutoff, "cutoff": v.cutoff.astype(float)}
for t in sorted(set(v.mtype)):
    FEATS[f"is_{t}"] = (v.mtype == t).astype(float)
for k in ("obsfrac", "cutoff"):
    x = FEATS[k]; FEATS[k] = (x - x[rows].mean()) / x[rows].std()


def gate_fit(Ma, Mb, z, tr):
    q, ww = Q[tr], w[tr]; A, B = Ma[tr], Mb[tr]; zz = z[tr]

    def f(g):
        s = 1 / (1 + np.exp(-(g[0] + g[1] * zz)))
        P = s[:, None] * A + (1 - s[:, None]) * B
        return (ww * ((P - q) ** 2).sum(1)).sum()
    best = None
    for g0 in (-1.0, 0.0, 1.0):
        r = minimize(f, [g0, 0.0], method="BFGS", options={"gtol": 1e-10})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def gate_pred(Ma, Mb, z, g, idx):
    s = 1 / (1 + np.exp(-(g[0] + g[1] * z[idx])))
    return s[:, None] * Ma[idx] + (1 - s[:, None]) * Mb[idx]


def main():
    t0 = time.time()
    G = {"BEST_dev": pts(BEST)}
    print("BEST dev", pts(BEST), evaluate(BEST, v, "dev", bootstrap=0)["points"])
    # member table
    corr = {}
    eb = (BEST - Q)[rows]
    for n in ALL:
        e = (M[n] - Q)[rows]
        corr[n] = float(np.corrcoef(e.ravel(), eb.ravel())[0, 1])
    indiv = {n: pts(M[n]) for n in ALL}
    G["member_dev_points"] = indiv; G["member_err_corr_vs_BEST"] = corr
    for n in ALL:
        print(f"  {n:13s} dev {indiv[n]:.2f} corr vs BEST {corr[n]:.3f}")
    # ---- 1. compose
    comp = []
    for i, j in itertools.product(ALL, ALL):
        P = np.column_stack([1 - parts(M[i])[0], parts(M[i])[0][:, None] * parts(M[j])[1]])
        comp.append({"pw": i, "shares": j, "points": pts(P)})
    comp.sort(key=lambda d: -d["points"])
    top1 = comp[:10]
    for d in top1:
        i, j = d["pw"], d["shares"]
        P = np.column_stack([1 - parts(M[i])[0], parts(M[i])[0][:, None] * parts(M[j])[1]])
        ev = evaluate(P, v, "dev", bootstrap=300); dc = diff_ci(P, BEST, v, "dev", bootstrap=300)
        d.update(diff_vs_BEST=dc["diff"], ci=dc["ci95"], ratios=ev["condition_ratios"])
        print("S1", i, j, round(d["points"], 3), round(dc["diff"], 3), [round(x, 2) for x in dc["ci95"]])
    G["stage1_n"] = len(comp); G["stage1_top10"] = top1
    # ---- member selection: 8 by score and low corr
    cand = sorted([n for n in ALL if n != "TM_COMBINED3"], key=lambda n: -(indiv[n] - 30 * max(corr[n] - 0.9, 0)))
    SEL = ["TM_COMBINED3"] + cand[:7]
    # force-diverse: ensure TM_E7 and L7 are present
    for must in ("TM_E7", "L7"):
        if must not in SEL:
            SEL[-1] = must
    SEL = list(dict.fromkeys(SEL))
    while len(SEL) < 8:
        SEL.append(next(n for n in cand if n not in SEL))
    G["selected8"] = SEL; print("SEL", SEL)
    # ---- 2. convex blends
    res2 = []
    for k in (2, 3, 4):
        for sub in itertools.combinations(SEL, k):
            Ms = [M[n] for n in sub]
            fit_fn = lambda tr, Ms=Ms: simplex_fit(Ms, tr)
            pred_fn = lambda a, idx, Ms=Ms: blend([m[idx] for m in Ms], a)
            out = oof_generic(fit_fn, pred_fn)
            res2.append({"members": sub, "points": pts(out), "w_full": simplex_fit(Ms, rows).round(3).tolist()})
    res2.sort(key=lambda d: -d["points"]); G["stage2_n"] = len(res2); G["stage2_top10"] = res2[:10]
    for d in res2[:10]:
        print("S2", d["members"], round(d["points"], 3), d["w_full"])
    print("t", time.time() - t0)
    json.dump(G, open(HERE / "grids" / "K1.json", "w"), indent=1, default=str)
    json.dump({"G": G, "comp_all": comp}, open("/tmp/x/k1_state.json", "w"), default=str)
    return G, SEL, res2, comp


def pts_in(P):
    return pts(P)




def nested(SEL, comp):
    """10 random splits (stratified by condition, grouped by episode). On half A pick the best of: stage-1 compositions
    (no fit) and stage-2 convex blends (OOF inside A, 3 folds); refit chosen on A, score on half B; gain over BEST on B."""
    rng = np.random.RandomState(7)
    combos = [("S1", i, j) for i, j in itertools.product(ALL, ALL)] + \
             [("S2", sub) for k in (2, 3, 4) for sub in itertools.combinations(SEL, k)]
    out = []
    S1 = {}
    for i, j in itertools.product(ALL, ALL):
        S1[(i, j)] = np.column_stack([1 - parts(M[i])[0], parts(M[i])[0][:, None] * parts(M[j])[1]])
    for s in range(10):
        A = np.zeros(72, bool)
        for c in range(4):
            idx = np.flatnonzero(epcond == c); A[rng.choice(idx, len(idx) // 2, replace=False)] = True
        Bm = ~A
        epsA = [e for e in dev if A[ep_of[e]]]
        # inside-A folds
        fa = {e: k % 3 for k, e in enumerate(rng.permutation(epsA))}
        Arows = np.concatenate([v.episode_rows(e) for e in epsA])
        sub_f = [(np.concatenate([v.episode_rows(e) for e in epsA if fa[e] != f]),
                  np.concatenate([v.episode_rows(e) for e in epsA if fa[e] == f])) for f in range(3)]
        best = (-1, None)
        for cb in combos:
            if cb[0] == "S1":
                P = S1[(cb[1], cb[2])]
            else:
                Ms = [M[n] for n in cb[1]]; P = BEST.copy()
                for tr, te in sub_f:
                    P[te] = blend([m[te] for m in Ms], simplex_fit(Ms, tr))
            sc = pts(P, A)
            if sc > best[0]:
                best = (sc, cb)
        cb = best[1]
        if cb[0] == "S1":
            P = S1[(cb[1], cb[2])]
        else:
            Ms = [M[n] for n in cb[1]]; a = simplex_fit(Ms, Arows); P = blend(Ms, a)
        gain = pts(P, Bm) - pts(BEST, Bm)
        out.append({"split": s, "chosen": [str(x) for x in cb], "insample_A_gain": best[0] - pts(BEST, A), "heldout_B_gain": gain})
        print("nest", s, cb, round(best[0] - pts(BEST, A), 3), round(gain, 3), flush=True)
    return out, len(combos)


def save_candidates(res2):
    """K1a/b/c = three distinct member-set blends from stage 2 (weights >0.01 on every member in the full-dev fit)."""
    picks, seen = [], set()
    for d in res2:
        act = tuple(n for n, x in zip(d["members"], d["w_full"]) if x > 0.01)
        if len(act) < 2 or act in seen or any(set(act) <= set(p) or set(p) <= set(act) for p in seen):
            continue
        seen.add(act); picks.append(d)
        if len(picks) == 3: break
    info = []
    for tag, d in zip("abc", picks):
        Ms = [M[n] for n in d["members"]]
        fit_fn = lambda tr: simplex_fit(Ms, tr)
        pred_fn = lambda a, idx: blend([m[idx] for m in Ms], a)
        out = full_audit(oof_generic(fit_fn, pred_fn), fit_fn, pred_fn)
        to_csv(HERE / "preds" / f"K1{tag}_validation.csv", v, out)
        ev = evaluate(out, v, "dev", bootstrap=1000); dc = diff_ci(out, BEST, v, "dev")
        info.append({"id": f"K1{tag}", "members": d["members"], "w_full": d["w_full"], "points": ev["points"], "ci95": ev["ci95"],
                     "ratios": ev["condition_ratios"], "vs_BEST": dc})
        print(info[-1])
        if tag == "a":
            write_result("K1", "sai-worker", "convex blend of BEST with C2 (episode-OOF, dev)", {"members": d["members"], "w": d["w_full"]},
                         v, out, 0.0)
    return info


if __name__ == "__main__" and len(sys.argv) > 1:
    G, SEL, res2, comp = main()
    info = save_candidates(res2)
    nest, ncomb = nested(SEL, comp)
    G["candidates"] = info; G["nested"] = nest; G["nested_n_combos"] = ncomb
    G["nested_mean_heldout_gain"] = float(np.mean([x["heldout_B_gain"] for x in nest]))
    G["nested_mean_insample_gain"] = float(np.mean([x["insample_A_gain"] for x in nest]))
    json.dump(G, open(HERE / "grids" / "K1.json", "w"), indent=1, default=str)
    print("NESTED mean heldout gain", G["nested_mean_heldout_gain"], "insample", G["nested_mean_insample_gain"])
