"""M9b: gate between POOLED (episode-shared) and OWN (per-machine) BOCPD evidence (follow-up to M9).

pw = g*pw_pool + (1-g)*pw_own, where each pw is the B3 ridge win prob with BOCPD survival row weights
(K=2, alpha=2, floor .05, mode as variant). OWN = same BOCPD on the machine's own stream (rho=0); POOLED = M9 (rho
as variant, peers' ticks < cutoff). Gate g is per episode, from ticks < cutoff only, no condition labels:
  C    = |sum_j z_j|/sqrt(m): common-shift stat; z_j = standardized (win rate of last W ticks - earlier) of machine j,
         over the m machines of the episode (incl. self). Independent drift gives ~N(0,1); shared shift gives large C.
  frac = fraction of PEERS whose own BOCPD posterior puts > 0.5 mass on a regime start in the last W ticks.
  cnt  = number of peers with that property.
Pre-declared grids (nothing selected on shared_change; selection = total dev points):
  base: pool {(rho1,surv),(rho.5,surv*b3)} x h {1/200,1/5000}   (own uses same h, mode, K=2); W=40 fixed for C/frac
  gate types per base:
   soft1: g=sigmoid(s*(C-th)), th {1,1.5,2,3} x s {1,2,4}      (12)
   hard1: g=1[C>th], th {1,1.5,2,3}                            (4)
   soft2: g=sigmoid(s*(C-th))*sigmoid(8*(frac-f0)), s=2, th {1,1.5,2} x f0 {0.15,0.3,0.5}  (9)
   kofn : g=1[cnt>=k] with W in {40,80} for the mass test, k {1,2,3,4}                (8)
  = 33 cells per base, 132 total (+2 reference cells per base). Own-only and pool-only reference cells included.
 ROUND 2 (declared after round 1 showed own-BOCPD-surv is a terrible fallback: drift 1.15): fallback fb='b3' =
  plain B3 holdout half-life memory (the standard baseline) instead of own BOCPD; pool in {surv,b3} x h {1/200,1/5000}
  same 33 gate cells + refs = 140 more cells (fb='own' cells are round 1). Selection over all cells by dev points.
Run: python3 methods/M9b.py [--smoke]
"""
import sys, os, json, time, itertools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import _mem as m
import _cp as cp
from harness.scoring import _episode_table, _resamples, BOOT_SEED
from harness.data import CONDITIONS

_POST = {}
_PW = {}
_FEAT = {}
POOLS = {"surv": (1.0, "surv"), "b3": (0.5, "surv*b3")}


def own_post(split, q, h):
    k = (split.name, q.i, h, "own")
    if k not in _POST:
        _POST[k] = cp.bocpd_post([q.g], q.c, h, 2.0, 2, 1.0)
    return _POST[k]


def pool_post(split, q, h, rho):
    k = (split.name, q.i, h, rho)
    if k not in _POST:
        peers = split.peers(q.i)
        _POST[k] = cp.bocpd_post([q.g] + [split.grid[j] for j in peers], q.c, h, 2.0, 2, rho)
    return _POST[k]


def pw_var(split, q, h, pool, which):
    """win prob of query q under variant; which in own|pool."""
    if not q.gappy:
        return m.pw_hl_cached(q, m.INF)
    if which == "b3":
        return m.pw_hl_cached(q, m.b3_hl(q))
    rho, mode = POOLS[pool]
    k = (split.name, q.i, h, pool, which)
    if k not in _PW:
        post = own_post(split, q, h) if which == "own" else pool_post(split, q, h, rho)
        w = cp.surv_weights(q, post, 0.05)
        if mode == "surv*b3":
            w = w * m.hl_w(q.ticks, q.c, m.b3_hl(q))
        _PW[k] = m.pw_from_w(q, w)
    return _PW[k]


def zstat(g, c, W):
    ok = ~np.isnan(g[:c])
    t = np.flatnonzero(ok)
    x = (g[t] > 0).astype(float)
    rec = t >= c - W
    n1, n0 = rec.sum(), (~rec).sum()
    if n1 < 5 or n0 < 10:
        return 0.0
    p = x.mean()
    sd = np.sqrt(max(p * (1 - p), 1e-3) * (1 / n1 + 1 / n0))
    return float((x[rec].mean() - x[~rec].mean()) / sd)


def feats(split, qs, i, h, W):
    """(C, frac, cnt) for the episode of query i (excluding peers' outcomes at cutoff; only grid < cutoff)."""
    k = (split.name, i, h, W)
    if k not in _FEAT:
        peers = split.peers(i)
        c = qs[i].c
        zs = [zstat(split.grid[j], c, W) for j in [i] + peers]
        C = abs(sum(zs)) / np.sqrt(len(zs))
        rec = []
        for j in peers:
            post = own_post(split, qs[j], h)
            rec.append(post[max(c - W, 0):].sum() > 0.5)
        cnt = int(sum(rec))
        _FEAT[k] = (float(C), cnt / max(len(peers), 1), cnt)
    return _FEAT[k]


def sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def gate(p, f1, f2):
    t = p["gate"]
    if t == "none_own":
        return 0.0
    if t == "none_pool":
        return 1.0
    if t == "soft1":
        return float(sig(p["s"] * (f1[0] - p["th"])))
    if t == "hard1":
        return float(f1[0] > p["th"])
    if t == "soft2":
        return float(sig(p["s"] * (f1[0] - p["th"])) * sig(8 * (f1[1] - p["f0"])))
    if t == "kofn":
        return float(f2[2] >= p["k"])


def make(p):
    def predict(split):
        qs = m.prep(split)
        pws = []
        fb = p.get("fb", "own")
        for q in qs:
            g = gate(p, feats(split, qs, q.i, p["h"], 40) if p["gate"] != "kofn" else None,
                     feats(split, qs, q.i, p["h"], p.get("W", 40)) if p["gate"] == "kofn" else None) \
                if p["gate"] not in ("none_own", "none_pool") else gate(p, None, None)
            if g <= 0:
                pw = pw_var(split, q, p["h"], p["pool"], fb)
            elif g >= 1:
                pw = pw_var(split, q, p["h"], p["pool"], "pool")
            else:
                pw = g * pw_var(split, q, p["h"], p["pool"], "pool") + (1 - g) * pw_var(split, q, p["h"], p["pool"], fb)
            pws.append(pw)
        return m.assemble(split, pws)
    return predict


def cells():
    out = []
    for fb, pool, h in itertools.product(("own", "b3"), ("surv", "b3"), (1 / 200, 1 / 5000)):
        b = {"pool": pool, "h": h, "fb": fb}
        out += [dict(b, gate="none_own"), dict(b, gate="none_pool")]
        out += [dict(b, gate="soft1", th=th, s=s) for th in (1, 1.5, 2, 3) for s in (1, 2, 4)]
        out += [dict(b, gate="hard1", th=th) for th in (1, 1.5, 2, 3)]
        out += [dict(b, gate="soft2", s=2, th=th, f0=f0) for th in (1, 1.5, 2) for f0 in (0.15, 0.3, 0.5)]
        out += [dict(b, gate="kofn", W=W, k=k) for W in (40, 80) for k in (1, 2, 3, 4)]
    return out


def cond_diff_ci(P, Pb, split, reps=1000):
    """paired bootstrap of (ratio_method - ratio_B3) per condition, dev episodes only, harness resampling."""
    ca, ea, ba = _episode_table(m_prep(P), split, "dev")
    _, eb, _ = _episode_table(m_prep(Pb), split, "dev")
    res = {}
    rs = _resamples(ca, reps)
    for ci, c in enumerate(CONDITIONS):
        d0 = ea[ca == ci].mean() / ba[ca == ci].mean() - eb[ca == ci].mean() / ba[ca == ci].mean()
        bs = []
        for idx in rs:
            ii = idx[ca[idx] == ci]
            bs.append((ea[ii].mean() - eb[ii].mean()) / ba[ii].mean())
        res[c] = {"diff": float(d0), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    return res


def m_prep(P):
    P = np.asarray(P, float)
    return P / P.sum(1, keepdims=True)


if __name__ == "__main__":
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        assert os.environ.get(k) == "1", k
    v = m.load_split("validation")
    sm = m.smoke_split(v)
    t0 = time.perf_counter()
    cs = cells()
    for c in (cs[0], cs[5], cs[-1]):
        P = make(c)(sm)
        print("smoke", c, round(m.cell_eval(P, sm, "smoke")[0], 2), flush=True)
    print(f"smoke {time.perf_counter()-t0:.1f}s", flush=True)
    if "--smoke" in sys.argv:
        sys.exit()
    b3 = m.read_pred_csv(str(m.SAI / "preds/B3_validation.csv"), v)
    out, best = [], None
    t0 = time.perf_counter()
    for c in cs:
        P = make(c)(v)
        pts, cr = m.cell_eval(P, v, "dev")
        out.append({"params": c, "points": pts, "condition_ratios": cr})
        print(f"{pts:.3f} {json.dumps(c)} {json.dumps({k: round(x, 3) for k, x in cr.items()})}", flush=True)
        if best is None or pts > best[0]:
            best = (pts, c)
        (m.SAI / "grids/M9b.json").write_text(json.dumps(out, indent=1, default=float))
    print("grid time", time.perf_counter() - t0)
    c = best[1]
    d, P = m.run_method("M9b", "claude-w2", "gated pooled/own episode BOCPD", c, make(c), v)
    dd = m.diff_ci(P, b3, v, "dev")
    bg = m.blend_gain(P, b3, v)
    cr0 = json.load(open(m.SAI / "results/B3.json"))["condition_ratios"]
    cd = cond_diff_ci(P, b3, v)
    d["extra"] = {"vs_B3": dd, "blend_gain": {"points": bg["points"], "ci95": bg["ci95"], "vs_base": bg["vs_base"], "w_mean": bg["w_mean"]},
                  "cond_ratio_diff_vs_B3_ci": cd, "grid_cells_completed": len(out)}
    # per-condition CI for the best-shared_change-ratio cell and pure pooled references (reporting only)
    refs = {}
    for lab, pr in (("pool_only_surv_h5000", {"pool": "surv", "h": 1 / 5000, "gate": "none_pool"}),
                    ("pool_only_surv_h200", {"pool": "surv", "h": 1 / 200, "gate": "none_pool"}),
                    ("own_only_surv_h5000", {"pool": "surv", "h": 1 / 5000, "gate": "none_own"})):
        refs[lab] = cond_diff_ci(make(pr)(v), b3, v)
    d["extra"]["reference_cond_ci"] = refs
    (m.SAI / "results/M9b.json").write_text(json.dumps(d, indent=1, default=float))
    print("BEST", d["points"], d["ci95"], dd, bg["points"], bg["vs_base"])
    print(json.dumps(cd, indent=0)); print(json.dumps(refs, indent=0))
