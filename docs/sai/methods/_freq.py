"""Shared core for W1 frequency/shrinkage methods (A1-A8). Reuses _mem.py (B3 pieces) read-only.

Model family: P(payout) = pw * size_shares (B3 form) where
  mode "ridge": pw = B3 pair-ridge win prob (m.b3_hl + m.pw_hl), optional A5 row weights, optional win-prior shrink
  mode "freq":  pw = recency-weighted win frequency (half-life hlw) shrunk to a type/cell win-rate prior (aw pseudo-ticks)
  size_shares: class counts = alpha_k * prior_share_k + sum of recency/density-weighted payouts of class k
  (B3 = prior "self" [type shares from the split's own pre-cutoff histories], alpha 8, hl 64).
Priors "type"/"cell" are estimated from the TRAIN split histories (ticks < cutoff), then applied to any split.
Short history is detected from tick gaps only (observed fraction < 0.6 of cutoff, m.GAPPY_FRAC), never from labels.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import _mem as m  # noqa: E402
from _mem import (load_split, smoke_split, evaluate, diff_ci, blend_gain, run_method, read_pred_csv, SAI,  # noqa: E402
                  PAYV, INF)
from harness import class_shares  # noqa: E402
import harness  # noqa: E402

_TRAIN = {}
_QX = {}
_PW = {}


# ---------------- train-estimated priors ----------------
def train_priors():
    if "p" in _TRAIN:
        return _TRAIN["p"]
    t = load_split("train")
    cnt, tk = {}, {}
    for i in range(t.n):
        g = t.grid[i, :t.cutoff[i]]
        ob = g[~np.isnan(g)]
        c5 = np.array([np.sum(ob == 0)] + [np.sum(ob == p) for p in PAYV], float)
        for key in (("t", t.mtype[i]), ("c", t.mtype[i], int(t.signal[i]))):
            cnt[key] = cnt.get(key, 0) + c5
    pri = {k: c / c.sum() for k, c in cnt.items()}      # per-tick class rates (index 0 = loss)
    _TRAIN["p"] = pri
    return pri


def prior_rates(q, split, kind):
    """per-tick 5-class rate vector prior for query q."""
    if kind == "self":
        r = np.concatenate([[0.0], q.tshare]); r[0] = 0.0
        return r
    pri = train_priors()
    mt = split.mtype[q.i]
    return pri[("c", mt, int(split.signal[q.i]))] if kind == "cell" else pri[("t", mt)]


def prior_shares(q, split, kind):
    if kind == "self":
        return q.tshare
    r = prior_rates(q, split, kind)[1:]
    return r / r.sum()


# ---------------- per-query extras ----------------
class QX:
    pass


def prep_x(split):
    key = (split.name, split.n, str(split.qid[0]))
    if key in _QX:
        return _QX[key]
    qs = m.prep(split)
    out = []
    for q in qs:
        x = QX()
        g, c = q.g, q.c
        x.obs_t = np.flatnonzero(~np.isnan(g))
        x.win_o = (g[x.obs_t] > 0).astype(float)
        x.age_o = (c - 1 - x.obs_t).astype(float)
        s = c
        while s > 0 and not np.isnan(g[s - 1]):
            s -= 1
        x.s = s                                   # dense tail starts here (0 = fully dense)
        n_old = int(np.sum(x.obs_t < s))
        x.sparse = (not q.gappy) and n_old > 0
        x.fac = (s / n_old) if x.sparse else 1.0   # one sampled older tick stands for this many ticks
        x.age_p = (c - 1 - q.pay_t).astype(float)
        out.append(x)
    _QX[key] = out
    return out


def mult(x, t_arr, e):
    if e == 0 or not x.sparse:
        return np.ones(len(t_arr))
    return np.where(t_arr < x.s, x.fac ** e, 1.0)


def dec(age, hl):
    return np.ones_like(age) if hl >= INF else 0.5 ** (age / hl)


def eff_w(q, x, hl, e):
    return float(np.sum(mult(x, x.obs_t, e) * dec(x.age_o, hl)))


def pay_sums(q, x, hl4, e):
    """weighted payout sums per class (4), class-specific half-lives hl4."""
    S = np.zeros(4)
    if len(q.pay_t) == 0:
        return S
    mp = mult(x, q.pay_t, e)
    for k in range(4):
        sel = q.pay_k == k
        if sel.any():
            S[k] = np.sum(mp[sel] * dec(x.age_p[sel], hl4[k]))
    return S


def as4(v):
    return [float(a) for a in v] if isinstance(v, (list, tuple)) else [float(v)] * 4


# ---------------- size shares ----------------
def size_shares(q, x, prior4, p, wmed):
    hl4, a4 = as4(p.get("hl", 64)), as4(p.get("alpha", 8))
    e = p.get("e5", 0)
    g = p.get("g", 0.0)
    if g:
        sc = (eff_w(q, x, 64, e) / wmed) ** g
        a4 = [a * sc for a in a4]
    S = pay_sums(q, x, hl4, e)
    if p.get("norm", False):
        wref = eff_w(q, x, 64, e)
        S = S * np.array([wref / max(eff_w(q, x, h, e), 1e-9) for h in hl4])
    cnt = np.array(a4) * prior4 + S
    return cnt / cnt.sum()


def peer_prior(split, qs, xs, i, p, base4):
    """A8: blend type prior with pooled recency-weighted peer payout shares (leave-one-out: self excluded)."""
    a8 = p["a8"]
    lam = a8["lam"]
    if lam == 0:
        return base4
    peers = split.peers(i)
    if a8["scope"] == "type":
        peers = [j for j in peers if split.mtype[j] == split.mtype[i]]
    if not peers:
        return base4
    S = np.zeros(4)
    for j in peers:
        S += pay_sums(qs[j], xs[j], [a8.get("hl", 64)] * 4, 0)
    sh = (S + a8.get("k", 8) * base4) / (S.sum() + a8.get("k", 8))
    return (1 - lam) * base4 + lam * sh


# ---------------- win probability ----------------
def pw_ridge(split, qs, xs, e5r):
    key = (split.name, split.n, str(split.qid[0]), e5r)
    if key in _PW:
        return _PW[key]

    def build():
        out = []
        for q, x in zip(qs, xs):
            hl = m.b3_hl(q)
            w = m.hl_w(q.ticks, q.c, hl)
            if e5r:
                w = w * mult(x, q.ticks, e5r)
            out.append(m.pw_from_w(q, w))
        return np.array(out)

    v = harness.cached(f"freq_pwridge_{split.name}_{split.n}_{split.qid[0]}_{e5r}_v1", build)
    _PW[key] = v
    return v


def win_prior_rate(q, split, kind):
    r = prior_rates(q, split, "type" if kind == "self" else kind)
    return 1.0 - r[0]


def pw_freq(q, x, split, kind, hlw, aw, e):
    wt = mult(x, x.obs_t, e) * dec(x.age_o, hlw)
    rw = win_prior_rate(q, split, kind)
    return float((np.sum(wt * x.win_o) + aw * rw) / (np.sum(wt) + aw))


# ---------------- prediction ----------------
def predict_params(split, p):
    qs, xs = m.prep(split), prep_x(split)
    n = split.n
    kind = p.get("prior", "type")
    e = p.get("e5", 0)
    wmed = float(np.median([eff_w(q, x, 64, e) for q, x in zip(qs, xs)]))
    mode = p.get("mode", "ridge")
    if mode == "ridge":
        pws = pw_ridge(split, qs, xs, p.get("e5r", 0))
    P = np.zeros((n, 5))
    for k, (q, x) in enumerate(zip(qs, xs)):
        base4 = prior_shares(q, split, kind)
        pr4 = peer_prior(split, qs, xs, q.i, p, base4) if "a8" in p else base4
        sh = size_shares(q, x, pr4, p, wmed)
        if mode == "ridge":
            pw = float(pws[k])
            wk = p.get("wk", 0)
            if wk:
                pw = (q.nobs * pw + wk * win_prior_rate(q, split, kind)) / (q.nobs + wk)
        else:
            pw = pw_freq(q, x, split, kind, p.get("hlw", 64), p.get("aw", 20), e)
        pw = float(np.clip(pw, 0.005, 0.995))
        fact = np.concatenate([[1 - pw], pw * sh])
        if "a7" in p and p["a7"]["w"] > 0:          # blend with direct 5-class frequencies
            a7 = p["a7"]
            S = pay_sums(q, x, [a7["dhl"]] * 4, e)
            W = eff_w(q, x, a7["dhl"], e)
            r = prior_rates(q, split, "type" if kind == "self" else kind)[1:]
            rate = (S + a7["da"] * r) / (W + a7["da"])
            direct = np.concatenate([[1 - rate.sum()], rate])
            fact = (1 - a7["w"]) * fact + a7["w"] * direct
        P[k] = fact
    P = np.clip(P, 1e-6, None)
    return P / P.sum(1, keepdims=True)


def make(p):
    return lambda split: predict_params(split, p)


# ---------------- runner ----------------
def run_ids(mid, cells, desc):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        assert os.environ.get(k) == "1", k
    v = load_split("validation")
    sm = smoke_split(v)
    t0 = time.perf_counter()
    pts = [evaluate(make(c)(sm), sm, "smoke", bootstrap=0)["points"] for c in cells[:2]]
    dt = time.perf_counter() - t0
    print(f"[{mid}] smoke cells0-1 {pts} in {dt:.1f}s; est grid {dt/2*len(cells)*(v.n/sm.n)/60:.1f} min", flush=True)
    if "--smoke" in sys.argv:
        return
    b3 = read_pred_csv(str(SAI / "preds/B3_validation.csv"), v)
    b3r = json.load(open(SAI / "results/B3.json"))
    out, best, bm = [], None, {}
    gpath = SAI / "grids" / f"{mid}.json"
    for p in cells:
        t0 = time.perf_counter()
        P = make(p)(v)
        dt = time.perf_counter() - t0
        ev = evaluate(P, v, "dev", bootstrap=0)
        out.append({"params": p, "points": ev["points"], "condition_ratios": ev["condition_ratios"], "seconds": round(dt, 1)})
        print(f"[{mid}] {ev['points']:.3f} {json.dumps(p)} {dt:.1f}s", flush=True)
        if best is None or ev["points"] > best[0]:
            best = (ev["points"], p)
        md = p.get("mode", "ridge")
        if md not in bm or ev["points"] > bm[md][0]:
            bm[md] = (ev["points"], p)
        gpath.write_text(json.dumps(out, indent=1, default=float))
    p = best[1]
    d, P = run_method(mid, "claude-w1", desc, p, make(p), v)
    extra = {"vs_B3": diff_ci(P, b3, v, "dev"), "grid_cells_completed": len(out),
             "class_shares_B3": class_shares(b3, v, "dev"), "by_mode": {}}
    for md, (pt, pp) in bm.items():
        Pm = make(pp)(v)
        bg = blend_gain(Pm, b3, v)
        extra["by_mode"][md] = {"params": pp, "points": pt, "ci95": evaluate(Pm, v, "dev")["ci95"],
                                "vs_B3": diff_ci(Pm, b3, v, "dev"),
                                "blend_gain": {"points": bg["points"], "ci95": bg["ci95"], "vs_base": bg["vs_base"],
                                               "w_mean": bg["w_mean"]},
                                "class_shares": class_shares(Pm, v, "dev"),
                                "condition_ratios": evaluate(Pm, v, "dev", bootstrap=0)["condition_ratios"]}
    d["extra"] = extra
    (SAI / "results" / f"{mid}.json").write_text(json.dumps(d, indent=1, default=float))
    print(f"[{mid}] BEST {d['points']:.3f} {d['ci95']} vsB3 {extra['vs_B3']}")
    for md, r in extra["by_mode"].items():
        print(f"  mode {md}: {r['points']:.3f} {r['ci95']} vsB3 {r['vs_B3']} blend {r['blend_gain']['points']:.3f} vs_base {r['blend_gain']['vs_base']} cls "
              f"{ {k: round(x, 3) for k, x in r['class_shares'].items()} }")
