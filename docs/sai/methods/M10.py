"""M10: half-life implied by a local-level Kalman filter's steady-state gain (sanity check of B3's selected half-lives).

Per machine on observed win flags: p = mean, R = p(1-p). Block means of `blk` consecutive observed ticks; level
innovation variance per observation Q = max(Var(diff of block means) - 2R/blk, 0)/blk (random-walk level).
Steady-state prior variance P- = (Q + sqrt(Q^2+4QR))/2, gain K = P-/(P-+R), EWMA half-life hl_k = ln(.5)/ln(1-K),
scaled by `mult` and clipped to [hl_min, inf] (hl_k > 1e4 -> inf). Ridge uses hl_k directly ('kalman') or Kalman
hl_k as the centre of a 3-point choice {hl_k/2,hl_k,2*hl_k} picked by B3's 40-tick holdout ('kalman+hold').
Also reports corr(log hl_k, log B3 hl) and fraction of queries whose B3 half-life equals the nearest grid value to hl_k.
Non-gappy machines hl=inf as B3. B3 win model and size split.
Pre-declared grid (2 x 3 x 3 x 2 = 36 cells): blk {8,16}, mult {0.5,1,2}, hl_min {8,16,32}, mode {kalman,kalman+hold}.
Run: python3 methods/M10.py [--smoke]
"""
import sys, itertools, json
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
import _mem as m


def kalman_hl(q, blk):
    x = q.win[~np.isnan(q.win)]
    n = len(x)
    p = x.mean(); R = max(p * (1 - p), 1e-3)
    nb = n // blk
    if nb < 4:
        return m.INF
    bm = x[n - nb * blk:].reshape(nb, blk).mean(1)
    Q = max(np.var(np.diff(bm)) - 2 * R / blk, 0.0) / blk
    if Q <= 1e-12:
        return m.INF
    Pm = (Q + np.sqrt(Q * Q + 4 * Q * R)) / 2
    K = Pm / (Pm + R)
    return float(np.log(0.5) / np.log(1 - K))


def pick(q, p):
    hk = kalman_hl(q, p["blk"]) * p["mult"]
    hk = m.INF if hk > 1e4 else max(hk, p["hl_min"])
    if p["mode"] == "kalman" or hk >= m.INF:
        return hk
    cand = tuple(max(h, p["hl_min"]) for h in (hk / 2, hk, hk * 2))
    return cand[int(np.argmin(m.holdout_errs(q, cand, 40)))]


def make(p):
    def predict(split):
        qs = m.prep(split)
        pws = [m.pw_hl_cached(q, m.INF) if not q.gappy else m.pw_hl_cached(q, round(pick(q, p), 3)) for q in qs]
        return m.assemble(split, pws)
    return predict


cells = [{"blk": a, "mult": b, "hl_min": c, "mode": d} for a, b, c, d in
         itertools.product((8, 16), (1.0, 0.5, 2.0), (16, 8, 32), ("kalman", "kalman+hold"))]


def sanity(split, p):
    qs = [q for q in m.prep(split) if q.gappy]
    hk = np.array([min(kalman_hl(q, p["blk"]), 1e4) for q in qs])
    hb = np.array([min(m.b3_hl(q), 1e4) for q in qs])
    return {"corr_log_hl": float(np.corrcoef(np.log(hk), np.log(hb))[0, 1]), "median_kalman_hl": float(np.median(hk)),
            "frac_kalman_inf": float(np.mean(hk >= 1e4)), "frac_b3_inf": float(np.mean(hb >= 1e4)),
            "b3_hl_counts": {str(k): int(np.sum(hb == k)) for k in sorted(set(hb))}}


if __name__ == "__main__":
    out = m.main_runner("M10", __doc__, cells, make, "Kalman steady-state gain half-life")
    if out:
        v = m.load_split("validation")
        s = sanity(v, out[0]["params"] if isinstance(out[0], dict) else cells[0])
        (m.SAI / "grids" / "M10_sanity.json").write_text(json.dumps(s, indent=1))
        print(s)
