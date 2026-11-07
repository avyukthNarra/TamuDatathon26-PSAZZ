"""M2: fixed-share (Herbster-Warmuth) over half-life experts replacing B3's holdout argmin.

Experts = ridge fits with half-life h in pool; each is scored on block-refit one-step-ahead win-flag predictions
over the last `window` observed ticks (refit every 8 ticks, using ticks before the block only). Weights:
w <- w*exp(-eta*loss), renormalise, w <- (1-a)w + a/N. Output: probability mixture of the experts' full-history fits
at the cutoff ('mix') or the max-weight expert ('argmax'). Non-gappy machines keep hl=inf as in B3. Win model (pair
ridge) and size split are B3's.
Pre-declared grid (3 x 3 x 4 x 3 x 2 = 216 cells): pool in {B3, B3+8, wide(inf..4)}, window in {40,80,120},
  eta in {2,5,10,20} (squared loss), share a in {0.01,0.05,0.2}, out in {mix, argmax}.
Extension (declared after M3 showed longer windows help): pool B3, window {160,180}, eta {2,5,10}, a {0.01,0.05},
  out {mix,argmax} = 24 more cells.
Run: python3 methods/M2.py [--smoke]
"""
import sys, itertools
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
import _mem as m


def weights_fs(q, pool, window, eta, a, loss="sq"):
    te, y, P = m.seq_preds(q, pool, window, 8)
    N = len(pool)
    w = np.ones(N) / N
    for j in range(len(te)):
        l = (P[:, j] - y[j]) ** 2 if loss == "sq" else -np.log(np.where(y[j] > 0, P[:, j], 1 - P[:, j]))
        w = w * np.exp(-eta * l)
        w /= w.sum()
        w = (1 - a) * w + a / N
    return w


def make(p):
    pool = m.POOLS[p["pool"]]
    def predict(split):
        qs = m.prep(split)
        pws = []
        for q in qs:
            if not q.gappy:
                pws.append(m.pw_hl_cached(q, m.INF)); continue
            w = weights_fs(q, pool, p["window"], p["eta"], p["a"])
            ph = np.array([m.pw_hl_cached(q, h) for h in pool])
            pws.append(float(w @ ph) if p["out"] == "mix" else float(ph[int(np.argmax(w))]))
        return m.assemble(split, pws)
    return predict


cells = [{"pool": a, "window": b, "eta": c, "a": d, "out": e} for a, b, c, d, e in
         itertools.product(("B3", "B3+8", "wide"), (40, 80, 120), (2, 5, 10, 20), (0.01, 0.05, 0.2), ("mix", "argmax"))]
cells += [{"pool": "B3", "window": w, "eta": e, "a": a, "out": o} for w in (160, 180) for e in (2, 5, 10)
          for a in (0.01, 0.05) for o in ("mix", "argmax")]
if __name__ == "__main__":
    m.main_runner("M2", __doc__, cells, make, "fixed-share over half-life experts replacing holdout argmin")
