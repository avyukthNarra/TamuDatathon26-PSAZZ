"""M3: dynamic model averaging (Raftery et al.) over half-life experts replacing B3's holdout argmin.

Same experts and block-refit one-step predictions as M2. Each tick: model probs pi <- pi^alpha / sum(pi^alpha)
(forgetting alpha), then pi <- pi * Bernoulli predictive likelihood of the observed win flag, renormalised.
Output: pi-weighted probability mixture of the experts' full-history fits ('mix') or max-pi expert ('argmax').
Non-gappy machines: hl=inf as B3. Win model and size split are B3's.
Pre-declared grid (3 x 3 x 4 x 2 = 72 cells): pool {B3, B3+8, wide}, window {40,80,120},
  forgetting alpha {0.9,0.95,0.99,1.0}, out {mix, argmax}.
Extension (window best at edge, monotone gain 40<80<120): pool B3, window {160,180}, alpha {0.99,1.0},
  out {argmax,mix} = 8 more cells (cutoffs are >=192 so these stay in range).
Run: python3 methods/M3.py [--smoke]
"""
import sys, itertools
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
import _mem as m


def dma_pi(q, pool, window, alpha):
    te, y, P = m.seq_preds(q, pool, window, 8)
    N = len(pool)
    pi = np.ones(N) / N
    for j in range(len(te)):
        pi = pi ** alpha
        pi /= pi.sum()
        pi = pi * np.where(y[j] > 0, P[:, j], 1 - P[:, j])
        pi /= pi.sum()
    return pi


def make(p):
    pool = m.POOLS[p["pool"]]
    def predict(split):
        qs = m.prep(split)
        pws = []
        for q in qs:
            if not q.gappy:
                pws.append(m.pw_hl_cached(q, m.INF)); continue
            pi = dma_pi(q, pool, p["window"], p["alpha"])
            ph = np.array([m.pw_hl_cached(q, h) for h in pool])
            pws.append(float(pi @ ph) if p["out"] == "mix" else float(ph[int(np.argmax(pi))]))
        return m.assemble(split, pws)
    return predict


cells = [{"pool": a, "window": b, "alpha": c, "out": d} for a, b, c, d in
         itertools.product(("B3", "B3+8", "wide"), (80, 40, 120), (0.99, 0.9, 0.95, 1.0), ("argmax", "mix"))]
cells += [{"pool": "B3", "window": w, "alpha": a, "out": o} for w in (160, 180) for a in (0.99, 1.0) for o in ("argmax", "mix")]
if __name__ == "__main__":
    m.main_runner("M3", __doc__, cells, make, "dynamic model averaging over half-life experts")
