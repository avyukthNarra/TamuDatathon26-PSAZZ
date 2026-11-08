"""M6: JSD/TV-gated blend of a short-memory and a long-memory pair-ridge fit.

D = divergence between 5-class payout frequencies of the last `ws` observed ticks and the earlier history (add-0.5
smoothing). Gate g = sigmoid(a*(u - qthr)), u = percentile rank of D among the queries of the split being predicted
(unlabeled, no conditions used). pw = g*pw(hl_s) + (1-g)*pw(hl_l). hl_l 'b3' = B3's holdout-selected half-life.
Non-gappy machines: hl=inf only. Win model and size split are B3's.
Pre-declared grid (2 x 3 x 2 x 3 x 3 x 3 = 324 cells): dist {jsd,tv}, ws {16,24,40}, hl_s {16,32},
  hl_l {inf,128,b3}, qthr {0.3,0.5,0.7}, slope a {5,10,20}.
Run: python3 methods/M6.py [--smoke]
"""
import sys, itertools
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
import _mem as m

PAYS = (0, 1, 2, 5, 20)


def freq(g):
    v = g[~np.isnan(g)]
    c = np.array([np.sum(v == p) for p in PAYS], float) + 0.5
    return c / c.sum()


def divergence(q, ws, dist):
    s = freq(q.g[q.c - ws:]); l = freq(q.g[:q.c - ws])
    if dist == "tv":
        return 0.5 * np.abs(s - l).sum()
    mm = 0.5 * (s + l)
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return 0.5 * kl(s, mm) + 0.5 * kl(l, mm)


def make(p):
    def predict(split):
        qs = m.prep(split)
        D = np.array([divergence(q, p["ws"], p["dist"]) for q in qs])
        u = (np.argsort(np.argsort(D)) + 0.5) / len(D)
        g = 1 / (1 + np.exp(-p["a"] * (u - p["qthr"])))
        pws = []
        for q, gi in zip(qs, g):
            if not q.gappy:
                pws.append(m.pw_hl_cached(q, m.INF)); continue
            hl = m.b3_hl(q) if p["hl_l"] == "b3" else p["hl_l"]
            pws.append(gi * m.pw_hl_cached(q, p["hl_s"]) + (1 - gi) * m.pw_hl_cached(q, hl))
        return m.assemble(split, pws)
    return predict


cells = [{"dist": a, "ws": b, "hl_s": c, "hl_l": d, "qthr": e, "a": f} for a, b, c, d, e, f in
         itertools.product(("jsd", "tv"), (24, 16, 40), (16, 32), (m.INF, 128, "b3"), (0.5, 0.3, 0.7), (10, 5, 20))]
if __name__ == "__main__":
    m.main_runner("M6", __doc__, cells, make, "JSD/TV-gated blend of short and long memory pair-ridge fits")
