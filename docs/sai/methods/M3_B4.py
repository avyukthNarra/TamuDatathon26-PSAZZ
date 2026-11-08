"""M3_B4: dynamic model averaging over half-life experts on the B4 base (lags (7,), near-OLS ridge).
Experts/likelihood as M3. Pre-declared grids (scope dev):
  Round A (lam 0.05): pool {B3, B3+8, wide} x window {40,80,120,180} x alpha {0.9,0.95,0.99,1.0} x out {argmax,mix} = 96 cells
  Round B (lam {0.5,2}): pool B3 x window {120,180} x alpha {0.99,1.0} x out {argmax,mix} = 16 cells
  Round C (extension, if best at the window edge): pool {B3,long(inf,256,128,64,32)} x window {160,190} x alpha {0.99,1.0}
     x out {argmax,mix}, lam 0.05 = 16 cells
Run: python3 methods/M3_B4.py [--smoke]
"""
import sys, itertools
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
import _mem4 as m
import M3
M3.m = m
from M3 import dma_pi
m.POOLS["long"] = (m.INF, 256, 128, 64, 32)


def make(p):
    pool = m.POOLS[p["pool"]]
    def predict(split):
        m.set_lam(p["lam"])
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


cells = [{"lam": 0.05, "pool": a, "window": b, "alpha": c, "out": d} for a, b, c, d in
         itertools.product(("B3", "B3+8", "wide"), (180, 40, 80, 120), (0.99, 0.9, 0.95, 1.0), ("argmax", "mix"))]
cells += [{"lam": l, "pool": "B3", "window": w, "alpha": a, "out": o} for l in (0.5, 2.0) for w in (120, 180)
          for a in (0.99, 1.0) for o in ("argmax", "mix")]
cells += [{"lam": 0.05, "pool": pl, "window": w, "alpha": a, "out": o} for pl in ("B3", "long") for w in (160, 190)
          for a in (0.99, 1.0) for o in ("argmax", "mix")]
if __name__ == "__main__":
    m.main_runner("M3_B4", __doc__, cells, make, "DMA over half-life experts on B4 base")
