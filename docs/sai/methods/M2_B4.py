"""M2_B4: fixed-share over half-life experts on the B4 base (lags (7,), near-OLS ridge), replacing holdout argmin.
Experts/loss/output exactly as M2 (block refit every 8 ticks, squared loss, non-gappy keep hl=inf).
Pre-declared grids (scope dev):
  Round A (lam 0.05): pool {B3, B3+8, wide(inf..4)} x window {40,80,120,180} x eta {2,5,10,20} x share a {0.01,0.05,0.2}
     x out {mix, argmax} = 288 cells
  Round B (lam {0.5,2}): pool B3 x window {120,180} x eta {2,5} x a {0.01,0.05} x out {mix,argmax} = 32 cells
  Round C (extension, declared if best at window edge): pool {B3, long(inf,256,128,64,32)} x window {160,190} x eta {2,5}
     x a {0.01} x out {argmax,mix}, lam 0.05 = 16 cells
Run: python3 methods/M2_B4.py [--smoke]
"""
import sys, itertools
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
import _mem4 as m
from M2 import weights_fs
import M2
m.POOLS["long"] = (m.INF, 256, 128, 64, 32)
M2.m = m  # weights_fs uses m.seq_preds


def make(p):
    pool = m.POOLS[p["pool"]]
    def predict(split):
        m.set_lam(p["lam"])
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


cells = [{"lam": 0.05, "pool": a, "window": b, "eta": c, "a": d, "out": e} for a, b, c, d, e in
         itertools.product(("B3", "B3+8", "wide"), (40, 80, 120, 180), (2, 5, 10, 20), (0.01, 0.05, 0.2), ("mix", "argmax"))]
cells += [{"lam": l, "pool": "B3", "window": w, "eta": e, "a": a, "out": o} for l in (0.5, 2.0) for w in (120, 180)
          for e in (2, 5) for a in (0.01, 0.05) for o in ("mix", "argmax")]
cells += [{"lam": 0.05, "pool": pl, "window": w, "eta": e, "a": 0.01, "out": o} for pl in ("B3", "long") for w in (160, 190)
          for e in (2, 5) for o in ("argmax", "mix")]
if __name__ == "__main__":
    m.main_runner("M2_B4", __doc__, cells, make, "fixed-share over half-life experts on B4 base")
