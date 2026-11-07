"""M4: Bayesian online changepoint detection (Adams & MacKay, Dirichlet-multinomial) as the memory selector.

Per machine, run-length (regime start) posterior at the cutoff over ticks < cutoff (missing ticks carry no evidence,
constant hazard h, symmetric Dirichlet prior alpha over K classes: K=2 win/loss or K=5 payouts). Ridge row weight of
tick u = P(regime start <= u) (survival), floored at `floor`; mode 'surv' uses it directly, 'surv*b3' multiplies it by
B3's holdout-selected half-life weights (a changepoint reset on top of B3 memory), 'map' uses a hard window from the
MAP regime start. Non-gappy machines: hl=inf as in B3. Win model and size split are B3's.
Pre-declared grid (4 x 2 x 2 x 2 x 3 = 96 cells): hazard {1/20,1/50,1/100,1/200}, alpha {0.5,2}, K {2,5},
  floor {0,0.05}, mode {surv, surv*b3, map}. Cap 10 min per cell.
Extension (declared after the first 96 cells showed the best hazard at the grid edge): hazard {1/500,1/1000,1/5000}
  x alpha {0.5,2} x K {2,5} x floor {0,0.05} x mode {surv*b3} = 24 more cells.
Run: python3 methods/M4.py [--smoke]
"""
import sys, itertools
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
import _mem as m
import _cp as cp

_P = {}


def post_for(q, split, p):
    key = (split.name, q.i, p["h"], p["alpha"], p["K"])
    if key not in _P:
        _P[key] = cp.bocpd_post([q.g], q.c, p["h"], p["alpha"], p["K"])
    return _P[key]


def make(p):
    def predict(split):
        qs = m.prep(split)
        pws = []
        for q in qs:
            if not q.gappy:
                pws.append(m.pw_hl_cached(q, m.INF)); continue
            post = post_for(q, split, p)
            if p["mode"] == "map":
                w = cp.start_weights(q, int(np.argmax(post)), p["floor"])
            else:
                w = cp.surv_weights(q, post, p["floor"])
                if p["mode"] == "surv*b3":
                    w = w * m.hl_w(q.ticks, q.c, m.b3_hl(q))
            pws.append(m.pw_from_w(q, w))
        return m.assemble(split, pws)
    return predict


cells = [{"h": a, "alpha": b, "K": c, "floor": d, "mode": e} for a, b, c, d, e in
         itertools.product((1 / 50, 1 / 20, 1 / 100, 1 / 200), (0.5, 2.0), (2, 5), (0.0, 0.05), ("surv", "surv*b3", "map"))]
cells += [{"h": a, "alpha": b, "K": c, "floor": d, "mode": "surv*b3"} for a, b, c, d in
          itertools.product((1 / 500, 1 / 1000, 1 / 5000), (0.5, 2.0), (2, 5), (0.0, 0.05))]
if __name__ == "__main__":
    m.main_runner("M4", __doc__, cells, make, "BOCPD run-length posterior as ridge row weights")
