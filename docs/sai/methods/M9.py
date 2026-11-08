"""M9: BOCPD with changepoint evidence shared across an episode's machines.

Same BOCPD as M4 (regime-start posterior, Dirichlet-multinomial, hazard h), but each tick's likelihood multiplies the
query machine's predictive and its peers' predictives (peers' ticks < cutoff; episodes have a single cutoff, so no
peer tick at/after the query cutoff is used), peers' log-likelihood scaled by rho. Weights: survival P(start<=u),
floored at 0.05; mode 'surv' directly or 'surv*b3' (multiplied by B3 half-life weights). Non-gappy machines hl=inf.
Win model and size split are B3's.
Pre-declared grid (5 x 2 x 2 x 2 = 40 cells): hazard {1/20,1/50,1/200,1/1000,1/5000}, rho {0.5,1}, K {2,5},
  mode {surv, surv*b3}; alpha=2, floor 0.05 fixed. Cap 10 min per cell (posteriors cached per (h,rho,K)).
Run: python3 methods/M9.py [--smoke]
"""
import sys, itertools
sys.path.insert(0, __import__("os").path.dirname(__file__))
import _mem as m
import _cp as cp

_P = {}


def make(p):
    def predict(split):
        qs = m.prep(split)
        pws = []
        for q in qs:
            if not q.gappy:
                pws.append(m.pw_hl_cached(q, m.INF)); continue
            key = (split.name, q.i, p["h"], p["rho"], p["K"])
            if key not in _P:
                peers = split.peers(q.i)
                _P[key] = cp.bocpd_post([q.g] + [split.grid[j] for j in peers], q.c, p["h"], 2.0, p["K"], p["rho"])
            w = cp.surv_weights(q, _P[key], 0.05)
            if p["mode"] == "surv*b3":
                w = w * m.hl_w(q.ticks, q.c, m.b3_hl(q))
            pws.append(m.pw_from_w(q, w))
        return m.assemble(split, pws)
    return predict


cells = [{"h": a, "rho": b, "K": c, "mode": d} for a, b, c, d in
         itertools.product((1 / 200, 1 / 20, 1 / 50, 1 / 1000, 1 / 5000), (1.0, 0.5), (2, 5), ("surv*b3", "surv"))]
if __name__ == "__main__":
    m.main_runner("M9", __doc__, cells, make, "episode-shared BOCPD run-length posterior as ridge row weights")
