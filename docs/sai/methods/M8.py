"""M8: HMM regimes (Rabiner) as the memory selector.

Per machine, K-state Bernoulli-emission HMM on the win-flag stream (ticks < cutoff, missing = no emission), fixed
deterministic init (state win probs = history mean +- 0.15 steps, symmetric transitions with self-prob `stay`),
exactly 10 Baum-Welch iterations. pi_next = filtered state posterior at the last tick propagated one step. Ridge row
weight of tick u = (sum_k pi_next(k)*gamma_u(k))^power, i.e. how much tick u looks like the regime the machine is
about to be in; floored; mode 'hmm' direct, 'hmm*b3' multiplied by B3's holdout half-life weights. Non-gappy machines
hl=inf as B3. B3 win model and size split.
Pre-declared grid (2 x 2 x 2 x 2 x 2 = 32 cells): K {2,3}, stay {0.9,0.98}, power {1,2}, floor {0,0.1},
  mode {hmm, hmm*b3}. Cap 10 min per cell (HMM fits cached per (K,stay)).
Extension (stay best at the edge 0.98): stay 0.995 x K {2,3} x power 1 x floor 0.1 x mode hmm*b3 = 2 cells.
Run: python3 methods/M8.py [--smoke]
"""
import sys, itertools
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
import _mem as m
import _cp as cp

_H = {}


def make(p):
    def predict(split):
        qs = m.prep(split)
        pws = []
        for q in qs:
            if not q.gappy:
                pws.append(m.pw_hl_cached(q, m.INF)); continue
            key = (split.name, q.i, p["K"], p["stay"])
            if key not in _H:
                _H[key] = cp.hmm_gamma(q, p["K"], p["stay"])
            gam, pn = _H[key]
            w = (gam[q.ticks] @ pn) ** p["power"]
            w = np.maximum(w, p["floor"])
            if p["mode"] == "hmm*b3":
                w = w * m.hl_w(q.ticks, q.c, m.b3_hl(q))
            pws.append(m.pw_from_w(q, w))
        return m.assemble(split, pws)
    return predict


cells = [{"K": a, "stay": b, "power": c, "floor": d, "mode": e} for a, b, c, d, e in
         itertools.product((2, 3), (0.98, 0.9), (1, 2), (0.1, 0.0), ("hmm*b3", "hmm"))]
cells += [{"K": k, "stay": 0.995, "power": 1, "floor": 0.1, "mode": "hmm*b3"} for k in (2, 3)]
if __name__ == "__main__":
    m.main_runner("M8", __doc__, cells, make, "HMM regime posterior as ridge row weights")
