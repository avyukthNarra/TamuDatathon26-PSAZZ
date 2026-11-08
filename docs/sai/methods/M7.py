"""M7: CUSUM / rescaled-range drift detection on each machine's win-flag stream.

Statistics on observed ticks (mean mu, binomial sd): backward CUSUM scan z = max_k |sum_{last k}(x-mu)| /
(sd*sqrt(k(1-k/n))) with k in [minseg, n-minseg] (the change point is k observations back), and the rescaled
range V = (max-min of partial sums of x-mu)/(sd*sqrt(n)). Modes:
  win      : change declared if z>thr; row weight 1 after the change tick, `floor` before (no change: all 1)
  win*b3   : the same multiplied by B3's holdout half-life weights
  gate     : stat (z or V) > thr -> B3 holdout selection, otherwise hl=inf (range feature as a drift gate)
Non-gappy machines: hl=inf as B3. B3 win model and size split.
Pre-declared grid: thr {1.5,2,2.5,3,4} (z) x minseg {20} x [win,win*b3] x floor {0,0.1,0.3} = 30 cells;
  gate: stat {z,V} x thr {1,1.5,2,2.5,3,4} = 12 cells (V thresholds in V units, typically 0.5-3).
Run: python3 methods/M7.py [--smoke]
"""
import sys, itertools
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
import _mem as m
import _cp as cp

_S = {}


def stats(q, split, minseg=20):
    key = (split.name, q.i, minseg)
    if key not in _S:
        _S[key] = cp.scan_stats(q, minseg)
    return _S[key]


def make(p):
    def predict(split):
        qs = m.prep(split)
        pws = []
        for q in qs:
            if not q.gappy:
                pws.append(m.pw_hl_cached(q, m.INF)); continue
            zs, kb, n, V = stats(q, split)
            if p["mode"] == "gate":
                s = zs if p["stat"] == "z" else V
                hl = m.b3_hl(q) if s > p["thr"] else m.INF
                pws.append(m.pw_hl_cached(q, hl)); continue
            st = cp.scan_start(q, kb) if zs > p["thr"] else 0
            w = cp.start_weights(q, st, p["floor"])
            if p["mode"] == "win*b3":
                w = w * m.hl_w(q.ticks, q.c, m.b3_hl(q))
            pws.append(m.pw_from_w(q, w))
        return m.assemble(split, pws)
    return predict


cells = [{"mode": md, "thr": t, "floor": f} for md, t, f in itertools.product(("win*b3", "win"), (2.5, 1.5, 2, 3, 4), (0.0, 0.1, 0.3))]
cells += [{"mode": "gate", "stat": s, "thr": t} for s in ("z", "V") for t in (2.0, 1.0, 1.5, 2.5, 3.0, 4.0)]
if __name__ == "__main__":
    m.main_runner("M7", __doc__, cells, make, "CUSUM scan / rescaled range drift detection")
