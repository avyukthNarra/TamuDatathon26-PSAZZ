"""M5: ADWIN adaptive window on each machine's win-flag stream as the memory selector.

ADWIN (Hoeffding-bound cut test, delta' = delta/window length, minimum sub-window `minseg` observations) yields the
start tick of the surviving window. Ridge row weight = 1 inside the window, `floor` before it ('win'), optionally
multiplied by B3's holdout half-life weights ('win*b3'). Non-gappy machines: hl=inf as B3. B3 win model and size split.
Pre-declared grid (4 x 2 x 3 x 2 = 48 cells): delta {0.002,0.01,0.05,0.2}, minseg {10,20}, floor {0,0.1,0.3},
  mode {win, win*b3}.
Run: python3 methods/M5.py [--smoke]
"""
import sys, itertools
sys.path.insert(0, __import__("os").path.dirname(__file__))
import _mem as m
import _cp as cp

_S = {}


def make(p):
    def predict(split):
        qs = m.prep(split)
        pws = []
        for q in qs:
            if not q.gappy:
                pws.append(m.pw_hl_cached(q, m.INF)); continue
            key = (split.name, q.i, p["delta"], p["minseg"])
            if key not in _S:
                _S[key] = cp.adwin_start(q, p["delta"], p["minseg"])
            w = cp.start_weights(q, _S[key], p["floor"])
            if p["mode"] == "win*b3":
                w = w * m.hl_w(q.ticks, q.c, m.b3_hl(q))
            pws.append(m.pw_from_w(q, w))
        return m.assemble(split, pws)
    return predict


cells = [{"delta": a, "minseg": b, "floor": c, "mode": d} for a, b, c, d in
         itertools.product((0.002, 0.01, 0.05, 0.2), (10, 20), (0.0, 0.1, 0.3), ("win", "win*b3"))]
if __name__ == "__main__":
    m.main_runner("M5", __doc__, cells, make, "ADWIN window start as ridge row weights")
