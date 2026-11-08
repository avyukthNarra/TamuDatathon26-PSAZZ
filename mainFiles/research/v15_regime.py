import math, sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import v13_eval as E
lg = E.lg
def v(wr=0.0, ww=0.4):
    def f(r, so):
        a, b, g = E.v11(r), 1 - r["v7c"][0], r["regime"]
        z = (1 - ww - wr) * lg(a) + ww * lg(b) + wr * lg(g) + 0.5 * E.flat(r) + 0.05 * r["signal"]
        win = 1 / (1 + math.exp(-z))
        sb = [x / b for x in r["v7c"][1:]]
        raw = [max(x, 1e-4) for x in [1 - win] + [win * s for s in sb]]; t = sum(raw)
        return [x / t for x in raw]
    return f
E.BASE = v()
E.evaluate("v15", v())
for wr in (0.1, 0.2):
    E.evaluate(f"+ regime model {wr} (taken from ours)", v(wr=wr))
    E.evaluate(f"+ regime model {wr} (taken from v7c)", v(wr=wr, ww=0.4 - wr))
