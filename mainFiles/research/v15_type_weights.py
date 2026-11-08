import math, sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import v13_eval as E
lg = E.lg
def v(ww=None):
    ww = ww or {}
    def f(r, so):
        w = ww.get(r["type"], 0.4)
        a, b = E.v11(r), 1 - r["v7c"][0]
        win = 1 / (1 + math.exp(-((1 - w) * lg(a) + w * lg(b) + 0.5 * E.flat(r) + 0.05 * r["signal"])))
        sb = [x / b for x in r["v7c"][1:]]
        raw = [max(x, 1e-4) for x in [1 - win] + [win * s for s in sb]]; t = sum(raw)
        return [x / t for x in raw]
    return f
E.BASE = v()
E.evaluate("v15", v())
for t in ("steady", "balanced", "jackpot"):
    for w in (0.2, 0.6):
        E.evaluate(f"{t}: v7c win weight {w}", v({t: w}))
