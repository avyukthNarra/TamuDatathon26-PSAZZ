import math, sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import subcell as S
    import size_flip as F
E, lg = S.E, S.lg
def v(lam=0.0):
    base = S.v(1.0, 16, {"jackpot"})   # = v17/v18 on the cache
    def f(r, so):
        p = base(r, so)
        if lam and abs(r["flip_d"]) > 1e-9 and r["type"] in F.MODES:
            A, B = F.MODES[r["type"]]; w = 1 - p[0]
            s = [x / w for x in p[1:]]
            s = [(1 - lam) * x + lam * 0.5 * (a + b) for x, a, b in zip(s, A, B)]
            raw = [max(x, 1e-4) for x in [1 - w] + [w * y for y in s]]; t = sum(raw); p = [x / t for x in raw]
        return p
    return f
E.BASE = v()
E.evaluate("v18 (cache)", v())
for lam in (0.05, 0.1, 0.2):
    E.evaluate(f"hedge size toward 50/50 in flip episodes {lam}", v(lam))
