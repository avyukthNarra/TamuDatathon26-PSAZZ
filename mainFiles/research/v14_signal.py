import math, sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import v13_eval as E
V14 = E.make(ww=0.4, extra_peer=0.5)
E.BASE = V14   # compare against v14
def with_signal(cs):
    def f(r, so):
        p = V14(r, so); w = 1 - p[0]
        z = math.log(max(w, 1e-6) / max(1 - w, 1e-6)) + cs * r["signal"]
        nw = 1 / (1 + math.exp(-z))
        raw = [max(x, 1e-4) for x in [1 - nw] + [nw * s / w for s in p[1:]]]; t = sum(raw)
        return [x / t for x in raw]
    return f
for cs in (0.0, 0.05, 0.1, 0.15):
    E.evaluate(f"v14 + signal term {cs}", with_signal(cs))
