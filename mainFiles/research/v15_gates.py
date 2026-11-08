import math, sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import v13_eval as E
lg = E.lg
for rows in (E.P.VAL, E.P.TRAIN):
    for r in rows:
        r["is_short"] = sum(1 for t in r["seq"] if t < r["cutoff"]) < 120
def v(ww_long=0.4, ww_short=0.4, peer=0.5, sig=0.05, peer_short=None):
    def f(r, so):
        ww = ww_short if r["is_short"] else ww_long
        pe = peer_short if (peer_short is not None and r["is_short"]) else peer
        a, b = E.v11(r), 1 - r["v7c"][0]
        win = 1 / (1 + math.exp(-((1 - ww) * lg(a) + ww * lg(b) + pe * E.flat(r) + sig * r["signal"])))
        sb = [x / b for x in r["v7c"][1:]]
        raw = [max(x, 1e-4) for x in [1 - win] + [win * s for s in sb]]; t = sum(raw)
        return [x / t for x in raw]
    return f
E.BASE = v()   # v15
E.evaluate("v15", v())
for ws in (0.2, 0.3, 0.5):
    E.evaluate(f"short histories: v7c win weight {ws}", v(ww_short=ws))
for wl in (0.3, 0.5):
    E.evaluate(f"long histories: v7c win weight {wl}", v(ww_long=wl))
for ps in (0.25, 1.0):
    E.evaluate(f"short histories: extra peer {ps}", v(peer_short=ps))
