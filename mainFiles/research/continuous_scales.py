"""Task 1: continuous add-on scales chosen on held-out TRAIN rows, checked on validation."""
import math, sys
from collections import defaultdict
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import subcell as S
E, lg, stack = S.E, S.lg, S.E.stack
BASE = {"ww": 0.3, "peer": 0.5, "signal": 0.05, "flip": 1.0, "sub": 1.0}
for rows in (E.P.VAL, E.P.TRAIN):
    for r in rows: r["subc"] = S.corr(r, 16) if r["type"] == "jackpot" else 0.0
def f_of(c):
    def f(r, so):
        a, b = E.v11(r), 1 - r["v7c"][0]
        za = lg(a) + c["flip"] * r["flip_d"] + c["sub"] * r["subc"]
        z = (1 - c["ww"]) * za + c["ww"] * lg(b) + c["peer"] * E.flat(r) + c["signal"] * r["signal"]
        win = 1 / (1 + math.exp(-z)); sb = [x / b for x in r["v7c"][1:]]
        raw = [max(x, 1e-4) for x in [1 - win] + [win * s for s in sb]]; t = sum(raw)
        return [x / t for x in raw]
    return f
def train_loss(c):  # metric-weighted window Brier on the out-of-fold train rows
    f = f_of(c); per = defaultdict(list)
    for r in E.P.TRAIN:
        per[(r["condition"], r["episode"])].append(stack.brier(f(r, None), r["outcome"]))
    byc = defaultdict(list)
    for (cnd, _), v in per.items(): byc[cnd].append(sum(v) / len(v))
    return sum(sum(v) / len(v) / stack.BASE_ERR[cnd] for cnd, v in byc.items()) / len(byc)
c = dict(BASE)
for sweep in range(2):
    for k in ("peer", "signal", "flip", "sub", "ww"):
        grid = [BASE[k] * m for m in (0, 0.25, 0.5, 0.75, 1.0, 1.25)] if k != "ww" else [0.1, 0.2, 0.3, 0.4, 0.5]
        best = min(grid, key=lambda g: train_loss(dict(c, **{k: g})))
        c[k] = best
    print(f"sweep {sweep}: train-chosen settings {c}", flush=True)
E.BASE = f_of(BASE)
E.evaluate("v18 settings", f_of(BASE))
E.evaluate("train-chosen continuous settings", f_of(c))
shrunk = {k: 0.5 * BASE[k] + 0.5 * c[k] for k in BASE}
E.evaluate("halfway to train-chosen (L2 shrink)", f_of(shrunk))
print("shrunk settings", shrunk)
