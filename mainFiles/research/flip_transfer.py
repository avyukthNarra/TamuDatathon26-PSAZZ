"""Transfer v7c's episode flip swap to our win model: our logit += lam * (logit v7c with swap - without)."""
import json, math, sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import v13_eval as E
lg = E.lg
FL = json.load(open("runs/research-cache/v7c_flip.json"))
for r in E.P.VAL:
    a, b = FL[f"{r['qid']}|v"]; r["flip_d"] = lg(1 - a[0]) - lg(1 - b[0])
for r in E.P.TRAIN:
    a, b = FL[f"{r['qid']}|{r['back']}"]; r["flip_d"] = lg(1 - a[0]) - lg(1 - b[0])
print("share of rows where the flip swap changes v7c's win: val %.3f train %.3f" % (
    sum(abs(r["flip_d"]) > 1e-9 for r in E.P.VAL) / len(E.P.VAL), sum(abs(r["flip_d"]) > 1e-9 for r in E.P.TRAIN) / len(E.P.TRAIN)))
def v(lam=0.0, ww=0.4):
    def f(r, so):
        a, b = E.v11(r), 1 - r["v7c"][0]
        z = (1 - ww) * (lg(a) + lam * r["flip_d"]) + ww * lg(b) + 0.5 * E.flat(r) + 0.05 * r["signal"]
        win = 1 / (1 + math.exp(-z))
        sb = [x / b for x in r["v7c"][1:]]
        raw = [max(x, 1e-4) for x in [1 - win] + [win * s for s in sb]]; t = sum(raw)
        return [x / t for x in raw]
    return f
E.BASE = v()
E.evaluate("v15", v())
for lam in (0.5, 1.0):
    E.evaluate(f"flip transfer {lam} to our win", v(lam))
E.evaluate("flip transfer 1.0, v7c weight 0.3", v(1.0, 0.3))
