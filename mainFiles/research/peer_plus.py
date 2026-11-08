"""Refining the peer correction (run from mainFiles/). All fitted on TRAIN; validation untouched."""
import math, random, sys
from collections import defaultdict
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
sys.argv = sys.argv[:1] + ["--methods", "none"]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import peer_correction as P
stack = P.stack
for rows in (P.VAL, P.TRAIN):
    for r in rows:
        for k in range(1, 9):
            r["x"][f"lag{k}"] = P.share(r, [k])
        sh = r["x"]["peers 1-5"]
        r["short"] = 1.0 if sum(1 for t in r["seq"] if t < r["cutoff"]) < 120 else 0.0
        r["x"]["short x peers"] = None if sh is None else r["short"] * sh
        r["x"]["signal"] = float(r["signal"])
for k in [f"lag{k}" for k in range(1, 9)] + ["short x peers", "signal"]:
    v = [r["x"][k] for r in P.TRAIN if r["x"][k] is not None]; P.MEANS[k] = sum(v) / len(v)
def dr(names, b, which, base_names=("peers 1-5",), base_b=(1.0,)):
    per_q = defaultdict(list)
    for r in P.TRAIN:
        if which == "cutoff" and r["back"]: continue
        per_q[(r["condition"], r["episode"], r["qid"])].append(stack.brier(stack.as_probs(r, P.corrected(r, names, b)), r["outcome"]) - stack.brier(stack.as_probs(r, P.corrected(r, list(base_names), list(base_b))), r["outcome"]))
    dd = defaultdict(lambda: defaultdict(list))
    for (c, e, _), v in per_q.items(): dd[c][e].append(sum(v) / len(v))
    def score(sel=None):
        return sum(sum(sum(dd[c][x]) / len(dd[c][x]) for x in (dd[c] if sel is None else sel[c])) / len(dd[c] if sel is None else sel[c]) / stack.BASE_ERR[c] for c in dd) / len(dd)
    rng = random.Random(0); bb = sorted(score({c: [rng.choice(list(e)) for _ in e] for c, e in dd.items()}) for _ in range(500))
    return score(), bb[12], bb[487]
def report(label, names, b):
    s = P.val_score(names, b); cu, wi = dr(names, b, "cutoff"), dr(names, b, "window")
    print(f"{label:46s} validation {s[0]['technical_points']:.2f} ({s[1]['technical_points']:.2f} / {s[2]['technical_points']:.2f}) | train vs v10: cutoff {cu[0]:+.4f} [{cu[1]:+.4f},{cu[2]:+.4f}] window {wi[0]:+.4f} [{wi[1]:+.4f},{wi[2]:+.4f}]", flush=True)
report("v10 (peers 1-5 x 1.0)", ["peers 1-5"], [1.0])
lags = [f"lag{k}" for k in range(1, 9)]
b = P.fit(P.TRAIN, lags, l2=1e-2); print("train per-lag weights:", [round(x, 2) for x in b])
for scale in (0.4, 0.55, 0.7):
    report(f"per-lag weights x{scale}", lags, [x * scale for x in b])
b2 = P.fit(P.TRAIN, ["peers 1-5", "short x peers"]); print("train peers / short x peers:", [round(x, 2) for x in b2])
for scale in (0.55, 0.7, 1.0):
    report(f"peers + short interaction x{scale}", ["peers 1-5", "short x peers"], [x * scale for x in b2])
report("peers 1.0 + short x peers 0.75", ["peers 1-5", "short x peers"], [1.0, 0.75])
report("peers 1.0 + short x peers 0.75 + signal 0.05", ["peers 1-5", "short x peers", "signal"], [1.0, 0.75, 0.05])
