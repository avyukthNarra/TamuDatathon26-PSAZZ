import sys, math, random
from collections import defaultdict
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
sys.argv = sys.argv[:1] + ["--methods", "none"]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import peer_correction as P
stack = P.stack
def lg(p): p = min(max(p, 1e-4), 1 - 1e-4); return math.log(p / (1 - p))
def win(r, cs):
    sh = r["x"]["peers 1-5"]
    z = lg(r["f"]) + (0 if sh is None else 1.0 * (sh - 0.4187)) + cs * r["signal"]
    return 1 / (1 + math.exp(-z))
def dr(cs, which):
    per_q = defaultdict(list)
    for r in P.TRAIN:
        if which == "cutoff" and r["back"]: continue
        per_q[(r["condition"], r["episode"], r["qid"])].append(stack.brier(stack.as_probs(r, win(r, cs)), r["outcome"]) - stack.brier(stack.as_probs(r, win(r, 0)), r["outcome"]))
    dd = defaultdict(lambda: defaultdict(list))
    for (c, e, _), v in per_q.items(): dd[c][e].append(sum(v) / len(v))
    def score(sel=None):
        return sum(sum(sum(dd[c][x]) / len(dd[c][x]) for x in (dd[c] if sel is None else sel[c])) / len(dd[c] if sel is None else sel[c]) / stack.BASE_ERR[c] for c in dd) / len(dd)
    rng = random.Random(0); b = sorted(score({c: [rng.choice(list(e)) for _ in e] for c, e in dd.items()}) for _ in range(1000))
    return score(), b[25], b[974]
for cs in (0.0, 0.05, 0.1, 0.15, 0.2):
    preds = {r["qid"]: stack.as_probs(r, win(r, cs)) for r in P.VAL}
    s = [stack.subset_score(preds, stack.vt, stack.vcond, stack.ref, e, 60.0)["technical_points"] for e in (stack.ALL_EP, *stack.HALVES)]
    cu, wi = dr(cs, "cutoff"), dr(cs, "window")
    print(f"signal coef {cs:4}: validation {s[0]:.2f} ({s[1]:.2f} / {s[2]:.2f}) | train vs peers-only: cutoff {cu[0]:+.4f} [{cu[1]:+.4f},{cu[2]:+.4f}] window {wi[0]:+.4f} [{wi[1]:+.4f},{wi[2]:+.4f}]", flush=True)
