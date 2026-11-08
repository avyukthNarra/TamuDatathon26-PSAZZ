import math, random, sys
from collections import defaultdict
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
sys.argv = sys.argv[:1] + ["--methods", "none"]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import residuals2 as R2
stack = R2.P.stack
def win(r, cap): return min(r["f2"], cap)
def dr(cap, which):
    per_q = defaultdict(list)
    for r in R2.TRAIN:
        if which == "cutoff" and r["back"]: continue
        per_q[(r["condition"], r["episode"], r["qid"])].append(stack.brier(stack.as_probs(r, win(r, cap)), r["outcome"]) - stack.brier(stack.as_probs(r, r["f2"]), r["outcome"]))
    dd = defaultdict(lambda: defaultdict(list))
    for (c, e, _), v in per_q.items(): dd[c][e].append(sum(v) / len(v))
    def score(sel=None):
        return sum(sum(sum(dd[c][x]) / len(dd[c][x]) for x in (dd[c] if sel is None else sel[c])) / len(dd[c] if sel is None else sel[c]) / stack.BASE_ERR[c] for c in dd) / len(dd)
    rng = random.Random(0); b = sorted(score({c: [rng.choice(list(e)) for _ in e] for c, e in dd.items()}) for _ in range(1000))
    return score(), b[25], b[974]
for cap in (1.0, 0.97, 0.96, 0.955, 0.95):
    preds = {r["qid"]: stack.as_probs(r, win(r, cap)) for r in R2.VAL}
    s = [stack.subset_score(preds, stack.vt, stack.vcond, stack.ref, e, 60.0)["technical_points"] for e in (stack.ALL_EP, *stack.HALVES)]
    cu, wi = dr(cap, "cutoff"), dr(cap, "window")
    print(f"ceiling {cap}: validation {s[0]:.2f} ({s[1]:.2f} / {s[2]:.2f}) | train cutoff {cu[0]:+.4f} [{cu[1]:+.4f},{cu[2]:+.4f}] window {wi[0]:+.4f} [{wi[1]:+.4f},{wi[2]:+.4f}]", flush=True)
