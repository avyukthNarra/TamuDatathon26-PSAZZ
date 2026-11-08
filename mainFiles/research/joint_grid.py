import math, random, sys
from collections import defaultdict
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
sys.argv = sys.argv[:1] + ["--methods", "none"]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import peer_correction as P
stack = P.stack
def lg(p): p = min(max(p, 1e-4), 1 - 1e-4); return math.log(p / (1 - p))
def win(r, wm, c):
    z = wm * lg(1 - r["mixer"][0]) + (1 - wm) * lg(r["hmm"])
    sh = r["x"]["peers 1-5"]
    return 1 / (1 + math.exp(-(z + (0 if sh is None else c * (sh - 0.4187)))))
def dr(wm, c, which):
    per_q = defaultdict(list)
    for r in P.TRAIN:
        if which == "cutoff" and r["back"]: continue
        per_q[(r["condition"], r["episode"], r["qid"])].append(stack.brier(stack.as_probs(r, win(r, wm, c)), r["outcome"]) - stack.brier(stack.as_probs(r, win(r, 0.7, 1.0)), r["outcome"]))
    dd = defaultdict(lambda: defaultdict(list))
    for (cc, e, _), v in per_q.items(): dd[cc][e].append(sum(v) / len(v))
    def score(sel=None):
        return sum(sum(sum(dd[x][y]) / len(dd[x][y]) for y in (dd[x] if sel is None else sel[x])) / len(dd[x] if sel is None else sel[x]) / stack.BASE_ERR[x] for x in dd) / len(dd)
    rng = random.Random(0); b = sorted(score({x: [rng.choice(list(e)) for _ in e] for x, e in dd.items()}) for _ in range(500))
    return score(), b[12], b[487]
for wm in (0.55, 0.6, 0.65, 0.7, 0.75):
    for c in (0.75, 1.0, 1.25):
        preds = {r["qid"]: stack.as_probs(r, win(r, wm, c)) for r in P.VAL}
        s = [stack.subset_score(preds, stack.vt, stack.vcond, stack.ref, e, 60.0)["technical_points"] for e in (stack.ALL_EP, *stack.HALVES)]
        cu, wi = dr(wm, c, "cutoff"), dr(wm, c, "window")
        both = s[1] >= 56.53 and s[2] >= 56.61 and wi[2] < 0
        print(f"mixer {wm:.2f} / HMM {1-wm:.2f}, peers {c:4}: validation {s[0]:.2f} ({s[1]:.2f} / {s[2]:.2f}) | train vs v10 cutoff {cu[0]:+.4f} window {wi[0]:+.4f} [{wi[1]:+.4f},{wi[2]:+.4f}]{'  <-- both' if both else ''}", flush=True)
