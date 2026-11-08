import sys, random
from collections import defaultdict
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
sys.argv = sys.argv[:1] + ["--methods", "none"]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import peer_correction as P
stack = P.stack
names = ["peers 1-5"]
def train_dr(b, which):
    per_q = defaultdict(list)
    for r in P.TRAIN:
        if which == "cutoff" and r["back"]: continue
        per_q[(r["condition"], r["episode"], r["qid"])].append(
            stack.brier(stack.as_probs(r, P.corrected(r, names, [b])), r["outcome"]) - stack.brier(stack.as_probs(r, r["f"]), r["outcome"]))
    dd = defaultdict(lambda: defaultdict(list))
    for (c, e, _), v in per_q.items(): dd[c][e].append(sum(v) / len(v))
    def score(sel=None):
        return sum(sum(sum(dd[c][x]) / len(dd[c][x]) for x in (dd[c] if sel is None else sel[c])) / len(dd[c] if sel is None else sel[c]) / stack.BASE_ERR[c] for c in dd) / len(dd)
    rng = random.Random(0); boot = sorted(score({c: [rng.choice(list(e)) for _ in e] for c, e in dd.items()}) for _ in range(1000))
    return score(), boot[25], boot[974]
for b in (0.5, 0.75, 1.0, 1.25):
    s = P.val_score(names, [b]); cu, wi = train_dr(b, "cutoff"), train_dr(b, "window")
    print(f"coef {b:4}: validation {s[0]['technical_points']:.2f} ({s[1]['technical_points']:.2f} / {s[2]['technical_points']:.2f}) | train cutoff {cu[0]:+.4f} [{cu[1]:+.4f},{cu[2]:+.4f}] window {wi[0]:+.4f} [{wi[1]:+.4f},{wi[2]:+.4f}]", flush=True)
