"""Average near-equivalent variants in log-odds (variance reduction), judged on both checks."""
import math, random, sys
from collections import defaultdict
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
sys.argv = sys.argv[:1] + ["--methods", "none"]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import peer_plus as PP
P = PP.P; stack = P.stack
def lg(p): p = min(max(p, 1e-4), 1 - 1e-4); return math.log(p / (1 - p))
LAGW = [x * 0.55 for x in P.fit(P.TRAIN, [f"lag{k}" for k in range(1, 9)], l2=1e-2)]
def peers(r, kind):
    if kind == "flat":
        sh = r["x"]["peers 1-5"]; return 0.0 if sh is None else (sh - 0.4187)
    return sum(w * ((r["x"][f"lag{k}"] if r["x"][f"lag{k}"] is not None else P.MEANS[f"lag{k}"]) - P.MEANS[f"lag{k}"]) for k, w in zip(range(1, 9), LAGW))
VARIANTS = {
    "v10":            lambda r: 0.7 * lg(1 - r["mixer"][0]) + 0.3 * lg(r["hmm"]) + peers(r, "flat"),
    "0.6/0.4 peers":  lambda r: 0.6 * lg(1 - r["mixer"][0]) + 0.4 * lg(r["hmm"]) + peers(r, "flat"),
    "per-lag peers":  lambda r: 0.7 * lg(1 - r["mixer"][0]) + 0.3 * lg(r["hmm"]) + peers(r, "lag"),
    "regime blend":   lambda r: (lambda w: lg(w))((1 - r["mixer"][0] + r["hmm"] + r["regime"]) / 3) + peers(r, "flat"),
}
def combo(names):
    return lambda r: 1 / (1 + math.exp(-sum(VARIANTS[n](r) for n in names) / len(names)))
def check(fn):
    preds = {r["qid"]: stack.as_probs(r, fn(r)) for r in P.VAL}
    s = [stack.subset_score(preds, stack.vt, stack.vcond, stack.ref, e, 60.0)["technical_points"] for e in (stack.ALL_EP, *stack.HALVES)]
    base = combo(["v10"]); out = {}
    for which in ("cutoff", "window"):
        per_q = defaultdict(list)
        for r in P.TRAIN:
            if which == "cutoff" and r["back"]: continue
            per_q[(r["condition"], r["episode"], r["qid"])].append(stack.brier(stack.as_probs(r, fn(r)), r["outcome"]) - stack.brier(stack.as_probs(r, base(r)), r["outcome"]))
        dd = defaultdict(lambda: defaultdict(list))
        for (c, e, _), v in per_q.items(): dd[c][e].append(sum(v) / len(v))
        def score(sel=None):
            return sum(sum(sum(dd[c][x]) / len(dd[c][x]) for x in (dd[c] if sel is None else sel[c])) / len(dd[c] if sel is None else sel[c]) / stack.BASE_ERR[c] for c in dd) / len(dd)
        rng = random.Random(0); b = sorted(score({c: [rng.choice(list(e)) for _ in e] for c, e in dd.items()}) for _ in range(500))
        out[which] = (score(), b[12], b[487])
    return s, out
for names in (["v10"], ["v10", "0.6/0.4 peers"], ["v10", "per-lag peers"], ["v10", "0.6/0.4 peers", "per-lag peers"],
              ["v10", "regime blend"], ["v10", "0.6/0.4 peers", "per-lag peers", "regime blend"]):
    s, o = check(combo(names))
    print(f"{' + '.join(names):55s} validation {s[0]:.2f} ({s[1]:.2f} / {s[2]:.2f}) | train vs v10 cutoff {o['cutoff'][0]:+.4f} [{o['cutoff'][1]:+.4f},{o['cutoff'][2]:+.4f}] window {o['window'][0]:+.4f} [{o['window'][1]:+.4f},{o['window'][2]:+.4f}]", flush=True)
