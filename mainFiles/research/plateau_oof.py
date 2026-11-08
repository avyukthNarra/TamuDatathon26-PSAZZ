"""Plateau average with the per-lag peer weights refit WITHOUT each held-out train fold (honest)."""
import math, random, sys
from collections import defaultdict
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
sys.argv = sys.argv[:1] + ["--methods", "none"]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import plateau_avg as PA
P, stack, lg = PA.P, PA.stack, PA.lg
LAGS = [f"lag{k}" for k in range(1, 9)]
eps = sorted({r["episode"] for r in P.TRAIN}); fold = {e: i % 4 for i, e in enumerate(eps)}
W = {k: [x * 0.55 for x in P.fit([r for r in P.TRAIN if fold[r["episode"]] != k], LAGS, l2=1e-2)] for k in range(4)}
def lagp(r, w):
    return sum(wi * ((r["x"][n] if r["x"][n] is not None else P.MEANS[n]) - P.MEANS[n]) for n, wi in zip(LAGS, w))
def flat(r):
    sh = r["x"]["peers 1-5"]; return 0.0 if sh is None else sh - 0.4187
def z(r, w, with_regime):
    a = 0.7 * lg(1 - r["mixer"][0]) + 0.3 * lg(r["hmm"]) + flat(r)
    b = 0.6 * lg(1 - r["mixer"][0]) + 0.4 * lg(r["hmm"]) + flat(r)
    c = 0.7 * lg(1 - r["mixer"][0]) + 0.3 * lg(r["hmm"]) + lagp(r, w)
    parts = [a, b, c] + ([lg((1 - r["mixer"][0] + r["hmm"] + r["regime"]) / 3) + flat(r)] if with_regime else [])
    return sum(parts) / len(parts)
sig = lambda x: 1 / (1 + math.exp(-x))
for with_regime in (False, True):
    out = {}
    for which in ("cutoff", "window"):
        per_q = defaultdict(list)
        for r in P.TRAIN:
            if which == "cutoff" and r["back"]: continue
            base = sig(0.7 * lg(1 - r["mixer"][0]) + 0.3 * lg(r["hmm"]) + flat(r))
            per_q[(r["condition"], r["episode"], r["qid"])].append(stack.brier(stack.as_probs(r, sig(z(r, W[fold[r["episode"]]], with_regime))), r["outcome"]) - stack.brier(stack.as_probs(r, base), r["outcome"]))
        dd = defaultdict(lambda: defaultdict(list))
        for (c, e, _), v in per_q.items(): dd[c][e].append(sum(v) / len(v))
        def score(sel=None):
            return sum(sum(sum(dd[c][x]) / len(dd[c][x]) for x in (dd[c] if sel is None else sel[c])) / len(dd[c] if sel is None else sel[c]) / stack.BASE_ERR[c] for c in dd) / len(dd)
        rng = random.Random(0); b = sorted(score({c: [rng.choice(list(e)) for _ in e] for c, e in dd.items()}) for _ in range(1000))
        out[which] = (score(), b[25], b[974])
    full_w = [x * 0.55 for x in P.fit(P.TRAIN, LAGS, l2=1e-2)]
    preds = {r["qid"]: stack.as_probs(r, sig(z(r, full_w, with_regime))) for r in P.VAL}
    s = [stack.subset_score(preds, stack.vt, stack.vcond, stack.ref, e, 60.0)["technical_points"] for e in (stack.ALL_EP, *stack.HALVES)]
    print(f"{'with' if with_regime else 'without'} regime: validation {s[0]:.2f} ({s[1]:.2f} / {s[2]:.2f}) | HONEST train vs v10: cutoff {out['cutoff'][0]:+.4f} [{out['cutoff'][1]:+.4f},{out['cutoff'][2]:+.4f}] window {out['window'][0]:+.4f} [{out['window'][1]:+.4f},{out['window'][2]:+.4f}]", flush=True)
