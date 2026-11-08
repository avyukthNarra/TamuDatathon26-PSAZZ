"""v11 win chance with zhill's size-rule HMM sizes vs the mixer's sizes, on both checks."""
import math, random, sys
from collections import defaultdict
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
sys.argv = sys.argv[:1] + ["--methods", "none"]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import plateau_avg as PA
P, stack, lg = PA.P, PA.stack, PA.lg
from slot_forecasting.io import load_inputs, read_targets
from slot_forecasting.size_rule_hmm import SizeRuleHMM, core_from_train, default_rules, modes_from_targets
from slot_forecasting.schema import Query
tq, th = load_inputs("public/train"); vq, _ = load_inputs("public/validation"); vt = read_targets("public/validation/targets.csv")
LAGS = [f"lag{k}" for k in range(1, 9)]
W = [x * 0.55 for x in P.fit(P.TRAIN, LAGS, l2=1e-2)]
def v11(r):
    flat = 0.0 if r["x"]["peers 1-5"] is None else r["x"]["peers 1-5"] - 0.4187
    lagp = sum(w * ((r["x"][n] if r["x"][n] is not None else P.MEANS[n]) - P.MEANS[n]) for n, w in zip(LAGS, W))
    lm, lh = lg(1 - r["mixer"][0]), lg(r["hmm"])
    return 1 / (1 + math.exp(-((0.7 * lm + 0.3 * lh + flat) + (0.6 * lm + 0.4 * lh + flat) + (0.7 * lm + 0.3 * lh + lagp)) / 3))
def size_splits(rows, split, modes, core):
    model = SizeRuleHMM(modes, prior=default_rules(core_weight=0.97, core=core), switch=0.0)
    out = {}
    for r in rows:
        if r["type"] not in modes: continue
        A, B = modes[r["type"]]
        seq = {t: v for t, v in P.R.HIST[split][(r["episode"], r["machine"])].items() if t < r["cutoff"]}
        pb = model.big_probability(seq, r["cutoff"], A, B)
        out[id(r)] = [(1 - pb) * a + pb * b for a, b in zip(A, B)]
    return out
def probs(r, win, size):
    if size is None:
        return stack.as_probs(r, win)
    raw = [max(x, 1e-4) for x in [1 - win] + [win * s for s in size]]
    t = sum(raw); return [x / t for x in raw]
SV = {}
for scored, other in zip(stack.HALVES, reversed(stack.HALVES)):
    oth = [t for t in vt if t.episode_id in other]
    modes = modes_from_targets(oth, vq)
    core = core_from_train(tq, th, modes)
    SV.update(size_splits([r for r in P.VAL if r["episode"] in scored], "validation", modes, core))
modes = modes_from_targets(vt, vq); core = core_from_train(tq, th, modes)
print("modes for types:", sorted(modes), "| core rules from train:", core)
ST = size_splits(P.TRAIN, "train", modes, core)
for label, use in (("v11 (mixer sizes)", False), ("v11 win + zhill size HMM", True)):
    preds = {r["qid"]: probs(r, v11(r), SV.get(id(r)) if use else None) for r in P.VAL}
    s = [stack.subset_score(preds, stack.vt, stack.vcond, stack.ref, e, 60.0) for e in (stack.ALL_EP, *stack.HALVES)]
    out = {}
    for which in ("cutoff", "window"):
        per_q = defaultdict(list)
        for r in P.TRAIN:
            if which == "cutoff" and r["back"]: continue
            per_q[(r["condition"], r["episode"], r["qid"])].append(stack.brier(probs(r, v11(r), ST.get(id(r)) if use else None), r["outcome"]) - stack.brier(probs(r, v11(r), None), r["outcome"]))
        dd = defaultdict(lambda: defaultdict(list))
        for (c, e, _), v in per_q.items(): dd[c][e].append(sum(v) / len(v))
        def score(sel=None):
            return sum(sum(sum(dd[c][x]) / len(dd[c][x]) for x in (dd[c] if sel is None else sel[c])) / len(dd[c] if sel is None else sel[c]) / stack.BASE_ERR[c] for c in dd) / len(dd)
        rng = random.Random(0); b = sorted(score({c: [rng.choice(list(e)) for _ in e] for c, e in dd.items()}) for _ in range(1000))
        out[which] = (score(), b[25], b[974])
    bc = s[0]["by_condition"]
    print(f"{label:28s} cross-fit validation {s[0]['technical_points']:.2f} ({s[1]['technical_points']:.2f} / {s[2]['technical_points']:.2f}) " + " ".join(f"{c[:5]} {bc[c]['loss_ratio']:.3f}" for c in sorted(bc)) + f" | train vs v11 cutoff {out['cutoff'][0]:+.4f} [{out['cutoff'][1]:+.4f},{out['cutoff'][2]:+.4f}] window {out['window'][0]:+.4f} [{out['window'][1]:+.4f},{out['window'][2]:+.4f}]", flush=True)
