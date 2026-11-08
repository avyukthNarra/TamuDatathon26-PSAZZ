"""Recent-surprise tempering (run from mainFiles/).

S = mean over the machine's last W observed ticks of (won_t - p_t)^2 - p_t (1 - p_t), with p_t the
pair mixer's own forecast at that tick (made before the tick): ~0 when the model is calibrated,
> 0 when it has recently been surprised (a changing machine). Correction on the v10 forecast:
    logit p = logit p_v10 / (1 + g * max(S, 0))       (pull toward 0.5 when surprised)
Validation (truth) and train (outcomes) checked separately; g is chosen on train.
"""
import math, random, sys
from collections import defaultdict
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
sys.argv = sys.argv[:1] + ["--methods", "none"]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import residuals2 as R2
stack = R2.P.stack
from slot_forecasting.io import load_inputs, read_labels, read_targets
from slot_forecasting.pair_mixer import CLASS_INDEX, adapt, size_modes_from_targets, win_prior_by_type_from_targets
tq, th = load_inputs("public/train"); vq, vh = load_inputs("public/validation"); vt = read_targets("public/validation/targets.csv")
mixer = adapt(tq, th, read_labels("public/train/labels.csv", tq), log=lambda *_: None,
              size_modes=size_modes_from_targets(vt, vq), win_prior=win_prior_by_type_from_targets(vt, vq), hmm_weight=0)
kind = {(q.episode_id, q.machine_id): (q.machine_type, q.signal, q.cutoff) for q in tq + vq}
EMIT = {}
for split, rows in (("validation", R2.VAL), ("train", R2.TRAIN)):
    seqs = R2.P.R.HIST[split]
    for key in {(r["episode"], r["machine"]) for r in rows}:
        mt, sig, cut = kind[key]; s = seqs[key]
        own = [None] * (cut + 1)
        for t, v in s.items(): own[t] = CLASS_INDEX[v]
        em = {}
        mixer.walk(own, mt, cut, em.__setitem__, sig)
        EMIT[(split, key)] = {t: 1 - p[0] for t, p in em.items()}
def surprise(r, split, W):
    s = R2.P.R.HIST[split][(r["episode"], r["machine"])]; em = EMIT[(split, (r["episode"], r["machine"]))]
    ts = sorted(t for t in s if t < r["cutoff"])[-W:]
    if len(ts) < 5: return 0.0
    return sum(((s[t] > 0) - em[t]) ** 2 - em[t] * (1 - em[t]) for t in ts) / len(ts)
def lg(p): p = min(max(p, 1e-4), 1 - 1e-4); return math.log(p / (1 - p))
for W in (10, 20, 40):
    for split, rows in (("validation", R2.VAL), ("train", R2.TRAIN)):
        for r in rows: r[f"S{W}"] = surprise(r, split, W)
def win(r, g, W): return 1 / (1 + math.exp(-lg(r["f2"]) / (1 + g * max(r[f"S{W}"], 0.0))))
def train_dr(g, W, which):
    per_q = defaultdict(list)
    for r in R2.TRAIN:
        if which == "cutoff" and r["back"]: continue
        per_q[(r["condition"], r["episode"], r["qid"])].append(stack.brier(stack.as_probs(r, win(r, g, W)), r["outcome"]) - stack.brier(stack.as_probs(r, r["f2"]), r["outcome"]))
    dd = defaultdict(lambda: defaultdict(list))
    for (c, e, _), v in per_q.items(): dd[c][e].append(sum(v) / len(v))
    def score(sel=None):
        return sum(sum(sum(dd[c][x]) / len(dd[c][x]) for x in (dd[c] if sel is None else sel[c])) / len(dd[c] if sel is None else sel[c]) / stack.BASE_ERR[c] for c in dd) / len(dd)
    rng = random.Random(0); b = sorted(score({c: [rng.choice(list(e)) for _ in e] for c, e in dd.items()}) for _ in range(500))
    return score(), b[12], b[487]
for W in (10, 20, 40):
    for g in (1.0, 2.0, 4.0, 8.0):
        preds = {r["qid"]: stack.as_probs(r, win(r, g, W)) for r in R2.VAL}
        s = [stack.subset_score(preds, stack.vt, stack.vcond, stack.ref, e, 60.0) for e in (stack.ALL_EP, *stack.HALVES)]
        cu, wi = train_dr(g, W, "cutoff"), train_dr(g, W, "window")
        bc = s[0]["by_condition"]
        print(f"W {W:2d} g {g:3}: validation {s[0]['technical_points']:.2f} ({s[1]['technical_points']:.2f} / {s[2]['technical_points']:.2f}) drift {bc['drift']['loss_ratio']:.3f} shared {bc['shared_change']['loss_ratio']:.3f} | train cutoff {cu[0]:+.4f} [{cu[1]:+.4f},{cu[2]:+.4f}] window {wi[0]:+.4f} [{wi[1]:+.4f},{wi[2]:+.4f}]", flush=True)
