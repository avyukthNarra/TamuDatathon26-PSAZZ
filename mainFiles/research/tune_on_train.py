"""Tune the regime model on TRAIN labels (validation stays untouched for one final check).

  python research/tune_on_train.py '[{"hazard": 0.001, "temper": 0.5}, ...]'

Uses the cached out-of-fold train rows (cutoff + last 10 observed ticks). For each config it
recomputes the regime model's win chance on every row (history strictly before the row's tick,
win prior from validation targets, lag pairs from train), then reports dR vs the 56.51 model of
  regime alone, and the blend 0.7 x (56.51 model) + 0.3 x regime,
at the cutoff and over the window. Negative = better.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.argv, CONFIGS = sys.argv[:1] + ["--methods", "none"], json.loads(sys.argv[1])
import stack  # noqa: E402  (loads caches, features and metric helpers)
from slot_forecasting.changepoint import RegimeModel  # noqa: E402
from slot_forecasting.io import load_inputs  # noqa: E402
from slot_forecasting.pair_mixer import choose_pairs, win_prior_by_type_from_targets  # noqa: E402

tq, th = load_inputs(stack.pub / "train")
vq, _ = load_inputs(stack.pub / "validation")
PAIRS = choose_pairs(tq, th)
PRIOR = win_prior_by_type_from_targets(stack.vt, vq)
SEQ = stack.HIST["train"]


def regime_wins(cfg):
    model = RegimeModel(PAIRS, PRIOR, **cfg)
    out = []
    for r in stack.TRAIN:
        key = f"{r['type']}|{r['signal']}"
        if (key, "all") not in model.priors:
            key = r["type"]
        seq = {t: v for t, v in SEQ[(r["episode"], r["machine"])].items() if t < r["cutoff"]}
        out.append(model.win_probability(seq, r["cutoff"], key))
    return out


def dr(pred_fn, which):
    per_q = defaultdict(list)
    for i, r in enumerate(stack.TRAIN):
        if which == "cutoff" and r["back"]:
            continue
        base = 0.7 * r["wins"][0] + 0.3 * r["wins"][1]
        d = stack.brier(stack.as_probs(r, pred_fn(i, r)), r["outcome"]) - stack.brier(stack.as_probs(r, base), r["outcome"])
        per_q[(r["condition"], r["episode"], r["qid"])].append(d)
    diff = defaultdict(lambda: defaultdict(list))
    for (c, e, _), v in per_q.items():
        diff[c][e].append(sum(v) / len(v))
    per_c = {c: sum(sum(x) / len(x) for x in eps.values()) / len(eps) / stack.BASE_ERR[c] for c, eps in diff.items()}
    return sum(per_c.values()) / len(per_c), per_c


for cfg in CONFIGS:
    g = regime_wins(cfg)
    alone_c, _ = dr(lambda i, r: g[i], "cutoff")
    alone_w, _ = dr(lambda i, r: g[i], "window")
    blend = lambda i, r: 0.7 * (0.7 * r["wins"][0] + 0.3 * r["wins"][1]) + 0.3 * g[i]  # noqa: E731
    blend_c, _ = dr(blend, "cutoff")
    blend_w, per = dr(blend, "window")
    print(f"{json.dumps(cfg):48s} regime alone: cutoff {alone_c:+.4f} window {alone_w:+.4f} | "
          f"0.3 blend: cutoff {blend_c:+.4f} window {blend_w:+.4f} {({k[:5]: round(v, 4) for k, v in per.items()})}", flush=True)
