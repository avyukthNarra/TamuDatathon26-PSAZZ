"""Reverse-direction stacking: learn on TRAIN outcomes, score on VALIDATION truth (run from mainFiles/).

  python research/stack_on_train.py

The stacker is fitted on the out-of-fold train rows (cutoff + last 10 observed ticks), with the
observed win (0/1) as target and the metric's condition weights; it never sees validation. Its
validation score (all 96 episodes and both halves) is therefore an untouched check. Train-side
skill is estimated with 4 episode folds of the stacker itself (fit on 3/4 of train episodes,
score the rest).
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.argv = sys.argv[:1] + ["--methods", "none"]
import stack  # noqa: E402

for r in stack.TRAIN:
    r["truth"] = [0.0 if r["outcome"] else 1.0]  # p_0 "truth" = observed loss, so 1 - truth[0] = won


def val_score(predict):
    preds = {r["qid"]: stack.as_probs(r, predict(r)) for r in stack.VAL}
    return [stack.subset_score(preds, stack.vt, stack.vcond, stack.ref, e, 60.0)["technical_points"]
            for e in (stack.ALL_EP, *stack.HALVES)]


def train_cv(method, base):
    episodes = sorted({r["episode"] for r in stack.TRAIN})
    fold = {e: i % 4 for i, e in enumerate(episodes)}
    diff = defaultdict(lambda: defaultdict(list))
    for k in range(4):
        model = method([r for r in stack.TRAIN if fold[r["episode"]] != k])
        per_q = defaultdict(list)
        for r in stack.TRAIN:
            if fold[r["episode"]] == k:
                per_q[(r["condition"], r["episode"], r["qid"])].append(
                    stack.brier(stack.as_probs(r, model(r)), r["outcome"]) - stack.brier(stack.as_probs(r, base(r)), r["outcome"]))
        for (c, e, _), v in per_q.items():
            diff[c][e].append(sum(v) / len(v))
    return sum(sum(sum(x) / len(x) for x in eps.values()) / len(eps) / stack.BASE_ERR[c]
               for c, eps in diff.items()) / len(diff)


base = stack.fixed((0.7, 0.3, 0.0))([])
if "--two" in sys.argv[0:] or True:
    # mixer + HMM only: the regime model's settings were chosen on validation, so drop it here
    for r in stack.TRAIN + stack.VAL:
        r["wins"] = (r["wins"][0], r["wins"][1], r["wins"][1])
for name, method in (("56.51 model", stack.fixed((0.7, 0.3, 0.0))),
                     ("logistic stack, no interactions (mixer+HMM)", stack.logistic(interact=False)),
                     ("logistic stack + interactions, L2 1e-2 (mixer+HMM)", stack.logistic(l2=1e-2)),
                     ("logistic stack + interactions, L2 1e-1 (mixer+HMM)", stack.logistic(l2=1e-1))):
    vs = val_score(method(stack.TRAIN))
    cv = train_cv(method, base)
    print(f"{name:52s} learned on train -> validation {vs[0]:.2f} ({vs[1]:.2f} / {vs[2]:.2f}) | train 4-fold window dR {cv:+.4f}", flush=True)
