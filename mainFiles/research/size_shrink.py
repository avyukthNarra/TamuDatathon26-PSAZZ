"""Payout-size research: shrink the mixer's size forecast toward the type's overall size mix.

  python research/size_shrink.py

size = (1 - lam) * mixer size + lam * type size mix, with lam = A / (A + wins in the last 40
observed spins) (more shrinkage when the machine has few recent wins). The type mix is counted
from TRAIN history only. Win chance = the 56.51 blend throughout. Scored on cross-fitted
validation (halves) and out-of-fold train labels (dR vs the 56.51 model; cutoff and window).
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.argv = sys.argv[:1] + ["--methods", "none"]
import stack  # noqa: E402
from slot_forecasting.io import load_inputs  # noqa: E402
from slot_forecasting.pair_mixer import CLASS_INDEX  # noqa: E402

counts = defaultdict(lambda: [0.0] * 4)
tq, th = load_inputs(stack.pub / "train")
kind = {(q.episode_id, q.machine_id): q.machine_type for q in tq}
for p in th:
    if p.payout > 0:
        counts[kind[(p.episode_id, p.machine_id)]][CLASS_INDEX[p.payout] - 1] += 1
TYPE_MIX = {k: [c / sum(v) for c in v] for k, v in counts.items()}


def recent_wins(r, split):
    seq = stack.HIST[split][(r["episode"], r["machine"])]
    ticks = sorted(t for t in seq if t < r["cutoff"])[-40:]
    return sum(1 for t in ticks if seq[t] > 0)


for split, rows in (("validation", stack.VAL), ("train", stack.TRAIN)):
    for r in rows:
        r["wins40"] = recent_wins(r, split)


def probs(r, A, short_only):
    pw = 0.7 * r["wins"][0] + 0.3 * r["wins"][1]
    m = r["mixer"]
    size = [x / (1 - m[0]) for x in m[1:]]
    lam = 0.0 if (A == 0 or (short_only and not r["short"])) else A / (A + r["wins40"])
    size = [(1 - lam) * s + lam * t for s, t in zip(size, TYPE_MIX[r["type"]])]
    raw = [max(x, 1e-4) for x in [1 - pw] + [pw * s for s in size]]
    total = sum(raw)
    return [x / total for x in raw]


def evaluate(A, short_only):
    preds = {r["qid"]: probs(r, A, short_only) for r in stack.VAL}
    cf = [stack.subset_score(preds, stack.vt, stack.vcond, stack.ref, e, 60.0)["technical_points"]
          for e in (stack.ALL_EP, *stack.HALVES)]
    res = {}
    for which in ("cutoff", "window"):
        per_q = defaultdict(list)
        for r in stack.TRAIN:
            if which == "cutoff" and r["back"]:
                continue
            per_q[(r["condition"], r["episode"], r["qid"])].append(
                stack.brier(probs(r, A, short_only), r["outcome"]) - stack.brier(probs(r, 0.0, False), r["outcome"]))
        diff = defaultdict(lambda: defaultdict(list))
        for (c, e, _), v in per_q.items():
            diff[c][e].append(sum(v) / len(v))
        res[which] = sum(sum(sum(x) / len(x) for x in eps.values()) / len(eps) / stack.BASE_ERR[c]
                         for c, eps in diff.items()) / len(diff)
    print(f"A={A:<5} short_only={short_only!s:5}  cross-fit {cf[0]:.2f} ({cf[1]:.2f} / {cf[2]:.2f}) | "
          f"train dR cutoff {res['cutoff']:+.4f} window {res['window']:+.4f}", flush=True)


for A, short_only in ((0.0, False), (1.0, False), (2.0, False), (4.0, False), (2.0, True), (4.0, True), (8.0, True)):
    evaluate(A, short_only)
