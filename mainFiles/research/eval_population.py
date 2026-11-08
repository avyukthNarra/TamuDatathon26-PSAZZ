"""Honest check of population priors (run from mainFiles/):

  python research/eval_population.py '[[win_strength, size_strength], ...]'

Win chance = 0.7 mixer + 0.3 HMM pooled in log-odds (the 56.54 model); only the mixer changes.
  validation: priors from the other half; population learned on train (validation never used)
  train:      8 episode folds; mixer AND population learned without the fold; cutoff + 10-tick
              window; dR vs strength (0, 0) (= the 56.54 model). Negative = better.
"""
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slot_forecasting.experiments import episode_folds, subset_score  # noqa: E402
from slot_forecasting.io import load_inputs, read_labels, read_submission, read_targets  # noqa: E402
from slot_forecasting.pair_mixer import (CLASS_INDEX, adapt, episode_kfolds, size_modes_from_targets,  # noqa: E402
                                         split_episodes, win_prior_by_type_from_targets)
from slot_forecasting.population import population_mixer  # noqa: E402
from slot_forecasting.rule_hmm import HiddenRuleHMM, levels_from_prior  # noqa: E402
from slot_forecasting.schema import Query  # noqa: E402

CONFIGS = [tuple(c) for c in json.loads(sys.argv[1])]
ALL = [(0.0, 0.0)] + [c for c in CONFIGS if c != (0.0, 0.0)]
pub = Path("public")
quiet = lambda *_: None  # noqa: E731
tq, th = load_inputs(pub / "train"); labels = read_labels(pub / "train/labels.csv", tq)
tcond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(pub / "train/conditions.csv"))}
vq, vh = load_inputs(pub / "validation"); vt = read_targets(pub / "validation/targets.csv")
ref = read_submission(pub / "validation/reference.csv")
vcond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(pub / "validation/conditions.csv"))}
HALVES = episode_folds(vt, vcond); ALL_EP = {t.episode_id for t in vt}
_be = defaultdict(lambda: defaultdict(list))
for t in vt:
    _be[vcond[t.query_id]][t.episode_id].append(sum((a - b) ** 2 for a, b in zip(ref[t.query_id], t.probabilities)))
BASE_ERR = {c: sum(sum(v) / len(v) for v in e.values()) / len(e) for c, e in _be.items()}


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def blend(mix, hmm_win):
    ow = 1 - mix[0]
    pw = 1 / (1 + math.exp(-(0.7 * logit(ow) + 0.3 * logit(hmm_win))))
    raw = [max(x, 1e-4) for x in [1 - pw] + [pw * x / ow for x in mix[1:]]]
    s = sum(raw)
    return [x / s for x in raw]


def sub(history, qs):
    keys = {(q.episode_id, q.machine_id) for q in qs}
    return tuple(p for p in history if (p.episode_id, p.machine_id) in keys)


# ---- validation ----
val = {c: {} for c in ALL}
for scored, other in zip(HALVES, reversed(HALVES)):
    pt = [t for t in vt if t.episode_id in other]
    prior = win_prior_by_type_from_targets(pt, vq)
    base = adapt(tq, th, labels, log=quiet, size_modes=size_modes_from_targets(pt, vq), win_prior=prior, hmm_weight=0)
    qs = tuple(q for q in vq if q.episode_id in scored); hist = sub(vh, qs)
    hmm = HiddenRuleHMM(base.pairs, levels_from_prior(prior)).win_chances(qs, hist)
    for c in ALL:
        mix = population_mixer(base, tq, th, *c).predict(qs, hist)
        val[c].update({k: blend(mix[k], hmm[k]) for k in mix})
print("validation done", flush=True)

# ---- train out of fold ----
prior = win_prior_by_type_from_targets(vt, vq); modes = size_modes_from_targets(vt, vq)
err = {c: defaultdict(list) for c in ALL}
folds = episode_kfolds(tq, tcond)
for fold in sorted(set(folds.values())):
    held = {e for e, f in folds.items() if f == fold}
    (hq, hh), (fq, fh) = split_episodes(tq, th, held)
    base = adapt(fq, fh, {q.query_id: labels[q.query_id] for q in fq}, log=quiet, size_modes=modes, win_prior=prior, hmm_weight=0)
    models = {c: population_mixer(base, fq, fh, *c) for c in ALL}
    hmm_model = HiddenRuleHMM(base.pairs, levels_from_prior(prior))
    own = {(p.episode_id, p.machine_id, p.tick): p.payout for p in hh}
    for back in range(10, -1, -1):
        qs, out = [], {}
        for q in hq:
            t = q.cutoff - back
            v = labels[q.query_id] if back == 0 else own.get((q.episode_id, q.machine_id, t))
            if t > 0 and v is not None:
                qs.append(Query(q.query_id, q.episode_id, q.machine_id, q.machine_type, q.signal, t)); out[q.query_id] = CLASS_INDEX[v]
        qs = tuple(qs); lim = {(q.episode_id, q.machine_id): q.cutoff for q in qs}
        hist = tuple(p for p in hh if (p.episode_id, p.machine_id) in lim and p.tick < lim[(p.episode_id, p.machine_id)])
        hw = hmm_model.win_chances(qs, hist)
        for c, m in models.items():
            mix = m.predict(qs, hist)
            for q in qs:
                p = blend(mix[q.query_id], hw[q.query_id])
                err[c][(q.query_id, back)] = sum((x - (1.0 if i == out[q.query_id] else 0.0)) ** 2 for i, x in enumerate(p))
    print(f"train fold {fold} done", flush=True)

ep = {q.query_id: q.episode_id for q in tq}


def dr(c, which):
    per_q = defaultdict(list)
    for (qid, back), e in err[c].items():
        if which == "cutoff" and back:
            continue
        per_q[qid].append(e - err[(0.0, 0.0)][(qid, back)])
    diff = defaultdict(lambda: defaultdict(list))
    for qid, v in per_q.items():
        diff[tcond[qid]][ep[qid]].append(sum(v) / len(v))

    def score(sel=None):
        vals = []
        for cnd, eps in diff.items():
            ch = eps if sel is None else sel[cnd]
            vals.append(sum(sum(eps[x]) / len(eps[x]) for x in ch) / len(ch) / BASE_ERR[cnd])
        return sum(vals) / len(vals)
    rng = random.Random(0)
    boot = sorted(score({cnd: [rng.choice(list(e)) for _ in e] for cnd, e in diff.items()}) for _ in range(1000))
    return score(), boot[25], boot[974]


for c in ALL:
    s = [subset_score(val[c], vt, vcond, ref, e, 60.0) for e in (ALL_EP, *HALVES)]
    sh = s[0]["by_condition"]["short_history"]["loss_ratio"]
    cu, wi = dr(c, "cutoff"), dr(c, "window")
    print(f"win {c[0]:<4} size {c[1]:<4}: validation {s[0]['technical_points']:.2f} ({s[1]['technical_points']:.2f} / "
          f"{s[2]['technical_points']:.2f}) short {sh:.4f} | train cutoff {cu[0]:+.4f} [{cu[1]:+.4f},{cu[2]:+.4f}] "
          f"window {wi[0]:+.4f} [{wi[1]:+.4f},{wi[2]:+.4f}]", flush=True)
