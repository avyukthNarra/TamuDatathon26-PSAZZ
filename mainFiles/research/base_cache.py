"""Cache honest base-model forecasts for stacking research (run from mainFiles/).

  python research/base_cache.py --out runs/research-cache

Writes <out>/{validation,train,test}.json: for each query (and, for train, each of the cutoff
and the 10 observed ticks before it) the pair mixer's 5 probabilities, the hidden-rule HMM's and
the regime model's win chances, the outcome / true probabilities where known, and its
(episode, machine, type, signal, cutoff).

  validation  priors from the OTHER validation half (cross-fitted), mixer fit on all of train
  train       8 episode folds; mixer pairs / cell mode refit without the fold; priors from all
              validation targets (train labels never reach the priors)
  test        priors from all validation targets (what the submission would use)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slot_forecasting.changepoint import RegimeModel  # noqa: E402
from slot_forecasting.experiments import episode_folds  # noqa: E402
from slot_forecasting.io import load_inputs, read_labels, read_targets  # noqa: E402
from slot_forecasting.pair_mixer import (CLASS_INDEX, adapt, choose_pairs, episode_kfolds,  # noqa: E402
                                         size_modes_from_targets, split_episodes,
                                         win_prior_by_type_from_targets)
from slot_forecasting.rule_hmm import HiddenRuleHMM, levels_from_prior  # noqa: E402
from slot_forecasting.schema import Query  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="public")
parser.add_argument("--out", default="runs/research-cache")
parser.add_argument("--window", type=int, default=10)
args = parser.parse_args()
pub, out = Path(args.data), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
quiet = lambda *_: None  # noqa: E731

tq, th = load_inputs(pub / "train")
labels = read_labels(pub / "train/labels.csv", tq)
tcond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(pub / "train/conditions.csv"))}
vq, vh = load_inputs(pub / "validation")
vt = read_targets(pub / "validation/targets.csv")
vcond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(pub / "validation/conditions.csv"))}
truth = {t.query_id: list(t.probabilities) for t in vt}


def members(pairs, prior):
    return HiddenRuleHMM(pairs, levels_from_prior(prior)), RegimeModel(pairs, prior, temper=0.5)


def rows(queries, history, mixer, hmm, regime, extra):
    m = mixer.predict(queries, history)
    h = hmm.win_chances(queries, history)
    g = regime.win_chances(queries, history)
    return [dict(qid=q.query_id, episode=q.episode_id, machine=q.machine_id, type=q.machine_type,
                 signal=q.signal, cutoff=q.cutoff, mixer=list(m[q.query_id]), hmm=h[q.query_id],
                 regime=g[q.query_id], **extra(q)) for q in queries]


def subset(history, queries):
    keys = {(q.episode_id, q.machine_id) for q in queries}
    return tuple(p for p in history if (p.episode_id, p.machine_id) in keys)


# ---- validation: cross-fitted priors ----
val_rows = []
for scored, other in zip(episode_folds(vt, vcond), reversed(episode_folds(vt, vcond))):
    prior_targets = [t for t in vt if t.episode_id in other]
    prior = win_prior_by_type_from_targets(prior_targets, vq)
    mixer = adapt(tq, th, labels, log=quiet, size_modes=size_modes_from_targets(prior_targets, vq),
                  win_prior=prior, hmm_weight=0)
    qs = tuple(q for q in vq if q.episode_id in scored)
    val_rows += rows(qs, subset(vh, qs), mixer, *members(mixer.pairs, prior),
                     lambda q: dict(condition=vcond[q.query_id], truth=truth[q.query_id]))
json.dump(val_rows, open(out / "validation.json", "w"))
print(f"validation: {len(val_rows)} rows", flush=True)

# ---- test: full priors ----
prior = win_prior_by_type_from_targets(vt, vq)
modes = size_modes_from_targets(vt, vq)
mixer = adapt(tq, th, labels, log=quiet, size_modes=modes, win_prior=prior, hmm_weight=0)
sq, sh = load_inputs(pub / "test")
json.dump(rows(sq, sh, mixer, *members(mixer.pairs, prior), lambda q: {}), open(out / "test.json", "w"))
print("test done", flush=True)

# ---- train: out of fold, cutoff + window ----
train_rows = []
folds = episode_kfolds(tq, tcond)
for fold in sorted(set(folds.values())):
    held = {e for e, f in folds.items() if f == fold}
    (hq, hh), (fq, fh) = split_episodes(tq, th, held)
    mixer = adapt(fq, fh, {q.query_id: labels[q.query_id] for q in fq}, log=quiet, size_modes=modes,
                  win_prior=prior, hmm_weight=0)
    hmm, regime = members(choose_pairs(fq, fh), prior)
    own = {(p.episode_id, p.machine_id, p.tick): p.payout for p in hh}
    for back in range(args.window, -1, -1):
        qs, outcome = [], {}
        for q in hq:
            t = q.cutoff - back
            value = labels[q.query_id] if back == 0 else own.get((q.episode_id, q.machine_id, t))
            if t > 0 and value is not None:
                qs.append(Query(q.query_id, q.episode_id, q.machine_id, q.machine_type, q.signal, t))
                outcome[q.query_id] = CLASS_INDEX[value]
        qs = tuple(qs)
        limit = {(q.episode_id, q.machine_id): q.cutoff for q in qs}
        hist = tuple(p for p in hh if (p.episode_id, p.machine_id) in limit and p.tick < limit[(p.episode_id, p.machine_id)])
        train_rows += rows(qs, hist, mixer, hmm, regime,
                           lambda q: dict(condition=tcond[q.query_id], back=q.cutoff and 0, outcome=outcome[q.query_id]))
        for r in train_rows[-len(qs):]:
            r["back"] = back
    print(f"train fold {fold} done", flush=True)
json.dump(train_rows, open(out / "train.json", "w"))
print(f"train: {len(train_rows)} rows", flush=True)
