"""Compare PairMixer settings honestly on train cutoff labels, balanced by condition.

Usage (from mainFiles/):
  python tune_pair_mixer.py '[{"alpha": 0.01}, {"decay_missing": false}]'  [--val]

Each candidate is scored against the current defaults, paired query by query, on
train/labels.csv outcomes AT THE CUTOFF TICK (history ticks rank settings
differently; see AGENTS.md section 12). Outcome Brier = error vs true probs +
a constant, so paired differences estimate true-error differences. Per
condition the difference is divided by that condition's frozen-baseline error
(taken from validation/reference.csv vs targets), giving an estimated change
in R, the score that sets points = 60 / (1 + R). Negative = better.
A 95% interval comes from resampling whole episodes within each condition.
--val also scores each candidate once on validation truth (use sparingly).
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from slot_forecasting.contest import score_contest_files
from slot_forecasting.io import load_inputs, read_labels, read_targets, write_submission
from slot_forecasting.pair_mixer import PairMixer
from slot_forecasting.rounds import VERSION, round_metadata
from slot_forecasting.schema import PAYOUTS

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("candidates", help="JSON list of PairMixer keyword settings")
parser.add_argument("--data", default=str(HERE / "public"))
parser.add_argument("--val", action="store_true")
parser.add_argument("--scratch", default=str(HERE / "runs" / "tuning"))
args = parser.parse_args()

public = Path(args.data)
train_q, train_h = load_inputs(public / "train")
labels = read_labels(public / "train/labels.csv", train_q)
conditions = {r["query_id"]: r["condition"] for r in csv.DictReader(open(public / "train/conditions.csv"))}
episode = {q.query_id: q.episode_id for q in train_q}
val = public / "validation"
reference = {r["query_id"]: [float(r[c]) for c in ("p_0", "p_1", "p_2", "p_5", "p_20")]
             for r in csv.DictReader(open(val / "reference.csv"))}
val_cond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(val / "conditions.csv"))}
ref_err = defaultdict(list)
for t in read_targets(val / "targets.csv"):
    ref_err[val_cond[t.query_id]].append(sum((a - b) ** 2 for a, b in zip(reference[t.query_id], t.probabilities)))
baseline_error = {c: sum(v) / len(v) for c, v in ref_err.items()}


def cutoff_brier(settings):
    preds = PairMixer(**settings).fit(train_q, train_h).predict(train_q, train_h)
    return {q: sum((p - (1.0 if PAYOUTS[k] == labels[q] else 0.0)) ** 2 for k, p in enumerate(preds[q]))
            for q in preds}


def delta_r(diff, groups):
    by_cond = defaultdict(list)
    for cond, eps in groups.items():
        for e in eps:
            by_cond[cond].extend(diff_by_episode[cond][e])
    return sum(sum(v) / len(v) / baseline_error[c] for c, v in by_cond.items()) / len(by_cond)


base = cutoff_brier({})
event = {"max_points": round_metadata(public)["max_points"], "version": VERSION}
scratch = Path(args.scratch)
scratch.mkdir(parents=True, exist_ok=True)
for settings in json.loads(args.candidates):
    cand = cutoff_brier(settings)
    diff_by_episode = defaultdict(lambda: defaultdict(list))
    for q in cand:
        diff_by_episode[conditions[q]][episode[q]].append(cand[q] - base[q])
    groups = {c: sorted(e) for c, e in diff_by_episode.items()}
    point = delta_r(None, groups)
    rng = random.Random(0)
    boot = sorted(delta_r(None, {c: [rng.choice(e) for _ in e] for c, e in groups.items()}) for _ in range(400))
    per = {c[:5]: round(sum(sum(v) for v in d.values()) / sum(len(v) for v in d.values()) / baseline_error[c], 4)
           for c, d in sorted(diff_by_episode.items())}
    line = f"{json.dumps(settings)}  dR={point:+.4f} [{boot[10]:+.4f},{boot[389]:+.4f}]  per-cond {per}"
    if args.val:
        vq, vh = load_inputs(val)
        write_submission(scratch / "val.csv", PairMixer(**settings).fit(train_q, train_h).predict(vq, vh))
        report = score_contest_files(scratch / "val.csv", val / "targets.csv", val / "queries.csv",
                                     val / "conditions.csv", val / "reference.csv", **event)
        line += f"  VAL {report['technical_points']:.2f} R={report['mean_loss_ratio']:.4f}"
    print(line, flush=True)
