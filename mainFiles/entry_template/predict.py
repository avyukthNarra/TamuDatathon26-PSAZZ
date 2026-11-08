"""House Edge entry: lag-pair win mixture, optional flip relabel, train-learned size.

Usage: python predict.py --data /data --out /output/submission.csv
On <data>/train it rediscovers which lag pairs drive wins, picks match/differ vs
4-cell experts by Brier on train labels.csv at the cutoff, and reads the dead/live
win-chance prior from <data>/validation/targets.csv when that file is present.
Without targets.csv it uses priors.json saved at build time. Which of the size
rule, the episode flip, and the round-two discovery patch are on is
entry_config.py, written when this entry was built. Standard library only.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slot_forecasting.entry_config import USE_FLIP, USE_ROUND2, USE_SIZE, USE_SIZE_GATE  # noqa: E402
from slot_forecasting.forecast import fit_flip, fit_size, predict_split  # noqa: E402
from slot_forecasting.io import load_inputs, read_labels, read_targets, write_submission  # noqa: E402
from slot_forecasting.pair_mixer import adapt, size_modes_from_targets, win_prior_by_type_from_targets  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--split", default="test")
args = parser.parse_args()
data = Path(args.data)
if (data / "public").is_dir():
    data = data / "public"
train_queries, train_history = load_inputs(data / "train")
labels_path = data / "train" / "labels.csv"
labels = read_labels(labels_path, train_queries) if labels_path.exists() else None
# Two priors come from this round's validation answers: the exact payout-size mixes per
# machine type, and the dead/live win-chance levels. If that file is missing, use the copy
# saved next to this script when the entry was built.
targets_path = data / "validation" / "targets.csv"
saved = Path(__file__).resolve().parent / "priors.json"
size_modes = win_prior = None
if targets_path.exists():
    targets = read_targets(targets_path)
    validation_queries = load_inputs(data / "validation")[0]
    size_modes = size_modes_from_targets(targets, validation_queries)
    win_prior = win_prior_by_type_from_targets(targets, validation_queries, two_means=USE_ROUND2)
elif saved.exists():
    priors = json.loads(saved.read_text(encoding="utf-8"))
    size_modes, win_prior = priors["size_modes"], priors["win_prior"]
model = adapt(train_queries, train_history, labels, size_modes=size_modes, win_prior=win_prior,
              round2=USE_ROUND2)
size_rule = fit_size(train_queries, train_history, USE_SIZE, labels, USE_SIZE_GATE)
gate = fit_flip(model, train_queries, train_history, USE_FLIP)
if size_rule is not None:
    print(f"size lags: {size_rule.lags}  head: {size_rule.gate_choice} "
          f"static prior: {size_rule.static_prior:.4f}")
    if size_rule.gate_diagnostics is not None:
        print(f"size gate diagnostics: {json.dumps(size_rule.gate_diagnostics, sort_keys=True)}")
if gate is not None:
    print(f"flip gate: {gate.threshold:.4f}")
queries, history = load_inputs(data / args.split)
write_submission(args.out, predict_split(model, queries, history, size_rule, gate))
print(f"wrote {args.out}")
