"""House Edge entry: per-machine online mixture of lag-pair (match/differ) experts.

Usage: python predict.py --data /data --out /output/submission.csv
On <data>/train it rediscovers which lag pairs drive wins, picks match/differ vs
4-cell experts by Brier on train labels.csv at the cutoff, fits payout-size
priors, and reads two priors from <data>/validation/targets.csv (if present): the two
payout-size modes per machine type and the dead/live win-chance levels; then predicts <data>/test.
Standard library only, deterministic.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slot_forecasting.io import load_inputs, read_labels, read_targets, write_submission  # noqa: E402
from slot_forecasting.pair_mixer import adapt, size_modes_from_targets, win_prior_from_targets  # noqa: E402

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
# Two priors come from this round's validation answers: the two exact payout-size mixes per
# machine type, and the dead/live win-chance levels. If that file is missing, use the copy
# saved next to this script when the entry was built.
targets_path = data / "validation" / "targets.csv"
saved = Path(__file__).resolve().parent / "priors.json"
size_modes = win_prior = None
if targets_path.exists():
    targets = read_targets(targets_path)
    size_modes = size_modes_from_targets(targets, load_inputs(data / "validation")[0])
    win_prior = win_prior_from_targets(targets)
elif saved.exists():
    priors = json.loads(saved.read_text(encoding="utf-8"))
    size_modes, win_prior = priors["size_modes"], priors["win_prior"]
model = adapt(train_queries, train_history, labels, size_modes=size_modes, win_prior=win_prior)
queries, history = load_inputs(data / args.split)
write_submission(args.out, model.predict(queries, history))
print(f"wrote {args.out}")
