"""House Edge entry: pair mixer + hidden-rule HMM + change-point regime model (win chance ensemble).

Usage: python predict.py --data /data --out /output/submission.csv
On <data>/train it rediscovers which lag pairs drive wins and which lags drive payout
size, picks match/differ vs 4-cell experts by Brier on train labels.csv at the cutoff,
fits payout-size priors, and reads two priors from <data>/validation/targets.csv (if
present): the payout-size modes per machine type (any number; two in the main round) and
the dead/live/middle win-chance mixture; then predicts <data>/test. Without targets.csv
it uses priors.json saved at build time, and without that a win prior fitted to train.
Standard library only, deterministic.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slot_forecasting.io import load_inputs, read_labels, read_targets, write_submission  # noqa: E402
from slot_forecasting.ensemble import build_ensemble  # noqa: E402
from slot_forecasting.pair_mixer import size_modes_from_targets, win_prior_by_type_from_targets  # noqa: E402

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
    win_prior = win_prior_by_type_from_targets(targets, validation_queries)
elif saved.exists():
    priors = json.loads(saved.read_text(encoding="utf-8"))
    size_modes, win_prior = priors["size_modes"], priors["win_prior"]
model = build_ensemble(train_queries, train_history, labels, size_modes, win_prior)
queries, history = load_inputs(data / args.split)
write_submission(args.out, model.predict(queries, history))
print(f"wrote {args.out}")
