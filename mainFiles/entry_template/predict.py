"""House Edge entry: per-machine online mixture of lag-pair (match/differ) experts.

Usage: python predict.py --data /data --out /output/submission.csv
On <data>/train it rediscovers which lag pairs drive wins, picks match/differ vs
4-cell experts by Brier on train labels.csv at the cutoff, and fits payout-size
priors; then predicts <data>/test. Standard library only, deterministic.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slot_forecasting.io import load_inputs, read_labels, write_submission  # noqa: E402
from slot_forecasting.pair_mixer import adapt  # noqa: E402

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
model = adapt(train_queries, train_history, labels)
queries, history = load_inputs(data / args.split)
write_submission(args.out, model.predict(queries, history))
print(f"wrote {args.out}")
