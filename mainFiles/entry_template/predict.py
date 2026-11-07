"""House Edge entry: per-machine online mixture of lag-pair (match/differ) experts.

Usage: python predict.py --data /data --out /output/submission.csv
Fits the only learned constants (payout-size priors per machine type) on
<data>/train, then predicts <data>/test. Standard library only, deterministic.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slot_forecasting.io import load_inputs, write_submission  # noqa: E402
from slot_forecasting.pair_mixer import PairMixer  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--split", default="test")
args = parser.parse_args()
data = Path(args.data)
if (data / "public").is_dir():
    data = data / "public"
train_queries, train_history = load_inputs(data / "train")
model = PairMixer().fit(train_queries, train_history)
queries, history = load_inputs(data / args.split)
write_submission(args.out, model.predict(queries, history))
print(f"wrote {args.out}")
