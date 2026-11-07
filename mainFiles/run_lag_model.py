"""Train the lag-feature model on every past spin, score it on validation, write a submission.

Usage:  python run_lag_model.py --out lag-run [--blend 0.8] [--learning-rate 0.01] [--epochs 6]
Prints the validation score per condition next to the pooled baseline's.
"""
import argparse
from pathlib import Path
from time import perf_counter

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.contest import score_contest_files
from slot_forecasting.io import load_inputs, read_labels, write_json, write_submission
from slot_forecasting.lag_model import LagForecaster
from slot_forecasting.rounds import VERSION, round_metadata
from slot_forecasting.starter import public_directory

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=str(Path(__file__).resolve().parent / "public"))
parser.add_argument("--out", default="lag-run")
parser.add_argument("--epochs", type=int, default=6)
parser.add_argument("--learning-rate", type=float, default=0.01)
parser.add_argument("--l2", type=float, default=1e-5)
parser.add_argument("--blend", type=float, default=0.8,
                    help="weight on the lag model; the rest goes to pooled_recent (1.0 = lag model only; 0.8 scored best of 1.0 and 0.8 on validation)")
args = parser.parse_args()

started = perf_counter()
public = public_directory(args.data)
out = Path(args.out).expanduser().resolve()
train_queries, train_history = load_inputs(public / "train")
labels = read_labels(public / "train/labels.csv", train_queries)
splits = {split: load_inputs(public / split) for split in ("validation", "test")}

baseline = FrequencyForecaster("pooled_recent").fit(train_queries, train_history, labels)
model = LagForecaster(epochs=args.epochs, learning_rate=args.learning_rate, l2=args.l2, blend=args.blend)
model.fit(train_queries, train_history, labels, log=print)
print(f"trained in {perf_counter() - started:.0f}s")

files = {"validation": out / "validation_predictions.csv", "test": out / "submission.csv"}
for split, (queries, history) in splits.items():
    write_submission(files[split], model.predict(queries, history, baseline.predict(queries, history)))
model.save(out / "lag_model.json")

event = {"max_points": round_metadata(public)["max_points"], "version": VERSION}
val = public / "validation"
report = score_contest_files(files["validation"], val / "targets.csv", val / "queries.csv",
                             val / "conditions.csv", val / "reference.csv", **event)
write_json(out / "validation_score.json", report)
print(f"\n{'condition':<15}{'error':>9}{'baseline':>10}{'ratio':>8}")
for name, row in report["by_condition"].items():
    print(f"{name:<15}{row['error']:>9.4f}{row['reference_error']:>10.4f}{row['loss_ratio']:>8.3f}")
print(f"\nmean ratio R = {report['mean_loss_ratio']:.3f}  ->  "
      f"{report['technical_points']:.2f} / {report['max_points']:g} points (pooled baseline = 30)")
print("Submission:", files["test"], f"  ({perf_counter() - started:.0f}s total)")
