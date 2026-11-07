"""Recreate our test forecast: python predict.py --data <public folder> --out <submission.csv>

Fits the rule model on <data>/train (deterministic, no randomness), then predicts <data>/test.
Standard library only. Works unchanged for the main round and round two (adaptation).
"""
import argparse
from pathlib import Path

from slot_forecasting.io import load_inputs, read_labels, write_submission

from rule_model import RuleForecaster


# v4 (validation 52.89 / 60): average of two rule models that make different mistakes.
#  - "full":     rules are graded on predicting the exact payout
#  - "win/lose": rules are graded on win vs lose only (better for short histories)
# Both take payout SIZE from the machine's own wins blended toward its type's average, then refine
# it with per-machine size rules (machines with at least 40 wins).
SETTINGS = dict(size_strength=80.0, strength=0.5, trust_forget=0.99,
                size_rule_strength=20.0, size_rule_beta=1.0, size_rule_min_wins=40)


class Blend:
    def __init__(self, models):
        self.models = models

    def fit(self, queries, history, labels):
        for model in self.models:
            model.fit(queries, history, labels)
        return self

    def predict(self, queries, history):
        forecasts = [model.predict(queries, history) for model in self.models]
        return {key: tuple(sum(f[key][i] for f in forecasts) / len(forecasts) for i in range(5))
                for key in forecasts[0]}


def build(data):
    data = Path(data)
    queries, history = load_inputs(data / "train")
    labels = read_labels(data / "train" / "labels.csv", queries)
    model = Blend([RuleForecaster(**SETTINGS), RuleForecaster(**SETTINGS, binary=True)])
    return model.fit(queries, history, labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Folder with train/ and test/ (e.g. /data or public)")
    parser.add_argument("--out", required=True, help="Where to write the submission CSV")
    parser.add_argument("--split", default="test", help="test (default) or validation")
    args = parser.parse_args()
    model = build(args.data)
    queries, history = load_inputs(Path(args.data) / args.split)
    write_submission(args.out, model.predict(queries, history))


if __name__ == "__main__":
    main()
