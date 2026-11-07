"""Recreate our test forecast: python predict.py --data <public folder> --out <submission.csv>

Combined model (validation 54.48 / 60): 30% our rule model + 70% the team's pair mixer
(slot_forecasting/pair_mixer.py, COMBINED_SOLUTIONS branch). Both refit on <data>/train
(deterministic, no randomness), then predict <data>/test. Standard library only.
Works unchanged for the main round and round two (adaptation).
"""
import argparse
from pathlib import Path

from slot_forecasting.io import load_inputs, read_labels, write_submission
from slot_forecasting.pair_mixer import adapt

from rule_model import RuleForecaster

# Weight of our rule model in the final blend; the pair mixer gets the rest. Chosen on validation:
# 0.3 -> 54.48 (tune 54.42, audit 54.56) vs rule model 52.89 and pair mixer 54.21 alone.
# The rule model is better on stable machines, the pair mixer on short_history / drift / shared_change.
RULE_WEIGHT = 0.3


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


class Weighted:
    """Weighted average of already-fitted models' forecasts."""

    def __init__(self, parts):
        self.parts = parts  # [(weight, model)]

    def predict(self, queries, history):
        forecasts = [(w, model.predict(queries, history)) for w, model in self.parts]
        total = sum(w for w, _ in forecasts)
        return {key: tuple(sum(w * f[key][i] for w, f in forecasts) / total for i in range(5))
                for key in forecasts[0][1]}


def build(data, log=lambda *_: None):
    data = Path(data)
    queries, history = load_inputs(data / "train")
    labels = read_labels(data / "train" / "labels.csv", queries)
    rules = Blend([RuleForecaster(**SETTINGS), RuleForecaster(**SETTINGS, binary=True)])
    rules.fit(queries, history, labels)
    mixer = adapt(queries, history, labels, log=log)  # rediscovers lag pairs on this round's train
    return Weighted([(RULE_WEIGHT, rules), (1.0 - RULE_WEIGHT, mixer)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Folder with train/ and test/ (e.g. /data or public)")
    parser.add_argument("--out", required=True, help="Where to write the submission CSV")
    parser.add_argument("--split", default="test", help="test (default) or validation")
    args = parser.parse_args()
    data = Path(args.data)
    if (data / "public").is_dir():
        data = data / "public"
    model = build(data)
    queries, history = load_inputs(data / args.split)
    write_submission(args.out, model.predict(queries, history))


if __name__ == "__main__":
    main()
