"""Recreate our test forecast: python predict.py --data <public folder> --out <submission.csv>

Combined model: a weighted blend of our rule model (rule_model.py) and the team's pair mixer
(slot_forecasting/pair_mixer.py). The blend weights live in blend_config.json, written by
auto_tune.py after it checks a blend on the audit half of validation. Without that file the default
is 20% rule + 80% mixer (validation 54.45 / 60 on main).

Every model refits on <data>/train (deterministic, no randomness), then predicts <data>/test.
Standard library only. Works unchanged for the main round and round two (adaptation).
"""
import argparse
import json
from pathlib import Path

from slot_forecasting.io import load_inputs, read_labels, write_submission
from slot_forecasting.pair_mixer import ALL_PAIRS, PairMixer, adapt

from rule_model import RuleForecaster

HERE = Path(__file__).resolve().parent

# Default blend. On main: 0.2 rule -> 54.45, 0.3 -> 54.48, mixer alone 54.21, rule alone 52.89.
# An independent check (fit on validation, Brier on train outcomes) slightly favoured the mixer
# alone, so 0.2 is the compromise. auto_tune.py may replace this via blend_config.json.
DEFAULT_WEIGHTS = {"rule": 0.2, "mixer": 0.8}

# Rule model v4 (52.89 / 60 alone): average of two variants that make different mistakes.
#  - "full":     rules are graded on predicting the exact payout
#  - "win/lose": rules are graded on win vs lose only (better for short histories)
# Both take payout SIZE from the machine's own wins blended toward its type's average, then refine
# it with per-machine size rules (machines with at least 40 wins).
SETTINGS = dict(size_strength=80.0, strength=0.5, trust_forget=0.99,
                size_rule_strength=20.0, size_rule_beta=1.0, size_rule_min_wins=40)


class Blend:
    """Equal average of several models; fit() fits each."""

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


def rule_model():
    return Blend([RuleForecaster(**SETTINGS), RuleForecaster(**SETTINGS, binary=True)])


def blend_weights():
    path = HERE / "blend_config.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["weights"]
    return dict(DEFAULT_WEIGHTS)


def build(data, weights=None, log=lambda *_: None):
    data = Path(data)
    weights = weights or blend_weights()
    queries, history = load_inputs(data / "train")
    labels = read_labels(data / "train" / "labels.csv", queries)
    fitted = {}
    if "mixer" in weights or "mixer_all" in weights:
        fitted["mixer"] = adapt(queries, history, labels, log=log)  # lag pairs rediscovered on train
    if "mixer_all" in weights:
        fitted["mixer_all"] = PairMixer(pairs=ALL_PAIRS, cell_mode=fitted["mixer"].cell_mode).fit(queries, history)
    if "rule" in weights:
        fitted["rule"] = rule_model().fit(queries, history, labels)
    unknown = set(weights) - set(fitted)
    if unknown:
        raise SystemExit(f"blend_config.json names unknown models: {sorted(unknown)}")
    return Weighted([(w, fitted[name]) for name, w in weights.items() if w > 0])


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
