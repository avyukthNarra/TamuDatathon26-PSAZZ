"""House Edge combo entry: 20% rule model + 80% pair mixer v3.

Usage: python predict.py --data <public folder or its parent> --out <csv> [--split test|validation]

Both models are fitted on <data>/train (deterministic, no randomness). The pair mixer's two
payout-size modes per machine type are read from <data>/validation/targets.csv with
size_modes_from_targets and passed to adapt(); if that file is missing the mixer falls back to
free frequencies. Lag pairs, size modes and priors are all learned from the folder given.
The forecast for <data>/<split> is 0.2 * rule + 0.8 * mixer, renormalised per query.
Standard library only.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if not (HERE / "slot_forecasting").is_dir():  # running inside the repo: use mainFiles/ modules
    sys.path.insert(1, str(HERE.parent))

from slot_forecasting.io import load_inputs, read_labels, read_targets, write_submission  # noqa: E402
from slot_forecasting.pair_mixer import adapt, size_modes_from_targets  # noqa: E402

from rule_model import RuleForecaster  # noqa: E402

RULE_WEIGHT = 0.2

# Same two rule-model settings as zhill-branch predict.py: one graded on the exact payout, one on
# win/lose only (better for short histories); their forecasts are averaged.
SETTINGS = dict(size_strength=80.0, strength=0.5, trust_forget=0.99,
                size_rule_strength=20.0, size_rule_beta=1.0, size_rule_min_wins=40)


def fit_rules(queries, history, labels):
    models = [RuleForecaster(**SETTINGS), RuleForecaster(**SETTINGS, binary=True)]
    for model in models:
        model.fit(queries, history, labels)
    return models


def blend(rule_models, mixer, queries, history, rule_weight=RULE_WEIGHT):
    rule_forecasts = [model.predict(queries, history) for model in rule_models]
    mixer_forecast = mixer.predict(queries, history)
    out = {}
    for key, mix in mixer_forecast.items():
        rule = [sum(f[key][i] for f in rule_forecasts) / len(rule_forecasts) for i in range(5)]
        p = [rule_weight * r + (1.0 - rule_weight) * m for r, m in zip(rule, mix)]
        total = sum(p)
        out[key] = tuple(x / total for x in p)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Folder with train/ and test/ (or its parent, e.g. /data)")
    parser.add_argument("--out", required=True, help="Where to write the submission CSV")
    parser.add_argument("--split", default="test", choices=("test", "validation"))
    args = parser.parse_args()
    data = Path(args.data)
    if (data / "public").is_dir():
        data = data / "public"

    queries, history = load_inputs(data / "train")
    labels = read_labels(data / "train" / "labels.csv", queries)
    rules = fit_rules(queries, history, labels)

    targets_path = data / "validation" / "targets.csv"
    size_modes = None
    if targets_path.exists():
        size_modes = size_modes_from_targets(read_targets(targets_path), load_inputs(data / "validation")[0])
    mixer = adapt(queries, history, labels, log=lambda *_: None, size_modes=size_modes)

    split_queries, split_history = load_inputs(data / args.split)
    write_submission(args.out, blend(rules, mixer, split_queries, split_history))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
