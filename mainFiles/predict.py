"""Recreate our test forecast: python predict.py --data <public folder> --out <submission.csv>

Combined model: a weighted blend of our rule model (rule_model.py) and the team's pair mixer v3
(slot_forecasting/pair_mixer.py: two-mode payout sizes + dead/live win-cell prior, both read from
<data>/validation/targets.csv, with a saved copy in priors.json). The blend weights live in blend_config.json, written by
auto_tune.py after it checks a blend on the audit half of validation. Without that file the default
is 20% rule + 80% mixer (validation 54.45 / 60 on main).

Every model refits on <data>/train (deterministic, no randomness), then predicts <data>/test.
Standard library only. Works unchanged for the main round and round two (adaptation).
"""
import argparse
import json
from pathlib import Path

from slot_forecasting.io import load_inputs, read_labels, read_targets, write_submission
from slot_forecasting.pair_mixer import (ALL_PAIRS, WIN_PRIOR_ETA, PairMixer, adapt, size_modes_from_targets,
                                         win_prior_by_type_from_targets)

from rule_model import RuleForecaster
from hmm_model import HiddenRuleHMM, levels_from_targets
from size_hmm import SizeRuleHMM, modes_from_targets, prior_from_train
from slot_forecasting.pair_mixer import choose_pairs

HERE = Path(__file__).resolve().parent

# Default blend. On main: 0.2 rule -> 54.45, 0.3 -> 54.48, mixer alone 54.21, rule alone 52.89.
# An independent check (fit on validation, Brier on train outcomes) slightly favoured the mixer
# alone, so 0.2 is the compromise. auto_tune.py may replace this via blend_config.json.
# v5 (Nov 7, 8 PM): + hidden-rule HMM (hmm_model.py) at 50%: validation 56.28 on all 96 episodes
# (tune 56.27, audit 56.31) vs 55.58 for 0.2 rule + 0.8 mixer. The HMM takes its payout-size split
# from the rule+mixer part, so "hmm" here only replaces the win chance.
# v6 (Nov 7, 10 PM, after merging COMBINED_SOLUTIONS mixer6 = per-type win prior + Pranav's rule_hmm):
# mixer6 alone 56.51; 0.1 rule + 0.1 our HMM + 0.8 mixer6 = 56.59 (tune 56.55, audit 56.65) on all 96.
DEFAULT_WEIGHTS = {"rule": 0.1, "mixer": 0.8, "hmm": 0.1}

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


def mixer_priors(data):
    """Pair mixer v3 priors from this round's validation answers (allowed: RULES.md "learn from the
    training and validation data"): the two exact payout-size mixes per machine type and the dead/live
    win-chance levels. Falls back to the copy saved in priors.json when the entry was built."""
    targets_path = Path(data) / "validation" / "targets.csv"
    if targets_path.exists():
        targets = read_targets(targets_path)
        vq = load_inputs(Path(data) / "validation")[0]
        # per-type win prior (mixer6): adapt() then also blends in Pranav's rule_hmm at 0.3
        return {"size_modes": size_modes_from_targets(targets, vq),
                "win_prior": win_prior_by_type_from_targets(targets, vq)}
    saved = HERE / "priors.json"
    if saved.exists():
        priors = json.loads(saved.read_text(encoding="utf-8"))
        return {"size_modes": priors["size_modes"], "win_prior": priors["win_prior"]}
    return {"size_modes": None, "win_prior": None}


def hmm_levels(data):
    """Per-type live/dead win levels for the HMM, from validation answers, else priors.json."""
    targets_path = Path(data) / "validation" / "targets.csv"
    if targets_path.exists():
        return levels_from_targets(read_targets(targets_path), load_inputs(Path(data) / "validation")[0])
    saved = HERE / "priors.json"
    if saved.exists():
        return {k: tuple(v) for k, v in json.loads(saved.read_text(encoding="utf-8")).get("hmm_levels", {}).items()}
    return {}


def size_rule_modes(data):
    """The two exact size mixes per type for the size-rule HMM, from validation answers, else priors.json."""
    targets_path = Path(data) / "validation" / "targets.csv"
    if targets_path.exists():
        return modes_from_targets(read_targets(targets_path), load_inputs(Path(data) / "validation")[0])
    saved = HERE / "priors.json"
    if saved.exists():
        return {k: tuple(tuple(m) for m in v)
                for k, v in json.loads(saved.read_text(encoding="utf-8")).get("size_rule_modes", {}).items()}
    return {}


def saved_priors(data):
    """Everything priors.json stores, so the entry also runs without validation answers."""
    return {**mixer_priors(data), "hmm_levels": hmm_levels(data), "size_rule_modes": size_rule_modes(data)}


# Size-rule HMM (size_hmm.py): replaces the payout-size split given a win. No switching (the size
# rule does not drift on main). Validation 56.59 -> 57.30 with the EM rule prior (57.32 with a
# hand-set 97% core prior, which loses 0.2-0.6 / 40 on simulated round-two kits).
SIZE_HMM = dict(none_weight=0.02, switch=0.0)


class WithSizeHMM:
    """Keeps the base win chance; payout size given a win comes from the size-rule HMM
    (machine types without known size mixes keep the base split)."""

    def __init__(self, base, size_hmm):
        self.base, self.size_hmm = base, size_hmm

    def predict(self, queries, history):
        base = self.base.predict(queries, history)
        splits = self.size_hmm.splits(queries, history)
        out = {}
        for k, p in base.items():
            if k in splits:
                out[k] = tuple([p[0]] + [(1 - p[0]) * x for x in splits[k]])
            else:
                out[k] = p
        return out


class WithHMM:
    """The HMM replaces the win chance; its payout sizes come from the rule+mixer blend."""

    def __init__(self, base, hmm, weight):
        self.base, self.hmm, self.weight = base, hmm, weight

    def predict(self, queries, history):
        base = self.base.predict(queries, history)
        hmm = self.hmm.predict(queries, history, size_from=base)
        w = self.weight
        return {k: tuple(w * a + (1 - w) * b for a, b in zip(hmm[k], base[k])) for k in base}


def build(data, weights=None, log=lambda *_: None):
    data = Path(data)
    weights = weights or blend_weights()
    queries, history = load_inputs(data / "train")
    labels = read_labels(data / "train" / "labels.csv", queries)
    fitted = {}
    if "mixer" in weights or "mixer_all" in weights:
        priors = mixer_priors(data)
        # lag pairs rediscovered on this round's train
        fitted["mixer"] = adapt(queries, history, labels, log=log, **priors)
    if "mixer_all" in weights:
        extra = {"win_prior": priors["win_prior"], "eta": WIN_PRIOR_ETA} if priors["win_prior"] else {}
        fitted["mixer_all"] = PairMixer(pairs=ALL_PAIRS, cell_mode=fitted["mixer"].cell_mode,
                                        size_modes=priors["size_modes"], **extra).fit(queries, history)
    if "rule" in weights:
        fitted["rule"] = rule_model().fit(queries, history, labels)
    unknown = set(weights) - set(fitted) - {"hmm"}
    if unknown:
        raise SystemExit(f"blend_config.json names unknown models: {sorted(unknown)}")
    base = Weighted([(w, fitted[name]) for name, w in weights.items() if w > 0 and name != "hmm"])
    if weights.get("hmm", 0) > 0:
        hmm = HiddenRuleHMM(choose_pairs(queries, history), levels=hmm_levels(data))
        base = WithHMM(base, hmm, weights["hmm"])
    modes = size_rule_modes(data)
    if modes:
        # rule prior fitted by EM on this round's train (lag rules, fixed mixes, none), so a round
        # with a different size mechanism re-learns it. Main: ~91% on payout >= 2x at lag 1/2/4.
        rules = prior_from_train(queries, history, modes, none_weight=SIZE_HMM["none_weight"])
        log("size-rule prior (top):", sorted(rules, key=rules.get, reverse=True)[:6])
        base = WithSizeHMM(base, SizeRuleHMM(modes, rules, SIZE_HMM["switch"]))
    return base


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
