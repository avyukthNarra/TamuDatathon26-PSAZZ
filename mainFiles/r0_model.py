"""Train-only combined forecast wrapper for the R0 baseline experiment."""
from math import isfinite

from rule_model import RuleForecaster
from slot_forecasting.pair_mixer import adapt
from slot_forecasting.train_priors import estimate_train_priors


RULE_SETTINGS = dict(size_strength=80.0, strength=0.5, trust_forget=0.99,
                     size_rule_strength=20.0, size_rule_beta=1.0, size_rule_min_wins=40)


class CombinedForecaster:
    """Fit the PairMixer, optionally blended with the copied 20% zhill rule model.

    ``prior_source='train'`` estimates both priors solely from train queries/history.
    ``prior_source='validation'`` is an explicitly diagnostic mode and requires
    ``provided_priors`` to be injected by the caller; this class never reads files.
    """

    def __init__(self, prior_source="train", provided_priors=None, rule_weight=0.0):
        if prior_source not in ("train", "validation"):
            raise ValueError("prior_source must be 'train' or 'validation'.")
        if not isinstance(rule_weight, (int, float)) or not isfinite(rule_weight) or not 0 <= rule_weight <= 1:
            raise ValueError("rule_weight must be a finite number in [0, 1].")
        if prior_source == "validation" and provided_priors is None:
            raise ValueError("validation prior_source requires explicitly injected provided_priors.")
        self.prior_source = prior_source
        self.provided_priors = provided_priors
        self.rule_weight = float(rule_weight)
        self.mixer = None
        self.rules = None

    def fit(self, queries, history, labels=None):
        if self.prior_source == "train":
            size_modes, win_prior = estimate_train_priors(queries, history)
        else:
            size_modes = self.provided_priors.get("size_modes") or {}
            win_prior = self.provided_priors.get("win_prior")
        self.mixer = adapt(queries, history, labels, size_modes=size_modes, win_prior=win_prior)
        if self.rule_weight > 0:
            self.rules = [RuleForecaster(**RULE_SETTINGS),
                          RuleForecaster(**RULE_SETTINGS, binary=True)]
            for model in self.rules:
                model.fit(queries, history, labels)
        return self

    def predict(self, queries, history):
        if self.mixer is None:
            raise RuntimeError("fit must be called before predict.")
        mixer_predictions = self.mixer.predict(queries, history)
        if not self.rule_weight:
            return mixer_predictions
        rule_predictions = [model.predict(queries, history) for model in self.rules]
        blended = {}
        for key, mixer in mixer_predictions.items():
            rule = tuple(sum(prediction[key][i] for prediction in rule_predictions) / len(rule_predictions)
                         for i in range(len(mixer)))
            blended[key] = tuple((1 - self.rule_weight) * mixer[i] + self.rule_weight * rule[i]
                                 for i in range(len(mixer)))
        return blended


def make_model(prior_source="train", provided_priors=None):
    """Construct the safe mixer-only model."""
    return CombinedForecaster(prior_source=prior_source, provided_priors=provided_priors, rule_weight=0.0)


def make_blend(prior_source="train", provided_priors=None):
    """Construct the safe 20% rule / 80% mixer blend."""
    return CombinedForecaster(prior_source=prior_source, provided_priors=provided_priors, rule_weight=0.2)
