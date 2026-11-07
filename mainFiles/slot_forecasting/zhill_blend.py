"""Reproduce the ``predict.py`` blend with a configurable adapted PairMixer memory."""

from __future__ import annotations

from typing import Any

from predict import Blend, RULE_WEIGHT, SETTINGS, Weighted
from rule_model import RuleForecaster
from slot_forecasting.pair_mixer import PairMixer, choose_pairs, cutoff_brier


def adapt_with_memory(queries, history, labels=None, *, half_life=(12.0, 40.0), log=print):
    """Mirror ``pair_mixer.adapt`` while setting memory before mode selection.

    Pair discovery and the parity/full cutoff-Brier comparison receive only the
    inputs passed here. The wrapper calls this with the training split in
    ``fit``, matching ``predict.py``'s ``adapt(train_queries, train_history,
    train_labels)`` flow.
    """
    pairs = choose_pairs(queries, history)
    log(f"lag pairs found on train: {pairs}")
    candidates = [
        PairMixer(pairs=pairs, cell_mode=mode, half_life=half_life).fit(
            queries, history
        )
        for mode in ("parity", "full")
    ]
    if labels is None:
        return candidates[0]

    scores = [cutoff_brier(model, queries, history, labels) for model in candidates]
    for model, score in zip(candidates, scores):
        log(f"cell_mode={model.cell_mode[0]}: train cutoff Brier {score:.5f}")
    return candidates[scores.index(min(scores))]


class ZhillBlendForecaster:
    """The shipped two-rule / PairMixer blend, fitted through the kit interface."""

    def __init__(
        self,
        *,
        half_life=(12.0, 40.0),
        rule_weight: float = RULE_WEIGHT,
        settings: dict[str, Any] | None = None,
        log=lambda *_: None,
    ) -> None:
        if not 0.0 <= rule_weight <= 1.0:
            raise ValueError("rule_weight must be in [0, 1]")
        self.half_life = tuple(half_life) if isinstance(half_life, (list, tuple)) else half_life
        self.rule_weight = float(rule_weight)
        self.settings = dict(SETTINGS if settings is None else settings)
        self.log = log
        self.rules = None
        self.mixer = None

    def fit(self, queries, history, labels=None):
        self.rules = Blend([
            RuleForecaster(**self.settings),
            RuleForecaster(**self.settings, binary=True),
        ]).fit(queries, history, labels)
        self.mixer = adapt_with_memory(
            queries,
            history,
            labels,
            half_life=self.half_life,
            log=self.log,
        )
        return self

    def predict(self, queries, history):
        if self.rules is None or self.mixer is None:
            raise RuntimeError("fit must be called before predict")
        return Weighted([
            (self.rule_weight, self.rules),
            (1.0 - self.rule_weight, self.mixer),
        ]).predict(queries, history)
