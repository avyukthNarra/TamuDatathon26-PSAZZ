"""Select per-machine memory separately for combo trees and logistic models.

The tree and logistic paths each use a prequential Brier score over recent
historical rows. Selection is soft, and the final component predictions are
fit on all valid rows strictly before the query cutoff.
"""

from __future__ import annotations

from math import exp, isfinite, log
import warnings
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .combo_trees_blend import ComboForecaster
from .pair_logistic import _lag_vector
from .shape_head import ShapeHeadForecaster


class ComboAdaptiveMemoryForecaster(ComboForecaster):
    """Combo forecaster with independent soft memory selection per win model.

    Prequential model fits use at most the most recent 24 valid rows, scoring
    every fourth row after at least 16 earlier training rows. The final query
    prediction refits each selected memory candidate on all pre-cutoff rows;
    those tree fits retain the combo's full estimator count.
    """

    def __init__(
        self,
        *args: Any,
        half_lives: tuple[int | None, ...] = (None, 16, 32, 64),
        selection_temperature: float = 128.0,
        max_eval_rows: int = 24,
        eval_stride: int = 4,
        min_train: int = 16,
        selector_tree_estimators: int = 8,
        **kwargs: Any,
    ) -> None:
        # Keep the tested combo settings unless a caller explicitly overrides them.
        kwargs.setdefault("mix", 0.5)
        kwargs.setdefault("short_alpha", 0.6)
        super().__init__(*args, **kwargs)

        if not half_lives:
            raise ValueError("half_lives must be non-empty")
        for half_life in half_lives:
            if half_life is not None and (
                isinstance(half_life, bool)
                or not isfinite(half_life)
                or half_life <= 0
            ):
                raise ValueError("half_lives must contain only None or finite positive values")
        if not isfinite(selection_temperature) or selection_temperature <= 0:
            raise ValueError("selection_temperature must be finite and positive")
        for value, name in (
            (max_eval_rows, "max_eval_rows"),
            (eval_stride, "eval_stride"),
            (min_train, "min_train"),
            (selector_tree_estimators, "selector_tree_estimators"),
        ):
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        # Preserve declaration order and remove duplicates for stable diagnostics.
        self.half_lives = tuple(dict.fromkeys(half_lives))
        self.selection_temperature = float(selection_temperature)
        self.max_eval_rows = int(max_eval_rows)
        self.eval_stride = int(eval_stride)
        self.min_train = int(min_train)
        self.selector_tree_estimators = int(selector_tree_estimators)
        self.selection_diagnostics: dict[tuple[Any, ...], dict[str, Any]] = {}

    @staticmethod
    def _candidate_name(half_life: int | None) -> str:
        return "half_life=none" if half_life is None else f"half_life={half_life:g}"

    @staticmethod
    def _weights(ticks: np.ndarray, cutoff: float, half_life: int | None) -> np.ndarray:
        if half_life is None:
            return np.ones(len(ticks), dtype=float)
        ages = cutoff - 1.0 - ticks.astype(float)
        return np.exp2(-ages / half_life)

    @staticmethod
    def _smoothed_prior(labels: np.ndarray, prior_win: float) -> float:
        return float((float(labels.sum()) + 2.0 * prior_win) / (len(labels) + 2.0))

    def _estimator(self, family: str, n_estimators: int):
        if family == "logistic":
            return LogisticRegression(C=self.C, max_iter=1000)
        classifier = ExtraTreesClassifier if self.kind == "extra" else RandomForestClassifier
        return classifier(
            n_estimators=n_estimators,
            min_samples_leaf=self.min_leaf,
            random_state=self.seed,
            n_jobs=1,
        )

    def _fit_probability(
        self,
        family: str,
        features: np.ndarray,
        labels: np.ndarray,
        ticks: np.ndarray,
        query_features: np.ndarray,
        cutoff: float,
        half_life: int | None,
        prior_win: float,
        *,
        for_selection: bool,
    ) -> float:
        if len(labels) < self.min_train or np.unique(labels).size < 2:
            return self._smoothed_prior(labels, prior_win)

        sample_weights = self._weights(ticks, cutoff, half_life)
        n_estimators = (
            self.selector_tree_estimators if for_selection else self.n_estimators
        )
        model = self._estimator(family, n_estimators)
        if family == "logistic":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(features, labels, sample_weight=sample_weights)
        else:
            model.fit(features, labels, sample_weight=sample_weights)
        class_index = int(np.flatnonzero(model.classes_ == 1)[0])
        probability = float(model.predict_proba(query_features.reshape(1, -1))[0, class_index])
        return float(np.clip(probability, 1e-6, 1 - 1e-6))

    def _select_family(
        self,
        family: str,
        features: np.ndarray,
        labels: np.ndarray,
        ticks: np.ndarray,
        query_features: np.ndarray,
        cutoff: float,
        prior_win: float,
    ) -> tuple[float, dict[str, Any]]:
        evaluation_indices = np.arange(
            max(0, len(labels) - self.max_eval_rows), len(labels), self.eval_stride
        )
        losses: dict[str, list[float]] = {
            self._candidate_name(half_life): [] for half_life in self.half_lives
        }
        prequential: list[dict[str, Any]] = []

        for index in evaluation_indices:
            prefix = np.flatnonzero(ticks < ticks[index])
            if len(prefix) < self.min_train:
                continue
            predictions: dict[str, float] = {}
            for half_life in self.half_lives:
                name = self._candidate_name(half_life)
                probability = self._fit_probability(
                    family,
                    features[prefix],
                    labels[prefix],
                    ticks[prefix],
                    features[index],
                    float(ticks[index]),
                    half_life,
                    prior_win,
                    for_selection=True,
                )
                predictions[name] = probability
                losses[name].append((probability - int(labels[index])) ** 2)
            prequential.append({"tick": float(ticks[index]), "probabilities": predictions})

        mean_losses = {
            name: (float(np.mean(values)) if values else None)
            for name, values in losses.items()
        }
        available = [
            half_life for half_life in self.half_lives
            if mean_losses[self._candidate_name(half_life)] is not None
        ]
        if available:
            best_loss = min(mean_losses[self._candidate_name(hl)] for hl in available)
            scores = {
                self._candidate_name(hl): exp(
                    -(mean_losses[self._candidate_name(hl)] - best_loss)
                    * self.selection_temperature
                )
                for hl in available
            }
            total = sum(scores.values())
            selected = {name: score / total for name, score in scores.items()}
        else:
            selected = {self._candidate_name(self.half_lives[0]): 1.0}

        final_probabilities: dict[str, float] = {}
        for half_life in self.half_lives:
            name = self._candidate_name(half_life)
            if selected.get(name, 0.0) == 0.0:
                continue
            final_probabilities[name] = self._fit_probability(
                family,
                features,
                labels,
                ticks,
                query_features,
                cutoff,
                half_life,
                prior_win,
                for_selection=False,
            )

        probability = sum(
            weight * final_probabilities[name]
            for name, weight in selected.items()
            if name in final_probabilities
        )
        diagnostics = {
            "mean_brier_loss": mean_losses,
            "selected_weights": selected,
            "prequential_ticks": [row["tick"] for row in prequential],
            "prequential_predictions": prequential,
            "final_probabilities": final_probabilities,
            "fallback": not bool(available),
        }
        return float(np.clip(probability, 1e-6, 1 - 1e-6)), diagnostics

    def _prior_win(self) -> float:
        global_prior = getattr(self.base, "global_prior", None)
        if global_prior is None:
            return 0.4
        return float(np.clip(1.0 - float(global_prior[0]), 0.0, 1.0))

    def _postprocess_combo_probability(self, probability: float, seq: dict, cutoff: int) -> float:
        probability = float(np.clip(probability, 1e-6, 1 - 1e-6))
        if self.temp != 1.0:
            logit = log(probability / (1.0 - probability)) / self.temp
            probability = 1.0 / (1.0 + exp(-logit))

        alpha = self.alpha
        if self.short_alpha is not None and sum(tick < cutoff for tick in seq) < 100:
            alpha = self.short_alpha
        base_probability = self._base_pw
        if base_probability is None:
            base_probability = self._prior_win()
        return float(alpha * probability + (1.0 - alpha) * base_probability)

    def win_probability(self, seq: dict[int, int], cutoff: int) -> float | None:
        tree_query = _lag_vector(seq, cutoff, self.lags)
        logistic_query = ShapeHeadForecaster._features(self, seq, cutoff)
        if tree_query is None or logistic_query is None:
            return None

        tree_rows: list[np.ndarray] = []
        logistic_rows: list[np.ndarray] = []
        labels: list[int] = []
        ticks: list[int] = []
        for tick, payout in sorted(seq.items()):
            if tick >= cutoff:
                continue
            tree_row = _lag_vector(seq, tick, self.lags)
            logistic_row = ShapeHeadForecaster._features(self, seq, tick)
            if tree_row is None or logistic_row is None:
                continue
            tree_rows.append(tree_row)
            logistic_rows.append(logistic_row)
            labels.append(1 if payout > 0 else 0)
            ticks.append(tick)

        target = np.asarray(labels, dtype=int)
        tick_values = np.asarray(ticks, dtype=float)
        if len(target) < self.min_rows or np.unique(target).size < 2:
            return None
        tree_features = np.asarray(tree_rows, dtype=float)
        logistic_features = np.asarray(logistic_rows, dtype=float)
        prior_win = self._prior_win()

        tree_probability, tree_diagnostics = self._select_family(
            "tree", tree_features, target, tick_values, tree_query,
            float(cutoff), prior_win,
        )
        logistic_probability: float | None = None
        logistic_diagnostics: dict[str, Any] | None = None
        if self.mix > 0.0:
            logistic_probability, logistic_diagnostics = self._select_family(
                "logistic", logistic_features, target, tick_values, logistic_query,
                float(cutoff), prior_win,
            )

        if self.mix <= 0.0 or logistic_probability is None:
            combined = tree_probability
        else:
            combined = self.mix * logistic_probability + (1.0 - self.mix) * tree_probability

        key = (cutoff, tuple(sorted(seq.items())))
        self.selection_diagnostics[key] = {
            "tree": tree_diagnostics,
            "logistic": logistic_diagnostics,
            "eligible_rows": len(target),
            "mix": self.mix,
            "short_alpha": self.short_alpha,
        }
        return self._postprocess_combo_probability(combined, seq, cutoff)
