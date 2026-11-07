"""Per-machine selection of history memory for a binary win head."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class _Candidate:
    half_life: int | None
    c: float

    @property
    def name(self) -> str:
        memory = "none" if self.half_life is None else str(self.half_life)
        return f"half_life={memory},C={self.c:g}"


class AdaptiveMemorySelector:
    """Choose recency weighting for one machine using prequential loss.

    ``X`` must contain features that are already causal at each row. ``ticks``
    are actual tick numbers, so gaps remain gaps when recency weights are made.
    """

    def __init__(
        self,
        half_lives: tuple[int | None, ...] = (None, 16, 32, 64),
        Cs: tuple[float, ...] = (1.0, 3.0),
        *,
        mode: str = "hard",
        temperature: float = 10.0,
        max_eval_rows: int = 32,
        eval_stride: int = 4,
        min_train: int = 16,
    ) -> None:
        if mode not in {"hard", "soft"}:
            raise ValueError("mode must be 'hard' or 'soft'")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not half_lives or not Cs or any(c <= 0 for c in Cs):
            raise ValueError("half_lives and positive Cs must be non-empty")
        self.candidates = tuple(
            _Candidate(hl, float(c))
            for hl in half_lives
            for c in Cs
        )
        self.mode = mode
        self.temperature = float(temperature)
        self.max_eval_rows = int(max_eval_rows)
        self.eval_stride = int(eval_stride)
        self.min_train = int(min_train)

    @staticmethod
    def _prior(y: np.ndarray, prior_win: float) -> float:
        # Two prior pseudo-observations keep short and one-class histories stable.
        return float((float(y.sum()) + 2.0 * prior_win) / (len(y) + 2.0))

    @staticmethod
    def _weights(ticks: np.ndarray, cutoff: float, half_life: int | None) -> np.ndarray:
        if half_life is None:
            return np.ones(len(ticks), dtype=float)
        return np.exp2(-(cutoff - 1.0 - ticks.astype(float)) / half_life)

    def _fit_probability(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ticks: np.ndarray,
        query_features: np.ndarray,
        cutoff: float,
        candidate: _Candidate,
        prior_win: float,
    ) -> tuple[float, np.ndarray]:
        weights = self._weights(ticks, cutoff, candidate.half_life)
        if len(y) < self.min_train or np.unique(y).size < 2:
            return self._prior(y, prior_win), weights
        model = LogisticRegression(C=candidate.c, max_iter=500)
        model.fit(X, y, sample_weight=weights)
        probability = float(model.predict_proba(query_features.reshape(1, -1))[0, 1])
        return probability, weights

    def predict_win(
        self,
        X: Any,
        y: Any,
        ticks: Any,
        query_features: Any,
        cutoff: float,
        prior_win: float = 0.4,
    ) -> tuple[float, dict[str, Any]]:
        """Predict win probability and return model-selection diagnostics."""
        features = np.asarray(X, dtype=float)
        labels = np.asarray(y, dtype=int).reshape(-1)
        tick_values = np.asarray(ticks, dtype=float).reshape(-1)
        query = np.asarray(query_features, dtype=float).reshape(-1)
        if features.ndim != 2 or len(features) != len(labels) or len(labels) != len(tick_values):
            raise ValueError("X, y, and ticks must have matching row counts; X must be 2D")
        if features.shape[1] != len(query):
            raise ValueError("query_features must match the feature width of X")
        if np.any((labels != 0) & (labels != 1)):
            raise ValueError("y must contain binary labels")
        if np.any(np.diff(tick_values) < 0):
            raise ValueError("ticks must be sorted")
        if not 0.0 <= prior_win <= 1.0:
            raise ValueError("prior_win must be in [0, 1]")

        eligible = tick_values < cutoff
        train_X, train_y, train_ticks = features[eligible], labels[eligible], tick_values[eligible]
        eval_indices = np.flatnonzero(eligible)
        eval_indices = eval_indices[-self.max_eval_rows :: self.eval_stride]
        losses: dict[str, list[float]] = {c.name: [] for c in self.candidates}
        scored_ticks: list[float] = []

        # Each evaluated outcome is predicted using only rows strictly before its tick.
        for index in eval_indices:
            prefix = np.flatnonzero(tick_values < tick_values[index])
            if len(prefix) < self.min_train:
                continue
            scored_ticks.append(float(tick_values[index]))
            for candidate in self.candidates:
                probability, _ = self._fit_probability(
                    features[prefix], labels[prefix], tick_values[prefix],
                    features[index], tick_values[index], candidate, prior_win,
                )
                probability = float(np.clip(probability, 1e-6, 1 - 1e-6))
                target = int(labels[index])
                losses[candidate.name].append(
                    -(target * log(probability) + (1 - target) * log(1 - probability))
                )

        mean_losses = {
            name: (float(np.mean(values)) if values else None)
            for name, values in losses.items()
        }
        # Candidate declaration order favors no decay, then longer half-lives;
        # a stable sort therefore resolves exact ties conservatively.
        available = [c for c in self.candidates if mean_losses[c.name] is not None]
        if self.mode == "hard" and available:
            best_loss = min(mean_losses[c.name] for c in available)
            chosen = next(c for c in available if mean_losses[c.name] == best_loss)
            selected = {chosen.name: 1.0}
        elif self.mode == "soft" and available:
            best_loss = min(mean_losses[c.name] for c in available)
            scores = {
                c.name: exp(-(mean_losses[c.name] - best_loss) * self.temperature)
                for c in available
            }
            total = sum(scores.values())
            selected = {name: score / total for name, score in scores.items()}
        else:
            # With too little history, preserve the same conservative ordering.
            selected = {self.candidates[0].name: 1.0}

        final_probabilities: dict[str, float] = {}
        final_weights: dict[str, list[float]] = {}
        for candidate in self.candidates:
            if candidate.name not in selected:
                continue
            p, weights = self._fit_probability(
                train_X, train_y, train_ticks, query, cutoff,
                candidate, prior_win,
            )
            final_probabilities[candidate.name] = float(np.clip(p, 1e-6, 1 - 1e-6))
            final_weights[candidate.name] = weights.tolist()
        probability = sum(
            weight * final_probabilities[name] for name, weight in selected.items()
        )
        diagnostics: dict[str, Any] = {
            "mode": self.mode,
            "selected_weights": selected,
            "mean_logloss": mean_losses,
            "prequential_ticks": scored_ticks,
            "eligible_rows": len(train_y),
            "final_probabilities": final_probabilities,
            "final_sample_weights": final_weights,
            "fallback": len(train_y) < self.min_train or np.unique(train_y).size < 2,
        }
        return float(np.clip(probability, 1e-6, 1 - 1e-6)), diagnostics
