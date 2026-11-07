"""E3: train-only ranking of lag-pair interactions for the E1 forecaster.

Each training machine gets a chronological tail holdout. A C=1 main-effects
logit is compared with 28 C=1 logits that add one pair at a time. Machine-level
Brier deltas are averaged equally, so each machine can learn a different sign.
On the exploratory TRAIN diagnostic (not a validation benchmark), the strongest
single-pair candidates were lags 2x3, 4x6, 3x4, 1x2, and 1x6; the gains were
small and noisy. Tick-keyed lag lookup is required because train histories have
gaps; absent ticks remain unusable.
"""

from itertools import combinations
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression

from .pair_logistic import PairLogisticForecaster, _lag_vector
from .schema import index_history, unique_queries


class TrainSelectedPairForecaster(PairLogisticForecaster):
    """E1 with a small set of lag interactions selected using training histories.

    Parameters match :class:`PairLogisticForecaster`, with ``top_k`` controlling
    how many ranked pairs each machine-specific prediction model includes.
    Pair ranking uses only ``history`` passed to ``fit``; conditions, signal,
    machine type, labels, and validation data do not enter the ranking score.
    """

    def __init__(self, top_k=5, lags=8, pairs=True, C=3.0, half_life=32.0,
                 min_rows=12, base_half_life=24.0, clip=1e-4,
                 ranking_C=1.0, min_train=32, min_test=8,
                 full_pairs_above_rows=None):
        super().__init__(lags=lags, pairs=pairs, C=C, half_life=half_life,
                         min_rows=min_rows, base_half_life=base_half_life,
                         clip=clip)
        self.top_k = int(top_k)
        self.ranking_C = float(ranking_C)
        self.min_train, self.min_test = int(min_train), int(min_test)
        self.full_pairs_above_rows = (None if full_pairs_above_rows is None
                                      else int(full_pairs_above_rows))
        if not 0 <= self.top_k <= self.lags * (self.lags - 1) // 2:
            raise ValueError("top_k must be between 0 and the number of lag pairs")
        if self.ranking_C <= 0 or self.min_train < 1 or self.min_test < 1:
            raise ValueError("ranking_C, min_train, and min_test must be positive")
        if self.full_pairs_above_rows is not None and self.full_pairs_above_rows < 1:
            raise ValueError("full_pairs_above_rows must be positive or None")
        self.all_pairs = tuple(combinations(range(self.lags), 2))
        self.selected_pairs = ()
        self.ranking_diagnostics = None

    @staticmethod
    def _interaction_features(x, pairs):
        if not pairs:
            return np.asarray(x, dtype=float)
        return np.concatenate([np.asarray(x, dtype=float),
                               [x[a] * x[b] for a, b in pairs]])

    def _pairs_for_rows(self, valid_rows):
        if (self.full_pairs_above_rows is not None
                and valid_rows >= self.full_pairs_above_rows):
            return self.all_pairs
        return self.selected_pairs

    def _rank_pairs(self, queries, history):
        query_map = unique_queries(queries)
        indexed = index_history(queries, history)
        candidates = tuple(combinations(range(self.lags), 2))
        deltas = [[] for _ in candidates]
        eligible = 0

        for query in query_map.values():
            plays = indexed.get((query.episode_id, query.machine_id), ())
            seq = {play.tick: play.payout for play in plays}
            ordered = sorted(seq)
            if not ordered:
                continue
            test_count = max(24, int(np.ceil(len(ordered) * 0.20)))
            split = len(ordered) - test_count
            xs, ys, positions = [], [], []
            for position, tick in enumerate(ordered):
                x = _lag_vector(seq, tick, self.lags)
                if x is None:
                    continue
                xs.append(x)
                ys.append(int(seq[tick] > 0))
                positions.append(position)
            if not xs:
                continue
            X = np.asarray(xs, dtype=float)
            y = np.asarray(ys, dtype=int)
            positions = np.asarray(positions)
            train = positions < split
            test = positions >= split
            if (train.sum() < self.min_train or test.sum() < self.min_test
                    or np.unique(y[train]).size < 2):
                continue

            base = self._fit_rank_logit(X[train], y[train])
            p0 = base.predict_proba(X[test])[:, 1]
            base_loss = float(np.mean((p0 - y[test]) ** 2))
            machine_deltas = []
            for pair in candidates:
                X_pair = np.column_stack((X, X[:, pair[0]] * X[:, pair[1]]))
                model = self._fit_rank_logit(X_pair[train], y[train])
                pred = model.predict_proba(X_pair[test])[:, 1]
                machine_deltas.append(
                    float(np.mean((pred - y[test]) ** 2) - base_loss))
            for pair_index, delta in enumerate(machine_deltas):
                deltas[pair_index].append(delta)
            eligible += 1

        means = [float(np.mean(values)) if values else float("inf")
                 for values in deltas]
        # Stable tuple ordering breaks exact ties lexicographically.
        order = sorted(range(len(candidates)), key=lambda i: (means[i], candidates[i]))
        self.selected_pairs = tuple(candidates[i] for i in order[:self.top_k]
                                    if np.isfinite(means[i]))
        self.ranking_diagnostics = {
            "machines_total": len(query_map),
            "machines_eligible": eligible,
            "pairs_scored": len(candidates),
            "mean_brier_delta": {candidates[i]: means[i] for i in range(len(candidates))},
            "selected_pairs": self.selected_pairs,
        }

    def _fit_rank_logit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return LogisticRegression(C=self.ranking_C, max_iter=500,
                                      solver="liblinear").fit(X, y)

    def fit(self, queries, history, labels=None):
        queries, history = tuple(queries), tuple(history)
        super().fit(queries, history, labels)
        self._rank_pairs(queries, history)
        return self

    def win_probability(self, seq, cutoff):
        """Fit selected interactions on observations strictly before cutoff."""
        xq = _lag_vector(seq, cutoff, self.lags)
        if xq is None:
            return None
        lag_rows, ys, weights = [], [], []
        for tick, payout in seq.items():
            if tick >= cutoff:
                continue
            x = _lag_vector(seq, tick, self.lags)
            if x is None:
                continue
            lag_rows.append(x)
            ys.append(int(payout > 0))
            age = cutoff - 1 - tick
            weights.append(1.0 if self.half_life is None
                           else 2.0 ** (-age / self.half_life))
        if len(lag_rows) < self.min_rows or len(set(ys)) < 2:
            return None
        active_pairs = self._pairs_for_rows(len(lag_rows))
        rows = [self._interaction_features(x, active_pairs) for x in lag_rows]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LogisticRegression(C=self.C, max_iter=1000)
            model.fit(np.asarray(rows), ys, sample_weight=np.asarray(weights))
        x_query = self._interaction_features(xq, active_pairs)
        return float(model.predict_proba(x_query[None, :])[0, 1])
