"""E6: per-machine logistic forecaster with configurable lag interactions.

Degree 1 uses main lag features, degree 2 adds every pair product, and degree 3
adds every triple product. Binary encoding matches E1's 0/1 win indicators;
signed encoding maps losses/wins to -1/+1 before making products.
"""
from itertools import combinations
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression

from .pair_logistic import PairLogisticForecaster, _lag_vector


class InteractionSweepForecaster(PairLogisticForecaster):
    """E1 shape head with selectable per-machine logit interaction degree."""

    def __init__(self, degree=2, encoding='binary', family='all', lags=8,
                 C=3.0, half_life=32.0, min_rows=12,
                 base_half_life=24.0, clip=1e-4, penalty='l2'):
        # This class owns feature expansion rather than E1's pairs switch.
        super().__init__(lags=lags, pairs=False, C=C, half_life=half_life,
                         min_rows=min_rows, base_half_life=base_half_life,
                         clip=clip)
        self.degree = int(degree)
        self.encoding = str(encoding)
        self.family = str(family)
        self.penalty = str(penalty)
        if self.degree not in (1, 2, 3):
            raise ValueError('degree must be 1, 2, or 3')
        if self.encoding not in ('binary', 'signed'):
            raise ValueError("encoding must be 'binary' or 'signed'")
        if self.family != 'all':
            raise ValueError("family currently supports only 'all'")
        if self.penalty not in ('l1', 'l2'):
            raise ValueError("penalty must be 'l1' or 'l2'")
        self._interaction_indices = {
            size: np.asarray(list(combinations(range(self.lags), size)), dtype=int)
            for size in range(2, self.degree + 1)
        }
        self.diagnostics = {
            'successful_fits': 0,
            'max_iter_hits': 0,
            'fallbacks': 0,
        }

    def feature_matrix(self, X):
        """Expand a batch of main lag vectors using vectorized interaction blocks."""
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.lags:
            raise ValueError(f'X must have shape (n_rows, {self.lags})')
        if self.encoding == 'signed':
            values = 2.0 * values - 1.0
        blocks = [values]
        for size in range(2, self.degree + 1):
            indices = self._interaction_indices[size]
            blocks.append(values[:, indices].prod(axis=2))
        return np.concatenate(blocks, axis=1)

    def feature_vector(self, x):
        """Expand one lag vector through the same batch implementation."""
        values = np.asarray(x, dtype=float).reshape(1, -1)
        return self.feature_matrix(values)[0]

    def predict(self, queries, history):
        """Reset per-predict fit diagnostics before using the inherited path."""
        self.diagnostics = {
            'successful_fits': 0,
            'max_iter_hits': 0,
            'fallbacks': 0,
        }
        return super().predict(queries, history)

    def win_probability(self, seq, cutoff):
        """Fit on exact-lag rows strictly before cutoff and predict that tick."""
        xq = _lag_vector(seq, cutoff, self.lags)
        if xq is None:
            self.diagnostics['fallbacks'] += 1
            return None
        rows, ys, weights = [], [], []
        for tick, payout in seq.items():
            if tick >= cutoff:
                continue
            x = _lag_vector(seq, tick, self.lags)
            if x is None:
                continue
            rows.append(x)
            ys.append(int(payout > 0))
            age = cutoff - 1 - tick
            weights.append(1.0 if self.half_life is None
                           else 2.0 ** (-age / self.half_life))
        if len(rows) < self.min_rows or len(set(ys)) < 2:
            self.diagnostics['fallbacks'] += 1
            return None
        features = self.feature_matrix(np.asarray(rows, dtype=float))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if self.penalty == 'l1':
                model = LogisticRegression(
                    C=self.C, penalty='l1', solver='liblinear',
                    intercept_scaling=100, random_state=0, max_iter=1000,
                )
            else:
                model = LogisticRegression(C=self.C, max_iter=1000)
            model.fit(features, ys, sample_weight=np.asarray(weights))
        self.diagnostics['successful_fits'] += 1
        if np.max(model.n_iter_) >= model.max_iter:
            self.diagnostics['max_iter_hits'] += 1
        return float(model.predict_proba(self.feature_vector(xq)[None, :])[0, 1])
