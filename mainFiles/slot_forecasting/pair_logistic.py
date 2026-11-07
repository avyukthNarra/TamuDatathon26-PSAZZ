"""E1: per-machine win-chance model with lag-pair interactions and recency weighting.

Why: ~94% of the starter's error is the win chance p0 = P(payout == 0), and the
true p0 depends on machine-specific interactions between the last eight spins
(pooled lags explain only ~20% of it). So each query's machine gets its own
L2-regularised logistic regression for P(win) fitted on its own pre-cutoff
history, with recent spins weighted more (drift / shared change).

Payout shape: P(payout = k | win) is taken from the kit's pooled_recent
forecast, whose positive part is machine-specific counts shrunk to a
type|signal prior. E5 refines this.

Leakage rules: training rows for a machine are ticks t < cutoff whose lags
t-1..t-L are all observed (tick lags, not row offsets; missing ticks are
unknown and simply make that row unusable). The query is predicted from the
L ticks before its cutoff.
"""
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression

from .baseline import FrequencyForecaster
from .schema import index_history, unique_queries


def _lag_vector(seq, tick, lags):
    values = [seq.get(tick - lag) for lag in range(1, lags + 1)]
    if any(v is None for v in values):
        return None
    return np.array([1.0 if v > 0 else 0.0 for v in values])


def _expand(x, pairs):
    if not pairs:
        return x
    n = len(x)
    return np.concatenate([x, [x[a] * x[b] for a in range(n) for b in range(a + 1, n)]])


class PairLogisticForecaster:
    def __init__(self, lags=8, pairs=True, C=3.0, half_life=32.0, min_rows=12,
                 base_half_life=24.0, clip=1e-4):
        self.lags, self.pairs, self.C = int(lags), bool(pairs), float(C)
        self.half_life = None if half_life is None else float(half_life)
        self.min_rows, self.clip = int(min_rows), float(clip)
        self.base = FrequencyForecaster("pooled_recent", half_life=base_half_life)

    def fit(self, queries, history, labels=None):
        self.base.fit(queries, history, labels)
        return self

    def win_probability(self, seq, cutoff):
        """P(win) for the spin at `cutoff`, or None when the machine can't support a fit."""
        xq = _lag_vector(seq, cutoff, self.lags)
        if xq is None:
            return None
        rows, ys, ws = [], [], []
        for tick, pay in seq.items():
            x = _lag_vector(seq, tick, self.lags)
            if x is None:
                continue
            rows.append(_expand(x, self.pairs))
            ys.append(1 if pay > 0 else 0)
            age = cutoff - 1 - tick
            ws.append(1.0 if self.half_life is None else 2.0 ** (-age / self.half_life))
        if len(rows) < self.min_rows or len(set(ys)) < 2:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LogisticRegression(C=self.C, max_iter=1000)
            model.fit(np.array(rows), ys, sample_weight=np.array(ws))
        return float(model.predict_proba(_expand(xq, self.pairs)[None])[0, 1])

    def predict(self, queries, history):
        queries, history = tuple(queries), tuple(history)
        base = self.base.predict(queries, history)
        indexed = index_history(queries, history)
        out = {}
        for query in sorted(unique_queries(queries).values(), key=lambda q: q.query_id):
            b = base[query.query_id]
            seq = {p.tick: p.payout for p in indexed.get((query.episode_id, query.machine_id), [])}
            pw = self.win_probability(seq, query.cutoff)
            if pw is None:
                out[query.query_id] = tuple(b)
                continue
            pw = min(max(pw, self.clip), 1.0 - self.clip)
            pos = np.array(b[1:]) / sum(b[1:])
            probs = np.concatenate([[1.0 - pw], pw * pos])
            out[query.query_id] = tuple(float(v) for v in probs / probs.sum())
        return out
