"""E5: payout-shape head and payout-size inputs on top of the E1 win model.

Shape = P(payout = k | win), k in (1, 2, 5, 20). The positive part of
pooled_recent shrinks machine counts toward a five-category prior, which
mixes win-rate evidence into the shape. Here the shape is estimated
separately: a type|signal positive-payout prior fitted on train history
(plus train labels), blended with the machine's own recency-weighted
positive spins.

`size_inputs=True` also feeds "big win" (payout >= 2) indicators at lags 1-8
into the win model, since the rules say win sizes can matter too.
"""
from collections import defaultdict

import numpy as np

from .pair_logistic import PairLogisticForecaster, _expand
from .schema import index_history, unique_queries

POSITIVE = (1, 2, 5, 20)


class ShapeHeadForecaster(PairLogisticForecaster):
    def __init__(self, shape_strength=20.0, shape_half_life=None, size_inputs=False, **kw):
        super().__init__(**kw)
        self.shape_strength = float(shape_strength)
        self.shape_half_life = None if shape_half_life is None else float(shape_half_life)
        self.size_inputs = bool(size_inputs)
        self.shape_prior = {}

    def fit(self, queries, history, labels=None):
        super().fit(queries, history, labels)
        queries = tuple(queries)
        indexed = index_history(queries, history)
        counts = defaultdict(lambda: np.full(4, 0.5))
        for q in unique_queries(queries).values():
            outs = [p.payout for p in indexed.get((q.episode_id, q.machine_id), [])]
            if labels is not None:
                outs.append(int(labels[q.query_id]))
            for o in outs:
                if o > 0:
                    counts[f"{q.machine_type}|{q.signal}"][POSITIVE.index(o)] += 1
        self.shape_prior = {k: v / v.sum() for k, v in counts.items()}
        return self

    def _features(self, seq, tick):
        values = [seq.get(tick - lag) for lag in range(1, self.lags + 1)]
        if any(v is None for v in values):
            return None
        win = np.array([1.0 if v > 0 else 0.0 for v in values])
        x = _expand(win, self.pairs)
        if self.size_inputs:
            x = np.concatenate([x, [1.0 if v >= 2 else 0.0 for v in values]])
        return x

    def win_probability(self, seq, cutoff):
        if not self.size_inputs:
            return super().win_probability(seq, cutoff)
        import warnings
        from sklearn.linear_model import LogisticRegression
        xq = self._features(seq, cutoff)
        if xq is None:
            return None
        rows, ys, ws = [], [], []
        for tick, pay in seq.items():
            if tick >= cutoff:
                continue
            x = self._features(seq, tick)
            if x is None:
                continue
            rows.append(x); ys.append(1 if pay > 0 else 0)
            age = cutoff - 1 - tick
            ws.append(1.0 if self.half_life is None else 2.0 ** (-age / self.half_life))
        if len(rows) < self.min_rows or len(set(ys)) < 2:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LogisticRegression(C=self.C, max_iter=1000).fit(np.array(rows), ys, sample_weight=np.array(ws))
        return float(m.predict_proba(xq[None])[0, 1])

    def shape(self, query, seq):
        prior = self.shape_prior.get(f"{query.machine_type}|{query.signal}", np.full(4, 0.25))
        c = self.shape_strength * prior
        for tick, pay in seq.items():
            if pay > 0 and tick < query.cutoff:
                age = query.cutoff - 1 - tick
                c[POSITIVE.index(pay)] += 1.0 if self.shape_half_life is None else 2.0 ** (-age / self.shape_half_life)
        return c / c.sum()

    def predict(self, queries, history):
        queries, history = tuple(queries), tuple(history)
        base = self.base.predict(queries, history)
        indexed = index_history(queries, history)
        out = {}
        for q in sorted(unique_queries(queries).values(), key=lambda q: q.query_id):
            seq = {p.tick: p.payout for p in indexed.get((q.episode_id, q.machine_id), [])}
            pw = self.win_probability(seq, q.cutoff)
            if pw is None:
                pw = 1.0 - base[q.query_id][0]
            pw = min(max(pw, self.clip), 1.0 - self.clip)
            probs = np.concatenate([[1.0 - pw], pw * self.shape(q, seq)])
            out[q.query_id] = tuple(float(v) for v in probs / probs.sum())
        return out
