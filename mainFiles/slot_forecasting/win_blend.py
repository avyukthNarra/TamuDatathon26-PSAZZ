"""E9: blend or temperature-scale the per-machine win probability.

A per-machine fit on ~30-200 rows is noisy, so P(win) is post-processed
with one or two global knobs, which are chosen on tuning episodes:
- alpha: pw = alpha * model + (1 - alpha) * pooled_recent's P(win)
- temp:  pw = sigmoid(logit(model) / temp); temp < 1 sharpens, temp > 1 softens
- short_alpha: a separate alpha for machines with < 100 observed spins
  (short histories), using observation count, which is visible at test time.
"""
import math

from .shape_head import ShapeHeadForecaster


class WinBlendForecaster(ShapeHeadForecaster):
    def __init__(self, alpha=1.0, temp=1.0, short_alpha=None, **kw):
        kw.setdefault("C", 10.0)
        kw.setdefault("shape_strength", 50)
        super().__init__(**kw)
        self.alpha, self.temp = float(alpha), float(temp)
        self.short_alpha = None if short_alpha is None else float(short_alpha)
        self._base_pw = None

    def win_probability(self, seq, cutoff):
        pw = super().win_probability(seq, cutoff)
        if pw is None:
            return None
        pw = min(max(pw, 1e-6), 1 - 1e-6)
        if self.temp != 1.0:
            z = math.log(pw / (1 - pw)) / self.temp
            pw = 1 / (1 + math.exp(-z))
        a = self.alpha
        if self.short_alpha is not None and sum(1 for t in seq if t < cutoff) < 100:
            a = self.short_alpha
        return a * pw + (1 - a) * self._base_pw

    def predict(self, queries, history):
        # Make the base P(win) for the current query available to win_probability.
        queries, history = tuple(queries), tuple(history)
        base = self.base.predict(queries, history)
        from .schema import index_history, unique_queries
        import numpy as np
        indexed = index_history(queries, history)
        out = {}
        for q in sorted(unique_queries(queries).values(), key=lambda q: q.query_id):
            seq = {p.tick: p.payout for p in indexed.get((q.episode_id, q.machine_id), [])}
            self._base_pw = 1.0 - base[q.query_id][0]
            pw = self.win_probability(seq, q.cutoff)
            if pw is None:
                pw = self._base_pw
            pw = min(max(pw, self.clip), 1.0 - self.clip)
            probs = np.concatenate([[1.0 - pw], pw * self.shape(q, seq)])
            out[q.query_id] = tuple(float(v) for v in probs / probs.sum())
        return out
