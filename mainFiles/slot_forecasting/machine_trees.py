"""E10: per-machine tree ensembles for P(win) instead of logistic regression.

Trees can represent sharp higher-order lag rules (three-way and more) that
the pairwise logistic model can't. Inputs are raw win indicators at tick
lags 1..8; the trees find the interactions themselves. Rows are weighted by
the same recency half-life. Optionally averaged with the logistic model.
"""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from .pair_logistic import _lag_vector
from .shape_head import ShapeHeadForecaster


class MachineTreeForecaster(ShapeHeadForecaster):
    def __init__(self, kind="extra", n_estimators=100, min_leaf=4, mix=0.0, seed=0, **kw):
        kw.setdefault("C", 10.0)
        kw.setdefault("shape_strength", 50)
        super().__init__(**kw)
        self.kind, self.n_estimators, self.min_leaf = kind, int(n_estimators), int(min_leaf)
        self.mix, self.seed = float(mix), int(seed)

    def win_probability(self, seq, cutoff):
        xq = _lag_vector(seq, cutoff, self.lags)
        if xq is None:
            return None
        rows, ys, ws = [], [], []
        for tick, pay in seq.items():
            if tick >= cutoff:
                continue
            x = _lag_vector(seq, tick, self.lags)
            if x is None:
                continue
            rows.append(x); ys.append(1 if pay > 0 else 0)
            age = cutoff - 1 - tick
            ws.append(1.0 if self.half_life is None else 2.0 ** (-age / self.half_life))
        if len(rows) < self.min_rows or len(set(ys)) < 2:
            return None
        cls = ExtraTreesClassifier if self.kind == "extra" else RandomForestClassifier
        m = cls(n_estimators=self.n_estimators, min_samples_leaf=self.min_leaf,
                random_state=self.seed, n_jobs=1).fit(np.array(rows), ys, sample_weight=np.array(ws))
        p_tree = float(m.predict_proba(xq[None])[0, 1])
        if self.mix <= 0:
            return p_tree
        p_lr = super().win_probability(seq, cutoff)
        return p_tree if p_lr is None else self.mix * p_lr + (1 - self.mix) * p_tree
