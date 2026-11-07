"""Stack-member test: does method X add anything to base B (default B3)? Episode-OOF convex blend."""
import hashlib

import numpy as np

from .data import CONDITIONS
from .scoring import _prep, diff_ci, evaluate, scope_episodes


def blend_gain(Pm, Pb, split, scope="dev", folds=6, grid=21, bootstrap=1000):
    """Per-fold blend weight w (0..1) fit on the other folds' episodes to minimise squared error to the
    true q; evaluate the out-of-fold blend. Returns OOF predictions, mean w, points, diff vs Pb with CI."""
    Pm, Pb = _prep(Pm, split), _prep(Pb, split)
    eps = scope_episodes(split, scope)
    order = sorted(eps, key=lambda e: hashlib.sha256(f"fold-v1|{e}".encode()).hexdigest())
    fold = {e: i % folds for i, e in enumerate(order)}
    rows = {e: split.episode_rows(e) for e in eps}
    ws = np.linspace(0, 1, grid)
    out = Pb.copy()
    used = []
    for f in range(folds):
        tr = np.concatenate([rows[e] for e in eps if fold[e] != f])
        losses = [(((w * Pm[tr] + (1 - w) * Pb[tr]) - split.truth[tr]) ** 2).sum(1).mean() for w in ws]
        w = ws[int(np.argmin(losses))]
        used.append(float(w))
        for e in eps:
            if fold[e] == f:
                out[rows[e]] = w * Pm[rows[e]] + (1 - w) * Pb[rows[e]]
    ev = evaluate(out, split, scope, bootstrap=bootstrap)
    return {"oof": out, "w_per_fold": used, "w_mean": float(np.mean(used)), "points": ev["points"],
            "ci95": ev.get("ci95"), "condition_ratios": ev["condition_ratios"],
            "vs_base": diff_ci(out, Pb, split, scope, bootstrap=bootstrap)}
