"""Contest scorer (per-condition loss ratios vs the frozen reference, points = max/(1+R)),
episode-level bootstrap CI, per-class error shares, and the reserved audit set.

Mirrors slot_forecasting.contest.contest_score; selfcheck.py cross-checks the two.
Scopes: "dev" = 72 validation episodes (tune here), "audit" = 24 reserved (6 per condition),
"all" = 96. Touching "audit"/"all" requires who=<id> and is appended to results/audit_access.log.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .data import CONDITIONS, PC

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
AUDIT_PER_CONDITION = 6
BOOT_SEED = 314159


def audit_episodes(split):
    """Frozen audit set: first 6 episodes per condition by sha256('audit-v1|episode'). Saved on first use."""
    path = HERE / f"audit_{split.round}.json"
    if path.is_file():
        return set(json.loads(path.read_text())["episodes"])
    by = {c: [] for c in CONDITIONS}
    for e in split.episodes:
        by[split.condition[split.episode_rows(e)[0]]].append(e)
    chosen = []
    for c in CONDITIONS:
        by[c].sort(key=lambda e: hashlib.sha256(f"audit-v1|{e}".encode()).hexdigest())
        chosen += by[c][:AUDIT_PER_CONDITION]
    path.write_text(json.dumps({"rule": "first 6 per condition by sha256('audit-v1|'+episode_id)",
                                "episodes": sorted(chosen)}, indent=1))
    return set(chosen)


def scope_episodes(split, scope):
    a = audit_episodes(split)
    if scope == "audit":
        return [e for e in split.episodes if e in a]
    if scope == "dev":
        return [e for e in split.episodes if e not in a]
    if scope == "all":
        return list(split.episodes)
    raise ValueError("scope must be dev|audit|all")


def _guard(scope, who):
    if scope in ("audit", "all"):
        if not who:
            raise PermissionError("audit/all scope needs who=<method id>; it is logged")
        RESULTS.mkdir(exist_ok=True)
        with open(RESULTS / "audit_access.log", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{scope}\t{who}\n")


def _prep(P, split):
    P = np.asarray(P, float)
    if P.shape != (split.n, 5) or not np.isfinite(P).all() or (P < 0).any() or (P > 1).any():
        raise ValueError("predictions must be finite (n,5) in [0,1]")
    s = P.sum(1, keepdims=True)
    if np.abs(s - 1).max() > 1e-6:
        raise ValueError("rows must sum to 1 within 1e-6")
    return P / s


def _episode_table(P, split, scope):
    """Per-episode mean error and reference error, grouped by condition."""
    err = ((P - split.truth) ** 2).sum(1)
    base = ((split.ref - split.truth) ** 2).sum(1)
    eps = scope_episodes(split, scope)
    cond, e, b = [], [], []
    for ep in eps:
        r = split.episode_rows(ep)
        cond.append(CONDITIONS.index(split.condition[r[0]]))
        e.append(err[r].mean())
        b.append(base[r].mean())
    return np.array(cond), np.array(e), np.array(b)


def _points(cond, e, b, max_points, idx=None):
    if idx is not None:
        cond, e, b = cond[idx], e[idx], b[idx]
    ratios = []
    for c in range(len(CONDITIONS)):
        m = cond == c
        ratios.append(e[m].mean() / max(b[m].mean(), 1e-9))
    R = float(np.mean(ratios))
    return max_points / (1 + R), ratios, R


def _resamples(cond, reps, seed=BOOT_SEED):
    """Same index draws for every method, stratified by condition (episodes are the unit)."""
    rng = np.random.RandomState(seed)
    groups = [np.flatnonzero(cond == c) for c in range(len(CONDITIONS))]
    return [np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(reps)]


def evaluate(P, split, scope="dev", who=None, bootstrap=1000):
    """Return points, per-condition ratios, CI95 on points, R, mean 5-class error, class shares."""
    _guard(scope, who)
    P = _prep(P, split)
    cond, e, b = _episode_table(P, split, scope)
    pts, ratios, R = _points(cond, e, b, split.max_points)
    out = {"scope": scope, "points": float(pts), "R": R, "n_episodes": int(len(e)),
           "condition_ratios": {c: float(r) for c, r in zip(CONDITIONS, ratios)},
           "mean_error": float(e.mean())}
    if bootstrap:
        boots = [_points(cond, e, b, split.max_points, idx)[0] for idx in _resamples(cond, bootstrap)]
        out["ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    out["class_error_shares"] = class_shares(P, split, scope)
    return out


def diff_ci(Pa, Pb, split, scope="dev", who=None, bootstrap=1000):
    """Paired bootstrap of points(A) - points(B). 'CI over B' means the lower bound > 0."""
    _guard(scope, who)
    Pa, Pb = _prep(Pa, split), _prep(Pb, split)
    ca, ea, ba = _episode_table(Pa, split, scope)
    _, eb, bb = _episode_table(Pb, split, scope)
    d = _points(ca, ea, ba, split.max_points)[0] - _points(ca, eb, bb, split.max_points)[0]
    boots = [_points(ca, ea, ba, split.max_points, i)[0] - _points(ca, eb, bb, split.max_points, i)[0]
             for i in _resamples(ca, bootstrap)]
    return {"diff": float(d), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def class_shares(P, split, scope="dev"):
    rows = np.concatenate([split.episode_rows(e) for e in scope_episodes(split, scope)])
    se = ((np.asarray(P)[rows] - split.truth[rows]) ** 2).mean(0)
    return {k: float(v / se.sum()) for k, v in zip(PC, se)}
