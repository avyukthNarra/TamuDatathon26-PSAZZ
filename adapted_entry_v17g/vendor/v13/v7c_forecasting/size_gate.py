"""Train-only, episode-cross-fitted choice between the new size rule and v6.

The gate compares conditional payout-size distributions at win ticks. Its two
folds are fixed by sorted episode id; each evaluated episode is excluded from
both the atom fit and the v6 size-prior fit. No validation inputs are used.
"""

import math
from collections import defaultdict

from .pair_mixer import CLASS_INDEX, adapt
from .schema import unique_queries
from .size_rule import SizeRule

MIN_EPISODES = 8
Z_VALUE = 1.96
_WIN_CLASS = {1: 0, 2: 1, 5: 2, 20: 3}


def _normalized(values):
    values = [float(value) for value in values]
    total = math.fsum(values)
    if (len(values) != 4 or any(not math.isfinite(value) or value < 0 for value in values)
            or not math.isfinite(total) or total <= 0 or abs(total - 1.0) > 1e-6):
        raise ValueError("size gate received an invalid conditional distribution")
    return [value / total for value in values]


def _brier(probabilities, outcome):
    return math.fsum((p - (1.0 if i == outcome else 0.0)) ** 2
                     for i, p in enumerate(probabilities))


def _logloss(probabilities, outcome):
    return -math.log(max(1e-15, probabilities[outcome]))


def _size_modes(rule):
    """V6 baseline gets only the atoms and type priors learned in this fold."""
    return {machine_type: {"modes": [list(atom) for atom in atoms],
                           "weights": [0.5, 0.5]}
            for machine_type, atoms in rule.atoms.items()}


def _conditional(probabilities):
    if (len(probabilities) != 5
            or any(not math.isfinite(float(p)) or p < 0 for p in probabilities)
            or abs(math.fsum(probabilities) - 1.0) > 1e-6):
        raise ValueError("v6 size head emitted an invalid five-class distribution")
    win_mass = math.fsum(probabilities[1:])
    if win_mass <= 0:
        raise ValueError("v6 size head emitted zero conditional win mass")
    return _normalized([value / win_mass for value in probabilities[1:]])


def _fold_predict(rule, baseline, held_queries, held_history, labels):
    by_machine = defaultdict(list)
    for play in held_history:
        by_machine[(play.episode_id, play.machine_id)].append(play)
    scored = []
    for query in held_queries:
        plays = sorted(by_machine[(query.episode_id, query.machine_id)], key=lambda p: p.tick)
        pay = [None] * (query.cutoff + 1)
        own = [None] * (query.cutoff + 1)
        for play in plays:
            if play.tick < query.cutoff:
                pay[play.tick] = play.payout
                own[play.tick] = CLASS_INDEX[play.payout]
        target = labels.get(query.query_id)
        if target in _WIN_CLASS:
            pay[query.cutoff] = target
            own[query.cutoff] = CLASS_INDEX[target]

        candidate_at = dict(rule.prequential(pay, query.machine_type))
        baseline_at = {}
        baseline.walk(own, query.machine_type, query.cutoff,
                      lambda tick, probs: baseline_at.__setitem__(tick, _conditional(probs)))

        observed = [(play.tick, play.payout) for play in plays
                    if play.tick < query.cutoff and play.payout in _WIN_CLASS]
        # The optional cutoff target is forecast before it is absorbed by either head.
        if target in _WIN_CLASS:
            observed.append((query.cutoff, target))
        for tick, payout in observed:
            outcome = _WIN_CLASS[payout]
            if tick not in candidate_at or tick not in baseline_at:
                raise ValueError(f"size gate prediction stream omitted scored tick {tick}")
            new_mix = _normalized(candidate_at[tick])
            v6_mix = _normalized(baseline_at[tick])
            scored.append((query.episode_id, _brier(new_mix, outcome),
                           _brier(v6_mix, outcome), _logloss(new_mix, outcome),
                           _logloss(v6_mix, outcome)))
    return scored


def fit_size_gate(queries, history, labels=None, *, min_episodes=MIN_EPISODES):
    """Fit and evaluate a fixed 2-fold causal gate using TRAIN data only.

    Args:
        queries/history: training ``Query``/``Play`` records.
        labels: optional ``query_id -> raw payout`` mapping. When a held-out
            cutoff label is supplied, both predictions are emitted before that
            payout is absorbed; it is never included in either fold fit.
        min_episodes: fixed-support guard (production default 8).

    Returns a dict with ``use_candidate`` and JSON-serializable diagnostics.
    A positive Brier delta means candidate minus v6. The rule is selected only
    when at least ``min_episodes`` contribute wins and mean + 1.96 episode-SE
    is strictly below zero. Otherwise the gate safely defaults to v6.
    """
    labels = labels or {}
    query_map = unique_queries(queries)
    episodes = sorted({q.episode_id for q in query_map.values()})
    episode_delta = defaultdict(list)
    all_scores = []
    folds = []
    for fold_index in (0, 1):
        held_ids = {episode for index, episode in enumerate(episodes)
                    if index % 2 == fold_index}
        fit_queries = tuple(q for q in query_map.values() if q.episode_id not in held_ids)
        held_queries = tuple(q for q in query_map.values() if q.episode_id in held_ids)
        fit_history = tuple(p for p in history if p.episode_id not in held_ids)
        held_history = tuple(p for p in history if p.episode_id in held_ids)
        if not fit_queries or not held_queries:
            continue

        # Both heads receive only the complementary episode fold.
        try:
            fitted_rule = SizeRule.learn(fit_queries, fit_history)
            baseline = adapt(fit_queries, fit_history, labels=None, log=lambda _message: None,
                             size_modes=_size_modes(fitted_rule), round2=True)
            scores = _fold_predict(fitted_rule, baseline, held_queries, held_history, labels)
        except Exception as exc:
            return {"use_candidate": False,
                    "diagnostics": {"criterion": "conditional_size_brier_episode_mean_plus_1.96_se_lt_0",
                                    "chosen_head": "v6_size_head", "reason": str(exc),
                                    "min_episodes": int(min_episodes), "z_value": Z_VALUE,
                                    "n_episodes": 0, "n_wins": 0, "folds": folds}}
        all_scores.extend(scores)
        fold_wins = 0
        for episode, new_brier, v6_brier, new_ll, v6_ll in scores:
            episode_delta[episode].append(new_brier - v6_brier)
            fold_wins += 1
        folds.append({"fold": fold_index,
                      "fit_episodes": len({q.episode_id for q in fit_queries}),
                      "heldout_episodes": len({q.episode_id for q in held_queries}),
                      "scored_wins": fold_wins,
                      "size_lags": list(fitted_rule.lags),
                      "static_prior": float(fitted_rule.static_prior),
                      "v6_type_size_priors": {
                          key: list(value) for key, value in baseline.type_size_priors.items()},
                      "atoms": {key: [list(atom) for atom in value]
                                for key, value in fitted_rule.atoms.items()}})

    # Give each episode equal weight, regardless of machine/query count.
    per_episode = {episode: math.fsum(values) / len(values)
                   for episode, values in episode_delta.items() if values}
    deltas = list(per_episode.values())
    n_episodes = len(deltas)
    mean_delta = math.fsum(deltas) / n_episodes if n_episodes else None
    if n_episodes > 1:
        variance = math.fsum((value - mean_delta) ** 2 for value in deltas) / (n_episodes - 1)
        episode_se = math.sqrt(variance / n_episodes)
    else:
        episode_se = None
    upper_ci = (mean_delta + Z_VALUE * episode_se
                if mean_delta is not None and episode_se is not None else None)
    candidate_wins = math.fsum(row[1] for row in all_scores) / len(all_scores) if all_scores else None
    v6_wins = math.fsum(row[2] for row in all_scores) / len(all_scores) if all_scores else None
    candidate_logloss = math.fsum(row[3] for row in all_scores) / len(all_scores) if all_scores else None
    v6_logloss = math.fsum(row[4] for row in all_scores) / len(all_scores) if all_scores else None
    use_candidate = bool(n_episodes >= min_episodes and upper_ci is not None and upper_ci < 0.0)
    return {"use_candidate": use_candidate,
            "diagnostics": {"criterion": "conditional_size_brier_episode_mean_plus_1.96_se_lt_0",
                            "chosen_head": "size_rule" if use_candidate else "v6_size_head",
                            "min_episodes": int(min_episodes), "z_value": Z_VALUE,
                            "n_episodes": n_episodes, "n_wins": len(all_scores),
                            "mean_delta_brier_candidate_minus_v6": mean_delta,
                            "episode_se": episode_se, "upper_ci": upper_ci,
                            "candidate_brier": candidate_wins, "v6_brier": v6_wins,
                            "candidate_logloss": candidate_logloss,
                            "v6_logloss": v6_logloss, "folds": folds}}
