"""Train-only cross-fitted guard for the full payout-win cell mode.

Call only when round-two cutoff selection would otherwise choose ``full``.
Full is retained only when paired whole-episode held-out TRAIN Brier supports
it with the fixed mean + 1.96 SE rule. Validation data is never accepted here.
"""

import math
from collections import defaultdict

from .pair_mixer import CLASS_INDEX, PairMixer, choose_pairs_robust, discover_size_keys
from .schema import unique_queries
from .size_rule import SizeRule

MIN_EPISODES = 8
Z_VALUE = 1.96


def _brier(probabilities, outcome):
    if (len(probabilities) != 5
            or any(not math.isfinite(float(value)) or value < 0 or value > 1
                   for value in probabilities)
            or abs(math.fsum(probabilities) - 1.0) > 1e-6):
        raise ValueError("cell-mode model emitted an invalid five-class forecast")
    return math.fsum((value - (1.0 if index == outcome else 0.0)) ** 2
                     for index, value in enumerate(probabilities))


def _size_modes(rule):
    return {machine_type: {"modes": [list(atom) for atom in atoms],
                           "weights": [0.5, 0.5]}
            for machine_type, atoms in rule.atoms.items()}


def _default(reason, min_episodes, folds=()):
    return {"allow_full": False,
            "diagnostics": {"criterion": "full_minus_parity_episode_mean_plus_1.96_se_lt_0",
                            "chosen_mode": "parity", "reason": reason,
                            "min_episodes": int(min_episodes), "z_value": Z_VALUE,
                            "n_episodes": 0, "n_queries": 0, "folds": list(folds)}}


def guard_full_cell_mode(queries, history, labels, *, min_episodes=MIN_EPISODES):
    """Return ``allow_full`` only after a train-only episode cross-fit approves it.

    The caller invokes this only if its initial round-two cutoff-Brier choice is
    full. For each fold, SizeRule atoms, robust pair support, size keys, and
    both PairMixer variants are fit on the other episodes only. ``labels`` are
    read exclusively for held-out cutoff scoring; they never enter a fit.
    Shared win priors are deliberately omitted for both modes.
    """
    query_map = unique_queries(queries)
    episodes = sorted({query.episode_id for query in query_map.values()})
    episode_deltas = defaultdict(list)
    fold_results = []
    total_queries = 0

    for fold_index in (0, 1):
        held_ids = {episode for index, episode in enumerate(episodes)
                    if index % 2 == fold_index}
        fit_queries = tuple(query for query in query_map.values()
                            if query.episode_id not in held_ids)
        held_queries = tuple(query for query in query_map.values()
                             if query.episode_id in held_ids)
        fit_history = tuple(play for play in history if play.episode_id not in held_ids)
        held_history = tuple(play for play in history if play.episode_id in held_ids)
        if not fit_queries or not held_queries:
            continue

        try:
            size_rule = SizeRule.learn(fit_queries, fit_history)
            modes = _size_modes(size_rule)
            pairs = choose_pairs_robust(fit_queries, fit_history)
            size_keys = discover_size_keys(fit_queries, fit_history, on_empty=((),))
            models = {
                mode: PairMixer(pairs=pairs, cell_mode=mode, size_modes=modes,
                                size_keys=size_keys).fit(fit_queries, fit_history)
                for mode in ("parity", "full")
            }
            predictions = {mode: model.predict(held_queries, held_history)
                           for mode, model in models.items()}
            fold_scores = defaultdict(list)
            for query in held_queries:
                if query.query_id not in labels:
                    raise ValueError(f"missing held-out TRAIN label: {query.query_id}")
                outcome = CLASS_INDEX[labels[query.query_id]]
                parity_loss = _brier(predictions["parity"][query.query_id], outcome)
                full_loss = _brier(predictions["full"][query.query_id], outcome)
                fold_scores[query.episode_id].append(full_loss - parity_loss)
            total_queries += sum(len(values) for values in fold_scores.values())
            for episode, values in fold_scores.items():
                episode_deltas[episode].append(math.fsum(values) / len(values))
            fold_results.append({
                "fold": fold_index,
                "fit_episodes": len({query.episode_id for query in fit_queries}),
                "heldout_episodes": len({query.episode_id for query in held_queries}),
                "scored_queries": sum(len(values) for values in fold_scores.values()),
                "pairs": [list(pair) for pair in pairs],
                "size_keys": [list(key) for key in size_keys],
                "size_lags": list(size_rule.lags),
                "static_prior": float(size_rule.static_prior),
                "type_size_priors": {
                    key: list(value) for key, value in models["parity"].type_size_priors.items()},
            })
        except Exception as exc:
            return _default(str(exc), min_episodes, fold_results)

    per_episode = {episode: math.fsum(deltas) / len(deltas)
                   for episode, deltas in episode_deltas.items() if deltas}
    values = list(per_episode.values())
    count = len(values)
    mean_delta = math.fsum(values) / count if count else None
    if count > 1:
        variance = math.fsum((value - mean_delta) ** 2 for value in values) / (count - 1)
        standard_error = math.sqrt(variance / count)
    else:
        standard_error = None
    upper = (mean_delta + Z_VALUE * standard_error
             if mean_delta is not None and standard_error is not None else None)
    allow_full = bool(count >= min_episodes and upper is not None and upper < 0.0)
    return {"allow_full": allow_full,
            "diagnostics": {"criterion": "full_minus_parity_episode_mean_plus_1.96_se_lt_0",
                            "chosen_mode": "full" if allow_full else "parity",
                            "min_episodes": int(min_episodes), "z_value": Z_VALUE,
                            "n_episodes": count,
                            "n_queries": total_queries,
                            "mean_delta_brier_full_minus_parity": mean_delta,
                            "episode_se": standard_error, "upper_ci": upper,
                            "folds": fold_results}}
