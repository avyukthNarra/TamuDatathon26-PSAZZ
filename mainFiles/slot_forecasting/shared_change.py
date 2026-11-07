"""Find and use one shared win-rate change point across an episode."""

from dataclasses import dataclass
import math
from typing import Mapping

from .pair_logistic import PairLogisticForecaster, _expand, _lag_vector


@dataclass(frozen=True)
class SharedChange:
    """Detection result: penalized evidence, per-machine direction and support."""

    boundary: int | None
    evidence: float
    directions: dict[str, int]
    observations: dict[str, tuple[int, int]]
    candidates: int
    raw_gain: float


def _bernoulli_log_likelihood(wins: int, count: int) -> float:
    if count == 0 or wins == 0 or wins == count:
        return 0.0
    probability = wins / count
    return wins * math.log(probability) + (count - wins) * math.log1p(-probability)


def estimate_shared_change(
    streams: Mapping[str, Mapping[int, int]],
    cutoff: int,
    min_side: int = 16,
    step: int = 4,
    evidence_threshold: float = 0.0,
) -> SharedChange:
    """Estimate a common tick boundary using independent Bernoulli rates.

    ``streams`` maps machine IDs to ``{tick: payout}`` observations. A win is
    payout > 0. Ticks at or after cutoff are ignored, and absent ticks remain
    unobserved. Every nonempty stream must have ``min_side`` observations on
    each side of a candidate. Candidate ticks are spaced ``step`` apart.

    The summed profile log-likelihood gain gives each machine its own rates
    before and after the shared split, so directions may differ. Evidence is
    the maximum gain minus ``(M/2)*log(N/M) + log(C)``: a BIC-style cost for
    the M added rates and the C-point scan. A boundary is returned only when
    evidence meets the threshold. Raw gain remains available if gated out.
    """
    if min_side < 1:
        raise ValueError("min_side must be positive")
    if step < 1:
        raise ValueError("step must be positive")

    binary: dict[str, tuple[list[int], list[int]]] = {}
    for machine, observations in streams.items():
        pairs = sorted((int(tick), int(payout) > 0)
                       for tick, payout in observations.items()
                       if int(tick) < cutoff)
        if pairs:
            binary[str(machine)] = ([tick for tick, _ in pairs],
                                    [int(win) for _, win in pairs])

    active = len(binary)
    total_obs = sum(len(ticks) for ticks, _ in binary.values())
    if not active:
        return SharedChange(None, float("-inf"), {}, {}, 0, 0.0)

    earliest = min(ticks[0] for ticks, _ in binary.values())
    # Split tick s places observations at t < s on the left. Include the
    # final boundary when exactly min_side observations remain on the right.
    candidate_ticks = range(earliest + min_side, cutoff - min_side + 1, step)
    best_gain = float("-inf")
    best_boundary = None
    best_directions: dict[str, int] = {}
    best_counts: dict[str, tuple[int, int]] = {}
    candidate_count = 0

    for boundary in candidate_ticks:
        gain = 0.0
        directions: dict[str, int] = {}
        counts: dict[str, tuple[int, int]] = {}
        admissible = True
        for machine, (ticks, wins) in binary.items():
            left_n = left_wins = 0
            for tick, win in zip(ticks, wins):
                if tick < boundary:
                    left_n += 1
                    left_wins += win
            right_n = len(ticks) - left_n
            if left_n < min_side or right_n < min_side:
                admissible = False
                break
            right_wins = sum(wins) - left_wins
            gain += (
                _bernoulli_log_likelihood(left_wins, left_n)
                + _bernoulli_log_likelihood(right_wins, right_n)
                - _bernoulli_log_likelihood(sum(wins), len(wins))
            )
            left_rate = left_wins / left_n
            right_rate = right_wins / right_n
            directions[machine] = (right_rate > left_rate) - (right_rate < left_rate)
            counts[machine] = (left_n, right_n)
        if admissible:
            candidate_count += 1
            if gain > best_gain:
                best_gain = gain
                best_boundary = boundary
                best_directions = directions
                best_counts = counts

    if best_boundary is None:
        return SharedChange(None, float("-inf"), {}, {}, 0, 0.0)

    penalty = (active / 2) * math.log(total_obs / active) + math.log(candidate_count)
    evidence = best_gain - penalty
    accepted = evidence >= evidence_threshold
    return SharedChange(
        best_boundary if accepted else None,
        evidence,
        best_directions if accepted else {},
        best_counts if accepted else {},
        candidate_count,
        best_gain,
    )


class SharedChangeForecaster(PairLogisticForecaster):
    """E1 pair-logistic model with an episode-wide shared-change boundary."""

    def __init__(self, before_weight=0.1, evidence_threshold=5.0,
                 min_side=16, change_step=4, **kwargs):
        super().__init__(**kwargs)
        self.before_weight = float(before_weight)
        if not 0.0 <= self.before_weight <= 1.0:
            raise ValueError("before_weight must be in [0, 1]")
        self.evidence_threshold = float(evidence_threshold)
        self.min_side = int(min_side)
        self.change_step = int(change_step)
        self.change_diagnostics = {}

    def predict(self, queries, history):
        from collections import defaultdict
        import numpy as np

        from .schema import index_history, unique_queries

        queries, history = tuple(queries), tuple(history)
        query_map = unique_queries(queries)
        indexed = index_history(queries, history)
        by_episode_cutoff = defaultdict(list)
        for query in query_map.values():
            by_episode_cutoff[(query.episode_id, query.cutoff)].append(query)

        changes = {}
        self.change_diagnostics = {}
        for (episode_id, cutoff), episode_queries in by_episode_cutoff.items():
            streams = {
                query.machine_id: {
                    play.tick: play.payout
                    for play in indexed.get((episode_id, query.machine_id), ())
                    if play.tick < cutoff
                }
                for query in episode_queries
            }
            change = estimate_shared_change(
                streams, cutoff, min_side=self.min_side, step=self.change_step,
                evidence_threshold=self.evidence_threshold,
            )
            changes[(episode_id, cutoff)] = change
            self.change_diagnostics[(episode_id, cutoff)] = {
                "boundary": change.boundary,
                "evidence": change.evidence,
                "raw_gain": change.raw_gain,
                "candidates": change.candidates,
                "directions": change.directions,
            }

        # Keep E1's pooled payout shape, routing each query its episode's
        # boundary explicitly so identical streams in other episodes cannot collide.
        base = self.base.predict(queries, history)
        output = {}
        for (episode_id, cutoff), episode_queries in by_episode_cutoff.items():
            boundary = changes[(episode_id, cutoff)].boundary
            for query in episode_queries:
                sequence = {
                    play.tick: play.payout
                    for play in indexed.get((episode_id, query.machine_id), ())
                }
                win_probability = self._fit_win(sequence, cutoff, boundary)
                fallback = base[query.query_id]
                if win_probability is None:
                    output[query.query_id] = tuple(fallback)
                    continue
                win_probability = min(max(win_probability, self.clip), 1.0 - self.clip)
                positive_shape = np.array(fallback[1:]) / sum(fallback[1:])
                probabilities = np.concatenate(
                    [[1.0 - win_probability], win_probability * positive_shape]
                )
                output[query.query_id] = tuple(
                    float(value) for value in probabilities / probabilities.sum()
                )
        return output

    def win_probability(self, sequence, cutoff):
        """E1-compatible single-stream prediction without shared context."""
        return self._fit_win(sequence, cutoff, boundary=None)

    def _fit_win(self, sequence, cutoff, boundary):
        import warnings
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        query_features = _lag_vector(sequence, cutoff, self.lags)
        if query_features is None:
            return None
        rows, labels, weights = [], [], []
        for tick, payout in sequence.items():
            if tick >= cutoff:
                continue
            features = _lag_vector(sequence, tick, self.lags)
            if features is None:
                continue
            rows.append(_expand(features, self.pairs))
            labels.append(1 if payout > 0 else 0)
            age = cutoff - 1 - tick
            weight = 1.0 if self.half_life is None else 2.0 ** (-age / self.half_life)
            if boundary is not None and tick < boundary:
                weight *= self.before_weight
            weights.append(weight)
        if len(rows) < self.min_rows or len(set(labels)) < 2:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LogisticRegression(C=self.C, max_iter=1000)
            model.fit(np.array(rows), labels, sample_weight=np.array(weights))
        return float(model.predict_proba(_expand(query_features, self.pairs)[None])[0, 1])
