"""Estimate PairMixer priors from training histories only.

These estimates are deliberately approximate. Win levels pool the strongest supported
pair-parity contrast per machine. Payout-size modes are cross-fitted across alternating
actual ticks so a machine's size rule is selected on one half and contributes counts on
the other; type-level modes then pool those held-out counts with symmetric smoothing.
"""
from collections import defaultdict
import math
import statistics

from .pair_mixer import choose_pairs
from .schema import index_history, unique_queries


SIZE_LAGS = (1, 2, 4)
PAYOUTS = (1, 2, 5, 20)
MIN_WIN_CELL = 20
MIN_SIZE_CELL = 3
MIN_WIN_MACHINES = 3
MIN_WIN_SEPARATION = 0.3


def estimate_train_priors(queries, history):
    """Return ``(size_modes, win_prior)`` using only the supplied train queries/history.

    Win cells use Laplace-smoothed rates and require 20 observations in both parity
    cells plus a rate gap of at least 0.3. The median low/high rates across at least
    three qualifying machines define the dead/live mixture means. Size-rule selection
    uses multinomial deviance on one actual-tick parity and accumulates counts only on
    the other parity; the halves are swapped. A 0.5 pseudocount per payout class keeps
    every returned size distribution normalized and finite.
    """
    queries = tuple(queries)
    history = tuple(history)
    unique_queries(queries)
    indexed = index_history(queries, history)
    train_pairs = choose_pairs(queries, history)
    by_machine = {(q.episode_id, q.machine_id): q for q in queries}
    level_low, level_high = [], []
    pooled_size = defaultdict(lambda: [[0.0] * len(PAYOUTS) for _ in range(2)])

    for key, plays in indexed.items():
        q = by_machine[key]
        seq = {p.tick: p.payout for p in plays}
        _collect_win_levels(seq, level_low, level_high, train_pairs)
        _collect_size_counts(seq, pooled_size[q.machine_type])

    size_modes = {}
    for machine_type, counts in pooled_size.items():
        if any(math.fsum(mode) <= 0 for mode in counts):
            continue
        normalized = []
        for mode in counts:
            smoothed = [value + 0.5 for value in mode]
            total = math.fsum(smoothed)
            normalized.append(tuple(value / total for value in smoothed))
        size_modes[machine_type] = tuple(normalized)

    win_prior = None
    if len(level_low) >= MIN_WIN_MACHINES and len(level_high) >= MIN_WIN_MACHINES:
        low, high = statistics.median(level_low), statistics.median(level_high)
        if high - low >= MIN_WIN_SEPARATION:
            win_prior = [(0.45, round(low, 4), 6.0),
                         (0.45, round(high, 4), 6.0),
                         (0.1, 0.5, 2.0)]
    return size_modes, win_prior


def _collect_win_levels(seq, lows, highs, pairs):
    best = None
    for pair in pairs:
        cells = [[0, 0], [0, 0]]  # (wins, observations), indexed by lag-parity
        for tick, payout in seq.items():
            first, second = seq.get(tick - pair[0]), seq.get(tick - pair[1])
            if first is None or second is None:
                continue
            parity = int((first > 0) == (second > 0))
            cells[parity][0] += payout > 0
            cells[parity][1] += 1
        if min(cell[1] for cell in cells) < MIN_WIN_CELL:
            continue
        rates = [(wins + 1.0) / (n + 2.0) for wins, n in cells]
        low, high = sorted(rates)
        separation = high - low
        if separation < MIN_WIN_SEPARATION:
            continue
        candidate = (separation, sum(cell[1] for cell in cells), low, high)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is not None:
        lows.append(best[2])
        highs.append(best[3])


def _collect_size_counts(seq, pooled):
    ticks = sorted(seq)
    for fit_parity in (0, 1):
        best = None
        for lag in SIZE_LAGS:
            cells = [[0] * len(PAYOUTS) for _ in range(2)]
            for tick in ticks:
                if tick % 2 != fit_parity or seq[tick] == 0:
                    continue
                previous = seq.get(tick - lag)
                if previous is None:
                    continue
                group = int(previous >= 5)
                cells[group][PAYOUTS.index(seq[tick])] += 1
            if min(map(sum, cells)) < MIN_SIZE_CELL:
                continue
            score = _multinomial_deviance(cells)
            means = [math.fsum(n * payout for n, payout in zip(cell, PAYOUTS)) / sum(cell)
                     for cell in cells]
            # Keep the low-expected-payout mode first, independent of flag direction.
            small_group = 0 if means[0] <= means[1] else 1
            candidate = (score, -lag, small_group)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is None:
            continue
        selected_lag, small_group = -best[1], best[2]
        holdout_parity = 1 - fit_parity
        for tick in ticks:
            if tick % 2 != holdout_parity or seq[tick] == 0:
                continue
            previous = seq.get(tick - selected_lag)
            if previous is None:
                continue
            group = int(previous >= 5)
            mode = 0 if group == small_group else 1
            pooled[mode][PAYOUTS.index(seq[tick])] += 1.0


def _multinomial_deviance(cells):
    total_by_class = [sum(cell[i] for cell in cells) for i in range(len(PAYOUTS))]
    total_n = sum(total_by_class)
    score = 0.0
    for cell in cells:
        n = sum(cell)
        for observed, pooled in zip(cell, total_by_class):
            if observed:
                expected = n * pooled / total_n
                score += 2.0 * observed * math.log(observed / expected)
    return score
