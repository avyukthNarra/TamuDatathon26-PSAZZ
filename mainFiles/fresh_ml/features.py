"""Causal, gap-aware in-context summaries; no target probabilities are accepted."""
from itertools import combinations

import numpy as np

PAYOUTS = (0, 1, 2, 5, 20)
CONTEXTS = [(k,) for k in range(1, 9)] + list(combinations(range(1, 9), 2))


def previous_sum(values, window=None):
    """Row t sums [max(0,t-window),t), never row t itself."""
    cumulative = np.concatenate([np.zeros_like(values[:1]), np.cumsum(values, axis=0)])
    times = np.arange(len(values))
    return cumulative[times] - cumulative[np.maximum(0, times - window) if window else 0]


def machine_features(sequence, peers, machine_type, signal):
    """Dense tick arrays use -1 for unobserved; includes a final query row."""
    sequence, peers = np.asarray(sequence), np.asarray(peers)
    n = len(sequence)
    observed = sequence >= 0
    wins = sequence > 0
    onehot = np.stack([sequence == k for k in range(5)], axis=1).astype(float)
    parts, names, groups = [], [], []

    def add(values, labels, group):
        values = np.asarray(values)
        if values.ndim == 1:
            values = values[:, None]
        parts.append(values)
        names.extend(labels)
        groups.extend([group] * values.shape[1])

    add(np.tile([machine_type == k for k in ('steady', 'balanced', 'jackpot')], (n, 1)),
        ['type_' + k for k in ('steady', 'balanced', 'jackpot')], 'base')
    # signal is query metadata, assumed fixed for this machine's episode.
    add(np.full(n, signal), ['signal'], 'base')
    add(np.arange(n), ['tick'], 'base')
    lagged = np.full((n, 8), -1, dtype=int)
    for lag in range(1, 9):
        lagged[lag:, lag - 1] = sequence[:-lag]
        add(np.stack([lagged[:, lag - 1], lagged[:, lag - 1] >= 0,
                      lagged[:, lag - 1] > 0], axis=1),
            [f'lag{lag}_{s}' for s in ('category', 'observed', 'win')], 'base')

    summaries = {}
    for window in (16, 48, 128, None):
        counts = previous_sum(onehot, window)
        total = counts.sum(axis=1)
        # Fixed weak symmetric smoothing, independent of every outcome.
        probs = (counts + 0.2) / (total[:, None] + 1)
        sizes = (counts[:, 1:] + 0.2) / (counts[:, 1:].sum(axis=1, keepdims=True) + 0.8)
        label = str(window or 'all')
        add(np.column_stack([probs, sizes, np.log1p(total)]),
            [f'self{label}_p{k}' for k in PAYOUTS] +
            [f'self{label}_size{k}' for k in PAYOUTS[1:]] + [f'self{label}_logn'], 'base')
        summaries[window] = (counts, total, probs, sizes)
    add(summaries[16][2] - summaries[None][2], [f'change_p{k}' for k in PAYOUTS], 'base')
    last_observed = np.maximum.accumulate(np.where(observed, np.arange(n), -1))
    last_win = np.maximum.accumulate(np.where(wins, np.arange(n), -1))
    for label, values in [('observed', last_observed), ('win', last_win)]:
        earlier = np.r_[-1, values[:-1]]
        add(np.minimum(256, np.arange(n) - earlier), [f'age_{label}'], 'base')

    peer_observed = (peers >= 0).sum(axis=0)
    peer_counts = np.stack([(peers == k).sum(axis=0) for k in range(5)], axis=1)
    for window in (1, 8, 32, 96):
        counts = previous_sum(peer_counts, window)
        total = previous_sum(peer_observed, window)
        add(np.column_stack([(counts + 0.2) / (total[:, None] + 1), np.log1p(total)]),
            [f'peer{window}_p{k}' for k in PAYOUTS] + [f'peer{window}_logn'], 'peer')
    for lag in range(1, 9):
        values = np.zeros((n, 2))
        values[lag:, 0] = ((peers[:, :-lag] > 0).sum(axis=0) + 0.5) / (peer_observed[:-lag] + 1)
        values[lag:, 1] = peer_observed[:-lag]
        add(values, [f'peer_lag{lag}_win', f'peer_lag{lag}_n'], 'peer')

    # Estimate the response to the CURRENT lag state using only PREVIOUS
    # examples of that state, separately on each machine. All pairs are
    # enumerated: the global learner must discover useful delays and signs.
    for context in CONTEXTS:
        values = lagged[:, np.array(context) - 1]
        valid = (values >= 0).all(axis=1)
        state = (values > 0).astype(int) @ (2 ** np.arange(len(context)))
        tag = '_'.join(map(str, context))
        for window in (48, None):
            local = np.zeros((n, 5))
            for s in range(2 ** len(context)):
                matches = valid & (state == s)
                counts = previous_sum(onehot * matches[:, None], window)
                local[matches] = counts[matches]
            count = local.sum(axis=1)
            prior = summaries[window][2]
            probs = (local + 8 * prior) / (count[:, None] + 8)
            win = 1 - probs[:, 0]
            columns = [win, win - (1 - prior[:, 0]), np.log1p(count), valid]
            labels = [f'ctx{tag}_{window}_{s}' for s in ('win', 'effect', 'logn', 'valid')]
            if len(context) == 1:
                # Local evidence for the conditional positive payout mix.
                sizes = (local[:, 1:] + 8 * summaries[window][3]) / (local[:, 1:].sum(axis=1, keepdims=True) + 8)
                columns.extend(sizes.T)
                labels.extend(f'ctx{tag}_{window}_size{k}' for k in PAYOUTS[1:])
            add(np.column_stack(columns), labels, 'context')
    result = np.column_stack(parts).astype(np.float32)
    assert np.isfinite(result).all()
    return result, names, groups
