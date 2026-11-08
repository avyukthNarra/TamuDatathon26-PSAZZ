"""Episode-level parity flip: one shared tick, then the pair mixer keeps learning.

The gate is the midpoint of the widest gap in the training episodes' joint
log Bayes factors (flip versus one direction the whole time). An episode above
that gate exchanges agree and differ counts at its posterior mode. The tick is
not a fraction of the cutoff, and the gate is not a constant from another round.

The detector is the one scored at +0.18 validation points on the v6 pair mixer.
``min_side`` is a sample-size floor (8 observed parity rows on each side of a
candidate tick), not a location on the episode.
"""

import math
from collections import defaultdict

from .pair_mixer import CLASS_INDEX
from .schema import index_history, unique_queries

MIN_SIDE = 8


def _logsumexp(values):
    if not values:
        return -math.inf
    top = max(values)
    if top == -math.inf:
        return -math.inf
    return top + math.log(math.fsum(math.exp(v - top) for v in values))


def episode_series(queries, history):
    """{episode_id: [(query, own class-index list), ...]}. own[cutoff] stays None."""
    indexed = index_history(queries, history)
    by_episode = defaultdict(list)
    for query in unique_queries(queries).values():
        own = [None] * (query.cutoff + 1)
        for play in indexed.get((query.episode_id, query.machine_id), []):
            own[play.tick] = CLASS_INDEX[play.payout]
        by_episode[query.episode_id].append((query, own))
    for episode_id, rows in by_episode.items():
        cutoffs = {query.cutoff for query, _own in rows}
        if len(cutoffs) != 1:
            raise RuntimeError(f"{episode_id} has mixed cutoffs {sorted(cutoffs)}")
    return dict(by_episode)


def bernoulli_ll(wins, n, p):
    if n <= 0:
        return 0.0
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return wins * math.log(p) + (n - wins) * math.log(1.0 - p)


def _two_rate_ll(w_agree, n_agree, w_diff, n_diff, agree_is_live, prior_a, prior_b):
    strength = prior_a + prior_b
    if agree_is_live:
        w_live, n_live, w_dead, n_dead = w_agree, n_agree, w_diff, n_diff
    else:
        w_live, n_live, w_dead, n_dead = w_diff, n_diff, w_agree, n_agree
    p_live = (w_live + prior_a) / (n_live + strength)
    p_dead = (w_dead + prior_a) / (n_dead + strength)
    if agree_is_live:
        return bernoulli_ll(w_agree, n_agree, p_live) + bernoulli_ll(w_diff, n_diff, p_dead)
    return bernoulli_ll(w_agree, n_agree, p_dead) + bernoulli_ll(w_diff, n_diff, p_live)


def _flip_ll(left, right, agree_was_live, prior_a, prior_b):
    """left/right are (w_agree, n_agree, w_diff, n_diff). Rates are tied across the flip."""
    strength = prior_a + prior_b
    if agree_was_live:
        w_live = left[0] + right[2]
        n_live = left[1] + right[3]
        w_dead = left[2] + right[0]
        n_dead = left[3] + right[1]
        p_before_agree = (w_live + prior_a) / (n_live + strength)
        p_after_agree = (w_dead + prior_a) / (n_dead + strength)
        p_before_diff = (w_dead + prior_a) / (n_dead + strength)
        p_after_diff = (w_live + prior_a) / (n_live + strength)
    else:
        w_live = left[2] + right[0]
        n_live = left[3] + right[1]
        w_dead = left[0] + right[2]
        n_dead = left[1] + right[3]
        p_before_agree = (w_dead + prior_a) / (n_dead + strength)
        p_after_agree = (w_live + prior_a) / (n_live + strength)
        p_before_diff = (w_live + prior_a) / (n_live + strength)
        p_after_diff = (w_dead + prior_a) / (n_dead + strength)
    return (bernoulli_ll(left[0], left[1], p_before_agree) + bernoulli_ll(left[2], left[3], p_before_diff)
            + bernoulli_ll(right[0], right[1], p_after_agree) + bernoulli_ll(right[2], right[3], p_after_diff))


def machine_flip_table(own, pair, cutoff, min_side, prior=(1.0, 1.0)):
    """(ll_stable, {tau: ll_flip}) for one pair. Each ll marginalizes direction.

    tau is the first tick of the new direction. Candidates need at least
    min_side observed parity rows on each side.
    """
    lag_a, lag_b = pair
    prior_a, prior_b = prior
    rows = []
    for t in range(lag_b, cutoff):
        if own[t] is None or t - lag_a < 0 or own[t - lag_a] is None or own[t - lag_b] is None:
            continue
        agree = (own[t - lag_a] > 0) == (own[t - lag_b] > 0)
        rows.append((t, agree, 1.0 if own[t] > 0 else 0.0))
    if len(rows) < 2 * min_side:
        return None
    n = len(rows)
    ag_w = [0.0] * (n + 1)
    ag_n = [0.0] * (n + 1)
    df_w = [0.0] * (n + 1)
    df_n = [0.0] * (n + 1)
    for i, (_t, agree, won) in enumerate(rows):
        ag_w[i + 1] = ag_w[i] + (won if agree else 0.0)
        ag_n[i + 1] = ag_n[i] + (1.0 if agree else 0.0)
        df_w[i + 1] = df_w[i] + (0.0 if agree else won)
        df_n[i + 1] = df_n[i] + (0.0 if agree else 1.0)
    total = (ag_w[n], ag_n[n], df_w[n], df_n[n])
    ll_stable = _logsumexp([
        _two_rate_ll(*total, agree_is_live, prior_a, prior_b) for agree_is_live in (True, False)
    ]) - math.log(2.0)
    by_tau = {}
    for i in range(min_side, n - min_side + 1):
        left = (ag_w[i], ag_n[i], df_w[i], df_n[i])
        right = (ag_w[n] - ag_w[i], ag_n[n] - ag_n[i], df_w[n] - df_w[i], df_n[n] - df_n[i])
        ll = _logsumexp([
            _flip_ll(left, right, agree_was_live, prior_a, prior_b) for agree_was_live in (True, False)
        ]) - math.log(2.0)
        by_tau[rows[i][0]] = ll
    return ll_stable, by_tau


def detect_episode(rows, pairs, min_side=MIN_SIDE, prior=(1.0, 1.0)):
    """Joint posterior over one flip tick. Flat prior on the candidate ticks."""
    cutoff = rows[0][0].cutoff
    per_machine = []
    for _query, own in rows:
        best = None
        for pair in pairs:
            table = machine_flip_table(own, pair, cutoff, min_side, prior)
            if table is None:
                continue
            ll_stable, by_tau = table
            if not by_tau:
                continue
            ll_flip = _logsumexp(list(by_tau.values())) - math.log(len(by_tau))
            score = max(ll_stable, ll_flip)
            if best is None or score > best[0]:
                best = (score, ll_stable, by_tau)
        per_machine.append(best)
    n_obs = [sum(value is not None for value in own[:cutoff]) for _query, own in rows]
    missing = 1.0 - (sum(n_obs) / (len(rows) * cutoff)) if cutoff else 1.0
    empty = {
        "logbf": 0.0, "map_tau": None, "map_frac": None, "sd_frac": 1.0,
        "missing": missing, "n_machines": 0, "cutoff": cutoff,
    }
    usable = [row for row in per_machine if row is not None]
    empty["n_machines"] = len(usable)
    if len(usable) < 2:
        return empty
    counts = defaultdict(int)
    for _score, _stable, by_tau in usable:
        for tau in by_tau:
            counts[tau] += 1
    need = max(2, int(math.ceil(0.75 * len(usable))))
    taus = sorted(tau for tau, count in counts.items() if count >= need)
    if len(taus) < 2:
        taus = sorted(counts)
    if len(taus) < 2:
        return empty
    stable_sum = math.fsum(row[1] for row in usable)
    joint = []
    for tau in taus:
        total = 0.0
        for _score, ll_stable, by_tau in usable:
            total += by_tau.get(tau, ll_stable)
        joint.append(total)
    logbf = _logsumexp(joint) - math.log(len(joint)) - stable_sum
    top = _logsumexp(joint)
    weights = [math.exp(v - top) for v in joint]
    mean = math.fsum(w * tau for w, tau in zip(weights, taus))
    var = math.fsum(w * (tau - mean) ** 2 for w, tau in zip(weights, taus))
    map_tau = taus[max(range(len(taus)), key=lambda i: joint[i])]
    return {
        "logbf": logbf,
        "map_tau": map_tau,
        "map_frac": map_tau / cutoff,
        "sd_frac": math.sqrt(max(var, 0.0)) / cutoff,
        "missing": missing,
        "n_machines": len(usable),
        "cutoff": cutoff,
    }


def detect_all(by_episode, pairs, min_side=MIN_SIDE):
    return {episode_id: detect_episode(rows, pairs, min_side=min_side)
            for episode_id, rows in by_episode.items()}


def widest_gap(values):
    """(width, midpoint, left, right) of the largest gap in a list of log Bayes factors.

    Fewer than two values has no gap: width 0 and a midpoint of +inf, so nothing
    is swapped.
    """
    ordered = sorted(float(v) for v in values)
    if len(ordered) < 2:
        return 0.0, math.inf, None, None
    width, mid, left, right = max(
        ((ordered[i + 1] - ordered[i], 0.5 * (ordered[i] + ordered[i + 1]), ordered[i], ordered[i + 1])
         for i in range(len(ordered) - 1)),
        key=lambda item: (item[0], -item[1]),
    )
    return width, mid, left, right


class FlipGate:
    """Train-only threshold. ``swaps`` reads the split it is asked to forecast."""

    def __init__(self, threshold, width, left, right, train_detect):
        self.threshold = threshold
        self.width = width
        self.left = left
        self.right = right
        self.train_detect = train_detect

    @classmethod
    def fit(cls, pairs, queries, history):
        detect = detect_all(episode_series(queries, history), pairs)
        width, mid, left, right = widest_gap(row["logbf"] for row in detect.values())
        return cls(mid, width, left, right, detect)

    def swaps(self, queries, history, pairs):
        """{episode_id: tau} for episodes whose logBF is above the train gap midpoint."""
        detect = detect_all(episode_series(queries, history), pairs)
        chosen = {}
        for episode_id, row in detect.items():
            tau = row["map_tau"]
            if tau is not None and row["logbf"] > self.threshold:
                chosen[episode_id] = tau
        return chosen, detect
