"""Train-fitted mixture of static and lag-driven payout-size states.

A win's payout, given that it won, is emitted by one of two atoms per machine
type. A static state emits atom ``s``; a dynamic state emits atom
``s XOR 1{payout at t - L >= 2}``. Joint EM fits both state families on train.
The train-only static mass compares normalized class marginal likelihoods;
the dynamic state prior is pooled by the machine's supported win-lag pair.
Unobserved lag bits are marginalized over the two atoms.

The lag set is learned over candidates 1..8 (or all candidates when no lag
exceeds uniform posterior share). Prequential forecasts use only prior ticks.

Standard library only. Deterministic: EM restarts use fixed seeds.
"""

import math
import random

from .schema import index_history, unique_queries

WIN_PAYOUTS = (1, 2, 5, 20)
WIN_INDEX = {payout: index for index, payout in enumerate(WIN_PAYOUTS)}
CANDIDATE_LAGS = tuple(range(1, 9))
EM_SEEDS = tuple(range(6))
EM_ITERS = 200
FLOOR = 1e-4
_LOG_FLOOR = 1e-300


def _log(p):
    return math.log(p) if p > _LOG_FLOOR else math.log(_LOG_FLOOR)


def _feature(pay, tick, lag):
    """1 if the lag tick paid at least 2, 0 if it paid 0 or 1, None if unknown."""
    index = tick - lag
    if index < 0 or index >= len(pay) or pay[index] is None:
        return None
    return 1 if pay[index] >= 2 else 0


def select_lags(mass, lags):
    """Lags whose share of posterior mass exceeds the uniform share.

    A flat posterior has no lag above ``1 / len(lags)``, and the caller then
    keeps every candidate. ``mass`` maps a lag to an un-normalized total.
    """
    lags = tuple(lags)
    total = math.fsum(mass.get(lag, 0.0) for lag in lags)
    if not lags or total <= 0.0:
        return lags
    share = 1.0 / len(lags)
    tolerance = 1e-12
    kept = tuple(lag for lag in lags if mass.get(lag, 0.0) / total > share + tolerance)
    return kept or lags


def size_posterior(pay, cutoff, atoms, lags, prior=None):
    """Posterior over ``(L, s)`` and mixture at ``cutoff``.

    ``pay[t]`` is an int payout or None. Only wins with ``t < cutoff`` enter
    the likelihood. Returns ``(mix, posterior)`` where ``mix`` is the
    four-class distribution on (1, 2, 5, 20).
    """
    atom = (tuple(atoms[0]), tuple(atoms[1]))
    lags = tuple(lags)
    wins = [(tick, WIN_INDEX[pay[tick]]) for tick in range(cutoff) if pay[tick] in WIN_INDEX]
    states = [(lag, polarity) for lag in lags for polarity in (0, 1)]
    loglik = {}
    if prior is None:
        state_prior = {state: 1.0 / len(states) for state in states}
    else:
        mass = math.fsum(max(0.0, prior.get(state, 0.0)) for state in states)
        state_prior = ({state: max(0.0, prior.get(state, 0.0)) / mass for state in states}
                       if mass > 0.0 else {state: 1.0 / len(states) for state in states})
    for lag, polarity in states:
        total = _log(state_prior[(lag, polarity)])
        for tick, outcome in wins:
            feature = _feature(pay, tick, lag)
            if feature is None:
                probability = 0.5 * (atom[0][outcome] + atom[1][outcome])
            else:
                probability = atom[polarity ^ feature][outcome]
            total += _log(probability)
        loglik[(lag, polarity)] = total
    if not states:
        return [0.25, 0.25, 0.25, 0.25], {}
    top = max(loglik.values())
    raw = {state: math.exp(value - top) for state, value in loglik.items()}
    normalizer = math.fsum(raw.values())
    posterior = {state: weight / normalizer for state, weight in raw.items()}
    mix = [0.0, 0.0, 0.0, 0.0]
    for (lag, polarity), weight in posterior.items():
        feature = _feature(pay, cutoff, lag)
        if feature is None:
            distribution = [0.5 * (atom[0][i] + atom[1][i]) for i in range(4)]
        else:
            distribution = atom[polarity ^ feature]
        for i in range(4):
            mix[i] += weight * distribution[i]
    return mix, posterior


def _logsumexp(values):
    top = max(values)
    return top + math.log(math.fsum(math.exp(value - top) for value in values))


def hybrid_posterior(pay, cutoff, atoms, lags, static_mass, dynamic_prior=None):
    """Posterior over static ``(None,s)`` and dynamic ``(L,s)`` states."""
    lags = tuple(lags)
    wins = [(tick, WIN_INDEX[pay[tick]]) for tick in range(cutoff)
            if pay[tick] in WIN_INDEX]
    states = [("static", None, s) for s in (0, 1)]
    states += [("dynamic", lag, s) for lag in lags for s in (0, 1)]
    q_static = min(1.0, max(0.0, static_mass))
    dynamic_states = [(lag, s) for lag in lags for s in (0, 1)]
    if dynamic_prior is None:
        dyn_prior = {state: 1.0 / len(dynamic_states) for state in dynamic_states}
    else:
        total = math.fsum(max(0.0, dynamic_prior.get(state, 0.0)) for state in dynamic_states)
        dyn_prior = ({state: max(0.0, dynamic_prior.get(state, 0.0)) / total
                      for state in dynamic_states} if total > 0 else
                     {state: 1.0 / len(dynamic_states) for state in dynamic_states})
    log_terms = {}
    for kind, lag, polarity in states:
        if kind == "static":
            prior = q_static * 0.5
            emission = math.fsum(_log(atoms[polarity][outcome]) for _, outcome in wins)
        else:
            prior = (1.0 - q_static) * dyn_prior[(lag, polarity)]
            emission = 0.0
            for tick, outcome in wins:
                feature = _feature(pay, tick, lag)
                if feature is None:
                    probability = 0.5 * (atoms[0][outcome] + atoms[1][outcome])
                else:
                    probability = atoms[polarity ^ feature][outcome]
                emission += _log(probability)
        log_terms[(kind, lag, polarity)] = _log(prior) + emission
    if not states:
        return [0.25] * 4, {}, -math.inf, -math.inf
    total = _logsumexp(list(log_terms.values()))
    posterior = {state: math.exp(value - total) for state, value in log_terms.items()}
    mix = [0.0] * 4
    for (kind, lag, polarity), weight in posterior.items():
        if kind == "static":
            distribution = atoms[polarity]
        else:
            feature = _feature(pay, cutoff, lag)
            distribution = ([(a + b) * 0.5 for a, b in zip(atoms[0], atoms[1])]
                            if feature is None else atoms[polarity ^ feature])
        for index, value in enumerate(distribution):
            mix[index] += weight * value
    static_ll, dynamic_ll = _class_log_evidence(pay, cutoff, atoms, lags, dyn_prior)
    return mix, posterior, static_ll, dynamic_ll


def _class_log_evidence(pay, cutoff, atoms, lags, dynamic_prior=None):
    """Normalized class-conditional marginal likelihoods on the same wins."""
    lags = tuple(lags)
    wins = [(tick, WIN_INDEX[pay[tick]]) for tick in range(cutoff)
            if pay[tick] in WIN_INDEX]
    static = []
    for polarity in (0, 1):
        emission = math.fsum(_log(atoms[polarity][outcome]) for _, outcome in wins)
        static.append(math.log(0.5) + emission)
    dynamic_states = [(lag, s) for lag in lags for s in (0, 1)]
    if not dynamic_states:
        return _logsumexp(static), -math.inf
    if dynamic_prior is None:
        prior = {state: 1.0 / len(dynamic_states) for state in dynamic_states}
    else:
        total = math.fsum(max(0.0, dynamic_prior.get(state, 0.0)) for state in dynamic_states)
        prior = ({state: max(0.0, dynamic_prior.get(state, 0.0)) / total
                  for state in dynamic_states} if total > 0 else
                 {state: 1.0 / len(dynamic_states) for state in dynamic_states})
    dynamic = []
    for lag, polarity in dynamic_states:
        emission = 0.0
        for tick, outcome in wins:
            feature = _feature(pay, tick, lag)
            probability = (0.5 * (atoms[0][outcome] + atoms[1][outcome]) if feature is None
                           else atoms[polarity ^ feature][outcome])
            emission += _log(probability)
        dynamic.append(_log(prior[(lag, polarity)]) + emission)
    return _logsumexp(static), _logsumexp(dynamic)


_PAIR_CANDIDATES = tuple((a, b) for a in range(1, 9) for b in range(a + 1, 9))
_PAIR_MIN_CELL = 8
_PAIR_PSEUDOCOUNT = 0.5


def _pair_gain(pay, a, b):
    """Jeffreys-smoothed gain for win ~ XOR(win[t-a], win[t-b])."""
    n = [0, 0]
    wins = [0, 0]
    for tick, outcome in enumerate(pay):
        if outcome is None:
            continue
        left, right = tick - a, tick - b
        if left < 0 or right < 0 or pay[left] is None or pay[right] is None:
            continue
        cell = int((pay[left] > 0) ^ (pay[right] > 0))
        n[cell] += 1
        wins[cell] += int(outcome > 0)
    if min(n) < _PAIR_MIN_CELL:
        return None

    def ll(count, success):
        p = (success + 0.5) / (count + 1.0)
        return success * math.log(p) + (count - success) * math.log1p(-p)

    two = ll(n[0], wins[0]) + ll(n[1], wins[1])
    one = ll(sum(n), sum(wins))
    return two - one


def _best_pair(pay, pairs):
    scored = [(gain, pair) for pair in pairs
              if (gain := _pair_gain(pay, *pair)) is not None]
    if not scored:
        return None
    return max(scored, key=lambda item: (item[0], -item[1][0], -item[1][1]))[1]


def _gain_from_counts(cells):
    n = [cells[i][0] for i in (0, 1)]
    wins = [cells[i][1] for i in (0, 1)]
    if min(n) < _PAIR_MIN_CELL:
        return None

    def ll(count, success):
        p = (success + 0.5) / (count + 1.0)
        return success * math.log(p) + (count - success) * math.log1p(-p)

    return ll(n[0], wins[0]) + ll(n[1], wins[1]) - ll(sum(n), sum(wins))


def _best_pair_from_counts(counts, pairs):
    scored = [(gain, pair) for pair in pairs
              if (gain := _gain_from_counts(counts[pair])) is not None]
    if not scored:
        return None
    return max(scored, key=lambda item: (item[0], -item[1][0], -item[1][1]))[1]


def _learn_pair_prior(rows, atoms, lags, weights=None):
    """Learn soft state priors from train-only win-pair assignments."""
    votes = {pair: 0 for pair in _PAIR_CANDIDATES}
    assignments = []
    for machine_type, pay, _wins in rows:
        pair = _best_pair(pay, _PAIR_CANDIDATES)
        assignments.append((machine_type, pay, pair))
        if pair is not None:
            votes[pair] += 1
    max_votes = max(votes.values(), default=0)
    selected = tuple(pair for pair in _PAIR_CANDIDATES
                     if max_votes and votes[pair] >= 0.3 * max_votes)
    if not selected:
        return {}, (), assignments

    states = tuple((lag, polarity) for lag in lags for polarity in (0, 1))
    counts = {pair: {state: _PAIR_PSEUDOCOUNT for state in states} for pair in selected}
    for index, (machine_type, pay, _global_pair) in enumerate(assignments):
        pair = _best_pair(pay, selected)
        if pair not in counts or machine_type not in atoms:
            continue
        _mix, posterior = size_posterior(pay, len(pay), atoms[machine_type], lags)
        weight = 1.0 if weights is None else weights[index]
        for state in states:
            counts[pair][state] += weight * posterior.get(state, 0.0)
    priors = {}
    for pair, bucket in counts.items():
        total = math.fsum(bucket.values())
        priors[pair] = {state: value / total for state, value in bucket.items()}
    return priors, selected, assignments


def _joint_em(rows, lags, seed, iters):
    """EM with equal normalized static/dynamic class priors during atom fitting."""
    rng = random.Random(seed)
    types = sorted({machine_type for machine_type, _, _ in rows})
    lag_stats = _sufficient(rows, lags)
    stats = []
    for (machine_type, pay, wins), (_typ, observed, missing) in zip(rows, lag_stats):
        static_counts = [0, 0, 0, 0]
        for _tick, outcome in wins:
            static_counts[outcome] += 1
        stats.append((machine_type, pay, static_counts, observed, missing))
    atoms = {machine_type: [_simplex(rng), _simplex(rng)] for machine_type in types}
    pi = [1.0 / len(lags)] * len(lags)
    previous = None
    total = 0.0
    for _ in range(iters):
        counts = {machine_type: [[1e-9] * 4, [1e-9] * 4] for machine_type in types}
        lag_mass = [0.0] * len(lags)
        total = 0.0
        for machine_type, pay, static_counts, observed, missing in stats:
            logs = {}
            for polarity in (0, 1):
                logs[("static", None, polarity)] = math.fsum(
                    static_counts[outcome] * _log(atoms[machine_type][polarity][outcome])
                    for outcome in range(4))
            for lag in lags:
                index = lags.index(lag)
                for polarity in (0, 1):
                    value = 0.0
                    for outcome in range(4):
                        for feature in (0, 1):
                            count = observed[index][feature][outcome]
                            if count:
                                value += count * _log(atoms[machine_type][polarity ^ feature][outcome])
                        if missing[index][outcome]:
                            both = 0.5 * (atoms[machine_type][0][outcome] +
                                          atoms[machine_type][1][outcome])
                            value += missing[index][outcome] * _log(both)
                    logs[("dynamic", lag, polarity)] = value
            terms = {}
            for state, value in logs.items():
                kind, lag, _polarity = state
                state_prior = 0.25 if kind == "static" else 0.25 * pi[lags.index(lag)]
                terms[state] = _log(state_prior) + value
            norm = _logsumexp(list(terms.values()))
            total += norm
            for state, value in terms.items():
                responsibility = math.exp(value - norm)
                kind, lag, polarity = state
                if kind == "static":
                    for outcome in range(4):
                        counts[machine_type][polarity][outcome] += (
                            responsibility * static_counts[outcome])
                    continue
                index = lags.index(lag)
                lag_mass[index] += responsibility
                for outcome in range(4):
                    for feature in (0, 1):
                        count = observed[index][feature][outcome]
                        if count:
                            counts[machine_type][polarity ^ feature][outcome] += responsibility * count
                    count = missing[index][outcome]
                    if count:
                        both = 0.5 * (atoms[machine_type][0][outcome] +
                                      atoms[machine_type][1][outcome])
                        counts[machine_type][0][outcome] += (
                            responsibility * count * atoms[machine_type][0][outcome] / (2.0 * both))
                        counts[machine_type][1][outcome] += (
                            responsibility * count * atoms[machine_type][1][outcome] / (2.0 * both))
        for machine_type in types:
            for polarity in (0, 1):
                atom_total = math.fsum(counts[machine_type][polarity])
                atoms[machine_type][polarity] = [value / atom_total
                                                  for value in counts[machine_type][polarity]]
        lag_total = math.fsum(lag_mass)
        if lag_total > 0:
            pi = [value / lag_total for value in lag_mass]
        if previous is not None and abs(total - previous) < 1e-7 * max(1.0, abs(total)):
            break
        previous = total
    return atoms, pi, total


def _histories(queries, history):
    """One row per machine: (type, pay, wins), machines in episode/machine order."""
    indexed = index_history(queries, history)
    rows = []
    ordered = sorted(unique_queries(queries).values(), key=lambda q: (q.episode_id, q.machine_id))
    for query in ordered:
        pay = [None] * query.cutoff
        for play in indexed.get((query.episode_id, query.machine_id), ()):
            pay[play.tick] = play.payout
        wins = [(tick, WIN_INDEX[pay[tick]]) for tick in range(query.cutoff) if pay[tick] in WIN_INDEX]
        rows.append((query.machine_type, pay, wins))
    return rows


def _sufficient(rows, lags):
    """Per machine, win counts by lag, feature bit, and payout class.

    ``n[lag][feature][class]`` counts wins whose lag tick is observed.
    ``nu[lag][class]`` counts wins whose lag tick is missing.
    """
    stats = []
    for machine_type, pay, wins in rows:
        observed = [[[0, 0, 0, 0], [0, 0, 0, 0]] for _ in lags]
        missing = [[0, 0, 0, 0] for _ in lags]
        for tick, outcome in wins:
            for index, lag in enumerate(lags):
                feature = _feature(pay, tick, lag)
                if feature is None:
                    missing[index][outcome] += 1
                else:
                    observed[index][feature][outcome] += 1
        stats.append((machine_type, observed, missing))
    return stats


def _simplex(rng):
    draws = [rng.gammavariate(1.0, 1.0) for _ in range(4)]
    total = math.fsum(draws)
    return [draw / total for draw in draws]


def _em(stats, lags, seed, iters):
    """EM for per-type atoms and a shared lag prior. Returns atoms, pi, loglik."""
    rng = random.Random(seed)
    types = sorted({machine_type for machine_type, _, _ in stats})
    atoms = {machine_type: [_simplex(rng), _simplex(rng)] for machine_type in types}
    pi = [1.0 / len(lags)] * len(lags)
    previous = None
    total = 0.0
    for _ in range(iters):
        counts = {machine_type: [[1e-9] * 4, [1e-9] * 4] for machine_type in types}
        lag_mass = [0.0] * len(lags)
        total = 0.0
        for machine_type, observed, missing in stats:
            both = [0.5 * (atoms[machine_type][0][y] + atoms[machine_type][1][y]) for y in range(4)]
            loglik = []
            for index in range(len(lags)):
                for polarity in (0, 1):
                    value = 0.0
                    for outcome in range(4):
                        for feature in (0, 1):
                            count = observed[index][feature][outcome]
                            if count:
                                value += count * _log(atoms[machine_type][polarity ^ feature][outcome])
                        if missing[index][outcome]:
                            value += missing[index][outcome] * _log(both[outcome])
                    loglik.append(value)
            top = max(loglik)
            weights = []
            for index in range(len(lags)):
                for polarity in (0, 1):
                    weights.append(pi[index] * 0.5 * math.exp(loglik[index * 2 + polarity] - top))
            normalizer = math.fsum(weights)
            total += top + math.log(normalizer)
            for index in range(len(lags)):
                for polarity in (0, 1):
                    responsibility = weights[index * 2 + polarity] / normalizer
                    lag_mass[index] += responsibility
                    for outcome in range(4):
                        for feature in (0, 1):
                            count = observed[index][feature][outcome]
                            if count:
                                counts[machine_type][polarity ^ feature][outcome] += responsibility * count
                        count = missing[index][outcome]
                        if count:
                            split = both[outcome] * 2.0
                            counts[machine_type][0][outcome] += (
                                responsibility * count * atoms[machine_type][0][outcome] / split)
                            counts[machine_type][1][outcome] += (
                                responsibility * count * atoms[machine_type][1][outcome] / split)
        for machine_type in types:
            for polarity in (0, 1):
                atom_total = math.fsum(counts[machine_type][polarity])
                atoms[machine_type][polarity] = [value / atom_total for value in counts[machine_type][polarity]]
        lag_total = math.fsum(lag_mass)
        pi = [value / lag_total for value in lag_mass]
        if previous is not None and abs(total - previous) < 1e-7 * abs(total):
            break
        previous = total
    return atoms, pi, total


def _canon(atoms):
    """Atom 0 is the one with the larger chance of paying 1, then 2, 5, 20."""
    ordered = {}
    for machine_type, (left, right) in atoms.items():
        pair = sorted((tuple(left), tuple(right)), key=lambda vector: tuple(-value for value in vector))
        ordered[machine_type] = pair
    return ordered


class SizeRule:
    """Train-fitted static/dynamic payout-size mixture."""

    def __init__(self, atoms, lags, mass, pair_prior=None, pair_candidates=(),
                 static_prior=0.5, assignments=()):
        self.atoms = {machine_type: (tuple(pair[0]), tuple(pair[1])) for machine_type, pair in atoms.items()}
        self.lags = tuple(lags)
        self.mass = dict(mass)
        self.pair_prior = dict(pair_prior or {})
        self.pair_candidates = tuple(pair_candidates)
        self.static_prior = float(static_prior)
        self._pair_assignments = tuple(assignments)
        self.loglik = None
        self.restarts = ()

    @classmethod
    def learn(cls, queries, history, lags=CANDIDATE_LAGS, seeds=EM_SEEDS, iters=EM_ITERS):
        lags = tuple(lags)
        rows = _histories(queries, history)
        if not lags:
            lags = CANDIDATE_LAGS
        stats = _sufficient(rows, lags)
        best = None
        restarts = []
        for seed in seeds:
            atoms, _pi, total = _joint_em(rows, lags, seed, iters)
            restarts.append((seed, total))
            if best is None or total > best[0]:
                best = (total, atoms)
        atoms = _canon(best[1])
        # Weight lag selection and pair pooling by each train machine's
        # normalized dynamic-class evidence so static machines do not invent
        # support for arbitrary lag rules.
        dynamic_weights = []
        mass = {lag: 0.0 for lag in lags}
        for machine_type, pay, _wins in rows:
            static_ll, dynamic_ll = _class_log_evidence(pay, len(pay), atoms[machine_type], lags)
            both = _logsumexp((math.log(0.5) + static_ll, math.log(0.5) + dynamic_ll))
            p_dynamic = math.exp(math.log(0.5) + dynamic_ll - both)
            dynamic_weights.append(p_dynamic)
            _mix, posterior = size_posterior(pay, len(pay), atoms[machine_type], lags)
            for lag in lags:
                mass[lag] += p_dynamic * (posterior[(lag, 0)] + posterior[(lag, 1)])
        kept = select_lags(mass, lags)
        count = float(len(rows) or 1)
        pair_prior, pair_candidates, assignments = _learn_pair_prior(rows, atoms, kept,
                                                                       weights=dynamic_weights)
        static_wins = 0
        for (machine_type, pay, _wins), weight in zip(rows, dynamic_weights):
            pair = _best_pair(pay, pair_candidates)
            dynamic_ll = _class_log_evidence(pay, len(pay), atoms[machine_type], kept,
                                             pair_prior.get(pair))[1]
            static_ll = _class_log_evidence(pay, len(pay), atoms[machine_type], kept,
                                            pair_prior.get(pair))[0]
            static_wins += int(static_ll > dynamic_ll)
        empirical_static = static_wins / count
        static_prior = min(0.99, max(0.01, empirical_static))
        rule = cls(atoms, kept, {lag: mass[lag] / count for lag in lags},
                   pair_prior, pair_candidates, static_prior, assignments)
        rule.loglik = best[0]
        rule.restarts = tuple(restarts)
        return rule

    def prequential(self, pay, machine_type):
        """Yield `(tick, mix4)` before observing each raw payout in ``pay``.

        The pair key, state likelihood, and prediction all use only earlier
        ticks. This is intended for causal train-fold gating and online use.
        """
        atoms = self.atoms.get(machine_type)
        if atoms is None or not self.lags:
            for tick, _value in enumerate(pay):
                yield tick, [0.25] * 4
            return
        static_ll = [0.0, 0.0]
        dynamic_ll = {(lag, s): 0.0 for lag in self.lags for s in (0, 1)}
        pair_counts = {pair: [[0, 0], [0, 0]] for pair in self.pair_candidates}
        prior_uniform = {state: 1.0 / (2 * len(self.lags))
                         for state in dynamic_ll}
        for tick, payout in enumerate(pay):
            prefix = pay[:tick]
            pair = _best_pair_from_counts(pair_counts, self.pair_candidates)
            dynamic_prior = self.pair_prior.get(pair, prior_uniform)
            prior_total = math.fsum(dynamic_prior.get(state, 0.0) for state in dynamic_ll)
            if prior_total <= 0:
                dynamic_prior = prior_uniform
                prior_total = 1.0
            terms = {("static", None, s): math.log(self.static_prior * 0.5) + static_ll[s]
                     for s in (0, 1)}
            for state, value in dynamic_ll.items():
                lag, polarity = state
                prior = ((1.0 - self.static_prior) * dynamic_prior.get(state, 0.0) /
                         prior_total)
                terms[("dynamic", lag, polarity)] = _log(prior) + value
            norm = _logsumexp(tuple(terms.values()))
            mix = [0.0] * 4
            for (kind, lag, polarity), value in terms.items():
                weight = math.exp(value - norm)
                if kind == "static":
                    component = atoms[polarity]
                else:
                    feature = _feature(prefix, tick, lag)
                    component = ([(a + b) * 0.5 for a, b in zip(atoms[0], atoms[1])]
                                 if feature is None else atoms[polarity ^ feature])
                for index, probability in enumerate(component):
                    mix[index] += weight * probability
            yield tick, mix

            outcome = WIN_INDEX.get(payout)
            if outcome is not None:
                for polarity in (0, 1):
                    static_ll[polarity] += _log(atoms[polarity][outcome])
                for lag in self.lags:
                    feature = _feature(prefix, tick, lag)
                    for polarity in (0, 1):
                        probability = (0.5 * (atoms[0][outcome] + atoms[1][outcome])
                                       if feature is None else atoms[polarity ^ feature][outcome])
                        dynamic_ll[(lag, polarity)] += _log(probability)
            if payout is None:
                continue
            for pair in self.pair_candidates:
                a, b = pair
                if tick < max(a, b):
                    continue
                left, right = pay[tick - a], pay[tick - b]
                if left is None or right is None:
                    continue
                cell = int((left > 0) ^ (right > 0))
                pair_counts[pair][cell][0] += 1
                pair_counts[pair][cell][1] += int(payout > 0)

    def apply(self, predictions, queries, history):
        """Keep each row's ``p_0``. Replace the size split, floor at 1e-4, renormalize."""
        indexed = index_history(queries, history)
        result = {}
        for query in unique_queries(queries).values():
            pay = [None] * query.cutoff
            for play in indexed.get((query.episode_id, query.machine_id), ()):
                pay[play.tick] = play.payout
            atoms = self.atoms.get(query.machine_type)
            if atoms is None or not self.lags:
                mix = [0.25, 0.25, 0.25, 0.25]
            else:
                mix = self.predict_mix(pay, query.machine_type, query.cutoff)
            p0 = predictions[query.query_id][0]
            raw = [p0] + [(1.0 - p0) * value for value in mix]
            floored = [max(value, FLOOR) for value in raw]
            total = math.fsum(floored)
            result[query.query_id] = tuple(value / total for value in floored)
        return result

    def predict_mix(self, pay, machine_type, cutoff):
        """Causal conditional payout mix at ``cutoff`` using only ``pay[:cutoff]``."""
        atoms = self.atoms.get(machine_type)
        if atoms is None or not self.lags:
            return [0.25] * 4
        prefix = list(pay[:cutoff])
        pair = _best_pair(prefix, self.pair_candidates)
        mix, _posterior, _static_ll, _dynamic_ll = hybrid_posterior(
            prefix, cutoff, atoms, self.lags, self.static_prior, self.pair_prior.get(pair))
        return mix
