"""Hidden-rule HMM win model, blended into the pair mixer's win chance.

Adapted from hmm_model.HiddenRuleHMM on the zhill-56.28-hmm branch. The pair mixer learns decayed
tally tables per lag pair, so it lags a machine that switches rules. This model states the rule
outright and filters over it:

  hidden state  (lag pair, "match" | "differ") for each pair from choose_pairs, plus a "no rule"
                state that pays at the machine's smoothed base rate
  emission      P(win | state) = the live level if the state says the machine is live given the
                spins `a` and `b` back, else the dead level
  switching     each tick the state is kept with prob 1 - switch, else redrawn from the prior
                (Bayesian change-point behaviour instead of exponential forgetting)

Forward filtering over the machine's observed spins gives the rule posterior at the cutoff; the win
chance is the posterior-weighted emission. Live / dead levels come from the same win prior the
mixer uses (win_prior_by_type_from_targets): the "type|signal" entry if present, else the type's.
Only ticks strictly below the cutoff are read. Standard library only, deterministic.
"""
import math

from .schema import index_history, probabilities, unique_queries

HMM_WEIGHT = 0.3   # share of the HMM in the blended win chance
HMM_SWITCH = 0.001
MIN_PROBABILITY = 1e-4  # same floor as PairMixer.predict


def levels_from_prior(win_prior):
    """{key: (live, dead)} from a {key: [(w, dead, k), (w, live, k), ...]} win prior."""
    return {key: (components[1][1], components[0][1]) for key, components in (win_prior or {}).items()
            if isinstance(components, (list, tuple)) and len(components) >= 2}


class HiddenRuleHMM:
    def __init__(self, pairs, levels, switch=HMM_SWITCH, none_weight=0.1, base_rate=0.45):
        self.pairs = [tuple(p) for p in pairs]
        self.levels = dict(levels)
        self.switch = switch
        self.none_weight = none_weight
        self.base_rate = base_rate
        self.states = [(pair, direction) for pair in self.pairs for direction in ("match", "differ")] + [None]
        share = (1.0 - none_weight) / (len(self.states) - 1)
        self.prior = [share] * (len(self.states) - 1) + [none_weight]

    @staticmethod
    def _live(seq, t, state):
        (a, b), direction = state
        if t - a not in seq or t - b not in seq:
            return None
        agree = (seq[t - a] > 0) == (seq[t - b] > 0)
        return agree if direction == "match" else not agree

    def _emission(self, seq, t, state, rate, live, dead):
        if state is None:
            return rate
        on = self._live(seq, t, state)
        return rate if on is None else (live if on else dead)

    def win_probability(self, seq, cutoff, live, dead):
        post = list(self.prior)
        last = None
        wins = observed = 0
        for t in sorted(seq):
            if last is not None:
                stay = (1.0 - self.switch) ** (t - last)
                post = [stay * p + (1.0 - stay) * q for p, q in zip(post, self.prior)]
            last = t
            won = seq[t] > 0
            rate = (wins + 4.0 * self.base_rate) / (observed + 4.0)
            post = [p * (e if won else 1.0 - e)
                    for p, e in zip(post, (self._emission(seq, t, s, rate, live, dead) for s in self.states))]
            total = math.fsum(post)
            post = [p / total for p in post]
            wins += won
            observed += 1
        if last is not None:
            stay = (1.0 - self.switch) ** (cutoff - last)
            post = [stay * p + (1.0 - stay) * q for p, q in zip(post, self.prior)]
        rate = (wins + 4.0 * self.base_rate) / (observed + 4.0)
        return math.fsum(p * self._emission(seq, cutoff, s, rate, live, dead) for p, s in zip(post, self.states))

    def win_chances(self, queries, history):
        indexed = index_history(queries, history)
        out = {}
        for query in unique_queries(queries).values():
            key = f"{query.machine_type}|{query.signal}"
            live, dead = self.levels.get(key) or self.levels.get(query.machine_type) or (0.898, 0.0435)
            seq = {p.tick: p.payout for p in indexed.get((query.episode_id, query.machine_id), [])}
            out[query.query_id] = self.win_probability(seq, query.cutoff, live, dead)
        return out


class WithRuleHMM:
    """The pair mixer's forecast with its win chance blended toward the HMM's (sizes unchanged)."""

    def __init__(self, mixer, hmm, weight=HMM_WEIGHT):
        self.mixer, self.hmm, self.weight = mixer, hmm, weight
        self.pairs, self.cell_mode = mixer.pairs, mixer.cell_mode

    def predict(self, queries, history):
        base = self.mixer.predict(queries, history)
        hmm = self.hmm.win_chances(queries, history)
        out = {}
        for query_id, probs in base.items():
            win = 1.0 - probs[0]
            blended = (1.0 - self.weight) * win + self.weight * hmm[query_id]
            sizes = [p / win for p in probs[1:]]
            raw = [max(p, MIN_PROBABILITY) for p in [1.0 - blended] + [blended * s for s in sizes]]
            total = math.fsum(raw)
            out[query_id] = probabilities([p / total for p in raw], query_id)
        return out
