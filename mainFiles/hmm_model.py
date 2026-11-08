"""Hidden-rule HMM: an explicit switching model of the simulator's win rule.

Every other team model learns tally tables per lag pair and forgets old spins exponentially.
This one states the rule outright and filters over it:

  hidden state h = (rule, direction):
      rule      a lag pair (a, b) or a single lag (k,), from the repo's pair discovery on train
      direction "match" (live when the two spins agree / the spin won) or "differ" (the opposite)
    plus one "no rule" state that pays at the machine's base rate.
  emission  P(win at t | h) = LIVE if h says the machine is live given spins t-a, t-b, else DEAD
            (LIVE/DEAD are the dead/live win levels from validation, as in pair mixer v4)
  switching each tick, the machine keeps its rule with prob 1 - switch, else jumps to any other
            state (Bayesian change-point behaviour, instead of exponential forgetting)

Forward filtering over the machine's observed spins gives P(h | spins before the cutoff), and the
win chance at the cutoff is the posterior-weighted emission. Payout size given a win is taken
from another model (size_from), since this model only answers "will it pay".
Only ticks strictly below each machine's cutoff are read. Standard library only.
"""
import math

from slot_forecasting.schema import index_history


class HiddenRuleHMM:
    def __init__(self, rules, live=0.898, dead=0.0435, switch=0.001, none_weight=0.1, rule_prior=None,
                 base_rate=0.45, levels=None):
        self.rules = [tuple(r) for r in rules]
        self.live, self.dead = live, dead
        # levels: {machine_type: (live, dead)}; true live/dead win chances differ a lot by type
        self.levels = dict(levels or {})
        self.switch = switch
        self.none_weight = none_weight
        self.rule_prior = rule_prior or {}
        self.base_rate = base_rate

    def fit(self, queries, history, labels=None):
        return self

    def _states(self):
        states, prior = [], []
        for rule in self.rules:
            w = self.rule_prior.get(rule, 1.0)
            for direction in ("match", "differ"):
                states.append((rule, direction))
                prior.append(w)
        total = math.fsum(prior)
        prior = [(1 - self.none_weight) * p / total for p in prior]
        states.append(None)
        prior.append(self.none_weight)
        return states, prior

    @staticmethod
    def _live(seq, t, rule, direction):
        values = [seq.get(t - k) for k in rule]
        if any(v is None for v in values):
            return None
        if len(rule) == 2:
            agree = (values[0] > 0) == (values[1] > 0)
        else:
            agree = values[0] > 0
        return agree if direction == "match" else not agree

    def win_probability(self, seq, cutoff):
        states, prior = self._states()
        post = list(prior)
        n = len(states)
        last = None
        wins = obs = 0
        for t in sorted(seq):
            if last is not None:  # switching: with prob stay keep the state, else jump to the prior
                stay = (1 - self.switch) ** (t - last)
                post = [stay * p + (1 - stay) * q for p, q in zip(post, prior)]
            last = t
            won = seq[t] > 0
            rate = (wins + self.base_rate * 4) / (obs + 4)
            like = []
            for state in states:
                if state is None:
                    p = rate
                else:
                    live = self._live(seq, t, *state)
                    p = rate if live is None else (self.live if live else self.dead)
                like.append(p if won else 1 - p)
            post = [p * l for p, l in zip(post, like)]
            total = math.fsum(post)
            post = [p / total for p in post]
            wins += won
            obs += 1
        if last is not None:
            stay = (1 - self.switch) ** (cutoff - last)
            post = [stay * p + (1 - stay) * q for p, q in zip(post, prior)]
        rate = (wins + self.base_rate * 4) / (obs + 4)
        pw = 0.0
        for state, p in zip(states, post):
            if state is None:
                pw += p * rate
            else:
                live = self._live(seq, cutoff, *state)
                pw += p * (rate if live is None else (self.live if live else self.dead))
        return pw

    def predict(self, queries, history, size_from=None):
        """size_from: {query_id: 5 probs} whose payout-size split (given a win) is reused."""
        indexed = index_history(queries, history)
        out = {}
        for q in queries:
            seq = {p.tick: p.payout for p in indexed.get((q.episode_id, q.machine_id), [])}
            if q.machine_type in self.levels:
                self.live, self.dead = self.levels[q.machine_type]
            pw = min(max(self.win_probability(seq, q.cutoff), 1e-4), 1 - 1e-4)
            if size_from is not None:
                other = size_from[q.query_id]
                ow = 1 - other[0]
                split = [x / ow for x in other[1:]] if ow > 0 else [0.25] * 4
            else:
                split = [0.25] * 4
            out[q.query_id] = tuple([1 - pw] + [pw * s for s in split])
        return out


def levels_from_targets(targets, queries):
    """Median true win chance above / below 0.5, per machine type, from validation targets.csv."""
    kind = {q.query_id: q.machine_type for q in queries}
    wins = {}
    for t in targets:
        if t.query_id in kind:
            wins.setdefault(kind[t.query_id], []).append(1.0 - t.probabilities[0])
    out = {}
    for k, w in wins.items():
        live = sorted(x for x in w if x >= 0.5)
        dead = sorted(x for x in w if x < 0.5)
        if len(live) >= 10 and len(dead) >= 10:
            out[k] = (live[len(live) // 2] if len(live) % 2 else (live[len(live) // 2 - 1] + live[len(live) // 2]) / 2,
                      dead[len(dead) // 2] if len(dead) % 2 else (dead[len(dead) // 2 - 1] + dead[len(dead) // 2]) / 2)
    return out
