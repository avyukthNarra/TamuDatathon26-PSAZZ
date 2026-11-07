"""Per-machine lag-PAIR rule model, built on top of pooled_recent.

What we found (MODEL_PLAN.md): most machines are "live" (pay out ~88%) or "dead" (pay out ~10%)
depending on whether they WON at two specific earlier spins — e.g. live when exactly one of the
spins 1 and 3 back was a win (XOR), or the opposite (XNOR). Common pairs: (1,3), (2,5), (4,7).

How the model works, for each machine:
  1. Safe guess = pooled_recent.
  2. For every candidate "rule" (a pair of lags, a single lag, or no rule) keep a tally:
     situation = win/lose at those lags  ->  counts of the next payout.
  3. Walk through the machine's own history in time order. Before adding each spin to the tally,
     score how well the rule WOULD HAVE predicted it. Rules that keep predicting well get trusted.
     (This is a fair test: a rule is only ever graded on spins it hadn't seen yet.)
  4. Old spins fade (forgetting), so if a machine's rule changes, the new rule takes over.
  5. Final forecast = trust-weighted average of every rule's forecast for the cutoff spin.

Only ticks strictly below each machine's cutoff are read. Standard library only.
"""
import itertools
import math

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.schema import PAYOUTS, index_history

N = len(PAYOUTS)
INDEX = {p: i for i, p in enumerate(PAYOUTS)}


def default_rules(max_lag=8):
    pairs = list(itertools.combinations(range(1, max_lag + 1), 2))
    singles = [(k,) for k in range(1, max_lag + 1)]
    return [()] + singles + pairs  # () = "no rule": just the machine's overall counts


class RuleForecaster:
    """Defaults are the tuned settings (validation 48.87 / 60, see MODEL_PLAN.md)."""

    def __init__(self, rules=None, strength=1.0, forget=(0.95, 1.0), trust_forget=1.0, beta=1.0,
                 floor=1e-4, base_strength=20.0, base_half_life=24.0, rule_prior=None,
                 prior_weight=3.0, prior_min_history=150, size_strength=None, size_half_life=None, binary=False,
                 size_rule_strength=None, size_rule_beta=1.0, size_rule_min_wins=0):
        self.binary = binary                  # True = rules score win/lose only (needs size_strength)
        self.size_rule_strength = size_rule_strength  # None = no per-machine size rules
        self.size_rule_beta = size_rule_beta
        self.size_rule_min_wins = size_rule_min_wins
        self.size_strength = size_strength    # None = sizes come from the rule tallies (v2 behaviour)
        self.size_half_life = size_half_life
        self.size_prior = {}
        self.rules = [tuple(r) for r in (rules if rules is not None else default_rules())]
        # "head start" for rules that are common in the TRAINING machines. If rule_prior is None it is
        # learned in fit(); prior_weight=0 turns it off.
        self.prior_weight = prior_weight
        self.prior_min_history = prior_min_history
        self.strength = strength          # pretend-examples of the safe guess in each tally cell
        # per-tick fade of old spins in the tallies; several values = each machine picks its memory
        self.forget = tuple(forget) if isinstance(forget, (tuple, list)) else (forget,)
        self.trust_forget = trust_forget  # per-tick fade of old evidence when scoring rules
        self.beta = beta                  # how sharply to prefer the best-scoring rules
        self.floor = floor
        self.rule_prior = None if rule_prior is None else dict(rule_prior)  # log-prior bonus per rule
        self.rule_wins = {}
        self.base = FrequencyForecaster("pooled_recent", prior_strength=base_strength, half_life=base_half_life)

    def settings(self):
        return {k: v for k, v in vars(self).items() if k not in ("base", "rules", "rule_prior", "rule_wins", "size_prior")} | {
            "n_rules": len(self.rules), "base_strength": self.base.prior_strength,
            "base_half_life": self.base.half_life,
            "rule_wins": {"-".join(map(str, r)) or "none": n for r, n in sorted(self.rule_wins.items())}}

    def fit(self, queries, history, labels=None):
        self.base.fit(queries, history, labels)
        self._learn_size_prior(queries, history)
        if self.rule_prior is None:
            self.rule_prior = self._learn_rule_prior(queries, history)
        return self

    def _learn_rule_prior(self, queries, history):
        """Count which rule best explains each long training machine (no forgetting, fair
        predict-then-learn scoring), and turn the counts into a log head start."""
        queries, history = tuple(queries), tuple(history)
        base_preds = self.base.predict(queries, history)
        indexed = index_history(queries, history)
        wins = {}
        for q in sorted(queries, key=lambda q: q.query_id):
            seq = {p.tick: p.payout for p in indexed.get((q.episode_id, q.machine_id), [])}
            if len(seq) < self.prior_min_history:
                continue
            prior = [2.0 * b for b in base_preds[q.query_id]]
            best = None
            for rule in self.rules:
                tally, score = {}, 0.0
                for t in sorted(seq):
                    ctx = self._context(seq, t, rule)
                    if ctx is None:
                        continue
                    counts = tally.setdefault(ctx, [0.0] * N)
                    i = INDEX[seq[t]]
                    score += math.log((counts[i] + prior[i]) / (math.fsum(counts) + 2.0))
                    counts[i] += 1.0
                if best is None or score > best[0]:
                    best = (score, rule)
            wins[best[1]] = wins.get(best[1], 0) + 1
        self.rule_wins = wins
        total = sum(wins.values())
        return {r: self.prior_weight * math.log((wins.get(r, 0) + 1) / (total + len(self.rules)))
                for r in self.rules}

    def _forecast(self, seq, cutoff, base):
        ticks = sorted(seq)
        prior = [self.strength * b for b in base]
        forecasts, scores = [], []
        for rule, forget in itertools.product(self.rules, self.forget):
            tally = {}          # situation -> [weighted counts], each cell decays over time
            stamp = {}          # situation -> tick of last update (for lazy decay)
            score, last = 0.0, None
            for t in ticks:
                ctx = self._context(seq, t, rule)
                if ctx is None:
                    continue
                counts = tally.get(ctx)
                if counts is not None:
                    fade = forget ** (t - stamp[ctx])
                    counts = [c * fade for c in counts]
                else:
                    counts = [0.0] * N
                n = math.fsum(counts)
                p = (counts[INDEX[seq[t]]] + prior[INDEX[seq[t]]]) / (n + self.strength)
                if last is not None:
                    score *= self.trust_forget ** (t - last)
                score += math.log(p)
                last = t
                counts[INDEX[seq[t]]] += 1.0
                tally[ctx], stamp[ctx] = counts, t
            ctx = self._context(seq, cutoff, rule)
            if ctx is None:
                continue
            if last is not None:
                score *= self.trust_forget ** (cutoff - last)
            counts = tally.get(ctx)
            if counts is not None:
                fade = forget ** (cutoff - stamp[ctx])
                counts = [c * fade for c in counts]
            else:
                counts = [0.0] * N
            n = math.fsum(counts)
            forecasts.append([(c + pr) / (n + self.strength) for c, pr in zip(counts, prior)])
            scores.append(self.beta * score + self.rule_prior.get(rule, 0.0))
        if not forecasts:
            return list(base)
        top = max(scores)
        weights = [math.exp(s - top) for s in scores]
        total = math.fsum(weights)
        return [math.fsum(w * f[i] for w, f in zip(weights, forecasts)) / total for i in range(N)]

    @staticmethod
    def _context(seq, t, rule):
        bits = []
        for k in rule:
            v = seq.get(t - k)
            if v is None:   # missing tick = unknown
                return None
            bits.append(v > 0)
        return tuple(bits)

    def _learn_size_prior(self, queries, history):
        """Payout-size mix GIVEN a win (1x, 2x, 5x, 20x), per machine type, from training history."""
        types = {(q.episode_id, q.machine_id): q.machine_type for q in queries}
        counts = {}
        for play in history:
            if play.payout > 0:
                c = counts.setdefault(types[(play.episode_id, play.machine_id)], [0.0] * (N - 1))
                c[INDEX[play.payout] - 1] += 1.0
        self.size_prior = {t: [(v + 0.5) / (math.fsum(c) + 0.5 * (N - 1)) for v in c] for t, c in counts.items()}

    def _size_split(self, seq, cutoff, machine_type, fallback):
        """This machine's own payout-size mix (recent wins count more), blended toward its type's mix."""
        prior = self.size_prior.get(machine_type, fallback)
        counts = [0.0] * (N - 1)
        for t, y in seq.items():
            if y > 0:
                w = 1.0 if self.size_half_life is None else 2 ** (-(cutoff - 1 - t) / self.size_half_life)
                counts[INDEX[y] - 1] += w
        k = self.size_strength.get(machine_type, 20.0) if isinstance(self.size_strength, dict) else self.size_strength
        total = math.fsum(counts) + k
        return [(c + k * p) / total for c, p in zip(counts, prior)]

    def _size_rules(self, seq, cutoff, prior_split):
        """Same idea as the win rules, but for payout SIZE among wins: for each candidate rule
        (win bits or big-payout bits at a lag pair / single lag), tally the size of each win,
        grade the rule on wins it hadn't seen yet, and trust-weight the forecasts."""
        wins = sorted(t for t, v in seq.items() if v > 0)
        if len(wins) < self.size_rule_min_wins:
            return prior_split  # too few wins to learn a size rule (e.g. short histories)
        prior = [self.size_rule_strength * p for p in prior_split]
        forecasts, scores = [], []
        for feature in ("win", "big"):
            limit = 1 if feature == "win" else 2
            for rule in self.rules:
                if feature == "big" and not rule:
                    continue  # "no rule" only needs to appear once
                tally, score = {}, 0.0
                for t in wins:
                    ctx = self._context_bits(seq, t, rule, limit)
                    if ctx is None:
                        continue
                    counts = tally.setdefault(ctx, [0.0] * (N - 1))
                    i = INDEX[seq[t]] - 1
                    score += math.log((counts[i] + prior[i]) / (math.fsum(counts) + self.size_rule_strength))
                    counts[i] += 1.0
                ctx = self._context_bits(seq, cutoff, rule, limit)
                if ctx is None:
                    continue
                counts = tally.get(ctx, [0.0] * (N - 1))
                n = math.fsum(counts)
                forecasts.append([(c + p) / (n + self.size_rule_strength) for c, p in zip(counts, prior)])
                scores.append(self.size_rule_beta * score)
        if not forecasts:
            return prior_split
        top = max(scores)
        weights = [math.exp(s - top) for s in scores]
        total = math.fsum(weights)
        return [math.fsum(w * f[i] for w, f in zip(weights, forecasts)) / total for i in range(N - 1)]

    @staticmethod
    def _context_bits(seq, t, rule, limit):
        bits = []
        for k in rule:
            v = seq.get(t - k)
            if v is None:
                return None
            bits.append(v >= limit)
        return tuple(bits)

    def predict(self, queries, history):
        queries, history = tuple(queries), tuple(history)
        base_preds = self.base.predict(queries, history)
        indexed = index_history(queries, history)  # also rejects any future leak
        out = {}
        for q in queries:
            seq = {p.tick: p.payout for p in indexed.get((q.episode_id, q.machine_id), [])}
            base = base_preds[q.query_id]
            if self.binary and self.size_strength is not None:
                # rules only judge WIN vs LOSE (payout size is handled separately below)
                wl = {t: int(v > 0) for t, v in seq.items()}  # 0 -> index 0, 1 -> index 1
                p = self._forecast(wl, q.cutoff, [base[0], 1.0 - base[0], 0.0, 0.0, 0.0])
                p = [p[0]] + [p[1] * b / (1.0 - base[0]) for b in base[1:]]  # placeholder split
            else:
                p = self._forecast(seq, q.cutoff, base)
            if self.size_strength is not None:
                # part 1 (rules) says WILL it pay; part 2 says HOW MUCH if it does
                win = 1.0 - p[0]
                rule_split = [v / win for v in p[1:]] if win > 0 else [0.25] * 4
                split = self._size_split(seq, q.cutoff, q.machine_type, rule_split)
                if self.size_rule_strength is not None:
                    split = self._size_rules(seq, q.cutoff, split)
                p = [p[0]] + [win * s for s in split]
            p = [max(self.floor, v) for v in p]
            s = math.fsum(p)
            out[q.query_id] = tuple(v / s for v in p)
        return out
