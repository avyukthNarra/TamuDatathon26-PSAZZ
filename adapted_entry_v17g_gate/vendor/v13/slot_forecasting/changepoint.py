"""Regime model: Bayesian online change-point detection over the machine's win rule.

A different approach from the pair mixer (decaying tallies + expert weights) and the
hidden-rule HMM (fixed type levels). For each machine we keep an exact-ish posterior over
the hidden story behind its spins:

  segment start s   when the current regime began (Adams & MacKay 2007 run length)
  rule k            which lag pair (a, b) drives wins in this regime, or "no rule"
  cell levels       each cell's own win level, never forgotten within the regime:
                    the two parity cells of pair (a, b) ("the spins a and b back agree" /
                    "disagree"), or one cell for "no rule"

Each cell's level has the Beta-mixture prior of the machine's type and signal (dead / live /
middle levels, from win_prior_by_type_from_targets), so the predictive for a cell with w wins in
n spins since the regime began is that prior's posterior mean (exact, integer counts, cached).
A regime ends with hazard H per tick (a gap of g ticks: 1 - (1 - H)^g), after which the rule and
the levels are redrawn. Hypotheses (s, k) are weighted by how well they predicted every spin
since s; the lowest-weight segments are pruned to keep the work bounded.

The win chance at the cutoff is the posterior-weighted predictive. Payout sizes come from the
pair mixer. Only ticks strictly before the cutoff are read. Standard library only, deterministic.
"""
import math
from functools import lru_cache

from .schema import index_history, unique_queries

HAZARD = 0.001
MAX_SEGMENTS = 24


class RegimeModel:
    def __init__(self, pairs, win_prior, hazard=HAZARD, max_segments=MAX_SEGMENTS, no_rule_weight=0.1, coupled=True, temper=1.0):
        self.pairs = [tuple(p) for p in pairs]
        self.hazard = hazard
        self.max_segments = max_segments
        # coupled: each pair rule comes in two directions ("agree" cell live / "differ" cell live);
        #   the live cell's level has the prior's live + middle components, the dead cell's the
        #   dead component, so seeing one cell pay tells the model the other is dead.
        self.coupled = coupled
        # temper < 1 flattens how fast hypotheses are ruled out (hedges between rules on short data)
        self.temper = temper
        self.rules = [(k, d) for k in range(len(self.pairs)) for d in ((0, 1) if coupled else (None,))]
        rule_weight = (1.0 - no_rule_weight) / len(self.rules)
        self.rules.append((len(self.pairs), None))
        self.rule_prior = [rule_weight] * (len(self.rules) - 1) + [no_rule_weight]
        # prior components per key: [(log weight, a, b)], a = mean * strength
        self.priors = {}
        for key, comps in win_prior.items():
            parts = {"all": comps, "dead": comps[:1], "live": comps[1:]}
            for role, sub in parts.items():
                total = math.fsum(w for w, _, _ in sub)
                self.priors[(key, role)] = tuple((math.log(w / total), m * k, (1.0 - m) * k) for w, m, k in sub)
        self._rate = lru_cache(maxsize=None)(self._rate_uncached)

    def _rate_uncached(self, key, w, n):
        """Posterior mean win level of a cell with w wins in n spins under the key's Beta mixture."""
        logs, means = [], []
        for lw, a, b in self.priors[key]:  # key = (prior key, role)
            logs.append(lw + math.lgamma(a + w) + math.lgamma(b + n - w) - math.lgamma(a + b + n)
                        - math.lgamma(a) - math.lgamma(b) + math.lgamma(a + b))
            means.append((a + w) / (a + b + n))
        top = max(logs)
        post = [math.exp(v - top) for v in logs]
        return math.fsum(p * m for p, m in zip(post, means)) / math.fsum(post)

    def _cell(self, seq, t, k):
        """Cell of rule k at tick t: 0/1 for the parity cells, 2 for "no rule", None if a lag is missing."""
        if k == len(self.pairs):
            return 2
        a, b = self.pairs[k]
        if t - a not in seq or t - b not in seq:
            return None
        return 0 if (seq[t - a] > 0) == (seq[t - b] > 0) else 1

    def win_probability(self, seq, cutoff, key):
        rate = self._rate
        n_rules = len(self.rules)

        def roles(direction, cell):
            if direction is None:
                return "all"
            return "live" if cell == direction else "dead"

        # hypotheses: [log weight, rule index, counts [w0, n0, w1, n1, w2, n2]]
        hyps = [[math.log(p), i, [0, 0, 0, 0, 0, 0]] for i, p in enumerate(self.rule_prior)]
        last = None

        def predict(t):
            preds, cells = [], []
            for _, i, c in hyps:
                k, d = self.rules[i]
                cell = self._cell(seq, t, k)
                cells.append(cell)
                if cell is None:  # lag missing: the regime's pooled level
                    preds.append(rate((key, "all"), c[0] + c[2], c[1] + c[3]))
                else:
                    preds.append(rate((key, roles(d, cell)), c[2 * cell], c[2 * cell + 1]))
            return preds, cells

        def changepoint(gap):
            if gap <= 0:
                return
            p_change = 1.0 - (1.0 - self.hazard) ** gap
            top = max(h[0] for h in hyps)
            log_new = top + math.log(math.fsum(math.exp(h[0] - top) for h in hyps)) + math.log(p_change)
            keep = math.log(1.0 - p_change)
            for h in hyps:
                h[0] += keep
            hyps.extend([log_new + math.log(p), i, [0, 0, 0, 0, 0, 0]] for i, p in enumerate(self.rule_prior))

        for t in sorted(seq):
            changepoint(t - last if last is not None else 0)
            last = t
            won = 1 if seq[t] > 0 else 0
            preds, cells = predict(t)
            for h, p, cell in zip(hyps, preds, cells):
                h[0] += self.temper * math.log(p if won else 1.0 - p)
                if cell is not None:
                    h[2][2 * cell] += won
                    h[2][2 * cell + 1] += 1
            if len(hyps) > self.max_segments * n_rules:
                hyps.sort(key=lambda h: -h[0])
                del hyps[self.max_segments * n_rules:]
            top = max(h[0] for h in hyps)
            for h in hyps:
                h[0] -= top
        changepoint(cutoff - last if last is not None else 0)
        preds, _ = predict(cutoff)
        top = max(h[0] for h in hyps)
        ws = [math.exp(h[0] - top) for h in hyps]
        return math.fsum(w * p for w, p in zip(ws, preds)) / math.fsum(ws)

    def win_chances(self, queries, history):
        indexed = index_history(queries, history)
        out = {}
        for query in unique_queries(queries).values():
            key = f"{query.machine_type}|{query.signal}"
            if (key, "all") not in self.priors:
                key = query.machine_type
            seq = {p.tick: p.payout for p in indexed.get((query.episode_id, query.machine_id), [])}
            out[query.query_id] = self.win_probability(seq, query.cutoff, key)
        return out
