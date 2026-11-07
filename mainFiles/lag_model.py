"""Per-machine lag model with shrinkage, built on top of pooled_recent.

Plain-English version (see MODEL_PLAN.md section 7):
  1. Start from the safe guess (pooled_recent).
  2. For each lag k, tally from this machine's OWN history:
     "when payout X happened k spins earlier, what came next?"
  3. Blend each tally with the safe guess (S pretend examples) so tiny tallies don't overreact.
  4. Look up the payout at each lag before the cutoff; each gives a clue. Combine the clues.
  5. Optional nudge from the other machines in the episode.
  6. Floor tiny probabilities and renormalise.

Only ticks strictly below each machine's cutoff are ever read. Standard library only.
"""
import math
from collections import defaultdict

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.schema import PAYOUTS, index_history

N = len(PAYOUTS)  # 5 payouts: 0, 1, 2, 5, 20
INDEX = {p: i for i, p in enumerate(PAYOUTS)}


def bucket(payout, buckets):
    """How finely to split the 'what happened k spins ago' situation."""
    if buckets == 5:
        return INDEX[payout]
    if buckets == 3:
        return 0 if payout == 0 else 1 if payout == 1 else 2
    return 0 if payout == 0 else 1  # 2 buckets: lose / win


class LagForecaster:
    def __init__(self, lags=(1, 2, 3, 4, 5, 6, 7, 8), strength=10.0, base_weight=1.0,
                 lag_weights=None, combine="log", buckets=3, tally_half_life=None,
                 base_strength=20.0, base_half_life=24.0, floor=1e-3,
                 neighbour=0.0, neighbour_window=16, select=None):
        self.lags = tuple(lags)
        self.select = select  # (g0, scale): per-machine lag selection by evidence, or None
        self.strength = float(strength)
        self.base_weight = float(base_weight)
        self.lag_weights = dict(lag_weights or {})
        self.combine = combine
        self.buckets = buckets
        self.tally_half_life = tally_half_life
        self.floor = floor
        self.neighbour = neighbour
        self.neighbour_window = neighbour_window
        self.base = FrequencyForecaster("pooled_recent", prior_strength=base_strength, half_life=base_half_life)

    def settings(self):
        return {k: v for k, v in vars(self).items() if k != "base"} | {
            "base_strength": self.base.prior_strength, "base_half_life": self.base.half_life}

    def fit(self, queries, history, labels=None):
        self.base.fit(queries, history, labels)
        return self

    # --- step 2 + 3: one clue per lag -------------------------------------------------
    def _table(self, seq, k, cutoff):
        """Tally sheet for lag k: table[bucket of payout k spins earlier] = counts of next payout."""
        table = defaultdict(lambda: [0.0] * N)
        for t, y in seq.items():
            x = seq.get(t - k)
            if x is None:
                continue
            w = 1.0 if self.tally_half_life is None else 2 ** (-(cutoff - 1 - t) / self.tally_half_life)
            table[bucket(x, self.buckets)][INDEX[y]] += w
        return table

    @staticmethod
    def _evidence(table):
        """G-test statistic: how strongly this lag's tally differs from 'no habit at all'."""
        rows = {x: math.fsum(c) for x, c in table.items()}
        cols = [math.fsum(c[i] for c in table.values()) for i in range(N)]
        n = math.fsum(rows.values())
        if n <= 0:
            return 0.0
        return 2 * math.fsum(c[i] * math.log(c[i] * n / (rows[x] * cols[i]))
                             for x, c in table.items() for i in range(N) if c[i] > 0)

    def _clues(self, seq, cutoff, base):
        clues = []
        for k in self.lags:
            prev = seq.get(cutoff - k)
            if prev is None:  # missing tick = unknown, skip this clue
                continue
            table = self._table(seq, k, cutoff)
            counts = table.get(bucket(prev, self.buckets), [0.0] * N)
            n = math.fsum(counts)
            clue = [(c + self.strength * b) / (n + self.strength) for c, b in zip(counts, base)]
            weight = self.lag_weights.get(k, 1.0)
            if self.select is not None:
                # trust a lag only as much as its habit stands out from chance
                g0, scale = self.select
                weight *= min(1.0, max(0.0, (self._evidence(table) - g0) / scale))
            if weight > 0:
                clues.append((weight, clue))
        return clues

    # --- step 4: combine ---------------------------------------------------------------
    def _combine(self, base, clues):
        if not clues:
            return list(base)
        if self.combine == "linear":
            total = self.base_weight + math.fsum(w for w, _ in clues)
            return [(self.base_weight * b + math.fsum(w * c[i] for w, c in clues)) / total
                    for i, b in enumerate(base)]
        # "log": start at the base, each clue multiplies in how much it differs from the base
        logs = [math.log(b) + math.fsum(w * (math.log(c[i]) - math.log(b)) for w, c in clues)
                for i, b in enumerate(base)]
        top = max(logs)
        exp = [math.exp(v - top) for v in logs]
        return [v / math.fsum(exp) for v in exp]

    # --- step 5: neighbours ----------------------------------------------------------
    def _neighbour_shift(self, query, seqs, episode_members):
        """How much the other machines' recent win rate is above their long-run win rate."""
        diffs = []
        for other in episode_members[query.episode_id]:
            if other.machine_id == query.machine_id:
                continue
            seq = seqs[(other.episode_id, other.machine_id)]
            seen = [y for t, y in seq.items() if t < query.cutoff]
            recent = [y for t, y in seq.items() if query.cutoff - self.neighbour_window <= t < query.cutoff]
            if len(seen) >= 20 and len(recent) >= 4:
                diffs.append(sum(y > 0 for y in recent) / len(recent) - sum(y > 0 for y in seen) / len(seen))
        return math.fsum(diffs) / len(diffs) if diffs else 0.0

    def predict(self, queries, history):
        queries, history = tuple(queries), tuple(history)
        base_preds = self.base.predict(queries, history)
        indexed = index_history(queries, history)  # also rejects any future leak
        seqs = {key: {p.tick: p.payout for p in plays} for key, plays in indexed.items()}
        for q in queries:
            seqs.setdefault((q.episode_id, q.machine_id), {})
        members = defaultdict(list)
        for q in queries:
            members[q.episode_id].append(q)
        out = {}
        for q in queries:
            base = base_preds[q.query_id]
            seq = seqs[(q.episode_id, q.machine_id)]
            p = self._combine(base, self._clues(seq, q.cutoff, base))
            if self.neighbour:
                shift = self.neighbour * self._neighbour_shift(q, seqs, members)
                # move probability mass between "lose" and the win payouts, keeping win shape
                win = 1 - p[0]
                new_win = min(0.999, max(0.001, win + shift))
                p = [1 - new_win] + [v * new_win / win for v in p[1:]] if win > 0 else p
            p = [max(self.floor, v) for v in p]
            s = math.fsum(p)
            out[q.query_id] = tuple(v / s for v in p)
        return out
