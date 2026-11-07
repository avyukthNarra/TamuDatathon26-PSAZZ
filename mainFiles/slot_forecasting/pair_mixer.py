"""Per-machine online mixture of lag-pair experts.

Why: in this simulator a machine's next win chance depends on the *combination*
of its own results at two earlier lags (on main-round train data the pairs
(1,3), (4,7) and (2,5) dominate, each used by ~1/3 of machines), and that
mapping changes over time. A straight-line model on single lags cannot see a
pair interaction, and a table learned once goes stale.

How: every candidate pair is an "expert". For each machine, each expert keeps
exponentially decayed counts of outcomes in each cell of its pair (won/lost at
lag a x won/lost at lag b). Its forecast is that cell's decayed payout mix,
smoothed toward the machine's own recent mix. Experts are combined with
exponential weights on their recent squared error (Vovk / Herbster-Warmuth
style), so each machine settles on its own pair and can switch if it changes.

Everything is online: the forecast for tick t is produced before tick t is
absorbed, so it only uses ticks < t. Training-time evaluation and test-time
prediction run the same `walk` function. Standard library only.
"""

import itertools
import math
from collections import Counter

from .schema import PAYOUTS, ValidationError, index_history, probabilities, unique_queries

CLASSES = len(PAYOUTS)
CLASS_INDEX = {value: index for index, value in enumerate(PAYOUTS)}
ALL_PAIRS = tuple(itertools.combinations(range(1, 9), 2))
DEFAULT_PAIRS = ((1, 3), (4, 7), (2, 5))
DEFAULT_SIZE_KEYS = ((),)  # lag-keyed size tables did not help (size mode looks random)
MIN_PROBABILITY = 1e-4


class PairMixer:
    # Defaults = best on train cutoff labels, confirmed once on validation (53.35 / 60).
    def __init__(self, pairs=DEFAULT_PAIRS, pair_prior=None, half_life=14.0, k_win=0.25, eta=16.0,
                 size_keys=DEFAULT_SIZE_KEYS, size_half_life=40.0, k_size=32.0, k_machine=32.0,
                 size_eta=8.0, type_size_priors=None, rate_prior=(1.0, 2.0), cell_mode="parity"):
        self.pairs = tuple(tuple(p) for p in pairs)
        self.pair_prior = list(pair_prior) if pair_prior is not None else [0.0] * len(self.pairs)
        self.k_win = k_win
        self.eta = eta
        self.size_keys = tuple(tuple(k) for k in size_keys)
        self.size_half_life = size_half_life
        self.k_size = k_size
        self.k_machine = k_machine
        self.size_eta = size_eta
        self.type_size_priors = type_size_priors or {}
        self.rate_prior = rate_prior
        named = {"parity": ("parity",), "full": ("full",), "both": ("parity", "full")}
        modes = named.get(cell_mode, (cell_mode,)) if isinstance(cell_mode, str) else tuple(cell_mode)
        if not modes or any(m not in ("parity", "full", "sym") for m in modes):
            raise ValidationError("cell_mode must use parity, full and/or sym.")
        self.cell_mode = modes
        # half_life may be one number or several; each becomes its own set of experts.
        self.half_lives = tuple(half_life) if isinstance(half_life, (list, tuple)) else (half_life,)

    # ---- fitting: only the population size prior per machine type is learned from data ----
    def fit(self, queries, history, labels=None):
        counts = {}
        machine_type = {(q.episode_id, q.machine_id): q.machine_type for q in queries}
        for play in history:
            if play.payout > 0:
                row = counts.setdefault(machine_type[(play.episode_id, play.machine_id)], [0.5] * (CLASSES - 1))
                row[CLASS_INDEX[play.payout] - 1] += 1.0
        self.type_size_priors = {t: [c / math.fsum(v) for c in v] for t, v in counts.items()}
        return self

    def _size_prior(self, machine_type):
        return self.type_size_priors.get(machine_type, [0.25] * (CLASSES - 1))

    def walk(self, own, machine_type, stop, emit):
        """own[t] = payout class index or None. Calls emit(t, probs) for t in 0..stop before absorbing t.

        Win chance: mixture over lag-pair experts (decayed win table per pair cell).
        Payout size given a win: separate mixture over size experts keyed by
        () = machine-level, (a,) = one lag, (a, b) = a lag pair.
        """
        size_decay = 2 ** (-1.0 / self.size_half_life)
        sizes = CLASSES - 1
        size_tab = [dict() for _ in self.size_keys]      # cell -> [4 size counts]
        base_decay = 2 ** (-1.0 / self.half_lives[0])
        rate = [0.0, 0.0]
        machine_size = [0.0] * sizes
        # Win experts = (pair, mode, decay). Modes:
        #   full   - 4 cells (won/lost at lag a x won/lost at lag b)
        #   parity - 2 cells: did the two lags agree? (the simulator's rule is XOR/XNOR-like)
        #   sym    - 1 number: evidence for "win iff agree" vs "win iff differ", pooled over both cells
        experts, prior = [], []
        for pair, weight in zip(self.pairs, self.pair_prior):
            for mode in self.cell_mode:
                for half_life in self.half_lives:
                    experts.append((pair, mode, 2 ** (-1.0 / half_life)))
                    prior.append(weight)
        win_tab = [dict() for _ in experts]              # key -> [wins (or xnor evidence), n]
        logw = list(prior)
        size_logw = [0.0] * len(self.size_keys)
        type_size = self._size_prior(machine_type)
        a0, b0 = self.rate_prior

        def cell_of(t, lags):
            values = []
            for lag in lags:
                v = own[t - lag] if t - lag >= 0 else None
                if v is None:
                    return None
                values.append(v > 0)
            return tuple(values)

        for t in range(stop + 1):
            base = (rate[0] + a0) / (rate[1] + b0)
            msum = math.fsum(machine_size)
            msize = [(c + self.k_machine * p) / (msum + self.k_machine) for c, p in zip(machine_size, type_size)]
            pair_cells = {pair: cell_of(t, pair) for pair in self.pairs}
            win_keys, win_preds = [], []
            for i, (pair, mode, _) in enumerate(experts):
                cell = pair_cells[pair]
                if cell is None:
                    win_keys.append(None)
                    win_preds.append(base)
                    continue
                agree = cell[0] == cell[1]
                if mode == "sym":
                    e, n = win_tab[i].get("sym", (0.0, 0.0))
                    q = (e + self.k_win * 0.5) / (n + self.k_win)
                    win_keys.append(("sym", agree))
                    win_preds.append(q if agree else 1.0 - q)
                    continue
                key = (agree,) if mode == "parity" else cell
                w, n = win_tab[i].get(key, (0.0, 0.0))
                win_keys.append(key)
                win_preds.append((w + self.k_win * base) / (n + self.k_win))
            size_cells = [cell_of(t, key) for key in self.size_keys]
            size_preds = []
            for i, cell in enumerate(size_cells):
                sc = size_tab[i].get(cell) if cell is not None else None
                if sc is None:
                    size_preds.append(msize)
                else:
                    ssum = math.fsum(sc)
                    size_preds.append([(c + self.k_size * m) / (ssum + self.k_size) for c, m in zip(sc, msize)])
            top = max(logw)
            weights = [math.exp(v - top) for v in logw]
            pw = math.fsum(w * p for w, p in zip(weights, win_preds)) / math.fsum(weights)
            top = max(size_logw)
            sweights = [math.exp(v - top) for v in size_logw]
            stotal = math.fsum(sweights)
            size = [math.fsum(w * sp[k] for w, sp in zip(sweights, size_preds)) / stotal for k in range(sizes)]
            emit(t, [1.0 - pw] + [pw * s for s in size])
            if t == stop:
                break
            value = own[t]
            if value is not None:
                won = 1.0 if value > 0 else 0.0
                for i, p in enumerate(win_preds):
                    logw[i] -= self.eta * (p - won) ** 2
                if value > 0:
                    for i, sp in enumerate(size_preds):
                        size_logw[i] -= self.size_eta * math.fsum(
                            (q - (1.0 if k == value - 1 else 0.0)) ** 2 for k, q in enumerate(sp))
            for (_, _, decay), tab in zip(experts, win_tab):
                for c in tab.values():
                    c[0] *= decay
                    c[1] *= decay
            for tab in size_tab:
                for c in tab.values():
                    for k in range(sizes):
                        c[k] *= size_decay
            rate[0] *= base_decay
            rate[1] *= base_decay
            for k in range(sizes):
                machine_size[k] *= size_decay
            if value is None:
                continue
            won = 1.0 if value > 0 else 0.0
            rate[0] += won
            rate[1] += 1.0
            for i, key in enumerate(win_keys):
                if key is None:
                    continue
                if key[0] == "sym":
                    c = win_tab[i].setdefault("sym", [0.0, 0.0])
                    c[0] += 1.0 if (won == 1.0) == key[1] else 0.0
                else:
                    c = win_tab[i].setdefault(key, [0.0, 0.0])
                    c[0] += won
                c[1] += 1.0
            if value > 0:
                machine_size[value - 1] += 1.0
                for i, cell in enumerate(size_cells):
                    if cell is not None:
                        size_tab[i].setdefault(cell, [0.0] * sizes)[value - 1] += 1.0

    def predict(self, queries, history, baseline=None):
        query_map = unique_queries(queries)
        indexed = index_history(queries, history)
        result = {}
        for query in query_map.values():
            own = [None] * (query.cutoff + 1)
            for play in indexed.get((query.episode_id, query.machine_id), []):
                own[play.tick] = CLASS_INDEX[play.payout]
            captured = {}
            self.walk(own, query.machine_type, query.cutoff, lambda t, p: captured.__setitem__(t, p))
            probs = [max(p, MIN_PROBABILITY) for p in captured[query.cutoff]]
            total = math.fsum(probs)
            result[query.query_id] = probabilities([p / total for p in probs], query.query_id)
        return result

    def outcome_brier(self, queries, history, start=40):
        """Mean 5-class Brier score vs observed outcomes over every observed tick >= start.

        In expectation this equals squared error vs the true probabilities plus a
        constant, so it ranks settings like the competition metric does.
        """
        indexed = index_history(queries, history)
        query_map = unique_queries(queries)
        errors = []
        for query in query_map.values():
            own = [None] * query.cutoff
            for play in indexed.get((query.episode_id, query.machine_id), []):
                own[play.tick] = CLASS_INDEX[play.payout]

            def emit(t, probs, own=own):
                if t >= start and own[t] is not None:
                    errors.append(math.fsum((p - (1.0 if k == own[t] else 0.0)) ** 2 for k, p in enumerate(probs)))
            if query.cutoff > 0:
                self.walk(own, query.machine_type, query.cutoff - 1, emit)
        if not errors:
            raise ValidationError("No ticks to score.")
        return math.fsum(errors) / len(errors)


def discover_pairs(queries, history, top=None):
    """Count, per machine, which lag pair separates its win rate best; return pair frequencies."""
    indexed = index_history(queries, history)
    winners = Counter()
    for plays in indexed.values():
        wins = {p.tick: p.payout > 0 for p in plays}
        best, best_score = None, -1.0
        for a, b in ALL_PAIRS:
            cells = {}
            for t, won in wins.items():
                if t - a in wins and t - b in wins:
                    c = cells.setdefault((wins[t - a], wins[t - b]), [0, 0])
                    c[0] += won
                    c[1] += 1
            n = sum(c[1] for c in cells.values())
            if n < 20:
                continue
            mean = sum(c[0] for c in cells.values()) / n
            score = sum(c[1] * (c[0] / c[1] - mean) ** 2 for c in cells.values()) / n
            if score > best_score:
                best, best_score = (a, b), score
        if best is not None:
            winners[best] += 1
    return winners.most_common(top)
