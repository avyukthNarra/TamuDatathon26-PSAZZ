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
# Payout-size experts: machine-level plus lags 1, (1,2), 2 and (2,3), each cell keyed on the
# EXACT earlier payout (size_cell="class"). Idea from a second team's rule model, which keys
# size rules on big/small earlier payouts. Train cutoff dR -0.0145 [-0.029,-0.001], validation
# 53.75 -> 54.21. Keyed on win/loss only (the old default) these experts can't see payout size.
DEFAULT_SIZE_KEYS = ((), (1,), (1, 2), (2,), (2, 3))
# With size_modes (the two exact size mixes per type), the size experts are keyed on single lags
# 1-4: on main train the size mode is set by whether the spin 1, 2 or 4 back was a big win.
# Train cutoff dR -0.023 [-0.042, -0.001]; validation 54.21 -> 54.90 (tuning/audit halves 54.44 / 55.43).
MODE_SIZE_KEYS = ((), (1,), (2,), (3,), (4,))
MIN_PROBABILITY = 1e-4


class PairMixer:
    # Defaults: kept only if better on BOTH train cutoff labels (tune_pair_mixer.py) and
    # validation truth. half_life (12, 40) = a fast and a slow expert per pair.
    def __init__(self, pairs=DEFAULT_PAIRS, pair_prior=None, half_life=(12.0, 40.0), k_win=0.25, eta=16.0,
                 size_keys=None, size_half_life=80.0, k_size=16.0, k_machine=32.0,
                 size_eta=8.0, type_size_priors=None, rate_prior=(1.0, 2.0), cell_mode="parity",
                 decay_missing=True, alpha=0.0, size_cell="class", size_modes=None, mode_temp=1.0):
        # size_modes: {machine_type: (small_mix, big_mix)}, the two exact payout-size mixes
        #   (1x, 2x, 5x, 20x) a machine of that type switches between (size_modes_from_targets).
        #   Each size cell then holds one number, P(big mode), from its decayed size counts by
        #   Bayes' rule (log-likelihood ratio x mode_temp), instead of 4 free frequencies, so it
        #   learns from far fewer wins. Without modes the size experts use free frequencies.
        self.size_modes = {t: (tuple(a), tuple(b)) for t, (a, b) in (size_modes or {}).items()}
        self.mode_temp = mode_temp
        if size_keys is None:
            size_keys = MODE_SIZE_KEYS if self.size_modes else DEFAULT_SIZE_KEYS
        # size_cell: what a size expert's cell records at each lag: "win" (paid or not),
        # "big" (lost / paid 1-2x / paid 5-20x) or "class" (the exact payout).
        self.pairs = tuple(tuple(p) for p in pairs)
        self.pair_prior = list(pair_prior) if pair_prior is not None else [0.0] * len(self.pairs)
        self.k_win = k_win
        self.eta = eta
        self.size_keys = tuple(tuple(k) for k in size_keys)
        # size_cell may also be a list with one entry per size key.
        self.size_cell = (tuple(size_cell) if isinstance(size_cell, (list, tuple))
                          else (size_cell,) * len(self.size_keys))
        if len(self.size_cell) != len(self.size_keys) or any(m not in ("win", "big", "class") for m in self.size_cell):
            raise ValidationError("size_cell must be win, big or class (one per size key if a list).")
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
        # decay_missing=False: memory only fades on observed ticks, so the 24 sampled
        # older ticks of a short history are not wiped out by the gap before the recent window.
        self.decay_missing = decay_missing
        # alpha > 0: Fixed-Share (Herbster & Warmuth 1998); every expert keeps at least
        # alpha of its prior weight, so a machine that switches pair/rule can switch experts.
        self.alpha = alpha
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
        modes = self.size_modes.get(machine_type)
        mode_llr = [math.log(max(b, 1e-3) / max(a, 1e-3)) for a, b in zip(*modes)] if modes else None
        a0, b0 = self.rate_prior

        def cell_of(t, lags, how="win"):
            values = []
            for lag in lags:
                v = own[t - lag] if t - lag >= 0 else None
                if v is None:
                    return None
                if how == "win":
                    values.append(v > 0)
                elif how == "big":
                    values.append(0 if v == 0 else 1 if v <= 2 else 2)  # lost / 1-2x / 5-20x
                else:
                    values.append(v)
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
            size_cells = [cell_of(t, key, how) for key, how in zip(self.size_keys, self.size_cell)]
            size_preds = []
            for i, cell in enumerate(size_cells):
                sc = size_tab[i].get(cell) if cell is not None else None
                if modes is not None:
                    counts = sc if sc is not None else machine_size
                    llr = self.mode_temp * math.fsum(c * r for c, r in zip(counts, mode_llr))
                    big = 1.0 / (1.0 + math.exp(-max(min(llr, 50.0), -50.0)))
                    size_preds.append([(1.0 - big) * a + big * b for a, b in zip(*modes)])
                elif sc is None:
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
                if self.alpha > 0:
                    top = max(logw)
                    raw = [math.exp(v - top) for v in logw]
                    total = math.fsum(raw)
                    ptop = max(prior)
                    praw = [math.exp(v - ptop) for v in prior]
                    ptotal = math.fsum(praw)
                    logw = [math.log((1 - self.alpha) * r / total + self.alpha * q / ptotal)
                            for r, q in zip(raw, praw)]
                if value > 0:
                    for i, sp in enumerate(size_preds):
                        size_logw[i] -= self.size_eta * math.fsum(
                            (q - (1.0 if k == value - 1 else 0.0)) ** 2 for k, q in enumerate(sp))
            if value is None and not self.decay_missing:
                continue
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


def choose_pairs(queries, history, min_share=0.3):
    """Lag pairs used by many machines: count >= min_share x the most common pair's count."""
    found = discover_pairs(queries, history)
    if not found:
        return DEFAULT_PAIRS
    top = found[0][1]
    return tuple(sorted(pair for pair, count in found if count >= min_share * top))


def size_modes_from_targets(targets, queries):
    """The two exact payout-size mixes per machine type, read from validation targets.csv.

    Every true size-given-win vector in the main round is one of two per type (a "small"
    and a "big" mode). Returns {type: (small, big)}, small = the mode with more 1x, or {}
    if any type doesn't show exactly two clear modes (the model then uses free frequencies).
    """
    machine_type = {q.query_id: q.machine_type for q in queries}
    seen = {}
    for target in targets:
        p = target.probabilities
        win = 1.0 - p[0]
        if win < 0.05 or target.query_id not in machine_type:
            continue
        mix = tuple(round(x / win, 3) for x in p[1:])
        seen.setdefault(machine_type[target.query_id], Counter())[mix] += 1
    modes = {}
    for kind, counts in seen.items():
        groups = []  # merge vectors that differ only by rounding, then keep the two biggest groups
        for mix, n in counts.most_common():
            for group in groups:
                if max(abs(a - b) for a, b in zip(group[0], mix)) < 0.02:
                    group[1] += n
                    break
            else:
                groups.append([mix, n])
        groups.sort(key=lambda g: -g[1])
        if len(groups) < 2 or sum(g[1] for g in groups[2:]) > 0.05 * sum(g[1] for g in groups):
            return {}
        small, big = sorted((groups[0][0], groups[1][0]), key=lambda m: -m[0])
        modes[kind] = (tuple(x / sum(small) for x in small), tuple(x / sum(big) for x in big))
    return modes


def cutoff_brier(model, queries, history, labels):
    """Mean 5-class Brier vs labels.csv outcomes at each query's cutoff tick."""
    predictions = model.predict(queries, history)
    return math.fsum(math.fsum((p - (1.0 if PAYOUTS[k] == labels[q] else 0.0)) ** 2
                               for k, p in enumerate(predictions[q])) for q in labels) / len(labels)


def adapt(queries, history, labels=None, log=print, size_modes=None):
    """Round-two helper: rediscover the lag pairs and pick the cell mode on this round's train data.

    size_modes: from size_modes_from_targets on this round's validation targets, if available.
    """
    pairs = choose_pairs(queries, history)
    log(f"lag pairs found on train: {pairs}")
    log(f"payout-size modes: {'two per type, from validation targets' if size_modes else 'none (free frequencies)'}")
    candidates = [PairMixer(pairs=pairs, cell_mode=mode, size_modes=size_modes).fit(queries, history)
                  for mode in ("parity", "full")]
    if labels is None:
        return candidates[0]
    scores = [cutoff_brier(model, queries, history, labels) for model in candidates]
    for model, score in zip(candidates, scores):
        log(f"cell_mode={model.cell_mode[0]}: train cutoff Brier {score:.5f}")
    return candidates[scores.index(min(scores))]
