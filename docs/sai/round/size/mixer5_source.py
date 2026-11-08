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

import hashlib
import itertools
import math
from collections import Counter

from slot_forecasting.schema import PAYOUTS, ValidationError, index_history, probabilities, unique_queries

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
# With a win_prior (true win chances are U-shaped: most near a "dead" or a "live" level), the
# expert weights learn best with eta 8. Validation 54.90 -> 55.34 (halves 55.03 / 55.71).
WIN_PRIOR_ETA = 6.0
MIN_PROBABILITY = 1e-4


def _normalize_size_modes(spec):
    if isinstance(spec, dict):
        modes = tuple(tuple(float(x) for x in mode) for mode in spec["modes"])
        weights = tuple(float(x) for x in spec.get("weights", [1.0] * len(modes)))
    else:
        modes = tuple(tuple(float(x) for x in mode) for mode in spec)
        weights = tuple([1.0 / len(modes)] * len(modes))
    if not modes or len(weights) != len(modes) or any(w <= 0 for w in weights):
        raise ValidationError("size_modes needs one positive prior weight per mode.")
    if any(len(mode) != CLASSES - 1 or any(x < 0 for x in mode) or math.fsum(mode) <= 0 for mode in modes):
        raise ValidationError("Every size mode must be a nonempty four-class probability vector.")
    weights_sum = math.fsum(weights)
    normalized_weights = tuple(w / weights_sum for w in weights)
    normalized_modes = tuple(tuple(x / math.fsum(mode) for x in mode) for mode in modes)
    return {"modes": normalized_modes, "weights": normalized_weights}


def size_mode_mixture(counts, modes, priors, temperature=1.0, floor=1e-3):
    """Bayesian posterior mixture over any number of payout-size modes."""
    log_post = [math.log(prior) + temperature * math.fsum(
        count * math.log(max(probability, floor)) for count, probability in zip(counts, mode))
        for mode, prior in zip(modes, priors)]
    top = max(log_post)
    weights = [math.exp(value - top) for value in log_post]
    total = math.fsum(weights)
    return [math.fsum(weight * mode[k] for weight, mode in zip(weights, modes)) / total
            for k in range(CLASSES - 1)]


class PairMixer:
    # Defaults: kept only if better on BOTH train cutoff labels (tune_pair_mixer.py) and
    # validation truth. half_life (12, 40) = a fast and a slow expert per pair.
    def __init__(self, pairs=DEFAULT_PAIRS, pair_prior=None, half_life=(12.0, 40.0), k_win=0.25, eta=16.0,
                 size_keys=None, size_half_life=80.0, k_size=16.0, k_machine=32.0,
                 size_eta=8.0, type_size_priors=None, rate_prior=(1.0, 2.0), cell_mode="parity",
                 decay_missing=True, alpha=0.0, size_cell="class", size_modes=None, mode_temp=1.0, win_prior=None):
        # win_prior: [(weight, mean, strength), ...] - a mixture of Beta priors for each win cell's
        #   rate (e.g. a "dead" level near 0.03 and a "live" level near 0.93). A cell's estimate is
        #   its posterior mean, pulled toward whichever level its counts favour, instead of toward
        #   the machine's overall win rate. None = the k_win / base smoothing.
        def prepare_win_prior(components):
            return [(float(w), float(m) * float(k), (1 - float(m)) * float(k))
                    for w, m, k in components]
        if isinstance(win_prior, dict):
            self.win_prior = None
            self.win_priors = {kind: prepare_win_prior(components)
                               for kind, components in win_prior.items()}
        else:
            self.win_prior = prepare_win_prior(win_prior) if win_prior else None
            self.win_priors = {}
        # size_modes: {machine_type: (small_mix, big_mix)}, the two exact payout-size mixes
        #   (1x, 2x, 5x, 20x) a machine of that type switches between (size_modes_from_targets).
        #   Each size cell then holds one number, P(big mode), from its decayed size counts by
        #   Bayes' rule (log-likelihood ratio x mode_temp), instead of 4 free frequencies, so it
        #   learns from far fewer wins. Without modes the size experts use free frequencies.
        self.size_modes = {t: _normalize_size_modes(spec) for t, spec in (size_modes or {}).items()}
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
        mode_spec = self.size_modes.get(machine_type)
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
                has_cell_prior = self.win_priors.get(machine_type) or self.win_prior
                win_preds.append(self._cell_rate(w, n, machine_type) if has_cell_prior
                                 else (w + self.k_win * base) / (n + self.k_win))
            size_cells = [cell_of(t, key, how) for key, how in zip(self.size_keys, self.size_cell)]
            size_preds = []
            for i, cell in enumerate(size_cells):
                sc = size_tab[i].get(cell) if cell is not None else None
                if mode_spec is not None:
                    counts = sc if sc is not None else machine_size
                    size_preds.append(size_mode_mixture(counts, mode_spec["modes"], mode_spec["weights"],
                                                        self.mode_temp))
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

    def _cell_rate(self, w, n, machine_type=None):
        """Posterior mean win rate of a cell with (decayed) w wins in n spins under the Beta-mixture prior."""
        components = self.win_priors.get(machine_type) or self.win_prior
        logs, means = [], []
        for weight, a, b in components:
            logs.append(math.log(weight) + math.lgamma(a + w) + math.lgamma(b + n - w) - math.lgamma(a + b + n)
                        - math.lgamma(a) - math.lgamma(b) + math.lgamma(a + b))
            means.append((a + w) / (a + b + n))
        top = max(logs)
        post = [math.exp(v - top) for v in logs]
        return math.fsum(p * m for p, m in zip(post, means)) / math.fsum(post)

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


def discover_size_keys(queries, history, lags=range(1, 9), min_share=0.5, top=4, min_wins=12):
    """Lags whose exact payout class best predicts the next win's payout size, from train history.

    Per machine, splits winning spins by the payout class `lag` spins back and measures the
    log-likelihood gain over the machine's overall size mix, minus its chance level (half the
    degrees of freedom), so sparse cells don't look informative. Lags scoring at least
    min_share x the best are kept (main round: 1, 2, 4, clearly above the rest). Like
    choose_pairs, this lets a new round move the size rule to other lags.
    """
    indexed = index_history(queries, history)
    gain = Counter()
    machines = Counter()
    for plays in indexed.values():
        by_tick = {p.tick: p.payout for p in plays}
        for lag in lags:
            cells = {}
            for tick, payout in by_tick.items():
                if payout > 0 and tick - lag in by_tick:
                    row = cells.setdefault(CLASS_INDEX[by_tick[tick - lag]], [0] * (CLASSES - 1))
                    row[CLASS_INDEX[payout] - 1] += 1
            totals = [sum(row[k] for row in cells.values()) for k in range(CLASSES - 1)]
            n = sum(totals)
            if n < min_wins:
                continue
            g = math.fsum(x * math.log(x * n / (sum(row) * total)) for row in cells.values()
                          for x, total in zip(row, totals) if x)
            df = (len(cells) - 1) * (sum(1 for total in totals if total) - 1)
            gain[lag] += g - df / 2.0
            machines[lag] += 1
    scores = {lag: gain[lag] / machines[lag] for lag in machines}
    best = max(scores.values(), default=0.0)
    if best <= 0.0:
        return MODE_SIZE_KEYS
    kept = sorted(lag for lag, score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
                  if score >= min_share * best)
    return ((),) + tuple((lag,) for lag in kept)


def size_modes_from_targets(targets, queries):
    """Cluster conditional payout-size vectors, retaining any supported number of modes."""
    machine_type = {q.query_id: q.machine_type for q in queries}
    seen = {}
    for target in targets:
        p = target.probabilities
        win = 1.0 - p[0]
        if win < 0.05 or target.query_id not in machine_type:
            continue
        mix = tuple(x / win for x in p[1:])
        seen.setdefault(machine_type[target.query_id], []).append(mix)
    modes = {}
    for kind, vectors in seen.items():
        groups = []
        # Deterministic online clustering; tolerates small target rounding noise.
        for mix in sorted(vectors):
            nearest = min(groups, key=lambda g: max(abs(a-b) for a,b in zip(g[0], mix)), default=None)
            if nearest is not None and max(abs(a-b) for a,b in zip(nearest[0], mix)) < 0.025:
                n = nearest[1]
                nearest[0] = tuple((n*a+b)/(n+1) for a,b in zip(nearest[0], mix))
                nearest[1] += 1
            else:
                groups.append([mix, 1])
        groups.sort(key=lambda g: (-g[1], g[0]))
        if not groups:
            continue
        # Tiny isolated clusters are target noise; retain them only when they carry >= 5% mass.
        groups = [g for g in groups if g[1] / len(vectors) >= 0.05 or len(groups) == 1]
        groups.sort(key=lambda g: (-g[0][0], g[0]))
        # Equal prior per mode: with two modes this is exactly the old sigmoid(LLR) rule.
        modes[kind] = {"modes": [g[0] for g in groups], "weights": [1.0] * len(groups)}
    return modes


def _u_shaped_prior(rates, strength, low=0.2, high=0.8, min_side=20):
    """Three-component Beta mixture fitted to a set of win rates.

    "dead" (< low) and "live" (> high) components sit at their cluster medians with the given
    strength; the middle component is a flat Beta(1, 1). Each weight is its cluster's share of the
    rates (+1 each, so none is zero). None if either side has fewer than min_side rates.
    """
    dead = sorted(r for r in rates if r < low)
    live = sorted(r for r in rates if r > high)
    middle = [r for r in rates if low <= r <= high]
    if len(dead) < min_side or len(live) < min_side:
        return None
    counts = (len(dead) + 1, len(live) + 1, len(middle) + 1)
    total = sum(counts)
    return [(counts[0] / total, round(dead[len(dead) // 2], 4), strength),
            (counts[1] / total, round(live[len(live) // 2], 4), strength),
            (counts[2] / total, 0.5, 2.0)]


def win_prior_from_targets(targets, strength=6.0):
    """Beta-mixture prior for win cells, read from validation targets.csv.

    True win chances are U-shaped: most sit near a "dead" level or a "live" level. Uses the median
    true win chance below and above 0.5 as the two levels (main round: 0.044 and 0.898), each with
    weight 0.45 and the given strength, plus a weak middle component (0.1, mean 0.5, strength 2).
    Data-driven weights (0.46 / 0.32 / 0.22 from the shares below 0.2 / above 0.8) were tried:
    +0.03 validation but worse out of fold on train (dR +0.008 [+0.003, +0.013]), so not used.
    Returns None if either side has too few targets to estimate a level.
    """
    wins = sorted(1.0 - t.probabilities[0] for t in targets)
    dead = [w for w in wins if w < 0.5]
    live = [w for w in wins if w >= 0.5]
    if len(dead) < 20 or len(live) < 20:
        return None
    return [(0.45, round(dead[len(dead) // 2], 4), strength), (0.45, round(live[len(live) // 2], 4), strength),
            (0.1, 0.5, 2.0)]


def win_prior_from_train(queries, history, pairs=None, min_spins=10):
    """Win prior from train history alone: a data-weighted mixture (_u_shaped_prior) fitted to the
    empirical win rates of every (machine, pair, lag-outcome) cell with at least min_spins spins.

    NOT used by adapt(): out of fold on train it was worse than no prior (dR +0.030 at the
    cutoff), because whole-history cell rates mix regimes and land mostly in the middle.
    Kept for round two, where it can be re-tested with tune_pair_mixer.py '[{"win_prior": "train"}]'.
    """
    pairs = pairs or choose_pairs(queries, history)
    rates = []
    for plays in index_history(queries, history).values():
        won = {p.tick: p.payout > 0 for p in plays}
        for a, b in pairs:
            wins, spins = Counter(), Counter()
            for tick, value in won.items():
                if tick - a in won and tick - b in won:
                    cell = (won[tick - a], won[tick - b])
                    wins[cell] += value
                    spins[cell] += 1
            rates.extend(wins[cell] / n for cell, n in spins.items() if n >= min_spins)
    return _u_shaped_prior(rates, 6.0)


def win_prior_by_type_from_targets(targets, queries, strength=10.0, global_shrink=0.0):
    """Beta-mixture win prior per machine type: {type: components}, levels read per type.

    Types differ a lot (main round: jackpot is rarely live, only 10 of 265 targets above 0.8,
    while steady is live in 149 of 269), so one round-wide pair of levels misfits them. Each
    type gets win_prior_from_targets on its own targets, with levels optionally shrunk toward
    the round-wide ones (global_shrink 0 = pure type levels, which tested best). A type with
    too few targets on either side uses the round-wide prior. None if that is missing too.
    Out of fold on train vs the round-wide prior: late-window dR -0.011 [-0.015, -0.008];
    cross-fitted validation 55.47 -> 56.12.
    """
    global_prior = win_prior_from_targets(targets, strength)
    if global_prior is None or not 0.0 <= global_shrink <= 1.0:
        return None
    machine_type = {q.query_id: q.machine_type for q in queries}
    by_type = {}
    for target in targets:
        kind = machine_type.get(target.query_id)
        if kind is not None:
            by_type.setdefault(kind, []).append(target)
    priors = {}
    for kind in sorted(by_type):
        own = win_prior_from_targets(by_type[kind], strength)
        if own is None:
            priors[kind] = global_prior
            continue
        priors[kind] = [(w, round((1.0 - global_shrink) * m + global_shrink * g[1], 4), k)
                        for (w, m, k), g in zip(own, global_prior)]
    return priors


def episode_kfolds(queries, conditions, n_folds=8):
    """{episode_id: fold} with whole episodes per fold, spread evenly within each condition
    (episodes ordered by a hash of their id, so the split is fixed and label-blind)."""
    by_condition = {}
    for query in queries:
        by_condition.setdefault(conditions[query.query_id], set()).add(query.episode_id)
    folds = {}
    for condition in sorted(by_condition):
        ordered = sorted(by_condition[condition], key=lambda e: hashlib.sha256(e.encode()).hexdigest())
        for index, episode_id in enumerate(ordered):
            folds[episode_id] = index % n_folds
    return folds


def split_episodes(queries, history, episodes):
    """(queries, history) inside the given episodes, then (queries, history) outside them."""
    inside = tuple(q for q in queries if q.episode_id in episodes)
    outside = tuple(q for q in queries if q.episode_id not in episodes)
    return ((inside, tuple(p for p in history if p.episode_id in episodes)),
            (outside, tuple(p for p in history if p.episode_id not in episodes)))


def cutoff_brier(model, queries, history, labels):
    """Mean 5-class Brier vs labels.csv outcomes at each query's cutoff tick."""
    predictions = model.predict(queries, history)
    return math.fsum(math.fsum((p - (1.0 if PAYOUTS[k] == labels[q] else 0.0)) ** 2
                               for k, p in enumerate(predictions[q])) for q in labels) / len(labels)


def adapt(queries, history, labels=None, log=print, size_modes=None, win_prior=None):
    """Round-two helper: rediscover the lag pairs and pick the cell mode on this round's train data.

    size_modes: from size_modes_from_targets on this round's validation targets, if available.
    win_prior: from win_prior_by_type_from_targets (or win_prior_from_targets) on the same
      targets, if available; a {type: components} dict or one component list.
    """
    pairs = choose_pairs(queries, history)
    log(f"lag pairs found on train: {pairs}")
    size_keys = discover_size_keys(queries, history)
    log(f"payout-size keys discovered: {size_keys}")
    log(f"payout-size modes: {'target-derived mixture' if size_modes else 'none (free frequencies)'}")
    log(f"win-cell prior: {win_prior if win_prior else 'none (smooth toward base rate)'}")
    extra = {"win_prior": win_prior, "eta": WIN_PRIOR_ETA} if win_prior else {}
    candidates = [PairMixer(pairs=pairs, cell_mode=mode, size_modes=size_modes, size_keys=size_keys,
                            **extra).fit(queries, history)
                  for mode in ("parity", "full")]
    if labels is None:
        return candidates[0]
    scores = [cutoff_brier(model, queries, history, labels) for model in candidates]
    for model, score in zip(candidates, scores):
        log(f"cell_mode={model.cell_mode[0]}: train cutoff Brier {score:.5f}")
    return candidates[scores.index(min(scores))]
