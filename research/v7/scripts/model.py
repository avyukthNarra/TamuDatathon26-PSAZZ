"""Episode-level variants on top of the v6 pair mixer.

The per-machine walk is the v6 walk (pair_mixer.PairMixer.walk) with three
optional hooks, all of which default to off:

- shrink half-life log-probabilities toward the episode mean
- swap parity-cell counts at a tick (direction flip)
- clear parity-cell counts at a tick (reset)

No main-round lag, half-life, or changepoint location is hardcoded. Pairs and
half-lives come from the fitted mixer; the flip tick comes from the detector.
"""

import math
from collections import defaultdict

from slot_forecasting.pair_mixer import CLASS_INDEX, CLASSES, MIN_PROBABILITY, size_mode_mixture
from slot_forecasting.schema import index_history, probabilities, unique_queries


def _logsumexp(values):
    if not values:
        return -math.inf
    top = max(values)
    if top == -math.inf:
        return -math.inf
    return top + math.log(math.fsum(math.exp(v - top) for v in values))


def finalize_probs(raw, query_id):
    clipped = [max(p, MIN_PROBABILITY) for p in raw]
    total = math.fsum(clipped)
    return probabilities([p / total for p in clipped], query_id)


def dist_from_win(pw, size):
    pw = min(max(pw, 0.0), 1.0)
    return [1.0 - pw] + [pw * s for s in size]


class MachineState:
    """One machine's mixer state. forecast() then absorb() is one v6 tick."""

    def __init__(self, model, query, own):
        self.model = model
        self.query = query
        self.own = own
        self.stop = query.cutoff
        self.size_decay = 2 ** (-1.0 / model.size_half_life)
        self.sizes = CLASSES - 1
        self.size_tab = [dict() for _ in model.size_keys]
        self.base_decay = 2 ** (-1.0 / model.half_lives[0])
        self.rate = [0.0, 0.0]
        self.machine_size = [0.0] * self.sizes
        self.experts = []  # (pair, mode, decay, pair_index, hl_index)
        prior = []
        for pair_index, (pair, weight) in enumerate(zip(model.pairs, model.pair_prior)):
            for mode in model.cell_mode:
                for hl_index, half_life in enumerate(model.half_lives):
                    decay = 2 ** (-1.0 / half_life)
                    self.experts.append((pair, mode, decay, pair_index, hl_index))
                    prior.append(weight)
        self.n_hl = len(model.half_lives)
        self.pair_index = [item[3] for item in self.experts]
        self.hl_index = [item[4] for item in self.experts]
        self.win_tab = [dict() for _ in self.experts]
        self.logw = list(prior)
        self.prior = list(prior)
        self.size_logw = [0.0] * len(model.size_keys)
        self.type_size = model._size_prior(query.machine_type)
        self.mode_spec = model.size_modes.get(query.machine_type)
        self.a0, self.b0 = model.rate_prior
        self.win_preds = []
        self.win_keys = []
        self.size_preds = []
        self.size_cells = []
        self.pw = 0.5
        self.size = [0.25] * self.sizes
        self.probs = None

    def cell_of(self, t, lags, how="win"):
        values = []
        own = self.own
        for lag in lags:
            v = own[t - lag] if t - lag >= 0 else None
            if v is None:
                return None
            if how == "win":
                values.append(v > 0)
            elif how == "big":
                values.append(0 if v == 0 else 1 if v <= 2 else 2)
            else:
                values.append(v)
        return tuple(values)

    def swap_parity(self):
        """Exchange agree/differ counts so pre-flip data matches the new direction."""
        for tab in self.win_tab:
            agree = tab.pop((True,), None)
            differ = tab.pop((False,), None)
            if agree is not None:
                tab[(False,)] = agree
            if differ is not None:
                tab[(True,)] = differ
            if "sym" in tab:
                evidence, n = tab["sym"]
                tab["sym"][0] = n - evidence

    def reset_counts(self):
        for tab in self.win_tab:
            tab.clear()

    def forecast(self, t):
        model = self.model
        base = (self.rate[0] + self.a0) / (self.rate[1] + self.b0)
        msum = math.fsum(self.machine_size)
        msize = [(c + model.k_machine * p) / (msum + model.k_machine)
                 for c, p in zip(self.machine_size, self.type_size)]
        pair_cells = {pair: self.cell_of(t, pair) for pair in model.pairs}
        win_keys, win_preds = [], []
        has_cell_prior = model.win_priors.get(self.query.machine_type) or model.win_prior
        for i, (pair, mode, _decay, _pi, _hi) in enumerate(self.experts):
            cell = pair_cells[pair]
            if cell is None:
                win_keys.append(None)
                win_preds.append(base)
                continue
            agree = cell[0] == cell[1]
            if mode == "sym":
                evidence, n = self.win_tab[i].get("sym", (0.0, 0.0))
                q = (evidence + model.k_win * 0.5) / (n + model.k_win)
                win_keys.append(("sym", agree))
                win_preds.append(q if agree else 1.0 - q)
                continue
            key = (agree,) if mode == "parity" else cell
            w, n = self.win_tab[i].get(key, (0.0, 0.0))
            win_keys.append(key)
            if has_cell_prior:
                win_preds.append(model._cell_rate(w, n, self.query.machine_type))
            else:
                win_preds.append((w + model.k_win * base) / (n + model.k_win))
        size_cells = [self.cell_of(t, key, how)
                      for key, how in zip(model.size_keys, model.size_cell)]
        size_preds = []
        for i, cell in enumerate(size_cells):
            sc = self.size_tab[i].get(cell) if cell is not None else None
            if self.mode_spec is not None:
                counts = sc if sc is not None else self.machine_size
                size_preds.append(size_mode_mixture(
                    counts, self.mode_spec["modes"], self.mode_spec["weights"], model.mode_temp))
            elif sc is None:
                size_preds.append(msize)
            else:
                ssum = math.fsum(sc)
                size_preds.append([(c + model.k_size * m) / (ssum + model.k_size)
                                   for c, m in zip(sc, msize)])
        self.win_keys = win_keys
        self.win_preds = win_preds
        self.size_preds = size_preds
        self.size_cells = size_cells

    def mix(self):
        self.pw = mix_pw(self.logw, self.win_preds)
        top = max(self.size_logw)
        weights = [math.exp(v - top) for v in self.size_logw]
        total = math.fsum(weights)
        self.size = [math.fsum(w * sp[k] for w, sp in zip(weights, self.size_preds)) / total
                     for k in range(self.sizes)]
        self.probs = finalize_probs(dist_from_win(self.pw, self.size), self.query.query_id)

    def absorb(self, t):
        model = self.model
        value = self.own[t]
        if value is not None:
            won = 1.0 if value > 0 else 0.0
            eta = model.eta
            for i, p in enumerate(self.win_preds):
                self.logw[i] -= eta * (p - won) ** 2
            if model.alpha > 0:
                _fixed_share(self.logw, self.prior, model.alpha)
            if value > 0:
                for i, sp in enumerate(self.size_preds):
                    self.size_logw[i] -= model.size_eta * math.fsum(
                        (q - (1.0 if k == value - 1 else 0.0)) ** 2 for k, q in enumerate(sp))
        if value is None and not model.decay_missing:
            return
        for (_pair, _mode, decay, _pi, _hi), tab in zip(self.experts, self.win_tab):
            for cell in tab.values():
                cell[0] *= decay
                cell[1] *= decay
        for tab in self.size_tab:
            for cell in tab.values():
                for k in range(self.sizes):
                    cell[k] *= self.size_decay
        self.rate[0] *= self.base_decay
        self.rate[1] *= self.base_decay
        for k in range(self.sizes):
            self.machine_size[k] *= self.size_decay
        if value is None:
            return
        won = 1.0 if value > 0 else 0.0
        self.rate[0] += won
        self.rate[1] += 1.0
        for i, key in enumerate(self.win_keys):
            if key is None:
                continue
            if key[0] == "sym":
                cell = self.win_tab[i].setdefault("sym", [0.0, 0.0])
                cell[0] += 1.0 if (won == 1.0) == key[1] else 0.0
            else:
                cell = self.win_tab[i].setdefault(key, [0.0, 0.0])
                cell[0] += won
            cell[1] += 1.0
        if value > 0:
            self.machine_size[value - 1] += 1.0
            for i, cell_key in enumerate(self.size_cells):
                if cell_key is not None:
                    self.size_tab[i].setdefault(cell_key, [0.0] * self.sizes)[value - 1] += 1.0


def _fixed_share(logw, prior, alpha):
    top = max(logw)
    raw = [math.exp(v - top) for v in logw]
    total = math.fsum(raw)
    ptop = max(prior)
    praw = [math.exp(v - ptop) for v in prior]
    ptotal = math.fsum(praw)
    logw[:] = [math.log((1 - alpha) * r / total + alpha * q / ptotal) for r, q in zip(raw, praw)]


def mix_pw(logw, win_preds):
    top = max(logw)
    weights = [math.exp(v - top) for v in logw]
    total = math.fsum(weights)
    return math.fsum(w * p for w, p in zip(weights, win_preds)) / total


def half_life_logp(logw, n_hl):
    masses = [_logsumexp([logw[i] for i in range(len(logw)) if i % n_hl == h]) for h in range(n_hl)]
    z = _logsumexp(masses)
    return [m - z for m in masses]


def shrink_halflife(logws, n_hl, lam, how):
    """Pull each machine's half-life log-probabilities toward the episode mean.

    Pair preferences inside a half-life are untouched: every expert of half-life
    h is shifted by the same amount. `how` is 'uniform' or 'precision'
    (machines with a peaked half-life distribution pull harder, so a machine
    with no evidence does not drag the mean back to uniform).
    """
    if lam <= 0 or len(logws) < 2:
        return
    lps = [half_life_logp(logw, n_hl) for logw in logws]
    if how == "precision":
        weights = [max(lp) - min(lp) for lp in lps]
        wsum = math.fsum(weights)
        if wsum <= 1e-8:
            return
        mean = [math.fsum(w * lp[h] for w, lp in zip(weights, lps)) / wsum for h in range(n_hl)]
    elif how == "uniform":
        n = float(len(lps))
        mean = [math.fsum(lp[h] for lp in lps) / n for h in range(n_hl)]
    else:
        raise ValueError(how)
    for logw, lp in zip(logws, lps):
        mixed = [(1.0 - lam) * lp[h] + lam * mean[h] for h in range(n_hl)]
        mixed = [v - _logsumexp(mixed) for v in mixed]
        for h in range(n_hl):
            delta = mixed[h] - lp[h]
            if delta == 0.0:
                continue
            for i in range(len(logw)):
                if i % n_hl == h:
                    logw[i] += delta


def long_pw(state):
    """Win probability of the longest half-life on the machine's best pair."""
    n_pairs = max(state.pair_index) + 1
    masses = [_logsumexp([state.logw[i] for i in range(len(state.logw))
                           if state.pair_index[i] == pair_index])
              for pair_index in range(n_pairs)]
    best = max(range(n_pairs), key=lambda i: masses[i])
    longest = state.n_hl - 1
    chosen = [i for i in range(len(state.logw))
              if state.pair_index[i] == best and state.hl_index[i] == longest]
    return mix_pw([state.logw[i] for i in chosen], [state.win_preds[i] for i in chosen])


def snapshot(state):
    return {
        "query_id": state.query.query_id,
        "episode_id": state.query.episode_id,
        "machine_type": state.query.machine_type,
        "logw": list(state.logw),
        "win_preds": list(state.win_preds),
        "n_hl": state.n_hl,
        "pw": state.pw,
        "size": list(state.size),
        "probs": state.probs,
        "long_pw": long_pw(state),
    }


def episode_series(queries, history):
    """{episode_id: [(query, own class-index list with None at the cutoff), ...]}."""
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


def walk_episode(model, rows, swap_tick=None, reset_tick=None, pool=None):
    """Walk the machines of one episode in lockstep.

    pool = ('online'|'final', lam, 'uniform'|'precision') or None.
    Swap/reset happen at the start of that tick, before the forecast, so the
    cutoff forecast sees the corrected counts.
    """
    states = [MachineState(model, query, own) for query, own in rows]
    stop = rows[0][0].cutoff
    n_hl = states[0].n_hl
    for t in range(stop + 1):
        if swap_tick is not None and t == swap_tick:
            for state in states:
                state.swap_parity()
        if reset_tick is not None and t == reset_tick:
            for state in states:
                state.reset_counts()
        for state in states:
            state.forecast(t)
        if pool is not None and (pool[0] == "online" or (pool[0] == "final" and t == stop)):
            shrink_halflife([state.logw for state in states], n_hl, pool[1], pool[2])
        for state in states:
            state.mix()
        if t == stop:
            break
        for state in states:
            state.absorb(t)
    return states


def predict_split(model, by_episode, intervention):
    """intervention(episode_id, cutoff) -> (swap_tick, reset_tick, pool)."""
    predictions = {}
    snaps = {}
    for episode_id, rows in by_episode.items():
        cutoff = rows[0][0].cutoff
        swap_tick, reset_tick, pool = intervention(episode_id, cutoff)
        for state in walk_episode(model, rows, swap_tick=swap_tick, reset_tick=reset_tick, pool=pool):
            predictions[state.query.query_id] = state.probs
            snaps[state.query.query_id] = snapshot(state)
    return predictions, snaps


def apply_pool(grouped_snaps, lam, how):
    """Post-hoc half-life shrink at the cutoff, from captured independent log-weights."""
    predictions = {}
    for rows in grouped_snaps.values():
        logws = [list(row["logw"]) for row in rows]
        shrink_halflife(logws, rows[0]["n_hl"], lam, how)
        for row, logw in zip(rows, logws):
            pw = mix_pw(logw, row["win_preds"])
            predictions[row["query_id"]] = finalize_probs(dist_from_win(pw, row["size"]), row["query_id"])
    return predictions


def group_snaps(snaps, by_episode):
    return {episode_id: [snaps[query.query_id] for query, _own in rows]
            for episode_id, rows in by_episode.items()}


def long_predictions(snaps):
    """Cutoff distribution that keeps v6's size mix and the slow expert's win rate."""
    predictions = {}
    for row in snaps.values():
        predictions[row["query_id"]] = finalize_probs(
            dist_from_win(row["long_pw"], row["size"]), row["query_id"])
    return predictions


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
        p_before_agree, p_after_agree = (w_live + prior_a) / (n_live + strength), (w_dead + prior_a) / (n_dead + strength)
        p_before_diff, p_after_diff = (w_dead + prior_a) / (n_dead + strength), (w_live + prior_a) / (n_live + strength)
    else:
        w_live = left[2] + right[0]
        n_live = left[3] + right[1]
        w_dead = left[0] + right[2]
        n_dead = left[1] + right[3]
        p_before_agree, p_after_agree = (w_dead + prior_a) / (n_dead + strength), (w_live + prior_a) / (n_live + strength)
        p_before_diff, p_after_diff = (w_live + prior_a) / (n_live + strength), (w_dead + prior_a) / (n_dead + strength)
    return (bernoulli_ll(left[0], left[1], p_before_agree) + bernoulli_ll(left[2], left[3], p_before_diff)
            + bernoulli_ll(right[0], right[1], p_after_agree) + bernoulli_ll(right[2], right[3], p_after_diff))


def machine_flip_table(own, pair, cutoff, min_side, prior=(1.0, 1.0)):
    """(ll_stable, {tau: ll_flip}) for one pair. Each ll marginalizes direction.

    tau is the first tick of the new direction. Candidates need at least
    min_side observed parity rows on each side (a sample-size floor, not a
    location on the episode).
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


def detect_episode(rows, pairs, min_side=8, prior=(1.0, 1.0)):
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
        "machine_tau_sd_frac": 1.0,
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
    machine_taus = []
    for _score, ll_stable, by_tau in usable:
        personal = max(by_tau.items(), key=lambda item: item[1])
        if personal[1] - ll_stable > 2.0:
            machine_taus.append(personal[0])
    if len(machine_taus) >= 2:
        mmean = math.fsum(machine_taus) / len(machine_taus)
        msd = math.sqrt(math.fsum((tau - mmean) ** 2 for tau in machine_taus) / len(machine_taus))
        machine_sd_frac = msd / cutoff
    else:
        machine_sd_frac = 1.0
    return {
        "logbf": logbf,
        "map_tau": map_tau,
        "map_frac": map_tau / cutoff,
        "sd_frac": math.sqrt(max(var, 0.0)) / cutoff,
        "missing": missing,
        "n_machines": len(usable),
        "cutoff": cutoff,
        "machine_tau_sd_frac": machine_sd_frac,
    }


def detect_all(by_episode, pairs, min_side=8):
    return {episode_id: detect_episode(rows, pairs, min_side=min_side)
            for episode_id, rows in by_episode.items()}


def missing_threshold(train_detect):
    """Midpoint of the widest gap in train missing-fractions, if the gap is wide.

    Short histories are an observation mask, so on a round that uses one the
    missing fraction is bimodal. None if the largest gap is under 0.15, and the
    short-history filter then stays off.
    """
    values = sorted(row["missing"] for row in train_detect.values())
    if len(values) < 4:
        return None
    width, mid = max((values[i + 1] - values[i], 0.5 * (values[i] + values[i + 1]))
                     for i in range(len(values) - 1))
    if width < 0.15:
        return None
    return mid
