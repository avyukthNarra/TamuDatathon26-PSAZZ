"""PairMixer with a causal, episode-pooled memory-speed loss adjustment."""

import math
from collections import defaultdict

from .pair_mixer import CLASS_INDEX, CLASSES, MIN_PROBABILITY, PairMixer
from .schema import index_history, probabilities, unique_queries


class EpisodeMemoryMixer(PairMixer):
    """Pool cumulative expert losses by memory speed within each episode.

    Each machine's experts still learn from its own history. At a prediction
    cutoff, the episode's summed prequential expert loss for each half-life is
    applied once as a shared log-weight shift. No peer outcome at or after the
    cutoff can affect that prediction.
    """

    def __init__(self, *args, gamma=0.5, **kwargs):
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and non-negative")
        super().__init__(*args, **kwargs)
        self.gamma = float(gamma)
        self._expert_half_lives = tuple(
            half_life
            for _ in self.pairs
            for _ in self.cell_mode
            for half_life in self.half_lives
        )

    def _trace(self, own, machine_type, stop, capture_ticks):
        """Run one machine independently, retaining causal cutoff snapshots."""
        size_decay = 2 ** (-1.0 / self.size_half_life)
        sizes = CLASSES - 1
        size_tab = [dict() for _ in self.size_keys]
        base_decay = 2 ** (-1.0 / self.half_lives[0])
        rate = [0.0, 0.0]
        machine_size = [0.0] * sizes
        experts, prior = [], []
        for pair, weight in zip(self.pairs, self.pair_prior):
            for mode in self.cell_mode:
                for half_life in self.half_lives:
                    experts.append((pair, mode, half_life, 2 ** (-1.0 / half_life)))
                    prior.append(weight)
        win_tab = [dict() for _ in experts]
        logw = list(prior)
        size_logw = [0.0] * len(self.size_keys)
        half_life_index = {half_life: i for i, half_life in enumerate(self.half_lives)}
        cumulative_loss = [0.0] * len(self.half_lives)
        snapshots = {}
        type_size = self._size_prior(machine_type)
        a0, b0 = self.rate_prior

        def cell_of(t, lags, how="win"):
            values = []
            for lag in lags:
                value = own[t - lag] if t - lag >= 0 else None
                if value is None:
                    return None
                if how == "win":
                    values.append(value > 0)
                elif how == "big":
                    values.append(0 if value == 0 else 1 if value <= 2 else 2)
                else:
                    values.append(value)
            return tuple(values)

        for tick in range(stop + 1):
            base = (rate[0] + a0) / (rate[1] + b0)
            msum = math.fsum(machine_size)
            msize = [(c + self.k_machine * p) / (msum + self.k_machine)
                     for c, p in zip(machine_size, type_size)]
            pair_cells = {pair: cell_of(tick, pair) for pair in self.pairs}
            win_keys, win_preds = [], []
            for i, (pair, mode, _, _) in enumerate(experts):
                cell = pair_cells[pair]
                if cell is None:
                    win_keys.append(None)
                    win_preds.append(base)
                    continue
                agree = cell[0] == cell[1]
                if mode == "sym":
                    evidence, count = win_tab[i].get("sym", (0.0, 0.0))
                    q = (evidence + self.k_win * 0.5) / (count + self.k_win)
                    win_keys.append(("sym", agree))
                    win_preds.append(q if agree else 1.0 - q)
                    continue
                key = (agree,) if mode == "parity" else cell
                wins, count = win_tab[i].get(key, (0.0, 0.0))
                win_keys.append(key)
                win_preds.append((wins + self.k_win * base) / (count + self.k_win))

            size_cells = [cell_of(tick, key, how)
                          for key, how in zip(self.size_keys, self.size_cell)]
            size_preds = []
            for i, cell in enumerate(size_cells):
                counts = size_tab[i].get(cell) if cell is not None else None
                if counts is None:
                    size_preds.append(msize)
                else:
                    total = math.fsum(counts)
                    size_preds.append([(c + self.k_size * p) / (total + self.k_size)
                                       for c, p in zip(counts, msize)])

            if tick in capture_ticks:
                top = max(size_logw)
                size_weights = [math.exp(value - top) for value in size_logw]
                size_total = math.fsum(size_weights)
                size = [math.fsum(w * prediction[k]
                                  for w, prediction in zip(size_weights, size_preds)) / size_total
                        for k in range(sizes)]
                snapshots[tick] = {
                    "logw": tuple(logw),
                    "win_preds": tuple(win_preds),
                    "size": tuple(size),
                    "cumulative_loss": tuple(cumulative_loss),
                }

            if tick == stop:
                break
            value = own[tick]
            if value is not None:
                won = 1.0 if value > 0 else 0.0
                for i, prediction in enumerate(win_preds):
                    loss = (prediction - won) ** 2
                    logw[i] -= self.eta * loss
                    cumulative_loss[half_life_index[experts[i][2]]] += loss
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
                    for i, predictions in enumerate(size_preds):
                        size_logw[i] -= self.size_eta * math.fsum(
                            (p - (1.0 if k == value - 1 else 0.0)) ** 2
                            for k, p in enumerate(predictions))
            if value is None and not self.decay_missing:
                continue
            for (_, _, _, decay), table in zip(experts, win_tab):
                for counts in table.values():
                    counts[0] *= decay
                    counts[1] *= decay
            for table in size_tab:
                for counts in table.values():
                    for k in range(sizes):
                        counts[k] *= size_decay
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
                    counts = win_tab[i].setdefault("sym", [0.0, 0.0])
                    counts[0] += 1.0 if (won == 1.0) == key[1] else 0.0
                else:
                    counts = win_tab[i].setdefault(key, [0.0, 0.0])
                    counts[0] += won
                counts[1] += 1.0
            if value > 0:
                machine_size[value - 1] += 1.0
                for i, cell in enumerate(size_cells):
                    if cell is not None:
                        size_tab[i].setdefault(cell, [0.0] * sizes)[value - 1] += 1.0
        return snapshots

    def predict(self, queries, history, baseline=None):
        query_map = unique_queries(queries)
        indexed = index_history(queries, history)
        by_episode = defaultdict(list)
        for query in query_map.values():
            by_episode[query.episode_id].append(query)

        result = {}
        for episode_id, episode_queries in by_episode.items():
            machine_queries = defaultdict(list)
            machine_types = {}
            for query in episode_queries:
                machine_queries[query.machine_id].append(query)
                machine_types[query.machine_id] = query.machine_type
            cutoffs = {query.cutoff for query in episode_queries}
            max_cutoff = max(cutoffs)
            traces = {}
            for machine_id, machine_cutoffs in machine_queries.items():
                own = [None] * (max_cutoff + 1)
                for play in indexed.get((episode_id, machine_id), ()):
                    if play.tick <= max_cutoff:
                        own[play.tick] = CLASS_INDEX[play.payout]
                traces[machine_id] = self._trace(
                    own, machine_types[machine_id], max_cutoff,
                    cutoffs | set(q.cutoff for q in machine_cutoffs),
                )

            for query in episode_queries:
                pooled_loss = [0.0] * len(self.half_lives)
                for machine_id in sorted(traces):
                    machine_trace = traces[machine_id]
                    losses = machine_trace[query.cutoff]["cumulative_loss"]
                    for index, loss in enumerate(losses):
                        pooled_loss[index] += loss
                snapshot = traces[query.machine_id][query.cutoff]
                adjusted = [value - self.eta * self.gamma * pooled_loss[
                    half_life_index
                ] for value, half_life_index in zip(
                    snapshot["logw"],
                    (self.half_lives.index(h) for h in self._expert_half_lives),
                )]
                top = max(adjusted)
                weights = [math.exp(value - top) for value in adjusted]
                win_probability = math.fsum(w * p for w, p in zip(weights, snapshot["win_preds"])) / math.fsum(weights)
                raw = [max(1.0 - win_probability, MIN_PROBABILITY)]
                raw.extend(max(win_probability * p, MIN_PROBABILITY) for p in snapshot["size"])
                total = math.fsum(raw)
                result[query.query_id] = probabilities([p / total for p in raw], query.query_id)
        return result
