"""Lag-feature forecaster trained on every past spin, not only the official queries.

Each observed tick in history.csv becomes one training example: its payout is the
label and every input is built only from ticks strictly before it, on the same
machine and on the other machines of its episode. Prediction for a query at
``cutoff`` runs the exact same feature code at tick ``cutoff``, so training and
test inputs are computed identically.

Standard library only (the kit's base image is plain Python). The model is a
multinomial logistic regression fit with Adam on sparse rows.
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .schema import PAYOUTS, ValidationError, index_history, probabilities, unique_queries

MODEL_VERSION = "lag-logreg-v1"
TYPES = ("steady", "balanced", "jackpot")
SIGNALS = (-1, 0, 1)
CLASSES = len(PAYOUTS)
CLASS_INDEX = {value: index for index, value in enumerate(PAYOUTS)}
MAX_LAG = 8          # README: outcomes interact at delays up to eight ticks
NEIGHBOR_LAGS = 4
HALF_LIVES = (8.0, 32.0)
PRIOR_STRENGTH = 4.0
COND_HALF_LIFE = 64.0  # per-machine lag responses can slowly reverse (drift), so forget slowly
MIN_PROBABILITY = 1e-3


def _feature_names():
    names = ["bias"]
    names += [f"type={t}" for t in TYPES]
    names += [f"signal={s}" for s in SIGNALS]
    names += [f"type={t}|signal={s}" for t in TYPES for s in SIGNALS]
    for lag in range(1, MAX_LAG + 1):
        names += [f"lag{lag}={p}" for p in PAYOUTS] + [f"lag{lag}=missing"]
    for lag in range(1, MAX_LAG + 1):
        names += [f"type={t}|lag{lag}win" for t in TYPES]
    for half_life in HALF_LIVES:
        names += [f"logp_hl{half_life:g}_{p}" for p in PAYOUTS]
    names += [f"logp_all_{p}" for p in PAYOUTS]
    for lag in range(1, MAX_LAG + 1):
        names.append(f"self_cond_win_lag{lag}")
    for lag in range(1, MAX_LAG + 1):
        names += [f"cond_lag{lag}_logratio_{p}" for p in PAYOUTS]
    for lag in range(1, NEIGHBOR_LAGS + 1):
        names += [f"nb_lag{lag}_winrate", f"nb_lag{lag}_meanlog", f"nb_lag{lag}_missing"]
    names += ["nb_winrate_hl8", "nb_winrate_shift", "nb_cond_win_lag1", "since_win", "since_big", "log_observed",
              "recent_gap"]
    return tuple(names)


FEATURES = _feature_names()
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURES)}


def _series(plays, length):
    """Payout class index per tick, or None when the tick was not observed."""
    values = [None] * length
    for play in plays:
        if play.tick < length:
            values[play.tick] = CLASS_INDEX[play.payout]
    return values


class _MachineState:
    """Running statistics over ticks < t, advanced one tick at a time (no lookahead)."""

    def __init__(self, prior):
        self.prior = prior
        self.decayed = {h: [0.0] * CLASSES for h in HALF_LIVES}
        self.decay = {h: 2 ** (-1.0 / h) for h in HALF_LIVES}
        self.total = [0.0] * CLASSES
        self.observed = 0
        self.since_win = None
        self.since_big = None
        self.last_seen = None
        # cond[lag][state] = [wins, count] where state = 1 if the tick `lag` back was a win.
        self.cond = [[[0.0, 0.0], [0.0, 0.0]] for _ in range(MAX_LAG + 1)]
        self.nb_cond = [[0.0, 0.0], [0.0, 0.0]]
        # cond_dist[lag][state] = decayed payout-class counts after a win (1) / loss (0) `lag` back.
        self.cond_dist = [[[0.0] * CLASSES, [0.0] * CLASSES] for _ in range(MAX_LAG + 1)]
        self.cond_decay = 2 ** (-1.0 / COND_HALF_LIFE)

    def advance(self, own, neighbor_win, t):
        """Absorb tick t (its own value and neighbors' value) after predicting it."""
        value = own[t]
        for h in HALF_LIVES:
            factor = self.decay[h]
            counts = self.decayed[h]
            for index in range(CLASSES):
                counts[index] *= factor
            if value is not None:
                counts[value] += 1.0
        if self.since_win is not None:
            self.since_win += 1
        if self.since_big is not None:
            self.since_big += 1
        for lag in range(1, MAX_LAG + 1):
            for counts in self.cond_dist[lag]:
                for index in range(CLASSES):
                    counts[index] *= self.cond_decay
        if value is None:
            return
        self.total[value] += 1.0
        self.observed += 1
        self.last_seen = t
        win = 1.0 if value > 0 else 0.0
        if value > 0:
            self.since_win = 0
        if value >= 3:
            self.since_big = 0
        for lag in range(1, MAX_LAG + 1):
            if t - lag >= 0 and own[t - lag] is not None:
                state = 1 if own[t - lag] > 0 else 0
                cell = self.cond[lag][state]
                cell[0] += win
                cell[1] += 1.0
                self.cond_dist[lag][state][value] += 1.0
        if t - 1 >= 0 and neighbor_win[t - 1] is not None:
            state = 1 if neighbor_win[t - 1] >= 0.5 else 0
            self.nb_cond[state][0] += win
            self.nb_cond[state][1] += 1.0

    def win_rate(self):
        wins = sum(self.total[1:])
        prior_win = 1.0 - self.prior[0]
        return (wins + 2.0 * prior_win) / (self.observed + 2.0)


def _log_smoothed(counts, prior):
    total = math.fsum(counts) + PRIOR_STRENGTH
    return [math.log((c + PRIOR_STRENGTH * p) / total) for c, p in zip(counts, prior)]


def _features(query, own, state, neighbor_win, neighbor_mean, neighbor_seen, t):
    """Sparse (index, value) features for predicting tick t from ticks < t only."""
    row = {FEATURE_INDEX["bias"]: 1.0,
           FEATURE_INDEX[f"type={query.machine_type}"]: 1.0,
           FEATURE_INDEX[f"signal={query.signal}"]: 1.0,
           FEATURE_INDEX[f"type={query.machine_type}|signal={query.signal}"]: 1.0}
    for lag in range(1, MAX_LAG + 1):
        value = own[t - lag] if t - lag >= 0 else None
        if value is None:
            row[FEATURE_INDEX[f"lag{lag}=missing"]] = 1.0
        else:
            row[FEATURE_INDEX[f"lag{lag}={PAYOUTS[value]}"]] = 1.0
            if value > 0:
                row[FEATURE_INDEX[f"type={query.machine_type}|lag{lag}win"]] = 1.0
    for h in HALF_LIVES:
        for p, logp in zip(PAYOUTS, _log_smoothed(state.decayed[h], state.prior)):
            row[FEATURE_INDEX[f"logp_hl{h:g}_{p}"]] = logp
    for p, logp in zip(PAYOUTS, _log_smoothed(state.total, state.prior)):
        row[FEATURE_INDEX[f"logp_all_{p}"]] = logp
    base = state.win_rate()
    for lag in range(1, MAX_LAG + 1):
        value = own[t - lag] if t - lag >= 0 else None
        if value is None:
            continue
        wins, count = state.cond[lag][1 if value > 0 else 0]
        # Machine-specific: how this machine's win chance has differed after this state.
        row[FEATURE_INDEX[f"self_cond_win_lag{lag}"]] = (wins + 4.0 * base) / (count + 4.0) - base
    overall = _log_smoothed(state.decayed[HALF_LIVES[-1]], state.prior)
    for lag in range(1, MAX_LAG + 1):
        value = own[t - lag] if t - lag >= 0 else None
        if value is None:
            continue
        counts = state.cond_dist[lag][1 if value > 0 else 0]
        # This machine's payout mix after this lag state, relative to its usual mix.
        for p, logp, base_logp in zip(PAYOUTS, _log_smoothed(counts, [math.exp(x) for x in overall]), overall):
            row[FEATURE_INDEX[f"cond_lag{lag}_logratio_{p}"]] = logp - base_logp
    for lag in range(1, NEIGHBOR_LAGS + 1):
        s = t - lag
        if s < 0 or neighbor_win[s] is None:
            row[FEATURE_INDEX[f"nb_lag{lag}_missing"]] = 1.0
        else:
            row[FEATURE_INDEX[f"nb_lag{lag}_winrate"]] = neighbor_win[s]
            row[FEATURE_INDEX[f"nb_lag{lag}_meanlog"]] = neighbor_mean[s]
    nb_rate = neighbor_seen["decayed_win"] / neighbor_seen["decayed_n"] if neighbor_seen["decayed_n"] else 0.0
    row[FEATURE_INDEX["nb_winrate_hl8"]] = nb_rate
    if neighbor_seen["all_n"]:
        row[FEATURE_INDEX["nb_winrate_shift"]] = nb_rate - neighbor_seen["all_win"] / neighbor_seen["all_n"]
    if t - 1 >= 0 and neighbor_win[t - 1] is not None:
        wins, count = state.nb_cond[1 if neighbor_win[t - 1] >= 0.5 else 0]
        row[FEATURE_INDEX["nb_cond_win_lag1"]] = (wins + 4.0 * base) / (count + 4.0) - base
    row[FEATURE_INDEX["since_win"]] = min(state.since_win if state.since_win is not None else 32, 32) / 32.0
    row[FEATURE_INDEX["since_big"]] = min(state.since_big if state.since_big is not None else 64, 64) / 64.0
    row[FEATURE_INDEX["log_observed"]] = math.log1p(state.observed) / 6.0
    gap = (t - state.last_seen) if state.last_seen is not None else 64
    row[FEATURE_INDEX["recent_gap"]] = min(gap, 64) / 64.0
    return sorted(row.items())


def _neighbor_series(series_by_machine, machine_id, length):
    """Per tick: share of *other* observed machines that won, and their mean log(1+payout)."""
    win = [None] * length
    mean = [None] * length
    for t in range(length):
        wins = total = logs = 0.0
        for other, values in series_by_machine.items():
            if other == machine_id or t >= len(values) or values[t] is None:
                continue
            total += 1.0
            wins += 1.0 if values[t] > 0 else 0.0
            logs += math.log1p(PAYOUTS[values[t]])
        if total:
            win[t] = wins / total
            mean[t] = logs / total / math.log1p(20)
    return win, mean


def _walk_machine(query, own, neighbor_win, neighbor_mean, prior, stop, emit):
    """Walk ticks 0..stop; at each tick call emit(t, features) BEFORE absorbing tick t."""
    state = _MachineState(prior)
    neighbor_seen = {"decayed_win": 0.0, "decayed_n": 0.0, "all_win": 0.0, "all_n": 0.0}
    nb_decay = 2 ** (-1.0 / 8.0)
    for t in range(stop + 1):
        emit(t, _features(query, own, state, neighbor_win, neighbor_mean, neighbor_seen, t))
        if t == stop:
            break
        state.advance(own, neighbor_win, t)
        neighbor_seen["decayed_win"] *= nb_decay
        neighbor_seen["decayed_n"] *= nb_decay
        if neighbor_win[t] is not None:
            neighbor_seen["decayed_win"] += neighbor_win[t]
            neighbor_seen["decayed_n"] += 1.0
            neighbor_seen["all_win"] += neighbor_win[t]
            neighbor_seen["all_n"] += 1.0


def _episodes(queries, history):
    query_map = unique_queries(queries)
    indexed = index_history(queries, history)
    episodes = defaultdict(list)
    for query in query_map.values():
        episodes[query.episode_id].append(query)
    for episode_id in sorted(episodes):
        members = sorted(episodes[episode_id], key=lambda q: q.machine_id)
        length = max(q.cutoff for q in members)
        series = {q.machine_id: _series(indexed.get((q.episode_id, q.machine_id), []), length)
                  for q in members}
        yield members, series, length


def type_priors(queries, history):
    counts = {t: [0.5] * CLASSES for t in TYPES}
    machine_type = {(q.episode_id, q.machine_id): q.machine_type for q in queries}
    for play in history:
        counts[machine_type[(play.episode_id, play.machine_id)]][CLASS_INDEX[play.payout]] += 1.0
    return {t: [c / math.fsum(v) for c in v] for t, v in counts.items()}


def build_examples(queries, history, priors, labels=None, min_tick=MAX_LAG):
    """One example per observed tick (>= min_tick), plus each query's labelled cutoff.

    Inputs for tick t only ever read ticks < t (enforced by _walk_machine order).
    Returns (rows, targets, episode_ids) where targets are class indices.
    """
    rows, targets, groups = [], [], []
    for members, series, _ in _episodes(queries, history):
        for query in members:
            own = list(series[query.machine_id])
            label = labels.get(query.query_id) if labels else None
            if label is not None:
                own = own + [None] * (query.cutoff + 1 - len(own))
                own[query.cutoff] = CLASS_INDEX[label]
            neighbor_win, neighbor_mean = _neighbor_series(series, query.machine_id, len(own))

            def emit(t, features, own=own):
                if t >= min_tick and own[t] is not None:
                    rows.append(features)
                    targets.append(own[t])
                    groups.append(query.episode_id)

            stop = query.cutoff if label is not None else query.cutoff - 1
            _walk_machine(query, own, neighbor_win, neighbor_mean, priors[query.machine_type], stop, emit)
    return rows, targets, groups


def query_features(queries, history, priors):
    """Features for each query's cutoff tick, using only ticks < cutoff."""
    result = {}
    for members, series, length in _episodes(queries, history):
        for query in members:
            own = series[query.machine_id] + [None] * max(0, query.cutoff + 1 - length)
            neighbor_win, neighbor_mean = _neighbor_series(series, query.machine_id, len(own))
            captured = {}
            _walk_machine(query, own, neighbor_win, neighbor_mean, priors[query.machine_type],
                          query.cutoff, lambda t, f: captured.__setitem__(t, f))
            result[query.query_id] = captured[query.cutoff]
    return result


def _softmax(scores):
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    total = math.fsum(exps)
    return [e / total for e in exps]


class LagForecaster:
    def __init__(self, epochs=6, learning_rate=0.01, l2=1e-5, batch_size=256, seed=20261107,
                 blend=0.8):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.batch_size = batch_size
        self.seed = seed
        self.blend = blend
        self.weights = None
        self.priors = None

    def fit(self, queries, history, labels=None, log=None):
        self.priors = type_priors(queries, history)
        rows, targets, _ = build_examples(queries, history, self.priors, labels)
        if not rows:
            raise ValidationError("No training examples could be built.")
        self.fit_rows(rows, targets, log=log)
        return self

    def fit_rows(self, rows, targets, log=None):
        width = len(FEATURES)
        weights = [[0.0] * CLASSES for _ in range(width)]
        m = [[0.0] * CLASSES for _ in range(width)]
        v = [[0.0] * CLASSES for _ in range(width)]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        order = list(range(len(rows)))
        rng = random.Random(self.seed)
        step = 0
        for epoch in range(self.epochs):
            rng.shuffle(order)
            loss = 0.0
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                grad = defaultdict(lambda: [0.0] * CLASSES)
                for i in batch:
                    row = rows[i]
                    scores = [0.0] * CLASSES
                    for index, value in row:
                        w = weights[index]
                        for k in range(CLASSES):
                            scores[k] += w[k] * value
                    probs = _softmax(scores)
                    loss -= math.log(max(probs[targets[i]], 1e-12))
                    probs[targets[i]] -= 1.0
                    for index, value in row:
                        g = grad[index]
                        for k in range(CLASSES):
                            g[k] += probs[k] * value
                step += 1
                lr = self.learning_rate * math.sqrt(1 - beta2 ** step) / (1 - beta1 ** step)
                scale = 1.0 / len(batch)
                for index, g in grad.items():
                    w, mi, vi = weights[index], m[index], v[index]
                    for k in range(CLASSES):
                        gk = g[k] * scale + self.l2 * w[k]
                        mi[k] = beta1 * mi[k] + (1 - beta1) * gk
                        vi[k] = beta2 * vi[k] + (1 - beta2) * gk * gk
                        w[k] -= lr * mi[k] / (math.sqrt(vi[k]) + eps)
            if log:
                log(f"epoch {epoch + 1}/{self.epochs}: train log-loss {loss / len(rows):.4f}")
        self.weights = weights
        return self

    def predict_row(self, row):
        scores = [0.0] * CLASSES
        for index, value in row:
            w = self.weights[index]
            for k in range(CLASSES):
                scores[k] += w[k] * value
        probs = _softmax(scores)
        probs = [max(p, MIN_PROBABILITY) for p in probs]
        total = math.fsum(probs)
        return [p / total for p in probs]

    def predict(self, queries, history, baseline=None):
        if self.weights is None:
            raise ValidationError("Fit or load the model before predicting.")
        features = query_features(queries, history, self.priors)
        predictions = {}
        for query_id, row in features.items():
            probs = self.predict_row(row)
            if baseline is not None and self.blend < 1.0:
                probs = [self.blend * p + (1 - self.blend) * b for p, b in zip(probs, baseline[query_id])]
            predictions[query_id] = probabilities(probs, query_id)
        return predictions

    def save(self, path):
        if self.weights is None:
            raise ValidationError("Fit the model before saving.")
        Path(path).write_text(json.dumps({
            "model_version": MODEL_VERSION, "features": list(FEATURES), "weights": self.weights,
            "priors": self.priors, "settings": {"epochs": self.epochs, "learning_rate": self.learning_rate,
                                                "l2": self.l2, "batch_size": self.batch_size,
                                                "seed": self.seed, "blend": self.blend}},
            allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("model_version") != MODEL_VERSION or tuple(value.get("features", ())) != FEATURES:
            raise ValidationError("Saved lag model does not match this code's feature list.")
        model = cls(**value["settings"])
        model.weights = value["weights"]
        model.priors = value["priors"]
        return model
