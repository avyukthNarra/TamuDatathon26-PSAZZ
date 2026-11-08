"""Official score, uncertainty and diagnostic slices for one entry's forecasts.

Everything here is a pure function of CSV paths or in-memory dicts. Nothing
knows about a model: a prediction is just ``{query_id: (p_0, p_1, p_2, p_5,
p_20)}``. The only kit code touched is the pooled starter, and only inside
``reference_from_starter``.

Scoring follows RULES.md "How points work":

* per query: squared probability error summed over the five payout classes;
* per episode: mean over that episode's queries;
* ``E[c]``: mean over a condition's episodes;
* ``B[c]``: the same for the frozen reference;
* ``R``: mean over the four conditions of ``E[c] / max(B[c], 1e-9)``;
* ``points = max_points / (1 + R)`` with 60 for main and 40 for adaptation.
"""

from collections import defaultdict
import csv
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Mapping, Optional

PAYOUTS = (0, 1, 2, 5, 20)
PROB_COLUMNS = tuple(f"p_{value}" for value in PAYOUTS)
PREDICTION_COLUMNS = ("query_id", *PROB_COLUMNS)
CONDITIONS = ("stable", "short_history", "drift", "shared_change")
MAIN_MAX_POINTS = 60
ADAPTATION_MAX_POINTS = 40
SUM_TOLERANCE = 1e-6
REFERENCE_FLOOR = 1e-9
DEFAULT_SEED = 314159
DEFAULT_REPEATS = 1000
KIT_ROOT = Path(__file__).resolve().parent.parent
SHORT_HISTORY_TICKS = 64
SHORT_HISTORY_CONSECUTIVE = 40
CUTOFF_BINS = ((192, 207), (208, 223), (224, 239), (240, 256))


class ScoringError(ValueError):
    """Malformed input, or inputs that do not cover the same queries."""


# ---------------------------------------------------------------- readers

def _read_rows(path, columns, *, exact=False):
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames or []
            if len(set(header)) != len(header):
                raise ScoringError(f"{path}: duplicate column names")
            wanted = set(columns)
            if (set(header) != wanted) if exact else not wanted <= set(header):
                raise ScoringError(f"{path}: expected columns {', '.join(columns)}")
            rows = []
            for row in reader:
                if None in row or any(value is None or not value.strip() for value in row.values()):
                    raise ScoringError(f"{path}, line {reader.line_num}: missing, extra or empty field")
                rows.append(row)
    except (csv.Error, UnicodeError) as exc:
        raise ScoringError(f"{path}: unreadable CSV ({exc})") from exc
    if not rows:
        raise ScoringError(f"{path}: no data rows")
    return rows


def _probabilities(values, name):
    """Five finite probabilities in [0, 1] summing to 1 within tolerance.

    Mirrors the kit: after the tolerance check, values are divided by their sum
    to remove decimal rounding. Invalid rows are rejected, never repaired.
    """
    try:
        numbers = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ScoringError(f"{name}: probabilities must be numbers") from exc
    if len(numbers) != len(PAYOUTS):
        raise ScoringError(f"{name}: need exactly {len(PAYOUTS)} probabilities")
    if any(not math.isfinite(v) or v < 0 or v > 1 for v in numbers):
        raise ScoringError(f"{name}: probabilities must be finite and within [0, 1]")
    total = math.fsum(numbers)
    if abs(total - 1.0) > SUM_TOLERANCE:
        raise ScoringError(f"{name}: probabilities must sum to 1 (got {total:.12g})")
    return tuple(v / total for v in numbers)


def _require_same_ids(expected, actual, name):
    missing, extra = sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
    if missing or extra:
        raise ScoringError(f"{name}: query_id mismatch (missing {missing[:5]}, unexpected {extra[:5]})")


def read_predictions(source):
    """``{query_id: probs}`` from a submission CSV path, or a dict passed through."""
    if isinstance(source, Mapping):
        return {str(key): _probabilities(value, str(key)) for key, value in source.items()}
    result = {}
    for row in _read_rows(source, PREDICTION_COLUMNS, exact=True):
        key = row["query_id"]
        if key in result:
            raise ScoringError(f"{source}: duplicate query_id {key}")
        result[key] = _probabilities([row[c] for c in PROB_COLUMNS], key)
    return result


def read_queries(path):
    """``{query_id: {episode_id, machine_id, machine_type, signal, cutoff}}``."""
    result = {}
    for row in _read_rows(path, ("query_id", "episode_id", "machine_id", "machine_type", "signal", "cutoff")):
        key = row["query_id"]
        if key in result:
            raise ScoringError(f"{path}: duplicate query_id {key}")
        result[key] = {
            "episode_id": row["episode_id"], "machine_id": row["machine_id"],
            "machine_type": row["machine_type"], "signal": int(row["signal"]), "cutoff": int(row["cutoff"]),
        }
    return result


def read_conditions(path):
    result = {}
    for row in _read_rows(path, ("query_id", "condition")):
        if row["query_id"] in result:
            raise ScoringError(f"{path}: duplicate query_id {row['query_id']}")
        result[row["query_id"]] = row["condition"]
    return result


def _read_targets_full(path):
    """``(probs, outcomes, episodes)`` dicts keyed by query_id."""
    probs, outcomes, episodes = {}, {}, {}
    for row in _read_rows(path, ("query_id", "episode_id", *PROB_COLUMNS, "outcome")):
        key = row["query_id"]
        if key in probs:
            raise ScoringError(f"{path}: duplicate query_id {key}")
        probs[key] = _probabilities([row[c] for c in PROB_COLUMNS], key)
        outcomes[key] = int(float(row["outcome"]))
        episodes[key] = row["episode_id"]
    return probs, outcomes, episodes


def read_history_ticks(path):
    """``{(episode_id, machine_id): sorted observed ticks}`` from history.csv."""
    ticks = defaultdict(list)
    for row in _read_rows(path, ("episode_id", "machine_id", "tick", "payout")):
        ticks[(row["episode_id"], row["machine_id"])].append(int(row["tick"]))
    return {key: sorted(values) for key, values in ticks.items()}


def _coerce(value, loader):
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    return loader(value)


def _coerce_targets(value):
    if isinstance(value, Mapping):
        return {str(k): _probabilities(v, str(k)) for k, v in value.items()}
    return _read_targets_full(value)[0]


# ---------------------------------------------------------------- split loading

class Split:
    """One dataset split: queries, true probabilities, outcomes, conditions, reference.

    Any of targets/outcomes/conditions/reference is None when its file is
    absent (the real kit's test split has none of them).
    """

    def __init__(self, queries, targets, outcomes, conditions, reference):
        self.queries = queries
        self.targets = targets
        self.outcomes = outcomes
        self.conditions = conditions
        self.reference = reference

    def __repr__(self):
        return f"Split(queries={len(self.queries)}, targets={self.targets is not None})"


def _truth_directory(split_dir, truth_dir):
    """Resolve the `_truth` override: either `<root>/_truth` or `<root>/_truth/<split>`."""
    if truth_dir is None:
        return Path(split_dir)
    truth_dir, split_name = Path(truth_dir), Path(split_dir).name
    nested = truth_dir / split_name
    return nested if nested.is_dir() else truth_dir


def load_split(split_dir, truth_dir=None):
    """Load a split. ``truth_dir`` (DESIGN.md `_truth` override) replaces the answer files.

    Without ``truth_dir`` the answers come from ``split_dir`` itself. With it,
    answers come only from the truth directory, so a split folder that happens
    to hold answers is never silently used.
    """
    split_dir = Path(split_dir)
    queries = read_queries(split_dir / "queries.csv")
    answers = _truth_directory(split_dir, truth_dir)
    targets_path = answers / "targets.csv"
    targets = outcomes = episodes = None
    if targets_path.is_file():
        targets, outcomes, episodes = _read_targets_full(targets_path)
        _require_same_ids(queries, targets, str(targets_path))
        for key, episode in episodes.items():
            if queries[key]["episode_id"] != episode:
                raise ScoringError(f"{targets_path}: episode of {key} disagrees with queries.csv")
    conditions_path = answers / "conditions.csv"
    conditions = read_conditions(conditions_path) if conditions_path.is_file() else None
    reference_path = answers / "reference.csv"
    reference = read_predictions(reference_path) if reference_path.is_file() else None
    return Split(queries, targets, outcomes, conditions, reference)


# ---------------------------------------------------------------- core metric

def _squared_error(prediction, truth):
    return math.fsum((a - b) ** 2 for a, b in zip(prediction, truth))


def _mean(values):
    values = list(values)
    return math.fsum(values) / len(values)


def _align(pred, targets, queries, conditions, reference):
    """Validate that every input covers exactly the scored query set."""
    if not targets:
        raise ScoringError("no targets to score")
    expected = set(targets)
    for name, table in (("queries", queries), ("conditions", conditions),
                        ("prediction", pred), ("reference", reference)):
        if table is None:
            raise ScoringError(f"{name} is required")
        _require_same_ids(expected, table, name)
    return expected


def _episode_records(pred, targets, queries, conditions, reference):
    """``{episode_id: (condition, mean pred error, mean reference error)}``."""
    by_episode = defaultdict(list)
    for key in sorted(targets):
        by_episode[queries[key]["episode_id"]].append(key)
    records = {}
    for episode, ids in sorted(by_episode.items()):
        labels = {conditions[key] for key in ids}
        if len(labels) != 1:
            raise ScoringError(f"episode {episode} spans several conditions")
        pred_error = _mean(_squared_error(pred[k], targets[k]) for k in ids)
        ref_error = _mean(_squared_error(reference[k], targets[k]) for k in ids)
        records[episode] = (labels.pop(), pred_error, ref_error)
    return records


def _score_from_records(records, max_points):
    """Official points from ``(condition, pred_error, ref_error)`` episode triples."""
    per_condition = defaultdict(lambda: ([], []))
    for condition, pred_error, ref_error in records:
        per_condition[condition][0].append(pred_error)
        per_condition[condition][1].append(ref_error)
    if set(per_condition) != set(CONDITIONS):
        raise ScoringError(f"conditions must be exactly {CONDITIONS}, got {sorted(per_condition)}")
    E, B, ratio, episodes = {}, {}, {}, {}
    for condition in CONDITIONS:
        pred_errors, ref_errors = per_condition[condition]
        E[condition] = _mean(pred_errors)
        B[condition] = _mean(ref_errors)
        ratio[condition] = E[condition] / max(B[condition], REFERENCE_FLOOR)
        episodes[condition] = len(pred_errors)
    R = _mean(ratio[c] for c in CONDITIONS)
    return {"points": max_points / (1 + R), "R": R, "E": E, "B": B, "ratio": ratio,
            "n_episodes": episodes, "max_points": max_points}


def _inputs(pred, targets, queries, conditions, reference):
    """Resolve path-or-dict arguments and return the aligned episode records."""
    pred = read_predictions(pred)
    targets = _coerce_targets(targets)
    queries = _coerce(queries, read_queries)
    conditions = _coerce(conditions, read_conditions)
    reference = read_predictions(reference)
    _align(pred, targets, queries, conditions, reference)
    return _episode_records(pred, targets, queries, conditions, reference), pred, targets, queries, reference


def official_score(pred_csv, targets, queries, conditions, reference, max_points):
    """Official points and the per-condition pieces behind them.

    Arguments accept file paths (targets.csv, queries.csv, conditions.csv,
    reference.csv, the submission CSV) or dicts of the same shape. Returns
    ``{points, R, E, B, ratio, n_episodes, max_points}`` with per-condition dicts.
    """
    records, *_ = _inputs(pred_csv, targets, queries, conditions, reference)
    return _score_from_records(records.values(), max_points)


# ---------------------------------------------------------------- uncertainty

def _quantile(sorted_values, level):
    position = (len(sorted_values) - 1) * level
    low, high = math.floor(position), math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def _describe(values):
    values = list(values)
    return {"mean": _mean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}


def _strata(records):
    """``{condition: [(pred_error, ref_error), ...]}`` in sorted episode order."""
    strata = defaultdict(list)
    for episode in sorted(records):
        condition, pred_error, ref_error = records[episode]
        strata[condition].append((pred_error, ref_error))
    return strata


def _resample_draws(strata, repeats, seed):
    """Per repeat, per condition, episode indices drawn with replacement."""
    rng = random.Random(seed)
    draws = []
    for _ in range(repeats):
        draws.append({c: [rng.randrange(len(rows)) for _ in range(len(rows))]
                      for c, rows in sorted(strata.items())})
    return draws


def _points_for_draw(strata, draw, max_points):
    triples = []
    for condition, indices in draw.items():
        rows = strata[condition]
        triples.extend((condition, rows[i][0], rows[i][1]) for i in indices)
    return _score_from_records(triples, max_points)["points"]


def bootstrap_ci(pred, targets, queries, conditions, reference, max_points,
                 *, repeats=DEFAULT_REPEATS, seed=DEFAULT_SEED, level=0.95):
    """Condition-stratified episode bootstrap of the points.

    Episodes are resampled with replacement inside each condition. B is
    recomputed on every resample from the resampled reference episodes.
    """
    records, *_ = _inputs(pred, targets, queries, conditions, reference)
    strata = _strata(records)
    point = _score_from_records(records.values(), max_points)["points"]
    draws = _resample_draws(strata, repeats, seed)
    samples = sorted(_points_for_draw(strata, draw, max_points) for draw in draws)
    tail = (1 - level) / 2
    return {"points": point, "ci_low": _quantile(samples, tail), "ci_high": _quantile(samples, 1 - tail),
            "level": level, "repeats": repeats, "seed": seed,
            "bootstrap": _describe(samples)}


def paired_compare(pred_a, pred_b, targets, queries, conditions, reference, max_points,
                   *, repeats=DEFAULT_REPEATS, seed=DEFAULT_SEED, level=0.95):
    """Points of A minus points of B, with a paired bootstrap CI.

    Both predictions are resampled with the same episode draws, so the CI is for
    the difference rather than for each score separately.
    """
    records_a, *_ = _inputs(pred_a, targets, queries, conditions, reference)
    records_b, *_ = _inputs(pred_b, targets, queries, conditions, reference)
    if set(records_a) != set(records_b):
        raise ScoringError("paired comparison needs identical episodes")
    strata_a, strata_b = _strata(records_a), _strata(records_b)
    if {c: len(v) for c, v in strata_a.items()} != {c: len(v) for c, v in strata_b.items()}:
        raise ScoringError("paired comparison needs identical episode counts per condition")
    point_a = _score_from_records(records_a.values(), max_points)["points"]
    point_b = _score_from_records(records_b.values(), max_points)["points"]
    draws = _resample_draws(strata_a, repeats, seed)
    diffs = sorted(_points_for_draw(strata_a, draw, max_points) - _points_for_draw(strata_b, draw, max_points)
                   for draw in draws)
    tail = (1 - level) / 2
    return {"points_a": point_a, "points_b": point_b, "mean_diff": point_a - point_b,
            "ci_low": _quantile(diffs, tail), "ci_high": _quantile(diffs, 1 - tail),
            "share_a_better": sum(d > 0 for d in diffs) / len(diffs),
            "level": level, "repeats": repeats, "seed": seed, "bootstrap": _describe(diffs)}


def repeated_half_splits(pred, targets, queries, conditions, reference, max_points,
                         *, n_splits=20, seed=DEFAULT_SEED):
    """Random condition-stratified halves of the episodes, scored separately.

    Each split puts about half of each condition's episodes in half A and the
    rest in half B. B is recomputed on each half. Returns the distribution of
    points over splits for each half, so split-to-split variation is visible.
    """
    records, *_ = _inputs(pred, targets, queries, conditions, reference)
    by_condition = defaultdict(list)
    for episode in sorted(records):
        by_condition[records[episode][0]].append(episode)
    for condition in CONDITIONS:
        if len(by_condition.get(condition, [])) < 2:
            raise ScoringError(f"half splits need at least two episodes in {condition}")
    rng = random.Random(seed)
    splits = []
    for _ in range(n_splits):
        half_a, half_b = [], []
        for condition in sorted(by_condition):
            episodes = list(by_condition[condition])
            rng.shuffle(episodes)
            cut = (len(episodes) + 1) // 2
            half_a.extend(episodes[:cut])
            half_b.extend(episodes[cut:])
        points_a = _score_from_records([records[e] for e in half_a], max_points)["points"]
        points_b = _score_from_records([records[e] for e in half_b], max_points)["points"]
        splits.append({"half_a": points_a, "half_b": points_b,
                       "n_episodes": {"half_a": len(half_a), "half_b": len(half_b)}})
    return {"n_splits": n_splits, "seed": seed,
            "full_points": _score_from_records(records.values(), max_points)["points"],
            "half_a": _describe(s["half_a"] for s in splits),
            "half_b": _describe(s["half_b"] for s in splits),
            "splits": splits}


# ---------------------------------------------------------------- slices

def _history_kind(ticks, cutoff):
    """'short' if 64 observed ticks whose last 40 are consecutive; else 'long'.

    Matches DATA.md (short = 40 recent consecutive + 24 sampled older ticks).
    The condition labels in the kit's validation split agree with this rule
    on every query (192/192 short_history, 0 others).
    """
    observed = [t for t in ticks if t < cutoff]
    if len(observed) == SHORT_HISTORY_TICKS:
        recent = observed[-SHORT_HISTORY_CONSECUTIVE:]
        if all(b - a == 1 for a, b in zip(recent, recent[1:])):
            return "short"
    return "long"


def _cutoff_bin(cutoff):
    for low, high in CUTOFF_BINS:
        if low <= cutoff <= high:
            return f"{low}-{high}"
    return "other"


def slices(pred, targets, queries, conditions, reference, history=None):
    """Per-slice diagnostics. Each dimension's rows partition all queries.

    Returns ``{dimension: [row, ...]}`` for dimensions condition, machine_type,
    signal, history_kind, cutoff_bin and episode. Each row has n_queries, mse
    (query-averaged 5-class squared error), reference_mse, ratio (mse /
    max(reference_mse, 1e-9)) and win_prob_error (mean of |(1-p0_pred)-(1-p0_true)|).
    Slice MSE is a plain query mean, not the episode-weighted official E[c].
    """
    pred = read_predictions(pred)
    targets = _coerce_targets(targets)
    queries = _coerce(queries, read_queries)
    conditions = _coerce(conditions, read_conditions)
    reference = read_predictions(reference)
    _align(pred, targets, queries, conditions, reference)
    ticks = None
    if history is not None:
        ticks = history if isinstance(history, Mapping) else read_history_ticks(history)

    rows = []
    for key in sorted(targets):
        meta = queries[key]
        truth, forecast = targets[key], pred[key]
        if ticks is None:
            kind = "unknown"
        else:
            kind = _history_kind(ticks.get((meta["episode_id"], meta["machine_id"]), []), meta["cutoff"])
        rows.append({
            "condition": conditions[key], "machine_type": meta["machine_type"], "signal": str(meta["signal"]),
            "history_kind": kind, "cutoff_bin": _cutoff_bin(meta["cutoff"]), "episode": meta["episode_id"],
            "sq": _squared_error(forecast, truth), "ref_sq": _squared_error(reference[key], truth),
            "win": abs((1 - forecast[0]) - (1 - truth[0])),
        })

    result = {}
    for dimension in ("condition", "machine_type", "signal", "history_kind", "cutoff_bin", "episode"):
        groups = defaultdict(list)
        for row in rows:
            groups[row[dimension]].append(row)
        table = []
        for value in sorted(groups):
            members = groups[value]
            mse = _mean(r["sq"] for r in members)
            ref_mse = _mean(r["ref_sq"] for r in members)
            table.append({"value": value, "n_queries": len(members), "mse": mse, "reference_mse": ref_mse,
                          "ratio": mse / max(ref_mse, REFERENCE_FLOOR),
                          "win_prob_error": _mean(r["win"] for r in members)})
        result[dimension] = table
    return result


# ---------------------------------------------------------------- protocols

def kfold_validation_protocol(validation_dir, *, k=4, seed=DEFAULT_SEED):
    """Split validation episodes into k folds, stratified by condition.

    In fold i the episodes listed under ``scored_episodes`` are held out: the
    entry must run without their answers (labels, targets, conditions) and is
    scored on exactly those queries. The ``visible_*`` episodes are the rest.
    Each condition's episodes are spread evenly over the folds, and the scored
    query sets partition all validation queries.
    """
    if not isinstance(k, int) or k < 2:
        raise ScoringError("k must be an integer >= 2")
    base = Path(validation_dir)
    queries = read_queries(base / "queries.csv")
    conditions = read_conditions(base / "conditions.csv")
    _require_same_ids(queries, conditions, "conditions")
    episode_condition = {}
    for key, meta in queries.items():
        episode = meta["episode_id"]
        if episode_condition.setdefault(episode, conditions[key]) != conditions[key]:
            raise ScoringError(f"episode {episode} spans several conditions")
    by_condition = defaultdict(list)
    for episode, condition in sorted(episode_condition.items()):
        by_condition[condition].append(episode)
    for condition, episodes in by_condition.items():
        if len(episodes) < k:
            raise ScoringError(f"condition {condition} has {len(episodes)} episodes, fewer than k={k}")
    rng = random.Random(seed)
    fold_of = {}
    for condition in sorted(by_condition):
        episodes = list(by_condition[condition])
        rng.shuffle(episodes)
        for position, episode in enumerate(episodes):
            fold_of[episode] = position % k
    all_episodes = sorted(episode_condition)
    folds = []
    for fold in range(k):
        held = sorted(e for e in all_episodes if fold_of[e] == fold)
        held_set = set(held)
        scored_queries = sorted(q for q, m in queries.items() if m["episode_id"] in held_set)
        visible_queries = sorted(q for q, m in queries.items() if m["episode_id"] not in held_set)
        folds.append({
            "fold": fold,
            "scored_episodes": held,
            "scored_query_ids": scored_queries,
            "visible_episodes": [e for e in all_episodes if e not in held_set],
            "visible_query_ids": visible_queries,
            "scored_conditions": {c: sum(1 for e in held if episode_condition[e] == c) for c in sorted(by_condition)},
        })
    return folds


def reference_from_starter(dataset_root, *, split="validation", prior_strength=20.0):
    """Predictions of the kit's pooled starter for ``split``, fitted on train only.

    Reproduces the kit's ``starter.py`` pooled method (metadata: "Pooled,
    strength=20"). Labels are used as the starter uses them: train labels
    only, never validation or test answers.
    """
    root = str(KIT_ROOT)
    if root not in sys.path:
        sys.path.append(root)
    from slot_forecasting.baseline import FrequencyForecaster
    from slot_forecasting.io import load_inputs, read_labels
    from slot_forecasting.starter import public_directory

    public = public_directory(dataset_root)
    train_queries, train_history = load_inputs(public / "train")
    labels = read_labels(public / "train/labels.csv", train_queries)
    model = FrequencyForecaster("pooled", prior_strength=prior_strength).fit(train_queries, train_history, labels)
    score_queries, score_history = load_inputs(public / split)
    return model.predict(score_queries, score_history)
