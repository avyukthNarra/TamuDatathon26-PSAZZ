"""v13 (commit a70abc0) plus the short-history win patch.

submission: priors from validation/targets.csv when that file exists, else train-only
frequencies. crossfit: each validation half uses priors fit on the other half
(sha256 episode folds, the same split as run_strongest.py --cv-val).
"""
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ENTRY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENTRY))
sys.path.insert(0, str(ENTRY / "vendor" / "v13"))

from blend import write_predictions  # noqa: E402
from members.hooks import PAIRS_OVERRIDE  # noqa: E402


def resolve_data(path):
    data = Path(path)
    if (data / "public" / "train").is_dir():
        return data / "public"
    return data


def short_frame(data, split):
    from slot_forecasting.io import load_inputs

    queries, history = load_inputs(data / split)
    by_machine = {}
    counts = defaultdict(int)
    for row in history:
        by_machine.setdefault(row.machine_id, {})[row.tick] = row.payout
        counts[(row.episode_id, row.machine_id)] += 1
    cond_path = data / split / "conditions.csv"
    if cond_path.exists():
        with cond_path.open(newline="") as stream:
            cond = {row["query_id"]: row["condition"] for row in csv.DictReader(stream)}
    else:
        # Validation conditions match "fewer than 75% of ticks observed" on all 768 queries.
        cond = {}
        for query in queries:
            observed = counts[(query.episode_id, query.machine_id)]
            cond[query.query_id] = "short_history" if observed < query.cutoff * 0.75 else "stable"
    frame = {
        "Q": [
            {
                "query_id": query.query_id,
                "episode_id": query.episode_id,
                "machine_id": query.machine_id,
                "machine_type": query.machine_type,
                "signal": query.signal,
                "cutoff": query.cutoff,
            }
            for query in queries
        ],
        "H": by_machine,
        "cond": cond,
    }
    return frame


def _replace_win(predictions, probabilities):
    patched = {key: list(value) for key, value in predictions.items()}
    for query_id, p_win in probabilities.items():
        if query_id not in patched:
            continue
        base = patched[query_id]
        denominator = max(1.0 - base[0], 1e-12)
        size = [value / denominator for value in base[1:]]
        scale = sum(size)
        size = [value / scale for value in size]
        p_win = min(max(float(p_win), 1e-6), 1.0 - 1e-6)
        patched[query_id] = [1.0 - p_win] + [p_win * value for value in size]
    return patched


def _read_targets(path):
    out = {}
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            out[row["query_id"]] = [float(row[column]) for column in ("p_0", "p_1", "p_2", "p_5", "p_20")]
    return out


def _halves(frame):
    episode_condition = {}
    for query in frame["Q"]:
        episode_condition[query["episode_id"]] = frame["cond"].get(query["query_id"], "stable")
    first, second = set(), set()
    for condition in sorted(set(episode_condition.values())):
        episodes = sorted(episode for episode, name in episode_condition.items() if name == condition)
        for index, episode in enumerate(episodes):
            (first if index % 2 == 0 else second).add(episode)
    return first, second


def _levels_from_train(train, pairs):
    """Dead/live logit lines from stable train rates, when validation targets are absent."""
    import math

    import numpy as np

    default = {
        "dead": {"a": -2.0, "b": 0.0, "sd": 0.6},
        "live": {"a": 2.0, "b": 0.0, "sd": 0.6},
    }
    out = {}
    for kind in ("steady", "balanced", "jackpot"):
        rows = []
        for query in train["Q"]:
            if query["machine_type"] != kind or train["cond"].get(query["query_id"]) != "stable":
                continue
            history = {
                tick: int(payout > 0)
                for tick, payout in train["H"][query["machine_id"]].items()
                if tick < query["cutoff"]
            }
            best = None
            for lag_a, lag_b in pairs:
                cell_w, cell_n = [0, 0], [0, 0]
                for tick, bit in history.items():
                    if tick - lag_a in history and tick - lag_b in history:
                        agree = int(history[tick - lag_a] == history[tick - lag_b])
                        cell_w[agree] += bit
                        cell_n[agree] += 1
                if cell_n[0] < 5 or cell_n[1] < 5:
                    continue
                rates = [cell_w[i] / cell_n[i] for i in (0, 1)]
                gap = abs(rates[0] - rates[1])
                if best is None or gap > best[0]:
                    best = (gap, rates, cell_n)
            if best is None:
                continue
            _gap, rates, counts = best
            order = (0, 1) if rates[0] <= rates[1] else (1, 0)
            logits = []
            for index in order:
                p = min(max(rates[index], 0.5 / counts[index]), 1.0 - 0.5 / counts[index])
                logits.append(math.log(p / (1.0 - p)))
            rows.append((int(query["signal"]), *logits))
        if len(rows) < 4:
            out[kind] = default
            continue
        signal = np.array([row[0] for row in rows], dtype=float)
        spec = {}
        for offset, name in ((1, "dead"), (2, "live")):
            z = np.array([row[offset] for row in rows])
            design = np.column_stack([np.ones(len(rows)), signal])
            coef = np.linalg.lstsq(design, z, rcond=None)[0]
            residual = z - design @ coef
            spec[name] = {
                "a": round(float(coef[0]), 3),
                "b": round(float(coef[1]), 3),
                "sd": max(round(float(residual.std(ddof=1)), 3), 0.05),
            }
        out[kind] = spec
    return out


def _level_table(data, episodes):
    import short_win

    targets_path = data / "validation" / "targets.csv"
    if not targets_path.exists():
        return None
    frame = short_frame(data, "validation")
    frame["targets"] = _read_targets(targets_path)
    return short_win.fit_level_centres(frame, episodes)


def learned_pairs(data):
    import short_win

    train = short_frame(data, "train")
    if PAIRS_OVERRIDE:
        pairs = [tuple(pair) for pair in PAIRS_OVERRIDE]
        print("pair support override", pairs, flush=True)
        return pairs
    return [tuple(pair) for pair in short_win.learn_pairs(train)]


def apply_jackpot(predictions, data, split, pairs):
    import jackpot_model

    jackpot_model.set_support(pairs)
    train = jackpot_model.load_folder(data / "train")
    params = jackpot_model.fit_params(train)
    strength, _scores = jackpot_model._strength_brier(train, params)
    folder = jackpot_model.load_folder(data / split)
    wins = jackpot_model.predict_folder(folder, params, strength)
    print(f"jackpot patched {len(wins)} queries with {strength}", flush=True)
    return _replace_win(predictions, wins)


def apply_short(predictions, data, split, pairs, protocol):
    import short_win

    frame = short_frame(data, split)
    train = short_frame(data, "train")
    if protocol == "crossfit":
        first, second = _halves(frame)
        probabilities = {}
        for scored, prior_ids in ((first, second), (second, first)):
            try:
                levels = _level_table(data, prior_ids)
            except Exception as exc:
                print(f"short level prior failed ({exc}); using train rates", flush=True)
                levels = None
            levels = levels or _levels_from_train(train, pairs)
            fold = {**frame, "Q": [query for query in frame["Q"] if query["episode_id"] in scored]}
            fold_prob, _, _ = short_win.predict_short(
                fold, grid=True, marginalize=True, pairs=pairs, levels=levels,
            )
            probabilities.update(fold_prob)
            print(f"short crossfit half done, {len(probabilities)} queries", flush=True)
    else:
        if (data / "validation" / "targets.csv").exists():
            val = short_frame(data, "validation")
            try:
                levels = _level_table(data, {query["episode_id"] for query in val["Q"]})
            except Exception as exc:
                print(f"short level prior failed ({exc}); using train rates", flush=True)
                levels = None
        else:
            levels = None
            print("short: no validation targets, using train rate levels", flush=True)
        if levels is None:
            levels = _levels_from_train(train, pairs)
        probabilities, _, _ = short_win.predict_short(
            frame, grid=True, marginalize=True, pairs=pairs, levels=levels,
        )
    return _replace_win(predictions, probabilities)


def fit_predict_submission(data, split):
    from slot_forecasting.combo import V7C, combine
    from slot_forecasting.ensemble import build_ensemble
    from slot_forecasting.io import load_inputs, read_labels, read_targets
    from slot_forecasting.pair_mixer import size_modes_from_targets, win_prior_by_type_from_targets

    train_queries, train_history = load_inputs(data / "train")
    labels_path = data / "train" / "labels.csv"
    labels = read_labels(labels_path, train_queries) if labels_path.exists() else None
    targets_path = data / "validation" / "targets.csv"
    if targets_path.exists():
        targets = read_targets(targets_path)
        validation_queries = load_inputs(data / "validation")[0]
        size_modes = size_modes_from_targets(targets, validation_queries)
        win_prior = win_prior_by_type_from_targets(targets, validation_queries)
        have_targets = True
    else:
        size_modes = win_prior = None
        have_targets = False
        print("v13: no validation targets, using train-only frequencies", flush=True)
    model = build_ensemble(
        train_queries, train_history, labels, size_modes, win_prior,
        log=lambda *_: None, variant="strongest",
    )
    v7c = V7C(data, targets_available=have_targets, saved_priors=None, log=lambda *_: None)
    queries, history = load_inputs(data / split)
    return combine(model.predict(queries, history), v7c.predict_ids(data, split, None))


def fit_predict_crossfit(data):
    from slot_forecasting.combo import V7C, combine
    from slot_forecasting.ensemble import build_ensemble
    from slot_forecasting.experiments import episode_folds
    from slot_forecasting.io import load_inputs, read_labels, read_rows, read_targets
    from slot_forecasting.pair_mixer import size_modes_from_targets, win_prior_by_type_from_targets

    train_queries, train_history = load_inputs(data / "train")
    train_labels = read_labels(data / "train" / "labels.csv", train_queries)
    val_targets = read_targets(data / "validation" / "targets.csv")
    val_queries, val_history = load_inputs(data / "validation")
    groups = {
        row["query_id"]: row["condition"]
        for row in read_rows(data / "validation" / "conditions.csv", ("query_id", "condition"))
    }
    tuning, audit = episode_folds(val_targets, groups)
    crossfit = {}
    for scored, prior_ids in ((tuning, audit), (audit, tuning)):
        prior_targets = tuple(target for target in val_targets if target.episode_id in prior_ids)
        fold_model = build_ensemble(
            train_queries, train_history, train_labels,
            size_modes_from_targets(prior_targets, val_queries),
            win_prior_by_type_from_targets(prior_targets, val_queries),
            log=lambda *_: None, variant="strongest",
        )
        fold_queries = tuple(query for query in val_queries if query.episode_id in scored)
        fold_keys = {(query.episode_id, query.machine_id) for query in fold_queries}
        fold_history = tuple(row for row in val_history if (row.episode_id, row.machine_id) in fold_keys)
        fold_v7c = V7C(data, targets_available=True, prior_episodes=prior_ids, log=lambda *_: None)
        crossfit.update(combine(
            fold_model.predict(fold_queries, fold_history),
            fold_v7c.predict_ids(data, "validation", {query.query_id for query in fold_queries}),
        ))
        print(f"v13 crossfit half done, {len(crossfit)} queries", flush=True)
    return crossfit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--protocol", choices=("submission", "crossfit"), default="submission")
    args = parser.parse_args()
    data = resolve_data(args.data)
    pairs = learned_pairs(data)
    if args.protocol == "crossfit":
        predictions = fit_predict_crossfit(data)
        predictions = apply_jackpot(predictions, data, "validation", pairs)
        predictions = apply_short(predictions, data, "validation", pairs, args.protocol)
    else:
        predictions = fit_predict_submission(data, args.split)
        predictions = apply_jackpot(predictions, data, args.split, pairs)
        predictions = apply_short(predictions, data, args.split, pairs, args.protocol)
    write_predictions(args.out, predictions)
    print(f"v13_short wrote {len(predictions)} rows", flush=True)


if __name__ == "__main__":
    main()
