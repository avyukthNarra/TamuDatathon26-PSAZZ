"""Short-history win filter.

The stable kernel is one of three lag pairs and a fair direction.  A short
history leaves most lag partners unobserved, so each machine carries a
distribution over the last eight win bits.  An observed tick multiplies that
distribution by the parity likelihood and pins the newest bit.  A missing tick
adds both possible bits.  Levels are averaged on the type-by-signal logit
normal prior.  Only the win probability is replaced; v13 supplies the size
distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp

PAIRS = ((1, 3), (2, 5), (4, 7))
LEVELS = {
    "steady": {"a": -1.627, "b": 0.318, "c": 4.638, "s": 0.52, "r": 0.38},
    "balanced": {"a": -2.964, "b": 0.598, "c": 4.850, "s": 0.56, "r": 0.31},
    "jackpot": {"a": -4.196, "b": 0.428, "c": 4.190, "s": 0.63, "r": 0.07},
}
SHOCKS = np.array((-np.sqrt(3.0), 0.0, np.sqrt(3.0)))
SHOCK_WEIGHTS = np.array((1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0))
STATES = np.arange(256)


def hypotheses(machine_type, signal, *, grid):
    spec = LEVELS[machine_type]
    mean_dead = spec["a"] + spec["b"] * signal
    mean_live = mean_dead + spec["c"]
    shocks = SHOCKS if grid else np.zeros(1)
    weights = SHOCK_WEIGHTS if grid else np.ones(1)
    rows = []
    for pair_index, (lag_a, lag_b) in enumerate(PAIRS):
        for xnor in (1, 0):
            for first, first_weight in zip(shocks, weights):
                for second, second_weight in zip(shocks, weights):
                    logit_dead = mean_dead + spec["s"] * first
                    logit_live = mean_live + spec["s"] * (
                        spec["r"] * first + np.sqrt(1.0 - spec["r"] ** 2) * second
                    )
                    rows.append((
                        lag_a, lag_b, xnor,
                        float(expit(logit_dead)), float(expit(logit_live)),
                        (1.0 / 3.0) * 0.5 * float(first_weight * second_weight),
                        pair_index,
                    ))
    lag_a = np.array([row[0] for row in rows])
    lag_b = np.array([row[1] for row in rows])
    xnor = np.array([row[2] for row in rows], dtype=bool)
    p_dead = np.array([row[3] for row in rows])
    p_live = np.array([row[4] for row in rows])
    prior = np.array([row[5] for row in rows])
    prior /= prior.sum()
    bit_a = (STATES[None, :] >> (lag_a[:, None] - 1)) & 1
    bit_b = (STATES[None, :] >> (lag_b[:, None] - 1)) & 1
    agree = bit_a == bit_b
    live = np.where(xnor[:, None], agree, ~agree)
    p_table = np.where(live, p_live[:, None], p_dead[:, None])
    return {
        "lag_a": lag_a,
        "lag_b": lag_b,
        "p_dead": p_dead,
        "p_live": p_live,
        "prior": prior,
        "p_table": p_table,
    }


def _shift(log_alpha, log_lik, bit):
    sources = np.arange(128)
    destination = ((sources << 1) | bit) & 255
    updated = np.full_like(log_alpha, -np.inf)
    updated[:, destination] = np.logaddexp(
        log_alpha[:, sources] + log_lik[:, sources],
        log_alpha[:, sources + 128] + log_lik[:, sources + 128],
    )
    return updated


def filter_win(observed, cutoff, hypothesis, *, marginalize):
    """Return the posterior predictive win probability and log evidence."""
    log_alpha = np.full(hypothesis["p_table"].shape, -np.inf)
    log_alpha[:, 0] = np.log(hypothesis["prior"])
    marginal = 0.5 * (hypothesis["p_dead"] + hypothesis["p_live"])
    observed_loglik = 0.0
    for tick in range(cutoff):
        use_marginal = tick < hypothesis["lag_b"]
        probability = np.where(
            use_marginal[:, None], marginal[:, None], hypothesis["p_table"]
        )
        known = observed.get(tick)
        if known is not None:
            posterior = np.exp(log_alpha - logsumexp(log_alpha))
            predictive = float(np.sum(posterior * probability))
            observed_loglik += np.log(predictive if known else 1.0 - predictive)
        if known is None:
            loss = _shift(log_alpha, np.log(np.clip(1.0 - probability, 1e-15, 1.0)), 0)
            win = _shift(log_alpha, np.log(np.clip(probability, 1e-15, 1.0)), 1)
            log_alpha = np.logaddexp(loss, win)
        else:
            if marginalize:
                informative = np.ones(len(hypothesis["lag_b"]), dtype=bool)
            else:
                informative = np.array([
                    (tick - int(lag_a)) in observed and (tick - int(lag_b)) in observed
                    for lag_a, lag_b in zip(hypothesis["lag_a"], hypothesis["lag_b"])
                ])
            log_lik = np.where(
                informative[:, None],
                np.log(np.clip(probability if known else 1.0 - probability, 1e-15, 1.0)),
                0.0,
            )
            log_alpha = _shift(log_alpha, log_lik, int(known))
    predictive = np.where(
        (cutoff < hypothesis["lag_b"])[:, None],
        marginal[:, None],
        hypothesis["p_table"],
    )
    log_norm = logsumexp(log_alpha)
    posterior = np.exp(log_alpha - log_norm)
    return float(np.sum(posterior * predictive)), float(log_norm), float(observed_loglik)


def observed_wins(history):
    return {tick: int(payout > 0) for tick, payout in history.items()}


def machine_histories(data):
    grouped = defaultdict(dict)
    for query in data["Q"]:
        grouped[(query["episode_id"], query["machine_id"])] = {
            "query": query,
            "history": data["H"][query["machine_id"]],
        }
    return grouped


def predict_short(data, *, grid, marginalize):
    cache = {}
    predictions = {}
    evidence = []
    for query in data["Q"]:
        if data["cond"][query["query_id"]] != "short_history":
            continue
        key = (query["machine_type"], int(query["signal"]), grid)
        if key not in cache:
            cache[key] = hypotheses(query["machine_type"], int(query["signal"]), grid=grid)
        probability, log_evidence, observed_loglik = filter_win(
            observed_wins(data["H"][query["machine_id"]]),
            int(query["cutoff"]),
            cache[key],
            marginalize=marginalize,
        )
        predictions[query["query_id"]] = probability
        evidence.append((log_evidence, observed_loglik))
    return predictions, float(np.sum([row[0] for row in evidence])), float(np.sum([row[1] for row in evidence]))


def patch(v13, win_probabilities):
    patched = {key: value.copy() for key, value in v13.items()}
    for query_id, p_win in win_probabilities.items():
        base = patched[query_id]
        denominator = max(1.0 - base[0], 1e-12)
        size = base[1:] / denominator
        size /= size.sum()
        p_win = float(np.clip(p_win, 1e-6, 1.0 - 1e-6))
        patched[query_id] = np.concatenate(((1.0 - p_win,), p_win * size))
    return patched


def diagnose_short(train):
    """Train-only check for a direction flip inside short histories."""
    rows = []
    for query in train["Q"]:
        if train["cond"][query["query_id"]] != "short_history":
            continue
        history = observed_wins(train["H"][query["machine_id"]])
        best = None
        for lag_a, lag_b in PAIRS:
            cells = defaultdict(lambda: [0, 0])
            usable = []
            for tick, win in history.items():
                if tick - lag_a in history and tick - lag_b in history:
                    agree = int(history[tick - lag_a] == history[tick - lag_b])
                    cells[agree][0] += win
                    cells[agree][1] += 1
                    usable.append(tick)
            counts = sum(item[1] for item in cells.values())
            if counts < 8 or any(item[1] == 0 for item in cells.values()):
                continue
            rates = {key: value[0] / value[1] for key, value in cells.items()}
            gap = abs(rates.get(1, 0.0) - rates.get(0, 0.0))
            if best is None or gap > best[0]:
                best = (gap, counts, usable, rates)
        if best is None:
            continue
        _, counts, usable, rates = best
        ordered = sorted(usable)
        middle = ordered[len(ordered) // 2]
        early = late = None
        for lag_a, lag_b in PAIRS:
            for label, ticks in (("early", [t for t in ordered if t < middle]), ("late", [t for t in ordered if t >= middle])):
                pass
        early_gap = late_gap = None
        lag_a, lag_b = max(PAIRS, key=lambda pair: _gap(history, pair))
        early_rates = _rates(history, (lag_a, lag_b), [t for t in ordered if t < middle])
        late_rates = _rates(history, (lag_a, lag_b), [t for t in ordered if t >= middle])
        flip = None
        if early_rates and late_rates:
            early_live = max(early_rates, key=early_rates.get)
            late_live = max(late_rates, key=late_rates.get)
            flip = early_live != late_live
            early_gap = abs(early_rates.get(1, 0) - early_rates.get(0, 0))
            late_gap = abs(late_rates.get(1, 0) - late_rates.get(0, 0))
        rows.append({
            "usable": counts,
            "gap": best[0],
            "flip": flip,
            "early_gap": early_gap,
            "late_gap": late_gap,
            "episode": query["episode_id"],
        })
    return rows


def _rates(history, pair, ticks):
    cells = defaultdict(lambda: [0, 0])
    for tick in ticks:
        if tick - pair[0] in history and tick - pair[1] in history:
            agree = int(history[tick - pair[0]] == history[tick - pair[1]])
            cells[agree][0] += history[tick]
            cells[agree][1] += 1
    if len(cells) < 2 or any(value[1] < 3 for value in cells.values()):
        return None
    return {key: value[0] / value[1] for key, value in cells.items()}


def _gap(history, pair):
    rates = _rates(history, pair, list(history))
    if not rates:
        return -1.0
    return abs(rates.get(1, 0.0) - rates.get(0, 0.0))


def load_environment():
    orch = Path("/Users/saisreed/Desktop/Coding/TamuDatathon26-PSAZZ/worktrees/fresh-orch")
    sys.path.insert(0, str(orch))
    from orch.common import folds, load, points
    public = Path("/Users/saisreed/Desktop/Coding/TamuDatathon26-PSAZZ/worktrees/fresh-bayes/mainFiles/public")
    return load, points, folds, load("train", public), load("validation", public)


def score_variants(out_dir):
    load, points, folds, train, validation = load_environment()
    v13_path = Path("/Users/saisreed/Desktop/Coding/TamuDatathon26-PSAZZ/worktrees/_tm/v13/mainFiles/runs/v13/crossfit_validation_predictions.csv")
    import csv
    v13 = {
        row["query_id"]: np.array([float(row[column]) for column in ("p_0", "p_1", "p_2", "p_5", "p_20")])
        for row in csv.DictReader(v13_path.open())
    }
    base_points, base_ratios = points(v13, validation)
    truth = {
        query_id: 1.0 - validation["targets"][query_id][0]
        for query_id, condition in validation["cond"].items()
        if condition == "short_history"
    }
    oracle_points, oracle_ratios = points(patch(v13, truth), validation)
    diagnosis = diagnose_short(train)
    variants = {}
    fold_a, fold_b = folds(validation)
    short_ids = {
        "A": {q["query_id"] for q in validation["Q"] if q["episode_id"] in fold_a and validation["cond"][q["query_id"]] == "short_history"},
        "B": {q["query_id"] for q in validation["Q"] if q["episode_id"] in fold_b and validation["cond"][q["query_id"]] == "short_history"},
    }
    for name, grid, marginalize in (
        ("plugin_drop", False, False),
        ("plugin_marginal", False, True),
        ("grid_drop", True, False),
        ("grid_marginal", True, True),
    ):
        train_predictions, train_evidence, train_observed = predict_short(train, grid=grid, marginalize=marginalize)
        label_loglik = 0.0
        for query in train["Q"]:
            if train["cond"][query["query_id"]] != "short_history":
                continue
            probability = train_predictions[query["query_id"]]
            won = int(train["labels"][query["query_id"]] > 0)
            label_loglik += np.log(probability if won else 1.0 - probability)
        validation_predictions, _, _ = predict_short(validation, grid=grid, marginalize=marginalize)
        patched = patch(v13, validation_predictions)
        full_points, full_ratios = points(patched, validation)
        a_points, a_ratios = points(patched, validation, [q["query_id"] for q in validation["Q"] if q["episode_id"] in fold_a])
        b_points, b_ratios = points(patched, validation, [q["query_id"] for q in validation["Q"] if q["episode_id"] in fold_b])
        variants[name] = {
            "train_log_evidence": train_evidence,
            "train_observed_loglik": train_observed,
            "train_label_loglik": float(label_loglik),
            "full_points": full_points,
            "full_ratios": full_ratios,
            "fold_a_points": a_points,
            "fold_a_ratios": a_ratios,
            "fold_b_points": b_points,
            "fold_b_ratios": b_ratios,
            "predictions": validation_predictions,
        }
        print(name, round(full_points, 4), {k: round(float(v), 4) for k, v in full_ratios.items()}, "observed_ll", round(train_observed, 2), flush=True)
    chosen_b = max(variants, key=lambda name: variants[name]["fold_a_points"])
    chosen_a = max(variants, key=lambda name: variants[name]["fold_b_points"])
    cross = {key: value.copy() for key, value in v13.items()}
    for query in validation["Q"]:
        if validation["cond"][query["query_id"]] != "short_history":
            continue
        choice = chosen_a if query["episode_id"] in fold_a else chosen_b
        cross = patch(cross, {query["query_id"]: variants[choice]["predictions"][query["query_id"]]})
    cross_points, cross_ratios = points(cross, validation)
    train_choice = max(variants, key=lambda name: variants[name]["train_observed_loglik"])
    report = {
        "v13_crossfit_points": base_points,
        "v13_ratios": base_ratios,
        "oracle_short_win_points": oracle_points,
        "oracle_short_ratios": oracle_ratios,
        "diagnosis": {
            "machines": len(diagnosis),
            "mean_usable": float(np.mean([row["usable"] for row in diagnosis])),
            "mean_gap": float(np.mean([row["gap"] for row in diagnosis])),
            "flip_fraction": float(np.mean([row["flip"] for row in diagnosis if row["flip"] is not None])),
            "flip_comparisons": int(sum(row["flip"] is not None for row in diagnosis)),
            "mean_early_gap": float(np.nanmean([row["early_gap"] if row["early_gap"] is not None else np.nan for row in diagnosis])),
            "mean_late_gap": float(np.nanmean([row["late_gap"] if row["late_gap"] is not None else np.nan for row in diagnosis])),
        },
        "variants": {
            name: {key: value for key, value in row.items() if key != "predictions"}
            for name, row in variants.items()
        },
        "crossfit": {
            "fold_b_model": chosen_b,
            "fold_a_model": chosen_a,
            "points": cross_points,
            "ratios": cross_ratios,
        },
        "train_evidence_choice": train_choice,
        "train_evidence_points": variants[train_choice]["full_points"],
        "train_evidence_ratios": variants[train_choice]["full_ratios"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "short_report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps(report, indent=2, default=float))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="mainFiles/runs/short-win")
    args = parser.parse_args()
    score_variants(Path(args.out))


if __name__ == "__main__":
    main()
