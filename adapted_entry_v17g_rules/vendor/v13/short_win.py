"""Short-history win filter.

The kernel is the lag-pair support learned from this round's train histories.  A short
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

import math
from itertools import combinations

ALL_PAIRS = tuple(combinations(range(1, 9), 2))
MIN_SHARE = 0.05
MIN_MARGIN = 2.0
MIN_CELL = 5
SHOCKS = np.array((-np.sqrt(3.0), 0.0, np.sqrt(3.0)))
SHOCK_WEIGHTS = np.array((1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0))
STATES = np.arange(256)


def hypotheses(machine_type, signal, *, grid, pairs, levels):
    spec = levels[machine_type]
    shocks = SHOCKS if grid else np.zeros(1)
    weights = SHOCK_WEIGHTS if grid else np.ones(1)
    rows = []
    pair_weight = 1.0 / len(pairs)
    for pair_index, (lag_a, lag_b) in enumerate(pairs):
        for xnor in (1, 0):
            for first, first_weight in zip(shocks, weights):
                for second, second_weight in zip(shocks, weights):
                    logit_dead = spec["dead"]["a"] + spec["dead"]["b"] * signal + spec["dead"]["sd"] * first
                    logit_live = spec["live"]["a"] + spec["live"]["b"] * signal + spec["live"]["sd"] * second
                    rows.append((
                        lag_a, lag_b, xnor,
                        float(expit(logit_dead)), float(expit(logit_live)),
                        pair_weight * 0.5 * float(first_weight * second_weight),
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


def predict_short(data, *, grid, marginalize, pairs, levels, only_short=True):
    cache = {}
    predictions = {}
    evidence = []
    for query in data["Q"]:
        if only_short and data["cond"][query["query_id"]] != "short_history":
            continue
        key = (query["machine_type"], int(query["signal"]), grid, id(levels))
        if key not in cache:
            cache[key] = hypotheses(query["machine_type"], int(query["signal"]), grid=grid, pairs=pairs, levels=levels)
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


def diagnose_short(train, pairs):
    """Train-only check for a direction flip inside short histories."""
    rows = []
    for query in train["Q"]:
        if train["cond"][query["query_id"]] != "short_history":
            continue
        history = observed_wins(train["H"][query["machine_id"]])
        best = None
        for lag_a, lag_b in pairs:
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
        for lag_a, lag_b in pairs:
            for label, ticks in (("early", [t for t in ordered if t < middle]), ("late", [t for t in ordered if t >= middle])):
                pass
        early_gap = late_gap = None
        lag_a, lag_b = max(pairs, key=lambda pair: _gap(history, pair))
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


def _bern(wins, n):
    if n <= 0:
        return 0.0
    p = min(max(wins / n, 1e-6), 1.0 - 1e-6)
    return wins * math.log(p) + (n - wins) * math.log(1.0 - p)


def learn_pairs(train, min_share=MIN_SHARE, min_margin=MIN_MARGIN, min_cell=MIN_CELL):
    """Same vote as research_exact.model.learn_pairs, on the orch history dict."""
    votes = {}
    n = len(train["Q"])
    for query in train["Q"]:
        history = observed_wins(train["H"][query["machine_id"]])
        cutoff = int(query["cutoff"])
        wins = {tick: bit for tick, bit in history.items() if tick < cutoff}
        scored = []
        for lag_a, lag_b in ALL_PAIRS:
            cell_w = [0, 0]
            cell_n = [0, 0]
            for tick, bit in wins.items():
                if tick - lag_a in wins and tick - lag_b in wins:
                    agree = int(wins[tick - lag_a] == wins[tick - lag_b])
                    cell_w[agree] += bit
                    cell_n[agree] += 1
            if cell_n[0] < min_cell or cell_n[1] < min_cell:
                continue
            gain = (_bern(cell_w[0], cell_n[0]) + _bern(cell_w[1], cell_n[1])
                    - _bern(cell_w[0] + cell_w[1], cell_n[0] + cell_n[1]))
            scored.append((gain, (lag_a, lag_b)))
        if len(scored) < 2:
            continue
        scored.sort(reverse=True)
        if scored[0][0] - scored[1][0] < min_margin:
            continue
        votes[scored[0][1]] = votes.get(scored[0][1], 0) + 1
    kept = [pair for pair, count in votes.items() if n and count / n >= min_share]
    if not kept:
        kept = list(ALL_PAIRS)
    kept = sorted(kept)
    print("pair support", kept, "top", sorted(votes.items(), key=lambda item: -item[1])[:6], "of", n, flush=True)
    return kept


def fit_level_centres(data, episodes):
    """2-means logit centres on one validation fold. Dead is the lower centre."""
    by_id = {query["query_id"]: query for query in data["Q"]}
    out = {}
    for kind in ("steady", "balanced", "jackpot"):
        rows = []
        for query_id, prob in data["targets"].items():
            query = by_id[query_id]
            if query["machine_type"] != kind or query["episode_id"] not in episodes:
                continue
            p = min(max(1.0 - float(prob[0]), 1e-6), 1.0 - 1e-6)
            rows.append((int(query["signal"]), math.log(p / (1.0 - p))))
        z = np.array([row[1] for row in rows])
        signal = np.array([row[0] for row in rows], dtype=float)
        centres = np.array([np.quantile(z, 0.2), np.quantile(z, 0.8)])
        for _ in range(50):
            label = np.abs(z[:, None] - centres[None]).argmin(1)
            centres = np.array([z[label == k].mean() for k in (0, 1)])
        if centres[0] > centres[1]:
            centres = centres[::-1]
            label = 1 - label
        spec = {}
        for k, name in enumerate(("dead", "live")):
            mask = label == k
            design = np.column_stack([np.ones(int(mask.sum())), signal[mask]])
            coef, *_ = np.linalg.lstsq(design, z[mask], rcond=None)
            resid = z[mask] - design @ coef
            ddof = 2 if int(mask.sum()) > 2 else 1
            spec[name] = {
                "a": round(float(coef[0]), 3),
                "b": round(float(coef[1]), 3),
                "sd": max(round(float(resid.std(ddof=ddof)), 3), 0.05),
            }
        out[kind] = spec
    return out


def fit_size_atoms(data, episodes):
    """Two winning-size atoms per type, from the other fold's targets."""
    by_id = {query["query_id"]: query for query in data["Q"]}
    payouts = (1, 2, 5, 20)
    out = {}
    for kind in ("steady", "balanced", "jackpot"):
        rows = []
        for query_id, prob in data["targets"].items():
            query = by_id[query_id]
            if query["machine_type"] != kind or query["episode_id"] not in episodes:
                continue
            win = 1.0 - float(prob[0])
            if win < 0.02:
                continue
            rows.append(np.array(prob[1:], dtype=float) / win)
        if len(rows) < 4:
            pooled = np.ones(4) / 4.0 if not rows else np.mean(rows, axis=0)
            pooled = pooled / pooled.sum()
            out[kind] = np.vstack([pooled, pooled])
            continue
        x = np.array(rows)
        centres = np.quantile(x, [0.25, 0.75], axis=0)
        label = np.zeros(len(x), dtype=int)
        for _ in range(40):
            dist = ((x[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
            label = dist.argmin(1)
            for k in (0, 1):
                if np.any(label == k):
                    centres[k] = x[label == k].mean(0)
        centres = centres / centres.sum(1, keepdims=True)
        order = np.argsort(-centres[:, 0])
        out[kind] = centres[order]
    return out


def size_posterior(history, cutoff, atoms):
    counts = np.zeros(4)
    index = {1: 0, 2: 1, 5: 2, 20: 3}
    for tick, payout in history.items():
        if tick >= cutoff or payout == 0:
            continue
        counts[index[int(payout)]] += 1
    if counts.sum() == 0:
        mix = atoms.mean(0)
    else:
        log_lik = np.log(np.clip(atoms, 1e-12, 1.0)) @ counts
        weight = np.exp(log_lik - logsumexp(log_lik))
        mix = weight @ atoms
    return mix / mix.sum()


def compose_forecast(data, wins, atoms_for):
    out = {}
    for query in data["Q"]:
        query_id = query["query_id"]
        if query_id not in wins:
            continue
        size = size_posterior(data["H"][query["machine_id"]], int(query["cutoff"]), atoms_for(query))
        p_win = float(np.clip(wins[query_id], 1e-6, 1.0 - 1e-6))
        out[query_id] = np.concatenate(((1.0 - p_win,), p_win * size))
    return out


def load_environment(public=None):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "orch"))
    from common import folds, load, points
    public = Path(public) if public else root / "mainFiles" / "public"
    meta = json.loads((public / "metadata.json").read_text())
    return (load, points, folds, load("train", public), load("validation", public),
            public, float(meta.get("max_points", 60)))


def _slice(data, episodes):
    return {**data, "Q": [query for query in data["Q"] if query["episode_id"] in episodes]}


def score_variants(out_dir, public=None, v13_path=None):
    out_dir = Path(out_dir)
    load, points, folds, train, validation, public, max_points = load_environment(public)
    root = Path(__file__).resolve().parents[1]
    if v13_path is None:
        v13_path = root / "orch" / "results" / "v13_crossfit.csv"
    v13_path = Path(v13_path) if v13_path else None
    import csv
    v13 = None
    if v13_path is not None and v13_path.exists():
        v13 = {
            row["query_id"]: np.array([float(row[column]) for column in ("p_0", "p_1", "p_2", "p_5", "p_20")])
            for row in csv.DictReader(v13_path.open())
        }
    pairs = learn_pairs(train)
    fold_a, fold_b = folds(validation)
    levels = {"A": fit_level_centres(validation, fold_a), "B": fit_level_centres(validation, fold_b)}
    atoms = {"A": fit_size_atoms(validation, fold_a), "B": fit_size_atoms(validation, fold_b)} if v13 is None else None
    print("levels from A", levels["A"], flush=True)
    only_short = v13 is not None

    def forecast(wins):
        if v13 is not None:
            return patch(v13, wins)

        def atoms_for(query):
            side = "B" if query["episode_id"] in fold_a else "A"
            return atoms[side][query["machine_type"]]

        return compose_forecast(validation, wins, atoms_for)

    base_points = base_ratios = None
    if v13 is not None:
        base_points, base_ratios = points(v13, validation, max_points=max_points)
    diagnosis = diagnose_short(train, pairs)
    variants = {}
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
        # Fold A is scored with centres fit on B, and the reverse.
        pred_a, _, _ = predict_short(_slice(validation, fold_a), grid=grid, marginalize=marginalize, pairs=pairs, levels=levels["B"], only_short=only_short)
        pred_b, _, _ = predict_short(_slice(validation, fold_b), grid=grid, marginalize=marginalize, pairs=pairs, levels=levels["A"], only_short=only_short)
        validation_predictions = {**pred_a, **pred_b}
        patched = forecast(validation_predictions)
        full_points, full_ratios = points(patched, validation, max_points=max_points)
        a_ids = [q["query_id"] for q in validation["Q"] if q["episode_id"] in fold_a]
        b_ids = [q["query_id"] for q in validation["Q"] if q["episode_id"] in fold_b]
        a_points, a_ratios = points(patched, validation, a_ids, max_points=max_points)
        b_points, b_ratios = points(patched, validation, b_ids, max_points=max_points)
        variants[name] = {
            "full_points": full_points,
            "full_ratios": full_ratios,
            "fold_a_points": a_points,
            "fold_a_ratios": a_ratios,
            "fold_b_points": b_points,
            "fold_b_ratios": b_ratios,
            "predictions": validation_predictions,
        }
        print(name, round(full_points, 4), {k: round(float(v), 4) for k, v in full_ratios.items()}, flush=True)
    chosen_b = max(variants, key=lambda name: variants[name]["fold_a_points"])
    chosen_a = max(variants, key=lambda name: variants[name]["fold_b_points"])
    cross_wins = {}
    for query in validation["Q"]:
        if only_short and validation["cond"][query["query_id"]] != "short_history":
            continue
        choice = chosen_a if query["episode_id"] in fold_a else chosen_b
        cross_wins[query["query_id"]] = variants[choice]["predictions"][query["query_id"]]
    cross = forecast(cross_wins)
    cross_points, cross_ratios = points(cross, validation, max_points=max_points)
    report = {
        "pairs": pairs,
        "max_points": max_points,
        "v13_crossfit_points": base_points,
        "v13_ratios": base_ratios,
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
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "short_report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps(report, indent=2, default=float))
    return report


def write_selected(out_dir, model="grid_marginal"):
    """Cross-fit entry point. `model` is ignored; the other fold picks the variant."""
    del model
    return score_variants(out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="mainFiles/runs/short-win")
    parser.add_argument("--public", default=None)
    parser.add_argument("--v13", default=None)
    parser.add_argument("--no-v13", action="store_true")
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    if args.export:
        write_selected(out)
    else:
        score_variants(out, public=args.public, v13_path="" if args.no_v13 else args.v13)


if __name__ == "__main__":
    main()
