"""Compare PairMixer settings with episode-held-out cutoff predictions.

Usage (from mainFiles/):
  python tune_pair_mixer.py '[{"alpha": 0.01}, {"decay_missing": false}]' [--base '{...}'] [--val]

Each candidate is compared with the base settings using episode-held-out
predictions at the actual query cutoff. Episodes are kept together, so a query's
episode is excluded from fitting that fold's type-level payout prior. The score
uses train/labels.csv, not intermediate history ticks. Per-condition paired
Brier differences are normalized by the official validation reference error;
negative dR is better. The interval resamples whole episodes within conditions.
With --val, each half's priors are estimated from the opposite half, so the
scored episodes do not supply the target-derived priors used for their forecasts.

Two train numbers are reported, both out of fold (8 episode-grouped folds; pairs,
size keys and parity/full are re-chosen inside each fold):
  cutoff  - the 768 labels.csv outcomes at the cutoff tick (the scored tick);
  window  - the same plus the last --window observed history ticks before the
            cutoff, averaged per query first, so the late-episode regime is
            measured with ~10x more outcomes (tighter interval).
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from slot_forecasting.experiments import episode_folds, subset_score
from slot_forecasting.io import load_inputs, read_labels, read_targets
from slot_forecasting.pair_mixer import (CLASS_INDEX, MIN_PROBABILITY, PairMixer, WIN_PRIOR_ETA, choose_pairs,
                                         episode_kfolds, split_episodes,
                                         discover_size_keys, size_modes_from_targets,
                                         win_prior_by_type_from_targets, win_prior_from_targets,
                                         win_prior_from_train)
from slot_forecasting.rounds import round_metadata
from slot_forecasting.schema import PAYOUTS, index_history

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("candidates", help="JSON list of PairMixer keyword settings")
parser.add_argument("--data", default=str(HERE / "public"))
parser.add_argument("--val", action="store_true")
parser.add_argument("--base", default="{}", help="JSON settings to compare against (default: PairMixer defaults); "
                    "candidates are applied on top of it")
parser.add_argument("--window", type=int, default=10, help="history ticks before the cutoff in the window score")
parser.add_argument("--scratch", help="accepted for compatibility; comparisons no longer write scratch files")
args = parser.parse_args()

public = Path(args.data)
train_q, train_h = load_inputs(public / "train")
labels = read_labels(public / "train/labels.csv", train_q)
conditions = {r["query_id"]: r["condition"] for r in csv.DictReader(open(public / "train/conditions.csv"))}
episode = {q.query_id: q.episode_id for q in train_q}
val = public / "validation"
reference = {r["query_id"]: [float(r[c]) for c in ("p_0", "p_1", "p_2", "p_5", "p_20")]
             for r in csv.DictReader(open(val / "reference.csv"))}
val_cond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(val / "conditions.csv"))}
ref_err = defaultdict(lambda: defaultdict(list))
for t in read_targets(val / "targets.csv"):
    ref_err[val_cond[t.query_id]][t.episode_id].append(
        sum((a - b) ** 2 for a, b in zip(reference[t.query_id], t.probabilities)))
baseline_error = {c: sum(sum(v) / len(v) for v in episodes.values()) / len(episodes)
                  for c, episodes in ref_err.items()}


def resolve_settings(settings, prior_targets):
    """Resolve target-derived settings from the supplied validation episodes."""
    settings = dict(settings)
    win_prior_strength = settings.pop("win_prior_strength", 6.0)
    win_prior_shrink = settings.pop("win_prior_shrink", 0.0)
    if settings.get("size_modes") == "val":
        settings["size_modes"] = size_modes_from_targets(prior_targets, val_queries)
    if settings.get("win_prior") == "val-v3":  # combined-v3: fixed 0.45/0.45/0.1 weights, 0.5 split
        wins = sorted(1.0 - t.probabilities[0] for t in prior_targets)
        dead, live = [w for w in wins if w < 0.5], [w for w in wins if w >= 0.5]
        settings["win_prior"] = [(0.45, round(dead[len(dead) // 2], 4), win_prior_strength),
                                 (0.45, round(live[len(live) // 2], 4), win_prior_strength), (0.1, 0.5, 2.0)]
    elif settings.get("win_prior") == "val":
        settings["win_prior"] = win_prior_from_targets(prior_targets, strength=win_prior_strength)
    elif settings.get("win_prior") == "val-by-type":
        settings["win_prior"] = win_prior_by_type_from_targets(
            prior_targets, val_queries, strength=win_prior_strength, global_shrink=win_prior_shrink)
    return settings


val_queries, val_history = load_inputs(val)
val_targets = read_targets(val / "targets.csv")


def episode_oof_errors(settings, prior_targets):
    """Return each train query's Brier error from a model fitted without its episode."""
    fold_by_episode = episode_kfolds(train_q, conditions)
    errors = {}
    for fold in sorted(set(fold_by_episode.values())):
        heldout = {episode_id for episode_id, value in fold_by_episode.items() if value == fold}
        (heldout_queries, heldout_history), (fit_queries, fit_history) = split_episodes(train_q, train_h, heldout)
        fit_labels = {q.query_id: labels[q.query_id] for q in fit_queries}
        model = select_fold_model(settings, fit_queries, fit_history, fit_labels, prior_targets)
        errors.update(window_errors(model, heldout_queries, heldout_history))
    return errors


def brier(probs, outcome_class):
    return sum((p - (1.0 if k == outcome_class else 0.0)) ** 2 for k, p in enumerate(probs))


def window_errors(model, queries, history):
    """{query_id: (cutoff Brier, mean Brier over the cutoff + last args.window observed ticks)}.

    Uses the model's own walk, which emits each tick's forecast before absorbing that tick,
    and the same floor + renormalisation as PairMixer.predict.
    """
    indexed = index_history(queries, history)
    result = {}
    for query in queries:
        own = [None] * (query.cutoff + 1)
        for play in indexed.get((query.episode_id, query.machine_id), []):
            own[play.tick] = CLASS_INDEX[play.payout]
        captured = {}
        model.walk(own, query.machine_type, query.cutoff, captured.__setitem__)
        scores = []
        for tick in range(query.cutoff - args.window, query.cutoff + 1):
            outcome = CLASS_INDEX[labels[query.query_id]] if tick == query.cutoff else own[tick] if tick >= 0 else None
            if outcome is None:
                continue
            probs = [max(p, MIN_PROBABILITY) for p in captured[tick]]
            total = sum(probs)
            scores.append(brier([p / total for p in probs], outcome))
        result[query.query_id] = (scores[-1], sum(scores) / len(scores))
    return result


def cutoff_brier(model, queries, history, fold_labels):
    predictions = model.predict(queries, history)
    return sum(sum((p - (1.0 if PAYOUTS[k] == fold_labels[q.query_id] else 0.0)) ** 2
                   for k, p in enumerate(predictions[q.query_id])) for q in queries) / len(queries)


def select_fold_model(settings, fit_queries, fit_history, fit_labels, prior_targets):
    """Mirror adapt(): discover pairs and choose parity/full using only this fold's labels."""
    resolved = resolve_settings(settings, prior_targets)
    if resolved.get("win_prior") == "train":
        resolved["win_prior"] = win_prior_from_train(fit_queries, fit_history)
    if resolved.get("size_modes") == "train":
        resolved["size_modes"] = None
    if resolved.get("size_keys", "discover") == "discover":
        resolved["size_keys"] = discover_size_keys(fit_queries, fit_history)
    if "pairs" not in resolved:
        resolved["pairs"] = choose_pairs(fit_queries, fit_history)
    if resolved.get("win_prior") and "eta" not in resolved:
        resolved["eta"] = WIN_PRIOR_ETA
    if "cell_mode" in resolved:
        return PairMixer(**resolved).fit(fit_queries, fit_history)
    candidates = [PairMixer(**{**resolved, "cell_mode": mode}).fit(fit_queries, fit_history)
                  for mode in ("parity", "full")]
    scores = [cutoff_brier(model, fit_queries, fit_history, fit_labels) for model in candidates]
    return candidates[scores.index(min(scores))]


def delta_r(diff_by_episode, selected=None):
    """Official condition weighting: average query errors within episode, then episodes."""
    condition_deltas = []
    for condition, episodes in diff_by_episode.items():
        chosen = episodes if selected is None else selected[condition]
        episode_means = [sum(episodes[e]) / len(episodes[e]) for e in chosen]
        condition_deltas.append(sum(episode_means) / len(episode_means) / baseline_error[condition])
    return sum(condition_deltas) / len(condition_deltas)


def comparison_groups(candidate, base_errors, which=0):
    """which = 0 for the cutoff score, 1 for the window score."""
    diffs = defaultdict(lambda: defaultdict(list))
    for query in train_q:
        diffs[conditions[query.query_id]][query.episode_id].append(
            candidate[query.query_id][which] - base_errors[query.query_id][which])
    return diffs


def interval(diff_by_episode, draws=2000):
    rng = random.Random(0)
    samples = []
    for _ in range(draws):
        selected = {condition: [rng.choice(tuple(episodes)) for _ in episodes]
                    for condition, episodes in diff_by_episode.items()}
        samples.append(delta_r(diff_by_episode, selected))
    samples.sort()
    return samples[int(0.025 * draws)], samples[int(0.975 * draws)]


def validation_report(settings):
    """Score each half using priors computed from the opposite half."""
    tuning, audit = episode_folds(val_targets, val_cond)
    scored_predictions = {}
    half_scores = []
    max_points = round_metadata(public)["max_points"]
    for scored_episodes in (tuning, audit):
        prior_targets = tuple(t for t in val_targets if t.episode_id not in scored_episodes)
        model = select_fold_model(settings, train_q, train_h, labels, prior_targets)
        query_ids = {q.query_id for q in val_queries if q.episode_id in scored_episodes}
        queries = tuple(q for q in val_queries if q.query_id in query_ids)
        history = tuple(p for p in val_history if (p.episode_id, p.machine_id) in
                        {(q.episode_id, q.machine_id) for q in queries})
        scored_predictions.update(model.predict(queries, history))
        half_scores.append(subset_score(scored_predictions, val_targets, val_cond, reference,
                                        scored_episodes, max_points))
    all_episodes = {t.episode_id for t in val_targets}
    full_score = subset_score(scored_predictions, val_targets, val_cond, reference, all_episodes,
                              max_points)
    return full_score, half_scores


base_settings = json.loads(args.base)
base_errors = episode_oof_errors(base_settings, val_targets)
print("base: episode-held-out cutoff Brier computed", flush=True)
if args.val:
    full, halves = validation_report(base_settings)
    print(f"base VAL {full['technical_points']:.2f} (cross-fit halves "
          f"{halves[0]['technical_points']:.2f} / {halves[1]['technical_points']:.2f})", flush=True)
for settings in json.loads(args.candidates):
    settings = {**base_settings, **settings}
    cand = episode_oof_errors(settings, val_targets)
    line = json.dumps(settings)
    for which, name in ((0, "OOF cutoff"), (1, f"OOF window[-{args.window}..0]")):
        diff_by_episode = comparison_groups(cand, base_errors, which)
        point = delta_r(diff_by_episode)
        low, high = interval(diff_by_episode)
        per = {condition[:5]: round(sum(sum(v) / len(v) for v in episodes.values()) /
                                    len(episodes) / baseline_error[condition], 4)
               for condition, episodes in sorted(diff_by_episode.items())}
        line += f"\n    {name} dR={point:+.4f} [{low:+.4f},{high:+.4f}]  per-cond {per}"
    if args.val:
        full, halves = validation_report(settings)
        line += (f"\n    cross-fit VAL {full['technical_points']:.2f} "
                 f"(halves {halves[0]['technical_points']:.2f} / {halves[1]['technical_points']:.2f})")
    print(line, flush=True)
