"""Generic, train-only candidate evaluation against a supplied baseline."""

import argparse
import hashlib
import importlib
import json
import math
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from slot_forecasting.contest import contest_score
from slot_forecasting.experiments import episode_folds, subset_score
from slot_forecasting.io import (
    load_inputs,
    read_labels,
    read_rows,
    read_submission,
    read_targets,
    require_ids,
    write_json,
    write_submission,
)
from slot_forecasting.rounds import VERSION, round_metadata
from slot_forecasting.scoring import distribution_error
from slot_forecasting.schema import CONDITIONS, ValidationError

HERE = Path(__file__).resolve().parent


def _score_summary(predictions, targets, groups, reference, max_points):
    score = contest_score(predictions, targets, groups, reference, max_points=max_points, version=VERSION)
    return {
        "technical_points": score["technical_points"],
        "mean_loss_ratio": score["mean_loss_ratio"],
        "by_condition": score["by_condition"],
    }


def _percentile(values, level):
    ordered = sorted(values)
    position = (len(ordered) - 1) * level
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_points_bootstrap(candidate, baseline, reference, targets, groups, *,
                            max_points=60, repeats=1000, seed=314159):
    """Paired, condition-stratified episode bootstrap of candidate-minus-baseline points.

    Reference errors are recomputed from the same resampled episodes as both models on
    every draw, matching the event's per-condition error ratio.
    """
    targets = tuple(targets)
    ids = {target.query_id for target in targets}
    if not ids:
        raise ValidationError("Bootstrap requires at least one episode.")
    require_ids(candidate, ids)
    require_ids(baseline, ids)
    require_ids(reference, ids)
    require_ids(groups, ids)
    if set(groups.values()) != set(CONDITIONS):
        raise ValidationError("Bootstrap requires every announced condition.")

    episode_errors = {"candidate": defaultdict(dict), "baseline": defaultdict(dict), "reference": defaultdict(dict)}
    episode_conditions = {}
    for target in targets:
        condition = groups[target.query_id]
        previous = episode_conditions.setdefault(target.episode_id, condition)
        if previous != condition:
            raise ValidationError("An episode cannot belong to multiple conditions.")
        for label, predictions in (("candidate", candidate), ("baseline", baseline), ("reference", reference)):
            row = episode_errors[label][condition].setdefault(target.episode_id, [])
            row.append(distribution_error(predictions[target.query_id], target.probabilities))
    means = {
        label: {condition: {episode: math.fsum(errors) / len(errors) for episode, errors in by_episode.items()}
                for condition, by_episode in by_condition.items()}
        for label, by_condition in episode_errors.items()
    }
    if any(not means["candidate"].get(condition) for condition in CONDITIONS):
        raise ValidationError("Bootstrap requires episodes in every condition.")

    def points(sampled):
        ratios = []
        for condition in CONDITIONS:
            draws = sampled[condition]
            ref_error = math.fsum(means["reference"][condition][episode] for episode in draws) / len(draws)
            base_error = math.fsum(means["baseline"][condition][episode] for episode in draws) / len(draws)
            cand_error = math.fsum(means["candidate"][condition][episode] for episode in draws) / len(draws)
            denominator = max(1e-9, ref_error)
            ratios.append((cand_error / denominator, base_error / denominator))
        cand_ratio = math.fsum(pair[0] for pair in ratios) / len(CONDITIONS)
        base_ratio = math.fsum(pair[1] for pair in ratios) / len(CONDITIONS)
        return max_points / (1 + cand_ratio) - max_points / (1 + base_ratio)

    observed = {condition: sorted(means["candidate"][condition]) for condition in CONDITIONS}
    point = points(observed)
    repeats = int(repeats)
    if repeats < 1:
        raise ValidationError("Bootstrap repeats must be at least one.")
    rng = random.Random(seed)
    samples = []
    for _ in range(repeats):
        draw = {condition: [rng.choice(episodes) for _ in episodes] for condition, episodes in observed.items()}
        samples.append(points(draw))
    return {"point_delta": point, "ci95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
            "repeats": repeats, "seed": seed}


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True,
                              text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_branch():
    try:
        return subprocess.run(["git", "branch", "--show-current"], check=True,
                              capture_output=True, text=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _simple_config(model):
    """Keep constructor state that can be represented directly in JSON."""
    def clean(value):
        if value is None or isinstance(value, (str, int, bool)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else _UNSERIALIZABLE
        if isinstance(value, (tuple, list)):
            converted = [clean(item) for item in value]
            return converted if all(item is not _UNSERIALIZABLE for item in converted) else _UNSERIALIZABLE
        if isinstance(value, dict) and all(isinstance(key, str) for key in value):
            converted = {key: clean(item) for key, item in value.items()}
            return converted if all(item is not _UNSERIALIZABLE for item in converted.values()) else _UNSERIALIZABLE
        return _UNSERIALIZABLE

    config = {}
    for key, value in vars(model).items():
        cleaned = clean(value)
        if cleaned is not _UNSERIALIZABLE:
            config[key] = cleaned
    return config


class _Unserializable:
    pass


_UNSERIALIZABLE = _Unserializable()


def _prediction_map(value, ids):
    if isinstance(value, (str, Path)):
        return read_submission(value, ids)
    if not isinstance(value, dict):
        raise ValidationError("baseline_predictions must be a CSV path or prediction mapping.")
    require_ids(value, ids)
    return value


def run_experiment(factory, *, data, name, baseline_factory=None, baseline_predictions=None,
                   results_path=HERE / "runs" / "harness_results.jsonl", out=None,
                   repeats=1000, seed=314159):
    """Fit a candidate on training inputs and score it on validation episodes."""
    if (baseline_factory is None) == (baseline_predictions is None):
        raise ValidationError("Provide exactly one of baseline_factory or baseline_predictions.")
    public = Path(data).resolve()
    metadata = round_metadata(public)
    train_q, train_h = load_inputs(public / "train")
    train_labels = read_labels(public / "train/labels.csv", train_q)
    val_q, val_h = load_inputs(public / "validation")
    targets = read_targets(public / "validation/targets.csv")
    val_ids = {query.query_id for query in val_q}
    require_ids({target.query_id for target in targets}, val_ids)
    group_rows = read_rows(public / "validation/conditions.csv", ("query_id", "condition"))
    groups = {row["query_id"]: row["condition"] for row in group_rows}
    if len(groups) != len(group_rows):
        raise ValidationError("Duplicate condition query_id.")
    require_ids(groups, val_ids)
    reference = read_submission(public / "validation/reference.csv", val_ids)
    model = factory()
    model_config = _simple_config(model)
    started = perf_counter()
    model.fit(train_q, train_h, train_labels)
    candidate = model.predict(val_q, val_h)
    runtime_seconds = perf_counter() - started
    require_ids(candidate, val_ids)
    baseline_runtime_seconds = None
    baseline_path = None
    baseline_sha = None
    if baseline_factory is not None:
        baseline_model = baseline_factory()
        baseline_started = perf_counter()
        baseline_model.fit(train_q, train_h, train_labels)
        baseline = baseline_model.predict(val_q, val_h)
        baseline_runtime_seconds = perf_counter() - baseline_started
        require_ids(baseline, val_ids)
    else:
        baseline = _prediction_map(baseline_predictions, val_ids)
        if isinstance(baseline_predictions, (str, Path)):
            baseline_path = str(Path(baseline_predictions).resolve())
            baseline_sha = hashlib.sha256(Path(baseline_path).read_bytes()).hexdigest()
    if out is not None:
        write_submission(out, candidate)

    tuning, audit = episode_folds(targets, groups)
    splits = {"full": None, "tuning": tuning, "audit": audit}
    scores = {}
    bootstrap = {}
    for split_name, split_episodes in splits.items():
        selected = targets if split_episodes is None else tuple(
            target for target in targets if target.episode_id in split_episodes)
        selected_ids = {target.query_id for target in selected}
        selected_groups = {key: groups[key] for key in selected_ids}
        selected_reference = {key: reference[key] for key in selected_ids}
        cand = {key: candidate[key] for key in selected_ids}
        base = {key: baseline[key] for key in selected_ids}
        if split_episodes is None:
            candidate_score = _score_summary(cand, selected, selected_groups, selected_reference,
                                              metadata["max_points"])
            baseline_score = _score_summary(base, selected, selected_groups, selected_reference,
                                             metadata["max_points"])
        else:
            candidate_score = subset_score(candidate, targets, groups, reference, split_episodes,
                                           metadata["max_points"])
            baseline_score = subset_score(baseline, targets, groups, reference, split_episodes,
                                          metadata["max_points"])
            candidate_score = {key: candidate_score[key] for key in
                               ("technical_points", "mean_loss_ratio", "by_condition")}
            baseline_score = {key: baseline_score[key] for key in
                              ("technical_points", "mean_loss_ratio", "by_condition")}
        scores[split_name] = {"candidate": candidate_score, "baseline": baseline_score}
        bootstrap[split_name] = paired_points_bootstrap(
            cand, base, selected_reference, selected, selected_groups,
            max_points=metadata["max_points"], repeats=repeats, seed=seed)

    result = {
        "name": name,
        "branch": _git_branch(),
        "branch_sha": _git_sha(),
        "candidate_factory": f"{getattr(factory, '__module__', '')}:{getattr(factory, '__qualname__', repr(factory))}",
        "candidate_config": model_config,
        "prior_source": getattr(model, "prior_source", None),
        "max_points": metadata["max_points"],
        "round": metadata["round"],
        "season_id": metadata["season_id"],
        "runtime_seconds": runtime_seconds,
        "baseline_runtime_seconds": baseline_runtime_seconds,
        "baseline_factory": (f"{getattr(baseline_factory, '__module__', '')}:"
                             f"{getattr(baseline_factory, '__qualname__', repr(baseline_factory))}"
                             if baseline_factory is not None else None),
        "baseline_path": baseline_path,
        "baseline_sha256": baseline_sha,
        "scores": scores,
        "paired_bootstrap": bootstrap,
        "tuning_episodes": sorted(tuning),
        "audit_episodes": sorted(audit),
        "bootstrap_seed": seed,
        "bootstrap_repeats": repeats,
    }
    destination = Path(results_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    return result


def _load_factory(spec):
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValidationError("Candidate must use module:callable syntax.")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise ValidationError("Candidate factory must be callable.")
    return factory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="factory import path: module:callable")
    baseline = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument("--baseline", help="train-only baseline factory import path: module:callable")
    baseline.add_argument("--baseline-predictions", help="cached baseline validation prediction CSV")
    parser.add_argument("--name", required=True, help="candidate name for the result record")
    parser.add_argument("--data", default=str(HERE / "public"))
    parser.add_argument("--out", help="optional candidate validation prediction CSV, e.g. runs/name.csv")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--results", default=str(HERE / "runs" / "harness_results.jsonl"))
    args = parser.parse_args(argv)
    result = run_experiment(_load_factory(args.candidate), data=args.data, name=args.name,
                            baseline_factory=_load_factory(args.baseline) if args.baseline else None,
                            baseline_predictions=args.baseline_predictions,
                            results_path=args.results, out=args.out,
                            repeats=args.bootstrap, seed=args.seed)
    condition_order = ("stable", "short_history", "drift", "shared_change")
    for split in ("full", "tuning", "audit"):
        scored = result["scores"][split]["candidate"]
        errors = ",".join(f"{scored['by_condition'][condition]['error']:.4f}"
                           for condition in condition_order)
        interval = result["paired_bootstrap"][split]
        print(f"{split}: {scored['technical_points']:.2f} pts; errors [{errors}]; "
              f"paired Δ {interval['point_delta']:+.2f} [{interval['ci95'][0]:+.2f},{interval['ci95'][1]:+.2f}]")
    baseline_runtime = result["baseline_runtime_seconds"]
    baseline_text = "cached" if baseline_runtime is None else f"{baseline_runtime:.3f}s"
    print(f"runtime: candidate {result['runtime_seconds']:.3f}s, baseline {baseline_text}")


if __name__ == "__main__":
    main()
