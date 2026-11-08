"""Small, importable evaluator for the main-round validation set.

The module intentionally loads only public train/validation inputs.  It mirrors
``slot_forecasting.contest.contest_score`` while making episode folds and
episode bootstrap intervals easy to reuse in experiments.
"""

from collections import defaultdict
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
KIT_ROOT = ROOT if (ROOT / "mainFiles").is_dir() else ROOT.parent.parent / "fresh-eda"
sys.path.insert(0, str(KIT_ROOT))
from mainFiles.slot_forecasting.baseline import FrequencyForecaster
from mainFiles.slot_forecasting.contest import contest_score
from mainFiles.slot_forecasting.io import (
    load_inputs, read_labels, read_rows, read_submission, read_targets,
)
from mainFiles.slot_forecasting.rounds import round_metadata
from mainFiles.slot_forecasting.schema import CONDITIONS
from mainFiles.slot_forecasting.scoring import distribution_error


PUBLIC = KIT_ROOT / "mainFiles" / "public"
TARGETS = read_targets(PUBLIC / "validation" / "targets.csv")
TARGET_BY_ID = {target.query_id: target for target in TARGETS}
CONDITION_BY_ID = {
    row["query_id"]: row["condition"]
    for row in read_rows(PUBLIC / "validation" / "conditions.csv", ("query_id", "condition"))
}
REFERENCE = read_submission(
    PUBLIC / "validation" / "reference.csv", set(TARGET_BY_ID)
)
QUERIES, _ = load_inputs(PUBLIC / "validation")
QUERY_BY_ID = {query.query_id: query for query in QUERIES}
EPISODE_BY_ID = {query.query_id: query.episode_id for query in QUERIES}
EPISODES = tuple(sorted({target.episode_id for target in TARGETS}))


def _condition_for_episode(episode):
    conditions = {
        CONDITION_BY_ID[query_id]
        for query_id, query in QUERY_BY_ID.items()
        if query.episode_id == episode
    }
    if len(conditions) != 1:
        raise ValueError(f"Episode {episode!r} has inconsistent conditions.")
    return conditions.pop()


FOLDS = {
    fold: tuple(
        episode for condition in CONDITIONS
        for index, episode in enumerate(
            sorted(
                episode
                for episode in EPISODES
                if _condition_for_episode(episode) == condition
            )
        )
        if index % 2 == (0 if fold == "A" else 1)
    )
    for fold in ("A", "B")
}


def _predictions_for_episodes(preds_dict, episodes):
    allowed = set(episodes)
    return {
        query_id: prediction
        for query_id, prediction in preds_dict.items()
        if EPISODE_BY_ID.get(query_id) in allowed
    }


def _groups_for_episodes(episodes):
    allowed = set(episodes)
    return {
        query_id: condition
        for query_id, condition in CONDITION_BY_ID.items()
        if EPISODE_BY_ID.get(query_id) in allowed
    }


def score(preds_dict, episodes=None):
    """Return contest points, normalized error ratio, and condition ratios."""
    selected = EPISODES if episodes is None else tuple(episodes)
    targets = tuple(
        target for target in TARGETS if target.episode_id in set(selected)
    )
    predictions = _predictions_for_episodes(preds_dict, selected)
    reference = _predictions_for_episodes(REFERENCE, selected)
    groups = _groups_for_episodes(selected)
    result = contest_score(
        predictions,
        targets,
        groups,
        reference,
        max_points=round_metadata(PUBLIC)["max_points"],
    )
    return {
        "points": result["technical_points"],
        "R": result["mean_loss_ratio"],
        "per_condition_ratio": {
            condition: result["by_condition"][condition]["loss_ratio"]
            for condition in CONDITIONS
        },
    }


def _episode_errors(preds_dict, episodes):
    """Return normalized condition ratios for a chosen episode sample."""
    selected = tuple(episodes)
    targets = tuple(target for target in TARGETS if target.episode_id in set(selected))
    groups = _groups_for_episodes(selected)
    predictions = _predictions_for_episodes(preds_dict, selected)
    reference = _predictions_for_episodes(REFERENCE, selected)
    result = contest_score(predictions, targets, groups, reference,
                           max_points=round_metadata(PUBLIC)["max_points"])
    return {
        condition: result["by_condition"][condition]["loss_ratio"]
        for condition in CONDITIONS
    }


def bootstrap_ci(preds_dict, repeats=1000, seed=314159):
    """Return a percentile 95% CI, resampling all episodes per condition."""
    by_condition = {
        condition: tuple(
            sorted({
                target.episode_id for target in TARGETS
                if CONDITION_BY_ID[target.query_id] == condition
            })
        )
        for condition in CONDITIONS
    }
    episode_errors = {}
    for condition, episodes in by_condition.items():
        for episode in episodes:
            ids = [
                query_id for query_id, query_episode in EPISODE_BY_ID.items()
                if query_episode == episode
            ]
            episode_errors[(condition, episode)] = (
                sum(distribution_error(preds_dict[query_id], TARGET_BY_ID[query_id].probabilities)
                    for query_id in ids) / len(ids),
                sum(distribution_error(REFERENCE[query_id], TARGET_BY_ID[query_id].probabilities)
                    for query_id in ids) / len(ids),
            )
    rng = random.Random(seed)
    points = []
    for _ in range(repeats):
        ratios = {}
        for condition, episodes in by_condition.items():
            sampled = [rng.choice(episodes) for _ in episodes]
            prediction_error = sum(
                episode_errors[(condition, episode)][0] for episode in sampled
            ) / len(sampled)
            reference_error = sum(
                episode_errors[(condition, episode)][1] for episode in sampled
            ) / len(sampled)
            ratios[condition] = prediction_error / max(reference_error, 1e-9)
        ratio = sum(ratios.values()) / len(CONDITIONS)
        points.append(60.0 / (1.0 + ratio))
    points.sort()
    return (points[int(0.025 * (len(points) - 1))],
            points[int(0.975 * (len(points) - 1))])


def pooled_anchor():
    """Fit and return the frozen pooled starter's validation predictions."""
    train_queries, train_history = load_inputs(PUBLIC / "train")
    labels = read_labels(PUBLIC / "train" / "labels.csv", train_queries)
    model = FrequencyForecaster("pooled").fit(train_queries, train_history, labels)
    return model.predict(QUERIES, load_inputs(PUBLIC / "validation")[1])
