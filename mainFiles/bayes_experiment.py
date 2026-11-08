"""Empirical-Bayes latent-state forecasts and the required validation audit.

The model has a Gaussian population prior over machine-specific logistic
coefficients.  Each machine gets a Laplace posterior, while payout size is a
hierarchical Dirichlet posterior conditional on a win.  Static and
exponentially weighted states are compared with a Laplace pseudo-evidence
score; the mixture variant uses those scores as its state posterior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.contest import contest_score
from slot_forecasting.io import (
    load_inputs,
    read_labels,
    read_rows,
    read_submission,
    read_targets,
    write_json,
    write_submission,
)
from slot_forecasting.schema import PAYOUTS, CONDITIONS
from slot_forecasting.scoring import distribution_error


LAGS = tuple(range(1, 9))
STATE_HALF_LIVES = (None, 24.0)
FEATURE_NAMES = (
    "intercept",
    "type_steady",
    "type_balanced",
    "type_jackpot",
    "signal_neg",
    "signal_zero",
    "signal_pos",
    *(f"lag_{lag}_{part}" for lag in LAGS for part in ("event", "amount", "missing")),
    "peer_last_event",
    "peer_recent_event",
    "peer_long_event",
    "peer_recent_missing",
)


def sigmoid_logit(value: float) -> float:
    return float(expit(np.clip(value, -35.0, 35.0)))


def logdet(matrix: np.ndarray) -> float:
    sign, value = np.linalg.slogdet(matrix)
    if sign <= 0 or not np.isfinite(value):
        raise ValueError("Laplace Hessian is not positive definite")
    return float(value)


class DataView:
    def __init__(self, queries, history):
        self.queries = tuple(queries)
        self.history = tuple(history)
        self.query_by_machine = {
            (q.episode_id, q.machine_id): q for q in self.queries
        }
        self.sequence: dict[tuple[str, str], dict[int, int]] = defaultdict(dict)
        for play in self.history:
            self.sequence[(play.episode_id, play.machine_id)][play.tick] = play.payout
        self.episode_machines: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for key in self.query_by_machine:
            self.episode_machines[key[0]].append(key)

    def feature(self, key, tick: int, *, shared: bool) -> np.ndarray:
        query = self.query_by_machine[key]
        sequence = self.sequence[key]
        values = [
            1.0,
            float(query.machine_type == "steady"),
            float(query.machine_type == "balanced"),
            float(query.machine_type == "jackpot"),
            float(query.signal == -1),
            float(query.signal == 0),
            float(query.signal == 1),
        ]
        for lag in LAGS:
            payout = sequence.get(tick - lag)
            values.extend(
                (
                    float(payout is not None and payout > 0),
                    0.0 if payout is None else math.log1p(payout) / math.log1p(20),
                    float(payout is None),
                )
            )
        if shared:
            peers = [peer for peer in self.episode_machines[key[0]] if peer != key]
            rates = []
            for window in (1, 4, 8):
                observations = [
                    value
                    for peer in peers
                    for tick_offset in range(1, window + 1)
                    if (value := self.sequence[peer].get(tick - tick_offset)) is not None
                ]
                rates.append(
                    (sum(value > 0 for value in observations) / len(observations))
                    if observations
                    else 0.0
                )
            missing = float(
                not any(
                    self.sequence[peer].get(tick - offset) is not None
                    for peer in peers
                    for offset in range(1, 5)
                )
            )
            values.extend((*rates, missing))
        return np.asarray(values, dtype=float)


def objective(beta, x, y, weights, prior_mean, prior_precision):
    eta = x @ beta
    p = expit(np.clip(eta, -35.0, 35.0))
    residual = y - p
    log_likelihood = np.sum(weights * (y * np.log(np.maximum(p, 1e-15))
                                        + (1 - y) * np.log(np.maximum(1 - p, 1e-15))))
    delta = beta - prior_mean
    return float(log_likelihood - 0.5 * np.sum(prior_precision * delta * delta))


def laplace_map(
    x,
    y,
    weights,
    prior_mean,
    prior_variance,
    *,
    initial=None,
    iterations=16,
):
    dimension = x.shape[1]
    prior_precision = 1.0 / np.asarray(prior_variance, dtype=float)
    precision_matrix = np.diag(prior_precision)
    beta = np.asarray(prior_mean if initial is None else initial, dtype=float).copy()
    current = objective(beta, x, y, weights, prior_mean, prior_precision)
    for _ in range(iterations):
        p = expit(np.clip(x @ beta, -35.0, 35.0))
        gradient = x.T @ (weights * (y - p)) - prior_precision * (beta - prior_mean)
        hessian = (x.T * (weights * p * (1 - p))) @ x + precision_matrix
        step = np.linalg.solve(hessian, gradient)
        if np.linalg.norm(step, ord=np.inf) < 1e-6:
            break
        scale = 1.0
        while scale > 1 / 128:
            candidate = beta + scale * step
            value = objective(candidate, x, y, weights, prior_mean, prior_precision)
            if value >= current:
                beta, current = candidate, value
                break
            scale *= 0.5
        else:
            break
    p = expit(np.clip(x @ beta, -35.0, 35.0))
    hessian = (x.T * (weights * p * (1 - p))) @ x + precision_matrix
    covariance = np.linalg.inv(hessian)
    evidence = (
        current
        - 0.5 * logdet(hessian)
        + 0.5 * float(np.sum(np.log(prior_precision)))
    )
    return beta, covariance, float(evidence)


def weighted_age(tick: int, cutoff: int, half_life: float | None) -> float:
    if half_life is None:
        return 1.0
    age = max(0, cutoff - 1 - tick)
    return 2.0 ** (-age / half_life)


class BayesianLatentModel:
    def __init__(self, *, half_life: float | None, shared: bool, drop_missing: bool):
        self.half_life = half_life
        self.shared = shared
        self.drop_missing = drop_missing
        self.mean = None
        self.variance = None
        self.size_prior = None
        self.train_evidence = None
        self.machine_posteriors = {}

    def _examples(self, data: DataView, labels=None):
        rows = []
        for key, sequence in data.sequence.items():
            query = data.query_by_machine[key]
            for tick, payout in sorted(sequence.items()):
                rows.append(
                    (key, tick, int(payout > 0), payout, query.cutoff)
                )
            if labels is not None:
                rows.append(
                    (key, query.cutoff, int(labels[query.query_id] > 0),
                     labels[query.query_id], query.cutoff)
                )
        x, y, outcomes, groups, weights = [], [], [], [], []
        for key, tick, target, payout, cutoff in rows:
            vector = data.feature(key, tick, shared=self.shared)
            if self.drop_missing and any(
                vector[7 + (index * 3) + 2] for index in range(len(LAGS))
            ):
                continue
            x.append(vector)
            y.append(target)
            outcomes.append(payout)
            groups.append(key)
            weights.append(weighted_age(tick, cutoff, self.half_life))
        return (
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            outcomes,
            groups,
            np.asarray(weights, dtype=float),
        )

    def _learn_size_prior(self, outcomes, groups):
        counts_by_machine = defaultdict(lambda: np.zeros(4, dtype=float))
        for key, payout in zip(groups, outcomes):
            if payout > 0:
                counts_by_machine[key][PAYOUTS.index(payout) - 1] += 1
        candidates = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
        best = None
        best_score = -math.inf
        for concentration in candidates:
            score = 0.0
            for counts in counts_by_machine.values():
                total = float(counts.sum())
                alpha = np.full(4, concentration / 4)
                score += math.lgamma(concentration) - math.lgamma(concentration + total)
                score += sum(
                    math.lgamma(a + value) - math.lgamma(a)
                    for a, value in zip(alpha, counts)
                )
            if score > best_score:
                best_score, best = score, concentration
        total_counts = np.zeros(4, dtype=float)
        for counts in counts_by_machine.values():
            total_counts += counts
        self.size_prior = np.full(4, best / 4) + total_counts / max(
            1, len(counts_by_machine)
        )
        return float(best_score), float(best)

    def fit(self, data: DataView, labels):
        x, y, outcomes, groups, weights = self._examples(data, labels)
        dimension = x.shape[1]
        initial_variance = np.full(dimension, 1.0)
        prior_mean = np.zeros(dimension)
        global_beta, _, _ = laplace_map(
            x, y, weights, prior_mean, initial_variance, iterations=20
        )
        local_betas = []
        local_prior_variance = np.full(dimension, 1.0)
        grouped = defaultdict(list)
        for index, key in enumerate(groups):
            grouped[key].append(index)
        for indices in grouped.values():
            local_x, local_y, local_w = x[indices], y[indices], weights[indices]
            local_beta, _, _ = laplace_map(
                local_x,
                local_y,
                local_w,
                global_beta,
                local_prior_variance,
                initial=global_beta,
                iterations=10,
            )
            local_betas.append(local_beta)
        if len(local_betas) > 1:
            local_variance = np.var(np.asarray(local_betas), axis=0, ddof=1)
            local_prior_variance = np.clip(local_variance, 0.03, 2.0)
        self.mean, self.variance, self.train_evidence = global_beta, local_prior_variance, 0.0
        self.machine_posteriors = {}
        evidence = 0.0
        for key, indices in grouped.items():
            beta, covariance, local_evidence = laplace_map(
                x[indices],
                y[indices],
                weights[indices],
                self.mean,
                self.variance,
                initial=self.mean,
                iterations=16,
            )
            self.machine_posteriors[key] = (beta, covariance)
            evidence += local_evidence
        size_evidence, concentration = self._learn_size_prior(outcomes, groups)
        self.train_evidence = float(evidence + size_evidence)
        self.size_concentration = concentration
        return self

    def _predict_machine(self, data: DataView, query):
        key = (query.episode_id, query.machine_id)
        sequence = data.sequence[key]
        x = data.feature(key, query.cutoff, shared=self.shared)
        rows = [
            (tick, payout)
            for tick, payout in sorted(sequence.items())
            if tick < query.cutoff
        ]
        local_x = np.asarray(
            [data.feature(key, tick, shared=self.shared) for tick, _ in rows],
            dtype=float,
        )
        local_y = np.asarray([float(payout > 0) for _, payout in rows])
        local_w = np.asarray(
            [weighted_age(tick, query.cutoff, self.half_life) for tick, _ in rows]
        )
        beta, covariance, evidence = laplace_map(
            local_x,
            local_y,
            local_w,
            self.mean,
            self.variance,
            initial=self.mean,
            iterations=16,
        )
        eta = float(x @ beta)
        variance_eta = float(x @ covariance @ x)
        p_win = sigmoid_logit(eta / math.sqrt(1 + math.pi * variance_eta / 8))
        counts = np.asarray(self.size_prior, dtype=float).copy()
        for tick, payout in rows:
            if payout > 0:
                counts[PAYOUTS.index(payout) - 1] += weighted_age(
                    tick, query.cutoff, self.half_life
                )
        size = counts / counts.sum()
        return np.asarray((1 - p_win, *(p_win * size))), float(evidence)

    def predict(self, data: DataView):
        predictions = {}
        evidences = {}
        for query in data.queries:
            prediction, evidence = self._predict_machine(data, query)
            predictions[query.query_id] = tuple(prediction / prediction.sum())
            evidences[query.query_id] = evidence
        return predictions, evidences


def fit_models(train_data, labels):
    models = {}
    settings = (
        ("static", None, False, False),
        ("recent24", 24.0, False, False),
        ("recent24_shared", 24.0, True, False),
        ("mixture_shared", None, True, False),
        ("static_drop_missing", None, False, True),
    )
    for name, half_life, shared, drop_missing in settings:
        if name == "mixture_shared":
            static = BayesianLatentModel(
                half_life=None, shared=True, drop_missing=drop_missing
            ).fit(train_data, labels)
            recent = BayesianLatentModel(
                half_life=24.0, shared=True, drop_missing=drop_missing
            ).fit(train_data, labels)
            models[name] = (static, recent)
        else:
            models[name] = BayesianLatentModel(
                half_life=half_life, shared=shared, drop_missing=drop_missing
            ).fit(train_data, labels)
    return models


def predict_models(models, data):
    result = {}
    evidence = {}
    for name, model in models.items():
        if isinstance(model, tuple):
            static_predictions, static_evidence = model[0].predict(data)
            recent_predictions, recent_evidence = model[1].predict(data)
            predictions = {}
            for query in data.queries:
                key = query.query_id
                scores = np.asarray((static_evidence[key], recent_evidence[key]))
                scores -= scores.max()
                weights = np.exp(scores)
                weights /= weights.sum()
                predictions[key] = tuple(
                    weights[0] * np.asarray(static_predictions[key])
                    + weights[1] * np.asarray(recent_predictions[key])
                )
            result[name] = predictions
            evidence[name] = {
                "static": model[0].train_evidence,
                "recent24": model[1].train_evidence,
            }
        else:
            result[name], _ = model.predict(data)
            evidence[name] = model.train_evidence
    return result, evidence


def episode_folds(targets, groups):
    by_condition = defaultdict(set)
    for target in targets:
        by_condition[groups[target.query_id]].add(target.episode_id)
    fold_a, fold_b = set(), set()
    for condition in CONDITIONS:
        ordered = sorted(by_condition[condition])
        fold_a.update(ordered[::2])
        fold_b.update(ordered[1::2])
    return fold_a, fold_b


def score_subset(predictions, targets, groups, reference, episodes):
    subset = [target for target in targets if target.episode_id in episodes]
    ids = {target.query_id for target in subset}
    return contest_score(
        {key: predictions[key] for key in ids},
        subset,
        {key: groups[key] for key in ids},
        {key: reference[key] for key in ids},
        max_points=60,
        version="temporal-adaptation-v3-60-40",
    )


def bootstrap_points(predictions, targets, groups, reference, repeats=1000):
    episodes = defaultdict(list)
    for target in targets:
        episodes[(groups[target.query_id], target.episode_id)].append(target)
    rng = np.random.default_rng(314159)
    values = []
    for _ in range(repeats):
        ratios = []
        for condition in CONDITIONS:
            condition_episodes = [
                episode for (name, episode) in episodes if name == condition
            ]
            sampled = rng.choice(condition_episodes, len(condition_episodes), replace=True)
            losses, baseline = [], []
            for episode in sampled:
                rows = episodes[(condition, episode)]
                losses.append(
                    np.mean([
                        distribution_error(predictions[t.query_id], t.probabilities)
                        for t in rows
                    ])
                )
                baseline.append(
                    np.mean([
                        distribution_error(reference[t.query_id], t.probabilities)
                        for t in rows
                    ])
                )
            ratios.append(np.mean(losses) / max(np.mean(baseline), 1e-9))
        values.append(60 / (1 + float(np.mean(ratios))))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="mainFiles/public")
    parser.add_argument("--out", default="mainFiles/runs/bayes-experiment")
    args = parser.parse_args()
    public = Path(args.data)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"Output already exists: {out}")
    train_q, train_h = load_inputs(public / "train")
    validation_q, validation_h = load_inputs(public / "validation")
    train_data = DataView(train_q, train_h)
    validation_data = DataView(validation_q, validation_h)
    labels = read_labels(public / "train/labels.csv", train_q)
    targets = read_targets(public / "validation/targets.csv")
    groups = {
        row["query_id"]: row["condition"]
        for row in read_rows(public / "validation/conditions.csv", ("query_id", "condition"))
    }
    reference = read_submission(public / "validation/reference.csv")
    models = fit_models(train_data, labels)
    predictions, evidence = predict_models(models, validation_data)
    fold_a, fold_b = episode_folds(targets, groups)
    comparison = {}
    for name, forecast in predictions.items():
        full = contest_score(
            forecast, targets, groups, reference, max_points=60,
            version="temporal-adaptation-v3-60-40",
        )
        a = score_subset(forecast, targets, groups, reference, fold_a)
        b = score_subset(forecast, targets, groups, reference, fold_b)
        comparison[name] = {
            "train_laplace_pseudo_log_evidence": evidence[name],
            "full_validation": full,
            "fold_a_points": a["technical_points"],
            "fold_b_points": b["technical_points"],
            "cross_fitted_mean_points": (a["technical_points"] + b["technical_points"]) / 2,
            "bootstrap_ci95_points": bootstrap_points(
                forecast, targets, groups, reference
            ),
        }
    def evidence_value(name):
        value = comparison[name]["train_laplace_pseudo_log_evidence"]
        return sum(value.values()) if isinstance(value, dict) else value

    by_train_evidence = max(comparison, key=evidence_value)
    chosen_for_audit_b = max(
        comparison, key=lambda name: comparison[name]["fold_a_points"]
    )
    chosen_for_audit_a = max(
        comparison, key=lambda name: comparison[name]["fold_b_points"]
    )
    cross_fitted_predictions = {
        target.query_id: predictions[
            chosen_for_audit_a if target.episode_id in fold_a else chosen_for_audit_b
        ][target.query_id]
        for target in targets
    }
    cross_fitted_score = contest_score(
        cross_fitted_predictions, targets, groups, reference, max_points=60,
        version="temporal-adaptation-v3-60-40",
    )
    cross_fitted_ci = bootstrap_points(
        cross_fitted_predictions, targets, groups, reference
    )
    pooled = FrequencyForecaster("pooled").fit(
        train_q, train_h, labels
    ).predict(validation_q, validation_h)
    pooled_full = contest_score(
        pooled, targets, groups, reference, max_points=60,
        version="temporal-adaptation-v3-60-40",
    )
    pooled_a = score_subset(pooled, targets, groups, reference, fold_a)
    pooled_b = score_subset(pooled, targets, groups, reference, fold_b)
    pooled_ci = bootstrap_points(pooled, targets, groups, reference)
    test_q, test_h = load_inputs(public / "test")
    test_predictions, _ = predict_models(models, DataView(test_q, test_h))
    selected_test = test_predictions[by_train_evidence]
    out.mkdir(parents=True)
    write_submission(out / "submission.csv", selected_test)
    write_submission(out / "validation_predictions.csv", cross_fitted_predictions)
    report = {
        "model": "hierarchical Bayesian logistic latent win state plus Dirichlet size state",
        "settings": {
            "lags": list(LAGS),
            "states": ["static", "recent24"],
            "posterior": "Laplace logistic-normal approximation",
            "missing_ticks": "explicit missing indicators; static_drop_missing is the comparison",
            "shared_change": "peer event rates at the previous 1, 4, and 8 ticks",
            "size_prior_concentration_grid": [1, 2, 5, 10, 20, 50],
            "size_prior_concentration": {
                name: (
                    model[0].size_concentration
                    if isinstance(model, tuple)
                    else model.size_concentration
                )
                for name, model in models.items()
            },
            "seed": 314159,
        },
        "folds": {"fold_a_episodes": sorted(fold_a), "fold_b_episodes": sorted(fold_b)},
        "selection": {
            "train_evidence": by_train_evidence,
            "cross_fitted": {
                "tune_fold_a_report_fold_b": chosen_for_audit_b,
                "tune_fold_b_report_fold_a": chosen_for_audit_a,
                "points": cross_fitted_score["technical_points"],
                "bootstrap_ci95_points": cross_fitted_ci,
                "by_condition": cross_fitted_score["by_condition"],
            },
        },
        "pooled_anchor": {
            "full_validation": pooled_full,
            "fold_a_points": pooled_a["technical_points"],
            "fold_b_points": pooled_b["technical_points"],
            "bootstrap_ci95_points": pooled_ci,
        },
        "comparison": comparison,
    }
    write_json(out / "bayes_report.json", report)
    print(json.dumps({
        name: {
            "train_evidence": row["train_laplace_pseudo_log_evidence"],
            "cross_fitted_points": row["cross_fitted_mean_points"],
            "full_points": row["full_validation"]["technical_points"],
            "ci95": row["bootstrap_ci95_points"],
        }
        for name, row in comparison.items()
    }, indent=2))
    print("train-evidence selection:", by_train_evidence)
    print(
        "cross-fitted selection:",
        chosen_for_audit_b,
        "on audit B;",
        chosen_for_audit_a,
        "on audit A;",
        "points",
        cross_fitted_score["technical_points"],
    )
    print("pooled anchor:", pooled_full["technical_points"])
    print("submission:", out / "submission.csv")


if __name__ == "__main__":
    main()
