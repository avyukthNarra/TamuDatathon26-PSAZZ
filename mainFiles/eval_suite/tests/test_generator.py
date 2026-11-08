"""Method-agnostic contract and causality tests for synthetic datasets."""

import hashlib
import re
import random
import statistics
import subprocess
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from eval_suite import scoring
from eval_suite.datachecks import check_dataset
from eval_suite.generator import (
    PRESETS, GeneratorConfig, LagInteraction, MachineParameters,
    conditional_probabilities, generate_dataset, preset_config, simulate_episode,
)
from eval_suite.calibration import compare_statistics, summarize_dataset
from slot_forecasting.io import (
    load_inputs,
    read_labels,
    read_rows,
    read_submission,
    read_targets,
    write_submission,
)

KIT = Path(__file__).resolve().parents[2]
EXPECTED_PRESETS = {
    "main_like", "round2_shift", "all_lags", "neighbour_heavy",
    "hidden_state_heavy", "fast_drift", "edge_cases",
}
CONDITIONS = ("stable", "short_history", "drift", "shared_change")


def _directory_hashes(root):
    root = Path(root)
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _point_from_checker(path, data_root):
    completed = subprocess.run(
        [sys.executable, str(KIT / "check_submission.py"), str(path),
         "--data", str(data_root), "--split", "validation"],
        cwd=KIT, capture_output=True, text=True, check=True,
    )
    match = re.search(r"Validation points: ([0-9.]+) / (\d+)", completed.stdout)
    if match is None:
        raise AssertionError(f"checker did not print validation points:\n{completed.stdout}")
    return float(match.group(1)), int(match.group(2))


class GeneratedDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory(prefix="generator-suite-")
        cls.tmp = Path(cls._temporary.name)
        cls.datasets = {}
        for name in sorted(EXPECTED_PRESETS):
            root = cls.tmp / name
            cls.datasets[name] = generate_dataset(
                root, preset=name, seed=2407,
                episodes_per_condition=2 if name == "edge_cases" else 1,
            )
        cls.replay_root = cls.tmp / "main_like-replay"
        generate_dataset(cls.replay_root, preset="main_like", seed=2407,
                         episodes_per_condition=1)

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def test_all_seven_presets_load_through_kit_io_and_keep_test_truth_private(self):
        self.assertEqual(set(PRESETS), EXPECTED_PRESETS)
        for name, root in self.datasets.items():
            with self.subTest(preset=name):
                public = Path(root)
                failed_checks = [result for result in check_dataset(public) if not result.passed
                                 and result.severity == "fail"]
                self.assertEqual(failed_checks, [], msg=f"{name}: {failed_checks}")
                for split_name in ("train", "validation", "test"):
                    queries, history = load_inputs(public / split_name)
                    self.assertEqual(len(queries), 64 if name == "edge_cases" else 32)
                    self.assertGreater(len(history), 0)
                train_q, _ = load_inputs(public / "train")
                validation_q, _ = load_inputs(public / "validation")
                labels = read_labels(public / "train" / "labels.csv", train_q)
                val_labels = read_labels(public / "validation" / "labels.csv", validation_q)
                val_targets = read_targets(public / "validation" / "targets.csv")
                self.assertEqual(set(labels), {q.query_id for q in train_q})
                self.assertEqual(set(val_labels), {q.query_id for q in validation_q})
                self.assertEqual({t.query_id for t in val_targets}, {q.query_id for q in validation_q})
                test_q_for_ids, _ = load_inputs(public / "test")
                self.assertTrue({q.query_id for q in train_q}.isdisjoint(q.query_id for q in validation_q))
                self.assertTrue({q.query_id for q in train_q}.isdisjoint(q.query_id for q in test_q_for_ids))
                self.assertTrue({q.query_id for q in validation_q}.isdisjoint(q.query_id for q in test_q_for_ids))

                test_dir = public / "test"
                for answer_name in ("targets.csv", "labels.csv", "conditions.csv", "reference.csv"):
                    self.assertFalse((test_dir / answer_name).exists(), answer_name)
                private_test = public / "_truth" / "test"
                test_q, _ = load_inputs(test_dir)
                truth = read_targets(private_test / "targets.csv")
                conditions = read_rows(private_test / "conditions.csv", ("query_id", "condition"))
                reference = read_submission(private_test / "reference.csv")
                ids = {query.query_id for query in test_q}
                self.assertEqual({target.query_id for target in truth}, ids)
                self.assertEqual({row["query_id"] for row in conditions}, ids)
                self.assertEqual(set(reference), ids)

    def test_all_splits_have_eight_machines_and_equal_condition_counts(self):
        for name, root in self.datasets.items():
            public = Path(root)
            for split in ("train", "validation", "test"):
                with self.subTest(preset=name, split=split):
                    queries, _ = load_inputs(public / split)
                    if split == "test":
                        condition_rows = read_rows(public / "_truth" / "test" / "conditions.csv",
                                                   ("query_id", "condition"))
                    else:
                        condition_rows = read_rows(public / split / "conditions.csv",
                                                   ("query_id", "condition"))
                    query_conditions = {row["query_id"]: row["condition"] for row in condition_rows}
                    self.assertEqual(set(query_conditions.values()), set(CONDITIONS))
                    per_condition = Counter(query_conditions.values())
                    self.assertEqual(set(per_condition.values()), {16 if name == "edge_cases" else 8})
                    per_episode = defaultdict(set)
                    for query in queries:
                        per_episode[query.episode_id].add(query.machine_id)
                    self.assertTrue(per_episode)
                    self.assertTrue(all(len(machines) == 8 for machines in per_episode.values()))

    def test_determinism_is_byte_for_byte_for_same_seed(self):
        self.assertEqual(_directory_hashes(self.datasets["main_like"]),
                         _directory_hashes(self.replay_root))

    def test_main_like_short_histories_are_exactly_64_observations(self):
        public = Path(self.datasets["main_like"])
        queries, history = load_inputs(public / "train")
        rows = read_rows(public / "train" / "conditions.csv", ("query_id", "condition"))
        conditions = {row["query_id"]: row["condition"] for row in rows}
        plays = defaultdict(set)
        for play in history:
            plays[(play.episode_id, play.machine_id)].add(play.tick)
        short = [query for query in queries if conditions[query.query_id] == "short_history"]
        self.assertEqual(len(short), 8)
        for query in short:
            ticks = plays[(query.episode_id, query.machine_id)]
            self.assertEqual(len(ticks), 64)
            self.assertEqual(ticks & set(range(query.cutoff - 40, query.cutoff)),
                             set(range(query.cutoff - 40, query.cutoff)))
            older = ticks - set(range(query.cutoff - 40, query.cutoff))
            self.assertEqual(len(older), 24)
            self.assertTrue(all(tick < query.cutoff - 40 for tick in older))
            self.assertTrue(all(tick < query.cutoff for tick in ticks))

    def test_main_like_truth_and_frozen_strength_20_reference_scores(self):
        public = Path(self.datasets["main_like"])
        validation = public / "validation"
        targets = read_targets(validation / "targets.csv")
        truth = {target.query_id: target.probabilities for target in targets}
        reference = read_submission(validation / "reference.csv")
        data = (validation / "targets.csv", validation / "queries.csv",
                validation / "conditions.csv", validation / "reference.csv")
        exact = scoring.official_score(truth, *data, max_points=60)
        baseline = scoring.official_score(reference, *data, max_points=60)
        self.assertAlmostEqual(exact["points"], 60.0, delta=1e-10)
        self.assertAlmostEqual(baseline["points"], 30.0, delta=1e-9)

        regenerated = scoring.reference_from_starter(public, split="validation", prior_strength=20.0)
        self.assertEqual(set(regenerated), set(reference))
        for query_id in reference:
            for actual, expected in zip(reference[query_id], regenerated[query_id]):
                self.assertAlmostEqual(actual, expected, delta=1e-12, msg=query_id)

        with tempfile.TemporaryDirectory(prefix="generator-checker-") as tmp:
            truth_csv = Path(tmp) / "truth.csv"
            write_submission(truth_csv, truth)
            true_points, true_max = _point_from_checker(truth_csv, public)
            ref_points, ref_max = _point_from_checker(validation / "reference.csv", public)
        self.assertEqual(true_max, 60)
        self.assertEqual(ref_max, 60)
        self.assertAlmostEqual(true_points, 60.0, delta=1e-6)
        self.assertAlmostEqual(ref_points, 30.0, delta=1e-6)

        adaptation = Path(self.datasets["round2_shift"])
        adaptation_validation = adaptation / "validation"
        adaptation_targets = read_targets(adaptation_validation / "targets.csv")
        adaptation_truth = {target.query_id: target.probabilities for target in adaptation_targets}
        adaptation_reference = read_submission(adaptation_validation / "reference.csv")
        adaptation_data = (adaptation_validation / "targets.csv",
                           adaptation_validation / "queries.csv",
                           adaptation_validation / "conditions.csv",
                           adaptation_validation / "reference.csv")
        self.assertAlmostEqual(scoring.official_score(adaptation_truth, *adaptation_data,
                                                      max_points=40)["points"], 40.0, delta=1e-10)
        self.assertAlmostEqual(scoring.official_score(adaptation_reference, *adaptation_data,
                                                      max_points=40)["points"], 20.0, delta=1e-9)
        with tempfile.TemporaryDirectory(prefix="adaptation-checker-") as tmp:
            truth_csv = Path(tmp) / "truth.csv"
            write_submission(truth_csv, adaptation_truth)
            true_points, true_max = _point_from_checker(truth_csv, adaptation)
            ref_points, ref_max = _point_from_checker(adaptation_validation / "reference.csv", adaptation)
        self.assertEqual((true_max, ref_max), (40, 40))
        self.assertAlmostEqual(true_points, 40.0, delta=1e-6)
        self.assertAlmostEqual(ref_points, 20.0, delta=1e-6)

    def test_preset_configurations_encode_documented_shifts(self):
        configs = {name: preset_config(name) for name in EXPECTED_PRESETS}
        main = configs["main_like"]
        self.assertNotEqual(configs["round2_shift"].recurring_lags, main.recurring_lags)
        self.assertEqual(set(configs["all_lags"].recurring_lags), set(range(1, 9)))
        self.assertGreater(configs["neighbour_heavy"].neighbour_strength, main.neighbour_strength)
        self.assertGreater(configs["hidden_state_heavy"].hidden_state_strength,
                           main.hidden_state_strength)
        self.assertLess(configs["fast_drift"].drift_duration, main.drift_duration)

    def test_stress_presets_activate_the_documented_mechanisms(self):
        rng = random.Random(931)
        for name in ("round2_shift", "all_lags", "neighbour_heavy", "hidden_state_heavy", "fast_drift"):
            with self.subTest(preset=name):
                config = preset_config(name)
                episode = simulate_episode(f"stress-{name}", "stable", rng, config,
                                           force_all_lags=name == "all_lags")
                active_lags = {lag for params in episode.parameters for interaction in params.interactions
                               for lag in interaction.lags}
                if name == "round2_shift":
                    self.assertTrue(active_lags)
                    self.assertTrue(active_lags <= set(config.recurring_lags))
                    self.assertNotEqual(config.recurring_lags, preset_config("main_like").recurring_lags)
                elif name == "all_lags":
                    self.assertEqual(active_lags, set(range(1, 9)))
                elif name == "neighbour_heavy":
                    self.assertTrue(all(p.neighbour_offset is not None for p in episode.parameters))
                elif name == "hidden_state_heavy":
                    self.assertGreater(config.hidden_state_strength,
                                       preset_config("main_like").hidden_state_strength)
                    low = (0.0,) * 8
                    high = (1.0,) * 8
                    kwargs = dict(prefix={}, machine_types=episode.machine_types,
                                  signals=episode.signals, params=episode.parameters, tick=10,
                                  config=config)
                    low_vectors = conditional_probabilities(latent_state=low, **kwargs)
                    high_vectors = conditional_probabilities(latent_state=high, **kwargs)
                    self.assertTrue(any(a != b for a, b in zip(low_vectors, high_vectors)))
                elif name == "fast_drift":
                    self.assertLess(config.drift_duration, preset_config("main_like").drift_duration)
                    self.assertTrue(all(p.drift_duration == config.drift_duration for p in episode.parameters))

    def test_single_available_lag_generates_without_replacement_errors(self):
        config = preset_config("main_like", {"recurring_lags": (1,)})
        episode = simulate_episode("single-lag", "stable", random.Random(101), config)
        lags = {lag for machine in episode.parameters for interaction in machine.interactions
                for lag in interaction.lags}
        self.assertEqual(lags, {1})

    def test_edge_cases_cover_cutoffs_and_probability_support(self):
        public = Path(self.datasets["edge_cases"])
        for split in ("train", "validation", "test"):
            queries, _ = load_inputs(public / split)
            self.assertIn(192, {query.cutoff for query in queries})
            self.assertIn(256, {query.cutoff for query in queries})
        edge_episode = simulate_episode("edge-fixture", "stable", random.Random(18),
                                        preset_config("edge_cases"), edge_index=0)
        all_loss = next(i for i, params in enumerate(edge_episode.parameters) if params.all_loss)
        rare_jackpot = next(i for i, params in enumerate(edge_episode.parameters) if params.rare_jackpot)
        self.assertEqual(edge_episode.machine_types[rare_jackpot], "jackpot")
        target_by_machine = {target.query_id.rsplit(":", 1)[-1]: target for target in edge_episode.targets}
        all_loss_target = target_by_machine[f"m{all_loss:02d}"]
        rare_target = target_by_machine[f"m{rare_jackpot:02d}"]
        self.assertEqual(all_loss_target.probabilities, (1.0, 0.0, 0.0, 0.0, 0.0))
        self.assertGreater(rare_target.probabilities[4], 0.0)
        self.assertGreater(rare_target.probabilities[4],
                           rare_target.probabilities[1] * 2.0)

    def test_shared_change_is_one_episode_changepoint(self):
        result = simulate_episode("shared-fixture", "shared_change", random.Random(38),
                                  preset_config("main_like"))
        self.assertIsNotNone(result.shared_change_tick)
        self.assertGreater(result.shared_change_tick, 0)
        self.assertLess(result.shared_change_tick, result.cutoff)
        self.assertEqual(len(result.parameters), 8)
        unchanged = simulate_episode("stable-fixture", "stable", random.Random(38),
                                      preset_config("main_like"))
        self.assertIsNone(unchanged.shared_change_tick)

    def test_conditional_vectors_ignore_current_and_future_peer_outcomes(self):
        params = tuple(MachineParameters(
            interactions=(LagInteraction((1,), "single", 1),),
            neighbour_offset=1, neighbour_lag=1, neighbour_sign=1,
            size_lag=1, size_shift_sign=1, drift_sign=1,
            drift_start=0, drift_duration=64, shared_sign=1,
        ) for _ in range(8))
        types = ("steady", "balanced", "jackpot", "steady", "balanced", "jackpot", "steady", "balanced")
        signals = (-1, 0, 1, -1, 0, 1, -1, 0)
        config = GeneratorConfig()
        baseline = {f"m{i}": {t: (20 if (i + t) % 2 else 0) for t in range(12)} for i in range(8)}
        mutated = {machine: dict(history) for machine, history in baseline.items()}
        # Mutate every machine's current/contemporaneous and later outcomes.
        for history in mutated.values():
            for tick in range(10, 12):
                history[tick] = 20 if history[tick] == 0 else 0
        kwargs = dict(machine_types=types, signals=signals, params=params, tick=10,
                      latent_state=(1.0, -1.0) * 4, config=config,
                      machine_ids=tuple(f"m{i}" for i in range(8)))
        before = conditional_probabilities(baseline, **kwargs)
        after = conditional_probabilities(mutated, **kwargs)
        self.assertEqual(before, after)
        # A genuinely observed lag-1 change can affect the vector, so the test
        # would catch a function that simply ignored all history.
        changed_past = {machine: dict(history) for machine, history in baseline.items()}
        changed_past["m0"][9] = 20 if changed_past["m0"][9] == 0 else 0
        self.assertNotEqual(before, conditional_probabilities(changed_past, **kwargs))

    def test_calibration_counts_observed_contexts_and_rejects_bad_sampling_pattern(self):
        public = Path(self.datasets["main_like"])
        queries, history = load_inputs(public / "train")
        summary = summarize_dataset(public)
        self.assertTrue(summary["history_sampling"]["condition_specific_pattern_exact"])
        query_by_machine = {(q.episode_id, q.machine_id): q for q in queries}
        # Marginals must be computed from every observed history outcome.
        type_signal = defaultdict(list)
        machine_wins = defaultdict(list)
        wins_at = {}
        for play in history:
            query = query_by_machine[(play.episode_id, play.machine_id)]
            type_signal[(query.machine_type, query.signal)].append(play.payout)
            machine_wins[(play.episode_id, play.machine_id)].append(play.payout > 0)
            wins_at[(play.episode_id, play.machine_id, play.tick)] = play.payout > 0
        for (machine_type, signal), outcomes in type_signal.items():
            cell = summary["type_signal_win_rates"][machine_type][str(signal)]
            self.assertEqual(cell["n"], len(outcomes))
            self.assertEqual(cell["wins"], sum(outcome > 0 for outcome in outcomes))
            self.assertAlmostEqual(cell["rate"], sum(outcome > 0 for outcome in outcomes) / len(outcomes))
        for machine_type in {q.machine_type for q in queries}:
            positive = [p.payout for p in history
                        if query_by_machine[(p.episode_id, p.machine_id)].machine_type == machine_type
                        and p.payout > 0]
            mix = summary["positive_payout_mix_by_type"][machine_type]
            self.assertEqual(mix["n_positive"], len(positive))
            for payout in (1, 2, 5, 20):
                self.assertEqual(mix["counts"][str(payout)], positive.count(payout))
        machine_rates = [sum(values) / len(values) for values in machine_wins.values()]
        self.assertAlmostEqual(summary["machine_win_rate_spread"]["sd"], statistics.stdev(machine_rates))

        # For each lag, count every pair whose two exact ticks are observed;
        # sparse holes are skipped rather than imputed.
        expected = {lag: 0 for lag in range(1, 9)}
        for episode_id, machine_id in machine_wins:
            ticks = {play.tick for play in history
                     if play.episode_id == episode_id and play.machine_id == machine_id}
            for tick in ticks:
                expected.update({lag: expected[lag] + int(tick - lag in ticks) for lag in range(1, 9)})
        actual = {int(lag): row["n"] for lag, row in
                  summary["win_lag_correlations"]["overall"].items()}
        self.assertEqual(actual, expected)

        malformed = summarize_dataset(public)
        malformed["history_sampling"]["condition_specific_pattern_exact"] = False
        comparison = compare_statistics(summary, malformed)
        self.assertFalse(comparison["metrics"]["sampling_pattern_exact"])
        self.assertFalse(comparison["within_descriptive_bands"]["sampling_pattern_exact"])
        unavailable = summarize_dataset(public)
        unavailable["type_signal_win_rates"]["steady"]["-1"]["rate"] = None
        unavailable_comparison = compare_statistics(summary, unavailable)
        self.assertIsNone(unavailable_comparison["metrics"]["type_signal_win_rate_max_abs"])
        self.assertFalse(unavailable_comparison["within_descriptive_bands"]["type_signal_win_rate_max_abs"])



if __name__ == "__main__":
    unittest.main()
