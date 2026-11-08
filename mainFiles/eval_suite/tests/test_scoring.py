"""Tests of the suite's scoring, not of any team method.

Run from the kit root: python -m unittest discover -s eval_suite/tests -t . -v
"""

import csv
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from eval_suite import scoring as S

KIT = Path(__file__).resolve().parents[2]
PUBLIC = KIT / "public"
VALIDATION = PUBLIC / "validation"
MAX_MAIN = 60


def _write_predictions(path, predictions):
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(S.PREDICTION_COLUMNS)
        for key in sorted(predictions):
            writer.writerow([key, *[repr(float(v)) for v in predictions[key]]])


def _read_reference():
    return S.read_predictions(VALIDATION / "reference.csv")


def _checker(csv_path):
    """Run the kit's check_submission.py exactly as a participant would and parse it."""
    completed = subprocess.run(
        [sys.executable, str(KIT / "check_submission.py"), str(csv_path), "--split", "validation"],
        cwd=KIT, capture_output=True, text=True, check=True)
    output = completed.stdout
    points = float(re.search(r"Validation points: ([0-9.]+) / (\d+)", output).group(1))
    ratios = {name: float(value) for name, value in re.findall(r"^\s+(\w+): loss ratio ([0-9.]+)", output, re.M)}
    return points, ratios


class ScoringTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="scoring-test-")
        cls.targets = S.load_split(VALIDATION).targets
        cls.split = S.load_split(VALIDATION)
        cls.reference = _read_reference()
        cls.uniform = {key: [0.2] * 5 for key in cls.targets}
        cls.truth_csv = Path(cls.tmp) / "truth.csv"
        _write_predictions(cls.truth_csv, cls.targets)
        cls.uniform_csv = Path(cls.tmp) / "uniform.csv"
        _write_predictions(cls.uniform_csv, cls.uniform)
        cls.blend = {k: [0.5 * a + 0.5 * b for a, b in zip(cls.reference[k], cls.uniform[k])]
                     for k in cls.reference}
        cls.blend_csv = Path(cls.tmp) / "blend.csv"
        _write_predictions(cls.blend_csv, cls.blend)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def score(self, prediction, max_points=MAX_MAIN):
        return S.official_score(prediction, VALIDATION / "targets.csv", VALIDATION / "queries.csv",
                                VALIDATION / "conditions.csv", VALIDATION / "reference.csv", max_points)

    def assert_matches_checker(self, csv_path, prediction):
        printed_points, printed_ratios = _checker(csv_path)
        mine = self.score(prediction)
        # The checker prints six decimals, so compare at that precision.
        self.assertLessEqual(abs(mine["points"] - printed_points), 5e-7 + 1e-12)
        for name, ratio in printed_ratios.items():
            self.assertLessEqual(abs(mine["ratio"][name] - ratio), 5e-7 + 1e-12, name)
        # Full precision: the kit's own scorer, which check_submission calls in-process.
        from slot_forecasting.rounds import score_round
        kit = score_round(csv_path, PUBLIC)
        self.assertAlmostEqual(mine["points"], kit["technical_points"], delta=1e-9)
        for name, row in kit["by_condition"].items():
            self.assertAlmostEqual(mine["ratio"][name], row["loss_ratio"], delta=1e-9, msg=name)
        return mine


class OfficialScoreTests(ScoringTestCase):
    def test_points_match_checker_on_reference_file(self):
        self.assert_matches_checker(VALIDATION / "reference.csv", self.reference)

    def test_points_match_checker_on_uniform_file(self):
        mine = self.assert_matches_checker(self.uniform_csv, self.uniform)
        self.assertLess(mine["points"], MAX_MAIN / 2)

    def test_points_match_checker_on_blend_file(self):
        self.assert_matches_checker(self.blend_csv, self.blend)

    def test_truth_as_prediction_gives_max_points(self):
        mine = self.score(self.targets)
        self.assertAlmostEqual(mine["points"], MAX_MAIN, delta=1e-9)
        for ratio in mine["ratio"].values():
            self.assertAlmostEqual(ratio, 0.0, delta=1e-12)

    def test_reference_as_prediction_gives_half_max(self):
        mine = self.score(self.reference)
        self.assertAlmostEqual(mine["points"], MAX_MAIN / 2, delta=1e-9)
        self.assertAlmostEqual(mine["R"], 1.0, delta=1e-9)

    def test_uniform_is_below_half_max(self):
        self.assertLess(self.score(self.uniform)["points"], MAX_MAIN / 2)

    def test_adaptation_max_points_scales_points(self):
        self.assertAlmostEqual(self.score(self.reference, 40)["points"], 20.0, delta=1e-9)

    def test_rejects_missing_prediction(self):
        partial = dict(self.reference)
        partial.pop(next(iter(partial)))
        with self.assertRaises(S.ScoringError):
            self.score(partial)

    def test_rejects_bad_probability_row(self):
        bad = dict(self.reference)
        key = next(iter(bad))
        bad[key] = (0.5, 0.5, 0.5, 0.0, 0.0)
        with self.assertRaises(S.ScoringError):
            self.score(bad)


class LoadSplitTests(ScoringTestCase):
    def test_validation_counts(self):
        self.assertEqual(len(self.split.queries), 768)
        self.assertEqual(len(self.split.targets), 768)
        self.assertEqual(len(self.split.reference), 768)
        self.assertEqual(len(set(self.split.conditions.values())), 4)

    def test_truth_override_replaces_answers(self):
        with tempfile.TemporaryDirectory(prefix="split-") as tmp:
            split_dir = Path(tmp) / "validation"
            split_dir.mkdir()
            shutil.copy(VALIDATION / "queries.csv", split_dir / "queries.csv")
            truth_dir = Path(tmp) / "_truth" / "validation"
            truth_dir.mkdir(parents=True)
            for name in ("targets.csv", "conditions.csv", "reference.csv"):
                shutil.copy(VALIDATION / name, truth_dir / name)
            loaded = S.load_split(split_dir, truth_dir=Path(tmp) / "_truth")
            self.assertEqual(loaded.targets, self.targets)
            self.assertEqual(loaded.conditions, self.split.conditions)
            self.assertEqual(loaded.reference, self.reference)

    def test_missing_answers_are_none_without_override(self):
        loaded = S.load_split(PUBLIC / "test")
        self.assertIsNone(loaded.targets)
        self.assertIsNone(loaded.conditions)
        self.assertIsNone(loaded.reference)
        self.assertGreater(len(loaded.queries), 0)


class ReferenceFromStarterTests(unittest.TestCase):
    def test_reproduces_released_reference(self):
        rebuilt = S.reference_from_starter(PUBLIC)
        released = S.read_predictions(VALIDATION / "reference.csv")
        self.assertEqual(set(rebuilt), set(released))
        max_diff = max(abs(a - b) for key in released for a, b in zip(rebuilt[key], released[key]))
        print(f"\nreference reproduction max abs diff: {max_diff:.3e}")
        self.assertLess(max_diff, 1e-12)


class BootstrapTests(ScoringTestCase):
    def test_ci_contains_point_estimate(self):
        result = S.bootstrap_ci(self.blend, self.targets, self.split.queries, self.split.conditions,
                                self.reference, MAX_MAIN, repeats=300, seed=7)
        self.assertLessEqual(result["ci_low"], result["points"])
        self.assertGreaterEqual(result["ci_high"], result["points"])
        self.assertLess(result["ci_low"], result["ci_high"])

    def test_paired_compare_same_prediction_is_zero(self):
        result = S.paired_compare(self.blend, self.blend, self.targets, self.split.queries,
                                  self.split.conditions, self.reference, MAX_MAIN, repeats=200, seed=7)
        self.assertEqual(result["mean_diff"], 0.0)
        self.assertEqual(result["ci_low"], 0.0)
        self.assertEqual(result["ci_high"], 0.0)

    def test_paired_compare_orders_reference_above_uniform(self):
        result = S.paired_compare(self.reference, self.uniform, self.targets, self.split.queries,
                                  self.split.conditions, self.reference, MAX_MAIN, repeats=200, seed=7)
        self.assertGreater(result["mean_diff"], 0)
        self.assertGreater(result["ci_low"], 0)

    def test_half_splits_shape(self):
        result = S.repeated_half_splits(self.blend, self.targets, self.split.queries, self.split.conditions,
                                        self.reference, MAX_MAIN, n_splits=5, seed=7)
        self.assertEqual(len(result["splits"]), 5)
        for key in ("half_a", "half_b"):
            self.assertEqual(set(result[key]), {"mean", "sd", "min", "max"})
            self.assertLessEqual(result[key]["min"], result[key]["mean"])
            self.assertLessEqual(result[key]["mean"], result[key]["max"])
        self.assertEqual(result["splits"][0]["n_episodes"], {"half_a": 48, "half_b": 48})


class SliceTests(ScoringTestCase):
    def test_slice_counts_sum_to_queries(self):
        table = S.slices(self.blend, self.targets, self.split.queries, self.split.conditions,
                         self.reference, history=VALIDATION / "history.csv")
        for dimension, rows in table.items():
            self.assertEqual(sum(r["n_queries"] for r in rows), len(self.targets), dimension)

    def test_history_kind_matches_condition_labels(self):
        table = S.slices(self.blend, self.targets, self.split.queries, self.split.conditions,
                         self.reference, history=VALIDATION / "history.csv")
        kinds = {row["value"]: row["n_queries"] for row in table["history_kind"]}
        short = {row["value"]: row["n_queries"] for row in table["condition"]}["short_history"]
        self.assertEqual(kinds.get("short"), short)

    def test_win_probability_error_definition(self):
        table = S.slices(self.blend, self.targets, self.split.queries, self.split.conditions,
                         self.reference, history=None)
        expected = {}
        for key, truth in self.targets.items():
            cond = self.split.conditions[key]
            expected.setdefault(cond, []).append(abs((1 - self.blend[key][0]) - (1 - truth[0])))
        for row in table["condition"]:
            values = expected[row["value"]]
            self.assertAlmostEqual(row["win_prob_error"], sum(values) / len(values), delta=1e-12)
        self.assertTrue(all(row["ratio"] >= 0 for row in table["condition"]))


class KFoldTests(unittest.TestCase):
    def test_folds_partition_validation(self):
        folds = S.kfold_validation_protocol(VALIDATION, k=4, seed=11)
        self.assertEqual(len(folds), 4)
        scored = [q for fold in folds for q in fold["scored_query_ids"]]
        self.assertEqual(sorted(scored), sorted(S.read_queries(VALIDATION / "queries.csv")))
        for fold in folds:
            self.assertFalse(set(fold["scored_episodes"]) & set(fold["visible_episodes"]))
            self.assertEqual(fold["scored_conditions"], {c: 6 for c in S.CONDITIONS})


if __name__ == "__main__":
    unittest.main()
