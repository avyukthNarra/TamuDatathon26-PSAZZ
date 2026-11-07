import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from slot_forecasting.schema import Query, Target, ValidationError

import exp_harness


def bootstrap_fixture():
    conditions = ("stable", "short_history", "drift", "shared_change")
    targets, groups, candidate, reference = [], {}, {}, {}
    for ci, condition in enumerate(conditions):
        for ei, error_level in enumerate((0.01, 0.16)):
            episode = f"e{ci}-{ei}"
            query_id = f"q{ci}-{ei}"
            truth = (0.5, 0.5, 0.0, 0.0, 0.0)
            targets.append(Target(query_id, episode, truth, 0))
            groups[query_id] = condition
            candidate[query_id] = truth
            # Two reference error levels per condition make a frozen denominator wrong.
            reference[query_id] = (0.5 + error_level**0.5 / 2**0.5,
                                   0.5 - error_level**0.5 / 2**0.5, 0.0, 0.0, 0.0)
    return tuple(targets), groups, candidate, reference


class PairedBootstrapTest(unittest.TestCase):
    def test_identical_predictions_have_zero_interval(self):
        targets, groups, candidate, _ = bootstrap_fixture()
        result = exp_harness.paired_points_bootstrap(
            candidate, candidate, candidate, targets, groups, repeats=100, seed=7)
        self.assertEqual(result["point_delta"], 0.0)
        self.assertEqual(result["ci95"], [0.0, 0.0])

    def test_reference_denominator_is_recomputed_for_each_shared_draw(self):
        targets, groups, perfect, reference = bootstrap_fixture()
        result = exp_harness.paired_points_bootstrap(
            perfect, reference, reference, targets, groups, repeats=1000, seed=314159)
        self.assertAlmostEqual(result["point_delta"], 30.0, places=12)
        self.assertEqual(result["ci95"], [30.0, 30.0])

    def test_prediction_id_mismatch_is_rejected(self):
        targets, groups, candidate, reference = bootstrap_fixture()
        candidate.pop(next(iter(candidate)))
        with self.assertRaises(ValidationError):
            exp_harness.paired_points_bootstrap(
                candidate, reference, reference, targets, groups, repeats=10)


class ResultAppendTest(unittest.TestCase):
    def test_second_run_appends_without_replacing_first(self):
        conditions = ("stable", "short_history", "drift", "shared_change")
        queries, targets, groups, predictions = [], [], {}, {}
        for ci, condition in enumerate(conditions):
            for ei in range(2):
                episode, query_id = f"e{ci}-{ei}", f"q{ci}-{ei}"
                queries.append(Query(query_id, episode, f"m{ci}-{ei}", "steady", 0, 1))
                targets.append(Target(query_id, episode, (1, 0, 0, 0, 0), 0))
                groups[query_id] = condition
                predictions[query_id] = (1, 0, 0, 0, 0)
        queries = tuple(queries)
        targets = tuple(targets)
        train_queries = (Query("train-q", "train-e", "train-m", "steady", 0, 1),)
        reference = dict(predictions)
        fit_calls = []

        class Model:
            def fit(self, train_q, train_h, train_labels):
                fit_calls.append((tuple(train_q), tuple(train_h), dict(train_labels)))

            def predict(self, val_q, val_h):
                return dict(predictions)

        with tempfile.TemporaryDirectory() as temporary:
            result_path = Path(temporary) / "runs" / "results.jsonl"
            with patch.object(exp_harness, "round_metadata", return_value={
                    "max_points": 60, "round": "main", "season_id": "0" * 24}), \
                 patch.object(exp_harness, "load_inputs", side_effect=[
                     (train_queries, ()), (queries, ()), (train_queries, ()), (queries, ())]), \
                 patch.object(exp_harness, "read_labels", return_value={"train-q": 0}), \
                 patch.object(exp_harness, "read_targets", return_value=targets), \
                 patch.object(exp_harness, "read_rows", return_value=[
                     {"query_id": key, "condition": value} for key, value in groups.items()]), \
                 patch.object(exp_harness, "read_submission", return_value=reference):
                for name in ("first", "second"):
                    exp_harness.run_experiment(Model, data=temporary, name=name,
                                               baseline_factory=Model,
                                               results_path=result_path, repeats=5)
            lines = result_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(fit_calls), 4)
        self.assertTrue(all(call == (train_queries, (), {"train-q": 0}) for call in fit_calls))
        self.assertEqual(len(lines), 2)
        self.assertEqual([json.loads(line)["name"] for line in lines], ["first", "second"])
        records = [json.loads(line) for line in lines]
        self.assertTrue(all(record["baseline_factory"].endswith(
            "ResultAppendTest.test_second_run_appends_without_replacing_first.<locals>.Model")
            for record in records))
        self.assertTrue(all(record["baseline_runtime_seconds"] >= 0 for record in records))


if __name__ == "__main__":
    unittest.main()
