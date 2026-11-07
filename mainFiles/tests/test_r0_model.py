"""Tests for the safe entry wrapper and its packaged imports."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from slot_forecasting.io import read_submission, write_rows
import r0_model
from r0_model import CombinedForecaster


ROOT = Path(__file__).resolve().parents[1]


class R0ModelTest(unittest.TestCase):
    def test_validation_priors_must_be_injected(self):
        with self.assertRaises(ValueError):
            CombinedForecaster(prior_source="validation")

    def test_make_blend_uses_twenty_percent_average_of_two_rule_variants(self):
        query = object()
        mixer_probs = (0.5, 0.2, 0.15, 0.1, 0.05)
        full_probs = (0.4, 0.3, 0.15, 0.1, 0.05)
        binary_probs = (0.6, 0.2, 0.1, 0.05, 0.05)

        class FakeMixer:
            def predict(self, *_args):
                return {"q": mixer_probs}

        class FakeRule:
            def __init__(self, binary=False, **_kwargs):
                self.probabilities = binary_probs if binary else full_probs

            def fit(self, *_args):
                return self

            def predict(self, *_args):
                return {"q": self.probabilities}

        with patch.object(r0_model, "estimate_train_priors", return_value=({}, None)), \
                patch.object(r0_model, "adapt", return_value=FakeMixer()), \
                patch.object(r0_model, "RuleForecaster", FakeRule):
            model = r0_model.make_blend().fit((query,), (), {})
        expected_rule = tuple((a + b) / 2 for a, b in zip(full_probs, binary_probs))
        expected = tuple(0.8 * mixer + 0.2 * rule for mixer, rule in zip(mixer_probs, expected_rule))
        self.assertEqual(model.predict((query,), ())["q"], expected)

    def test_train_entry_does_not_call_target_reader(self):
        spec = importlib.util.spec_from_file_location("r0_test_entry", ROOT / "entry_template/predict.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.read_targets = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("safe train entry attempted to read validation targets"))

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "public"
            train, test = data / "train", data / "test"
            train.mkdir(parents=True)
            test.mkdir()
            train_queries, train_history, labels = [], [], []
            test_queries, test_history = [], []
            for machine in range(3):
                mid = f"m{machine}"
                train_queries.append({"query_id": f"train{machine}", "episode_id": "train_ep",
                                     "machine_id": mid, "machine_type": "typeA", "signal": "0", "cutoff": "48"})
                labels.append({"query_id": f"train{machine}", "outcome": "1"})
                for tick in range(48):
                    train_history.append({"episode_id": "train_ep", "machine_id": mid,
                                          "tick": str(tick), "payout": str((tick + machine) % 3 == 0 and 1 or 0)})
                test_queries.append({"query_id": f"test{machine}", "episode_id": "test_ep",
                                     "machine_id": mid, "machine_type": "typeA", "signal": "0", "cutoff": "12"})
                for tick in range(12):
                    test_history.append({"episode_id": "test_ep", "machine_id": mid,
                                         "tick": str(tick), "payout": str((tick + machine) % 3 == 0 and 1 or 0)})
            write_rows(train / "queries.csv", ("query_id", "episode_id", "machine_id", "machine_type", "signal", "cutoff"), train_queries)
            write_rows(train / "history.csv", ("episode_id", "machine_id", "tick", "payout"), train_history)
            write_rows(train / "labels.csv", ("query_id", "outcome"), labels)
            write_rows(test / "queries.csv", ("query_id", "episode_id", "machine_id", "machine_type", "signal", "cutoff"), test_queries)
            write_rows(test / "history.csv", ("episode_id", "machine_id", "tick", "payout"), test_history)
            output = Path(temporary) / "submission.csv"
            module.main(["--data", str(data), "--out", str(output)])
            predictions = read_submission(output)
            self.assertEqual(set(predictions), {f"test{machine}" for machine in range(3)})

    def test_packaged_entry_help_works_without_repository_pythonpath(self):
        with tempfile.TemporaryDirectory() as temporary:
            entry = Path(temporary) / "entry"
            package = entry / "slot_forecasting"
            package.mkdir(parents=True)
            for name in ("__init__.py", "schema.py", "io.py", "baseline.py", "pair_mixer.py", "train_priors.py"):
                shutil.copyfile(ROOT / "slot_forecasting" / name, package / name)
            shutil.copyfile(ROOT / "entry_template/predict.py", entry / "predict.py")
            shutil.copyfile(ROOT / "r0_model.py", entry / "r0_model.py")
            shutil.copyfile(ROOT / "rule_model.py", entry / "rule_model.py")
            environment = dict(os.environ)
            environment.pop("PYTHONPATH", None)
            result = subprocess.run([sys.executable, "predict.py", "--help"], cwd=entry,
                                    env=environment, check=True, capture_output=True, text=True)
            self.assertIn("--prior-source", result.stdout)


if __name__ == "__main__":
    unittest.main()
