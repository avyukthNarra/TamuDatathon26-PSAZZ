"""Run with python -m unittest fresh_ml.test_features."""
import unittest
from pathlib import Path
import tempfile

import numpy as np

from .features import CONTEXTS, machine_features, previous_sum


class CausalFeaturesTest(unittest.TestCase):
    def test_sums_are_strict_prefixes(self):
        x = np.arange(6.)[:, None]
        np.testing.assert_array_equal(previous_sum(x).ravel(), [0, 0, 1, 3, 6, 10])
        np.testing.assert_array_equal(previous_sum(x, 2).ravel(), [0, 0, 1, 3, 5, 7])

    def test_current_and_future_own_and_peer_outcomes_cannot_change_features(self):
        rng = np.random.default_rng(71)
        seq = rng.integers(-1, 5, 80)
        peers = rng.integers(-1, 5, (7, 80))
        before, _, _ = machine_features(seq, peers, 'steady', 0)
        for cutoff in (0, 1, 7, 8, 32, 49, 79):
            altered, peer_altered = seq.copy(), peers.copy()
            altered[cutoff:] = rng.integers(-1, 5, len(seq) - cutoff)
            peer_altered[:, cutoff:] = rng.integers(-1, 5, (7, len(seq) - cutoff))
            after, _, _ = machine_features(altered, peer_altered, 'steady', 0)
            np.testing.assert_array_equal(before[:cutoff + 1], after[:cutoff + 1])

    def test_gap_is_not_a_loss_or_a_shifted_lag(self):
        seq = np.array([1, -1, 0, -1])
        x, names, _ = machine_features(seq, np.full((2, 4), -1), 'balanced', -1)
        self.assertEqual(x[2, names.index('lag1_observed')], 0)
        self.assertEqual(x[2, names.index('lag1_category')], -1)
        self.assertEqual(x[2, names.index('lag2_category')], 1)
        self.assertEqual(x[3, names.index('lag1_category')], 0)
        self.assertEqual(x[3, names.index('lag1_observed')], 1)

    def test_conditional_evidence_excludes_current_label(self):
        seq = np.array([0, 1, 0, 1, 0, 4])
        x, names, _ = machine_features(seq, np.full((0, 6), -1), 'jackpot', 1)
        # At tick 5 current lag1 is a loss: earlier matching labelled ticks
        # are ticks 1 and 3. The jackpot at tick 5 must not enter counts.
        count = x[5, names.index('ctx1_None_logn')]
        self.assertAlmostEqual(count, np.log1p(2), places=6)
        self.assertEqual(len(CONTEXTS), 36)

    def test_query_signal_does_not_enter_other_features(self):
        seq = np.array([0, 1, -1, 2, 0])
        peers = np.array([[1, 0, 3, -1, -1]])
        first, names, _ = machine_features(seq, peers, 'steady', -1)
        second, _, _ = machine_features(seq, peers, 'steady', 1)
        mask = np.array(names) != 'signal'
        np.testing.assert_array_equal(first[:, mask], second[:, mask])

    def test_prediction_path_uses_train_choice_without_scoring_inputs(self):
        from .run import predict_without_targets
        from slot_forecasting.io import read_submission
        dataset = {'splits': np.array(['validation', 'test']), 'ids': np.array(['val', 'test'])}
        predictions = {'a': np.tile([.5, .2, .1, .1, .1], (2, 1)),
                       'b': np.tile([.6, .1, .1, .1, .1], (2, 1))}
        training = {'a': {'selection_logloss': 2}, 'b': {'selection_logloss': 1}}
        with tempfile.TemporaryDirectory() as tmp:
            result = predict_without_targets(Path(tmp), dataset, predictions, training)
            self.assertEqual(result['train_only_choice'], 'b')
            np.testing.assert_allclose(read_submission(Path(tmp) / 'submission.csv')['test'], predictions['b'][1])

    def test_dataset_builds_without_validation_answers_or_conditions(self):
        from .run import load_dataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ('train', 'validation', 'test'):
                folder = root / split
                folder.mkdir()
                (folder / 'queries.csv').write_text(
                    'query_id,episode_id,machine_id,machine_type,signal,cutoff\n'
                    f'{split}-q,{split}-e,m,steady,0,40\n')
                (folder / 'history.csv').write_text(
                    'episode_id,machine_id,tick,payout\n' +
                    ''.join(f'{split}-e,m,{t},{(0, 1, 2, 5, 20)[t % 5]}\n' for t in range(40)))
            (root / 'train/labels.csv').write_text('query_id,outcome\ntrain-q,1\n')
            output = root / 'run'
            output.mkdir()
            data = load_dataset(root, output, 20)
            self.assertEqual(data['qx'].shape[0], 3)
            self.assertTrue(np.isfinite(data['x']).all())
            np.testing.assert_array_equal(data['y'], data['row_ticks'] % 5)


if __name__ == '__main__':
    unittest.main()
