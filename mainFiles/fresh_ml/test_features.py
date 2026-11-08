"""Run with python -m unittest fresh_ml.test_features."""
import unittest

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


if __name__ == '__main__':
    unittest.main()
