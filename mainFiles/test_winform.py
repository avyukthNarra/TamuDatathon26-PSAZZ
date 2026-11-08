"""Checks for the win-form likelihood pieces. No dataset required."""

import unittest

import numpy as np

from winform_study import (
    CELLS,
    _cell_counts,
    blend_with_size_split,
    cutoff_cell,
    dist_from_p,
    level_stats,
    true_sizes,
    u_at_cutoff,
)


class LevelStatsTest(unittest.TestCase):
    def test_no_data_integrates_the_prior(self):
        logz, mean_p, mean_theta, sd = level_stats(0, 0, 0.0, 0.5)
        self.assertAlmostEqual(logz, 0.0, places=2)
        self.assertAlmostEqual(mean_theta, 0.0, places=2)
        self.assertAlmostEqual(sd, 0.5, delta=0.02)
        # E[sigmoid(0.5 Z)] is above 0.5 by symmetry only at mu=0, where it is 0.5.
        self.assertAlmostEqual(mean_p, 0.5, delta=0.01)

    def test_no_data_mean_moves_with_the_center(self):
        _, low, _, _ = level_stats(0, 0, -2.0, 0.4)
        _, high, _, _ = level_stats(0, 0, 2.0, 0.4)
        self.assertLess(low, 0.2)
        self.assertGreater(high, 0.8)

    def test_many_wins_overcome_a_low_prior(self):
        _, mean_p, _, _ = level_stats(900, 1000, -2.0, 1.0)
        self.assertAlmostEqual(mean_p, 0.9, delta=0.02)

    def test_point_mass_matches_the_sigmoid(self):
        logz, mean_p, mean_theta, sd = level_stats(0, 0, 1.5, 0.0)
        self.assertEqual(sd, 0.0)
        self.assertAlmostEqual(mean_theta, 1.5)
        self.assertAlmostEqual(mean_p, 1 / (1 + np.exp(-1.5)), places=6)
        self.assertEqual(logz, 0.0)


class ScheduleAndPackingTest(unittest.TestCase):
    def test_ramp_anchored_past_its_length_is_finished_at_the_cutoff(self):
        self.assertEqual(u_at_cutoff(96, 62), 1.0)
        self.assertAlmostEqual(u_at_cutoff(31, 62), 0.5)
        self.assertEqual(u_at_cutoff(0, 62), 0.0)

    def test_v13_size_split_scales_with_our_win_probability(self):
        v13 = np.array([0.5, 0.2, 0.2, 0.05, 0.05])
        blended = blend_with_size_split(0.2, v13)
        self.assertTrue(np.allclose(blended, [0.8, 0.08, 0.08, 0.02, 0.02]))
        self.assertAlmostEqual(float(blended.sum()), 1.0)
        # The conditional size split is v13's, not a flat or true-size vector.
        self.assertAlmostEqual(blended[1] / 0.2, 0.4)

    def test_true_sizes_rebuild_the_target(self):
        target = np.array([0.2, 0.5, 0.2, 0.08, 0.02])
        rebuilt = dist_from_p(0.8, true_sizes(target))
        self.assertTrue(np.allclose(rebuilt, target))
        self.assertAlmostEqual(float(rebuilt.sum()), 1.0)

    def test_flip_reassigns_levels_after_the_tick(self):
        cutoff = 6
        seq = np.array([1, 0, 1, 0, 1, 0, -1], dtype=np.int16)
        alpha = np.array([1, 0, 1, 0, 1, 0, 1], dtype=float)
        use = np.ones(cutoff, dtype=bool)
        n, k = _cell_counts(seq, alpha, direction=1, use=use, flip_at=3)
        # See the hand count in the test: live n=4 k=2, dead n=2 k=1.
        self.assertEqual(list(n), [2, 4])
        self.assertEqual(list(k), [1, 2])
        self.assertEqual(cutoff_cell(alpha, cutoff, direction=1, flipped=True), "dead")
        self.assertEqual(cutoff_cell(alpha, cutoff, direction=1, flipped=False), "live")
        self.assertIn("dead", CELLS)


if __name__ == "__main__":
    unittest.main()
