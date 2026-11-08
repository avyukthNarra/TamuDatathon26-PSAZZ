"""Run from mainFiles/:  python -m unittest discover -s tests -v"""
import math
import unittest

from slot_forecasting.combo import combine
from slot_forecasting.size_rule_hmm import SizeRuleHMM, default_rules
from slot_forecasting.strongest import two_modes


class ComboTest(unittest.TestCase):
    def test_win_is_a_log_odds_average_and_sizes_come_from_v7c(self):
        ours = {"q": (0.2, 0.4, 0.2, 0.15, 0.05)}       # win 0.8
        theirs = {"q": (0.5, 0.1, 0.1, 0.2, 0.1)}        # win 0.5, sizes 0.2 / 0.2 / 0.4 / 0.2
        p = combine(ours, theirs)["q"]
        z = 0.5 * math.log(0.8 / 0.2) + 0.5 * math.log(1.0)
        self.assertAlmostEqual(1 - p[0], 1 / (1 + math.exp(-z)), places=6)
        self.assertAlmostEqual(p[3] / (1 - p[0]), 0.4, places=6)
        self.assertAlmostEqual(sum(p), 1.0, places=9)

    def test_two_modes_orders_small_then_big(self):
        modes = two_modes({"steady": {"modes": [(0.2, 0.3, 0.3, 0.2), (0.7, 0.2, 0.08, 0.02)], "weights": [1, 1]}})
        self.assertEqual(modes["steady"][0], (0.7, 0.2, 0.08, 0.02))

    def test_size_rule_hmm_learns_a_lag_two_rule(self):
        A, B = (0.8, 0.15, 0.04, 0.01), (0.1, 0.2, 0.4, 0.3)
        seq, payouts = {}, (1, 2, 5, 20)
        import random
        rng = random.Random(1)
        for t in range(200):
            big = t >= 2 and seq.get(t - 2, 0) >= 2
            seq[t] = rng.choices(payouts, B if big else A)[0] if rng.random() < 0.7 else 0
        hmm = SizeRuleHMM({"steady": (A, B)}, prior=default_rules(core_weight=0.97), switch=0.0)
        pb = hmm.big_probability(seq, 200, A, B)
        self.assertAlmostEqual(pb, 1.0 if seq[198] >= 2 else 0.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
