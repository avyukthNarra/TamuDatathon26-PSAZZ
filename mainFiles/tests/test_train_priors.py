"""Tests for priors estimated only from training histories."""
import random
import unittest

from slot_forecasting.schema import Play, Query, ValidationError
from slot_forecasting.train_priors import estimate_train_priors


class TrainPriorsTest(unittest.TestCase):
    def test_recovers_separated_pair_parity_levels(self):
        queries, history = [], []
        for machine in range(5):
            rng = random.Random(44 + machine)
            seq = []
            for tick in range(800):
                if tick < 8:
                    payout = int(rng.random() < 0.5)
                else:
                    agree = (seq[tick - 1] > 0) == (seq[tick - 3] > 0)
                    payout = int(rng.random() < (0.04 if agree else 0.94))
                seq.append(payout)
            episode, machine_id = f"e{machine}", f"m{machine}"
            queries.append(Query(f"q{machine}", episode, machine_id, "typeA", 0, len(seq)))
            history.extend(Play(episode, machine_id, tick, payout) for tick, payout in enumerate(seq))

        size_modes, win_prior = estimate_train_priors(queries, history)
        self.assertEqual(size_modes, {})
        self.assertIsNotNone(win_prior)
        self.assertAlmostEqual(sum(component[0] for component in win_prior), 1.0)
        self.assertLess(win_prior[0][1], 0.15)
        self.assertGreater(win_prior[1][1], 0.8)
        self.assertAlmostEqual(win_prior[2][1], 0.5)

    def test_cross_fitted_size_modes_are_normalized_and_ordered(self):
        queries, history = [], []
        for machine in range(5):
            rng = random.Random(410 + machine)
            seq = []
            for tick in range(600):
                if tick < 4:
                    payout = rng.choice((0, 1, 2, 5, 20))
                else:
                    big_previous = seq[tick - 1] >= 5
                    draw = rng.random()
                    if big_previous:
                        payout = 0 if draw < 0.12 else 5 if draw < 0.57 else 20
                    else:
                        payout = (0 if draw < 0.32 else 1 if draw < 0.63 else
                                  2 if draw < 0.94 else 5 if draw < 0.97 else 20)
                seq.append(payout)
            episode, machine_id = f"e{machine}", f"m{machine}"
            queries.append(Query(f"q{machine}", episode, machine_id, "typeA", 0, len(seq)))
            history.extend(Play(episode, machine_id, tick, payout) for tick, payout in enumerate(seq))

        size_modes, _ = estimate_train_priors(queries, history)
        self.assertEqual(set(size_modes), {"typeA"})
        small, big = size_modes["typeA"]
        for mode in (small, big):
            self.assertAlmostEqual(sum(mode), 1.0)
            self.assertTrue(all(value > 0 for value in mode))
        payout_values = (1, 2, 5, 20)
        expected_small = sum(p * value for p, value in zip(small, payout_values))
        expected_big = sum(p * value for p, value in zip(big, payout_values))
        self.assertLess(expected_small, expected_big)

    def test_observations_at_or_after_training_cutoff_are_rejected(self):
        query = Query("q", "e", "m", "typeA", 0, 2)
        history = (Play("e", "m", 0, 0), Play("e", "m", 1, 1), Play("e", "m", 2, 0))
        with self.assertRaises(ValidationError):
            estimate_train_priors((query,), history)


if __name__ == "__main__":
    unittest.main()
