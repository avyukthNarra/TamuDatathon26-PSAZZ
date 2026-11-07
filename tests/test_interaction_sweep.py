"""Focused checks for configurable per-machine lag interactions."""
from itertools import combinations
import unittest

import numpy as np

from mainFiles.slot_forecasting.interaction_sweep import InteractionSweepForecaster
from mainFiles.slot_forecasting.pair_logistic import PairLogisticForecaster
from mainFiles.slot_forecasting.schema import Play, Query


class InteractionSweepForecasterTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(41)
        self.seq = {tick: int(rng.random() < (0.35 + 0.3 * (tick % 3 == 0)))
                    for tick in range(100)}

    def test_feature_shapes_for_each_degree(self):
        x = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=float)
        degree1 = InteractionSweepForecaster(degree=1)
        degree2 = InteractionSweepForecaster(degree=2)
        degree3 = InteractionSweepForecaster(degree=3)
        self.assertEqual(degree1.feature_vector(x).shape, (8,))
        self.assertEqual(degree2.feature_vector(x).shape, (36,))
        self.assertEqual(degree3.feature_vector(x).shape, (92,))
        signed = InteractionSweepForecaster(degree=1, encoding='signed')
        np.testing.assert_array_equal(signed.feature_vector(x), 2 * x - 1)

    def test_vectorized_features_match_independent_combination_products(self):
        rng = np.random.default_rng(27)
        main_lags = rng.integers(0, 2, size=(19, 8)).astype(float)
        for degree in (1, 2, 3):
            for encoding in ('binary', 'signed'):
                with self.subTest(degree=degree, encoding=encoding):
                    model = InteractionSweepForecaster(degree=degree, encoding=encoding)
                    encoded = main_lags if encoding == 'binary' else 2 * main_lags - 1
                    expected_blocks = [encoded]
                    for size in range(2, degree + 1):
                        expected_blocks.append(np.asarray([
                            [np.prod(row[list(indexes)])
                             for indexes in combinations(range(8), size)]
                            for row in encoded
                        ]))
                    expected = np.concatenate(expected_blocks, axis=1)
                    np.testing.assert_array_equal(model.feature_matrix(main_lags), expected)
                    for index in (0, 7, len(main_lags) - 1):
                        np.testing.assert_array_equal(
                            model.feature_vector(main_lags[index]), expected[index]
                        )

    def test_degree_two_binary_matches_e1_prediction(self):
        e1 = PairLogisticForecaster(lags=8, pairs=True, C=1.2,
                                    half_life=16, min_rows=12)
        e6 = InteractionSweepForecaster(degree=2, encoding='binary', C=1.2,
                                        half_life=16, min_rows=12)
        clean = {tick: payout for tick, payout in self.seq.items() if tick < 80}
        self.assertAlmostEqual(e1.win_probability(clean, 80),
                               e6.win_probability(clean, 80), places=14)

    def test_current_and_future_rows_do_not_change_prediction(self):
        forecaster = InteractionSweepForecaster(degree=3, encoding='signed')
        cutoff = 80
        shorter = {tick: value for tick, value in self.seq.items() if tick < cutoff}
        self.assertEqual(forecaster.win_probability(self.seq, cutoff),
                         forecaster.win_probability(shorter, cutoff))

    def test_predict_returns_valid_distribution(self):
        query = Query('q0', 'ep0', 'm0', 'balanced', 1, 80)
        history = [Play('ep0', 'm0', tick, int(value))
                   for tick, value in self.seq.items() if tick < 80]
        labels = {'q0': 2}
        forecaster = InteractionSweepForecaster(degree=2, encoding='binary')
        forecaster.fit([query], history, labels)
        result = forecaster.predict([query], history)['q0']
        self.assertEqual(len(result), 5)
        self.assertTrue(all(np.isfinite(p) and 0 <= p <= 1 for p in result))
        self.assertAlmostEqual(sum(result), 1.0, places=12)
        self.assertEqual(forecaster.diagnostics['successful_fits'], 1)
        self.assertEqual(forecaster.diagnostics['fallbacks'], 0)
        self.assertGreaterEqual(forecaster.diagnostics['max_iter_hits'], 0)

    def test_l1_predict_returns_valid_probability_vector(self):
        query = Query('q1', 'ep1', 'm1', 'balanced', 1, 80)
        history = [Play('ep1', 'm1', tick, int(value))
                   for tick, value in self.seq.items() if tick < 80]
        model = InteractionSweepForecaster(
            degree=2, encoding='signed', penalty='l1', C=0.3,
            half_life=32, min_rows=12,
        )
        model.fit([query], history, {'q1': 2})

        result = model.predict([query], history)['q1']

        self.assertEqual(len(result), 5)
        self.assertTrue(all(np.isfinite(p) and 0 <= p <= 1 for p in result))
        self.assertAlmostEqual(sum(result), 1.0, places=12)
        self.assertEqual(model.diagnostics['successful_fits'], 1)
        self.assertEqual(model.diagnostics['fallbacks'], 0)

    def test_unsupported_penalty_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "penalty must be 'l1' or 'l2'"):
            InteractionSweepForecaster(penalty='elasticnet')


if __name__ == '__main__':
    unittest.main()
