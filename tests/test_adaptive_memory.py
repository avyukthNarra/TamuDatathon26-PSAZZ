import unittest

import numpy as np

from mainFiles.slot_forecasting.adaptive_memory import AdaptiveMemorySelector


class AdaptiveMemorySelectorTests(unittest.TestCase):
    def test_future_labels_do_not_change_cutoff_prediction(self):
        ticks = np.arange(40)
        X = np.column_stack((np.ones(40), ticks / 40))
        y = (ticks % 3 == 0).astype(int)
        selector = AdaptiveMemorySelector(half_lives=(None, 8), Cs=(1,), min_train=8)

        before, _ = selector.predict_win(X, y, ticks, [1, 1], cutoff=28)
        changed = y.copy()
        changed[28:] = 1 - changed[28:]
        after, diagnostics = selector.predict_win(X, changed, ticks, [1, 1], cutoff=28)

        self.assertEqual(before, after)
        self.assertEqual(diagnostics["eligible_rows"], 28)
        self.assertLess(max(diagnostics["prequential_ticks"]), 28)

    def test_prequential_prediction_does_not_use_its_current_label(self):
        ticks = np.arange(40)
        X = np.column_stack((np.ones(40), ticks / 40))
        y = (ticks % 3 == 0).astype(int)
        selector = AdaptiveMemorySelector(half_lives=(None,), Cs=(1,), min_train=8)

        _, original = selector.predict_win(X, y, ticks, [1, 1], cutoff=40)
        changed = y.copy()
        changed[20] = 1 - changed[20]
        _, altered = selector.predict_win(X, changed, ticks, [1, 1], cutoff=40)
        prediction_at_20 = lambda result: next(
            row["probabilities"]["half_life=none,C=1"]
            for row in result["prequential_predictions"] if row["tick"] == 20
        )

        self.assertEqual(prediction_at_20(original), prediction_at_20(altered))

    def test_recency_weights_use_tick_distance_across_gaps(self):
        ticks = np.array([1, 3, 8, 10])
        X = np.column_stack((np.ones(4), np.arange(4)))
        y = np.array([0, 1, 0, 1])
        selector = AdaptiveMemorySelector(half_lives=(2,), Cs=(1,), min_train=8)

        _, diagnostics = selector.predict_win(X, y, ticks, [1, 4], cutoff=11)

        np.testing.assert_allclose(
            diagnostics["final_sample_weights"]["half_life=2,C=1"],
            [2 ** (-9 / 2), 2 ** (-7 / 2), 2 ** (-2 / 2), 1.0],
        )

    def test_single_class_short_history_uses_finite_smoothed_fallback(self):
        ticks = np.array([0, 2, 5])
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.ones(3, dtype=int)
        selector = AdaptiveMemorySelector(half_lives=(None,), Cs=(1,), min_train=4)

        probability, diagnostics = selector.predict_win(X, y, ticks, [4.0], cutoff=6, prior_win=0.25)

        self.assertTrue(np.isfinite(probability))
        self.assertAlmostEqual(probability, (3 + 2 * 0.25) / 5)
        self.assertTrue(diagnostics["fallback"])

    def test_forecaster_uses_train_prior_and_filters_future_ticks(self):
        from mainFiles.slot_forecasting.adaptive_memory import AdaptiveMemoryForecaster

        model = AdaptiveMemoryForecaster(
            lags=2, pairs=False, half_lives=(None,), Cs=(1,), min_train=4,
            max_eval_rows=8, eval_stride=1,
        )
        model.base.global_prior = (0.25, 0.25, 0.2, 0.2, 0.1)
        history = {tick: (tick % 2) for tick in range(20)}
        extended = {**history, 20: 1, 21: 1}

        first = model.win_probability(history, 20)
        second = model.win_probability(extended, 20)

        self.assertEqual(first, second)
        key = (20, tuple(sorted(history.items())))
        self.assertEqual(model.selection_diagnostics[key]["eligible_rows"], 18)


if __name__ == "__main__":
    unittest.main()
