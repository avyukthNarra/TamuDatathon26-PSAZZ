import unittest
from unittest.mock import patch

import numpy as np

from mainFiles.slot_forecasting import combo_e2_memory
from mainFiles.slot_forecasting.combo_e2_memory import ComboAdaptiveMemoryForecaster


def _history(size=56):
    payouts = (0, 1, 0, 2, 0, 5, 1, 0, 20, 1, 0)
    return {tick: payouts[tick % len(payouts)] for tick in range(size)}


class ComboAdaptiveMemoryTests(unittest.TestCase):
    def _model(self, **kwargs):
        model = ComboAdaptiveMemoryForecaster(
            lags=2,
            pairs=False,
            min_rows=4,
            max_eval_rows=24,
            eval_stride=4,
            min_train=16,
            selector_tree_estimators=2,
            **kwargs,
        )
        model.base.global_prior = (0.6, 0.15, 0.1, 0.1, 0.05)
        return model

    def test_tree_and_logistic_select_memory_independently(self):
        model = self._model(n_estimators=4)
        probability = model.win_probability(_history(56), cutoff=55)

        self.assertTrue(np.isfinite(probability))
        self.assertEqual(model.mix, 0.5)
        self.assertEqual(model.short_alpha, 0.6)
        diagnostic = next(iter(model.selection_diagnostics.values()))
        for family in ("tree", "logistic"):
            weights = diagnostic[family]["selected_weights"]
            self.assertEqual(set(weights), {
                "half_life=none", "half_life=16", "half_life=32", "half_life=64",
            })
            self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_current_and_future_outcomes_do_not_change_cutoff_prediction(self):
        model = self._model(n_estimators=4)
        history = _history(55)
        original = model.win_probability(history, cutoff=55)

        changed = dict(history)
        changed.update({55: 20, 56: 0, 57: 5})
        altered = model.win_probability(changed, cutoff=55)

        self.assertEqual(original, altered)

    def test_prequential_predictions_exclude_the_evaluated_label(self):
        rng = np.random.default_rng(91)
        ticks = np.arange(40, dtype=float)
        features = rng.normal(size=(40, 3))
        labels = (np.arange(40) % 3 == 0).astype(int)
        query = np.array([0.2, -0.4, 0.8])
        model = self._model(n_estimators=3)

        for family in ("tree", "logistic"):
            _, before = model._select_family(
                family, features, labels, ticks, query, 40.0, 0.4,
            )
            changed = labels.copy()
            changed[20] = 1 - changed[20]
            _, after = model._select_family(
                family, features, changed, ticks, query, 40.0, 0.4,
            )
            before_row = next(row for row in before["prequential_predictions"]
                              if row["tick"] == 20.0)
            after_row = next(row for row in after["prequential_predictions"]
                             if row["tick"] == 20.0)
            self.assertEqual(before_row["probabilities"], after_row["probabilities"])

    def test_recency_weights_use_actual_tick_gaps(self):
        ticks = np.array([1, 3, 8, 10], dtype=float)
        weights = ComboAdaptiveMemoryForecaster._weights(ticks, cutoff=11, half_life=2)

        np.testing.assert_allclose(
            weights,
            [2 ** (-9 / 2), 2 ** (-7 / 2), 2 ** (-2 / 2), 1.0],
        )

    def test_prequential_trees_use_small_budget_and_final_trees_keep_combo_size(self):
        fitted_tree_counts = []

        class RecordingTrees:
            def __init__(self, n_estimators, **kwargs):
                self.n_estimators = n_estimators
                fitted_tree_counts.append(n_estimators)

            def fit(self, features, labels, sample_weight=None):
                self.classes_ = np.array([0, 1])
                return self

            def predict_proba(self, features):
                return np.tile([0.45, 0.55], (len(features), 1))

        model = ComboAdaptiveMemoryForecaster(
            lags=2, pairs=False, min_rows=4, max_eval_rows=24,
            eval_stride=4, min_train=16,
        )
        model.base.global_prior = (0.6, 0.15, 0.1, 0.1, 0.05)
        with patch.object(combo_e2_memory, "ExtraTreesClassifier", RecordingTrees):
            probability = model.win_probability(_history(56), cutoff=55)

        self.assertTrue(np.isfinite(probability))
        self.assertIn(8, fitted_tree_counts)
        self.assertEqual(fitted_tree_counts[-4:], [100, 100, 100, 100])


if __name__ == "__main__":
    unittest.main()
