import unittest
from unittest.mock import patch

from slot_forecasting import zhill_blend
from slot_forecasting.zhill_blend import ZhillBlendForecaster, adapt_with_memory


class ZhillBlendWrapperTests(unittest.TestCase):
    def test_adaptation_sets_memory_before_train_cutoff_mode_selection(self):
        made = []

        class FakeMixer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.cell_mode = (kwargs["cell_mode"],)
                self.fitted = False
                made.append(self)

            def fit(self, queries, history):
                self.fitted = True
                return self

        with patch.object(zhill_blend, "choose_pairs", return_value=((1, 3),)) as pairs:
            with patch.object(zhill_blend, "PairMixer", FakeMixer):
                with patch.object(zhill_blend, "cutoff_brier", side_effect=(0.4, 0.3)) as score:
                    selected = adapt_with_memory(
                        ("train-query",), ("train-history",), {"train-query": 1},
                        half_life=(12.0, 40.0, 1e9), log=lambda *_: None,
                    )

        pairs.assert_called_once_with(("train-query",), ("train-history",))
        self.assertEqual([m.kwargs["half_life"] for m in made], [
            (12.0, 40.0, 1e9), (12.0, 40.0, 1e9),
        ])
        self.assertEqual([m.kwargs["cell_mode"] for m in made], ["parity", "full"])
        self.assertTrue(all(m.fitted for m in made))
        self.assertEqual(score.call_count, 2)
        self.assertIs(selected, made[1])

    def test_default_and_never_forget_memory_settings(self):
        baseline = ZhillBlendForecaster()
        extended = ZhillBlendForecaster(half_life=(12.0, 40.0, 1e9))

        self.assertEqual(baseline.half_life, (12.0, 40.0))
        self.assertEqual(extended.half_life, (12.0, 40.0, 1e9))
        self.assertEqual(baseline.rule_weight, 0.2)

    def test_prediction_uses_exact_two_component_blend_weights(self):
        class ConstantModel:
            def __init__(self, result):
                self.result = result

            def predict(self, queries, history):
                return self.result

        query_id = "q"
        rules = {query_id: (0.5, 0.1, 0.1, 0.2, 0.1)}
        mixer = {query_id: (0.1, 0.2, 0.3, 0.2, 0.2)}
        model = ZhillBlendForecaster()
        model.rules = ConstantModel(rules)
        model.mixer = ConstantModel(mixer)

        result = model.predict((query_id,), ())
        expected = tuple(0.2 * r + 0.8 * m for r, m in zip(rules[query_id], mixer[query_id]))

        self.assertEqual(result[query_id], expected)


if __name__ == "__main__":
    unittest.main()
