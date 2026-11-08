import unittest

import numpy as np

from slot_forecasting.schema import Play, Query
from bayes_experiment import BayesianLatentModel, DataView, laplace_map


class BayesianExperimentTests(unittest.TestCase):
    def test_laplace_map_returns_finite_posterior(self):
        x = np.asarray([[1.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
        y = np.asarray([0.0, 1.0, 1.0])
        beta, covariance, evidence = laplace_map(
            x, y, np.ones(3), np.zeros(2), np.ones(2)
        )
        self.assertTrue(np.isfinite(beta).all())
        self.assertTrue(np.isfinite(covariance).all())
        self.assertTrue(np.isfinite(evidence))

    def test_missing_lag_is_explicit_and_forecast_normalizes(self):
        query = Query("q", "e", "m", "steady", 0, 3)
        history = (
            Play("e", "m", 0, 0),
            Play("e", "m", 2, 5),
        )
        data = DataView((query,), history)
        feature = data.feature(("e", "m"), 3, shared=False)
        self.assertEqual(feature[7 + 3 + 2], 1.0)

        model = BayesianLatentModel(
            half_life=None, shared=False, drop_missing=False
        ).fit(data, {"q": 2})
        prediction, _ = model.predict(data)
        self.assertAlmostEqual(sum(prediction["q"]), 1.0, places=12)
        self.assertTrue(np.isfinite(prediction["q"]).all())


if __name__ == "__main__":
    unittest.main()
