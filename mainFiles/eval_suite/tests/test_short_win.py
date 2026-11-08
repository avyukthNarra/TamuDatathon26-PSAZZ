import unittest

import numpy as np

from short_win import filter_win, hypotheses, patch


class ShortWinTests(unittest.TestCase):
    def test_marginal_filter_uses_a_missing_partner(self):
        observed = {tick: 1 for tick in range(12)}
        observed[1] = 0
        del observed[2]
        hypothesis = hypotheses("steady", 0, grid=False)
        marginal = filter_win(observed, 12, hypothesis, marginalize=True)[0]
        dropped = filter_win(observed, 12, hypothesis, marginalize=False)[0]
        self.assertTrue(np.isfinite(marginal))
        self.assertNotAlmostEqual(marginal, dropped, places=6)

    def test_patch_keeps_v13_size_split(self):
        base = {"q": np.array([0.2, 0.4, 0.2, 0.1, 0.1])}
        patched = patch(base, {"q": 0.5})
        self.assertAlmostEqual(float(patched["q"][1:].sum()), 0.5, places=12)
        self.assertTrue(np.allclose(patched["q"][1:] / patched["q"][1:].sum(), base["q"][1:] / 0.8))


if __name__ == "__main__":
    unittest.main()
