"""Protocol-level tests for caller-controlled bootstrap reproducibility."""

from pathlib import Path
import unittest

from eval_suite import scoring
from eval_suite.protocol import score_bundle


KIT = Path(__file__).resolve().parents[2]
VALIDATION = KIT / "public" / "validation"


class ScoreBundleSeedTests(unittest.TestCase):
    def test_nondefault_seed_changes_bootstrap_ci_draws(self):
        split = scoring.load_split(VALIDATION)
        # A uniform forecast has a non-degenerate bootstrap distribution here;
        # the exact reference would score exactly half on every resample.
        prediction = {query_id: (0.2,) * 5 for query_id in split.targets}
        kwargs = dict(pred_csv=prediction, split=split,
                      history_path=VALIDATION / "history.csv", max_points=60,
                      repeats=200, n_splits=2)

        seed_11 = score_bundle(**kwargs, seed=11)
        seed_29 = score_bundle(**kwargs, seed=29)

        self.assertNotEqual((seed_11["ci_low"], seed_11["ci_high"]),
                            (seed_29["ci_low"], seed_29["ci_high"]))


if __name__ == "__main__":
    unittest.main()
