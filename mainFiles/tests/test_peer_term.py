"""Run from mainFiles/:  python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "entry_template_super"))
import peer_term  # noqa: E402

ROWS = {"hot": [0.5, 0.2, 0.15, 0.1, 0.05], "cold": [0.5, 0.2, 0.15, 0.1, 0.05], "none": [0.5, 0.2, 0.15, 0.1, 0.05]}


class PeerTermTest(unittest.TestCase):
    def setUp(self):
        self.saved = peer_term.peer_shares
        peer_term.peer_shares = lambda split_dir: {"hot": 0.8, "cold": 0.2, "none": None}

    def tearDown(self):
        peer_term.peer_shares = self.saved

    def test_off_without_a_coefficient(self):
        self.assertIs(peer_term.apply(ROWS, "unused", 0.0), ROWS)

    def test_centred_on_the_split_mean_and_sizes_kept(self):
        out = peer_term.apply(ROWS, "unused", 0.75)
        self.assertGreater(1 - out["hot"][0], 0.5)    # share 0.8 > mean 0.5
        self.assertLess(1 - out["cold"][0], 0.5)
        self.assertAlmostEqual((1 - out["hot"][0]) + (1 - out["cold"][0]), 1.0, places=9)  # symmetric about the mean
        self.assertEqual(out["none"], ROWS["none"])   # no peers seen: unchanged
        for row in out.values():
            self.assertAlmostEqual(sum(row), 1.0, places=12)
        self.assertAlmostEqual(out["hot"][1] / (1 - out["hot"][0]), 0.4, places=9)

    def test_off_when_no_peer_json(self):
        self.assertFalse((peer_term.ENTRY / "peer.json").exists())  # the template ships without it: v20 behaviour
        self.assertEqual(peer_term.load_coef(), 0.0)


if __name__ == "__main__":
    unittest.main()
