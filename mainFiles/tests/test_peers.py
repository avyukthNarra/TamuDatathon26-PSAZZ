"""Run from mainFiles/:  python -m unittest discover -s tests -v"""
import unittest

from slot_forecasting.peers import PeerAdjusted, peer_share
from slot_forecasting.schema import Play, Query


class Flat:
    def predict(self, queries, history):
        return {q.query_id: (0.5, 0.2, 0.15, 0.1, 0.05) for q in queries}


def episode(peer_payout, cutoff=20, extra=()):
    queries = tuple(Query(f"q{m}", "e", f"m{m}", "steady", 0, cutoff) for m in range(3))
    history = [Play("e", "m0", t, 0) for t in range(cutoff)]
    history += [Play("e", f"m{m}", t, peer_payout) for m in (1, 2) for t in range(cutoff)]
    return queries, tuple(history) + tuple(extra)


class PeerTest(unittest.TestCase):
    def test_hot_peers_raise_and_cold_peers_lower_the_win_chance(self):
        model = PeerAdjusted(Flat(), center=0.5)
        hot = model.predict(*episode(5))["q0"]
        cold = model.predict(*episode(0))["q0"]
        self.assertGreater(1 - hot[0], 0.5)
        self.assertLess(1 - cold[0], 0.5)
        self.assertAlmostEqual(sum(hot), 1.0, places=9)
        self.assertAlmostEqual(hot[1] / (1 - hot[0]), 0.2 / 0.5, places=6)  # sizes unchanged

    def test_uses_only_peers_and_only_the_last_five_ticks(self):
        queries, history = episode(1)
        by_machine = {}
        for p in history:
            by_machine.setdefault((p.episode_id, p.machine_id), {})[p.tick] = p.payout
        self.assertEqual(peer_share(("e", "m0"), 20, by_machine), 1.0)  # own losses are ignored
        by_machine[("e", "m1")][10] = 0                                  # older than 5 ticks: ignored
        self.assertEqual(peer_share(("e", "m0"), 20, by_machine), 1.0)

    def test_no_future_ticks(self):
        model = PeerAdjusted(Flat(), center=0.5)
        queries, history = episode(1)
        base = model.predict(queries, history)
        changed = tuple(Play(p.episode_id, p.machine_id, p.tick, 0 if p.tick >= 19 and p.machine_id != "m0" else p.payout)
                        for p in history)
        self.assertNotEqual(model.predict(queries, changed)["q0"], base["q0"])   # tick 19 is used
        self.assertEqual(len(base), 3)


if __name__ == "__main__":
    unittest.main()
