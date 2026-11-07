"""Run from mainFiles/:  python -m unittest discover -s tests -v"""
import random
import unittest

from slot_forecasting.pair_mixer import CLASS_INDEX, PairMixer, adapt, choose_pairs, discover_pairs
from slot_forecasting.schema import PAYOUTS, Play, Query

MACHINES = ("e1-m0", "e1-m1", "e1-m2")


def make_data(seed=1, cutoff=60, skip=(), rule=None, lags=(1, 3)):
    rng = random.Random(seed)
    queries = tuple(Query(f"{m}-next", "e1", m, ("steady", "balanced", "jackpot")[i], (-1, 0, 1)[i], cutoff)
                    for i, m in enumerate(MACHINES))
    plays = []
    for m in MACHINES:
        won = {}
        for t in range(cutoff):
            if rule and t >= max(lags):
                win = (won[t - lags[0]] == won[t - lags[1]]) if rng.random() < 0.95 else rng.random() < 0.5
            else:
                win = rng.random() < 0.5
            won[t] = win
            if (m, t) not in skip:
                plays.append(Play("e1", m, t, rng.choice(PAYOUTS[1:]) if win else 0))
    return queries, tuple(plays)


def truncate(queries, history, cutoff):
    cut = tuple(Query(q.query_id, q.episode_id, q.machine_id, q.machine_type, q.signal, cutoff) for q in queries)
    return cut, tuple(p for p in history if p.tick < cutoff)


class PairMixerTest(unittest.TestCase):
    def setUp(self):
        self.queries, self.history = make_data()
        self.model = PairMixer().fit(self.queries, self.history)

    def test_future_rows_never_change_a_prediction(self):
        """Mutating or deleting ANY machine's rows at tick >= t leaves predictions at t unchanged."""
        for t in (10, 33, 59):
            q, h = truncate(self.queries, self.history, t)
            base = self.model.predict(q, h)
            rng = random.Random(t)
            mutated = tuple(Play(p.episode_id, p.machine_id, p.tick, rng.choice(PAYOUTS)) if p.tick >= t else p
                            for p in self.history)
            q2, h2 = truncate(self.queries, mutated, t)
            self.assertEqual(self.model.predict(q2, h2), base)

    def test_walk_emits_before_absorbing(self):
        """The forecast emitted for tick t is identical whatever tick t (and later) turn out to be."""
        own = [CLASS_INDEX[p.payout] for p in self.history if p.machine_id == "e1-m0"]
        seen = {}
        self.model.walk(own, "steady", 40, lambda t, p: seen.__setitem__(t, p))
        changed = own[:30] + [4] * (len(own) - 30)
        again = {}
        self.model.walk(changed, "steady", 40, lambda t, p: again.__setitem__(t, p))
        for t in range(31):
            self.assertEqual(seen[t], again[t], f"tick {t}")

    def test_deterministic(self):
        first = self.model.predict(self.queries, self.history)
        second = PairMixer().fit(self.queries, self.history).predict(self.queries, self.history)
        worst = max(abs(a - b) for k in first for a, b in zip(first[k], second[k]))
        self.assertLess(worst, 1e-12)

    def test_missing_tick_is_not_a_zero_payout(self):
        missing = make_data(skip={("e1-m0", 58), ("e1-m0", 56)})
        zeroed = tuple(Play(p.episode_id, p.machine_id, p.tick, 0) if (p.machine_id, p.tick) in
                       {("e1-m0", 58), ("e1-m0", 56)} else p for p in self.history)
        a = self.model.predict(*missing)["e1-m0-next"]
        b = self.model.predict(self.queries, zeroed)["e1-m0-next"]
        self.assertNotEqual(a, b)

    def test_probabilities_valid(self):
        for probs in self.model.predict(self.queries, self.history).values():
            self.assertAlmostEqual(sum(probs), 1.0, places=9)
            self.assertTrue(all(0 < p < 1 for p in probs))

    def test_learns_a_match_rule(self):
        """On data where a win means lag1 == lag3, the mixer should become confident."""
        queries, history = make_data(seed=5, cutoff=200, rule=True)
        model = PairMixer().fit(queries, history)
        own = {p.tick: p.payout > 0 for p in history if p.machine_id == "e1-m0"}
        predicted_win = 1 - model.predict(queries, history)["e1-m0-next"][0]
        expected = own[199] == own[197]
        self.assertGreater(predicted_win if expected else 1 - predicted_win, 0.8)
        self.assertEqual(discover_pairs(queries, history, top=1)[0][0], (1, 3))

    def test_adapts_to_new_lag_pairs(self):
        """Round two may use different pairs: rediscovery must find them and the mixer must use them."""
        queries, history = make_data(seed=7, cutoff=220, rule=True, lags=(2, 6))
        self.assertEqual(choose_pairs(queries, history), ((2, 6),))
        labels = {q.query_id: 0 for q in queries}
        model = adapt(queries, history, labels, log=lambda *_: None)
        self.assertEqual(model.pairs, ((2, 6),))
        won = {p.tick: p.payout > 0 for p in history if p.machine_id == "e1-m1"}
        predicted_win = 1 - model.predict(queries, history)["e1-m1-next"][0]
        self.assertGreater(predicted_win if won[218] == won[214] else 1 - predicted_win, 0.8)


if __name__ == "__main__":
    unittest.main()
