"""Run from mainFiles/:  python -m unittest discover -s tests -v"""
import random
import unittest

from slot_forecasting.pair_mixer import (CLASS_INDEX, PairMixer, adapt, choose_pairs, discover_pairs,
                                         size_modes_from_targets, win_prior_from_targets)
from slot_forecasting.schema import PAYOUTS, Play, Query, Target

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

    def test_size_experts_see_earlier_payout_size(self):
        """When a win's size copies the previous win's size, exact-payout size cells must pick it up."""
        rng = random.Random(3)
        query = Query("e1-m0-next", "e1", "e1-m0", "balanced", 0, 300)
        plays, last = [], 1
        for t in range(300):
            payout = 0 if t % 2 else PAYOUTS[last]  # wins on even ticks, losses on odd ticks
            if payout and rng.random() < 0.1:
                last = rng.randint(1, 4)
                payout = PAYOUTS[last]
            plays.append(Play("e1", "e1-m0", t, payout))
        history = tuple(plays)
        size_given_win = {}
        for mode in ("win", "class"):
            probs = PairMixer(size_cell=mode).fit((query,), history).predict((query,), history)[query.query_id]
            size_given_win[mode] = probs[last] / (1 - probs[0])
        self.assertGreater(size_given_win["class"], 0.6)
        self.assertGreater(size_given_win["class"], size_given_win["win"] + 0.1)

    def test_reads_two_size_modes_per_type(self):
        small, big = (0.5, 0.35, 0.13, 0.02), (0.21, 0.55, 0.21, 0.03)
        queries = tuple(Query(f"q{i}", "e1", f"m{i}", "balanced", 0, 10) for i in range(6))
        targets = [Target(f"q{i}", "e1", (0.4,) + tuple(0.6 * x for x in (small, big)[i % 2]), 0)
                   for i in range(6)]
        modes = size_modes_from_targets(targets, queries)
        for got, want in zip(modes["balanced"], (small, big)):
            self.assertTrue(all(abs(a - b) < 1e-3 for a, b in zip(got, want)))
        self.assertEqual(size_modes_from_targets(targets[:1], queries), {})  # one mode only -> fall back

    def test_two_mode_size_experts_learn_a_lag_rule(self):
        """Size mode = big iff the spin 2 back was a big win: the mode experts should call it at the cutoff."""
        small, big = (0.75, 0.2, 0.05, 0.0), (0.2, 0.3, 0.3, 0.2)
        rng = random.Random(11)
        payouts = []
        for t in range(400):
            mode = big if t >= 2 and payouts[t - 2] >= 5 else small
            payouts.append(rng.choices(PAYOUTS[1:], mode)[0] if rng.random() < 0.6 else 0)
        query = Query("e1-m0-next", "e1", "e1-m0", "steady", 0, 400)
        history = tuple(Play("e1", "e1-m0", t, v) for t, v in enumerate(payouts))
        model = PairMixer(size_modes={"steady": (small, big)}).fit((query,), history)
        probs = model.predict((query,), history)[query.query_id]
        want = big if payouts[398] >= 5 else small
        size = [p / (1 - probs[0]) for p in probs[1:]]
        self.assertLess(max(abs(a - b) for a, b in zip(size, want)), 0.01)  # free frequencies: 0.033

    def test_win_prior_levels_from_targets(self):
        wins = [0.02, 0.03, 0.05] * 10 + [0.88, 0.9, 0.95] * 10
        targets = [Target(f"q{i}", "e1", (1 - w, w, 0.0, 0.0, 0.0), 0) for i, w in enumerate(wins)]
        prior = win_prior_from_targets(targets)
        self.assertEqual([(c[0], c[1]) for c in prior[:2]], [(0.45, 0.03), (0.45, 0.9)])
        self.assertIsNone(win_prior_from_targets(targets[:30]))  # no live side -> no prior

    def test_win_prior_pulls_cells_to_the_nearest_level(self):
        """A cell with 3 wins in 3 spins should read near the live level, not the 0.5-ish base rate."""
        model = PairMixer(win_prior=[(0.45, 0.03, 6), (0.45, 0.93, 6), (0.1, 0.5, 2)])
        self.assertGreater(model._cell_rate(3.0, 3.0), 0.85)
        self.assertLess(model._cell_rate(0.0, 3.0), 0.1)
        self.assertAlmostEqual(model._cell_rate(0.0, 0.0), 0.45 * 0.03 + 0.45 * 0.93 + 0.1 * 0.5, places=9)

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
