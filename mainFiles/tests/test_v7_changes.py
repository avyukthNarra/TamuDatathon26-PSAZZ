"""Run from mainFiles/:  python -m unittest tests.test_v7_changes -v"""
import unittest

from slot_forecasting.episode_flip import detect_episode, widest_gap
from slot_forecasting.pair_mixer import (
    ALL_PAIRS, DEFAULT_PAIRS, MODE_SIZE_KEYS, PairMixer, choose_pairs_robust,
    discover_size_keys, swap_parity_counts, two_level_fit, win_prior_by_type_from_targets,
)
from slot_forecasting.schema import Play, Query, Target


def _probs(p0):
    rest = (1.0 - p0) / 4.0
    return (p0, rest, rest, rest, rest)


class SwapAndGateTests(unittest.TestCase):
    def test_parity_counts_exchange(self):
        tab = {(True,): [3.0, 4.0], (False,): [1.0, 5.0], "sym": [2.0, 9.0], (True, False): [8.0, 8.0]}
        swap_parity_counts([tab])
        self.assertEqual(tab[(False,)], [3.0, 4.0])
        self.assertEqual(tab[(True,)], [1.0, 5.0])
        self.assertEqual(tab["sym"][0], 7.0)
        self.assertEqual(tab[(True, False)], [8.0, 8.0])

    def test_widest_gap_midpoint(self):
        width, mid, left, right = widest_gap([-32.5, 237.9])
        self.assertAlmostEqual(mid, 102.7)
        self.assertAlmostEqual(width, 270.4)
        self.assertEqual((left, right), (-32.5, 237.9))
        _width, mid_with_zero, _left, _right = widest_gap([-32.5, 0.0, 237.9])
        self.assertGreater(mid_with_zero, -32.5)
        self.assertLess(mid_with_zero, 237.9)

    def test_detector_finds_synthetic_flip(self):
        cutoff = 160
        flip_at = 80
        pair = (1, 3)
        rows = []
        for index in range(4):
            own = [0, 1, 0]
            for t in range(3, cutoff):
                agree = (own[t - pair[0]] > 0) == (own[t - pair[1]] > 0)
                live = agree if t < flip_at else (not agree)
                own.append(1 if live else 0)
            query = Query(f"q{index}", "ep", f"m{index}", "steady", 0, cutoff)
            rows.append((query, own + [None]))
        found = detect_episode(rows, (pair,))
        self.assertGreater(found["logbf"], 20.0)
        self.assertIsNotNone(found["map_tau"])
        self.assertLess(abs(found["map_tau"] - flip_at), 12)

    def test_swap_tick_changes_forecast_only_when_set(self):
        cutoff = 80
        pair = (1, 3)
        plays = []
        for index in range(3):
            own = [0, 1, 0]
            for t in range(3, cutoff):
                agree = (own[t - pair[0]] > 0) == (own[t - pair[1]] > 0)
                live = agree if t < 40 else (not agree)
                own.append(1 if live else 0)
            for t, bit in enumerate(own):
                plays.append(Play("ep", f"m{index}", t, 1 if bit else 0))
        queries = tuple(Query(f"q{i}", "ep", f"m{i}", "steady", 0, cutoff) for i in range(3))
        history = tuple(plays)
        model = PairMixer(pairs=(pair,), half_life=400.0, k_win=0.5, eta=6.0, cell_mode="parity").fit(queries, history)
        base = model.predict(queries, history)
        same = model.predict(queries, history, swap_at={})
        moved = model.predict(queries, history, swap_at={"ep": 40})
        self.assertEqual(base["q0"], same["q0"])
        self.assertNotEqual(base["q0"], moved["q0"])


class Round2Tests(unittest.TestCase):
    def test_empty_search_is_not_the_main_pairs(self):
        queries = tuple(Query(f"q{i}", "ep", f"m{i}", "steady", 0, 15) for i in range(4))
        history = tuple(Play("ep", f"m{i}", t, 0) for i in range(4) for t in range(15))
        self.assertEqual(choose_pairs_robust(queries, history), ALL_PAIRS)
        self.assertNotEqual(choose_pairs_robust(queries, history), DEFAULT_PAIRS)
        self.assertEqual(discover_size_keys(queries, history), MODE_SIZE_KEYS)
        self.assertEqual(discover_size_keys(queries, history, on_empty=((),)), ((),))

    def test_two_means_does_not_copy_another_types_live_level(self):
        queries = []
        targets = []
        for i in range(25):
            queries.append(Query(f"s{i}", "ep", f"ms{i}", "steady", 0, 20))
            queries.append(Query(f"j{i}", "ep", f"mj{i}", "jackpot", 0, 20))
            targets.append(Target(f"s{i}", "ep", _probs(0.90 if i < 13 else 0.10), 0))
            targets.append(Target(f"j{i}", "ep", _probs(0.95 if i % 2 == 0 else 0.65), 0))
        for i in range(25, 50):
            queries.append(Query(f"s{i}", "ep", f"ms{i}", "steady", 0, 20))
            queries.append(Query(f"j{i}", "ep", f"mj{i}", "jackpot", 0, 20))
            targets.append(Target(f"s{i}", "ep", _probs(0.10), 1))
            targets.append(Target(f"j{i}", "ep", _probs(0.65), 0))
        stock = win_prior_by_type_from_targets(tuple(targets), tuple(queries), two_means=False)
        hybrid = win_prior_by_type_from_targets(tuple(targets), tuple(queries), two_means=True)
        stock_live = max(component[1] for component in stock["jackpot"])
        hybrid_live = max(component[1] for component in hybrid["jackpot"])
        self.assertGreater(stock_live, 0.8)
        self.assertLess(hybrid_live, 0.5)
        fitted = two_level_fit([0.05, 0.35] * 20)
        self.assertIsNotNone(fitted)
        self.assertLess(max(component[1] for component in fitted), 0.5)


if __name__ == "__main__":
    unittest.main()
