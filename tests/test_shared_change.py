import unittest
import random

from slot_forecasting.shared_change import estimate_shared_change, SharedChangeForecaster
from slot_forecasting.pair_logistic import PairLogisticForecaster
from slot_forecasting.schema import Play, Query


def alternating_streams(boundary=48, stop=96):
    ticks = range(stop)
    return {
        "up": {t: (1 if t < boundary else 0) for t in ticks},
        "down": {t: (0 if t < boundary else 1) for t in ticks},
    }


class SharedChangeTests(unittest.TestCase):
    def test_finds_common_boundary_with_opposite_machine_directions(self):
        result = estimate_shared_change(alternating_streams(), cutoff=96)
        self.assertIsNotNone(result.boundary)
        self.assertLessEqual(abs(result.boundary - 48), 4)
        self.assertGreater(result.raw_gain, 0)
        self.assertEqual(result.directions, {"down": 1, "up": -1})
        self.assertEqual(result.observations["up"], (result.boundary, 96 - result.boundary))

    def test_future_mutations_do_not_change_result(self):
        original = alternating_streams()
        mutated = {machine: dict(obs) for machine, obs in original.items()}
        for observations in mutated.values():
            for tick in range(96, 130):
                observations[tick] = 1000 if tick % 2 else -10
        a = estimate_shared_change(original, cutoff=96)
        b = estimate_shared_change(mutated, cutoff=96)
        self.assertEqual(a, b)

    def test_sparse_ticks_are_counted_as_observations_not_filled(self):
        streams = {
            "a": {t: (1 if t < 48 else 0) for t in range(0, 96, 2)},
            "b": {t: (0 if t < 48 else 1) for t in range(1, 96, 2)},
        }
        result = estimate_shared_change(streams, cutoff=96)
        self.assertIsNotNone(result.boundary)
        self.assertLessEqual(abs(result.boundary - 48), 4)
        self.assertEqual(result.observations["a"], (24, 24))
        self.assertEqual(result.observations["b"], (24, 24))

    def test_constant_streams_do_not_pass_penalty(self):
        streams = {"a": {t: 0 for t in range(96)}, "b": {t: 1 for t in range(96)}}
        result = estimate_shared_change(streams, cutoff=96)
        self.assertIsNone(result.boundary)
        self.assertLess(result.evidence, 0)
        self.assertEqual(result.raw_gain, 0)

    def test_too_short_streams_have_no_candidate(self):
        streams = {"a": {t: t % 2 for t in range(20)}, "b": {t: t % 2 for t in range(20)}}
        result = estimate_shared_change(streams, cutoff=20, min_side=16)
        self.assertIsNone(result.boundary)
        self.assertEqual(result.candidates, 0)

    def test_includes_last_boundary_with_exact_minimum_support(self):
        streams = {
            "up": {t: (1 if t < 16 else 0) for t in range(32)},
            "down": {t: (0 if t < 16 else 1) for t in range(32)},
        }
        result = estimate_shared_change(streams, cutoff=32, min_side=16, step=4)
        self.assertEqual(result.candidates, 1)
        self.assertEqual(result.boundary, 16)

    def test_one_other_stream_can_support_the_episode_boundary(self):
        queries = (
            Query("q-target", "episode", "target", "steady", 0, 96),
            Query("q-other", "episode", "other", "steady", 0, 96),
        )
        history = tuple(
            Play("episode", machine, tick, payout)
            for machine, payout_fn in (
                ("target", lambda tick: 0),
                ("other", lambda tick: 1 if tick < 48 else 0),
            )
            for tick in range(96)
            for payout in (payout_fn(tick),)
        )
        labels = {"q-target": 0, "q-other": 1}
        model = SharedChangeForecaster(
            evidence_threshold=0, min_side=16, change_step=4,
            lags=2, pairs=False, min_rows=12,
        ).fit(queries, history, labels)
        model.predict(queries, history)
        diagnostic = model.change_diagnostics[("episode", 96)]
        self.assertEqual(diagnostic["boundary"], 48)
        self.assertGreater(diagnostic["evidence"], 0)

    def test_before_weight_one_matches_e1_exactly(self):
        queries = tuple(
            Query(f"q-{m}", "episode", f"m{m}", "steady", 0, 96)
            for m in range(8)
        )
        history = tuple(
            Play("episode", f"m{m}", tick,
                 1 if (tick * (m + 3) + m) % 7 in (0, 1, 2) else 0)
            for m in range(8) for tick in range(96)
        )
        labels = {query.query_id: (i % 3) for i, query in enumerate(queries)}
        options = dict(lags=4, pairs=True, C=3.0, half_life=32,
                       min_rows=12, base_half_life=24)
        baseline = PairLogisticForecaster(**options).fit(queries, history, labels)
        shared = SharedChangeForecaster(
            **options, before_weight=1.0, evidence_threshold=-1e6,
        ).fit(queries, history, labels)
        expected = baseline.predict(queries, history)
        actual = shared.predict(queries, history)
        self.assertEqual(actual, expected)

    def test_identical_targets_use_their_own_episodes_neighbor_boundary(self):
        target_wins = set(random.Random(29).sample(range(96), 43))

        def positions(start, length, wins, seed):
            return {start + offset for offset in random.Random(seed).sample(range(length), wins)}

        # Equal total wins, but opposing shifts at different episode boundaries.
        neighbor_a = positions(0, 32, 12, 1) | positions(32, 64, 46, 2)
        neighbor_b = positions(0, 64, 46, 2) | positions(64, 32, 12, 1)
        queries = (
            Query("target-a", "episode-a", "target-a", "steady", 0, 96),
            Query("neighbor-a", "episode-a", "neighbor-a", "balanced", 1, 96),
            Query("target-b", "episode-b", "target-b", "steady", 0, 96),
            Query("neighbor-b", "episode-b", "neighbor-b", "balanced", 1, 96),
        )
        history = tuple(
            Play(episode, machine, tick, int(tick in wins))
            for episode, machine, wins in (
                ("episode-a", "target-a", target_wins),
                ("episode-a", "neighbor-a", neighbor_a),
                ("episode-b", "target-b", target_wins),
                ("episode-b", "neighbor-b", neighbor_b),
            )
            for tick in range(96)
        )
        labels = {query.query_id: i % 2 for i, query in enumerate(queries)}
        model = SharedChangeForecaster(
            C=3, lags=4, pairs=True, half_life=32, min_rows=12,
            before_weight=0.1, evidence_threshold=0, min_side=16, change_step=4,
        ).fit(queries, history, labels)

        batch = model.predict(queries, history)
        self.assertNotEqual(
            model.change_diagnostics[("episode-a", 96)]["boundary"],
            model.change_diagnostics[("episode-b", 96)]["boundary"],
        )
        separate_a = model.predict(queries[:2], history[:192])
        separate_b = model.predict(queries[2:], history[192:])
        self.assertEqual(batch["target-a"], separate_a["target-a"])
        self.assertEqual(batch["target-b"], separate_b["target-b"])
        self.assertNotEqual(batch["target-a"], batch["target-b"])

    def test_machine_scope_does_not_use_neighbor_evidence(self):
        queries = (
            Query("q-target", "episode", "target", "steady", 0, 96),
            Query("q-neighbor", "episode", "neighbor", "balanced", 1, 96),
        )
        history = tuple(
            Play("episode", machine, tick, payout)
            for machine, payout_fn in (
                ("target", lambda tick: 0),
                ("neighbor", lambda tick: 1 if tick < 48 else 0),
            )
            for tick in range(96)
            for payout in (payout_fn(tick),)
        )
        labels = {"q-target": 0, "q-neighbor": 1}
        model = SharedChangeForecaster(
            C=3, lags=2, pairs=False, min_rows=12,
            evidence_threshold=0, min_side=16, change_scope="machine",
        ).fit(queries, history, labels)
        model.predict(queries, history)
        target = model.change_diagnostics[("episode", 96, "target")]
        neighbor = model.change_diagnostics[("episode", 96, "neighbor")]
        self.assertIsNone(target["boundary"])
        self.assertEqual(neighbor["boundary"], 48)


if __name__ == "__main__":
    unittest.main()
