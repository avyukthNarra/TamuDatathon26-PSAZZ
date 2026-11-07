import random
import unittest

from slot_forecasting.pair_mixer import CLASS_INDEX, PairMixer
from slot_forecasting.pair_mixer_episode import EpisodeMemoryMixer
from slot_forecasting.schema import PAYOUTS, Play, Query, ValidationError


def make_data(episode="e1", cutoff=64, machines=8, seed=12):
    rng = random.Random(seed)
    queries, history = [], []
    for index in range(machines):
        machine = f"m{index}"
        queries.append(Query(f"{episode}-{machine}-next", episode, machine,
                             ("steady", "balanced", "jackpot")[index % 3],
                             (index % 3) - 1, cutoff))
        for tick in range(cutoff):
            payout = rng.choice(PAYOUTS)
            history.append(Play(episode, machine, tick, payout))
    return tuple(queries), tuple(history)


class EpisodeMemoryMixerTests(unittest.TestCase):
    def test_gamma_zero_matches_pair_mixer_with_same_half_lives(self):
        queries, history = make_data()
        history = tuple(p for p in history if (p.machine_id, p.tick) not in {
            ("m0", 17), ("m2", 31), ("m5", 8),
        })
        half_lives = (12.0, 40.0, 1e9)
        for mode in ("parity", "full"):
            expected = PairMixer(half_life=half_lives, cell_mode=mode).fit(
                queries, history
            ).predict(queries, history)
            actual = EpisodeMemoryMixer(gamma=0.0, half_life=half_lives,
                                        cell_mode=mode).fit(
                queries, history
            ).predict(queries, history)
            for query_id in expected:
                for left, right in zip(expected[query_id], actual[query_id]):
                    self.assertAlmostEqual(left, right, places=14)

    def test_peer_history_before_cutoff_affects_pooled_speed_weights(self):
        queries, history = make_data(cutoff=64)
        baseline = PairMixer(half_life=(12.0, 40.0, 1e9)).fit(queries, history)
        pooled = EpisodeMemoryMixer(gamma=0.5, half_life=(12.0, 40.0, 1e9)).fit(
            queries, history
        )
        base_prediction = baseline.predict(queries, history)[queries[0].query_id]
        pooled_prediction = pooled.predict(queries, history)[queries[0].query_id]
        self.assertGreater(max(abs(a - b) for a, b in zip(base_prediction, pooled_prediction)), 1e-7)

    def test_current_tick_outcome_does_not_enter_trace_loss(self):
        queries, history = make_data(cutoff=40, machines=1)
        model = EpisodeMemoryMixer(half_life=(12.0, 40.0, 1e9)).fit(queries, history)
        own = [None] * 41
        for play in history:
            own[play.tick] = CLASS_INDEX[play.payout]
        first = model._trace(own, queries[0].machine_type, 40, {40})[40]
        changed = list(own)
        changed[40] = 0
        second = model._trace(changed, queries[0].machine_type, 40, {40})[40]
        self.assertEqual(first["logw"], second["logw"])
        self.assertEqual(first["cumulative_loss"], second["cumulative_loss"])
        self.assertEqual(first["win_preds"], second["win_preds"])

    def test_episode_pooling_isolated_and_query_order_independent(self):
        q1, h1 = make_data(episode="e1", cutoff=48, seed=4)
        q2, h2 = make_data(episode="e2", cutoff=48, seed=9)
        model = EpisodeMemoryMixer(gamma=0.5, half_life=(12.0, 40.0, 1e9)).fit(q1, h1)
        only_first = model.predict(q1, h1)
        both = model.predict(q1 + q2, h1 + h2)
        reversed_order = model.predict(tuple(reversed(q1 + q2)), tuple(reversed(h1 + h2)))
        for query in q1:
            self.assertEqual(only_first[query.query_id], both[query.query_id])
            self.assertEqual(both[query.query_id], reversed_order[query.query_id])

    def test_future_peer_rows_are_rejected_by_cutoff_validation(self):
        queries, history = make_data(cutoff=40)
        earlier = tuple(Query(q.query_id, q.episode_id, q.machine_id, q.machine_type,
                              q.signal, 32) for q in queries)
        model = EpisodeMemoryMixer(gamma=0.5).fit(queries, history)
        with self.assertRaises(ValidationError):
            model.predict(earlier, history)

    def test_later_peer_outcomes_do_not_change_earlier_machine_cutoff(self):
        queries, history = make_data(cutoff=64)
        mixed_cutoffs = tuple(
            Query(q.query_id, q.episode_id, q.machine_id, q.machine_type, q.signal,
                  32 if q.machine_id == "m0" else 64)
            for q in queries
        )
        history = tuple(p for p in history if p.machine_id != "m0" or p.tick < 32)
        train_q, train_h = make_data(episode="train", cutoff=64, seed=71)
        model = EpisodeMemoryMixer(gamma=0.5, half_life=(12.0, 40.0, 1e9)).fit(
            train_q, train_h
        )
        before = model.predict(mixed_cutoffs, history)[mixed_cutoffs[0].query_id]
        changed = tuple(
            Play(p.episode_id, p.machine_id, p.tick,
                 PAYOUTS[(PAYOUTS.index(p.payout) + 1) % len(PAYOUTS)])
            if p.machine_id != "m0" and p.tick >= 32 else p
            for p in history
        )
        after = model.predict(mixed_cutoffs, changed)[mixed_cutoffs[0].query_id]
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
