"""Small CPU-only checks for the E7 contextual win head."""
import unittest

import numpy as np

from mainFiles.slot_forecasting.neural_context import NeuralContextForecaster
from mainFiles.slot_forecasting.schema import Play, Query


def synthetic_training_data(seed=19, episodes=10, machines_per_episode=2, ticks=96):
    rng = np.random.default_rng(seed)
    queries, history, labels = [], [], {}
    for episode_index in range(episodes):
        episode = f'ep{episode_index:03d}'
        for machine_index in range(machines_per_episode):
            machine = f'{episode}-m{machine_index}'
            outcomes = np.zeros(ticks + 1, dtype=int)
            outcomes[:2] = rng.integers(0, 2, size=2)
            for tick in range(2, ticks + 1):
                pair = outcomes[tick - 1] * outcomes[tick - 2]
                p = 1 / (1 + np.exp(-(-0.2 + 1.1 * pair)))
                outcomes[tick] = int(rng.random() < p)
            history.extend(Play(episode, machine, tick, int(outcomes[tick]))
                           for tick in range(ticks))
            query_id = f'q-{machine}'
            queries.append(Query(query_id, episode, machine, 'balanced', 0, ticks))
            labels[query_id] = 1 + int(rng.integers(0, 2))
    return queries, history, labels


class NeuralContextForecasterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = synthetic_training_data()
        cls.model = NeuralContextForecaster(
            max_epochs=4, patience=2, examples_per_machine=8,
            batch_size=64, min_context=16, seed=77,
        ).fit(*cls.training)

    def test_prefix_statistics_and_prediction_ignore_current_and_future_rows(self):
        _, history, _ = self.training
        query = self.training[0][0]
        seq = {p.tick: p.payout for p in history if p.machine_id == query.machine_id}
        cutoff = query.cutoff
        full_stats, full_count = self.model.context_statistics(seq, cutoff)
        clean = {tick: payout for tick, payout in seq.items() if tick < cutoff}
        clean_stats, clean_count = self.model.context_statistics(clean, cutoff)
        np.testing.assert_array_equal(full_stats, clean_stats)
        self.assertEqual(full_count, clean_count)
        self.assertEqual(self.model.win_probability(seq, cutoff),
                         self.model.win_probability(clean, cutoff))

    def test_training_prefix_summary_ignores_current_and_future_outcomes(self):
        rng = np.random.default_rng(101)
        original = {tick: int(rng.integers(0, 2)) for tick in range(81)}
        changed = dict(original)
        changed[70] = 1 - changed[70]
        changed.update({tick: int(rng.integers(0, 2)) for tick in range(81, 111)})
        Z1, y1, ticks1 = self.model._sequence_rows(original)
        Z2, y2, ticks2 = self.model._sequence_rows(changed)
        target_tick = 70
        index1 = int(np.flatnonzero(ticks1 == target_tick)[0])
        index2 = int(np.flatnonzero(ticks2 == target_tick)[0])
        prefix1 = self.model._prefix_summaries(
            Z1, y1, ticks1, terminal_query_tick=target_tick
        )[index1]
        prefix2 = self.model._prefix_summaries(
            Z2, y2, ticks2, terminal_query_tick=target_tick
        )[index2]
        np.testing.assert_array_equal(prefix1, prefix2)

    def test_machine_identity_is_not_an_input(self):
        _, history, _ = self.training
        query = self.training[0][0]
        seq = {p.tick: p.payout for p in history if p.machine_id == query.machine_id}
        renamed_query = Query(query.query_id + '-renamed', query.episode_id,
                              query.machine_id + '-renamed', query.machine_type,
                              query.signal, query.cutoff)
        renamed_history = [Play(play.episode_id, renamed_query.machine_id,
                                play.tick, play.payout)
                           for play in history if play.machine_id == query.machine_id]
        original = self.model.predict([query], [play for play in history
                                               if play.machine_id == query.machine_id])
        renamed = self.model.predict([renamed_query], renamed_history)
        np.testing.assert_allclose(original[query.query_id],
                                   renamed[renamed_query.query_id], rtol=0, atol=0)

    def test_training_is_deterministic_for_a_fixed_seed(self):
        second = NeuralContextForecaster(
            max_epochs=4, patience=2, examples_per_machine=8,
            batch_size=64, min_context=16, seed=77,
        ).fit(*self.training)
        self.assertEqual(self.model.training_diagnostics['best_epoch'],
                         second.training_diagnostics['best_epoch'])
        self.assertEqual(self.model.training_diagnostics['internal_train_holdout_loss'],
                         second.training_diagnostics['internal_train_holdout_loss'])
        query = self.training[0][0]
        seq = {p.tick: p.payout for p in self.training[1]
               if p.machine_id == query.machine_id}
        self.assertEqual(self.model.win_probability(seq, query.cutoff),
                         second.win_probability(seq, query.cutoff))

    def test_predict_returns_valid_five_probability_distribution(self):
        query = self.training[0][0]
        history = [p for p in self.training[1]
                   if p.machine_id == query.machine_id and p.tick < query.cutoff]
        probabilities = self.model.predict([query], history)[query.query_id]
        self.assertEqual(len(probabilities), 5)
        self.assertTrue(all(np.isfinite(p) and 0.0 <= p <= 1.0 for p in probabilities))
        self.assertAlmostEqual(sum(probabilities), 1.0, places=12)


if __name__ == '__main__':
    unittest.main()
