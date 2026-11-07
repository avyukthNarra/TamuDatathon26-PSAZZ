"""Focused synthetic checks for train-only lag-pair selection."""
import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from mainFiles.slot_forecasting.pair_logistic import _lag_vector
from mainFiles.slot_forecasting.pair_selection import TrainSelectedPairForecaster
from mainFiles.slot_forecasting.schema import Play, Query


def synthetic_data(seed=23, machines=20, ticks=800):
    rng = np.random.default_rng(seed)
    queries, history, labels = [], [], {}
    signs = np.array([1 if i < machines // 2 else -1 for i in range(machines)])
    rng.shuffle(signs)
    for m, sign in enumerate(signs):
        episode, machine = f'ep{m:03d}', f'm{m:03d}'
        y = np.zeros(ticks + 1, dtype=int)
        y[:2] = rng.integers(0, 2, size=2)
        for t in range(2, ticks + 1):
            pair = y[t - 1] * y[t - 2]
            p = 1 / (1 + np.exp(-(-0.35 + sign * 3.3 * pair)))
            y[t] = int(rng.random() < p)
        history.extend(Play(episode, machine, t, int(y[t])) for t in range(ticks))
        query_id = f'q{m:03d}'
        queries.append(Query(query_id, episode, machine, 'balanced', m % 2, ticks))
        labels[query_id] = int(y[ticks])
    return queries, history, labels


class TrainSelectedPairForecasterTests(unittest.TestCase):
    def test_relevant_pair_ranks_above_irrelevant_with_opposite_machine_signs(self):
        queries, history, labels = synthetic_data()
        model = TrainSelectedPairForecaster(top_k=5, min_train=100, min_test=20)
        model.fit(queries, history, labels)
        means = model.ranking_diagnostics['mean_brier_delta']
        self.assertEqual(model.ranking_diagnostics['machines_eligible'], len(queries))
        self.assertIn((0, 1), model.selected_pairs)
        self.assertLess(means[(0, 1)], means[(0, 6)])

        coefficients = []
        for query in queries:
            seq = {p.tick: p.payout for p in history if p.machine_id == query.machine_id}
            rows, outcomes = [], []
            for tick, payout in seq.items():
                lags = _lag_vector(seq, tick, model.lags)
                if lags is None:
                    continue
                rows.append(model._interaction_features(lags, model.selected_pairs))
                outcomes.append(int(payout > 0))
            fitted = LogisticRegression(C=model.C, max_iter=1000).fit(rows, outcomes)
            pair_feature = model.lags + model.selected_pairs.index((0, 1))
            coefficients.append(np.sign(fitted.coef_[0, pair_feature]))
        self.assertLess(min(coefficients), 0)
        self.assertGreater(max(coefficients), 0)

    def test_win_probability_ignores_cutoff_and_future_rows(self):
        model = TrainSelectedPairForecaster(top_k=1)
        model.selected_pairs = ((0, 1),)
        rng = np.random.default_rng(7)
        seq = {tick: int(rng.integers(0, 2)) for tick in range(100)}
        cutoff = 80
        short = {tick: payout for tick, payout in seq.items() if tick < cutoff}
        self.assertEqual(model.win_probability(seq, cutoff),
                         model.win_probability(short, cutoff))

    def test_full_pair_set_can_be_used_above_observed_support_threshold(self):
        model = TrainSelectedPairForecaster(top_k=1, lags=4,
                                            full_pairs_above_rows=64)
        model.selected_pairs = ((0, 1),)
        self.assertEqual(model._pairs_for_rows(63), ((0, 1),))
        self.assertEqual(model._pairs_for_rows(64),
                         ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))

    def test_ranking_ignores_labels_signal_and_machine_type(self):
        queries, history, labels = synthetic_data(seed=99, machines=8, ticks=500)
        changed = [Query(q.query_id, q.episode_id, q.machine_id, 'jackpot',
                         -q.signal, q.cutoff) for q in queries]
        altered_labels = {query_id: 0 for query_id in labels}
        first = TrainSelectedPairForecaster(top_k=4, min_train=100, min_test=20)
        second = TrainSelectedPairForecaster(top_k=4, min_train=100, min_test=20)
        first.fit(queries, history, labels)
        second.fit(changed, history, altered_labels)
        self.assertEqual(first.selected_pairs, second.selected_pairs)
        self.assertEqual(first.ranking_diagnostics['mean_brier_delta'],
                         second.ranking_diagnostics['mean_brier_delta'])


if __name__ == '__main__':
    unittest.main()
