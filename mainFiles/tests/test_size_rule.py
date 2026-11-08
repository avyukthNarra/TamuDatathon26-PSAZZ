"""Run from mainFiles/:  python -m unittest tests.test_size_rule -v"""
import random
import unittest
from unittest.mock import patch

from slot_forecasting.schema import PAYOUTS, Play, Query
from slot_forecasting import size_rule
from slot_forecasting.size_rule import SizeRule, select_lags, size_posterior

WIN = (1, 2, 5, 20)
# Atom 0 pays 1 more often. Separated enough that a short history identifies them.
ATOMS = (
    (0.72, 0.18, 0.08, 0.02),
    (0.12, 0.48, 0.28, 0.12),
)


def _draw(rng, weights):
    unit = rng.random()
    cumulative = 0.0
    for payout, weight in zip(WIN, weights):
        cumulative += weight
        if unit <= cumulative:
            return payout
    return WIN[-1]


def synthetic(n_machines=24, ticks=200, seed=1, lags=(1, 2, 4)):
    """Each machine uses one selected lag and polarity; wins follow that atom."""
    rng = random.Random(seed)
    queries, plays, truth = [], [], []
    for index in range(n_machines):
        lag = lags[index % len(lags)]
        polarity = index % 2
        machine = f"m{index}"
        queries.append(Query(f"q{index}", "e1", machine, "steady", 0, ticks))
        pay = []
        for tick in range(ticks):
            if rng.random() < 0.35:
                pay.append(0)
                continue
            if tick >= lag:
                feature = 1 if pay[tick - lag] >= 2 else 0
                weights = ATOMS[polarity ^ feature]
            else:
                weights = ATOMS[0]
            pay.append(_draw(rng, weights))
        plays.extend(Play("e1", machine, tick, value) for tick, value in enumerate(pay))
        truth.append((machine, lag, polarity))
    return tuple(queries), tuple(plays), truth


def synthetic_mixed(n_per_class=24, ticks=200, seed=11):
    """Balanced static and lag-driven size machines, with distinct train types."""
    rng = random.Random(seed)
    queries, plays, truth = [], [], []
    for index in range(2 * n_per_class):
        static = index < n_per_class
        type_name = "static" if static else "dynamic"
        local_index = index if static else index - n_per_class
        lag = (1, 2, 4)[local_index % 3]
        polarity = local_index % 2
        machine = f"mix{index}"
        query = Query(f"mq{index}", "mix-episode", machine, type_name, 0, ticks)
        queries.append(query)
        pay = []
        for tick in range(ticks):
            if rng.random() < 0.5:
                pay.append(0)
                continue
            if static or tick < lag:
                weights = ATOMS[polarity]
            else:
                feature = int(pay[tick - lag] >= 2)
                weights = ATOMS[polarity ^ feature]
            pay.append(_draw(rng, weights))
        plays.extend(Play("mix-episode", machine, tick, value)
                     for tick, value in enumerate(pay))
        truth.append((machine, "static" if static else "dynamic", lag, polarity, pay))
    return tuple(queries), tuple(plays), truth


class SizeRuleTest(unittest.TestCase):
    def test_lag_rule_keeps_mass_above_the_uniform_share(self):
        mass = {1: 10, 2: 10, 3: 1, 4: 10, 5: 1, 6: 1, 7: 1, 8: 1}
        self.assertEqual(select_lags(mass, tuple(range(1, 9))), (1, 2, 4))

    def test_flat_mass_falls_back_to_every_lag(self):
        mass = {lag: 1.0 for lag in range(1, 9)}
        self.assertEqual(select_lags(mass, tuple(range(1, 9))), tuple(range(1, 9)))

    def test_missing_lag_is_not_a_zero_payout(self):
        atoms = ((0.9, 0.05, 0.03, 0.02), (0.1, 0.4, 0.3, 0.2))
        _, missing = size_posterior([None, 1], 2, atoms, (1,))
        _, zero = size_posterior([0, 1], 2, atoms, (1,))
        self.assertAlmostEqual(missing[(1, 0)], 0.5)
        self.assertAlmostEqual(missing[(1, 1)], 0.5)
        self.assertGreater(zero[(1, 0)], 0.85)

    def test_ticks_at_or_after_the_cutoff_are_ignored(self):
        atoms = ((0.7, 0.2, 0.08, 0.02), (0.2, 0.5, 0.2, 0.1))
        pay = [1, 0, 1, 2, 5]
        mix, posterior = size_posterior(pay, 3, atoms, (1, 2))
        changed = [1, 0, 1, 20, 0]
        self.assertEqual(size_posterior(changed, 3, atoms, (1, 2)), (mix, posterior))

    def test_learn_is_deterministic_and_recovers_the_synthetic_atoms(self):
        queries, history, truth = synthetic()
        first = SizeRule.learn(queries, history)
        second = SizeRule.learn(queries, history)
        self.assertEqual(first.lags, second.lags)
        self.assertEqual(first.atoms, second.atoms)
        self.assertEqual(first.loglik, second.loglik)
        self.assertEqual(first.lags, (1, 2, 4))
        for got, want in zip(first.atoms["steady"], ATOMS):
            self.assertLess(max(abs(a - b) for a, b in zip(got, want)), 0.05)
        pay = [None] * queries[0].cutoff
        for play in history:
            if play.machine_id == truth[0][0]:
                pay[play.tick] = play.payout
        _mix, posterior = size_posterior(pay, len(pay), first.atoms["steady"], first.lags)
        self.assertEqual(max(posterior, key=posterior.get), truth[0][1:])

    def test_pair_prior_is_normalized_and_unseen_pairs_use_uniform_prior(self):
        queries, history, _truth = synthetic()
        rule = SizeRule.learn(queries, history)
        self.assertTrue(rule.pair_candidates)
        for prior in rule.pair_prior.values():
            self.assertAlmostEqual(sum(prior.values()), 1.0)
            self.assertTrue(all(value > 0.0 for value in prior.values()))
        unseen = next(pair for pair in ((a, b) for a in range(1, 9) for b in range(a + 1, 9))
                      if pair not in rule.pair_candidates)
        pay = [0, 1, 0, 2, 0, 5, 0, 1]
        uniform_mix, uniform_post = size_posterior(pay, len(pay), rule.atoms["steady"], rule.lags)
        fallback_mix, fallback_post = size_posterior(
            pay, len(pay), rule.atoms["steady"], rule.lags, rule.pair_prior.get(unseen))
        self.assertEqual((fallback_mix, fallback_post), (uniform_mix, uniform_post))

    def test_pair_prior_and_atoms_are_invariant_to_input_row_order(self):
        queries, history, _truth = synthetic()
        first = SizeRule.learn(queries, history)
        second = SizeRule.learn(tuple(reversed(queries)), tuple(reversed(history)))
        self.assertEqual(first.atoms, second.atoms)
        self.assertEqual(first.pair_candidates, second.pair_candidates)
        self.assertEqual(first.pair_prior, second.pair_prior)

    def test_pair_prior_refits_assignments_within_selected_pair_support(self):
        pairs = ((1, 2), (1, 3), (1, 4))
        rows = [("steady", [0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1], []) for _ in range(16)]
        global_choices = iter((pairs[0],) * 10 + (pairs[1],) * 4 + (pairs[2],) * 2)
        selected_choices = iter((pairs[0],) * 10 + (pairs[1],) * 6)

        def fake_best_pair(_pay, allowed):
            return next(global_choices if allowed == pairs else selected_choices)

        with patch.object(size_rule, "_PAIR_CANDIDATES", pairs), \
             patch.object(size_rule, "_best_pair", side_effect=fake_best_pair):
            priors, selected, _assignments = size_rule._learn_pair_prior(rows, {"steady": ATOMS}, (1,))

        self.assertEqual(selected, (pairs[0], pairs[1]))
        single = size_posterior(rows[0][1], len(rows[0][1]), ATOMS, (1,))[1]
        expected = {state: (0.5 + 6 * single[state]) / 7.0 for state in single}
        for state in expected:
            self.assertAlmostEqual(priors[pairs[1]][state], expected[state])

    def test_shifted_round_rediscovers_lags_outside_main_round(self):
        queries, history, truth = synthetic(n_machines=30, seed=19, lags=(3, 6, 8))
        rule = SizeRule.learn(queries, history)
        self.assertEqual(rule.lags, (3, 6, 8))
        for query, (_machine, lag, _polarity) in zip(queries, truth):
            pay = [None] * query.cutoff
            for play in history:
                if play.machine_id == query.machine_id:
                    pay[play.tick] = play.payout
            _mix, posterior = size_posterior(pay, len(pay), rule.atoms["steady"], rule.lags)
            self.assertIn(lag, {state[0] for state in posterior})

    def test_no_win_type_and_all_missing_history_are_deterministic(self):
        query = Query("q", "e", "m", "steady", 0, 20)
        no_wins = SizeRule.learn((query,), tuple(Play("e", "m", tick, 0) for tick in range(20)))
        self.assertEqual(no_wins.atoms["steady"], ((0.25,) * 4, (0.25,) * 4))
        self.assertEqual(no_wins.lags, tuple(range(1, 9)))
        self.assertEqual(no_wins.static_prior, 0.01)
        missing = SizeRule.learn((query,), ())
        self.assertEqual(missing.atoms["steady"], ((0.25,) * 4, (0.25,) * 4))
        self.assertEqual(missing.lags, tuple(range(1, 9)))

    def test_joint_static_dynamic_fit_recovers_both_regimes(self):
        queries, history, truth = synthetic_mixed()
        rule = SizeRule.learn(queries, history, seeds=(0, 1, 2), iters=120)
        self.assertGreater(rule.static_prior, 0.2)
        self.assertLess(rule.static_prior, 0.8)
        by_class = {"static": [], "dynamic": []}
        for machine, regime, _lag, _polarity, pay in truth:
            query = next(query for query in queries if query.machine_id == machine)
            selected_pair = size_rule._best_pair(pay, rule.pair_candidates)
            _, states, static_ll, dynamic_ll = size_rule.hybrid_posterior(
                pay, len(pay), rule.atoms[query.machine_type], rule.lags,
                rule.static_prior, rule.pair_prior.get(selected_pair))
            by_class[regime].append(static_ll - dynamic_ll)
        self.assertGreater(sum(by_class["static"]) / len(by_class["static"]), 0.0)
        self.assertLess(sum(by_class["dynamic"]) / len(by_class["dynamic"]), 0.0)

    def test_prequential_forecasts_ignore_current_and_future_payouts(self):
        queries, history, truth = synthetic_mixed(n_per_class=12, ticks=140, seed=17)
        rule = SizeRule.learn(queries, history, seeds=(0, 1), iters=80)
        _machine, regime, _lag, _polarity, pay = truth[-1]
        before = list(rule.prequential(pay, regime))
        changed = list(pay)
        changed[90:] = [20 if value != 20 else 1 for value in changed[90:]]
        after = list(rule.prequential(changed, regime))
        for (tick, mix_a), (other_tick, mix_b) in zip(before[:91], after[:91]):
            self.assertEqual(tick, other_tick)
            self.assertEqual(mix_a, mix_b)

    def test_prequential_matches_prefix_batch_prediction_as_pair_key_changes(self):
        queries, history, truth = synthetic_mixed(n_per_class=12, ticks=180, seed=23)
        rule = SizeRule.learn(queries, history, seeds=(0, 1), iters=80)
        _machine, regime, _lag, _polarity, pay = truth[-1]
        streamed = list(rule.prequential(pay, regime))
        prefix_pairs = set()
        for tick, stream_mix in streamed:
            prefix = pay[:tick]
            prefix_pairs.add(size_rule._best_pair(prefix, rule.pair_candidates))
            batch_mix = rule.predict_mix(pay, regime, tick)
            for got, want in zip(stream_mix, batch_mix):
                self.assertAlmostEqual(got, want, places=12)
        self.assertGreater(len(prefix_pairs), 1)

    def test_prequential_fit_is_deterministic_under_row_reordering(self):
        queries, history, truth = synthetic_mixed(n_per_class=10, ticks=120, seed=31)
        first = SizeRule.learn(queries, history, seeds=(0, 1), iters=80)
        second = SizeRule.learn(tuple(reversed(queries)), tuple(reversed(history)),
                                seeds=(0, 1), iters=80)
        self.assertEqual(first.atoms, second.atoms)
        self.assertEqual(first.static_prior, second.static_prior)
        self.assertEqual(first.pair_candidates, second.pair_candidates)
        self.assertEqual(first.pair_prior, second.pair_prior)
        machine, regime, _lag, _polarity, pay = truth[-1]
        self.assertEqual(list(first.prequential(pay, regime)),
                         list(second.prequential(pay, regime)))

    def test_apply_keeps_p0_and_floors(self):
        eps = 1e-8
        atoms = ((0.7, 0.2, 0.1 - eps, eps), (0.2, 0.5, 0.3 - eps, eps))
        rule = SizeRule({"steady": atoms}, (1,), {1: 1.0})
        query = Query("q", "e", "m", "steady", 0, 1)
        history = (Play("e", "m", 0, 0),)
        out = rule.apply({"q": (0.4, 0.15, 0.15, 0.15, 0.15)}, (query,), history)
        self.assertAlmostEqual(sum(out["q"]), 1.0)
        # No win yet and the lag bit is 0, so the mix is the average of the atoms.
        # p(20) is under the floor; after renormalizing, the classes that were
        # already above the floor keep their ratios, including p_0 to p_1.
        self.assertAlmostEqual(out["q"][1] / out["q"][0], (0.6 * 0.45) / 0.4)
        self.assertLess(out["q"][4], 1e-4)
        self.assertGreater(out["q"][4], 5e-5)
        bare = SizeRule({}, (1,), {})
        flat = bare.apply({"q": (0.2, 0.2, 0.2, 0.2, 0.2)}, (query,), history)
        self.assertEqual(len(flat["q"]), len(PAYOUTS))
        self.assertAlmostEqual(flat["q"][0], 0.2)
        self.assertTrue(all(abs(value - 0.2) < 1e-12 for value in flat["q"]))


if __name__ == "__main__":
    unittest.main()
