"""Run from mainFiles/:  python -m unittest discover -s tests -v"""
import random
import unittest

from slot_forecasting.pair_mixer import (CLASS_INDEX, MODE_SIZE_KEYS, PairMixer, adapt, choose_pairs,
                                         discover_pairs, episode_kfolds, split_episodes, win_prior_from_train,
                                         discover_size_keys, size_mode_mixture, size_modes_from_targets,
                                         win_prior_by_type_from_targets,
                                         win_prior_from_targets)
from slot_forecasting.rule_hmm import HiddenRuleHMM, WithRuleHMM, levels_from_prior
from slot_forecasting.schema import PAYOUTS, Play, Query, Target, ValidationError

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
        for got, want in zip(modes["balanced"]["modes"], (small, big)):
            self.assertTrue(all(abs(a - b) < 1e-3 for a, b in zip(got, want)))
        single = size_modes_from_targets(targets[:1], queries)
        self.assertEqual(len(single["balanced"]["modes"]), 1)

    def test_two_size_modes_equal_the_sigmoid_llr(self):
        a, b = (0.6, 0.3, 0.08, 0.02), (0.2, 0.3, 0.3, 0.2)
        counts = (3.0, 2.0, 1.0, 0.0)
        llr = sum(c * __import__("math").log(y / x) for c, x, y in zip(counts, a, b))
        big = 1 / (1 + __import__("math").exp(-llr))
        got = size_mode_mixture(counts, (a, b), (0.5, 0.5))
        self.assertTrue(all(abs(x - ((1-big)*u + big*v)) < 1e-12
                            for x, u, v in zip(got, a, b)))

    def test_discovers_size_lags_that_moved(self):
        """Round two may move the size rule: big mode iff lag 5 or lag 7 was a big win -> keys (5,), (7,)."""
        small, big = (0.75, 0.2, 0.05, 0.0), (0.2, 0.3, 0.3, 0.2)
        rng = random.Random(23)
        queries, plays = [], []
        for m in range(6):
            payouts = []
            for t in range(400):
                mode = big if (t >= 7 and (payouts[t - 5] >= 5 or payouts[t - 7] >= 5)) else small
                payouts.append(rng.choices(PAYOUTS[1:], mode)[0] if rng.random() < 0.6 else 0)
            queries.append(Query(f"q{m}", "e1", f"m{m}", "steady", 0, 400))
            plays.extend(Play("e1", f"m{m}", t, v) for t, v in enumerate(payouts))
        self.assertEqual(discover_size_keys(tuple(queries), tuple(plays)), ((), (5,), (7,)))

    def test_size_key_discovery_without_signal_keeps_the_defaults(self):
        q = (Query("q", "e", "m", "steady", 0, 4),)
        h = tuple(Play("e", "m", t, 1) for t in range(4))  # too few wins to score any lag
        self.assertEqual(discover_size_keys(q, h), MODE_SIZE_KEYS)

    def test_three_size_modes_from_targets_and_posterior_picks_the_right_one(self):
        mixes = ((0.7, 0.2, 0.08, 0.02), (0.4, 0.4, 0.15, 0.05), (0.1, 0.2, 0.4, 0.3))
        queries = tuple(Query(f"q{i}", "e1", f"m{i}", "jackpot", 0, 10) for i in range(30))
        targets = [Target(f"q{i}", "e1", (0.5,) + tuple(0.5 * x for x in mixes[i % 3]), 0) for i in range(30)]
        spec = size_modes_from_targets(targets, queries)["jackpot"]
        self.assertEqual(len(spec["modes"]), 3)
        for got, want in zip(spec["modes"], mixes):
            self.assertTrue(all(abs(a - b) < 1e-9 for a, b in zip(got, want)))
        # Counts drawn from the middle mode -> the posterior mixture sits on the middle mode.
        mixed = size_mode_mixture((40, 40, 15, 5), spec["modes"], spec["weights"])
        self.assertLess(max(abs(a - b) for a, b in zip(mixed, mixes[1])), 0.01)

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
        self.assertAlmostEqual(prior[0][1], 0.03)
        self.assertAlmostEqual(prior[1][1], 0.9)
        self.assertAlmostEqual(sum(c[0] for c in prior), 1.0)
        self.assertIsNone(win_prior_from_targets(targets[:30]))  # no live side -> no prior

    def test_win_prior_pulls_cells_to_the_nearest_level(self):
        """A cell with 3 wins in 3 spins should read near the live level, not the 0.5-ish base rate."""
        model = PairMixer(win_prior=[(0.45, 0.03, 6), (0.45, 0.93, 6), (0.1, 0.5, 2)])
        self.assertGreater(model._cell_rate(3.0, 3.0), 0.85)
        self.assertLess(model._cell_rate(0.0, 3.0), 0.1)
        self.assertAlmostEqual(model._cell_rate(0.0, 0.0), 0.45 * 0.03 + 0.45 * 0.93 + 0.1 * 0.5, places=9)

    def test_type_specific_win_priors_shrink_toward_global_levels(self):
        queries, targets = [], []
        levels = {"steady": (0.02, 0.90), "jackpot": (0.20, 0.70)}
        index = 0
        for machine_type, (dead, live) in levels.items():
            for win in (dead,) * 30 + (live,) * 30:
                query_id = f"q{index}"
                queries.append(Query(query_id, "e1", f"m{index}", machine_type, 0, 1))
                targets.append(Target(query_id, "e1", (1 - win, win, 0, 0, 0), 0))
                index += 1
        prior = win_prior_by_type_from_targets(targets, queries, global_shrink=0.5)
        self.assertAlmostEqual(prior["steady"][0][1], (0.02 + 0.20) / 2, places=4)
        self.assertAlmostEqual(prior["jackpot"][0][1], (0.20 + 0.20) / 2, places=4)
        model = PairMixer(win_prior=prior)
        self.assertLess(model._cell_rate(0.0, 3.0, "steady"),
                        model._cell_rate(0.0, 3.0, "jackpot"))

    def test_train_only_win_prior_is_u_shaped(self):
        """No targets: the fallback prior must still find a dead and a live level from history."""
        rng = random.Random(5)
        queries, plays = [], []
        for m in range(12):  # win iff lag 1 == lag 3 (95%): match cells ~0.97, differ cells ~0.03
            won = []
            for t in range(300):
                rule = t >= 3 and rng.random() < 0.95
                won.append(won[t - 1] == won[t - 3] if rule else rng.random() < 0.5)
            queries.append(Query(f"q{m}", "e1", f"m{m}", "steady", 0, 300))
            plays.extend(Play("e1", f"m{m}", t, 1 if w else 0) for t, w in enumerate(won))
        prior = win_prior_from_train(tuple(queries), tuple(plays), pairs=((1, 3),))
        self.assertLess(prior[0][1], 0.2)
        self.assertGreater(prior[1][1], 0.8)

    def test_episode_folds_never_mix_fit_and_heldout(self):
        queries, history = [], []
        conditions = {}
        for e in range(16):
            for m in range(2):
                q = Query(f"e{e}-m{m}", f"e{e}", f"m{m}", "steady", 0, 3)
                queries.append(q)
                conditions[q.query_id] = ("stable", "drift")[e % 2]
                history.extend(Play(f"e{e}", f"m{m}", t, 1) for t in range(3))
        folds = episode_kfolds(queries, conditions, n_folds=4)
        self.assertEqual(sorted(folds.values()).count(0), 4)  # 2 episodes per condition per fold
        for fold in range(4):
            held = {e for e, f in folds.items() if f == fold}
            (hq, hh), (fq, fh) = split_episodes(tuple(queries), tuple(history), held)
            self.assertFalse({q.episode_id for q in fq} & held)
            self.assertFalse({p.episode_id for p in fh} & held)
            self.assertEqual({q.episode_id for q in hq}, held)

    def test_fold_model_ignores_heldout_labels(self):
        """adapt() on the fit episodes must give the same forecasts whatever the held-out labels say."""
        queries, history = make_data(seed=7, cutoff=120, rule=True)
        fit = tuple(q for q in queries if q.machine_id != "e1-m2")
        fit_history = tuple(p for p in history if p.machine_id != "e1-m2")
        labels = {q.query_id: 0 for q in queries}
        flipped = dict(labels, **{"e1-m2-next": 20})
        quiet = lambda *_: None
        a = adapt(fit, fit_history, {k: labels[k] for k in (q.query_id for q in fit)}, log=quiet)
        b = adapt(fit, fit_history, {k: flipped[k] for k in (q.query_id for q in fit)}, log=quiet)
        self.assertEqual(a.predict(queries, history), b.predict(queries, history))

    def test_type_prior_middle_level_sits_at_the_type_median(self):
        queries, targets = [], []
        for i, win in enumerate((0.03,) * 25 + (0.9,) * 25 + (0.3, 0.35, 0.4) * 5):
            queries.append(Query(f"q{i}", "e1", f"m{i}", "jackpot", 0, 1))
            targets.append(Target(f"q{i}", "e1", (1 - win, win, 0, 0, 0), 0))
        middle = win_prior_by_type_from_targets(targets, queries)["jackpot"][2]
        self.assertEqual(middle[1:], (0.35, 10.0))
        flat = win_prior_by_type_from_targets(targets, queries, middle_strength=None)["jackpot"][2]
        self.assertEqual(flat[1:], (0.5, 2.0))

    def test_signal_prior_shifts_levels_and_is_used_by_the_walk(self):
        queries, targets = [], []
        for i in range(120):
            sig = (-1, 1)[i % 2]
            win = 0.02 if i % 4 < 2 else (0.70 if sig < 0 else 0.90)
            queries.append(Query(f"q{i}", "e1", f"m{i}", "balanced", sig, 1))
            targets.append(Target(f"q{i}", "e1", (1 - win, win, 0, 0, 0), 0))
        prior = win_prior_by_type_from_targets(targets, queries)
        low, high = prior["balanced|-1"][1][1], prior["balanced|1"][1][1]
        self.assertAlmostEqual(low, 0.5 * 0.70 + 0.5 * prior["balanced"][1][1], places=4)
        self.assertLess(low, high)
        model = PairMixer(win_prior=prior)
        self.assertLess(model._cell_rate(3.0, 3.0, "balanced|-1"), model._cell_rate(3.0, 3.0, "balanced|1"))

    def test_rule_hmm_follows_a_rule_switch_and_never_looks_ahead(self):
        """Wins follow lags (1,3) MATCH, then switch to (2,5) MATCH: the HMM must end on (2,5)."""
        rng = random.Random(4)
        won = [rng.random() < 0.5 for _ in range(5)]
        for t in range(5, 300):
            a, b = (1, 3) if t < 150 else (2, 5)
            won.append((won[t - a] == won[t - b]) if rng.random() < 0.97 else rng.random() < 0.5)
        query = Query("q", "e1", "m", "steady", 0, 300)
        history = tuple(Play("e1", "m", t, 1 if w else 0) for t, w in enumerate(won))
        hmm = HiddenRuleHMM(((1, 3), (2, 5)), {"steady": (0.95, 0.03)})
        p = hmm.win_chances((query,), history)["q"]
        self.assertAlmostEqual(p, 0.95 if won[298] == won[295] else 0.03, delta=0.1)
        later = history + tuple(Play("e1", "m", t, 1) for t in range(300, 310))
        with self.assertRaises(ValidationError):  # rows at/after the cutoff are refused outright
            hmm.win_chances((query,), later)

    def test_rule_hmm_blend_keeps_the_mixer_sizes(self):
        queries, history = make_data(seed=3, cutoff=80, rule=True)
        mixer = PairMixer().fit(queries, history)
        prior = {"steady": [(0.45, 0.03, 10), (0.45, 0.9, 10), (0.1, 0.5, 2)]}
        self.assertEqual(levels_from_prior(prior), {"steady": (0.9, 0.03)})
        blended = WithRuleHMM(mixer, HiddenRuleHMM(((1, 3),), levels_from_prior(prior)), 0.3).predict(queries, history)
        base = mixer.predict(queries, history)
        for qid, probs in blended.items():
            self.assertAlmostEqual(sum(probs), 1.0, places=9)
            for k in range(1, 5):
                self.assertAlmostEqual(probs[k] / (1 - probs[0]), base[qid][k] / (1 - base[qid][0]), places=3)

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
