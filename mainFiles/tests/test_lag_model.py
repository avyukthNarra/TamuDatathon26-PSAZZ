"""Run from mainFiles/:  python -m unittest discover -s tests -v"""
import random
import tempfile
import unittest
from pathlib import Path

from slot_forecasting.lag_model import (
    FEATURES, MIN_PROBABILITY, LagForecaster, build_examples, query_features, type_priors)
from slot_forecasting.schema import PAYOUTS, Play, Query

CUTOFF = 40
MACHINES = ("e1-m0", "e1-m1", "e1-m2")


def make_data(seed=1, cutoff=CUTOFF, skip=()):
    rng = random.Random(seed)
    queries = tuple(Query(f"{m}-next", "e1", m, ("steady", "balanced", "jackpot")[i], (-1, 0, 1)[i], cutoff)
                    for i, m in enumerate(MACHINES))
    history = tuple(Play("e1", m, t, rng.choice(PAYOUTS))
                    for m in MACHINES for t in range(cutoff) if (m, t) not in skip)
    return queries, history


class NoLookaheadTest(unittest.TestCase):
    def test_training_row_equals_prediction_row_built_without_the_future(self):
        """Tick t's training features must equal features computed from ticks < t only."""
        queries, history = make_data(skip={("e1-m0", 5), ("e1-m1", 12)})
        priors = type_priors(queries, history)
        rows, targets, _ = build_examples(queries, history, priors, min_tick=0)
        m0_rows = rows[:CUTOFF - 1]  # m0 is walked first; tick 5 is missing, so it has CUTOFF - 1 rows
        observed_ticks = [t for t in range(CUTOFF) if t != 5]
        for t in (9, 20, 39):
            past_queries = tuple(Query(q.query_id, q.episode_id, q.machine_id, q.machine_type, q.signal, t)
                                 for q in queries)
            past = tuple(p for p in history if p.tick < t)
            expected = query_features(past_queries, past, priors)["e1-m0-next"]
            self.assertEqual(m0_rows[observed_ticks.index(t)], expected, f"tick {t}")

    def test_changing_the_future_does_not_change_features(self):
        queries, history = make_data()
        priors = type_priors(queries, history)
        before = query_features(queries, history, priors)
        # Rewrite every machine's last 10 ticks; features for a cutoff-30 query must not move.
        early = tuple(Query(q.query_id, q.episode_id, q.machine_id, q.machine_type, q.signal, 30) for q in queries)
        a = query_features(early, tuple(p for p in history if p.tick < 30), priors)
        changed = tuple(Play(p.episode_id, p.machine_id, p.tick, 20 if p.tick >= 30 else p.payout) for p in history)
        b = query_features(early, tuple(p for p in changed if p.tick < 30), priors)
        self.assertEqual(a, b)
        self.assertNotEqual(before, a)

    def test_missing_tick_is_unknown_not_zero(self):
        queries, history = make_data(skip={("e1-m0", CUTOFF - 1)})
        priors = type_priors(queries, history)
        row = dict(query_features(queries, history, priors)["e1-m0-next"])
        self.assertEqual(row.get(FEATURES.index("lag1=missing")), 1.0)
        self.assertNotIn(FEATURES.index("lag1=0"), row)


class ModelTest(unittest.TestCase):
    def test_predictions_are_valid_and_reload_identically(self):
        queries, history = make_data(seed=2)
        labels = {q.query_id: 1 for q in queries}
        model = LagForecaster(epochs=2, blend=1.0).fit(queries, history, labels)
        predictions = model.predict(queries, history)
        for probs in predictions.values():
            self.assertAlmostEqual(sum(probs), 1.0, places=9)
            self.assertTrue(all(p >= MIN_PROBABILITY * 0.99 for p in probs))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            model.save(path)
            self.assertEqual(LagForecaster.load(path).predict(queries, history), predictions)

    def test_blend_mixes_with_baseline(self):
        queries, history = make_data(seed=3)
        model = LagForecaster(epochs=1, blend=0.5).fit(queries, history)
        uniform = {q.query_id: (0.2,) * 5 for q in queries}
        raw = LagForecaster(epochs=1, blend=1.0).fit(queries, history).predict(queries, history)
        mixed = model.predict(queries, history, uniform)
        for key in raw:
            for r, m in zip(raw[key], mixed[key]):
                self.assertAlmostEqual(m, 0.5 * r + 0.1, places=9)


if __name__ == "__main__":
    unittest.main()
