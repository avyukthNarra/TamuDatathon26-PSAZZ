"""Run from mainFiles/:  python -m unittest discover -s tests -v"""
import random
import unittest

from slot_forecasting.changepoint import RegimeModel
from slot_forecasting.ensemble import Ensemble
from slot_forecasting.pair_mixer import PairMixer
from slot_forecasting.schema import Play, Query, ValidationError

PRIOR = {"steady": [(0.45, 0.03, 10.0), (0.45, 0.9, 10.0), (0.1, 0.5, 10.0)]}


def rule_data(seed, n, switch_at=None, live=0.9):
    """Wins follow lags (1, 3) AGREE; after switch_at they follow lags (2, 5) DIFFER."""
    rng = random.Random(seed)
    won = [rng.random() < 0.5 for _ in range(5)]
    for t in range(5, n):
        (a, b), agree = ((1, 3), True) if switch_at is None or t < switch_at else ((2, 5), False)
        on = (won[t - a] == won[t - b]) == agree
        won.append(rng.random() < (live if on else 0.03))
    return won


def model(**kw):
    return RegimeModel(((1, 3), (2, 5), (4, 7)), PRIOR, **kw)


class RegimeModelTest(unittest.TestCase):
    def test_learns_the_rule(self):
        won = rule_data(1, 250)
        seq = {t: int(w) for t, w in enumerate(won)}
        p = model().win_probability(seq, 250, "steady")
        self.assertAlmostEqual(p, 0.9 if won[249] == won[247] else 0.03, delta=0.12)

    def test_follows_a_rule_switch(self):
        won = rule_data(2, 300, switch_at=150)
        seq = {t: int(w) for t, w in enumerate(won)}
        p = model().win_probability(seq, 300, "steady")
        self.assertAlmostEqual(p, 0.9 if won[298] != won[295] else 0.03, delta=0.12)

    def test_learns_a_machine_level_between_the_type_levels(self):
        """A machine live at 0.6 (not the type's 0.9) should be forecast near 0.6, not 0.9."""
        won = rule_data(3, 250, live=0.6)
        seq = {t: int(w) for t, w in enumerate(won)}
        live_cutoffs = [t for t in range(200, 250) if won[t - 1] == won[t - 3]]
        p = model().win_probability({k: v for k, v in seq.items() if k < live_cutoffs[-1]}, live_cutoffs[-1], "steady")
        self.assertLess(abs(p - 0.6), 0.15)

    def test_never_reads_the_future(self):
        q = Query("q", "e", "m", "steady", 0, 50)
        history = tuple(Play("e", "m", t, 1) for t in range(50))
        model().win_chances((q,), history)
        with self.assertRaises(ValidationError):
            model().win_chances((q,), history + (Play("e", "m", 50, 0),))

    def test_deterministic(self):
        seq = {t: int(w) for t, w in enumerate(rule_data(4, 200))}
        self.assertEqual(model().win_probability(seq, 200, "steady"), model().win_probability(seq, 200, "steady"))

    def test_ensemble_keeps_mixer_sizes_and_sums_to_one(self):
        queries = (Query("q", "e", "m", "steady", 0, 120),)
        history = tuple(Play("e", "m", t, 5 if w else 0) for t, w in enumerate(rule_data(5, 120)))
        mixer = PairMixer().fit(queries, history)
        base = mixer.predict(queries, history)["q"]
        probs = Ensemble(mixer, [model()], (0.5, 0.5)).predict(queries, history)["q"]
        self.assertAlmostEqual(sum(probs), 1.0, places=9)
        for k in range(1, 5):
            self.assertAlmostEqual(probs[k] / (1 - probs[0]), base[k] / (1 - base[0]), places=3)


if __name__ == "__main__":
    unittest.main()
