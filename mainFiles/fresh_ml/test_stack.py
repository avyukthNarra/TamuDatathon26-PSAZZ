import csv
import contextlib
import io
from pathlib import Path
import unittest
import tempfile

import numpy as np

from slot_forecasting.contest import contest_score
from slot_forecasting.io import read_queries, read_submission, read_targets
from slot_forecasting.schema import CONDITIONS, Query, Play
from .mixer6_snapshot import PairMixer
from .stack import Evaluator, blend, folds, normalized_curve, opposite_fold_targets, residual_predictions, read_conditions, evaluate


class StackingTest(unittest.TestCase):
    def test_mixer_curve_is_causal_with_missing_ticks(self):
        rng = np.random.default_rng(914)
        own = [int(v) if v >= 0 else None for v in rng.integers(-1, 5, 50)]
        model = PairMixer(pairs=((1, 2), (2, 3)), win_prior=[(.5, .05, 6), (.5, .9, 6)])
        expected = normalized_curve(model, own, 'steady', 49)
        for cutoff in (0, 1, 8, 25, 49):
            changed = own[:cutoff] + [None if v < 0 else int(v) for v in rng.integers(-1, 5, 50 - cutoff)]
            actual = normalized_curve(model, changed, 'steady', 49)
            np.testing.assert_allclose(actual[:cutoff + 1], expected[:cutoff + 1], atol=0, rtol=0)

    def test_mixer_curve_matches_predict(self):
        query = Query('q', 'e', 'm', 'steady', 0, 35)
        own = [None if t % 7 == 0 else t % 5 for t in range(35)] + [None]
        history = [Play('e', 'm', t, (0, 1, 2, 5, 20)[v]) for t, v in enumerate(own) if v is not None]
        model = PairMixer(pairs=((1, 3), (2, 4)))
        curve = normalized_curve(model, own, 'steady', 35)
        np.testing.assert_allclose(curve[-1], model.predict([query], history)['q'], atol=1e-15)

    def test_scored_fold_targets_are_excluded(self):
        from types import SimpleNamespace
        ts = [SimpleNamespace(episode_id='a'), SimpleNamespace(episode_id='b')]
        assignment = {'a': 'A', 'b': 'B'}
        self.assertEqual(opposite_fold_targets(ts, assignment, 'A'), (ts[1],))
        self.assertEqual(opposite_fold_targets(ts, assignment, 'B'), (ts[0],))

    def test_fold_schemes_and_episode_balance(self):
        qs, conditions = [], {}
        for c in CONDITIONS:
            for i in range(8):
                q = Query(f'{c}-{i}', f'{c}-{i}', 'm', 'steady', 0, 40)
                qs.append(q)
                conditions[q.query_id] = c
        ordered, hashed = folds(qs, conditions), folds(qs, conditions, hashed=True)
        self.assertNotEqual(ordered, hashed)
        for c in CONDITIONS:
            self.assertEqual([ordered[f'{c}-{i}'] for i in range(8)], ['A', 'B'] * 4)
            self.assertEqual(sum(hashed[f'{c}-{i}'] == 'A' for i in range(8)), 4)

    def test_blend_and_zero_residual_endpoints(self):
        a = np.array([[.4, .3, .1, .1, .1], [.8, .05, .05, .05, .05]])
        b = np.array([[.8, .05, .05, .05, .05], [.4, .3, .1, .1, .1]])
        np.testing.assert_allclose(blend(a, b, 0, 0), a)
        np.testing.assert_allclose(blend(a, b, 1, 1), b)
        np.testing.assert_allclose(residual_predictions(a, np.zeros_like(a)), a)
        p = blend(a, b, 0, 1)
        np.testing.assert_allclose(p[:, 0], a[:, 0])
        np.testing.assert_allclose(p[:, 1:] / (1 - p[:, 0, None]), b[:, 1:] / (1 - b[:, 0, None]))

    def test_fast_scorer_matches_official_all_folds(self):
        data = Path(__file__).resolve().parents[1] / 'public'
        qs = read_queries(data / 'validation/queries.csv')
        targets = read_targets(data / 'validation/targets.csv')
        groups = read_conditions(data)
        reference = read_submission(data / 'validation/reference.csv')
        ids = np.array([q.query_id for q in qs])
        assignment = folds(qs, groups)
        ev = Evaluator(data, ids, assignment)
        predictions = .98 * np.array([reference[k] for k in ids]) + .02 / 5
        for fold in ('all', 'A', 'B'):
            selected = [t for t in targets if fold == 'all' or assignment[t.episode_id] == fold]
            keep = {t.query_id for t in selected}
            report = contest_score({k: p for k, p in zip(ids, predictions) if k in keep}, selected,
                                   {k: groups[k] for k in keep}, {k: reference[k] for k in keep}, max_points=60)
            self.assertAlmostEqual(60 / (1 + ev.ratio(predictions, fold)), report['technical_points'], places=12)

    def test_outer_selection_uses_tuning_fold_prior_plan(self):
        # A naive selection on already-CF forecasts would choose y and score
        # perfectly. Correct outer tuning chooses x on both sides and exposes
        # its failure on the actual held-out folds.
        data = Path(__file__).resolve().parents[1] / 'public'
        qs = read_queries(data / 'validation/queries.csv')
        targets = {t.query_id: t for t in read_targets(data / 'validation/targets.csv')}
        assignment = folds(qs, read_conditions(data))
        mask_a = np.array([assignment[q.episode_id] == 'A' for q in qs])
        good = np.array([targets[q.query_id].probabilities for q in qs])
        bad = np.full_like(good, .2)
        plan_a = {'mixer': bad, 'x': np.where(mask_a[:, None], bad, good),
                  'y': np.where(mask_a[:, None], good, bad)}
        plan_b = {'mixer': bad, 'x': np.where(mask_a[:, None], good, bad),
                  'y': np.where(mask_a[:, None], bad, good)}
        training = {'mixer': {'selection_logloss': 3}, 'x': {'selection_logloss': 1}, 'y': {'selection_logloss': 2}}
        dataset = {'splits': np.array(['validation'] * len(qs)), 'ids': np.array([q.query_id for q in qs])}
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            report = evaluate(data, Path(tmp), dataset,
                              {'sorted_A': (plan_a, training), 'sorted_B': (plan_b, training)}, 'sorted')
        self.assertEqual(report['stack_selected_on_A_for_B'], 'x')
        self.assertEqual(report['stack_selected_on_B_for_A'], 'x')
        self.assertLess(report['stacked_crossfit']['points'], 59)


if __name__ == '__main__':
    unittest.main()
