"""Synthetic checks: simplex constraints, episode separation and target isolation."""
import importlib.util
from pathlib import Path
import unittest
import json
import tempfile
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location('combo_search', Path(__file__).with_name('search.py'))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
from harness.data import Split, CONDITIONS


class SearchTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(17)
        self.A = rng.dirichlet(np.ones(5), size=(3, 48))
        a = dict(qid=np.array([f'q{i}' for i in range(48)]),
                 episode=np.array([f'e{i//2}' for i in range(48)]),
                 machine=np.array(['a', 'b']*24), mtype=np.array(['x']*48),
                 signal=np.zeros(48), cutoff=np.ones(48), grid=np.zeros((48, 1)),
                 truth=.3*self.A[0]+.7*self.A[1], ref=np.full((48, 5), .2),
                 condition=np.array([CONDITIONS[(i//2)//6] for i in range(48)]))
        self.v = Split('synthetic', a, {})

    def test_known_linear_optimum_and_simplex(self):
        x = s.fit(self.A, 'linear', self.v.truth, np.ones(48))
        np.testing.assert_allclose(x, [.3, .7, 0], atol=1e-5)
        for family in s.FAMILIES:
            x = s.fit(self.A, family, self.v.truth, np.ones(48))
            p = s.forecast(self.A, family, x)
            self.assertTrue((p >= 0).all())
            np.testing.assert_allclose(p.sum(1), 1, atol=1e-10)
            if family == 'factorised':
                np.testing.assert_allclose([x[:3].sum(), x[3:].sum()], 1)
            else:
                self.assertAlmostEqual(x.sum(), 1)

    def test_geometric_known_optimum(self):
        truth = s.forecast(self.A, 'geometric', np.array([.2, .3, .5]))
        x = s.fit(self.A, 'geometric', truth, np.ones(48))
        np.testing.assert_allclose(x, [.2, .3, .5], atol=1e-5)

    def test_factorised_known_optimum(self):
        truth = s.forecast(self.A, 'factorised', np.array([.2, .3, .5, .6, .3, .1]))
        x = s.fit(self.A, 'factorised', truth, np.ones(48))
        np.testing.assert_allclose(s.forecast(self.A, 'factorised', x), truth, atol=1e-5)

    def test_episode_folds_and_heldout_target_isolation(self):
        episodes = self.v.episodes
        fold = s.folds(self.v, episodes, 6, 'outer')
        self.assertEqual(fold, s.folds(self.v, episodes, 6, 'outer'))
        self.assertEqual(set(fold.values()), set(range(6)))
        for c in CONDITIONS:
            eps = [e for e in episodes if self.v.condition[self.v.episode_rows(e)[0]] == c]
            self.assertEqual(sorted(fold[e] for e in eps), list(range(6)))
        te = s.rows_for(self.v, [e for e in episodes if fold[e] == 0])
        first, _ = s.cv(self.v, self.A, 'factorised', episodes, 6, 'outer')
        self.v.truth[te] = .2  # includes reference-loss denominator effects
        second, _ = s.cv(self.v, self.A, 'factorised', episodes, 6, 'outer')
        np.testing.assert_array_equal(first[te], second[te])

    def test_one_shot_guard_blocks_before_scoring(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / 'results.json').write_text(json.dumps({
                'source_sha256': {}, 'prediction_sha256': {}, 'top10': []}))
            (out / 'audit.once.json').write_text('{}')
            with patch.object(s, 'OUT', out), patch.object(s, 'HERE', out), \
                    patch.object(s, 'load_split', side_effect=AssertionError('must not load audit')):
                with self.assertRaises(FileExistsError):
                    s.audit()



if __name__ == '__main__':
    unittest.main()
