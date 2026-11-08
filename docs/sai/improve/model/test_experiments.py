"""Meaningful regression checks for the rewritten online win head."""
import unittest
import numpy as np
from experiments import win_head, train_structure
from improved_mixer import as_split
from pranav_v3 import PairMixer
from slot_forecasting.schema import Query, Play


class WinHeadTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(924)
        self.queries = [Query(f'q{i}','e',f'm{i}',kind,0,c)
                        for i,(kind,c) in enumerate(zip(('steady','balanced','jackpot'),(85,64,100)))]
        self.hist = [Play('e',f'm{i}',t,int(rng.choice([0,1,2,5,20])))
                     for i,q in enumerate(self.queries) for t in range(q.cutoff) if i == 1 or t%7 != 0]
        self.s = as_split(self.queries,self.hist)
        self.prior = {k:np.full(3,1/3) for k in ('steady','balanced','jackpot')}
        self.cells = {k:np.full((3,2),.5) for k in self.prior}
        self.beta = {k:[.1,.8] for k in self.prior}

    def test_baseline_matches_original_walk_with_gaps_and_varying_cutoffs(self):
        actual = win_head(self.s,self.prior,self.cells,self.beta)
        expected = []
        for i,q in enumerate(self.queries):
            own = [None if not np.isfinite(x) else [0,1,2,5,20].index(x) for x in self.s.grid[i,:q.cutoff]]
            raw = []
            PairMixer().walk(own,q.machine_type,q.cutoff,lambda t,p:raw.append(1-p[0]) if t == q.cutoff else None)
            expected.append(raw[0])
        np.testing.assert_allclose(actual,expected,atol=2e-15,rtol=0)

    def test_uniform_pair_prior_has_no_effect(self):
        a = win_head(self.s,self.prior,self.cells,self.beta)
        b = win_head(self.s,self.prior,self.cells,self.beta,prior_power=4.)
        np.testing.assert_allclose(a,b,atol=2e-15,rtol=0)

    def test_target_changes_do_not_affect_forecast(self):
        a = win_head(self.s,self.prior,self.cells,self.beta,beta_strength=6.,k_short=1.,k_long=.25)
        self.s.truth = np.full((3,5),np.nan)
        self.s.outcome = np.array([-1,-1,-1])
        b = win_head(self.s,self.prior,self.cells,self.beta,beta_strength=6.,k_short=1.,k_long=.25)
        np.testing.assert_array_equal(a,b)
        self.assertTrue(np.isfinite(a).all() and (a>=0).all() and (a<=1).all())

    def test_as_split_excludes_cutoff(self):
        # The original schema rejects future ticks rather than letting the model see them.
        with self.assertRaises(ValueError):
            as_split(self.queries,self.hist+[Play('e','m0',85,20)])


if __name__ == '__main__':
    unittest.main()
