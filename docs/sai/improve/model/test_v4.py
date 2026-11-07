"""Regression tests against the pinned 39e5757 v4 implementation."""
import unittest
import numpy as np
import test_experiments
from experiments import win_head
from pranav_v4 import PairMixer


class V4Tests(unittest.TestCase):
    setUp=test_experiments.WinHeadTests.setUp
    POP=((.45,.0435,6.),(.45,.898,6.),(.1,.5,2.))

    def test_v4_reproduces_original_beta_posterior_and_online_weights(self):
        actual=win_head(self.s,self.prior,self.cells,self.beta,population_prior=self.POP,eta_short=8.,eta_long=8.)
        expected=[]
        for i,q in enumerate(self.queries):
            own=[None if not np.isfinite(x) else [0,1,2,5,20].index(x) for x in self.s.grid[i,:q.cutoff]]
            raw=[]
            PairMixer(win_prior=self.POP,eta=8.).walk(own,q.machine_type,q.cutoff,
                lambda t,p:raw.append(1-p[0]) if t==q.cutoff else None)
            expected.append(raw[0])
        np.testing.assert_allclose(actual,expected,atol=3e-14,rtol=0)

    def test_hierarchical_pseudocounts_change_v4_prediction(self):
        args=(self.s,self.prior,self.cells,self.beta)
        a=win_head(*args,population_prior=self.POP,eta_short=8.,eta_long=8.)
        b=win_head(*args,population_prior=self.POP,eta_short=8.,eta_long=8.,k_short=2.,k_long=2.)
        self.assertGreater(abs(a-b).max(),1e-4)
        self.assertTrue(np.isfinite(b).all() and (b>0).all() and (b<1).all())

if __name__=='__main__':unittest.main()
