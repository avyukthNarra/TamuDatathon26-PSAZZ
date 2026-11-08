"""Causality, fractional counts and exact dense-gate regression checks."""
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parents[1]/'mainFiles'))
sys.path.insert(2, str(HERE/'improve/model'))
from harness.data import Split
from pranav_v4 import PairMixer
from sparse import beta_rate, gapped, win_head, forecast

PRIOR = ((.45,.0435,6.),(.45,.898,6.),(.1,.5,2.))


def fixture():
    rng = np.random.RandomState(341)
    grid = rng.choice([0.,1.,2.,5.,20.],size=(4,48))
    grid[1,4:28:2] = np.nan
    grid[3,8:30:3] = np.nan
    a = {'grid':grid,'qid':np.array(['a','b','c','d']),
         'episode':np.array(['e0','e0','e1','e1']),
         'machine':np.array(['m0','m1','m0','m1']),
         'mtype':np.array(['steady']*4),'signal':np.zeros(4),
         'cutoff':np.full(4,48)}
    return Split('synthetic',a,{})


class SparseTests(unittest.TestCase):
    def test_beta_posterior_matches_pinned_v4(self):
        m=PairMixer(win_prior=PRIOR)
        w=np.array([0.,.5,4.,8.]); n=np.array([0.,2.,8.,9.])
        expected=np.array([m._cell_rate(a,b) for a,b in zip(w,n)])
        np.testing.assert_allclose(beta_rate(w,n,PRIOR),expected,atol=1e-14)

    def test_unmodified_head_matches_original(self):
        s=fixture(); actual=win_head(s,PRIOR)
        expected=[]
        for i in range(s.n):
            own=[None if not np.isfinite(x) else [0,1,2,5,20].index(x) for x in s.grid[i]]
            raw=[]
            PairMixer(win_prior=PRIOR,eta=8.).walk(own,'steady',48,
                lambda t,p:raw.append(1-p[0]) if t==48 else None)
            expected.append(raw[0])
        np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=0)

    def test_sparse_gate_and_probability_validity(self):
        s=fixture(); np.testing.assert_array_equal(gapped(s),[False,True,False,True])
        baseline=np.random.RandomState(3).dirichlet(np.ones(5),size=s.n)
        sizes=baseline[:,1:]/baseline[:,1:].sum(1,keepdims=True)
        for params in [{'method':'expected'},{'method':'forward'},
                       {'decay_missing':False},{'extra':'both'}]:
            p=forecast(s,baseline,sizes,PRIOR,**params)
            self.assertEqual(p[~gapped(s)].tobytes(),baseline[~gapped(s)].tobytes())
            self.assertTrue(np.isfinite(p).all() and (p>=0).all())
            np.testing.assert_allclose(p.sum(1),1,atol=1e-12)

    def test_forward_predictions_never_read_future(self):
        s=fixture(); _,traces,_=win_head(s,PRIOR,method='forward',trace=True)
        for t in [0,7,20,34,47]:
            changed=fixture(); changed.grid[:,t:]=20.
            _,other,_=win_head(changed,PRIOR,method='forward',trace=True)
            np.testing.assert_array_equal(traces[:t+1],other[:t+1])
        changed=fixture(); changed.cutoff[:]=30
        changed.grid[:,30:]=np.nan
        a=win_head(changed,PRIOR,method='forward')
        changed.grid[:,30:]=20.
        b=win_head(changed,PRIOR,method='forward')
        np.testing.assert_array_equal(a,b)

    def test_missing_outcomes_are_never_training_targets(self):
        s=fixture();s.grid[:]=np.nan
        for method in ['expected','forward']:
            _,_,counts=win_head(s,PRIOR,method=method,trace=True)
            self.assertEqual(float(counts.sum()),0.)

    def test_fractional_updates_recover_lag_incomplete_exposures(self):
        s=fixture()
        _,_,baseline=win_head(s,PRIOR,trace=True)
        _,_,soft=win_head(s,PRIOR,method='expected',trace=True)
        self.assertGreater(soft[gapped(s)].sum(),baseline[gapped(s)].sum())
        np.testing.assert_array_equal(soft[~gapped(s)],baseline[~gapped(s)])


if __name__=='__main__':unittest.main()
