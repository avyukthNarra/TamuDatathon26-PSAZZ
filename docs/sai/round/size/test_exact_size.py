"""Verify direct discounted likelihood, missing-lag causality, and recovery."""
import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.special import softmax
from exact_size import ExactSizeSwitch,HYPOTHESIS_PRIOR
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'docs/sai'))
from harness.data import Split

MODES={'steady':{'modes':[[.8,.15,.04,.01],[.1,.2,.3,.4]],'weights':[1,1]}}


def split(grid,cutoff):
    n=len(grid)
    return Split('synthetic',{'qid':np.array([f'q{i}' for i in range(n)]),'episode':np.array([f'e{i}' for i in range(n)]),
        'machine':np.array([f'm{i}' for i in range(n)]),'mtype':np.array(['steady']*n),
        'signal':np.zeros(n,int),'cutoff':np.array(cutoff),'grid':np.array(grid,float)},{})


class ExactTests(unittest.TestCase):
    def test_direct_discounted_likelihood_with_missing_ticks_and_different_cutoffs(self):
        s=split([[1,np.nan,5,0,20,1,2,0],[0,1,np.nan,5,2,0,np.nan,np.nan]],[8,6])
        model=ExactSizeSwitch(MODES,{'steady':.2},(12,40,np.inf))
        actual,detail=model.predict(s,True)
        small,big=np.array(MODES['steady']['modes'])
        for i,T in enumerate(s.cutoff):
            def q(t):
                past=s.grid[i,:t];observed=past[np.isfinite(past)]
                fallback=(.4+(observed>=5).sum())/(2+len(observed))
                b=[float(s.grid[i,t-k]>=5) if t>=k and np.isfinite(s.grid[i,t-k]) else fallback for k in range(1,5)]
                p=np.r_[np.array([(x,1-x) for x in b]).ravel(),.5]
                return small+p[:,None]*(big-small)
            for h,half in enumerate((12,40,np.inf)):
                logp=np.log(HYPOTHESIS_PRIOR)
                for t in range(T):
                    if np.isfinite(s.grid[i,t]) and s.grid[i,t]>0:
                        cls=[1,2,5,20].index(s.grid[i,t])
                        logp+=2**(-(T-1-t)/half)*np.log(q(t)[:,cls])
                expected=softmax(logp)@q(T)
                np.testing.assert_allclose(actual[h,i],expected,atol=3e-15,rtol=0)
                np.testing.assert_allclose(detail['posterior'][h,i],softmax(logp),atol=3e-15,rtol=0)

    def test_recover_known_lag_one_polarity(self):
        rng=np.random.RandomState(803);grid=[];small,big=np.array(MODES['steady']['modes'])
        for t in range(600):
            mode=big if t and grid[-1]>=5 else small
            grid.append(rng.choice([1,2,5,20],p=mode))
        s=split([grid],[len(grid)])
        _,detail=ExactSizeSwitch(MODES,{'steady':.2},(np.inf,)).predict(s,True)
        self.assertGreater(detail['posterior'][0,0,0],.999)

    def test_no_outcome_probabilities_or_condition_needed(self):
        s=split([[1,5,np.nan,0,2,20]],[6]);model=ExactSizeSwitch(MODES,{'steady':.2})
        a=model.predict(s)
        s.truth=np.full((1,5),np.nan);s.condition=np.array(['arbitrary'])
        np.testing.assert_array_equal(a,model.predict(s))
        np.testing.assert_allclose(a.sum(2),1,atol=3e-15,rtol=0)


if __name__=='__main__':unittest.main()
