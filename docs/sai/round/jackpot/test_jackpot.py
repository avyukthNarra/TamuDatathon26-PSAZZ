"""Train-rule BIC, polarity and copied-mixer causality/type-isolation checks."""
import copy
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'worktrees/_tm/mixer6/mainFiles'))
from structure import designs,table_score
from mixer6_copy import PairMixer


class JackpotTests(unittest.TestCase):
    PRIOR={'jackpot':[(.45,.0126,10.),(.45,.6691,10.),(.1,.5,2.)],
           'balanced':[(.45,.0394,10.),(.45,.8396,10.),(.1,.5,2.)]}

    def test_bic_recovers_known_parity_and_either_polarity(self):
        rng=np.random.RandomState(991)
        lags=rng.choice([0,1,2,5,20],size=(1000,8))
        parity=(lags[:,0]>0)==(lags[:,2]>0)
        candidates=[r for r in designs(lags) if r[0]=='win_pair']
        for flipped in (False,True):
            y=np.logical_xor(parity,flipped).astype(float)
            scores=[table_score(key,y)[0] for _,_,key in candidates]
            chosen=candidates[int(np.argmax(scores))]
            self.assertEqual(chosen[1]['lags'],[1,3])
            gain,table=table_score(chosen[2],y)
            self.assertGreater(gain,100.)
            self.assertEqual(table['rates'][1]>table['rates'][0],not flipped)

    def test_constant_outcome_and_single_cell_bic_are_finite(self):
        self.assertEqual(table_score(np.zeros(20),np.zeros(20))[0],0.)
        for y in (np.zeros(20),np.ones(20)):
            score,table=table_score(np.arange(20)%2,y)
            self.assertTrue(np.isfinite(score))
            self.assertEqual(table['n'],[10,10])

    def test_copied_mixer_is_source_identical_except_import(self):
        source=(ROOT/'worktrees/_tm/mixer6/mainFiles/slot_forecasting/pair_mixer.py').read_text()
        copied=Path(__file__).with_name('mixer6_copy.py').read_text()
        self.assertEqual(copied,source.replace('from .schema import','from slot_forecasting.schema import',1))

    def test_jackpot_prior_change_leaves_other_type_exact(self):
        rng=np.random.RandomState(23)
        own=rng.choice(range(5),size=80).tolist()+[None]
        modified=copy.deepcopy(self.PRIOR);modified['jackpot'][1]=(.45,.59,21.)
        def emitted(priors,kind):
            result=[]
            PairMixer(win_prior=priors,eta=6.).walk(own,kind,80,lambda t,p:result.append(p))
            return np.array(result)
        np.testing.assert_array_equal(emitted(self.PRIOR,'balanced'),emitted(modified,'balanced'))
        self.assertGreater(abs(emitted(self.PRIOR,'jackpot')-emitted(modified,'jackpot')).max(),1e-4)

    def test_online_walk_never_uses_current_or_future_outcomes(self):
        rng=np.random.RandomState(18)
        own=rng.choice(range(5),size=80).tolist()+[None]
        def emitted(values):
            result=[]
            PairMixer(win_prior=self.PRIOR,eta=6.).walk(values,'jackpot',80,lambda t,p:result.append(p))
            return np.array(result)
        first=emitted(own)
        for t in (0,7,39,79):
            changed=own.copy();changed[t:]=[4]*(len(changed)-t)
            np.testing.assert_array_equal(first[:t+1],emitted(changed)[:t+1])


if __name__=='__main__':unittest.main()
