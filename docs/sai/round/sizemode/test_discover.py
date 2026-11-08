"""Check temporal causality, exact HMM likelihood, and original size-head parity."""
import itertools
import unittest
import numpy as np
from discover import RULES,HMM_SWITCH,rule_probability,hmm_likelihood,baseline_size_trace,pm
from harness.data import Split

MODES={'steady':{'modes':[[.8,.15,.04,.01],[.1,.2,.3,.4]],'weights':[1,1]}}


def split(own):
    return Split('synthetic',{'qid':np.array(['q']),'episode':np.array(['e']),'machine':np.array(['m']),
        'mtype':np.array(['steady']),'signal':np.array([0]),'cutoff':np.array([len(own)]),'grid':np.array([own],float)},{})


class DiscoveryTests(unittest.TestCase):
    def test_rules_polarities_missing_values_and_causality(self):
        own=np.array([1,np.nan,5,0,20,2,5],float);rates={'big':.2,'win':.6}
        p=rule_probability(own,rates)
        expected={'xor_big':1.,'and_big':0.,'or_big':1.,'xor_win':0.,'and_win':1.,'or_win':1.}
        for family,value in expected.items():
            for polarity in (1,-1):
                index=RULES.index((family,'big' if family.endswith('big') else 'win',(1,2),family.split('_')[0],polarity))
                self.assertEqual(p[6,index],value if polarity==1 else 1-value)
        index=RULES.index(('single_big','big',(1,),'single',1))
        self.assertAlmostEqual(p[2,index],.4/3)
        other=own.copy();other[4:]=20
        np.testing.assert_array_equal(p[:5],rule_probability(other,rates)[:5])

    def test_hmm_forward_likelihood_matches_hidden_path_enumeration(self):
        own=[1,np.nan,0,20];s=split(own);ma=np.array([MODES['steady']['modes']])
        full,prefix,_=hmm_likelihood(s,ma,np.array([2]))
        for j,prob in enumerate(HMM_SWITCH):
            likelihood=0.;pre=0.
            for path in itertools.product((0,1),repeat=4):
                weight=.5*np.prod([prob if path[t]!=path[t-1] else 1-prob for t in range(1,4)])
                all_em=1.;prefix_em=1.
                for t,value in enumerate(own):
                    if np.isfinite(value) and value>0:
                        emission=ma[0,path[t],[1,2,5,20].index(value)]
                        all_em*=emission
                        if t<2:prefix_em*=emission
                likelihood+=weight*all_em;pre+=weight*prefix_em
            self.assertAlmostEqual(full[0,j],np.log(likelihood),places=13)
            self.assertAlmostEqual(prefix[0,j],np.log(pre),places=13)

    def test_vector_size_head_matches_mixer6_walk(self):
        rng=np.random.RandomState(72);own=rng.choice([0,1,2,5,20],70).astype(float);own[3::8]=np.nan
        s=split(own);keys=((),(1,),(2,),(4,))
        actual=baseline_size_trace(s,MODES,keys)[0]
        classes=[None if not np.isfinite(x) else [0,1,2,5,20].index(x) for x in own]
        captured=[]
        pm.PairMixer(size_modes=MODES,size_keys=keys).walk(classes,'steady',len(own),lambda t,p:captured.append(p))
        captured=np.array(captured)
        expected=captured[:,1:]/captured[:,1:].sum(1,keepdims=True)
        np.testing.assert_allclose(actual,expected,atol=4e-14,rtol=0)

if __name__=='__main__':unittest.main()
