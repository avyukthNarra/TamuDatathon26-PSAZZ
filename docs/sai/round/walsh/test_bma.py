"""Causality, reference-code parity and exact support enumeration checks."""
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from bma import centered_history,walsh_bma_predict_horizon,predict,SUBSETS_K6,MULT_PENALTY_K6


class BMATests(unittest.TestCase):
    def test_supports_are_all_42_unique_subsets_and_penalized(self):
        self.assertEqual(len(SUBSETS_K6),42)
        self.assertEqual(len(set(tuple(x) for x in SUBSETS_K6)),42)
        self.assertTrue(all(x[0]==0 and len(x)<=4 for x in SUBSETS_K6))
        np.testing.assert_allclose(MULT_PENALTY_K6[:7],[0]+[np.log(36)]*6)

    def test_centering_is_strictly_causal_and_gaps_stay_zero(self):
        r=np.random.RandomState(18)
        sequence=r.choice([0.,1.,2.,np.nan],80)
        a=centered_history(sequence,12.)[0]
        for t in [0,8,37,70]:
            changed=sequence.copy();changed[t:]=20.
            b=centered_history(changed,12.)[0]
            np.testing.assert_array_equal(a[:t+1],b[:t+1])
        args=centered_history([1.,np.nan,0.],40.)
        self.assertTrue(args[4][1]);self.assertEqual(args[3][1],0.)
        self.assertEqual(args[0].shape,(3,36))

    def test_prediction_ignores_post_cutoff_and_truth(self):
        s=SimpleNamespace(n=1,grid=np.tile([0.,1.,0.,5.],(1,20)),cutoff=np.array([55]))
        first,_=predict(s)
        s.grid[:,55:]=20.;s.truth=np.ones((1,5));s.condition=['fake']
        second,_=predict(s)
        np.testing.assert_array_equal(first,second)
        self.assertTrue(np.isfinite(first).all() and (first>0).all() and (first<1).all())

    def test_bma_weights_and_empty_history(self):
        r=np.random.RandomState(3)
        args=centered_history(r.choice([0.,1.,5.],size=80),None)
        p,info=walsh_bma_predict_horizon(*args,1.,diagnostics=True)
        self.assertTrue(0<p<1)
        self.assertAlmostEqual(sum(info['model_weights']),1.)
        maximum=max(info['log_evidences'])
        self.assertTrue(all(w==0 or e>=maximum-3 for w,e in zip(info['model_weights'],info['log_evidences'])))
        self.assertEqual(walsh_bma_predict_horizon(*centered_history([np.nan]*30,None),1.),.5)

    def test_matches_supplied_reference_on_nontied_random_inputs(self):
        path=Path(__file__).resolve().parents[4]/'gemini-code-1794099642432.md'
        code=path.read_text().split('```python\n',1)[1].split('```',1)[0]
        namespace={}
        exec(compile(code,str(path),'exec'),namespace)
        reference=namespace['walsh_bma_predict_horizon']
        rng=np.random.RandomState(731)
        x=rng.normal(size=(70,36));y=rng.binomial(1,.4,size=70)
        observed=rng.binomial(1,.8,size=70).astype(bool)
        query=rng.normal(size=36);missing=rng.binomial(1,.2,size=36).astype(bool)
        query[missing]=0.
        for slab in (1.,1.35,2.):
            args=(x,y,observed,query,missing,2.**(-1/12.))
            self.assertAlmostEqual(walsh_bma_predict_horizon(*args,c_slab=slab),
                                   reference(*args,c_slab=slab),places=14)



if __name__=='__main__':unittest.main()
