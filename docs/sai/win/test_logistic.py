"""Numerical, sign-symmetry, missing-value and cutoff regression tests."""
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from logistic import feature_matrix,fit_laplace,objective,moderated,machine_rows,empirical_bayes,predict


class LogisticTests(unittest.TestCase):
    def setUp(self):
        r=np.random.RandomState(1729)
        self.x=np.column_stack([np.ones(100),r.choice([-1.,1.],size=(100,3))])
        self.y=r.binomial(1,expit(self.x@np.array([-.5,1.2,-2.,.3])))
        self.w=np.exp2(-np.arange(100)[::-1]/30.)
        self.var=np.array([2.,3.,3.,3.])

    def test_newton_matches_independent_optimizer(self):
        b,cov,info=fit_laplace(self.x,self.y,self.w,self.var)
        r=minimize(objective,np.zeros(4),args=(self.x,self.y,self.w,1/self.var),method='BFGS',tol=1e-9)
        np.testing.assert_allclose(b,r.x,atol=2e-6)
        self.assertLess(info['max_gradient'],1e-6)
        np.testing.assert_allclose(cov,cov.T,atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(cov).min(),0)

    def test_missing_is_zero_while_observed_loss_is_minus_one(self):
        x=feature_matrix([0.,np.nan,5.],[3],full=True)
        self.assertEqual(x.shape,(1,37))
        np.testing.assert_array_equal(x[0,:4],[1.,1.,0.,-1.])
        self.assertEqual(feature_matrix([0.],[0]).shape,(1,12))

    def test_zero_mean_prior_allows_sign_flips(self):
        a,c,_=fit_laplace(self.x,self.y,self.w,self.var)
        b,d,_=fit_laplace(self.x,1-self.y,self.w,self.var)
        np.testing.assert_allclose(a,-b,atol=1e-12)
        np.testing.assert_allclose(c,d,atol=1e-12)
        self.assertAlmostEqual(moderated(a,c,self.x[0])+moderated(b,d,self.x[0]),1.)

    def test_moderation_moves_uncertain_prediction_toward_half(self):
        b=np.array([2.]);q=np.ones(1)
        self.assertTrue(.5<moderated(b,np.array([[4.]]),q)<expit(2.))

    def test_post_cutoff_and_truth_never_affect_features_or_fit(self):
        s=SimpleNamespace(n=1,cutoff=np.array([30]),grid=np.r_[self.y[:30],np.nan,np.nan][None,:],mtype=['x'])
        a=machine_rows(s,True)
        s.grid[:,30:]=20.;s.truth=np.ones((1,5));s.condition=['fake']
        b=machine_rows(s,True)
        for first,second in zip(a[0][:4],b[0][:4]):np.testing.assert_array_equal(first,second)
        variance,history=empirical_bayes(a,None,iterations=2)
        pa,_=predict(a,variance,None);pb,_=predict(b,variance,None)
        np.testing.assert_array_equal(pa,pb)
        again,h2=empirical_bayes(a,None,iterations=2)
        np.testing.assert_array_equal(variance['x'],again['x'])

    def test_end_to_end_recovers_known_walsh_logistic_rule(self):
        rng=np.random.RandomState(298)
        sequence=np.zeros(1600)
        truth=np.full(1600,.5)
        for t in range(3,len(sequence)):
            parity=(1 if sequence[t-1]>0 else -1)*(1 if sequence[t-3]>0 else -1)
            truth[t]=expit(-.2+2.*parity)
            sequence[t]=rng.binomial(1,truth[t])
        for full in (False,True):
            x=feature_matrix(sequence,np.arange(8,1200),full)
            b,cov,_=fit_laplace(x,sequence[8:1200],np.ones(len(x)),np.full(x.shape[1],10.))
            test=feature_matrix(sequence,np.arange(1200,1600),full)
            p=np.array([moderated(b,cov,row) for row in test])
            self.assertLess(float(np.mean((p-truth[1200:])**2)),.004)



if __name__=='__main__':unittest.main()
