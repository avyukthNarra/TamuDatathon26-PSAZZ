"""Regression and independence checks for the research predictor.

Run: python3 -m unittest discover -s mainFiles -p sysid_tests.py -v
"""
from dataclasses import replace
from itertools import product
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import expit,logsumexp
from slot_forecasting.schema import Query,Target
from sysid_study import Machine,NAMES,PAIRS,load_machines
from sysid_bayes import infer_rules,static_predict,complete_data
from sysid_dynamic import machine_grid
from sysid_size import size_predict
from sysid_joint import combine_full
from sysid_eval import select_cross_fitted,official_score,bootstrap


def synthetic_machines(n=24,cutoff=320):
    rng=np.random.default_rng(73218);machines=[];labels={};conditions={}
    pairs=((1,8),(2,6),(3,5))
    for i in range(n):
        q=Query(f'q{i}',f'e{i//8}',f'm{i%8}','balanced',0,cutoff)
        payout=np.zeros(cutoff+1)
        payout[:8]=rng.integers(0,2,size=8)
        a,b=pairs[i%3];direction=1 if i%2 else -1
        for t in range(8,cutoff+1):
            feature=(2*(payout[t-a]>0)-1)*(2*(payout[t-b]>0)-1)
            payout[t]=float(rng.random()<expit(-.4+3.*direction*feature))
        F=np.full((cutoff+1,len(NAMES)),np.nan)
        for t in range(8,cutoff+1):
            w=2*(payout[t-np.arange(1,9)]>0)-1
            large=2*(payout[t-np.arange(1,9)]>1)-1
            F[t]=np.r_[w,[w[a-1]*w[b-1] for a,b in PAIRS],large,
                        [large[a-1]*large[b-1] for a,b in PAIRS],np.zeros(24)]
        labels[q.query_id]=int(payout[-1]);payout[-1]=np.nan
        machines.append(Machine(q,np.arange(cutoff),payout,F));conditions[q.query_id]='stable'
    return machines,labels,conditions


def win_model(rule='w1*w8'):
    return {'ruleids':[NAMES.index(rule)],'ruleweights':[1.], 'mu':[-.4]*3,'sigma':0.,
            'signal':0.,'amplitude':3.,'commonids':[],'common_coefficients':[]}


class SysidTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.machines,cls.labels,cls.conditions=synthetic_machines()

    def test_structure_search_relearns_alternate_lags(self):
        with tempfile.TemporaryDirectory() as tmp:
            ids,weights,mu,amp=infer_rules(self.machines,self.labels,self.conditions,Path(tmp))
        self.assertEqual({NAMES[i] for i in ids},{'w1*w8','w2*w6','w3*w5'})
        self.assertAlmostEqual(sum(weights),1.)

    def test_exact_binary_state_sum_matches_enumeration(self):
        m=self.machines[0];model=win_model();F,pay,t=complete_data(m)
        f=F[:,model['ruleids'][0]]
        ll=np.array([np.sum((pay>0)*(-.4+s*3*f)-np.logaddexp(0,-.4+s*3*f)) for s in (-1,1)])
        post=np.exp(ll-logsumexp(ll));nf=m.features[m.query.cutoff,model['ruleids'][0]]
        expected=post@expit(-.4+3*nf*np.array([-1,1]))
        p,_=static_predict(model,m)
        self.assertAlmostEqual(p,expected,places=12)

    def test_missing_lag_is_excluded_and_forecast_marginalized(self):
        m=self.machines[0];F=m.features.copy();F[100,:36]=np.nan;F[m.query.cutoff,:36]=np.nan
        mm=replace(m,features=F);_,_,ticks=complete_data(mm)
        self.assertNotIn(100,ticks)
        p,_=static_predict(win_model(),mm)
        expected=.5*(expit(-.4+3)+expit(-.4-3))
        self.assertAlmostEqual(p,expected,places=12)
        e,pp=machine_grid(win_model(),mm)
        self.assertTrue(np.all(np.isfinite(pp)))
        self.assertTrue(np.all((pp>0)&(pp<1)))

    def test_size_prediction_without_positive_history(self):
        m=self.machines[0];empty=replace(m,payouts=np.zeros_like(m.payouts))
        modes=np.array([[[.2,.4,.3,.1],[.5,.25,.2,.05]]]*3)
        model={'ruleids':[0],'ruleweights':[1.],'modes':modes.tolist()}
        p,_=size_predict(model,empty)
        np.testing.assert_allclose(p,modes[0].mean(axis=0),atol=1e-12)
        F=empty.features.copy();F[empty.query.cutoff]=np.nan
        p,_=size_predict(model,replace(empty,features=F))
        self.assertTrue(np.isfinite(p).all());self.assertAlmostEqual(sum(p),1.)

    def test_future_observations_do_not_enter_prefix_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp)
            rng=np.random.default_rng(12);past=rng.choice([0,1,2],size=200)
            grids=[]
            for variant in range(2):
                split=parent/f's{variant}';split.mkdir()
                (split/'queries.csv').write_text('query_id,episode_id,machine_id,machine_type,signal,cutoff\nq,e,m,balanced,0,200\n')
                p=past.copy()
                if variant:p[100:]=20
                (split/'history.csv').write_text('episode_id,machine_id,tick,payout\n'+''.join(f'e,m,{t},{int(y)}\n' for t,y in enumerate(p)))
                ms,_,_=load_machines(parent,f's{variant}')
                grids.append(machine_grid(win_model(),ms[0],end_tick=100))
            np.testing.assert_array_equal(grids[0][0],grids[1][0])
            np.testing.assert_array_equal(grids[0][1],grids[1][1])

    def test_shared_change_matches_brute_force_joint_probability(self):
        rng=np.random.default_rng(5);e=rng.uniform(.1,1,size=(2,3));p=rng.dirichlet(np.ones(5),size=(2,3));prior=np.array([.5,.25,.25])
        got,r=combine_full([(np.log(e[i]),p[i]) for i in range(2)])
        Z=np.array([np.prod(e[:,0]),np.prod(e[:,1:].mean(axis=1)),np.mean(np.prod(e[:,1:],axis=0))])
        comp=np.zeros((3,2,5));comp[0]=p[:,0]
        for i in range(2):
            comp[1,i]=np.sum(e[i,1:,None]*p[i,1:],axis=0)/e[i,1:].sum()
            ww=np.prod(e[:,1:],axis=0);comp[2,i]=np.sum(ww[:,None]*p[i,1:],axis=0)/ww.sum()
        posterior=prior*Z/(prior@Z);expected=np.einsum('c,cnd->nd',posterior,comp)
        np.testing.assert_allclose(got,expected,atol=1e-12)
        np.testing.assert_allclose(r['model_posterior'],posterior,atol=1e-12)

    def test_crossfit_uses_opposite_fold_choice(self):
        qs=[Query('a','e0','m','steady',0,200),Query('b','e1','m','steady',0,200)]
        candidates={'A':{'a':[1,0,0,0,0],'b':[0,1,0,0,0]},'B':{'a':[0,1,0,0,0],'b':[1,0,0,0,0]}}
        p,choice=select_cross_fitted(candidates,qs,np.array([0,1]),{'A':[60,30],'B':[30,60]})
        self.assertEqual(choice['tune_A_predict_B'],'A');self.assertEqual(choice['tune_B_predict_A'],'B')
        self.assertEqual(p['a'],candidates['B']['a']);self.assertEqual(p['b'],candidates['A']['b'])

    def test_official_score_and_episode_bootstrap_anchor(self):
        qs=[];ts=[];groups={};pred={};ref={}
        for c in ['stable','short_history','drift','shared_change']:
            for i in range(2):
                key=f'{c}{i}';q=Query(key,key,'m','steady',0,200);qs.append(q)
                ts.append(Target(key,key,(.5,.2,.15,.1,.05),0));groups[key]=c
                pred[key]=(.55,.15,.15,.1,.05);ref[key]=(.6,.1,.15,.1,.05)
        score=official_score(pred,qs,ts,groups,ref)
        self.assertAlmostEqual(score['technical_points'],48.,places=12)
        ci=bootstrap(pred,qs,ts,groups,ref,repeats=200)
        np.testing.assert_allclose(ci,[48.,48.],atol=1e-12)


if __name__=='__main__':unittest.main()
