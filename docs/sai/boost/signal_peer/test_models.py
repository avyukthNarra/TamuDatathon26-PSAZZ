"""Baseline reproduction and causal peer-trigger regression tests."""
import sys
import unittest
import numpy as np
from models import HERE,ROOT,METADATA,POP,win_walk,size_walk,trigger,fast_boost
from harness.data import Split
sys.path.insert(0,str(ROOT/'mainFiles'))
sys.path.insert(0,str(HERE/'improve/model'))
from pranav_v4 import PairMixer


class ModelTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.RandomState(72);grid=np.full((3,70),np.nan)
        cutoff=np.array([50,64,70])
        for i,c in enumerate(cutoff):
            grid[i,:c]=rng.choice([0,1,2,5,20],c)
            if i!=1:grid[i,np.arange(1,c,9)]=np.nan
        self.s=Split('synthetic',{'qid':np.array(['q0','q1','q2']),
            'episode':np.array(['e']*3),'machine':np.array(['m0','m1','m2']),
            'mtype':np.array(['steady','balanced','jackpot']),'signal':np.array([-1,0,1]),
            'cutoff':cutoff,'grid':grid},{})

    def test_batch_win_and_size_heads_match_original_v4_walk(self):
        win,_,pred,lw=win_walk(self.s);size=size_walk(self.s)
        for i,kind in enumerate(self.s.mtype):
            own=[None if not np.isfinite(x) else [0,1,2,5,20].index(x) for x in self.s.grid[i,:self.s.cutoff[i]]]
            raw=[]
            PairMixer(win_prior=POP.tolist(),size_modes=METADATA['size_modes'],eta=8.).walk(own,kind,int(self.s.cutoff[i]),
                lambda t,p:raw.extend(p) if t==self.s.cutoff[i] else None)
            np.testing.assert_allclose(win[i],1-raw[0],atol=3e-14,rtol=0)
            np.testing.assert_allclose(size[i],np.array(raw[1:])/sum(raw[1:]),atol=3e-14,rtol=0)
        np.testing.assert_allclose(fast_boost(pred,lw,np.zeros(3,bool),16),win,atol=2e-15,rtol=0)

    def test_peer_trigger_excludes_self_and_future_events(self):
        score=np.full((3,71),np.nan);threshold={k:.5 for k in self.s.mtype}
        score[0,49]=1. # Own evidence alone must not trigger peers.
        self.assertFalse(trigger(self.s,score,threshold,12,1)[0])
        score[1,51]=1. # Peer event AFTER machine0 cutoff must not be used.
        self.assertFalse(trigger(self.s,score,threshold,12,1)[0])
        score[1,50]=1.
        self.assertTrue(trigger(self.s,score,threshold,12,1)[0])
        self.assertFalse(trigger(self.s,score,threshold,12,2)[0])
        score[2,48]=1.
        self.assertTrue(trigger(self.s,score,threshold,12,2)[0])

    def test_signal_prior_changes_forecast_without_reading_targets(self):
        a=win_walk(self.s)[0]
        prior=np.broadcast_to(np.array([.8,.1,.1]),(3,3))
        b=win_walk(self.s,prior)[0]
        self.assertGreater(abs(a-b).max(),1e-5)
        self.s.truth=np.full((3,5),np.nan)
        np.testing.assert_array_equal(b,win_walk(self.s,prior)[0])

if __name__=='__main__':unittest.main()
