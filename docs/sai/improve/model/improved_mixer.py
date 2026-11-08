"""Reusable model-level version of IMP_M5. Existing NumPy/SciPy/harness required.

Supply the SAME size_modes as v3. fit() reads train histories only. This module
never reads validation targets or scores a split. Grid results select defaults;
these are not independently confirmed settings. Rule blending is external.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from experiments import ROOT, train_structure, win_head
from harness.data import Split
from pranav_v3 import PairMixer, CLASS_INDEX
from slot_forecasting.schema import index_history, probabilities


def as_split(queries, history):
    queries = tuple(queries)
    index = index_history(queries,history)
    grid = np.full((len(queries),max(q.cutoff for q in queries)),np.nan)
    for i,q in enumerate(queries):
        for play in index.get((q.episode_id,q.machine_id),[]):
            if play.tick < q.cutoff:
                grid[i,play.tick] = play.payout
    a = {'qid':np.array([q.query_id for q in queries]),'episode':np.array([q.episode_id for q in queries]),
         'machine':np.array([q.machine_id for q in queries]),'mtype':np.array([q.machine_type for q in queries]),
         'signal':np.array([q.signal for q in queries]),'cutoff':np.array([q.cutoff for q in queries]),'grid':grid}
    return Split('model-input',a,{})


class ImprovedPairMixer(PairMixer):
    """v3 sizes + train empirical-Bayes pair/cell priors + jackpot Beta mixture."""
    def fit(self, queries, history, labels=None):
        queries,history = tuple(queries),tuple(history)
        super().fit(queries,history,labels)
        self.learned = train_structure(as_split(queries,history))[:3]
        return self

    def predict(self, queries, history, baseline=None):
        s = as_split(queries,history)
        pw = win_head(s,*self.learned,prior_power=4.,eta_short=16.,k_short=1.,k_long=.25,
                      beta_strength=6.,beta_eta=8.)
        out = {}
        for i in range(s.n):
            own = [None if not np.isfinite(x) else CLASS_INDEX[int(x)] for x in s.grid[i,:s.cutoff[i]]]
            raw = []
            self.walk(own,str(s.mtype[i]),int(s.cutoff[i]),lambda t,p:raw.extend(p) if t == s.cutoff[i] else None)
            size = np.array(raw[1:])/sum(raw[1:])
            p = np.maximum(np.r_[1-pw[i],pw[i]*size],1e-4)
            out[str(s.qid[i])] = probabilities(p/p.sum(),str(s.qid[i]))
        return out
