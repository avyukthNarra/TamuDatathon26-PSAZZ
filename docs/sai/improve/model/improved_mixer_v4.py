"""Research adapter for M1-M3 on pinned v4. Baseline defaults are preserved.

Pass v4's original size_modes and win_prior. These inherited parameters were
fitted on validation targets; this class fits only additional train statistics.
Use win_options={} for baseline; exploratory options are in v4_results.json.
"""
from experiments import train_structure,win_head
from improved_mixer import as_split
from pranav_v4 import PairMixer,CLASS_INDEX
from slot_forecasting.schema import probabilities
import numpy as np


class ResearchPairMixerV4(PairMixer):
    def __init__(self, *, size_modes, win_prior, win_options=None):
        super().__init__(size_modes=size_modes,win_prior=win_prior,eta=8.)
        self.population_prior=tuple(tuple(p) for p in win_prior)
        self.win_options={'eta_short':8.,'eta_long':8.,**(win_options or {})}

    def fit(self,queries,history,labels=None):
        queries,history=tuple(queries),tuple(history)
        super().fit(queries,history,labels)
        self.learned=train_structure(as_split(queries,history))[:3]
        return self

    def predict(self,queries,history,baseline=None):
        s=as_split(queries,history)
        pw=win_head(s,*self.learned,population_prior=self.population_prior,**self.win_options)
        result={}
        for i in range(s.n):
            own=[None if not np.isfinite(x) else CLASS_INDEX[int(x)] for x in s.grid[i,:s.cutoff[i]]]
            raw=[]
            self.walk(own,str(s.mtype[i]),int(s.cutoff[i]),
                      lambda t,p:raw.extend(p) if t==s.cutoff[i] else None)
            size=np.array(raw[1:])/sum(raw[1:])
            p=np.maximum(np.r_[1-pw[i],pw[i]*size],1e-4)
            result[str(s.qid[i])]=probabilities(p/p.sum(),str(s.qid[i]))
        return result
