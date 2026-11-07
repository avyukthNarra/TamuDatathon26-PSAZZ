"""Synthetic causality and zero-change checks; no validation data/metrics."""
import sys
from pathlib import Path
import unittest
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[3]/'mainFiles')]
from _team_mixer_source import PairMixer
from TEAM_VOLATILITY import VolatilityMixer

def forecast(model,own,stop):
    captured=[]
    model.walk(own,'synthetic',stop,lambda t,p:captured.append(tuple(p)))
    return captured

class VolatilityTests(unittest.TestCase):
    def test_unchanged_control_matches_vendored_walk_with_missing(self):
        own=[(None if i%11==0 else (i*7)%5) for i in range(121)]
        self.assertEqual(forecast(PairMixer(),own,120),forecast(VolatilityMixer(),own,120))
    def test_same_and_future_ticks_never_change_emitted_forecast(self):
        old=[i%5 for i in range(101)]
        changed=old.copy();changed[70:]=[None]*31
        self.assertEqual(forecast(VolatilityMixer(adaptive=True,alpha=.05),old,70),forecast(VolatilityMixer(adaptive=True,alpha=.05),changed,70))
    def test_adaptive_gate_warmup_keeps_control(self):
        own=[i%5 for i in range(25)]
        self.assertEqual(forecast(VolatilityMixer(),own,24),forecast(VolatilityMixer(adaptive=True,alpha=.05),own,24))
    def test_missing_is_not_a_loss(self):
        own=[i%5 for i in range(71)];gap=own.copy();gap[30:60]=[None]*30
        self.assertNotEqual(forecast(VolatilityMixer(),gap,70)[-1],forecast(VolatilityMixer(),own,70)[-1])
if __name__=='__main__':unittest.main()
