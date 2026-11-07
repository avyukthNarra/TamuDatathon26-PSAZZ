"""Combo of E9 + E10: per-machine ExtraTrees/logistic mix for P(win), then
short-history shrinkage toward the pooled_recent win chance.

MRO: ComboForecaster -> WinBlendForecaster (shrink and temperature) ->
MachineTreeForecaster (trees + logistic mix) -> ShapeHeadForecaster.
"""
from .machine_trees import MachineTreeForecaster
from .win_blend import WinBlendForecaster


class ComboForecaster(WinBlendForecaster, MachineTreeForecaster):
    pass
