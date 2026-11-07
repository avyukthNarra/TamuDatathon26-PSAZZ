import exp_eval
from slot_forecasting.machine_trees import MachineTreeForecaster
exp_eval.CANDIDATES.update({
    "e10_extra": lambda: MachineTreeForecaster("extra"),
    "e10_rf": lambda: MachineTreeForecaster("rf"),
    "e10_extra_mix50": lambda: MachineTreeForecaster("extra", mix=0.5),
    "e10_rf_mix50": lambda: MachineTreeForecaster("rf", mix=0.5),
})
if __name__ == "__main__": exp_eval.main()
