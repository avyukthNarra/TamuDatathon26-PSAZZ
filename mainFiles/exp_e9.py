import exp_eval
from slot_forecasting.win_blend import WinBlendForecaster
exp_eval.CANDIDATES.update({
    "e9_a0.85": lambda: WinBlendForecaster(alpha=0.85),
    "e9_a0.7": lambda: WinBlendForecaster(alpha=0.7),
    "e9_t0.8": lambda: WinBlendForecaster(temp=0.8),
    "e9_t1.25": lambda: WinBlendForecaster(temp=1.25),
    "e9_short0.6": lambda: WinBlendForecaster(short_alpha=0.6),
    "e9_short0.3": lambda: WinBlendForecaster(short_alpha=0.3),
})
if __name__ == "__main__": exp_eval.main()
