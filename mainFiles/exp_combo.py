import exp_eval
from slot_forecasting.combo_trees_blend import ComboForecaster as F
exp_eval.CANDIDATES.update({
    "combo_mix50_s0.6": lambda: F(kind="extra", mix=0.5, short_alpha=0.6),
    "combo_mix50_s0.4": lambda: F(kind="extra", mix=0.5, short_alpha=0.4),
    "combo_mix50_s0.8": lambda: F(kind="extra", mix=0.5, short_alpha=0.8),
    "combo_mix30_s0.6": lambda: F(kind="extra", mix=0.3, short_alpha=0.6),
    "combo_mix50_s0.6_a0.9": lambda: F(kind="extra", mix=0.5, short_alpha=0.6, alpha=0.9),
    "combo_mix50_s0.6_leaf8": lambda: F(kind="extra", mix=0.5, short_alpha=0.6, min_leaf=8),
})
if __name__ == "__main__": exp_eval.main()
