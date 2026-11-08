# Signal-shifted win prior (sig-prior)

Branch `avyukth/exp-sig-prior`, commit `e2b7e06` (pushed). Option `signal_shift`, default off. Base: validation-prior 20% rule blend. Six harness runs, tuning split only; audit not used for selection.

| Config | Tuning /60 | Δ vs 55.256 | Paired 95% CI | Errors: stable / short / drift / shared |
|---|---:|---:|---|---|
| Baseline (validation blend) | 55.256 | - | - | - |
| d=0.05 | 55.269 | +0.013 | [-0.012, +0.048] | .0163 / .0226 / .0165 / .0132 |
| d=0.10 | 55.280 | +0.024 | [-0.029, +0.099] | .0160 / .0225 / .0165 / .0133 |
| d=0.15 | 55.289 | +0.033 | [-0.051, +0.154] | .0157 / .0225 / .0165 / .0135 |
| Per-type fitted (steady .094, balanced .082, jackpot .101) | 55.281 | +0.025 | [-0.025, +0.093] | .0161 / .0225 / .0165 / .0133 |
| Variant B: d=0.15, <100 ticks only | 55.265 | +0.009 | [-0.015, +0.035] | .0165 / .0225 / .0165 / .0131 |
| Train-prior blend (reference) | 54.185 | -1.071 | [-1.435, -0.641] | .0196 / .0224 / .0241 / .0205 |

Baseline and train-blend scores match the R0 report.

Finding: the shift helps monotonically in d, but every paired CI includes zero, and the best fixed value gains at most +0.03 points. Gating to short histories (variant B) gives less than the unconditional shift. The mechanism is real but negligible for this score. Defaults are unchanged.
