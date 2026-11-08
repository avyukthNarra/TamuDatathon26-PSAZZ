# R8 log-pool screening (tuning split only)

Branch `avyukth/exp-r8-logpool`, commit `b724047`. Worktree `worktrees/r8-logpool`, file `mainFiles/r0_model.py`.

Points are /60 on the tuning split. Paired delta is vs the R0 control with the same prior source (95% bootstrap CI). Errors are stable / short / drift / shared.

| Variant | Train: pts | Train errors | Train paired delta [95% CI] | Validation: pts | Validation errors | Validation paired delta [95% CI] |
|---|---:|---|---|---:|---|---|
| a) linear w=0.2 T=1 (control) | 54.19 | .0196 .0224 .0241 .0205 | 0 | 55.26 | .0165 .0227 .0165 .0131 | 0 |
| b) log w=0.2 T=1 | 54.22 | .0196 .0224 .0238 .0202 | +0.04 [-0.00, +0.08] | 55.24 | .0167 .0229 .0164 .0130 | -0.01 [-0.06, +0.03] |
| c) log w=0.3 T=1 | 54.32 | .0182 .0230 .0232 .0199 | +0.13 [-0.00, +0.27] | 55.25 | .0157 .0232 .0166 .0134 | -0.01 [-0.14, +0.11] |
| d) log w=0.2 T=1.1 | 53.78 | .0212 .0246 .0254 .0221 | -0.41 [-0.60, -0.23] | 55.10 | .0173 .0237 .0169 .0134 | -0.16 [-0.32, +0.01] |
| e) linear w=0.2 T=0.9 | 54.25 | .0194 .0215 .0244 .0202 | +0.06 [-0.14, +0.29] | 55.03 | .0176 .0230 .0176 .0142 | -0.22 [-0.42, -0.02] |
| e) linear w=0.2 T=1.1 | 53.70 | .0213 .0249 .0258 .0226 | -0.49 [-0.70, -0.30] | 55.07 | .0172 .0237 .0172 .0136 | -0.18 [-0.37, -0.00] |

Control reproduced: train 54.19 (R0 54.185), validation 55.26 (R0 55.256).

No variant beats the control on tuning with a CI clear of zero. Audit numbers were printed by the harness but not used to choose anything.

Factories: `r0_model:make_blend_*` (train) and `r0_model:make_blend_*_validation` (validation priors injected from `public/validation/targets.csv`). Existing tests pass (4).
