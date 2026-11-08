# T4: Bayesian logistic win-function test

The requested continuous Gaussian-prior logistic construction does not improve mixer v4. Both feature sets have much worse dev win-probability MSE in every condition. Every six-fold blend selects zero logistic weight; opposite-half model/weight selection also retains the reference. Stop here, as requested when the 37-term EB model is clearly worse. This conclusion applies to this specification and memory grid, not every possible continuous win model.

## Model and train-only selection

At tick t, observed lag payouts become +1 for a win or −1 for a loss, and unobserved/negative-time lags become zero. The compact dictionary has bias, eight single lags and products for (1,3), (2,5), (4,7): 12 terms. The full dictionary has bias, eight singles and all 28 pair products: 37 terms. Each machine is fitted independently to its pre-cutoff observed outcomes; zeros in features are not fabricated training outcomes.

Each coefficient has a zero-mean Gaussian prior, allowing signs to flip between machines of the same type. For each type/term, five fixed approximate empirical-Bayes updates set variance to the mean Laplace posterior second moment beta² + diagonal(covariance) across train machines. Initial variance is 4; updates are clipped to [0.001,100]. Histories only enter these updates; train cutoff labels are reserved for selecting the forgetting half-life. No validation targets fit these priors.

Exponentially weighted likelihood uses exp2(−(cutoff−1−tick)/half_life), with inf, 120, 60, 30 tested. Damped Newton uses analytic gradients/Hessians, Cholesky solves and a Gaussian prior for positive curvature. Prediction is sigmoid(mu/sqrt(1+pi*v/8)), where mu=x beta and v=x covariance x. All randomness is confined to seeded synthetic tests; production fits are deterministic.

| Features | Half-life | Train next-outcome log loss |
| --- | --- | --- |
| KNOWN (12) | inf | 0.573297 |
| KNOWN (12) | 120 | 0.490639 |
| KNOWN (12) | 60 | 0.435915 |
| KNOWN (12) | 30 | 0.380507 |
| FULL (37) | inf | 0.587677 |
| FULL (37) | 120 | 0.512799 |
| FULL (37) | 60 | 0.466833 |
| FULL (37) | 30 | 0.427004 |

Half-life 30 is selected independently for both dictionaries by the smallest train next-outcome log loss. No dev-driven memory/prior tuning was performed. All fitted variances, five EB iterations and convergence diagnostics are retained in `results.json`.

## Dev win-only MSE

Only 72 dev episodes are used. MSE is against true win probability 1−p0. Differences use condition-stratified episode bootstraps with 1,000 draws, seed 314159; positive MSE differences are worse.

| Model | Overall MSE | MSE minus reference [95% CI] | Stable | Short history | Drift | Shared change |
| --- | --- | --- | --- | --- | --- | --- |
| TM_COMBINED4 | 0.007787 | 0 | 0.006582 | 0.009055 | 0.007793 | 0.007718 |
| KNOWN | 0.032501 | +0.024714 [+0.021502, +0.028064] | 0.012662 | 0.023708 | 0.053943 | 0.039688 |
| FULL | 0.045819 | +0.038032 [+0.033424, +0.042738] | 0.023279 | 0.039358 | 0.067365 | 0.053273 |

Neither model helps stable or gapped histories; absolute errors are largest for drift and shared change. The full model’s excess MSE CI lies entirely above zero, triggering the requested early stop. Optional fast/slow mixtures and additional dev tuning were not attempted.

## Plug-in and held-out win blends

For the plug-in, replace the reference win probability with the logistic probability while holding its conditional payout shares fixed: P=[1−pw, pw×shares]. A convex blend of these two vectors is therefore exactly a blend of the two win probabilities; it does not mix size heads. Six-fold episode OOF blend weights use the existing harness, fitting w on the other five folds over a 0.05 grid to minimize five-class squared error.

| Forecast | Points [95% CI] | Paired delta vs reference [95% CI] | Ratios stable/short/drift/shared |
| --- | --- | --- | --- |
| TM_COMBINED4 | 55.345 [54.538, 56.106] | +0.000 | 0.063/0.121/0.070/0.082 |
| KNOWN plug-in | 47.193 [46.213, 48.054] | -8.152 [-9.117, -7.130] | 0.106/0.238/0.431/0.311 |
| KNOWN 6-fold blend | 55.345 [54.538, 56.106] | +0.000 [+0.000, +0.000] | 0.063/0.121/0.070/0.082 |
| FULL plug-in | 43.721 [42.478, 44.791] | -11.623 [-12.876, -10.372] | 0.184/0.359/0.536/0.410 |
| FULL 6-fold blend | 55.345 [54.538, 56.106] | +0.000 [+0.000, +0.000] | 0.063/0.121/0.070/0.082 |

All twelve fold weights (six per family) are zero. A separate condition-balanced 36/36-episode opposite-half check chooses the dictionary and blend weight using only the other half. Both halves select zero logistic weight: 55.345 [54.538, 56.106], delta +0.000 [+0.000, +0.000]. It reproduces the baseline win-only MSE. No dev-tuned dictionary/weight gain survives this control.

## Reproduce and verify

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
nice -n 10 python3 -m unittest discover -s docs/sai/win -p test_logistic.py -v
nice -n 10 python3 docs/sai/win/run.py
nice -n 10 python3 docs/sai/win/verify.py
```

Six tests pass, including Newton parity with an independent optimizer, sign symmetry, missing-feature semantics, cutoff isolation, posterior moderation, and end-to-end recovery of a simulated Walsh logistic rule. Independent selected-model prior refits are byte-identical; two prediction runs reproduce the stored raw win probabilities exactly. All five saved CSVs have 768 unique expected IDs, finite normalized probabilities and unchanged conditional size shares. CSV rounding/normalization differs from the raw constructed plug-in by at most a few floating-point units.

`WIN_KNOWN` and `WIN_FULL` are standalone plug-ins. Their `_BLEND_OOF` forecasts and `WIN_SELECTED_OPPOSITE_OOF` use held-out predictions on dev; audit rows are unchanged reference forecasts and have not been scored. Standalone plug-in audit rows are predictions only. The audit access log was unchanged throughout the run. No audit/all evaluator was called.

The reference’s original Beta win prior and payout-size modes were learned from public validation targets. The new logistic priors and memory choices use train only; preserving reference size shares inherits the reference’s size-mode dependency. No new dependencies, CI, submission predictor or config were changed. There is no configured lint command or CI; compilation, whitespace, six tests and saved-forecast integrity checks passed. No semver impact or user-visible behavior change; this is a local research/test branch with no PR or push. `RESEARCH_TODO.md` remains untouched and untracked.
