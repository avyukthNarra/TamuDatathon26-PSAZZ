# C1: exact Bayesian parity inversion — validation result

## Result

The exact inversion prototype scored **53.9639/60**, below the reproduced pair-mixer baseline (**56.2856/60**). Its paired 1000-resample, condition-stratified point change was **−2.3217** (95% CI **[−3.1629, −1.4748]**). This is a clear negative result for the full model on this validation set; do not replace the baseline with it.

| Forecast | Points / 60 | Δ points vs baseline [paired 95% CI] |
|---|---:|---:|
| Baseline pair mixer | 56.2856 | — |
| Exact model | 53.9639 | −2.3217 [−3.1305, −1.5699] |
| 50% baseline + 50% exact | 55.9159 | −0.3697 [−0.7259, −0.0171] |
| 80% baseline + 20% exact | 56.3381 | +0.0524 [−0.0787, +0.1977] |
| 80% exact + 20% baseline | 54.9191 | −1.3665 [−1.9997, −0.7970] |
| Exact win probability + baseline size mix | 56.3584 | +0.0728 [−0.4808, +0.5764] |
| Baseline win probability + exact size mix | 53.8769 | −2.4088 [−2.9931, −1.8963] |

The two modestly positive variants have intervals spanning zero, so neither establishes a gain. The component swaps suggest the size-mode posterior is the principal weakness: replacing only the baseline size mix causes a large loss, while exact win probability with baseline size is close to the baseline. That win-only swap has lower episode MSE on 51 episodes and higher error on 45, with a CI spanning zero. Exact has lower episode MSE than baseline on 27 episodes and higher error on 69; the baseline-win/exact-size swap loses on 75 and wins on 21. The 50/50 blend also loses on 57 episodes and wins on 39.

## By condition and type

Official condition errors use equal episode weighting, with 24 episodes and 192 queries in each condition. Ratios divide each forecast's error by the frozen reference error.

| Condition | Baseline error / ratio | Exact error / ratio | Exact win-probability MSE | Exact severity MSE* |
|---|---:|---:|---:|---:|
| Stable | .010622 / .05412 | .018183 / .09265 | .004464 | .014606 |
| Short history | .018568 / .10470 | .024997 / .14095 | .007695 | .017502 |
| Drift | .009262 / .04879 | .020682 / .10895 | .005664 | .016549 |
| Shared change | .011584 / .05635 | .021556 / .10486 | .003562 | .020012 |

| Machine type | Baseline MSE / ratio | Exact MSE / ratio | Exact win-probability MSE | Exact severity MSE* |
|---|---:|---:|---:|---:|
| Steady | .014932 / .06008 | .029910 / .12033 | .002630 | .030142 |
| Balanced | .012665 / .05725 | .023216 / .10494 | .004691 | .020504 |
| Jackpot | .012555 / .10699 | .013926 / .11868 | .008429 | .004577 |

The diagnostic decomposition reports squared error of `p(win)` and conditional payout-mix squared error weighted by the target win probability; the latter is a diagnostic, not an additive decomposition of the official MSE. It shows that exact `p(win)` is helpful on stable episodes and shared-change episodes but worse on short-history and drift, while payout severity error rises across all conditions and is especially large for steady machines.

## Model diagnostics and method

The evaluated artifact was the normalized 9×9-grid output in `outputs/validation-grid9/`, with 768 predictions over 96 episodes. Inference took **9.120 seconds**. The predictor reads validation `queries.csv` and `history.csv`; targets, conditions, and reference forecasts were loaded only afterward by the evaluator. The G1 parameter table and payout atoms summarize validation-target forensics, so this is not a strict train-only or unseen-validation estimate. No settings or blend weights were selected on validation.

The post-hoc episode scenario posterior (conditions were not supplied to the predictor) chooses the static/stable scenario for all 24 stable and 24 short-history episodes, and shared-change for all 24 actual shared-change episodes. It chooses shared-change for **all 24 drift episodes**, with mean drift posterior near zero. This points to drift discrimination as another failure mode; the condition labels were used only for this diagnostic after predictions were fixed.

As a separate post-hoc check, the validation target's conditional size vector was matched to its nearest documented small/big atom and compared with the history-only size posterior. The top posterior mode is correct on 524/768 machine queries, but is wrong with posterior ≥0.9 on 165 queries (of 578 high-confidence predictions). This quantifies the size-posterior failure without feeding targets into prediction or fitting parameters to validation.

The baseline file reproduces **56.2856/60** exactly via `slot_forecasting.contest.contest_score` with `max_points=60` (768 aligned query IDs). The evaluator strictly checks prediction, target, query, condition, and reference IDs. Paired bootstrap uses 1000 resamples, seed 314159, samples episodes with replacement within each condition, reuses identical episode draws across forecasts, and keeps official reference denominators fixed. Intervals are conditional on these observed reference error denominators and do not include denominator uncertainty.

Artifacts: `evaluate.py`, `outputs/evaluation-grid9-final/metrics.json`, and `outputs/evaluation-grid9-final/paired_episode.csv`. The paired CSV gives per-episode MSEs and deltas for every evaluated forecast. `outputs/validation-grid9/scenarios.json` is retained with candidate posterior diagnostics. No validation fitting or blend selection was performed; the G1 priors contain documented validation-derived forensic estimates.

## Prespecified train-only ablation

The common-mask ramp diagnostic improved train Brier MSE by .000640, which the parent judged too small to justify changing the primary model. The scored validation candidate remains the original grid-9 nuisance-ramp version above; no common-mask variant was scored or selected on validation.

## Recommendation

Do not ship or blend the full exact inversion into the current entry. Keep the baseline payout-size head: the exact size posterior is wrong with high confidence often enough to erase the near-baseline win-probability result. Future work should model the observed drift ramp and its timing directly from training histories, rather than using the current uninformative nuisance-window approximation; evaluate that as a new predeclared train-driven candidate.
