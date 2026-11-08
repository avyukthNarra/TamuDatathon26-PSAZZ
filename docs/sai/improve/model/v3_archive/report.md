# Model improvement handoff

All scores are 72-episode dev only; 1,000 episode-bootstrap replicates. Ratios are stable/short/drift/shared. All selected rule weights are 0.20.

| Model | Points [95% CI] | Difference vs reference [95% CI] | Ratios |
|---|---|---|---|
| TM_COMBINED3 | 54.774 [53.737,55.715] | reference | 0.066/0.149/0.076/0.091 |
| IMP_M1 | 54.836 [53.800,55.773] | +0.062 [+0.009,+0.136] | 0.065/0.144/0.076/0.091 |
| IMP_M2 | 55.006 [54.182,55.800] | +0.232 [+0.004,+0.517] | 0.061/0.132/0.079/0.091 |
| IMP_M3 | 54.975 [54.046,55.841] | +0.201 [+0.029,+0.417] | 0.066/0.133/0.076/0.091 |
| IMP_M4 | 55.001 [54.031,55.854] | +0.227 [+0.041,+0.454] | 0.061/0.140/0.077/0.084 |
| IMP_M5 | 55.109 [54.328,55.839] | +0.335 [+0.037,+0.706] | 0.060/0.131/0.079/0.085 |

M1 learns type-specific pair frequencies on train (prior power 4); its small gain disappears under fold selection.

M2 adds type-level cell pseudo-counts: 1 for 64-observation histories, 0.25 otherwise. It reduces short-history uncertainty; heavier shrinkage hurts.

M3 train check: peer pair agreement .340 versus null .333 (p=.196); type-centered rate covariance p=.013; changepoint alignment .417 versus .278 (p=.0033). Peer pooling gains -0.003 [-0.012,+0.004] versus matched type-only shrinkage: exclude it.

M4 jackpot-only Beta-mixture prior learns low/high win-cell levels .121/.612 from train; strength 6, expert learning rate 8. Jackpot train win rate is .224; sparse wins make cell estimates noisy. Dev jackpot p0 MSE falls .01143→.00955. Tested jackpot half-lives lose to this prior.

M5 combines M1/M2/M4: short-history ratio improves .149→.131. Reselecting every component and blend weight on opposite dev folds gives 54.997 [54.128,55.813], difference +.223 [.001,.501]. Selection remains exploratory; this is not independent confirmation.

Validation targets: inherited v3 size modes still read all validation targets, exactly as reference; new win structure/priors use train only. No audit/all scoring. No post-hoc recalibration duplicated.

Files: docs/sai/improve/model/ contains reusable ImprovedPairMixer, copied v3, runner, train statistics, complete 93-row grid/results, tests, verification and README; docs/sai/preds/IMP_M1_validation.csv through IMP_M5_validation.csv contain 768 rows each.

Checks: 4 tests passed; v3 reproduction max error 5.6e-16; reusable M5 reproduction 2.3e-16; two-run difference 0. No packages installed, existing files edited, Git mutations, commits or PR. Orchestrator owns changelog/commit. NumPy/SciPy research implementation; stdlib submission port skipped.
