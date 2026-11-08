# J1: jackpot win error is mostly active-level estimation

Train evidence supports the existing pair rules. Across 224 jackpot train machines, the observed win rate is 0.2244 and the median machine has only 37 wins (range 0–102); 55 machines have at most 20 wins and 31 at most 10. Short histories have median 20 wins. Among 57 stationary jackpot machines, all 47 with strong pair-rule evidence (ΔBIC >10 against a constant rate) select a known pair: (1,3):17, (2,5):14, (4,7):16. Both polarities occur: 29 match-high and 18 differ-high.

No convincing model gain was found. A train-fitted active-level prior gives +0.048 points but its paired CI spans zero; retain mixer6 pending independent evidence.

## Train rule discovery

`structure.py` compares all 28 pair rules under win/loss and big-win (payout≥5) input codings, all 56 parity triples for each coding, full four-cell binary pairs, exact-payout categorical pairs, counts of active known pairs under all polarity choices, and the joint three-pair lookup. Unrestricted two-cell probabilities cover both parity polarities. Each machine’s best BIC rule, cell counts/rates, and temporal holdout selection are retained in `structure.json`.

Comparisons use the same ticks with current outcome and all eight lag payouts observed. ΔBIC = 2(loglik_model−loglik_constant)−(occupied_cells−1)log(n). Search multiplicity is not corrected; weak/zero-evidence winners do not identify a generator rule. The following temporal check chooses the family’s rule on the first 70% of matched train ticks and scores the final 30% with smoothed cell rates. It uses no validation targets.

| Train family | Stationary jackpot temporal log loss | All-condition jackpot temporal log loss |
| --- | --- | --- |
| known_win_pair | 0.271 | 0.870 |
| win_pair | 0.290 | 0.865 |
| win_triple | 0.417 | 0.619 |
| big_pair | 0.414 | 0.672 |
| big_triple | 0.431 | 0.673 |
| win_full_pair | 0.291 | 0.860 |
| payout_pair | 0.307 | 0.819 |
| active_pair_count | 0.375 | 0.683 |
| joint_three_pairs | 0.283 | 0.827 |

Known win-coded pairs beat triples, big-win coding and additional-level tables on stationary temporal holdout. Across all conditions, only 65.2% of unrestricted pair winners are known pairs, but whole-history fits mix regimes and zero-signal cases select arbitrary ties. Triples appear less bad than stale static pairs in changing regimes; this does not establish a different generator interaction. Current mixer6 already adapts online. There is no clean stationary evidence here for a jackpot-specific pair dictionary or more than two within-machine rule cells.

## Dev target distribution

Only the 72 dev episodes are described or scored. Each target is one query’s generating win probability; distinct values across machines do not prove more than two latent levels within a single machine. Gap clusters below are descriptive, joining consecutive sorted values separated by ≤0.05.

| Type | Dev queries / distinct values at 12 decimals | q10 | q50 | q90 | Gap clusters (count: min–max) |
| --- | --- | --- | --- | --- | --- |
| balanced | 160 / 160 | 0.0230 | 0.6478 | 0.9203 | 78: 0.009–0.186; 82: 0.608–0.957 |
| jackpot | 207 / 207 | 0.0055 | 0.0230 | 0.7041 | 122: 0.002–0.057; 85: 0.269–0.862 |
| steady | 209 / 209 | 0.1049 | 0.9050 | 0.9721 | 89: 0.043–0.378; 1: 0.430–0.430; 119: 0.872–0.985 |

Jackpot deciles q10…q90: 0.0055, 0.0075, 0.0114, 0.0146, 0.0230, 0.2852, 0.4281, 0.5663, 0.7041. The 122 low cases range 0.00246–0.05677; the 85 active cases range 0.26947–0.86201. At a diagnostic boundary of 0.15 inside that gap, mixer6 puts all 207 jackpot cases in the correct band. No dead/active band confusion is needed to explain the residual error. A 0.5 probability threshold would misleadingly count many low-active cases as state errors.

## Where mixer6 misestimates the level

| True win range | n | Mean truth | Mean forecast | Bias | Win MSE |
| --- | --- | --- | --- | --- | --- |
| 0-0.05 | 120 | 0.0130 | 0.0206 | +0.0076 | 0.000644 |
| 0.05-0.1 | 2 | 0.0552 | 0.0190 | -0.0362 | 0.001406 |
| 0.2-0.5 | 30 | 0.3854 | 0.5258 | +0.1403 | 0.032116 |
| 0.5-0.8 | 47 | 0.6364 | 0.6218 | -0.0147 | 0.010672 |
| 0.8-1.0001 | 8 | 0.8373 | 0.6533 | -0.1840 | 0.040447 |

Overall jackpot dev win MSE is 0.009028, versus balanced 0.004350 and steady 0.005130. Jackpot active rates vary much more widely than a single population median suggests. The baseline active prior has mean 0.6691 and strength 10, while 45 stationary train machines with supported, adequately populated known-pair cells give an empirical active-rate mean 0.5922. These estimates still have selection/binomial uncertainty.

| Jackpot grouping | n | Mean truth | Bias | Win MSE |
| --- | --- | --- | --- | --- |
| 64_observed | 53 | 0.2559 | +0.0361 | 0.010520 |
| dense | 154 | 0.2356 | +0.0064 | 0.008514 |
| cutoff_lt224 | 104 | 0.2022 | +0.0152 | 0.007228 |
| cutoff_ge224 | 103 | 0.2798 | +0.0128 | 0.010845 |
| at_most20_wins | 59 | 0.1937 | +0.0302 | 0.005933 |
| 21_to40_wins | 57 | 0.2865 | +0.0065 | 0.012651 |
| over40_wins | 91 | 0.2427 | +0.0081 | 0.008764 |
| signal=-1 | 58 | 0.1930 | +0.0189 | 0.003210 |
| signal=0 | 79 | 0.2220 | +0.0254 | 0.011451 |
| signal=1 | 70 | 0.3017 | -0.0030 | 0.011114 |

These are unadjusted descriptive groups: cutoff and signal associations mix different true levels/conditions and do not establish a causal signal effect. Signal priors and post-hoc recalibration were not retried.

## Minimal jackpot-only prior changes

`mixer6_copy.py` is pinned source 814cb60, changed only to use an absolute local schema import. Constructor-supplied priors change the jackpot active component only; dead and middle components, pair dictionary, memories, expert loss, size head and all other types remain unchanged. The mandatory original probability floor remains in place.

- LIVE2 / LIVE6 keep active mean 0.6691 but weaken concentration to 2 or 6.
- TRAIN estimates the active mean and concentration from stationary train high-cell rates, requiring both cells to have ≥10 observations. Method-of-moments variance subtracts the estimated binomial sampling variance; the resulting mean is 0.592207 and strength 21.2624. This is not a validation-mean correction.

| Variant | Dev points [CI95] | Paired Δ [CI95] | Ratios stable/short/drift/shared | Jackpot win MSE |
| --- | --- | --- | --- | --- |
| J1_BASE | 56.009 [55.308,56.666] | +0.000 [+0.000,+0.000] | 0.0596/0.1095/0.0482/0.0677 | 0.009028 |
| J1_LIVE2 | 55.866 [55.124,56.575] | -0.143 [-0.277,-0.025] | 0.0604/0.1098/0.0532/0.0725 | 0.010172 |
| J1_LIVE6 | 55.981 [55.270,56.652] | -0.029 [-0.073,+0.011] | 0.0598/0.1089/0.0496/0.0689 | 0.009262 |
| J1_TRAIN | 56.058 [55.405,56.684] | +0.048 [-0.074,+0.214] | 0.0583/0.1061/0.0488/0.0681 | 0.008655 |

Two fixed, condition-balanced 36-episode dev halves each choose the candidate on the opposite half, including the unchanged baseline. Both choose J1_TRAIN. The stitched score is 56.058 [55.405,56.684], paired Δ +0.048 [-0.074,+0.214]. This does not establish a gain: the interval spans zero.

| Jackpot condition | Baseline win MSE | TRAIN win MSE |
| --- | --- | --- |
| stable | 0.010023 | 0.009411 |
| short_history | 0.010520 | 0.009254 |
| drift | 0.006709 | 0.006905 |
| shared_change | 0.008992 | 0.009134 |

The train prior improves stationary/short jackpot errors but slightly worsens changing regimes. It reduces low-active MSE (q=.2–.5) from .03212 to .02460, while worsening q>.8 from .04045 to .05102: a level bias/variance tradeoff remains. No candidate is recommended as a statistically supported replacement.

## Reproduce and verify

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
nice -n 10 python3 docs/sai/round/jackpot/diagnose.py
nice -n 10 python3 docs/sai/round/jackpot/structure.py
nice -n 10 python3 -m unittest discover -s docs/sai/round/jackpot -p test_jackpot.py -v
nice -n 10 python3 docs/sai/round/jackpot/fix.py
nice -n 10 python3 docs/sai/round/jackpot/verify.py
```

Five tests pass: parity/polarity recovery, degenerate BIC safety, exact copied-source provenance, cross-type isolation and strict pre-tick causality. Baseline reproduction error is 2.22e-16. Two independent TRAIN forecasts and their saved CSV match exactly. All five CSVs contain 768 valid unique IDs; all 503 non-jackpot rows are literally byte-identical to the reference.

Inherited baseline size modes and win priors were fitted on all 96 validation targets and reproduced unchanged before testing new train-only parameters. New model selection/scoring is dev-only; no audit/all evaluator was called. The audit log was unchanged during the model run. No installs, commits, branch changes or shared-file edits; the orchestrator owns commits and changelog. Existing lint/CI gaps remain; no semver impact from this experiment.
