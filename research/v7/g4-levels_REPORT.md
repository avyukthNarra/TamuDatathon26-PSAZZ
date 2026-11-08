# Win-level variation (validation targets)

The payout-size mix really is two atoms per machine type. The win probability is not. It is a live/dead parity of one lag pair, plus a level inside that cell that moves by machine and is only weakly tied to signal. Combined-v6 already tracks that level (p_win RMSE 0.074, R² 0.964). An explicit formula fit on histories does not beat it. The slice it still misses is jackpot-live.

Half-lives, L2, and feature sets were chosen on a train temporal holdout (last 35% of each machine’s ticks). Validation `targets.csv` (768 queries) was the exam. Even/odd episode splits are marked as cross-fit and were not used to pick the formula.

## 1. Where combined-v6’s error sits

Query-mean squared error of `runs/combined-v6/validation_predictions.csv` against the five true probabilities. `SE|true p` replaces the forecast’s win probability with the truth and keeps its size mix. `SE|true size` does the opposite. `frac p` is the share of SE removed by a perfect win probability.

| condition | type | n | SE | SE if true p_win | SE if true size | p_win MSE | frac removed by true p | frac removed by true size |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | all | 768 | 0.01251 | 0.00502 | 0.00757 | 0.00552 | 0.60 | 0.40 |
| all | steady | 269 | 0.01302 | 0.00672 | 0.00658 | 0.00437 | 0.48 | 0.49 |
| all | balanced | 234 | 0.01216 | 0.00686 | 0.00543 | 0.00391 | 0.44 | 0.55 |
| all | jackpot | 265 | 0.01230 | 0.00168 | 0.01047 | 0.00810 | 0.86 | 0.15 |
| stable | all | 192 | 0.01062 | 0.00275 | 0.00783 | 0.00557 | 0.74 | 0.26 |
| short_history | all | 192 | 0.01857 | 0.00952 | 0.00918 | 0.00675 | 0.49 | 0.51 |
| drift | all | 192 | 0.00926 | 0.00272 | 0.00666 | 0.00489 | 0.71 | 0.28 |
| shared_change | all | 192 | 0.01158 | 0.00510 | 0.00660 | 0.00485 | 0.56 | 0.43 |
| short_history | steady | 61 | 0.02465 | 0.01569 | 0.00934 | 0.00638 | 0.36 | 0.62 |
| stable | jackpot | 58 | 0.01287 | 0.00147 | 0.01115 | 0.00861 | 0.89 | 0.13 |

Official contest score of v6 is **56.29 / 60**. Plugging in the true win probability and keeping v6’s size mix scores **58.44**. That is the ceiling of any win-level work that does not also touch size: **+2.15 points**. Loss ratio falls from about 0.066 to 0.027. The last 1.6 points to 60 are size error.

Jackpot is the win-probability problem (86% of its SE). Short-history steady is a size problem (true p_win removes only 36% of SE 0.025). Globally, win probability is 60% of SE and size is 40%, not “nearly all.”

v6’s p_win itself: MAE 0.047, RMSE 0.074, R² 0.964, correlation 0.982 with the truth. Decile calibration is within ±0.02. The residual is scatter, not bias.

## 2. Size is two atoms

Every validation conditional size vector `(p_1,p_2,p_5,p_20)/p_win` matches one of two vectors per type, to 1e-9. The two-decimal figures in the brief are rounded. Nothing sits between the atoms (the mixture weight is 0 or 1 for all 768).

| type | atom A (n) | atom B (n) |
|---|---|---|
| steady | 0.7539893896, 0.1968084883, 0.0482180796, 0.0009840424 (137) | 0.4551220076, 0.4359023939, 0.1067960865, 0.0021795120 (132) |
| balanced | 0.4998531085, 0.3479282724, 0.1304731021, 0.0217455170 (131) | 0.2140661471, 0.5467365933, 0.2050262225, 0.0341710371 (103) |
| jackpot | 0.3351460319, 0.2449461988, 0.3149308270, 0.1049769423 (147) | 0.1207867301, 0.3239206784, 0.4164694436, 0.1388231479 (118) |

Steady’s 20× weight is about 0.001, not zero. Size state (big wins at lags 1, 2, 4) does not move p_win: those coefficients are +0.03 to +0.05 on the logit scale, and adding them to v6 in cross-fit changes p_win MSE by −0.3%.

## 3. What the level is

All 768 win probabilities are distinct out to 8 decimals. The level is continuous, not a small lookup table.

The largest gap inside each type splits dead from live at 0.651 (steady, gap 0.44), 0.378 (balanced, gap 0.38), and 0.163 (jackpot, gap 0.21).

**State.** A lag-pair parity, re-chosen with a decayed window, matches that split on 97.4% of queries where the pair is observed (stable 0.977, short 0.948, drift 0.983, shared_change 0.988; 691/768 observed). The winning pairs on the full history are (2,5) 226, (4,7) 205, (1,3) 163, then a tail. Canonical pairs are the best pair on 77% of machines, but only 58% of jackpots (jackpot separation is weaker, so the argmax is noisier). A single lag separates outcomes about half as hard as the best pair (mean separation score 3.1 vs 5.7).

Orienting the pair on the **whole** history matches the live/dead split only 60% of the time, because drift and shared change reverse the sign and the early counts cancel the late ones. Any formula that freezes the sign on the full history is wrong on those episodes.

**Level inside the cell.** Once the recent pair’s side is known, the standard deviation of true p_win falls from 0.37 to 0.12. Signal and a second pair barely move that:

| conditioning | within-cell SD of true p_win |
|---|---:|
| type only | 0.370 |
| type + recent side | 0.119 |
| type + side + signal | 0.114 |
| type + side + second pair | 0.112 |
| stable, type + side | 0.113 |
| short_history, type + side | 0.148 |
| drift, type + side | 0.112 |
| shared_change, type + side | 0.082 |

By the truth-side split (spread of the level, not pair-assignment error):

| cell | n | true p_win | v6 p_win | v6 RMSE | v6 R² inside the cell |
|---|---:|---|---|---:|---:|
| jackpot dead | 156 | 0.014 ± 0.011 | 0.019 ± 0.023 | 0.025 | negative (truth is tighter than v6) |
| jackpot live | 109 | 0.579 ± 0.164 | 0.596 ± 0.105 | 0.137 | 0.30 |
| steady dead | 120 | 0.162 ± 0.079 | 0.170 ± 0.089 | 0.088 | negative |
| steady live | 149 | 0.947 ± 0.027 | 0.943 ± 0.032 | 0.040 | negative |
| balanced dead | 108 | 0.054 ± 0.039 | 0.056 ± 0.045 | 0.052 | negative |
| balanced live | 126 | 0.834 ± 0.085 | 0.832 ± 0.061 | 0.070 | 0.32 |

Jackpot-live is the wide band in the brief (about 0.3–0.9). Steady-dead is the 0.1–0.4 band. Balanced-live is 0.6–1.0 but tighter than jackpot (SD 0.085). Three cells are nearly constants: jackpot-dead, steady-live, balanced-dead. v6’s error there is slightly larger than the truth’s own spread, so a constant would beat v6 on those cells. The point value of that is small (a few percent of global p_win MSE).

Jackpot-live holds about half of v6’s p_win MSE: 109/768 × 0.137² ≈ 0.0027, against a global p_win MSE of 0.00552.

**Signal.** Real, modest, and mostly already in v6. On stable episodes, in the recent high or low cell:

| type | signal −1 | signal 0 | signal +1 |
|---|---|---|---|
| balanced live | 0.78 ± 0.06 (n=10) | 0.88 ± 0.06 (11) | 0.90 ± 0.06 (15) |
| jackpot live | 0.44 ± 0.12 (3) | 0.46 ± 0.12 (7) | 0.63 ± 0.18 (10) |
| steady dead | 0.14 ± 0.06 (11) | 0.15 ± 0.05 (11) | 0.23 ± 0.09 (7) |

A pooled logistic on train (below) gives about +0.15 to +0.28 logit per signal step. From −1 to +1 that is ~0.14 probability near p=0.58, which matches the jackpot table. Cross-fitting signal on top of v6 removes **1.8%** of v6’s p_win MSE.

**Machine identity.** On a stable episode the two cell rates are a property of the machine. Estimating them from the early 55% of ticks and reading the current cell gives p_win RMSE 0.099 on stable queries, against v6’s 0.076. On stable steady it beats v6 (0.052 vs 0.076): v6 shrinks a nearly stable rate too hard. On stable jackpot the same formula is worse (0.159 vs 0.100), because a jackpot live cell has few wins and the empirical rate is noisy (binomial SE around 0.13 at n≈15, p≈0.6).

**Drift and shared change.** The sign flips (73% of drift machines, 88% of shared-change, 0% of stable, at a 55% time split). Keeping the early gap and only flipping the sign cuts RMSE from ~0.72 (stale cell) to 0.35 on drift and 0.23 on shared change. v6 is at 0.065–0.070. The regime change is not a pure reflection of a fixed strength. The level is redrawn. A machine’s strength does not survive the break.

**Neighbours, size-mode, extra lags, clock time.** Train logits see a neighbour term (mean of the other seven machines’ lag-1 wins, +0.30 logit) and small single-lag terms (+0.05). On validation, cross-fit on top of v6:

| added to logit(v6), even/odd | held-out p_win RMSE | fraction of v6 p_win MSE removed |
|---|---:|---:|
| nothing (recalibrated logit) | 0.0743 | 0.000 |
| signal | 0.0736 | +0.018 |
| recent cell rate | 0.0739 | +0.010 |
| neighbours | 0.0744 | −0.005 |
| big-win lags 1, 2, 4 | 0.0744 | −0.003 |
| time | 0.0744 | −0.004 |
| eight single lags | 0.0759 | −0.044 |
| recent rate + signal | 0.0729 | **+0.036** |

The best cross-fit correction removes 3.6% of v6’s win-probability MSE. Neighbours and size state add nothing once v6 is in the model.

**Logit symmetry of the two cells** (history rates, both cells n≥12): the gap averages 2.6–2.8 logits with SD 1.2–1.6, and the sum (twice the center) has SD 0.9–1.0. Machines differ in both strength and center. Jackpot centers vary most.

## 4. The formula, and its residual

Train late-window log loss picks a decayed parity-cell rate with half-life 12 (log loss 0.454, better than 8, 20, 40, 80). At the cutoff, take the lag pair whose decayed agree-vs-differ gap is largest, and predict that pair’s decayed win rate in the parity the cutoff actually falls in. Laplace (0.5)/(n+1) on the decayed counts.

That is the same family as v6 with one pair and one half-life, and no Beta pull toward a type level.

| predictor | p_win RMSE | p_win MAE | fraction of v6 p_win MSE removed |
|---|---:|---:|---:|
| combined-v6 | 0.074 | 0.047 | 0 |
| decayed cell rate, hl 12 (train-selected) | 0.191 | 0.123 | **−5.63** |
| same, hl 20 | 0.159 | 0.104 | −3.59 |
| same, hl 40 | 0.212 | 0.151 | −7.18 |
| canonical pairs only, hl 12 | ~0.19 | | worse than v6 in every condition |
| fixed dead/live constants (0.149/0.956, 0.039/0.840, 0.013/0.669) | 0.480 | 0.330 | −40.7 |
| per-machine logistic on 3 pairs, L2 chosen on train | 0.339 | 0.288 | −19.8 |
| pooled logistic + per-machine intercept | 0.379 | 0.299 | −25.0 |

By condition, half-life 12 vs v6 RMSE: stable 0.189 vs 0.075, short 0.205 vs 0.082, drift 0.185 vs 0.070, shared_change 0.186 vs 0.070. Jackpot is the worst type for the raw rate (0.270 vs 0.090).

Why the raw rate loses: effective n in the current cell is about 7–15, so the frequency wobbles by ~0.15, while the true level inside a regime is much smoother. v6’s three half-lives, three pairs, and Beta prior are what pull that noise down. Shrinking the hl-12 rate toward a cross-fit type×side×signal mean (prior strength grid 0…200, chosen per fold) only gets RMSE from 0.180 to 0.174.

The static logistic that train supports as a description of the spins, not as a cutoff forecast, is

```
logit p = α_type + β_type * signal + γ1_type * z1 + γ2_type * z2 + γ3_type * z3
          + 0.005 * (t/256) + 0.045 * lag1 + 0.069 * lag2 + 0.059 * lag4
          + 0.03 * big1 + 0.04 * big2 + 0.05 * big4
          + 0.305 * nbr_all_lag1
```

with z1, z2, z3 the machine’s top three pairs, oriented so +1 is the high cell, each zi in {−1,+1}. Coefficients (train, L2=10, the train-best spec):

| | intercept | signal | z1 | z2 | z3 |
|---|---:|---:|---:|---:|---:|
| steady | +0.32 | +0.20 | +1.04 | +0.34 | +0.26 |
| balanced | −0.34 | +0.15 | +0.96 | +0.32 | +0.27 |
| jackpot | −1.21 | +0.28 | +0.91 | +0.38 | +0.31 |

A per-machine intercept on top of a shared (z, signal) slope has SD 0.35 (steady), 0.39 (balanced), **0.66 (jackpot)**. That SD is the level variation: jackpot machines do not share one live probability. Evaluated at the validation cutoff with orientation from the full history, this logistic has RMSE 0.38, because the frozen sign is wrong after a reversal. Train late-window log loss already knew that: heavy L2, which ignores the pair, beat a sharp pair coefficient, and an EWMA beat both. The late window is where rules flip, so a stable logistic is punished.

On **stable** episodes only, the explicit machine formula “early-window cell rates of the best pair” is close to the truth and, for steady, better than v6:

| stable | n | cell-rate RMSE | v6 RMSE |
|---|---:|---:|---:|
| steady | 69 | 0.052 | 0.076 |
| balanced | 65 | 0.075 | 0.052 |
| jackpot | 49 | 0.159 | 0.100 |
| all stable | 183 | 0.099 | 0.076 |

Same formula on drift: RMSE 0.29–0.41. It is not a global replacement.

## 5. What to change in the model

1. **Keep the online pair mixer.** A closed form in lags, signal, neighbours, and size state does not reproduce the targets more closely. The train-selected formula has 6.6× v6’s p_win MSE. The richest cross-fit on top of v6 removes 3.6%.

2. **Jackpot live is the only cell worth chasing.** Truth SD 0.16, v6 SD 0.11, R² 0.30, RMSE 0.137, about half of all p_win MSE. The prior that piles mass near ~0.67 is too tight: train machine-intercepts already spread by 0.66 logit, which at the jackpot center is a wide probability band. Widen the jackpot **live** component (lower strength, or a continuous prior on the intercept with SD near 0.6 logit). Do not widen jackpot dead: truth is 0.014 ± 0.011 and v6 is already noisier than that.

3. **Steady live can be a constant.** Truth 0.947 ± 0.027. v6 RMSE 0.040 there is over-dispersion. A stronger pull to ~0.95 on the steady high cell, and the stable-steady result above (cell rate RMSE 0.052 vs v6 0.076), both say the steady high level is tighter than v6’s moving estimate. The global point value is a few percent of p_win MSE, not a new model.

4. **Do not add neighbour features, size-mode features, or a stack of single lags to the win head.** They lose the cross-fit. The second and third pairs are real on train (+0.3 logit, about a third of the primary pair) but they do not reduce validation within-cell SD (0.119 → 0.111) and they are already available to v6’s expert mix.

5. **Reset the level at a break; do not only flip the sign.** Sign flips are common (drift 73%, shared change 88%) and the old gap is the wrong magnitude afterwards (RMSE still 0.23–0.35). v6’s short half-life already re-estimates the level. A hard changepoint that reuses pre-break strength will not.

6. **Budget.** Perfect p_win is +2.15 points (56.29 → 58.44) with the current size mix. Short-history steady still has SE 0.025 after a true win probability, because the size mix is wrong. That cell is the size-side counterpart of jackpot-live. The exact atoms above are stable enough to hard-code; the remaining size error is uncertainty about which atom, especially when the short history does not show lags 1, 2, and 4 cleanly.

## What was run

- `probe_levels.py` — uniqueness, rounded-mode check, error split.
- `analyze_levels.py` — pair scan, variance tables, train-selected logistics, contest rescore of hybrids (`pred_*.csv`). Official v6 rescore 56.286.
- `analyze_recent.py` — decayed pair reselection, half-life chosen on train, cross-fit of v6 residuals, calibration.
- Inline checks — within-cell SD, shrinkage grid, early-magnitude / late-sign formula.

No hyperparameters were picked by minimizing validation target error. The validation tables for non-selected half-lives and L2 values are sensitivity, not the selection.
