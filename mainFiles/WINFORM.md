# winform

Cutoff win probability is one logit-normal draw per machine, around a mean that depends on type, parity cell, and signal. The parity cell at the query is known from two lags. On drift and shared-change that cell is the fully swapped one: every fitted ramp ends at least 34 ticks before the cutoff, and every shared flip lands at least 55 ticks before it. A point mass at the cell mean leaves the machine shock on the table. Shrinking the logit toward the cell mean with the machine's own counts is the estimate that history can support.

Worktree `fresh-winform`, branch `sai/fresh-winform`. Script `mainFiles/winform_study.py`. Numbers below are from `mainFiles/runs/winform/winform.json`.

## How the ceilings were scored

Fit on fold A, score fold B, then swap. Folds alternate sorted `episode_id` within condition. The headline prior uses only the other fold's **stable and short-history** targets, so drift and shared-change are a transfer test. Pair and direction are marginalized from ticks before the cutoff (three pairs, two directions). Condition labels choose the regime hypothesis; they are not used as numeric features. Each scored vector keeps that query's true conditional size distribution, so the points isolate P(win). An oracle that also plugs in the true P(win) scores 60.000. The pooled starter, scored as itself, is 30.000. v13's full cross-fit forecast (estimated sizes and estimated win chance) is 57.500; that is a different object from these ceilings.

Bootstrap intervals resample episodes within condition, 2,000 draws, seed 314159.

| Generator | Points | 95% CI | Stable | Short | Drift | Shared |
|---|---:|---:|---:|---:|---:|---:|
| Logit-normal posterior, separate slopes | **58.194** | **57.813–58.541** | 0.0297 | 0.0396 | 0.0319 | 0.0228 |
| Same, one slope per type | 58.194 |  | 0.0298 | 0.0395 | 0.0320 | 0.0229 |
| Same, no signal | 58.155 |  | 0.0311 | 0.0407 | 0.0324 | 0.0226 |
| Point mass at the cell mean | 57.571 | 57.146–57.940 | 0.0363 | 0.0564 | 0.0385 | 0.0376 |
| Point mass, no signal | 57.103 |  | 0.0482 | 0.0641 | 0.0456 | 0.0450 |
| Posterior, but never flip drift or shared | 30.853 | 29.828–31.885 | 0.0297 | 0.0396 | 2.0490 | 1.6604 |
| Posterior, episode-fitted ramp (still u=1) | 58.227 |  | 0.0297 | 0.0396 | 0.0296 | 0.0228 |
| Posterior, train-history prior only | 58.226 | 57.897–58.526 | 0.0297 | 0.0395 | 0.0301 | 0.0225 |
| Posterior plus a cutoff slope | 58.169 |  | 0.0296 | 0.0416 | 0.0319 | 0.0229 |
| Posterior plus an episode shift from peers | 58.210 |  | 0.0298 | 0.0386 | 0.0319 | 0.0227 |
| True P(win) and true sizes | 60.000 |  | 0 | 0 | 0 | 0 |
| Pooled starter | 30.000 |  | 1 | 1 | 1 | 1 |

The separate-slope and shared-slope posteriors differ by 0.0004 points. Dropping signal costs 0.040. Replacing the posterior with the cell-mean point mass costs 0.623, mostly on short histories (ratio 0.0564 versus 0.0396). Treating drift and shared-change as unflipped stable machines drops the score to 30.853. Train-only means, with no validation targets in the prior, score 58.226.

Win-probability RMSE of the headline posterior:

| | Steady | Balanced | Jackpot |
|---|---:|---:|---:|
| Stable | 0.044 | 0.052 | 0.095 |
| Short history | 0.059 | 0.063 | 0.087 |
| Drift | 0.055 | 0.055 | 0.085 |
| Shared change | 0.037 | 0.063 | 0.072 |

## The functional form

For machine type `t`, signal `s ∈ {−1,0,1}`, and parity cell `c ∈ {dead, live}`:

```
logit p = a[t,c] + b[t,c] * s + ε
ε ~ continuous, sd σ[t,c], one draw per machine, constant through the episode
```

The cell at the cutoff is determined by lags `(a,b) ∈ {(1,3),(2,5),(4,7)}` and the machine's original direction. Drift and shared-change have already swapped direction, so the query uses the opposite cell. Payout 0 is the loss; a payout of 1 is a win for the parity.

Descriptive fit on every confident validation target (both folds, all conditions). This table is the population summary. The ceilings above do **not** fit on the targets they score; they use the other fold's stable and short-history rows only.

| Type | Cell | n | a | b per signal step | σ | p at −1 / 0 / +1 |
|---|---|---:|---:|---:|---:|---|
| steady | dead | 118 | −1.771 | 0.421 | 0.498 | 0.100 / 0.145 / 0.206 |
| steady | live | 146 | 3.025 | 0.272 | 0.504 | 0.940 / 0.954 / 0.964 |
| balanced | dead | 102 | −3.089 | 0.514 | 0.590 | 0.027 / 0.044 / 0.071 |
| balanced | live | 124 | 1.745 | 0.383 | 0.573 | 0.796 / 0.851 / 0.894 |
| jackpot | dead | 120 | −4.493 | 0.374 | 0.668 | 0.008 / 0.011 / 0.016 |
| jackpot | live | 87 | 0.319 | 0.383 | 0.712 | 0.484 / 0.579 / 0.669 |

Jackpot live at signal 0 is probability 0.579 (logit 0.319). The train-history estimate of that intercept is 0.312, the same center. Fold-specific target fits are noisier (jackpot live intercept 0.018 on A, n=30, and 0.252 on B, n=22) and do not beat the train means: 58.194 versus 58.226.

Pair identification from history, median likelihood gap over the second pair: stable 70.9 nats, drift 26.1, shared-change 25.8, short history 9.8. The cutoff parity itself is observed for all 768 queries (the last 40 ticks are present even on short histories). Direction is uncertain on 2.1% of stable machines, 8.9% of drift, 8.3% of shared-change, and 17.7% of short histories; the posterior averages those hypotheses instead of dropping them.

## Why the probabilities are continuous

Inside a type × cell × signal bucket the logit still has sd about 0.50 (steady), 0.59 (balanced), and 0.67–0.71 (jackpot), skew near 0, excess kurtosis about −0.6 to −1.2. That is one cloud per cell. A two-mean split of those residuals puts roughly half the machines on each side of a 0.8–1.2 logit gap and cuts the sd to about 0.27–0.35, which is what one bell looks like when cut in half. The fraction of targets closer to the live/dead midpoint than to their own cell is 2.7% stable, 0.6% short, 0.6% drift, 0.6% shared. Drift cutoffs are not blends sitting between the two levels.

The machine shock is only partly visible in a finite history. Moving from the point mass to the posterior is worth 0.623 points. The remaining 1.806 points to 60 are the part of ε the counts do not pin down. Jackpot RMSE stays near 0.09 in every condition; short-history steady RMSE is 0.059 against 0.044 on stable steady. The best estimate given the history is the posterior mean of `sigmoid(θ)` under the logit-normal prior and the binomial counts in the pure regime, marginalized over pair and direction.

The other cell's history residual has correlation −0.036 with the leftover probability error (n=476, stable/short/drift). A covariance between dead and live does not buy a better cutoff probability once the cutoff cell has its own counts. Raw correlations of the two history logits on stable train machines are 0.41 steady (n=70), 0.32 balanced (n=61), and −0.08 jackpot (n=34). Subtracting a delta-method sampling variance from those margins is not reliable (jackpot-dead excess variance goes negative; the disattenuated correlation exceeds 1), so the prior sd should be taken from the target table above, not from that subtraction.

Machines in an episode do share a logit shock: held-out residual correlation is 0.42 stable, 0.38 short, 0.43 drift, 0.46 shared. Adding the other seven machines' history offsets to the prior gains 0.016 points (58.210). The shared piece is already inside each machine's own spins, so the peers are redundant once that machine is updated. A cutoff slope fit on one fold flips sign on the other (−0.086, se 0.038, versus +0.055, se 0.037) and costs 0.026 points.

## Drift is finished at the cutoff

On each drift episode a shared linear ramp was fit from history only. Rates were frozen on the first 35% of ticks; later ticks scored `u(t) = clip((t − (cutoff − anchor)) / length, 0, 1)`.

Validation, 24 episodes: anchor mean 85.4 (sd 7.9), length mean 42.3 (sd 9.6). `u` at the cutoff is 1 on 24/24 episodes. The ramp's last partial tick is 34 to 56 ticks before the cutoff (median 42). The same fit on train is 24/24 finished (anchor median 87, length median 43). Against a no-flip model the fitted ramp gains 90.6 nats per machine on the post-35% window. On the last 20 ticks, u=1 beats u=0.9 by 88.9 nats in total.

The target agrees. Define `u` so that 0 is the pre-change level of the current parity and 1 is the swapped level, using only pre-change counts shrunk to the other fold's prior. Median implied `u` is 1.006 on drift (p10 0.902, p25 0.975, n=181) and −0.004 on stable (n=191). Using each episode's own ramp, which is still u=1 but with a pure tail that starts a little earlier than the fixed 96/34 window, scores 58.227 against 58.194. The extra 0.033 is a cleaner window, not a partial blend.

## Shared flip timing

One tick per episode, saturated two-rate likelihood, grid from 45% to 85% of the cutoff. Validation: fraction `c/cutoff` mean 0.661 (sd 0.063), median 0.642. Ticks before the cutoff: minimum 55, median 78, mean 76.2 (sd 14.9). The peak is a single tick (median width of the 1-nat neighborhood is 0). Mean gain versus no flip is 474 nats per episode. Train: fraction mean 0.673 (sd 0.050), minimum 55 ticks before the cutoff. The query is on the swapped cell. Uncertainty about `c` changes how many post-flip spins you count, not which level is active.

## What failed

- A second discrete level inside the cell. The residuals are one cloud.
- A partial drift blend at the query. Every episode has u=1, and the late window prefers u=1 to u=0.9.
- A cutoff or absolute-tick term. The slope changes sign across folds.
- An episode intercept on top of the machine posterior. Correlation 0.42 in the targets, +0.016 points from peers.
- A live/dead covariance, once the cutoff cell is updated. Residual correlation −0.036.
- Separate signal slopes. They differ in the table (steady dead 0.42 versus live 0.27) and change the ceiling by 0.0004 points.
- Reading the prior sd off a sampling-variance correction of history logits. Those excess variances are too small and sometimes negative. The target residual sds are the ones that match the clouds above.

## Corrections for the exact filter

Encode this at the cutoff. Regime inference can stay as it is; the correction is the state **given** the regime.

1. **Level mean.** `logit p = a[type, cell] + b * signal + ε`. Use the table in "The functional form" (jackpot live `a = 0.319`, probability 0.579 at signal 0). A single slope per type is enough: it matched separate slopes to 0.0004 points. Train-history intercepts already agree, including jackpot live at 0.312, so a target-free filter can keep its train means.
2. **Level sd.** Use the target residual sds: steady 0.50 / 0.50, balanced 0.59 / 0.57, jackpot 0.67 / 0.71 (dead / live). Do not use a history excess near 0.2. Independent normals per cell are enough for the cutoff probability; the other cell's residual does not predict the remaining error (correlation −0.036).
3. **Drift.** `u(cutoff) = 1` on 24/24 validation episodes and 24/24 train episodes. The ramp ends 34–56 ticks before the cutoff (median 42). Predict the fully swapped level. A train anchor/length near 87/43 or 91/48 is consistent with that, because anchor > length. Do not blend the two levels on the query.
4. **Shared change.** One tick, fraction about 0.66–0.67, residual sd about 11–15 ticks, and never closer than 55 ticks to the cutoff in train or validation. The query cell is fully flipped. The tick still matters for which history rows use the old direction.
5. **Do not add** a cutoff coefficient, a peer/episode shift, or a neighbour term. Signal stays: dropping it costs 0.040 points.
6. **Where the points still are.** With true sizes, this posterior scores 58.19 cross-fit and 58.23 train-only. The other 1.81 points are ε that short histories and jackpots do not identify. Perfect knowledge of P(win) with these true sizes scores 60.

## Reproduce

```sh
cd /Users/saisreed/Desktop/Coding/TamuDatathon26-PSAZZ/worktrees/fresh-winform
python3 -m unittest discover -s mainFiles -p test_winform.py -v
OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python3 mainFiles/winform_study.py --out mainFiles/runs/winform
```

Python 3.13.3. The study run that produced these numbers took about 8 seconds. Unit tests: 7 passed. Outputs stay in `mainFiles/runs/winform/` (git-ignored).
