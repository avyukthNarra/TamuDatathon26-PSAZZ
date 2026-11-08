# G6: episode-level regime pooling

## Result

Swap the pair-mixer’s parity counts at one episode-level flip tick, then let the existing mixture keep learning. Validation score **56.46 / 60**, against the reproduced v6 baseline **56.2856** (the brief’s 56.29).

| | Points | Δ vs 56.2856 | Paired 95% CI |
|---|---:|---:|---:|
| v6 pair mixer (reproduced, max abs diff 1e-15) | 56.2856 | — | — |
| **Swap at the joint MAP tick** | **56.4618** | **+0.176** | **[+0.080, +0.292]** |
| Half-life pooling, even/odd cross-fit | 56.280 | −0.006 | [−0.030, +0.018] |
| c1 win blend, even/odd cross-fit | 56.287 | +0.001 | [−0.516, +0.441] |

The swap was the same forecast on both validation halves (even 56.632 vs 56.447; odd 56.319 vs 56.153). Stable and short-history episodes are untouched. The gain is drift and shared change.

Official file score: `slot_forecasting.contest.score_contest_files(..., max_points=60)` on `outputs/validation_flip_crossfit.csv`.

## What was run

Code: `model.py` (v6 walk plus three hooks) and `run.py`. The walker was checked against `runs/combined-v6/validation_predictions.csv` before any variant was scored (max absolute probability difference 9.99e-16).

v6 is `adapt()` on train: pairs rediscovered as (1,3), (2,5), (4,7), parity cells, half-lives 12/40/100, eta 6, size modes and the per-type win prior read from validation targets exactly as v6 does. Those priors are not retuned here. New constants are either fit on train or chosen by even/odd cross-fit within condition (12 episodes per condition per half). Bootstrap: 1000 resamples of episodes inside each condition, seed 314159, same draws for every forecast.

## Detector

For each machine and each train-discovered pair, prefix sums give the profile likelihood of “direction flipped at tau” versus “one direction the whole time”, marginalizing the two directions with a Beta(1,1) pseudocount. The machine keeps the pair with the higher evidence. The episode likelihood sums the machines at one shared tau. The prior on tau is flat over ticks that have at least 8 observed parity rows on each side. Nothing is pinned to a fraction of the episode.

Train log Bayes factor (flip versus stable), 24 episodes per condition:

| Condition | logBF mean (min, max) | Joint posterior sd / cutoff | Within-episode sd of per-machine MAP / cutoff | MAP tau / cutoff |
|---|---|---:|---:|---:|
| Stable | −92 (−122, −63) | 0.065 | 1.00 (machines do not agree) | scattered |
| Short history | −56 (−86, −33) | 0.004 | 1.00 | near the end, on a thin sample |
| Drift | +353 (+238, +508) | 0.0066 | 0.059 | 0.70 |
| Shared change | +496 (+388, +643) | 0.00002 | 0.0049 | 0.67 |

Validation tells the same story (shared min logBF +372, drift min +250, stable max −44, short max −26). The widest gap in the **train** logBF list runs from −32.5 to +237.9; its midpoint is **102.7**. Every precommitted gate inside that gap (40 and 120, with or without a posterior-width cap, with or without the short-history filter) fired on the same 48 validation episodes: all drift and all shared change, none of the stable or short-history episodes. The short-history filter is the midpoint of the widest gap in train missing-fractions (0.334). It does not change the swap, because those episodes already have a negative logBF.

Joint posterior width does **not** separate drift from a shared flip. Both regimes move all eight machines together, so one tau fits both. Per-machine MAP ticks do separate them (train: shared max 0.015, drift min 0.040). That split was not used. Swapping helps both regimes, so cutting drift out would throw away part of the gain.

## (1) Pooling half-life weights

At each tick, or only at the cutoff, each machine’s log-probabilities over the half-lives are shrunk toward the episode mean. Pair weights inside a half-life are left alone. Shrinkage 0.15–1.0, uniform or precision-weighted (a flat machine does not pull the mean).

Cross-fit does not beat v6. The even half liked online precision-weighted shrinkage at 0.5 (56.74 vs 56.45 on that half). The odd half liked a 0.15 uniform shrink and only by +0.001 on that half. Stitching the two choices scores **56.280**. The best full-sample pooling run, online precision 0.5, is 56.410 (Δ +0.124, CI [−0.022, +0.389]). Its stable mean error falls (0.01062 → 0.00877) but 14 stable episodes get worse and 10 get better, and the other three conditions do not move. Jackpot error is unchanged. Half-life choice is already a per-machine decision; sharing it across the episode does not travel from one half to the other.

## (2) Flip or reset the cell counts

Once a tau is chosen, the v6 walk is rerun with one of:

- **Swap.** At tau, exchange the agree and differ counts, then keep decaying and updating. Pre-flip spins are reused under the new direction.
- **Reset.** At tau, delete the win counts and relearn from later spins only.
- **Long only.** Same counts, but the cutoff win probability is taken from the slowest expert of the best pair instead of the mixture.

Swap, any gate in the train gap: **56.4618**.

| Condition | v6 error (ratio) | Swap error (ratio) | Mean paired episode diff | Episodes better / worse |
|---|---:|---:|---:|---:|
| Stable | 0.010622 (0.05412) | 0.010622 (0.05412) | 0 | unchanged |
| Short history | 0.018568 (0.10470) | 0.018568 (0.10470) | 0 | unchanged |
| Drift | 0.009262 (0.04879) | 0.008284 (0.04364) | −0.000977 | 16 / 8 |
| Shared change | 0.011584 (0.05635) | 0.009908 (0.04820) | −0.001676 | 17 / 7 |

Drift episode diffs: 10th percentile −0.00365, median −0.00022, 90th +0.00121. Shared change: −0.00486, −0.00122, +0.00124. About one episode in three still gets worse; the wins are larger. By machine type the squared error drops for all three: steady 0.01302 → 0.01249, balanced 0.01216 → 0.01154, jackpot 0.01230 → 0.01146.

Reset on the same episodes scores 56.383 (Δ +0.097, CI [+0.027, +0.167]). It helps both regimes and less than the swap, which is what you expect if the live and dead rates survive the flip and the old spins are still informative once the label is reversed. Forcing the slow expert after a swap scores 56.327 (CI includes 0). Forcing it after a reset scores 55.965 (CI [−0.578, −0.070]). The mixture should keep the right to prefer a shorter memory; only the counts should be relabeled.

## (3) Blend v6’s win probability with c1

c1’s validation-grid9 win probability is mixed into v6’s win probability. v6’s payout-size mix is kept. Weights are episode-level: c1’s stability posterior, c1’s shared-change posterior, and this detector’s quiet and sharp-flip scores. Sharp-flip requires logBF above 40, posterior sd under 8% of the cutoff, and a long history. Scales 0.25, 0.5, and 1.0 were cross-fit.

The stitch is **56.287**. The even half selected “quiet or sharp, scale 0.5” (56.86 on that half). The odd half selected “c1 says stable, scale 1” (56.42 on that half). Applied to the other half, full weight on c1’s stable class raises short-history error from 0.01857 to 0.02136, and the paired interval on the stitch covers zero.

Inside the grid, one blend does look strong and was **not** the cross-fit selection: quiet-or-sharp at scale 0.5 scores **56.605** (Δ +0.319, CI [+0.131, +0.545]) and beats v6 on both halves (even 56.861, odd 56.383), but the odd half preferred a different blend by 0.035. Condition errors for that unselected blend: stable 0.00890, drift 0.00819, shared change 0.00962, short history unchanged. c1’s shared-change posterior at scale 0.5 is almost the same number (56.487, CI [+0.098, +0.321]) because c1 calls every drift episode a shared flip; it does not touch stable. Treat 56.60 as an upper bound inside a 15-blend grid, not as a selected score. The selection rule that was fixed in advance does not beat v6.

## Recommendation

Put the swap in front of v6. On each episode, compute the joint flip logBF from the train-discovered pairs. If it is above the midpoint of the widest gap in the **train** logBF list (102.7 on this train; any cutoff between about −30 and +230 is the same forecast here), exchange agree and differ counts at the MAP tau and run the ordinary pair mixer from there. Do not hard-code the tick or a fraction of the cutoff. Do not pool half-life weights. Do not reset instead of swapping. Do not hand the cutoff to the slow expert. Do not ship the c1 blend until a weight chosen on one half still wins on the other.

Short history is still the weak condition (ratio 0.105). The detector correctly refuses to declare a flip there, and nothing in this pass moves it.

## Round two

Pairs, the size keys, and the logBF cutoff are functions of that round’s train histories, not of the numbers above. The flip tick is the posterior mode for that episode. The half-life grid and the validation-target win prior and size atoms are whatever v6’s `adapt()` fits on that round. The walk is deterministic.
