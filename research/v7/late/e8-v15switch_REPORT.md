# e8: episode-aware v15 blend

**Recommend `w_flip = 1.0`.** It is the train cross-fit choice, it clears every accept bar, and it is the closest of the grid to v7c on all three simulated kits. `run_strongest.py --flip-weight` now defaults to 1.0. `--flip-weight 0.4` (equal to `--win-weight`) is the flat v15 blend.

## What changed

v15's log-odds win blend stays at its weights (0.4 on v7c, extra peer 0.5, signal 0.05, sizes from v7c) except on episodes where v7c's own flip gate fires. There the v7c win share becomes `w_flip`. The gate is `FlipGate.swaps` in `v7c_forecasting/episode_flip.py`: the threshold is the midpoint of the widest gap in the training episodes' log Bayes factors, and an episode fires when its logBF is above that threshold and the posterior has a mode tick. Same detector the vendored forecast uses to swap agree and differ counts.

On this main kit the gate fired on all 24 drift and all 24 shared_change validation episodes (48/96) and on none of the stable or short_history episodes. Each simulated kit fired on its 6 drift and 6 shared_change episodes (12/24). Stable and short_history scores do not move.

## Accept

Main ≥ 57.45 and both cross-fit halves ≥ 57.37. Each sim ≥ v15 (37.44 / 35.62 / 36.64), and as close to v7c (37.72 / 36.45 / 37.63) as possible.

Every `w_flip` in {0.6, 0.7, 0.8, 0.9, 1.0} passes. 1.0 is the one to ship.

## Scores

`none` is the flat v15 blend. It reproduces the published numbers (main 57.55, halves 57.58 / 57.47, sims 37.44 / 35.62 / 36.64). Halves are opposite-half priors, same split as `run_strongest.py --cv-val`.

| w_flip | main | tuning | audit | disjoint | four | sharedlag | passes |
|---|---:|---:|---:|---:|---:|---:|---|
| none (v15) | 57.5467 | 57.5770 | 57.4687 | 37.4355 | 35.6170 | 36.6369 | baseline |
| 0.6 | 57.5562 | 57.5717 | 57.4937 | 37.5449 | 35.7338 | 37.0389 | yes |
| 0.7 | 57.5555 | 57.5632 | 57.5003 | 37.5860 | 35.7819 | 37.1906 | yes |
| 0.8 | 57.5511 | 57.5509 | 57.5031 | 37.6182 | 35.8224 | 37.3111 | yes |
| 0.9 | 57.5431 | 57.5346 | 57.5022 | 37.6413 | 35.8547 | 37.4008 | yes |
| **1.0** | **57.5316** | **57.5145** | **57.4977** | **37.6549** | **35.8782** | **37.4598** | **yes** |
| v7c (published) | 57.37 | 57.31 | 57.39 | 37.72 | 36.45 | 37.63 | |

Main loss ratios at 1.0 (v15 in parentheses): stable 0.036 (0.036), short_history 0.070 (0.070), drift 0.031 (0.029), shared_change 0.035 (0.036). The shared_change win gets better; drift gives a little back; the net on the real kit is −0.015 points. The simulated kits, where v15 was slow after a flip, move the other way. Gaps from 1.0 to v7c are 0.07 / 0.57 / 0.17, versus v15's gaps of 0.28 / 0.83 / 0.99. Four is still the one that does not get close to v7c.

## Train cross-fit

2-fold holdout of train episodes (same even/odd hash split as the validation halves). Each fold refits both models and the flip gate on the other fold. Priors stay the full-validation priors the build already uses. Objective is mean 5-class Brier on the held-out cutoff labels (768 queries). Tie would break toward the smaller weight. It did not tie.

| w_flip | OOF Brier |
|---|---:|
| none | 0.348236 |
| 0.6 | 0.348104 |
| 0.7 | 0.348059 |
| 0.8 | 0.348027 |
| 0.9 | 0.348011 |
| **1.0** | **0.348008** |

The gate fired on 48 held-out train episodes across the two folds. Higher weight is better at every step, so the choice is 1.0. That matches the sim ranking and does not use the validation scores.

## Runtime

One fit per kit, then every weight is a reblend. Main process 450s (full fit 153s, both halves 184s, train choice 112s). Sims, full score only: disjoint 55s, four 75s, sharedlag 68s. A single-weight `run_strongest.py --cv-val` still pays the fit and the two half-refits; the weight itself does not change that cost.

## Reproduce

```sh
cd wt/mainFiles
python3 run_strongest.py --data public --out ../../runs/w1 --cv-val
# or explicitly: --flip-weight 1.0
# flat v15: --flip-weight 0.4
```

Grid scorer (this directory): `python3 eval_flip.py --data <public> --out runs/<name> [--cv-val --train-choose]`.
