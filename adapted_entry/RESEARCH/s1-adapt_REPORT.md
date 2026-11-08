# Pair support is learned from train

Each win member now takes its lag pairs from that round's training histories. A machine votes for the pair `a < b ≤ 8` with the largest two-cell (agree versus differ) Bernoulli likelihood gain, and only if both cells have at least 5 ticks and the leader beats the runner-up by 2 nats. Pairs with at least 5% of train machines are kept. If none qualify, all 28 pairs stay eligible. There is no shipped list of main-round pairs.

Level centres for the short-history filter and the exact filter are a 2-means fit on the logit of the other validation fold, the same split `orch/priors/levels_from_truth.py` uses. On main those centres match that JSON to three decimals. The sysid rule library already screens 8 lags and 28 pair products on train and keeps rules with posterior mass above 2%. It had no pair literals. Its scorer now reads `max_points` from the kit's `metadata.json` (60 on main, 40 on these kits). The 69-candidate sysid catalog was not re-run in this time box; the published main figure is unchanged because the learner was not changed.

## Learned support

| Kit | Votes kept | Truth |
|---|---|---|
| main | (1,3), (2,5), (4,7) | those three |
| disjoint | (1,6), (2,4), (3,8) | those three |
| four | (1,6), (2,4), (3,8), (5,7) | those four; (5,7) is 5.2% of machines |
| sharedlag | (1,6), (1,8), (3,6) | those three |

Stray pairs stayed under 4%.

## Main validation (out of 60)

Cross-fit is the protocol in `wt/orch/RESULTS.md`: episodes sorted by id within condition, alternating folds. A choice used on one fold is fit on the other.

| Member | Published | Adapted | Halves A / B |
|---|---:|---:|---|
| v13 + short-history patch | 57.676 | **57.651** | both folds `grid_marginal` |
| exact Bayesian filter | 57.488 (57.41 / 57.56) | **57.488** (57.414 / 57.559) | ratios .0340 / .0713 / .0352 / .0343 |
| sysid rule library | 57.445 | not re-run | learner already train-only |

The exact filter matches the published point and the four condition ratios. The short-history patch is 0.025 under 57.676. The pairs are the original three, so that gap is the replacement of the fixed `LEVELS` table by the fold 2-means prior (separate dead and live slopes, independent shocks). v13's own cross-fit file still scores 57.500 under the same scorer.

## Round-two kits (out of 40)

v7c, from the brief: disjoint 37.72, four 36.45, sharedlag 37.63.

"Locked" forces the old pairs (1,3), (2,5), (4,7) and the old main level centres. For the exact filter those centres are `wt/orch/priors/levels_from_truth.json`. For the short filter they are the old `LEVELS` table written as dead/live means. The lock is a caller override, not a default in the member.

| | disjoint | four | sharedlag |
|---|---:|---:|---:|
| exact, learned | **38.389** | **37.605** | **37.525** |
| exact, locked | 17.668 | 18.915 | 18.114 |
| short filter, learned | 17.589 | 24.249 | 15.894 |
| short filter, locked | 17.281 | 15.290 | 18.006 |
| v7c | 37.72 | 36.45 | 37.63 |

The locked exact filter falls to the high teens on every kit. With the learned pairs it is ahead of v7c on disjoint and four, and 0.10 behind on sharedlag. Drift is the weak condition on four (loss ratio 0.115); short histories are the weak condition on sharedlag (0.121).

The short-history filter is not a full forecaster. On main it only replaces short-history win probabilities and keeps v13's size distribution. These kits have no v13 file, so the number above is the filter on every query plus a two-atom size mixture taken from the other fold's targets. That is why it sits in the teens and twenties. Pair learning still moves it: four, which also has shifted levels, goes from 15.29 locked to 24.25 learned. Sharedlag goes the other way (18.01 locked, 15.89 learned) because the filter is being asked to score drift and a late shared flip, which it does not model.

## What changed

- `wt/mainFiles/short_win.py`: pair vote and fold level centres replace `PAIRS` and `LEVELS`. Paths point at this tree. `--no-v13` scores a full submission when v13's file is absent.
- `wt/research_exact/model.py`: `learn_pairs` inside `fit`, stored on `params['pairs']`. A type with fewer than two separable stable machines gets a flat logit prior so a wrong pair set degrades instead of crashing `initial_levels`.
- `wt/research_exact/evaluate.py`: level centres are fit from the kit. `max_points` comes from `metadata.json`.
- `wt/mainFiles/sysid_eval.py` and `sysid_pipeline.py`: `max_points` comes from `metadata.json`. Rule selection was already train-only.

## Commands

From `/Users/avynarra/Documents/Datathon/research_pf/s1-adapt`, with one BLAS thread:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 wt/mainFiles/short_win.py --out runs/short-main
python3 wt/research_exact/evaluate.py \
  --data wt/mainFiles/public --out runs/exact-main --refit
```

Round two, learned support (`disjoint`, `four`, or `sharedlag`):

```sh
PUB=/Users/avynarra/Documents/Datathon/research_pf/g7-round2/data/disjoint/public
python3 wt/mainFiles/short_win.py --public "$PUB" --no-v13 --out runs/short-disjoint
python3 wt/research_exact/evaluate.py --data "$PUB" --out runs/exact-disjoint --refit
```

The locked columns rebind `learn_pairs` to `[(1, 3), (2, 5), (4, 7)]` and the level fitter to the main centres, then call `score_variants` or `evaluate`. That override is not in the member modules.
