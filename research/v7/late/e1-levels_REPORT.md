# e1 levels: per-machine win-level partial pooling

**Not kept.** The train-chosen pool lowers the main validation score. The switch defaults off (`USE_LEVEL_POOL = False`, `--level-pool` to turn it on). With the switch off, this copy is v7c: **57.37 / 60**, and the three simulated kits match v7c exactly (37.72 / 36.45 / 37.63).

## What was added

Inside `PairMixer._cell_rate`, for a pooled machine type, the highest-mean Beta component (the live atom) is replaced by a Beta whose mean is that type’s train live level at the query’s signal and whose strength `k` is a pseudo-count. The machine’s own decayed cell counts still update the Beta. The dead atom is left alone, so a cell full of losses still selects it.

The live level is `logit(p) = a + b·signal`, fit by least squares on each machine’s higher parity-cell rate in the early window (ticks before `cutoff − 110`), using pairs (1,3), (2,5), (4,7). `k` is chosen by a 2-fold prequential win Brier on train histories (half-life 40, equal weight on the three pairs). Fold centres are fit on the other fold. Nothing here is fit on validation.

Train centres, signal −1 / 0 / +1:

| type | n | slope | live −1 | live 0 | live +1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| jackpot | 135 | 0.345 | 0.495 | 0.580 | 0.661 |
| balanced | 198 | 0.347 | 0.794 | 0.845 | 0.885 |
| steady | 210 | 0.640 | 0.941 | 0.968 | 0.983 |

The current jackpot live atom is 0.669 at strength 10. The jackpot signal-0 centre 0.580 is above the generative centre near 0.50: the higher of two noisy cells is biased upward.

## Train grid (prequential win Brier)

`off` is the current mixture. Lower is better. 111,975 scored ticks.

| scope | k | all-type Brier | jackpot Brier |
| --- | ---: | ---: | ---: |
| balanced+jackpot+steady | 2 | **0.167218** | 0.134528 |
| balanced+jackpot+steady | 4 | 0.167362 | 0.134671 |
| balanced+jackpot+steady | 8 | 0.168002 | 0.135186 |
| jackpot | 2 | 0.168005 | **0.134528** |
| jackpot | 4 | 0.168045 | 0.134671 |
| jackpot | 8 | 0.168192 | 0.135186 |
| off |  | 0.168295 | 0.135550 |
| jackpot | 16 | 0.168428 | 0.136017 |
| jackpot | 32 | 0.168640 | 0.136759 |
| balanced+jackpot+steady | 16 | 0.168761 | 0.136017 |
| balanced+jackpot+steady | 32 | 0.169189 | 0.136759 |

Within the jackpot scope the train choice is **k = 2**. Across scopes the train choice is **all three types, k = 2**. Both were scored. The selector left in the code searches only the jackpot scope, which is the assigned cell. All-types won the proxy and then lost validation, so it is not searched again.

## Validation

Keep rule: full points must rise, and neither cross-fit half may drop by more than 0.02. v7c halves are 57.31 / 57.39.

| variant | points | halves | stable | short | drift | shared |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| v7c, pool off (reproduced) | **57.37** | 57.31 / 57.39 | 0.042 | 0.075 | 0.032 | 0.034 |
| jackpot only, k=2 | 57.33 | 57.18 / 57.45 | 0.041 | 0.074 | 0.036 | 0.035 |
| all types, k=2 | 57.26 | 57.14 / 57.36 | 0.038 | 0.079 | 0.038 | 0.036 |

Jackpot-only drops the tuning half by 0.13 and the full score by 0.04. All-types drops both halves (0.17 and 0.03) and the full score by 0.11. Neither is kept. Stable and short get slightly better under jackpot-only; drift and shared get worse.

### Win-probability RMSE by type

Live means true `p_win ≥ 0.2` (for jackpot that separates the two cells: dead max is about 0.06, live min about 0.27).

| cell | n | v7c | jackpot k=2 | all types k=2 |
| --- | ---: | ---: | ---: | ---: |
| steady | 269 | 0.0633 | 0.0633 | 0.0641 |
| balanced | 234 | 0.0588 | 0.0588 | 0.0629 |
| jackpot | 265 | **0.0863** | 0.0881 | 0.0881 |
| jackpot dead | 156 | **0.0199** | 0.0222 | 0.0222 |
| jackpot live | 109 | **0.1325** | 0.1347 | 0.1347 |

The pool does not reduce jackpot error. Live RMSE goes from 0.1325 to 0.1347, dead from 0.0199 to 0.0222. Train outcome-Brier liked the weak prior; squared error against the true cutoff probabilities did not. k = 2 spends almost no pseudo-counts, so a noisy cell rate replaces the strength-10 shrink toward 0.669, and the early-window centre is still high.

## Simulated kits (kept version, pool off)

| kit | points | v7c bar | stable | short | drift | shared |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| disjoint | 37.72 / 40 | 37.72 | 0.045 | 0.072 | 0.073 | 0.053 |
| four | 36.45 / 40 | 36.45 | 0.075 | 0.144 | 0.120 | 0.050 |
| sharedlag | 37.63 / 40 | 37.63 | 0.062 | 0.075 | 0.062 | 0.053 |

## Changed files

- `mainFiles/slot_forecasting/pair_mixer.py` — signal passed into the walk; live-atom retarget; train centre fit and prequential strength search, called from `adapt` when the flag is on.
- `mainFiles/slot_forecasting/entry_config.py` — `USE_LEVEL_POOL = False`.
- `mainFiles/run_pair_mixer.py` — `--level-pool` / `--no-level-pool`. Cross-fit folds reuse the train-chosen pool instead of selecting again.
- `mainFiles/entry_template/predict.py` — same flag.
- `mainFiles/tests/test_v7_changes.py` — signal moves a live-looking jackpot cell and does not move a dead one. `tests.test_v7_changes` and `tests.test_pair_mixer`: 31 tests, ok.

Runs: `mainFiles/runs/e1-baseline`, `e1-jackpot-pool`, `e1-level-pool`, `sim-disjoint`, `sim-four`, `sim-sharedlag`.
