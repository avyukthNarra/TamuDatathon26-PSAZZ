# e2 AdaHedge / train half-lives

Nothing is kept. Defaults stay off (`USE_ADAHEDGE = False`, `USE_HALF_LIFE = False`), so a build with no flags is v7c. Both switches are on the runner (`--adahedge`, `--half-life`) and in `entry_config.py`.

AdaHedge is de Rooij et al. (2014): per machine, `eta = -log(min prior weight) / cumulative mixability gap`, weights `∝ prior * exp(-eta * cumulative squared loss)`. It replaces both the win eta (6 with a win prior, else 16) and the size eta (4). No learning rate is fit. The size-rule head still replaces the mixer's size probabilities when the train gate turns it on, which it did on every run here, so the scored change is the win-expert rate.

Half-lives are one triple from a fixed grid, chosen by the existing train `outcome_brier` (mean 5-class Brier on observed ticks from 40 on). The grid is not fit on validation. An exact tie would have kept `(12, 40, 100)`.

## Main validation

| variant | points | tuning half | audit half | stable | short | drift | shared | keep |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| v7c baseline | 57.369 | 57.308 | 57.391 | 0.042 | 0.075 | 0.032 | 0.034 | — |
| AdaHedge | 57.439 | 57.450 | 57.380 | 0.038 | 0.074 | 0.032 | 0.034 | no: sims |
| half-life `(6, 24, 96)` | 57.223 | 57.064 | 57.361 | 0.045 | 0.078 | 0.036 | 0.035 | no |
| both, same triple | 57.314 | 57.247 | 57.342 | 0.040 | 0.077 | 0.035 | 0.035 | no |

AdaHedge is the only variant that raises the full score. Its audit half drops by 0.011, inside the 0.02 cap. The tuning half rises. The half-life pick fails the full score and drops the tuning half by 0.244. The combination fails the full score and both halves (drops 0.061 and 0.049).

Train prequential Brier, the selection criterion:

| half-lives | fixed eta | AdaHedge |
|---|---:|---:|
| (12, 40, 100) | 0.402326 | 0.402931 |
| (6, 24, 96) | 0.395445 | 0.395937 |
| (8, 32, 128) | 0.397036 | 0.397615 |
| (16, 64, 192) | 0.408535 | 0.409139 |
| (12, 48, 160) | 0.402336 | 0.402942 |

`(6, 24, 96)` wins that average on both weight rules, and loses the cutoff contest. The online average over ticks ≥ 40 likes a fast expert; the scored tick still wants the slower triple. Both cross-fit folds picked the same triple, so the choice is not an artifact of one prior half.

## Sims (AdaHedge only)

The other two variants already fail the main keep rule, so they were not run on the kits. Floors are the stated v7c scores. AdaHedge is worse on two of three.

| kit | v7c floor | AdaHedge | vs floor |
|---|---:|---:|---|
| disjoint | 37.72 | 37.698 | −0.022 |
| four | 36.45 | 36.475 | +0.025 |
| sharedlag | 37.63 | 37.557 | −0.073 |

## Changed files

- `mainFiles/slot_forecasting/pair_mixer.py` — AdaHedge update, `HALF_LIFE_GRID`, `adapt(..., adahedge=, tune_half_life=)`
- `mainFiles/slot_forecasting/entry_config.py` — `USE_ADAHEDGE`, `USE_HALF_LIFE` (both false)
- `mainFiles/run_pair_mixer.py` — `--adahedge`, `--half-life`, passed through both `adapt` calls
- `mainFiles/entry_template/predict.py` — same flags from `entry_config`

`python3 -m unittest tests.test_pair_mixer tests.test_v7_changes` — 30 tests, OK. A synthetic AdaHedge walk is deterministic and differs from fixed eta. Baseline on this copy matches the prompt: 57.37 / 60, halves 57.31 / 57.39, ratios 0.042, 0.075, 0.032, 0.034.
