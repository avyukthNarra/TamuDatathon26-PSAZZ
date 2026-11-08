# e3 size calibration

Kept one change: a missing `payout ≥ 2` lag bit is marginalized with the train base rate (about 0.224 on the main kit) instead of one half. Shrinkage, temperature, and polarity–direction coupling were not kept. The switch is `USE_SIZE_CAL` / `--size-cal` (default on; `--no-size-cal` restores v7c).

## Validation

Exact contest points. The two-decimal print of the full score is 57.37 either way; the exact score rises by 0.0034. Neither cross-fit half drops.

| variant | points | tuning half | audit half | stable | short | drift | shared |
|---|---:|---:|---:|---:|---:|---:|---:|
| v7c (`--no-size-cal`) | 57.36924 | 57.30829 | 57.39118 | 0.04190 | 0.07486 | 0.03236 | 0.03431 |
| **missing-bit rate (kept)** | **57.37267** | **57.30850** | **57.39868** | **0.04188** | **0.07465** | **0.03253** | **0.03412** |
| shrink `k/(n+k)` toward type frequency | not scored | | | | | | |
| temperature `1 + k/(1+n)` | not scored | | | | | | |
| polarity tied to win direction | not built | | | | | | |

Shrink and temperature were dropped on train cross-fit before any validation look. Both families were monotone worse as the strength grew (episode-mean conditional Brier on held-out wins: baseline 0.559886; shrink 0.560207 at k=1 up to 0.570509 at k=64; temper 0.560038 at k=1 up to 0.562675 at k=32). Adding shrink on top of the bit rate was also worse than the bit rate alone (0.559355).

Drift's loss ratio moves the wrong way (0.03236 → 0.03253). Short history and shared change move down, and that is enough for the full score and both halves to rise. Halves: tuning +0.00021, audit +0.0075. The keep rule is the full score and the two half scores, not each condition.

## Train cross-fit (how the constant was chosen)

Even/odd episode split, same cut as the size gate. Atoms are the usual EM fit. The choice is which transform to apply; the numeric rate is then the full-train share of observed ticks with payout ≥ 2, re-estimated on whatever kit is supplied. Fold rates were 0.230 and 0.218. Full-train rate 0.2238.

Sharp size posterior (max state ≥ 0.8) versus which XOR cell of the machine's pair wins more often, fit folds only, machines with at least 30 wins:

| direction \ polarity | s=0 | s=1 |
|---|---:|---:|
| 0 (agree cell higher) | 97 | 114 |
| 1 (differ cell higher) | 82 | 118 |

Agreement 215/411 = 0.523. That is not a correlation worth a prior, which matches g11 (direction in the prior key lost likelihood at every strength). No coupling was added.

## Size-part squared error

Query-mean size part, after replacing our win probability with the target's. v7c, then the kept rule, then the difference.

| condition | type | v7c | kept | delta |
|---|---|---:|---:|---:|
| stable | balanced | 0.00049 | 0.00050 | +0.00001 |
| stable | jackpot | 0.00054 | 0.00051 | −0.00003 |
| stable | steady | 0.00014 | 0.00016 | +0.00001 |
| stable | all | 0.00038 | 0.00038 | −0.00000 |
| short_history | balanced | 0.00566 | 0.00612 | +0.00046 |
| short_history | jackpot | 0.00152 | 0.00152 | −0.00000 |
| short_history | steady | 0.00597 | 0.00537 | −0.00060 |
| short_history | all | 0.00423 | 0.00418 | −0.00005 |
| drift | balanced | 0.00020 | 0.00024 | +0.00005 |
| drift | jackpot | 0.00071 | 0.00071 | −0.00000 |
| drift | steady | 0.00040 | 0.00046 | +0.00006 |
| drift | all | 0.00044 | 0.00048 | +0.00004 |
| shared_change | balanced | 0.00478 | 0.00479 | +0.00001 |
| shared_change | jackpot | 0.00126 | 0.00128 | +0.00002 |
| shared_change | steady | 0.00128 | 0.00114 | −0.00013 |
| shared_change | all | 0.00220 | 0.00216 | −0.00004 |

The size error that is left is still short-history balanced and steady, and shared-change balanced. The base rate helps short-history steady and hurts short-history balanced by a smaller amount. Jackpot's size part was already small; g11's jackpot misses are mostly the win probability, which this change does not touch.

## Simulated kits

Kept rule versus the same tree with `--no-size-cal`. Scores match to the printed float. Not worse than the given v7c points 37.72 / 36.45 / 37.63.

| kit | v7c | kept | printed |
|---|---:|---:|---:|
| disjoint | 37.71726 | 37.71726 | 37.72 / 40 |
| four | 36.45405 | 36.45405 | 36.45 / 40 |
| sharedlag | 37.62915 | 37.62915 | 37.63 / 40 |

## What changed

- `mainFiles/slot_forecasting/size_rule.py` — missing-bit emission takes a train rate; shrinkage and temperature helpers exist but stay at strength 0 unless a cross-fit selects them. `crossfit_calibrate` is train-only.
- `mainFiles/slot_forecasting/forecast.py` — `fit_size(..., calibrate=True)` runs that cross-fit and stamps the choice.
- `mainFiles/slot_forecasting/entry_config.py` — `USE_SIZE_CAL = True`.
- `mainFiles/run_pair_mixer.py` — `--size-cal` / `--no-size-cal`.
- `mainFiles/entry_template/predict.py` — passes the switch through.

Atoms, lags, the pair prior, and `p_0` are unchanged. The gate still sees the uncalibrated head and still selects `size_rule` (upper 95% Brier gap −0.00788). `python -m unittest tests.test_size_rule tests.test_size_gate` : 18 tests, OK. With the flag off, behavior is the previous one-half marginal.
