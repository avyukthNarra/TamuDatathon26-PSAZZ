# Residual hunt: branch `pranav/residual-hunt` (56.54 -> 56.59)

Instead of tuning, look at what the best model gets WRONG: residual = truth - forecast
(validation, exact) and outcome - forecast (train, 8,448 out-of-fold rows), against every
observable signal. A real missed effect must show the same sign with |z| > 2 on BOTH splits.
Scripts: `research/residuals.py`, `residuals2.py`, `peer_*.py`.

## Finding: machines win more when the other machines in their episode just won
| Peers' win share at lag | validation slope (z) | train slope (z) |
|---|---|---|
| 1 | +0.030 (+2.4) | +0.035 (+2.1) |
| 3 | +0.065 (+4.9) | +0.080 (+4.9) |
| 4 | +0.027 (+2.2) | +0.080 (+4.9) |
| 8 | +0.038 (+2.9) | +0.047 (+2.8) |
| mean of lags 1-5 | +0.076 (+4.0) | +0.161 (+6.0) |

Present in every condition, type and forecast level (largest for middle forecasts and short
histories). Not a specific neighbour (each id offset is small and inconsistent) and not an
episode "surprise" (peers' wins minus their forecasts is weaker): a collective, recent effect.
Per-machine responses spread no wider than a time-shifted null (sd 0.46 vs 0.50), so one shared
coefficient. The pair mixer and HMM never look at other machines, which is why it was missed;
the earlier neighbour test looked for pair rules copied across machines, not this.

## Correction (`slot_forecasting/peers.py`)
logit p = logit p_model + 1.0 x (peers' lag 1-5 win share - 0.4187 (train mean))

| Strength | Validation (halves) | Train cutoff dR | Train window dR |
|---|---|---|---|
| 0 (56.54 model) | 56.51 (56.51 / 56.53) | 0 | 0 |
| 0.5 | 56.56 (56.54 / 56.60) | -0.0024 [-0.0049, -0.0001] | -0.0022 [-0.0033, -0.0013] |
| 1.0 (chosen) | 56.56 (56.53 / 56.61) | -0.0041 [-0.0090, +0.0005] | -0.0037 [-0.0057, -0.0019] |
| train-fitted 1.76 | 56.47 | -0.0053 | -0.0046 |

Built entry (`run_ensemble.py --out runs/peers-v1 --cv-val`): validation 56.59 (56.586908),
cross-fit halves 56.52 / 56.60, R 0.0603 (stable 0.047, short 0.100, drift 0.041, shared 0.053),
122 s, reproduces byte for byte on Python 3.13 read-only, also without validation targets. Test
SHA-256 256bf174f26601318a5f2c741c02486b22c71964f54d9913b87a67f3177e2960.

## Round 2 (after the correction)
Only two leftovers pass |z| > 2 on both splits: win vs the signal column (+0.010 per step; a
logit term adds +0.03 validation, train within noise, not adopted) and size vs "own lag 7 won"
(weak). Everything else is clean.
