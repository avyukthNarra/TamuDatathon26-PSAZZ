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

## Round 3: per-machine hunts, calibration, tempering (no further gain)
- **Per-machine win hunt** (`research/per_machine_hunt.py`): residual of the pair mixer at every
  history tick >= 60 vs 28 lag-pair parities and 8 single lags, max |z| per machine vs a
  23-tick-shifted null. 14.8% of machines exceed the null's 95th percentile (5% = chance); the
  strongest patterns are the machines' OWN pairs (1,3)/(2,5)/(4,7). By condition: stable 3.6%
  (nothing), drift 24%, shared_change 17%, where the mixer is over-confident (residual -0.04 where
  it forecasts > 0.5, +0.02 where <= 0.5): it lags machines whose rule fades or changes.
- **Recent-surprise tempering** (pull the forecast toward 0.5 when the mixer's last 10-40
  forecasts were surprising): worse on BOTH splits at every strength (e.g. W 40 g 1: validation
  56.54 vs 56.55, train window +0.0008 [+0.0005, +0.0012]). The final model (HMM + peers) already
  handles what the mixer alone gets wrong at the cutoff.
- **Per-machine size hunt** (`research/per_machine_size.py`): 6.9% above the null's 95th percentile
  (5% = chance); pooled differences <= 0.01. Payout size is data-limited, not mis-specified.
- **Calibration** (`research/calibration.py`): good overall; forecasts above 0.95 run ~0.015 high on
  both splits, but a ceiling at 0.95-0.97 changes validation by +0.01 and train within noise.

Conclusion: after the peer effect, every residual test (population, per machine, win, size,
calibration) is at chance level. v10 (56.59) is the best verified model.

## Round 4: refining the peer correction (branch `pranav/peer-plus`, no change)
`research/peer_plus.py`, fitted on train, validation untouched, compared with v10:
| Variant | Validation (halves) | Train window dR vs v10 |
|---|---|---|
| v10 (peers lags 1-5 x 1.0) | 56.56 (56.53 / 56.61) | 0 |
| per-lag weights (train: 2-4 ~0.6, 7 0.25, others ~0) x0.55 | 56.58 (56.51 / 56.66) | -0.0014 [-0.0026, -0.0003] |
| per-lag weights x0.7 | 56.55 (56.47 / 56.66) | -0.0022 |
| + short-history interaction (train fits -0.05) | 56.48-56.57 | ~0 |
| forced short x 0.75 (+ signal) | 56.30-56.33 | +0.002 to +0.004 |
Per-lag weights help train but lower the validation tuning half: not adopted. v10 stays.
