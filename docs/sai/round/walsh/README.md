# R6 screened Walsh-logistic BMA: killed

At the reference default c_type=1.35 (all machine types), the fixed equal-weight pool of half-lives 12, 40 and infinity has dev win MSE 0.057439 versus 0.007787 for TM_COMBINED4: 7.38× worse. All six OOF blend weights are zero. Each individual horizon independently has MSE >2× reference and zero OOF weight. The kill criterion is satisfied; no further c_type tuning was performed.

## Implementation

`bma.py` implements Experiment 1 of the supplied root reference. It reuses T4’s raw lag extraction, then applies a strictly pre-tick exponentially weighted signed mean for each horizon, clipped to [-0.85,0.85]. Missing lags stay zero. All 36 single/pair features are screened by the specified |z| statistic, with deterministic index tie-breaking.

The top six features receive the reference two-pass Jaakkola–Jordan/Polya–Gamma update, intercept prior standard deviation 2, slab standard deviation 1.35, and exact enumeration of 42 supports of size ≤3. Supports share the seven-feature variational precision exactly as in the supplied code; this is not exact integration of the logistic posterior. Evidence uses log C(36,d), a three-nat Occam window, and MacKay prediction with the specified missing-lag variance contribution. No PairMixer weights, probabilities or source files were changed.

The task gives a horizon bank without an aggregation rule. The primary head uses a fixed uniform pool, without dev horizon selection. Individual-horizon diagnostics show the verdict does not depend on that pooling choice. Only c_type=1.35 was tested on dev; the allowed alternatives 1 and 2 were skipped following the kill rule.

## Dev diagnostics

| Head | Overall win MSE | Stable | Short history | Drift | Shared change | Mean OOF logistic weight | OOF points | Paired Δ [CI95] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TM_COMBINED4 | 0.007787 | 0.006582 | 0.009055 | 0.007793 | 0.007718 | — | 55.345 | 0 |
| UNIFORM | 0.057439 | 0.015294 | 0.030843 | 0.097902 | 0.085717 | 0.00 | 55.345 | +0.000 [+0.000,+0.000] |
| H12 | 0.048559 | 0.046177 | 0.051933 | 0.051719 | 0.044408 | 0.00 | 55.345 | +0.000 [+0.000,+0.000] |
| H40 | 0.055675 | 0.010782 | 0.029663 | 0.099203 | 0.083052 | 0.00 | 55.345 | +0.000 [+0.000,+0.000] |
| INF | 0.097529 | 0.011732 | 0.024683 | 0.182942 | 0.170761 | 0.00 | 55.345 | +0.000 [+0.000,+0.000] |

OOF points equal the baseline: 55.345 [54.538,56.106], paired delta 0 [0,0]. Per-condition OOF ratios are .063/.121/.070/.082. The uniform standalone plug-in scores 41.301, delta −14.044 [−15.119,−12.925]. Full metrics and all per-fold weights are in `results_c1p35.json`.

Every forecast retains TM_COMBINED4’s conditional payout-size shares. Six-fold episode-OOF weights use the existing harness’s 0.05 grid and are fitted on the other five dev folds only. All standalone heads worsen every condition. The reference retains its existing validation-derived prior/size-mode dependency; the BMA head reads pre-cutoff observed histories only.

## Reproduce and verify

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
nice -n 10 python3 -m unittest discover -s docs/sai/round/walsh -p test_bma.py -v
nice -n 10 python3 docs/sai/round/walsh/run.py
nice -n 10 python3 docs/sai/round/walsh/verify.py
```

Five tests pass, including numerical parity with the supplied reference at 14 decimal places, strict centering/cutoff causality, missing-feature handling, exact subset counts and Occam-weight normalization. An independent full-bank repeat reproduces all probabilities, screened supports and retained model counts byte for byte. Eight CSVs each contain 768 valid unique expected IDs and preserve conditional size shares within 3.4e-16. OOF audit rows contain reference forecasts and have not been evaluated; standalone audit rows are predictions only.

The experiment took 4.7 seconds, well within the 35-minute task timebox. No audit/all scorer was called and the access log was unchanged during the run. No installs, commits, branch changes, shared-source edits, changelog or config changes. Source, results and verification are confined to this folder plus R6-prefixed predictions; the orchestrator owns commits. Existing lint/CI gaps remain.
