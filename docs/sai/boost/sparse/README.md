# T1: sparse histories in mixer v4

Retain `TM_COMBINED4`. The best exploratory setting gains 0.078 points, but opposite-fold reselection loses 0.011, with its paired CI spanning zero. No supported improvement was found. All changes are gated by actual missing history ticks, never by condition labels.

## Reproduce

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
nice -n 10 python3 docs/sai/boost/sparse/diagnose.py
nice -n 10 python3 -m unittest discover -s docs/sai/boost/sparse -p test_sparse.py -v
nice -n 10 python3 docs/sai/boost/sparse/run.py
nice -n 10 python3 docs/sai/boost/sparse/verify.py
```

No new dependencies. Existing NumPy/SciPy and the pinned baseline size cache are used. These scripts only evaluate dev; no audit/all scorer is called. V4’s original size modes and win prior were fitted on all 96 validation targets and remain unchanged. No additional target-derived priors are introduced. Original lag pairs were discovered on train; the additional lag-1..8 expert dictionary is fixed before scoring.

## Gap diagnosis

`history_diagnosis.json` uses train and the 72 dev episodes. Short histories have exactly 64 observations: median train span 224 ticks with 21 gap runs; median dev span 207.5 ticks with 21 runs. Pair-lag completeness is about 55%, versus 97–98% for dense histories (initial lags explain the remaining dense loss). Other conditions have zero gaps.

## Experiments

- T1a splits each observed outcome’s count between uncertain cells, using the running win rate for missing lag values. Soft-update strengths 0.25, 0.5, 1 were tested.
- T1b freezes a missing tick’s own online win forecast as its soft lag value; later outcomes never revise that forecast. Missing outcomes themselves are never added as training targets. Same three strengths.
- T1c switches v4’s elapsed-tick forgetting to observed-tick forgetting and tests half-lives 6/20, 12/40, 24/80 with eta 4, 8, 16. Two elapsed-tick half-life controls separate slower memory from the clock change.
- T1d adds unary lags 1–8, the remaining 25 pair experts, or both, assigning 0.1 or 0.5 initial expert mass to extras. Six settings.

Every row below is a fixed 20% saved rule + 80% modified mixer forecast. IDs are `BOOST_T1_<number>`. CI95 and paired Δ use the shared scorer with 1,000 condition-stratified episode bootstraps. These full-dev exploratory scores include parameter-selection optimism. Ratios are stable/short_history/drift/shared_change.

| ID | Family | Settings | Dev [CI95] | Δ vs TM_COMBINED4 [CI95] | Condition ratios |
| --- | --- | --- | --- | --- | --- |
| BOOST_T1_01 | T1a | `{"method":"expected","soft_weight":0.25}` | 55.241 [54.336,56.087] | -0.104 [-0.338,+0.043] | 0.063/0.130/0.070/0.082 |
| BOOST_T1_02 | T1a | `{"method":"expected","soft_weight":0.5}` | 55.220 [54.264,56.076] | -0.125 [-0.419,+0.070] | 0.063/0.131/0.070/0.082 |
| BOOST_T1_03 | T1a | `{"method":"expected","soft_weight":1.0}` | 55.163 [54.198,56.022] | -0.182 [-0.519,+0.059] | 0.063/0.136/0.070/0.082 |
| BOOST_T1_04 | T1b | `{"method":"forward","soft_weight":0.25}` | 55.248 [54.345,56.090] | -0.097 [-0.321,+0.045] | 0.063/0.129/0.070/0.082 |
| BOOST_T1_05 | T1b | `{"method":"forward","soft_weight":0.5}` | 55.229 [54.284,56.082] | -0.116 [-0.393,+0.071] | 0.063/0.130/0.070/0.082 |
| BOOST_T1_06 | T1b | `{"method":"forward","soft_weight":1.0}` | 55.170 [54.213,56.008] | -0.175 [-0.494,+0.054] | 0.063/0.135/0.070/0.082 |
| BOOST_T1_07 | T1c | `{"decay_missing":false,"half_lives":[6.0,20.0],"eta":4.0}` | 55.066 [54.180,55.881] | -0.279 [-0.481,-0.097] | 0.063/0.143/0.070/0.082 |
| BOOST_T1_08 | T1c | `{"decay_missing":false,"half_lives":[6.0,20.0],"eta":8.0}` | 55.052 [54.186,55.872] | -0.292 [-0.527,-0.092] | 0.063/0.144/0.070/0.082 |
| BOOST_T1_09 | T1c | `{"decay_missing":false,"half_lives":[6.0,20.0],"eta":16.0}` | 55.052 [54.209,55.848] | -0.293 [-0.538,-0.096] | 0.063/0.144/0.070/0.082 |
| BOOST_T1_10 | T1c | `{"decay_missing":false,"half_lives":[12.0,40.0],"eta":4.0}` | 55.301 [54.472,56.073] | -0.044 [-0.108,+0.002] | 0.063/0.125/0.070/0.082 |
| BOOST_T1_11 | T1c | `{"decay_missing":false,"half_lives":[12.0,40.0],"eta":8.0}` | 55.335 [54.530,56.106] | -0.010 [-0.022,+0.000] | 0.063/0.122/0.070/0.082 |
| BOOST_T1_12 | T1c | `{"decay_missing":false,"half_lives":[12.0,40.0],"eta":16.0}` | 55.359 [54.548,56.112] | +0.014 [-0.016,+0.064] | 0.063/0.120/0.070/0.082 |
| BOOST_T1_13 | T1c | `{"decay_missing":false,"half_lives":[24.0,80.0],"eta":4.0}` | 55.338 [54.517,56.100] | -0.007 [-0.125,+0.101] | 0.063/0.122/0.070/0.082 |
| BOOST_T1_14 | T1c | `{"decay_missing":false,"half_lives":[24.0,80.0],"eta":8.0}` | 55.386 [54.577,56.141] | +0.042 [-0.023,+0.109] | 0.063/0.118/0.070/0.082 |
| BOOST_T1_15 | T1c | `{"decay_missing":false,"half_lives":[24.0,80.0],"eta":16.0}` | 55.423 [54.629,56.160] | +0.078 [+0.005,+0.157] | 0.063/0.115/0.070/0.082 |
| BOOST_T1_16 | T1c | `{"decay_missing":true,"half_lives":[6.0,20.0],"eta":8.0}` | 55.127 [54.288,55.922] | -0.218 [-0.365,-0.062] | 0.063/0.138/0.070/0.082 |
| BOOST_T1_17 | T1c | `{"decay_missing":true,"half_lives":[24.0,80.0],"eta":8.0}` | 55.382 [54.573,56.139] | +0.037 [-0.020,+0.095] | 0.063/0.118/0.070/0.082 |
| BOOST_T1_18 | T1d | `{"extra":"single","extra_mass":0.1}` | 55.311 [54.491,56.054] | -0.034 [-0.115,+0.038] | 0.063/0.124/0.070/0.082 |
| BOOST_T1_19 | T1d | `{"extra":"single","extra_mass":0.5}` | 55.275 [54.447,56.017] | -0.070 [-0.236,+0.065] | 0.063/0.127/0.070/0.082 |
| BOOST_T1_20 | T1d | `{"extra":"wide","extra_mass":0.1}` | 55.326 [54.519,56.083] | -0.019 [-0.036,-0.004] | 0.063/0.123/0.070/0.082 |
| BOOST_T1_21 | T1d | `{"extra":"wide","extra_mass":0.5}` | 55.124 [54.266,55.927] | -0.221 [-0.484,-0.024] | 0.063/0.139/0.070/0.082 |
| BOOST_T1_22 | T1d | `{"extra":"both","extra_mass":0.1}` | 55.303 [54.491,56.059] | -0.042 [-0.098,-0.001] | 0.063/0.125/0.070/0.082 |
| BOOST_T1_23 | T1d | `{"extra":"both","extra_mass":0.5}` | 55.155 [54.327,55.932] | -0.190 [-0.428,-0.008] | 0.063/0.136/0.070/0.082 |
| TM_COMBINED4 | control | `{}` | 55.345 [54.538,56.106] | +0.000 [+0.000,+0.000] | 0.063/0.121/0.070/0.082 |

## Opposite-fold check

Two deterministic condition-balanced dev halves each contain 36 episodes. Each half’s settings are selected using only the opposite half; the unchanged baseline is an eligible choice. No episode or machine crosses halves. The global check selects across all 23 settings.

| Selection | OOF dev [CI95] | Paired Δ [CI95] | Condition ratios |
| --- | --- | --- | --- |
| T1a | 55.345 [54.538,56.106] | +0.000 [+0.000,+0.000] | 0.063/0.121/0.070/0.082 |
| T1b | 55.345 [54.538,56.106] | +0.000 [+0.000,+0.000] | 0.063/0.121/0.070/0.082 |
| T1c | 55.334 [54.511,56.100] | -0.011 [-0.125,+0.089] | 0.063/0.122/0.070/0.082 |
| T1d | 55.345 [54.538,56.106] | +0.000 [+0.000,+0.000] | 0.063/0.121/0.070/0.082 |
| All settings | 55.334 [54.511,56.100] | -0.011 [-0.125,+0.089] | 0.063/0.122/0.070/0.082 |

Global held-out fold 0 selected observed-tick half-lives 24/80, eta 4; held-out fold 1 selected 24/80, eta 16. Their disagreement explains the optimistic full-dev eta choice. The `*_OOF_validation.csv` files apply these choices only on dev; their audit rows are unchanged baseline forecasts, never audit-scored.

## Verification and artifacts

All six tests pass: Beta-posterior parity with pinned v4, original head reproduction, probability validity, dense gating, strict forward causality (including post-cutoff mutations), no updates from missing outcomes, and recovery of incomplete-lag evidence. The full baseline reproduction error is 1.33e-15. Independent best-model reruns are byte-identical to each other and the saved best prediction.

`verification.json` validates all 29 CSVs: 768 unique expected IDs, finite probabilities, proper sums, and 576 dense-history rows per file literally byte-identical to the reference. The local writer copies unchanged reference row text to avoid the shared writer’s floating-point renormalization. All 23 settings, four family OOF forecasts, the global OOF forecast and best forecast are under the `BOOST_T1` prefix. `results.json` preserves all settings and fold selections.

Only this folder and T1-prefixed predictions were written. Git state, shared predictors, changelog and other workers’ files were not modified; the orchestrator owns commits. There is no semver impact, configured lint command or CI for these experiments.
