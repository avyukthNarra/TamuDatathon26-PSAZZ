# R0 baseline

Branch: `avyukth/exp-r0-baseline`; scored source: `7261af4`; original base: `COMBINED_SOLUTIONS@81d4fff`. Original predictions reproduced exactly (maximum difference 0). That branch contains a mixer, without zhill's rule blend.

| Priors / model | Full /60 | Tuning | Audit | Seconds |
|---|---:|---:|---:|---:|
| Validation / mixer | 55.338 | 55.026 | 55.704 | 14.15 |
| Validation / 20% rule blend | 55.578 | 55.256 | 55.956 | 33.92 |
| Train / mixer | 53.849 | 53.805 | 53.905 | 13.89 |
| Train / 20% rule blend | 54.241 | 54.185 | 54.311 | 33.44 |

Raw full-validation errors:

| Same row order | Stable | Short | Drift | Shared |
|---|---:|---:|---:|---:|
| Validation mixer | .013540 | .022982 | .012267 | .015171 |
| Validation blend | .011531 | .021370 | .012490 | .015048 |
| Train mixer | .019459 | .022385 | .022898 | .022800 |
| Train blend | .016210 | .022183 | .021188 | .021661 |

Originally, `entry_template/predict.py:33–42` read `validation/targets.csv` or validation-derived `priors.json`; `run_pair_mixer.py` also derived priors from validation targets. The new default, `--prior-source train`, uses training histories exclusively. Validation mode remains explicit and diagnostic; its constants are injected outside the model.

Train win centers are .1398/.7563 versus validation-derived .0435/.898. Size modes cross-fit lag selection and counts across alternating training ticks. The train-only estimates lose performance; full-history regime averaging is a possible explanation, untested here.

Each half contains 48 episodes. Paired bootstrap uses 1,000 condition-stratified draws, recomputing reference denominators. Validation-blend gains: tuning +.230 [95% CI +.012,+.475], audit +.252 [+.035,+.490]. All configurations were frozen before audit scoring.

From `mainFiles`:

```sh
python exp_harness.py --candidate r0_model:make_blend --baseline r0_model:make_model --data public --name r0-safe-blend --out runs/r0-safe-blend.csv
```

Pipeline files (relative to `mainFiles`):

- `r0_model.py`: fits mixer and optional blend.
- `slot_forecasting/train_priors.py`: estimates training-only priors.
- `slot_forecasting/pair_mixer.py`: online win/size experts.
- `rule_model.py`: exact zhill rules from `3bdf336`.
- `slot_forecasting/baseline.py`: rule-model frequency fallback.
- `entry_template/predict.py`: submission CLI and prior-source switch.
- `run_pair_mixer.py`: scores and packages entries.
- `exp_harness.py`: scoring, bootstrap, JSONL logging.

All split/condition metrics and intervals: `runs/harness_results.jsonl`; CSVs: `runs/r0-comparison/`. Tests: 23 passed; compile/diff checks passed. No lint/CI/manifest; standard-library only. Changelog updated; patch impact. Draft PR: https://github.com/avyukthNarra/TamuDatathon26-PSAZZ/pull/16
