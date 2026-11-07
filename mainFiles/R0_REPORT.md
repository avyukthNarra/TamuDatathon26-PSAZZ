# R0 train-only prior evaluation

Source scored: `avyukth/exp-r0-baseline` at `7261af44ededf71eeea10d9dbd3f461a88c18002`.
The original `COMBINED_SOLUTIONS` route is PairMixer-only; its entry does not include a
zhill rule blend. Its `run_pair_mixer.py` reads `validation/targets.csv` to derive priors,
and `entry_template/predict.py` also reads those targets at runtime (lines 33–42) or loads
the packaged validation-derived `priors.json` fallback. The score script uses the
validation labels as intended for evaluation. R0 makes train histories the default source;
validation priors remain available only as an explicit diagnostic option.

`slot_forecasting/train_priors.py` estimates win-prior centers using the discovered
training lag-pair list: each machine contributes its strongest supported parity contrast
with at least 20 observations in each cell and a smoothed separation of at least 0.3; the
two mixture centers are medians across at least three qualifying machines. Size modes are
estimated by selecting lag/orientation on one actual-tick parity and adding raw counts from
the other parity, swapping halves; type distributions use 0.5 pseudocounts per class and
are omitted if either held-out mode has no counts. This is a deliberately approximate
cross-fit, not a guarantee that sparse or regime-mixed training histories recover the
validation target modes. On this dataset the train win centers were 0.1398/0.7563, versus
historical target-derived 0.0435/0.898.

The original prediction CSV replayed with maximum probability difference 0 and exact
55.337655509346384 / 60. Four frozen variants were compared against that cached historical
mixer CSV; all used 1,000 condition-stratified episode bootstrap draws (seed 314159).

| Variant | Full | Tuning | Audit | Tuning paired delta (95% CI) |
|---|---:|---:|---:|---:|
| Historical priors, mixer | 55.338 | 55.026 | 55.704 | 0 (exact replay) |
| Historical priors, 20% zhill blend | 55.578 | 55.256 | 55.956 | +0.230 [+0.012, +0.475] |
| Train priors, mixer | 53.849 | 53.805 | 53.905 | -1.221 [-1.696, -0.689] |
| Train priors, 20% zhill blend | 54.241 | 54.185 | 54.311 | -0.841 [-1.372, -0.252] |

Full-validation raw probability errors by condition:

| Variant | Stable | Short history | Drift | Shared change |
|---|---:|---:|---:|---:|
| Historical priors, mixer | 0.013540 | 0.022982 | 0.012267 | 0.015171 |
| Historical priors, 20% zhill blend | 0.011531 | 0.021370 | 0.012490 | 0.015048 |
| Train priors, mixer | 0.019459 | 0.022385 | 0.022898 | 0.022800 |
| Train priors, 20% zhill blend | 0.016210 | 0.022183 | 0.021188 | 0.021661 |

Commands, per-variant full/tuning/audit condition metrics, bootstrap intervals and prediction
CSVs are recorded in ignored `runs/harness_results.jsonl` and `runs/r0-comparison/`.
The source baseline and priors are preserved in `runs/r0-original/`. The harness was run as:

```sh
PYTHONPATH=runs/r0-original python exp_harness.py --candidate validation_factories:make_historical_model --baseline-predictions runs/r0-original/validation_predictions.csv --data public --name r0-historical-mixer --out runs/r0-comparison/historical_mixer.csv --results runs/harness_results.jsonl
PYTHONPATH=runs/r0-original python exp_harness.py --candidate validation_factories:make_historical_blend --baseline-predictions runs/r0-original/validation_predictions.csv --data public --name r0-historical-blend-20pct-rule --out runs/r0-comparison/historical_blend.csv --results runs/harness_results.jsonl
python exp_harness.py --candidate r0_model:make_model --baseline-predictions runs/r0-original/validation_predictions.csv --data public --name r0-safe-mixer --out runs/r0-comparison/safe_mixer.csv --results runs/harness_results.jsonl
python exp_harness.py --candidate r0_model:make_blend --baseline-predictions runs/r0-original/validation_predictions.csv --data public --name r0-safe-blend --out runs/r0-comparison/safe_blend.csv --results runs/harness_results.jsonl
```

Historical factories inject the saved `original_priors.json` outside `CombinedForecaster`.
The copied `rule_model.py` is byte-identical to `origin/zhill-branch@3bdf336`. Validation
audit scores were computed only after the four settings were frozen; no test labels were
read. Tests: 23 passed, including train-only target-read guard and isolated entry import.
No lint configuration, CI workflow, or dependency manifest is present; the implementation
uses the existing standard-library-only runtime. Tests ran with `python -m unittest discover -s tests`.
