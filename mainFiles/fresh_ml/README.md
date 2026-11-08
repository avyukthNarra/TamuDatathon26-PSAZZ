# Independent self-supervised global learning

This experiment uses observed history ticks as next-spin labels. Each row's
features use strictly earlier ticks of its machine and its episode peers.
It enumerates all eight lags and all 28 pairs; no lag, pair, coefficient, or
setting is imported from prior research. No validation query outcome or target
probability enters feature building, fitting, calibration, or train-only choice.

The headline result is **52.9253/60 cross-fitted points**, with 95% episode
bootstrap CI **[51.9583, 53.8496]**. Both validation folds selected the same
calibrated 160-round, 31-leaf model. All choices made from train labels alone
selected 320 rounds and scored **52.6718**, CI **[51.7382, 53.5489]**.

| Forecast | Points | Stable ratio | Short ratio | Drift ratio | Shared ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cross-fitted HGB | 52.9253 | 0.170062 | 0.177890 | 0.088794 | 0.097947 |
| Train-only HGB choice | 52.6718 | 0.185683 | 0.178432 | 0.092586 | 0.099820 |
| Pooled anchor | 30.0000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Empirical 128 (train-chosen window) | 30.2817 | 0.99654 | 1.02679 | 0.96228 | 0.93997 |
| HGB without in-context summaries, calibrated | 34.4658 | 0.71028 | 0.79262 | 0.78408 | 0.67645 |
| HGB 15 leaves with in-context summaries, calibrated | 52.6050 | 0.16571 | 0.19460 | 0.09273 | 0.10926 |
| MLP 25 epochs, calibrated (diagnostic) | 51.2267 | 0.21657 | 0.19766 | 0.12749 | 0.14335 |
| HGB 640 rounds, calibrated (diagnostic) | 52.4228 | 0.19658 | 0.17145 | 0.09620 | 0.11392 |

Fold A/B scores for the cross-fitted choice are 52.5130/53.2642; pooled is
30.0000 on both. Pooled's frozen reference is reproduced exactly. Fold scores
are not averaged to form the headline: opposite-fold predictions are combined
and passed to the official scorer, preserving condition normalization.

The two heads are scikit-learn histogram gradient boosting classifiers:
win versus loss, followed by four positive payout classes. Machine-specific
conditional frequencies, counts, and changes let a global classifier learn
machine-specific effect directions. Missing ticks are encoded separately from
losses. A base-feature ablation removes those conditional summaries. Four
pure per-machine frequency forecasts are the empirical comparison.
The multiclass MLP has hidden layers (64,32), Adam, alpha=10 and snapshots at
10/25/45 epochs. The tree grid is 15 leaves without conditional summaries,
15/31 leaves with them, initially 160 rounds; the 31-leaf model is also
continued to 320 and 640 rounds. All candidates include raw and calibrated
predictions. Train-only log loss determines the submission candidate.

Run from `mainFiles/` with Python 3.13 and the already installed packages:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  python3 -m fresh_ml.run --out runs/fresh-ml-repro --iterations 160 \
    --include-mlp --extend-boosting --predict-only
python3 -m fresh_ml.run --out runs/fresh-ml-repro --evaluate-only
python3 -m unittest fresh_ml.test_features -v
python3 check_submission.py runs/fresh-ml-repro/validation_crossfit.csv --split validation
python3 check_submission.py runs/fresh-ml-repro/validation_predictions.csv --split validation
python3 check_submission.py runs/fresh-ml-repro/submission.csv
python3 check_submission.py runs/fresh-ml-repro/submission_validation_choice.csv
```

Versions used: numpy 2.3.3, scipy 1.17.0, scikit-learn 1.8.0. No packages are
installed or project dependencies changed. Seeds are fixed; thread pools are
limited to two. Run outputs, feature caches, and fitted pickle files remain in
the ignored run directory. Only load pickle files produced by this experiment.

Train-only choice means all model and calibration choices use only held-out
train-episode observed labels. Twelve train episodes calibrate the heads;
twelve disjoint train episodes select the candidate by multiclass log loss.
Other episodes provide fitting examples, including legal validation/test
histories. The alternative validation-tuned result selects on sorted,
alternating episodes within condition and reports opposite-fold predictions.
The pooled starter is scored with the same folds and frozen reference.

Bootstrap intervals resample episodes within condition 4,000 times, resampling
model and reference errors together. They condition on the fitted models and
selected fold choices; they do not include retraining/model-selection variance.

`signal` is excluded because its query-time value is not documented as an
observable at earlier pseudo-query ticks. Machine type is assumed fixed within
an episode, as represented by the kit. Historical labels start at tick 32;
calibration and selection use ticks >=128 to resemble final cutoffs. Train query
labels may be labels at their cutoffs, but are never added to feature histories.

The tests perturb all own-machine and peer observations at and after a cutoff
and require every earlier feature to remain identical. They also check strict
prefix sums, missing-tick lag alignment, and exclusion of the current label
from conditional counts.

Results are written to `runs/fresh-ml-repro/results.json`; the answer-free fit
records its complete runtime, peak memory, settings, versions, and input/source
hashes in `prediction_run.json`. It reads only train labels and the three sets
of input queries/histories. `--evaluate-only` opens scoring files afterwards.
`submission.csv` uses the train-only 320-round choice. When both validation
folds select the same model, `submission_validation_choice.csv` exports that
candidate's test forecast (160 rounds on this dataset).

To compare two independent complete fits (after running the command above
with two different fresh output directories), run:

```sh
python3 -m fresh_ml.verify runs/fresh-ml runs/fresh-ml-repro
```

Negative results: recent empirical frequencies stayed near the 30-point
anchor (best diagnostic window 48: 30.4087); an MLP did not beat the HGB;
640 boosting rounds hurt versus 160/320. Train selection loss was 0.900694
without conditional evidence versus 0.777672 with it (160 rounds), and
0.776920/0.778705 at 320/640 rounds. Calibration lifted the 160-round model
from 52.7500 to 52.9253. These numbers are diagnostics; the headline uses
opposite-fold selection rather than choosing the largest all-validation score.

Next experiment: use online regularized multivariate fits as additional
in-context evidence to separate correlated lag effects, with hyperparameters
chosen from train episodes. This has not been implemented or measured.
