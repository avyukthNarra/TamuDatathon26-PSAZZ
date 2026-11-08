# Independent self-supervised global learning

This experiment uses observed history ticks as next-spin labels. Each row's
features use strictly earlier ticks of its machine and its episode peers.
It enumerates all eight lags and all 28 pairs; no lag, pair, coefficient, or
setting is imported from prior research. No validation query outcome or target
probability enters feature building, fitting, calibration, or train-only choice.

The two heads are scikit-learn histogram gradient boosting classifiers:
win versus loss, followed by four positive payout classes. Machine-specific
conditional frequencies, counts, and changes let a global classifier learn
machine-specific effect directions. Missing ticks are encoded separately from
losses. A base-feature ablation removes those conditional summaries. Four
pure per-machine frequency forecasts are the empirical comparison.

Run from `mainFiles/` with Python 3.13 and the already installed packages:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  python3 -m fresh_ml.run --out runs/fresh-ml --iterations 160
python3 -m unittest fresh_ml.test_features -v
python3 check_submission.py runs/fresh-ml/validation_predictions.csv --split validation
python3 check_submission.py runs/fresh-ml/submission.csv
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

Results are written to `runs/fresh-ml/results.json`; reproduce the scoring only
with `python3 -m fresh_ml.run --out runs/fresh-ml --evaluate-only`.
