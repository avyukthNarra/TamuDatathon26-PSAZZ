# Research round: new logic on `pranav/stacked-model` (56.51 -> 56.54)

Goal: try entirely different ways of combining what we know, judged on BOTH honest checks:
cross-fitted validation (each half scored with priors / stackers fitted on the other half) and
out-of-fold train labels (8 episode folds; dR vs the 56.51 model at the cutoff and over the cutoff
+ last 10 observed ticks). All code is new (`mainFiles/research/`, `slot_forecasting/changepoint.py`,
`slot_forecasting/ensemble.py`, `run_ensemble.py`); the 56.51 model's files are not edited.

`research/base_cache.py` caches every base forecast once (validation cross-fit, train out of fold
incl. the window, test), so each method is scored in seconds (`research/stack.py`).

## Base models (alone)
| Model | Validation (halves) | Train window dR vs 56.51 |
|---|---|---|
| Pair mixer | 56.29 (56.14 / 56.47) | +0.0039 |
| Hidden-rule HMM | 56.06 (56.23 / 55.87) | +0.0066 |
| Change-point regime model (new) | 56.36 (56.34 / 56.41) | +0.0083 |

The regime model is best on validation and worst on train: its settings (coupling, hazard,
tempering) were chosen by looking at validation, which cross-fitting does not protect against.
Re-tuned on train (hazard 0.001-0.01, temper 0.5-1) it stays +0.007 to +0.012 worse than the mixer.

## Combination methods
| Method | Validation (halves) | Train cutoff dR | Train window dR |
|---|---|---|---|
| 56.51 model (linear 0.7 mixer + 0.3 HMM) | 56.48 (56.47 / 56.50) | 0 | 0 |
| Equal 3-way (with regime) | 56.60 (56.64 / 56.57) | +0.0031 | -0.0002 |
| Gated mixture of experts, fitted on validation truth (situation features: short history, own shift, episode-wide shift, model disagreement, type, signal) | 56.58-56.60 | +0.0029 to +0.0038 | -0.0002 to -0.0004 |
| Logistic stack (+ situation interactions), fitted on validation truth | 56.56-56.57 | +0.0039 to +0.0082 | -0.0027 to +0.0006 |
| Stackers fitted on TRAIN outcomes, scored on untouched validation | up to 56.70 (with regime), 56.63 (mixer + HMM) | | train 4-fold: -0.0001 to -0.0005 |
| **Log-odds pooling, same 0.7 / 0.3 weights** | **56.51 (56.51 / 56.53)** | **-0.0013 [-0.0021, -0.0004]** | -0.0004 [-0.0008, +0.0001] |
| Log-odds 0.6 / 0.4 | 56.53 (56.56 / 56.51) | -0.0014 (noise) | -0.0007 (noise) |
| Size shrink toward the type mix when few recent wins (short histories) | 56.50-56.53 | +0.0004 to +0.0014 | -0.0004 to -0.0010 |

Learned stackers raise validation but never beat the 56.51 model on held-out train: the gain is
validation-specific. What the train-fitted stacker actually learned was simply "average in
log-odds, ~50/50" (slope 1.02, intercept -0.05); the fixed log-odds pool of the existing 0.7 / 0.3
blend is the one change that improves both checks without new tuning.

## Result
`run_ensemble.py --out runs/logodds-v1 --cv-val`: validation 56.54 (56.544817), cross-fit halves
56.51 / 56.53, R 0.0611 (stable 0.047, short 0.103, drift 0.042, shared 0.052), 121 s, entry
reproduces byte for byte on Python 3.13 read-only, also without validation targets. Test SHA-256
bd43cea90fcc34f2ed032f8c9c3575c4ec7b6e1f9aefa03c7f857c7451a5cf29.

## Lesson
Every choice made while looking at validation (settings, stacker weights) shows up as a
validation-only gain. Tune new models on train, confirm once on validation.

Prompt used: "Completely change the logic on a new branch; try a bunch of new methods and see
what works best."
