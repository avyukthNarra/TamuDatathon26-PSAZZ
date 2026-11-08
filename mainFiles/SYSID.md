# Independent system identification of House Edge

This experiment learns recurring binary temporal rules from training histories,
then sums over a machine's rule, direction, intercept, and possible reversal time.
A shared reversal hypothesis combines all eight machines in an episode. Winning
payout size uses a separate finite-rule posterior. No lags, pair identities, or
physical coefficient values are copied from earlier team models.

## Reproduction

Tested with Python 3.13.3, NumPy 2.3.3, and SciPy 1.17.0. From the repository root:

```sh
OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python3 mainFiles/sysid_pipeline.py --out mainFiles/runs/sysid-final
python3 -m unittest discover -s mainFiles -p sysid_tests.py -v
python3 mainFiles/check_submission.py mainFiles/runs/sysid-final/validation_predictions.csv --split validation
python3 mainFiles/check_submission.py mainFiles/runs/sysid-final/submission.csv
```

The pipeline first fits every candidate on training inputs and labels, writes an
answer-free catalog, and predicts validation. Only the separate scoring phase
reads validation targets. `--phase prepare` and `--phase score` can run those
stages separately. All generated predictions and JSON artifacts stay under the
ignored output directory. `submission_train_only.csv` uses training-only model
selection; `submission.csv` selects a configuration on all validation after
cross-fitted evaluation. This is a research predictor, not an uploaded entry.

Additional calculations:

```sh
python3 mainFiles/sysid_study.py
python3 mainFiles/sysid_history.py
python3 mainFiles/sysid_oracle.py
python3 mainFiles/sysid_size.py
python3 mainFiles/inspect_sysid.py
```

`sysid_study.py` screens all single lags and pairs up to eight, including large
payout and peer features, using training histories. Its target diagnostics and
`sysid_oracle.py` use only validation Fold A. Oracle mode labels are confined to
those diagnostics and never enter forecasts. `sysid_history.py` compares
chronological holdout likelihoods for local logistic alternatives.

## Generative assumptions

A machine has one recurring win rule from a learned subset of single win lags
and pair products. Each win indicator is encoded -1/+1. The hidden direction is
-1/+1 and the intercept has a type/signal-specific Gaussian prior. Shared
physical parameters maximize stable-training marginal likelihood. The next
win probability is logistic(intercept + signed rule amplitude + learned common
lag/peer effects). A regime can remain fixed, reverse independently per
machine, or reverse at one shared episode time. Change times are integrated on
a fixed grid from 0.25 to 0.95 of the query cutoff; linear transition widths are
selected from training-history likelihood. Episode hypotheses have prior
weights 0.5 / 0.25 / 0.25, matching the supplied balanced training conditions.

The size generator chooses between two type-specific categorical profiles using
one binary temporal rule and a hidden direction. Its profiles and rule prior
are learned by EM on stable training histories. The constrained candidate shares
the non-1 winning-payout ratios between modes and learns a common mode log-odds
gap. Numeric profile values do not come from validation. Conditional independence
of the win and size parameters lets their posterior means form a five-class
forecast. A joint shared-reversal variant for size is measured separately.

Finite rule/sign/change states are summed exactly for this approximate model;
Gaussian intercepts use deterministic Gauss-Hermite quadrature. This does not
establish recovery of the exact simulator. Missing lag rows are excluded from
rule likelihoods; no missing outcome is treated as a loss. A missing query rule
feature is marginalized with a symmetric binary prior. The released queries all
have a complete recent 40-tick segment.

## Validation protocol

Episodes are sorted within each condition and assigned alternately to A and B.
All numerical model fitting uses training data. A fixed catalog is selected on
A and evaluated on B, then selected on B and evaluated on A. The headline scores
the union of these held-out predictions with the official 60-point scorer.
Training-only selection uses a BIC-style criterion for shared effects and size
constraints, and all-training history marginal likelihood for reversal shape.
For shared effects the criterion is -2 times the fitted penalized marginal
objective plus k log(n); common effects have weak Gaussian shrinkage.
The pooled starter is fitted from the same train history and labels. Bootstrap
intervals resample episodes within condition and recompute both forecast and
reference error from the same sampled episodes. They condition on the saved
forecasts and do not include model-selection uncertainty.

## Results

Final cross-fitted results and the oracle decomposition are generated in
`runs/sysid-final/final_results.json` and recorded in the coordinator report.

The kit inspected here is the **main** round (`max_points=60`), season
`5fffdc4d0c700bbf61c3f911`. Validation has 96 episodes / 768 queries;
test has 1,536 queries. Results below use the predeclared 69-candidate catalog.

| Forecast | Points / 60 | 95% episode bootstrap CI | Stable ratio | Short ratio | Drift ratio | Shared ratio |
|---|---:|---:|---:|---:|---:|---:|
| **Cross-fitted selection** | **57.2843** | **56.6589–57.7932** | 0.02837 | 0.09779 | 0.03006 | 0.03341 |
| Training-only selection | 57.4453 | 56.8825–57.8950 | 0.02990 | 0.09217 | 0.02414 | 0.03167 |
| Pooled starter | 30.0000 | 30.0000–30.0000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| Cross-fitted + oracle win chance | 59.2470 | 59.0499–59.4303 | 0.00396 | 0.03486 | 0.00230 | 0.00972 |
| Cross-fitted + oracle size distribution | 57.9901 | 57.3464–58.4921 | 0.02400 | 0.06334 | 0.02764 | 0.02366 |

These oracle interventions replace just one part of the saved forecast with
validation truth; they are diagnostics, not predictors. Their gains are not
additive. Perfect win chances are worth +1.9627 points; perfect sizes are worth
+0.7058 for the cross-fitted model. For training-only selection the oracle-win
and oracle-size scores are 59.2470 and 58.1593 respectively.

The selection made on A is `all_peers/dyn01/constrained`, evaluated on B at
57.4581. The selection made on B is `no_common/dyn0/constrained`, evaluated on A
at 57.0809. The headline rescales the combined episode errors; it is not an
arithmetic average of those two scores. The pooled starter scores 30.0000 on
both folds. Training-only selection is `strongest_peer/dyn01/constrained`;
its A/B scores are 57.3749 / 57.4871. Selection on all validation yields 57.4989
and writes `submission.csv`; that number is a selection-set diagnostic.

### What was identified

Stable-training evidence gives 99.105% of unpruned win-rule mass to pairs
(1,3), (2,5), and (4,7). Their renormalized prior weights are 0.37195, 0.32649,
and 0.30157. The search enumerated all eight single lags and all 28 pairs; these
three identities are learned outputs. The size screen enumerates 128 candidates
and selects large-payout lags 1, 2, and 4 from stable train histories. Its
constrained model learns weights 0.38905, 0.36229, and 0.24866.

For the training-only win model, the fitted signed-rule amplitude is 2.36663,
intercept standard deviation 0.33705, signal coefficient 0.38062, and learned
peer effect 0.48635 on the mean -1/+1 win indicator at lag 3. Type intercept
means are 0.58792 / -0.62754 / -1.97704 for steady / balanced / jackpot. The BIC-style criterion
selects that single peer term over zero, eight peer terms, or the larger own-lag
models: 24,691.76 versus 24,787.53 for no common effects and 24,743.50 for
eight peer terms. The identity of that peer lag is selected by the strongest
coefficient in the training peer fit, then compared by BIC.

Fold A has exactly two conditional winning-size profiles per type to nine
decimal places. Their counts are steady 66/57, balanced 55/60, jackpot 64/82.
The first component is P(payout=1 | win):

| Type | Lower P(1 | win) | Higher P(1 | win) | Log-odds gap | Ratio P(5)/P(2) | Ratio P(20)/P(2) |
|---|---:|---:|---:|---:|---:|
| Steady | 0.455122 | 0.753989 | 1.300000 | 0.245000 | 0.005000 |
| Balanced | 0.214066 | 0.499853 | 1.300000 | 0.375000 | 0.062500 |
| Jackpot | 0.120787 | 0.335146 | 1.300000 | 1.285714 | 0.428571 |

This target analysis motivated a compact family; the deployed profile values,
log-odds gap, and ratios are all re-estimated from training observations. The
training-only constrained model's P(1 | win) endpoints are approximately
0.45032/0.74771, 0.21663/0.50010, and 0.11567/0.32120 respectively. Winning
amounts are therefore a discrete-mode problem, while observed win chances span
0.002465–0.984939 on A.

Direct fits to A targets quantify why pooled additive regression is insufficient:

| Functional form fitted on A | Parameters | Logit P(win) RMSE |
|---|---:|---:|
| Type + signal | 4 | 2.45916 |
| Type + signal + 8 own win lags | 12 | 2.30034 |
| Type + signal + win lags/pairs | 40 | 2.19834 |
| Type + signal + 96 lag/size/peer features | 100 | 1.96587 |
| Type + signal + oracle two-cluster win mode | 5 | 0.59735 |
| Type + signal + type-specific oracle mode amplitude | 7 | 0.59517 |
| Oracle mode + own/peer lags + episode offsets | 70 | 0.41995 |

The last three use mode labels derived from targets exclusively for structural
diagnostics. They cannot be deployed as forecasts. The type-specific fitted
mode amplitudes are 2.33190 / 2.41909 / 2.45347; residual error remains large
relative to CSV precision. A common signed coefficient is a useful approximation,
not evidence of exact coefficient quantization in the hidden simulator.

### Does the model explain histories?

These are train-history marginal log likelihoods on identical observed binary
rows with all eight own lags present, excluding query labels. Higher is better.
Physical parameters were fitted on train, so this table is an in-sample model
check. The Beta-Bernoulli alternative integrates one constant win probability
per machine with a Beta(0.5,0.5) prior; the static rule model omits reversals.

| Condition | Binary observations | Constant frequency | Static rule | Reversal mixture |
|---|---:|---:|---:|---:|
| Stable | 40,824 | -24,683.67 | -12,237.24 | -12,253.88 |
| Short history | 6,174 | -4,000.06 | -2,308.06 | -2,313.36 |
| Drift | 41,264 | -26,106.25 | -31,925.20 | -13,946.10 |
| Shared change | 40,960 | -25,935.98 | -33,711.29 | -13,438.23 |

The reversal mixture pays its prior penalty on stable episodes and correctly
assigns its mass to the static component. It remedies the severe likelihood
failure of an unchanged rule on drift/shared episodes. For winning sizes, the
corresponding temporal-rule log likelihoods are -16,583.12 / -2,686.18 /
-17,773.85 / -16,922.45, versus -17,643.50 / -3,102.44 / -18,981.31 /
-18,081.34 for a constant per-machine Dirichlet(0.5,0.5,0.5,0.5) model.

### Negative results and remaining uncertainty

- Ignoring reversal regimes scored 23.3052 on the preliminary Fold A experiment;
  using a 16-tick forgetting half-life scored 56.8708, below the 57.1377 initial
  explicit reversal model. These use the initial basic win/size configuration.
- Chronological train holdouts (first 65% fitting, final 35% testing; identical
  45,864 held-out observations) gave mean log probability -0.66523 for an
  intercept-only model, -0.69655 for eight win lags, -0.84686 for win lags plus
  all pairs, and -0.77990 for all 96 features. Dense static local fits did not
  handle reversals. The finite-rule model also needs an explicit regime model.
- Making size-rule directions reverse with win-rule directions reduces training
  likelihood substantially. `inspect_sysid.py` reproduces that comparison with
  the final trained model: imposing these reversals loses 635.308 drift and
  623.877 shared log-likelihood units, with no stable-episode change.
- Selecting among 69 candidates on validation did not improve on training-only
  selection in the observed cross-fitted result (57.2843 versus 57.4453).
- Gaussian-prior integration is approximate: increasing 15 quadrature nodes to
  31 on all 384 A queries changes win probability by at most 0.000727 and by
  0.0000138 on average. The change-time grid and the single-rule assumption
  also remain approximations to the unidentified simulator.

The next useful investigation is a learned episode-level intercept or evolving
hidden bias, and an empirical distribution of rule amplitudes. Win probability
has substantially more remaining oracle headroom than payout size. Short
histories have the largest error ratio; a joint episode prior may improve their
remaining uncertainty. Temporal rules and physical coefficients must be
relearned on adaptation data; the synthetic regression check explicitly recovers
different pairs (1,8), (2,6), (3,5).

### Verification and repository handoff

A complete clean reproduction takes 128.34 seconds with BLAS thread counts
limited to two. Peak RSS is 598,638,592 bytes (0.558 GiB). Validation and test
CSVs for both selectors, plus `final_results.json`, are byte-identical across
the independent runs. The measurement command on this macOS host was:

```sh
/usr/bin/time -l env OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python3 mainFiles/sysid_pipeline.py --out mainFiles/runs/sysid-repro
```

Eight regression checks pass, covering alternate-rule recovery, exact binary and
shared-state sums, missing observations, empty positive histories, exclusion of
future observations in prefix predictions, opposite-fold selection, and official
scoring/bootstrap normalization. `py_compile` passes. The official checker
reports 768 valid validation rows and 1,536 valid test rows, and independently
reproduces 57.284283 points. No external APIs or new dependencies are used.

The assigned branch is `sai/fresh-sysid`. No push or PR is made, as required by
the task brief. The kit has source and evaluation tests; the project root lacks
a README, dependency manifest, .env.example, lint configuration, and CI.
Branch protection is documented as absent. Those infrastructure gaps were left
outside this research task. The added forecaster has a MINOR semver impact if
shipped; no release version or tag is created.

## Win-model follow-up: continuous latent levels and ceiling

The next analysis isolates `P(win)` after the team model identified the winning-size rule. It uses the train-selected history posterior from `strongest_peer/dyn01/constrained`. Every candidate input is built from validation histories; target logits are fitted on A and evaluated on B, then fitted on B and evaluated on A. The 4000-resample episode bootstrap is paired with the official pooled reference. The size vector is set to each held-out query's true conditional size distribution to report a win-only ceiling; this is an oracle diagnostic, not a deployable model.

The six size profiles are independently present in all validation targets: exactly two per type after rounding conditional probabilities to eight decimals. This agrees with the earlier train-only size fit and its large-payout indicators at lags 1, 2 and 4. That result does not make `P(win)` a two-level variable.

The all-validation `P(win)` deciles at 10%, ..., 90% are 0.0104, 0.0234, 0.0576, 0.1319, 0.3145, 0.6638, 0.8228, 0.9117, 0.9553. Type-specific distributions are more revealing: steady ranges from 0.073 at p5 to 0.976 at p95, balanced from 0.015 to 0.939, and jackpot from 0.004 to 0.792. The data contain a low and high band, but each band has continuous within-band spread. A target-derived two-cluster diagnostic leaves within-cluster logit RMSE 0.562 (steady), 0.712 (balanced), and 0.707 (jackpot). Type, signal, and a two-mode label alone leave target-logit RMSE 0.597. Adding episode offsets and observed lag/peer features reduces it to 0.420. Since the mode labels are computed from true targets, these figures diagnose the hidden structure and do not represent an available predictor.

The train-fitted posterior expresses this structure as a per-machine continuous intercept (Gaussian prior SD 0.337), a signed parity term of amplitude 2.367, signal coefficient 0.381, peer lag-three coefficient 0.486, and a posterior over static, independent reversal, and episode-shared reversal. Its reversal width is 0.1 of cutoff, selected by train history likelihood. Thus a machine's level shifts the baseline logit continuously; the pair parity moves it between bands, while signal, peers, and time change the resulting logit or its sign. Direct Fold-A fits support this decomposition: raw type/signal logit RMSE is 2.459, while the target-derived mode fit is 0.597. A global fit over 96 ordinary lag/size/peer terms still has RMSE 1.966. The latent state and continuous intercept matter together.

The fixed-form candidate ceilings, cross-fitted over folds, are:

| Candidate P(win) form | Win-only ceiling / 60 | Stable ratio | Short ratio | Drift ratio | Shared ratio |
|---|---:|---:|---:|---:|---:|
| Type + signal | 27.963 | 1.0780 | 1.2776 | 1.1034 | 1.1238 |
| Type + signal + history-posterior logit | 58.105 | 0.0258 | 0.0595 | 0.0218 | 0.0233 |
| Above + raw learned pair parities | 58.112 | 0.0258 | 0.0595 | 0.0215 | 0.0232 |
| Above + recent 32-tick win rate | 58.123 | 0.0252 | 0.0601 | 0.0210 | 0.0229 |
| Above + episode mean history logit | 58.116 | 0.0255 | 0.0584 | 0.0225 | 0.0233 |
| Exact P(win) targets + exact size | 60.000 | 0 | 0 | 0 | 0 |

The calibrated history-logit candidate scores 57.927 on A and 58.247 on B, with 95% episode-bootstrap CI 57.513–58.545. The history plus recent-32 variant's 58.123 cross-fit score is the best among the screened set and should be read as a selection-set diagnostic. Its larger ceiling does not establish that recent frequency is the hidden generator: adding recent frequency to the history posterior changes the score by only 0.018. The direct pair features add only 0.006, because the history posterior already integrates machine-specific rule and direction evidence.

Win RMSE for the calibrated history-logit model is 0.045 for steady, 0.070 for balanced, and **0.082 for jackpot**; by contest condition it is 0.055 drift, 0.059 shared change, 0.088 **short history**, and 0.062 stable. The screened history+recent model has RMSE 0.047 / 0.069 / 0.081 by type and 0.055 / 0.060 / 0.088 / 0.059 by condition. Short-history predictions and jackpot machines remain the largest misses. Direct A/B target calibration does not by itself recover every machine level; the remaining jackpot and short-history errors leave room for improved level inference.

Reproduce this analysis after the train-only pipeline with:

```sh
OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python3 mainFiles/sysid_win_form.py
```

It writes `mainFiles/runs/sysid-win/win_form.json`. Candidate definitions are frozen in `mainFiles/sysid_win_form.py`; each one is trained on one validation fold and scored only on the other. Ranking the fixed candidates uses both held-out scores, so the top-ranked candidate is reported as a screen result. The calibrated-history line and its interval use a single fixed form. All oracle-size values remain confined to win-only ceiling scoring.

## Fold-specific level-prior follow-up (new orchestrator measurement)

The new `PRIORS_FACTS.md` result is consistent with a binary parity state plus a continuous per-machine level: fitted pair amplitudes from validation truth are about 2.38–2.45 logits, while residual level SD is wider than the train model's 0.337 (`.47–.71` by fold/type). Signal slopes vary by type: balanced is about `.46–.48`, while jackpot is about `.36–.37`. Jackpot live intercepts fit to A/B are `.274/.367`, giving signal-zero live probabilities `.568/.591` before integrating the fitted residual SD. This corrects the earlier approximate `.50` history-based center, though it does not alone explain the largest jackpot misses.

I fit two-component logit levels separately by type on each tune fold, regressed dead/live centers on signal, and used the resulting type-specific midpoint, half-gap, signal slope, and pooled residual SD inside the finite parity/reversal likelihood. The other fold was scored with official validation targets and exact conditional sizes only for the win-only ceiling. Fold A/B level estimates were independent; no held-out target entered its model.

The best-supported predictive form is a logit followed by a logistic link: `z[i,t] = mu[type] + beta[type] * signal[t] + u[i] + A[type] * q[i] * x[i,t] + gamma * peer_win_lag3[t]`, with `P(win)=sigmoid(z)`. Here `x` is the learned one-of-three pair-parity feature, `q` is its machine-specific direction, `u` is a continuous machine intercept, and the time model averages static, per-machine reversal, and episode-shared reversal states that can flip the parity contribution. The signal term shifts the type-specific level; the parity state creates low/high bands; the Gaussian `u` and posterior uncertainty make individual probabilities continuous. This is a predictive approximation: the fold truth fits recover per-type dead/live level distributions and support the components, but do not prove every simulator term.

| Candidate win estimate | Cross-fitted win-only ceiling / 60 | 95% episode CI | Stable | Short | Drift | Shared |
|---|---:|---:|---:|---:|---:|---:|
| Train-selected history posterior, fixed-form target calibration | 58.105 | 57.513–58.545 | 0.0258 | 0.0595 | 0.0218 | 0.0233 |
| Fold-specific truth level priors in history likelihood | 58.107 | 57.513–58.574 | 0.0263 | 0.0587 | 0.0229 | 0.0224 |
| Screened history + recent-32 target fit (selection-set diagnostic) | 58.126 | 57.590–58.553 | 0.0214 | 0.0600 | 0.0214 | 0.0234 |
| Direct level-prior remap of history probability (failed) | 57.871 | 57.298–58.285 | 0.0268 | 0.0593 | 0.0290 | 0.0320 |
| Exact P(win), exact conditional size | 60.000 | oracle | 0 | 0 | 0 | 0 |

The fixed history calibration's fold scores are 57.927 on A and 58.247 on B; the direct type-specific likelihood candidate scores 57.927 on A and 58.255 on B. Its win RMSE is .0467/.0690/.0820 for steady/balanced/jackpot and .0563/.0584/.0877/.0620 for drift/shared/short/stable. The new level centers validate the generator decomposition but do not give a meaningful score gain over the existing posterior; direct probability remapping is worse. The largest measured remaining errors are jackpot (win RMSE .082) and short history (.088 under the earlier fixed form).

For reference, the complete train-only forecaster scores 57.445/60 (95% episode CI 56.882–57.895), versus pooled starter 30.000. Those scores include payout-size error and are not comparable to win-only ceilings above. Fold-prior fitting uses validation truth by design as a cross-fit system-identification diagnostic, not as a deployable train-only method. The executable is `mainFiles/sysid_level_prior.py`; run `OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python3 mainFiles/sysid_level_prior.py` after generating `mainFiles/runs/sysid-final/`. Results are written to ignored `mainFiles/runs/sysid-win/level_prior.json`.

## Handoff index and next experiment

Measured finding → reproducible source (all scripts run from the worktree root):

- Rule-pair enumeration and training-only model fit (99.105% posterior mass on `(1,3)/(2,5)/(4,7)`; amplitude 2.36663, intercept SD 0.33705, signal 0.38062, peer coefficient 0.48635): `mainFiles/sysid_bayes.py`, with feature names from `mainFiles/sysid_study.py`; final fitted model in ignored `mainFiles/runs/sysid-final/win_model*.json`.
- History likelihood, reversal and shared episode state checks (including condition-wise log likelihoods and the failed forgetting/static alternatives): `mainFiles/sysid_history.py`, `mainFiles/inspect_sysid.py`; full numbers in `mainFiles/SYSID.md`.
- True-target mode/continuous-logit diagnostics (Fold-A RMSE 2.45916 basic, 0.59735 with target-derived mode, 0.41995 with episode/own/peer terms): `mainFiles/sysid_oracle.py`.
- Conditional payout size structure and 2-per-type modes: `mainFiles/sysid_size.py`, independently checked across all validation targets in `mainFiles/sysid_win_form.py`.
- Win-form cross-fit ceilings and per-condition errors: `mainFiles/sysid_win_form.py`; cached diagnostic output `mainFiles/runs/sysid-win/win_form.json`. Primary fixed history candidate 58.1051; screened history+recent32 58.1258 (selection-set only).
- Fold-specific type x signal dead/live prior fit and likelihood replacement: `mainFiles/sysid_level_prior.py`; cached output `mainFiles/runs/sysid-win/level_prior.json`. Likelihood candidate 58.1069, direct remap 57.8709.
- End-to-end official score, train-only selection, and oracle decomposition: `mainFiles/sysid_pipeline.py`; official validation checker `mainFiles/check_submission.py`.

**Next experiment queued for Cursor:** extend the fixed history-logit candidate in `mainFiles/sysid_win_form.py` with (a) an episode intercept residual shared across machines and (b) a per-type amplitude/level uncertainty term, then compare against one additional candidate with a short-history-specific variance scale. Keep the current deterministic A/B alternation: fit the added coefficients on A targets and score B, then fit B and score A; report official win-only ceiling, paired episode-bootstrap CI, win RMSE for jackpot and short_history, and all four condition ratios. First test whether episode residuals correlate across machines after conditioning on history posterior (report within-episode residual correlation against shuffled-episode control); if absent, drop that candidate. Use no held-out target as a feature and retain the fixed 58.105 history form as the preregistered comparator. The aim is to address the measured jackpot RMSE .082 and short-history RMSE .088 without selecting on the fold being scored.

Completion marker: `worktrees/_coord/fresh/sysid.done` is intentionally empty. The branch remains local; do not push or open a PR under the shared brief.
