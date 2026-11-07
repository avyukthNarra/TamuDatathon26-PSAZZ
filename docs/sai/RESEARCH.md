# House Edge — Research and Modeling Notes

**Project:** TAMU Datathon 2026 · House Edge  
**Prepared:** November 7, 2026  
**Intended repo path:** `Sai/RESEARCH.md`  
**Status:** Literature-backed modeling plan; **not** an empirical report on the organizer CSVs.

> **Scope and evidence:** Challenge rules below are summarized from the [House Edge participant guide](https://docs.google.com/document/d/1CqNLjCiojRxKkZopoxBeilkke6Y4yCuz7FYSuefvCMo/edit). Research citations describe methods previously studied elsewhere. Their reported effectiveness **does not** imply an established improvement on this synthetic competition. Numerical examples are labeled as illustrative and are not measurements of the challenge data.

## TL;DR

Predict the **complete** five-category distribution of the next spin, not just the most likely result. Start with well-regularized, machine-type-specific frequencies; add true-tick lag features, recency weighting, and episode-level summaries. Evaluate separately for stable, short-history, drift, and shared-change conditions. Blend only models that produce out-of-episode validation improvements. Prioritize robustness, reproducibility, and correct cutoffs over expensive modeling.

## 1. Task specification

For each query, produce a distribution over payouts in this exact order:

`0×, 1×, 2×, 5×, 20×`.

With a machine-specific history `H_<t`, machine type `m`, observed signal `s`, and earlier histories of machines in the same episode, predict

\[
\hat{\mathbf p}_{e,j,t} = \big(\hat P(Y_{e,j,t}=0\mid\mathcal F_{t-1}),\ldots,\hat P(Y_{e,j,t}=20\mid\mathcal F_{t-1})\big).
\]

All five entries must be finite, nonnegative, and sum to 1. The model receives `query_id`, `episode_id`, `machine_id`, `machine_type` (`steady`, `balanced`, `jackpot`), `signal` (-1/0/1), and `cutoff`. Histories include `episode_id`, `machine_id`, `tick`, and `payout`.

**Crucial challenge features:**

- The data come from a **synthetic simulator**, not a real casino. Avoid importing real-world slot-machine house-edge numbers as priors without evidence.
- Payouts can interact at **lags up to eight actual ticks**; missing ticks are missing observations, **not** zeros.
- Machines of the same type may respond differently, including with opposite effects.
- Payout probabilities can drift, jump, or change together across machines in the same episode.
- Episodes are independent across train, validation, and test; keep whole episodes together in any resampling or model-selection split.
- Cutoffs range from 192–256. Short histories include 40 recent consecutive ticks and 24 sampled older ticks; inspect actual tick values.
- Conditions (`stable`, `short history`, `drift`, `shared change`) are labels in training/validation, **not** available to a test predictor.
- Training labels are **one realized next outcome** per query. Validation additionally exposes **true simulator probabilities**. Test does not expose answers.

Data kit: [organizer-provided ZIP](https://drive.google.com/file/d/1WXrMzPXHv_ZuwpKvdRrcKYoJaMR5J6VV/view). Keep competition inputs as the only modeling data unless rules expressly permit otherwise.

### Submission format

```csv
query_id,p_0,p_1,p_2,p_5,p_20
example_query,0.60,0.20,0.12,0.06,0.02
```

Main round: 80 possible points; adaptation: 20. Main deadline: Nov 7, 2026, 9 PM Central. Adaptation deadline: Nov 8, 2026, 11 AM Central. Confirm upload/packaging instructions against the participant guide, which controls over this note.

## 2. Scoring mathematics and implications

The error for a single query is the squared Euclidean distance between forecast probabilities `p` and hidden true probabilities `q`:

\[
L(p,q)=\sum_{k\in\{0,1,2,5,20\}}(p_k-q_k)^2.
\]

The contest averages errors within episode and then within condition, divides each condition's error by a frozen condition-specific baseline error (floored at `1e-9`), averages the four ratios into `R`, and awards

\[
\text{round points}=\frac{\text{round maximum}}{1+R}.
\]

**Implication:** optimize the contest's **condition-balanced, episode-balanced** validation measure, not simply the unweighted average query loss. Keep the baseline's validation technical points as the anchor. Avoid optimizing against hidden test outputs or trying to infer them.

### Why predicting observed outcomes is a valid training strategy

For one realized payout `Y ~ Categorical(q)`, define the multiclass Brier loss

\[
B(p,Y)=\sum_k(p_k-\mathbf 1\{Y=k\})^2.
\]

Its expectation is

\[
\mathbb E[B(p,Y)] = \sum_k(p_k-q_k)^2 + 1-\sum_kq_k^2.
\]

The last two terms do **not** depend on `p`, so the forecast minimizing expected Brier loss also minimizes the squared probability error. This is a special case of proper-scoring-rule theory [1]. In finite samples, however, realized outcome labels are noisy; validation's known true probabilities provide a more direct model-comparison signal.

### Rare-event sparsity: a concrete example

Suppose (illustratively) a machine has independent spins with fixed `P(20×)=0.02`. Then the probability of seeing **zero** 20× outcomes in `n` spins is `(0.98)^n`:

| Spins observed | Chance of zero 20× payouts |
|---:|---:|
| 20 | 66.8% |
| 40 | 44.6% |
| 64 | 27.4% |
| 128 | 7.5% |

These are *binomial calculations*, **not** measured simulator rates. They show why a raw observed frequency of zero can be weak evidence for a truly zero probability. Smoothing and partial pooling are especially important for short histories and the rare `20×` class.

## 3. Relevant published research

### [1] Proper scoring rules — foundational objective

**Gneiting, T. & Raftery, A. E. (2007).** *Strictly Proper Scoring Rules, Prediction, and Estimation*. Journal of the American Statistical Association.  
Paper: https://doi.org/10.1198/016214506000001437

- **Published result:** develops proper scoring rules, including quadratic scoring, which reward honest probability estimates.
- **House Edge use:** justifies forecasting five non-degenerate probabilities rather than a single categorical guess; connects training on outcomes to test scoring on actual distributions.
- **Limitation:** theory; does not choose between forecasting architectures for this dataset.

### [2] Online Bayesian changepoint detection — abrupt distribution shifts

**Adams, R. P. & MacKay, D. J. C. (2007).** *Bayesian Online Changepoint Detection*.  
Paper: https://arxiv.org/abs/0710.3742

- **Published result:** derives online inference for the posterior distribution of time since the most recent changepoint and demonstrates it on three real-world datasets.
- **House Edge use:** detect when earlier machine observations become less relevant; integrate over potential recent change times rather than using a fixed lookback.
- **Candidate implementation:** separate recent and long-run categorical models and assign a change probability based only on pre-cutoff observations. Full run-length filtering is optional.
- **Limitation:** detection under short, sparse histories may be unstable; use shrinkage and validate per condition.

### [3] Adaptive Windowing (ADWIN) — variable memory length

**Bifet, A. & Gavaldà, R. (2007).** *Learning from Time-Changing Data with Adaptive Windowing*. SIAM International Conference on Data Mining.  
Paper: https://doi.org/10.1137/1.9781611972771.42

- **Published result:** develops adaptive sliding windows for drifting sequences, with formal detection guarantees under its assumptions.
- **House Edge use:** compare short versus long histories in a data-driven way; long windows reduce variance on stable machines, shorter windows can adapt to drift.
- **Limitation:** a generic change detector should not automatically replace a benchmarked recency-weighted baseline.

### [4] Hidden Markov models — latent regimes

**Rabiner, L. R. (1989).** *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition*. Proceedings of the IEEE.  
Paper: https://doi.org/10.1109/5.18626

- **Published result:** reviews hidden-state transition/emission models and forward–backward/Viterbi inference.
- **House Edge use:** model hypothetical persistent hidden regimes whose probabilities imply different payouts. Filter state probabilities from prior ticks and forecast the next distribution.
- **Limitation:** no evidence yet that the simulator literally uses an HMM; the architecture is a useful approximation, not a reverse-engineered fact.

### [5] Hierarchical shrinkage for sparse multinomial probabilities

**Wickramasinghe et al. (2023).** *Bayesian Inference on Sparse Multinomial Data Using Smoothed Dirichlet Distribution*.  
Paper: https://doi.org/10.3233/MAS-221411

- **Published result:** studies pooling/shrinkage for sparse multinomial estimation.
- **House Edge use:** a robust category-probability estimate with type-level priors and machine-level observations; especially helpful where no 20× payout has been observed.
- **Limitation:** the exact prior strength must be tuned **only** using allowed training/validation procedures.

### [6] CatBoost — nonlinear lag and context interactions

**Prokhorenkova, L., et al. (2018).** *CatBoost: Unbiased Boosting with Categorical Features*. NeurIPS.  
Paper: https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html

- **Published result:** introduces ordered boosting and categorical-feature processing to reduce training-time prediction shift, with empirical benchmarks on other datasets.
- **House Edge use:** multiclass probability estimates based on lag-one through lag-eight payouts, machine type, signal, recent category proportions, and episode summaries.
- **Limitation:** avoid future-label leakage when generating supervised rows; the submission environment may require approved libraries or vendored compatible wheels.

### [7] Calibration — quality of probabilities matters

**Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017).** *On Calibration of Modern Neural Networks*. ICML/PMLR.  
Paper: https://proceedings.mlr.press/v70/guo17a.html

- **Published result:** shows that confidence calibration can be poor in modern neural models; temperature scaling improves calibration on many studied datasets.
- **House Edge use:** compare temperature scaling, simple shrinkage toward a prior, and uncalibrated predictions using *true validation probabilities*.
- **Limitation:** improved conventional calibration does not guarantee lower simulator-probability squared error.

### [8] Multivariate latent factors — shared episode effects (advanced)

**Kim, Düker, Fisher & Pipiras (2025 online / 2026 issue).** *Latent Gaussian Dynamic Factor Modeling and Forecasting for Multivariate Count Time Series*.  
Paper: https://doi.org/10.1111/jtsa.70016

- **Published result:** studies dynamic latent factors across related count series.
- **House Edge use:** motivates an episode-level latent component inferred from **earlier** activity on peer machines.
- **Limitation:** this is an analogy, not a direct categorical-payout solution; model complexity may not be worthwhile for a timed challenge.

## 4. Candidate models and equations

### A. Smoothed, hierarchical categorical baseline (highest priority)

For payout category `k`, let `n_{j,k}` be the count in machine `j`'s observable history, and let `q_{type(j),k}` be a prior probability estimated from the **training set** for the corresponding machine type. A simple posterior mean is

\[
\hat p_{j,k}=\frac{n_{j,k}+\alpha q_{type(j),k}}{N_j+\alpha}.
\]

Test several `alpha` values (e.g. 2, 8, 24, 64); don't treat these as known optima. More robust hierarchical possibilities mix global, type, and episode-level estimates with separately tuned weights. Re-estimate priors without leakage when generating pseudo-out-of-sample training features.

**Recency-weighted variant:** replace counts by `n^*_{j,k} = sum_{t_i<cutoff} w_i 1{y_i=k}` where `w_i=2^{-(cutoff-t_i)/h}` and `h` is a half-life in **actual ticks**. Normalize using total effective count `sum_i w_i`. Compare half-lives (e.g. 8, 16, 24, 48, infinity) by episode-held-out validation.

### B. Regularized multinomial logistic regression (very good next step)

Construct features containing:

- One-hot or embedded `machine_type`, `signal` and their interactions.
- Exact-tick categorical one-hot indicators at lags **1–8**; include **missing** as a distinct state.
- Payout frequencies in windows ending **before** cutoff (8, 16, 32, 64, all available).
- Time since each outcome, recent-versus-long-run frequency differences, and observed-history length.
- Episode peer-machine aggregates using **only earlier ticks**, with peer counts or coverage indicators.
- Machine identifiers only if they legitimately repeat across independent training contexts; avoid high-cardinality memorization.

Fit a regularized softmax classifier on observed next outcomes, using group/episode-safe evaluation. This can capture machine-type × signal × lag interactions without requiring an expensive model.

### C. Gradient boosted trees (strong nonlinear candidate)

Use the same leakage-safe features and train a multiclass estimator. CatBoost/LightGBM/XGBoost are options only if allowed by the reproduction environment. Use modest model complexity, early stopping through properly separated episodes, and average probability outputs across seeds if stable improvements justify it.

### D. Changepoint-aware mixture (target drift)

Combine long-run and recent estimates:

\[
\hat p = (1-g(H_{<t}))\,\hat p_{long} + g(H_{<t})\,\hat p_{recent}.
\]

`g` can be a bounded function of divergence between recent and long-run empirical distributions, a formal changepoint posterior, or a learned gating model. Fit it using training data, then check performance separately on stable and drift conditions. Avoid brittle near-0/near-1 gates.

### E. Hidden-state / shared-factor models (advanced)

For an HMM, define latent state `Z_t`, transition `P(Z_t|Z_{t-1})` and categorical emissions `P(Y_t|Z_t, type, signal)`. Filter the posterior of `Z_t` from observations at times `< cutoff` before forecasting the next emission distribution. For shared episodes, include peer history summaries or a jointly inferred episode state. Hidden state and peer effects are hypotheses to be tested, not known simulator mechanics.

### F. Ensemble and postprocessing

Blend eligible models by convex weights:

\[
\hat p^{ens}_k=\sum_{m=1}^{M}w_m\hat p^{(m)}_k,\qquad w_m\ge0,\ \sum_mw_m=1.
\]

Fit weights on an episode-held-out tuning subset, and evaluate on untouched episodes. If needed, shrink ensemble probabilities toward a type-level prior; renormalize and keep numerical precision. Report results for each of the four conditions; a strong global gain can hide a severe conditional regression.

## 5. Validation design: avoid false improvements

1. **Run and preserve the starter.** Use `run_starter.py`, its validation outputs, `validation_score.json`, and `run.json`. Record the frozen pooled reference's performance.
2. **Replicate the actual contest metric.** Compare the checker's `validation.contest.technical_points`; don't optimize the older diagnostic score by mistake.
3. **Split by episode** for additional training/selection folds. Never split individual queries from the same episode between training and evaluation.
4. **Use only data with `tick < cutoff`.** For a pseudo-training example at tick 150, use earlier data to build its features and use payout at 150 **only** as its label. Any observation at the cutoff from any peer machine is unavailable.
5. **Handle missing ticks explicitly.** A two-row lag is not the same as a two-tick lag. Never impute a missing payout as category 0.
6. **Treat validation true probabilities as an evaluation resource.** They are available in the provided validation split, but repeated extensive tuning on the entire split can overfit. Use the experiment runner's split-within-condition approach or reserve untouched validation episodes.
7. **Measure heterogeneity.** Report both overall technical points and per-condition episode-normalized squared errors. Add a runtime column.
8. **Check reliability.** Use multiple seeds or bootstrap resampling of **episodes**, not individual rows, for uncertainty around model comparisons.
9. **Preserve reproducibility.** Save model parameters, code, feature specs, random seeds, dependency versions, and hash/round identifiers.

### Suggested experiment matrix (test sequentially)

| ID | Change versus baseline | Main hypothesis | Watch for |
|---|---|---|---|
| E0 | Organizer `pooled` and `pooled_recent` | Establish reference | Correct technical score |
| E1 | Type-level Dirichlet smoothing | Sparse classes improve | Poor priors for unusual machines |
| E2 | Half-life grid / window lengths | Better drift adaptation | Stable-condition regression |
| E3 | Exact-tick lag 1–8 features | Temporal patterns matter | Missing-tick mistakes |
| E4 | Signal × type × lag interactions | Different machine responses | Overfitting / high variance |
| E5 | Peer episode summaries | Shared changes matter | Same-tick leakage |
| E6 | Gradient-boosted categorical model | Nonlinearities matter | Runtime / calibration |
| E7 | Changepoint gating | Recent regimes matter | False alarms on stable machines |
| E8 | Convex ensemble | Models make complementary errors | Tuning leakage |

An ablation table should contain: `experiment`, `seed`, `training groups`, `validation groups`, `technical_points`, `stable_error`, `short_error`, `drift_error`, `shared_error`, `runtime_sec`, and `notes`.

## 6. Statistics to compute **from the released data**

> These are **proposed descriptive statistics**, not published competition findings. Do not fill in values without running the analysis on released data.

1. Number of episodes, unique machines, queries and observation rows per split; category counts by machine type, signal, and condition.
2. Empirical overall and type-specific payout proportions; frequency of machine histories with zero observed `20×` payouts.
3. Lag-conditioned category distributions `P(Y_t=k | Y_{t-ell}=r, type, signal)` for `ell=1,...,8`; supply support counts and shrink heavily where rare.
4. Short-window vs long-window distribution divergences; compare predictive relevance to drift-condition labels **only on train/validation**.
5. Peer-machine lagged associations within episodes; compare with null controls and include group size/coverage.
6. Model error breakdown by true-probability quintiles, machine type, history availability, and condition.
7. Reliability plots comparing forecasted versus observed outcome frequencies as a **secondary** diagnostic; evaluate true-probability squared loss as the primary validation statistic.

When publishing any resulting table, label the split, number of episodes, number of queries, exact metric, and whether the statistic is observed-outcome based or true-probability based.

## 7. Tactical implementation plan

**First pass:** run the starter, preserve baseline outputs, score `pooled_recent`, and audit the exact CSV format. Then implement smoothed time-weighted categorical frequencies (machine-level and type-level). Add exact-tick 1–8 features to a regularized multinomial model; compare condition-by-condition. Add peer episode summaries only if they provide genuine validation gain. Test a simple convex ensemble last.

**Time permitting:** try nonlinear boosted trees and simple changepoint gating. Reserve full HMM/dynamic-factor implementations for clear evidence that simpler models miss persistent or shared structure.

**Packaging:** ship both `submission.csv` and a reproducible source ZIP according to the guide. The submission runner has no network; default Python is 3.13 standard-library-only, and the guide says to use an approved environment or provide compatible offline dependencies. The kit includes `check_submission.py` and `build_entry.py`; use them. Keep probability rounding errors well under the allowed tolerance and verify the final CSV matches the ZIP's recreated predictions.

## 8. Evidence and caveats

- **Directly established from the guide:** allowed payout categories; input schema; four conditions; tick/cutoff rules; challenge scoring and deadlines; hidden true odds; existence of validation true probabilities.
- **Established from publications:** proper scoring rules; online changepoint methods; adaptive windowing; HMM inference; sparse categorical shrinkage; boosted trees; and calibration principles.
- **Hypotheses requiring local testing:** which features are most useful, whether an HMM approximates the simulator, whether cross-machine sharing is strong, optimal priors and half-lives, and whether any sophisticated ensemble beats a frequency baseline.
- **Not claimed:** any measured competition score, learned machine odds, real casino payout frequencies, or empirical improvements from papers on this exact simulator.

## References

1. Gneiting, T. & Raftery, A. E. (2007). *Strictly Proper Scoring Rules, Prediction, and Estimation*. JASA. https://doi.org/10.1198/016214506000001437
2. Adams, R. P. & MacKay, D. J. C. (2007). *Bayesian Online Changepoint Detection*. https://arxiv.org/abs/0710.3742
3. Bifet, A. & Gavaldà, R. (2007). *Learning from Time-Changing Data with Adaptive Windowing*. SIAM SDM. https://doi.org/10.1137/1.9781611972771.42
4. Rabiner, L. R. (1989). *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition*. Proceedings of the IEEE. https://doi.org/10.1109/5.18626
5. Wickramasinghe et al. (2023). *Bayesian Inference on Sparse Multinomial Data Using Smoothed Dirichlet Distribution*. https://doi.org/10.3233/MAS-221411
6. Prokhorenkova, L. et al. (2018). *CatBoost: Unbiased Boosting with Categorical Features*. NeurIPS. https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html
7. Guo, C. et al. (2017). *On Calibration of Modern Neural Networks*. ICML. https://proceedings.mlr.press/v70/guo17a.html
8. Kim, Düker, Fisher & Pipiras. (2025/2026). *Latent Gaussian Dynamic Factor Modeling and Forecasting for Multivariate Count Time Series*. https://doi.org/10.1111/jtsa.70016
9. TAMU Datathon 2026. *House Edge — Participant Guide*. https://docs.google.com/document/d/1CqNLjCiojRxKkZopoxBeilkke6Y4yCuz7FYSuefvCMo/edit
