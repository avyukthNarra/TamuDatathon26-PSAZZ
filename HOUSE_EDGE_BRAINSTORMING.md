# House Edge: Research and Brainstorming

**Event:** TAMU Datathon 2026  
**Notes date:** November 7, 2026  
**Status:** Research and proposed experiments. No models in these notes have been implemented or benchmarked.

[Participant guide](https://docs.google.com/document/d/1CqNLjCiojRxKkZopoxBeilkke6Y4yCuz7FYSuefvCMo/edit?tab=t.0)

These notes connect the challenge to relevant research and outline experiments. Numerical examples are hypothetical. Applications to House Edge are proposed adaptations, not results established by the cited papers. Check the current participant kit, especially RULES.md and DATA.md, before implementation.

## 1. What we are predicting

For each query, estimate the next-spin probability distribution over payouts **0x, 1x, 2x, 5x, and 20x**.

Example:

~~~text
[0.60, 0.20, 0.12, 0.06, 0.02]
~~~

The output is a distribution, not a single guessed payout or an expected chip return. Payouts include the original wager: 1x breaks even, while 5x gives a net gain of four chips.

### Three different concepts

| Concept | Meaning |
| --- | --- |
| Episode | A group of related machines observed at the same cutoff. |
| Machine type | A provided category for an individual machine: steady, balanced, or jackpot. |
| Evaluation condition | A scenario used to evaluate forecasts: stable, short history, drift, or shared change. |

An episode can contain multiple machine types. Two machines of the same type can behave differently. The machine type **steady** and evaluation condition **stable** are different labels.

Each query asks about a particular machine in a particular episode at a cutoff tick. Histories must be grouped by both episode_id and machine_id.

### True odds versus observed outcomes

The simulator has actual payout probabilities. Each observed payout is a random draw from those probabilities.

Seeing no jackpots in a short history does not establish that the jackpot probability is zero. Likewise, a correct probability forecast can be followed by an unlikely outcome.

Some machine state is hidden, so exact recovery of the simulator's odds may be impossible.

## 2. Data and evaluation constraints

| Split | Queries and histories | Observed next payouts | True probability vectors | Condition labels |
| --- | --- | --- | --- | --- |
| Training | Yes | Yes | Not provided | Yes |
| Validation | Yes | Yes | Yes | Yes |
| Test | Yes | No | No | No |

Training, validation, and test contain separate episodes.

For cutoff 200, only observations with tick < 200 are valid inputs, including observations from other machines. Other-machine information must come from the same episode.

The guide describes cutoffs from 192 to 256. Long histories contain all earlier ticks; short histories contain 40 recent consecutive observations and 24 sampled older observations.

**A missing tick is unknown, not a zero payout.** Exact tick lags are not necessarily equivalent to row offsets.

The guide also states that past payouts can interact at delays up to eight ticks, machines of the same type may respond differently or in opposite directions, and changes may affect several machines together.

### What the four conditions test

| Condition | Core difficulty | Example research question |
| --- | --- | --- |
| Stable | Separating consistent behavior from random variation | Are we overreacting to a streak? |
| Short history | Making forecasts with limited evidence | How much should we borrow from other training machines? |
| Drift | Recognizing changing underlying behavior | Which older observations still describe the present? |
| Shared change | Using related machines to infer a common change | Does episode-wide evidence improve the target machine's forecast? |

Stable rules can still produce context-dependent next-spin odds. A predictable response to recent payouts is not automatically evidence of drift.

The condition labels describe scenarios. The probability error within each scenario measures forecast quality.

### Official score

For query q:

$$
L_q = \sum_{k \in \{0,1,2,5,20\}}(\hat p_{q,k}-p_{q,k})^2.
$$

The scorer averages within each episode, then across episodes within each condition. Each condition's error is divided by its frozen reference error, with a denominator floor of 1e-9. The four ratios are averaged to obtain R.

$$
\text{round points} = \frac{\text{maximum round points}}{1+R}.
$$

Main is worth 80 points and adaptation is worth 20. Matching the reference in both rounds gives 50 total points. Points are not classification accuracy.

Use the official checker and technical_points for model selection, rather than top-category accuracy or the older generic diagnostic score.

## 3. Research map

| Research direction | Question | Initial implementation idea |
| --- | --- | --- |
| Smoothing / shrinkage | How much should we trust a short machine history? | Blend local categorical frequencies with a learned prior. |
| Recency / change detection | Which observations remain relevant after behavior changes? | Decay old counts or average predictions over possible change times. |
| Context models | Does the order of previous payouts matter? | Smoothed conditional counts with fallback to shorter patterns. |
| Feature-based learning | How do several clues interact? | CatBoost with historical, lag, and episode features. |
| Hidden-state models | Is behavior switching between latent modes? | Categorical HMM with probabilistic state inference. |
| Neural sequence models | Can useful history representations be learned directly? | Small TCN or shared recurrent model with five-class outputs. |
| Hierarchical dynamics | How do shared patterns coexist with individual machine behavior? | Shared parameters plus local and episode states. |
| Calibration | Are predicted probabilities too concentrated or too diffuse? | Validate a low-dimensional probability adjustment. |

## 4. Papers and possible applications

### 4.1 Shrinkage estimation: combine local evidence with a prior

**Paper:** [Entropy Inference and the James-Stein Estimator, with Application to Nonlinear Gene Association Networks](https://jmlr.org/papers/v10/hausser09a.html)  
**Authors:** Jean Hausser and Korbinian Strimmer, 2009

The useful idea is regularizing categorical probability estimates when observations are limited. Read the Bayesian estimation and shrinkage portions before the gene-network application.

For a Dirichlet prior with total strength alpha and mean vector q:

$$
\hat p_k = \frac{n_k+\alpha q_k}{N+\alpha}.
$$

Here n_k is the machine's count for category k, N is its number of observations, and alpha represents the prior's effective sample size.

Example with N = 20 and alpha = 20:

| Payout | Machine count | Local frequency | Prior probability | Blended probability |
| --- | --- | --- | --- | --- |
| 0x | 15 | 75% | 60% | 67.5% |
| 1x | 3 | 15% | 20% | 17.5% |
| 2x | 2 | 10% | 12% | 11% |
| 5x | 0 | 0% | 6% | 3% |
| 20x | 0 | 0% | 2% | 1% |

**Proposed experiment:** Compare a global prior, a type-conditioned prior, local frequencies, and blends with several prior strengths.

The type-conditioned prior is a proposed application, not a specific result from the paper. More elaborate hierarchical pooling should allow individual machines to differ.

### 4.2 Bayesian Online Changepoint Detection: uncertain change times

**Paper:** [Bayesian Online Changepoint Detection](https://arxiv.org/abs/0710.3742)  
**Authors:** Ryan Prescott Adams and David J. C. MacKay, 2007

The method tracks a distribution over run length: time since the most recent change. It averages forecasts over plausible run lengths instead of selecting one change time with certainty.

Hypothetical simplified example:

| Explanation | Probability | Estimated chance of any payout |
| --- | --- | --- |
| A recent change occurred | 80% | 20% |
| A longer period remains relevant | 20% | 50% |

The combined chance is 0.8 * 0.20 + 0.2 * 0.50 = 0.26. The actual House Edge adaptation would average full five-category distributions.

**Proposed experiment:** Use categorical observations with a Dirichlet prior; compare with fixed windows and exponential decay.

**Caveats:** Basic segment models assume independent observations within a regime. Gradual drift and lag-dependent behavior may need extensions. Missing ticks also affect how elapsed time and change probabilities should be handled.

**Read first:** Introduction, Figure 1, and recursive run-length estimation.

### 4.3 Context-Tree Weighting: uncertain memory length

**Paper:** [The Context-Tree Weighting Method: Basic Properties](https://pure.tue.nl/ws/portalfiles/portal/1383848/Metis122608.pdf)  
**Authors:** Frans M. J. Willems, Yuri M. Shtarkov, and Tjalling J. Tjalkens, 1995

The paper combines predictive distributions from bounded-memory context structures. Its original formulation concerns binary sequences.

For this challenge, the transferable idea is to combine evidence based on no recent context, the last payout, the last two payouts, and longer patterns.

An unrestricted eight-payout context has 5^8 = 390,625 possible patterns. One machine's history cannot support independent reliable estimates for all of them.

**Proposed experiment:** A categorical context model with smoothing and fallback to shorter patterns. Include machine-specific evidence so pooled relationships do not erase opposing responses across machines.

This would be inspired by CTW; a simple fallback estimator should not be described as an exact CTW implementation.

**Caveats:** Five categories, sparse tick histories, drift, and interactions at selected delays need explicit handling.

### 4.4 CatBoost: combine features and learn interactions

**Paper:** [CatBoost: Unbiased Boosting with Categorical Features](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html)  
**Authors:** Liudmila Prokhorenkova et al., 2018

CatBoost is a gradient-boosting method with techniques for categorical features and reducing prediction shift from a particular kind of target leakage.

For House Edge, each training row would describe the available history and use the observed next payout as its target.

Candidate features:

- Machine type and provided signal.
- Observation count and missingness.
- Per-category frequencies over several windows.
- Exponentially weighted frequencies.
- Differences between recent and older distributions.
- Exact tick lags from one through eight.
- Selected lag-pair interactions.
- Machine-specific conditional-frequency summaries.
- Other-machine summaries within the same episode, before the cutoff.

**Proposed experiment:** Train a five-class classifier and evaluate predict_proba outputs with the official scorer.

**Caveats:** Internal ordered boosting does not enforce our cutoff rules or episode grouping. A categorical training objective is useful, but model selection must use the contest score. Do not expect an arbitrary unseen machine ID to provide transferable behavioral information.

### 4.5 Hidden Markov Models: recurring latent modes

**Paper:** [A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition](https://doi.org/10.1109/5.18626)  
**Author:** Lawrence R. Rabiner, 1989

An HMM represents an unobserved state that changes over time, with observations emitted according to that state.

A hypothetical machine could switch between a low-payout mode and a higher-payout mode. Histories provide evidence about its current state. Forecasting combines next-state probabilities with each state's payout probabilities.

A hidden state is a modeling assumption, not a machine type or evaluation condition.

**Proposed experiment:** Start with a small categorical HMM and compare it with simpler estimators. Consider pooling parameters across training sequences while retaining uncertainty about each sequence's state.

**Caveats:** A few recurring modes may not describe the simulator. Hidden-state probabilities are not themselves payout probabilities. Account for unobserved ticks when propagating state.

**Tutorial:** [hmmlearn Dishonest Casino example](https://hmmlearn.readthedocs.io/en/stable/auto_examples/plot_casino.html), which demonstrates switching between fair and loaded dice.

### 4.6 Temporal Convolutional Networks: learn delayed patterns

**Paper:** [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271)  
**Authors:** Shaojie Bai, J. Zico Kolter, and Vladlen Koltun, 2018

The paper describes a TCN built from causal convolutions, dilation, and residual connections. Dilation expands the historical receptive field without requiring a very large filter.

**Proposed experiment:** A small TCN consuming payout-category embeddings, observation masks, and context, with a five-category output.

Its filters could learn combinations involving different historical delays. Set the receptive field to cover relevant recent ticks and investigate whether longer context helps.

**Caveats:** Causal layers do not automatically prevent target leakage: predicting tick t must use payouts strictly before t. Missing ticks must be represented rather than removed from time. Neural complexity should be justified by validation gains.

**Read first:** Section 3 and Figure 1.

### 4.7 DeepAR: shared learning, sequence-specific memory

**Paper:** [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://arxiv.org/abs/1704.04110)  
**Authors:** David Salinas et al., first posted 2017

DeepAR learns a shared autoregressive recurrent forecasting model across many related time series. Its hidden state summarizes the particular sequence being processed.

The published experiments use Gaussian and negative-binomial likelihoods. House Edge would need a categorical output over the five payout categories.

**Proposed experiment:** One shared GRU or LSTM with contextual inputs and separate recurrent state for each machine sequence. Reset sequence state appropriately; do not let it accidentally carry information between unrelated machines.

**Caveats:** This is a categorical adaptation, not a standard demand-forecasting configuration. Shared network weights do not automatically model episode-wide changes; supply episode information or extend the architecture.

**Read first:** Model, likelihood choice, and training sections.

### 4.8 Deep state-space models: common structure with evolving local state

**Paper:** [Deep State Space Models for Time Series Forecasting](https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html)  
**Authors:** Syama Sundar Rangapuram et al., 2018

The paper combines a jointly learned recurrent network with per-series linear state-space models. It separates learned common structure from local evolving state.

A possible House Edge extension would include:

- Shared learned relationships.
- Machine-specific state and response parameters.
- An episode-level state inferred from related machines.
- A categorical observation distribution.

**Caveats:** Categorical emissions and a shared episode state are substantial proposed extensions, not ready-made components of the paper. They require additional modeling and inference work.

**Priority:** Later research direction, especially if simpler models reveal persistent problems with machine heterogeneity or shared changes.

### 4.9 Probability calibration: adjust confidence

**Paper:** [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)  
**Authors:** Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger, 2017

The paper studies mismatches between classifier confidence and empirical correctness, and investigates post-processing methods including temperature scaling.

For logits z and positive temperature T:

$$
p_k = \operatorname{softmax}(z/T)_k.
$$

Higher temperatures spread probabilities more broadly; lower temperatures make them more concentrated. Positive temperature scaling preserves the highest-scoring category.

**Proposed experiment:** Choose a low-dimensional confidence adjustment on tuning episodes, then evaluate on separate held-out episodes using the official score.

**Caveats:** Calibration cannot recover missing predictive information. Better calibration diagnostics do not necessarily guarantee a better contest score. One temperature cannot fix every category-specific error.

### 4.10 Proper scoring rules: probability quality is the objective

**Paper:** [Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)  
**Authors:** Tilmann Gneiting and Adrian E. Raftery, 2007

The paper develops the theory of scoring probabilistic forecasts, including logarithmic and quadratic/Brier scores.

For a true vector p, prediction q, and one-hot observed outcome e_Y:

$$
\mathbb{E}[\|q-e_Y\|^2]
=
\|q-p\|^2 + 1-\|p\|^2.
$$

For fixed p, the final terms do not depend on q. This explains why observed outcome labels can support learning probability forecasts despite their noise.

The competition evaluates squared error against true probabilities and adds its episode/condition aggregation. It is not simply observed-outcome classification accuracy.

**Read first:** Section 3 and the quadratic/Brier example.

## 5. Practical tools and tutorials

| Tool | Role | House Edge use |
| --- | --- | --- |
| [CatBoost](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_predict_proba) | Feature-based classifier with probability outputs | Combine historical summaries, lags, and episode evidence. |
| [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) | Nonoverlapping grouped evaluation | Group by episode_id when constructing internal evaluation folds. |
| [River ADWIN](https://riverml.xyz/dev/api/drift/ADWIN/) | Adaptive-window change detector | Supply change features or help select relevant history. |
| [ruptures](https://centre-borelli.github.io/ruptures-docs/) | Retrospective change-point detection | Analyze and segment the prefix available before each cutoff. |
| [hmmlearn tutorial](https://hmmlearn.readthedocs.io/en/stable/tutorial.html) | Hidden-state model fitting and inference | Prototype categorical HMMs and understand state transitions and emissions. |
| [PyMC multilevel tutorial](https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/multilevel_modeling.html) | Partial pooling and varying group effects | Learn how to share evidence while permitting individual machine differences. |
| [PyTorch GRU documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) | Custom neural sequence modeling | Build recurrent models with a five-class output; PyTorch also supports a custom TCN. |
| [Optuna](https://optuna.readthedocs.io/en/stable/) | Hyperparameter search | Tune smoothing, memory, complexity, and blend weights against the official validation objective. |
| [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html) | Probability evaluation and adjustment | Learn calibration methods; ensure calibration splits respect episode grouping. |

ADWIN and ruptures do not produce five-category forecasts by themselves. A detector must be paired with a probability estimator or used as a feature source.

For a scalar detector, per-payout indicator streams are candidate inputs. Average payout alone can conceal changes because different distributions can have the same mean.

## 6. Proposed experiment sequence

All experiments below are planned, not completed.

| ID | Change | Hypothesis | What to compare |
| --- | --- | --- | --- |
| E00 | Run the supplied starter and preserve its outputs | Establish a valid reproducible reference | Official total and condition scores |
| E01 | Compare local frequencies and learned priors | Local-only estimates are noisy on short histories | Stable and short-history errors |
| E02 | Tune shrinkage strength and prior grouping | Pooling can improve probability estimates | Total score and all conditions |
| E03 | Add windows or exponential decay | Recent evidence helps when behavior changes | Drift gains versus stable-condition cost |
| E04 | Add change-point or detector-derived estimates | Adaptive memory beats one fixed memory setting | Drift and shared-change errors |
| E05 | Add exact lag and conditional-pattern information | Order contains predictive information | Gains over frequency-only estimates |
| E06 | Train CatBoost on the same permitted evidence | Nonlinear combinations improve forecasts | Total score and per-condition errors |
| E07 | Add other-machine episode summaries | Related machines reveal common changes | Ablation of episode features |
| E08 | Compare a small HMM, TCN, or recurrent model | Learned sequence/state representations add value | Same episode-level holdout and runtime |
| E09 | Test calibration or a small probability blend | Models may make complementary errors | Independent held-out score |

Suggested first candidate settings, not simulator properties:

- Several prior strengths rather than one assumed value.
- Recent windows such as 8, 24, and 64 ticks.
- Exponential-decay half-lives such as 8, 24, and 64 ticks.
- Exact lag features from one through eight ticks.
- A small number of nonnegative ensemble weights summing to one.

An exponentially weighted categorical estimator can use:

$$
w_t = 2^{-(c-1-t)/h},
\qquad
\hat p_k =
\frac{\sum_{t<c} w_t \mathbf{1}(y_t=k)+\alpha q_k}
{\sum_{t<c} w_t+\alpha}.
$$

Here c is the cutoff and h is the half-life in ticks. This is a useful heuristic; fractional weighted counts are not the ordinary unweighted Dirichlet posterior update.

## 7. Evaluation and implementation discipline

- Keep entire episodes together in internal splits.
- Use only information strictly earlier than each prediction cutoff.
- If creating examples at historical training ticks, do not use that tick's payout as an input.
- Only attach context fields to historical examples if they were actually known at that time. Do not assume the query's current signal applied to older ticks.
- Do not treat validation or test condition labels as predictor inputs; test conditions are unavailable.
- Represent missing ticks explicitly.
- Avoid learning meaningless transferable identities from episode-local machine IDs.
- Evaluate each condition and the official total. Keep a genuinely held-out group of episodes when tuning.
- Do not repeatedly tune against a purported final holdout and continue calling it independent.
- Preserve a working baseline and use separate output directories for experiments.
- Record versions, settings, seeds, runtime, and exact model files.
- Confirm probability-column order, class-label mapping, normalization, and finite outputs.
- Keep model comparisons on the same evaluation episodes.
- Test probability blends on a separate assessment split after choosing their weights.

### Experiment log template

| Run | Hypothesis / change | Split definition | Settings / seed | Stable error | Short-history error | Drift error | Shared-change error | Official points | Runtime | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pending | Baseline | Pending | Pending | — | — | — | — | — | — | Pending |

## 8. Reproducibility and submission reminders

The guide requires a test CSV plus a matching code ZIP for each round.

~~~csv
query_id,p_0,p_1,p_2,p_5,p_20
example_query,0.60,0.20,0.12,0.06,0.02
~~~

- Replace the example ID with actual test query IDs.
- Include every required query exactly once and no extra columns.
- Probabilities must be finite, between zero and one, and sum to one within the required tolerance.
- Package the actual inference code, preprocessing, settings, saved model, and offline dependencies.
- The custom entry command must accept data and output paths.
- Reproduce the final CSV from a fresh folder before uploading.
- Saving a trained model can avoid expensive or random retraining during verification.
- Confirm the selected submission and reproduction status.

At the time these notes were written, the guide listed default verification limits of two CPU cores, 16 GiB RAM, and 30 minutes, with no network access. Confirm the frozen settings and approved environments on the submission page.

The main deadline is November 7 at 9 PM Central. Adaptation opens then and closes November 8 at 11 AM Central. Adaptation has new query IDs and requires a separate entry.

## 9. Open questions for inspecting the participant kit

- [ ] How many episodes, machines, and queries are in each split?
- [ ] What are the exact machine-type and outcome frequencies?
- [ ] How is signal defined, and is any historical signal available?
- [ ] How are condition labels organized in the released files?
- [ ] Which baseline performs best under the official score?
- [ ] Are apparent lag relationships consistent across machines or machine-specific?
- [ ] How much do recent frequencies differ from older frequencies?
- [ ] Do other-machine features add value after controlling for local history?
- [ ] Which libraries and saved-model formats are easiest to reproduce offline?
- [ ] Which improvements remain on an untouched episode-level assessment split?

## 10. Immediate next steps

1. Read the kit's rules, data description, metadata, and baseline implementation.
2. Run and preserve the official starter and its scores.
3. Implement or inspect smoothed local/category-prior estimates.
4. Compare recent weighting with long-memory estimates.
5. Build a feature-based model with lag and episode information.
6. Use the results to decide whether change-point, context, HMM, or neural modeling deserves deeper work.

