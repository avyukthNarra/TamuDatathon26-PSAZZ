# House Edge: 20 Additional Research Papers and Experiments

**Prepared:** November 7, 2026  
**Status:** Research notes and proposed experiments; no House Edge performance claims have been established.

This document supplements the original House Edge brainstorming notes with 20 additional papers. Each entry explains the research idea, a possible application, implementation effort, and limitations.

The applications are proposed adaptations, not results demonstrated by the papers on this competition. Effort estimates are judgments, not measured runtimes. Numerical examples and suggested settings are illustrative.

[Official participant guide](https://docs.google.com/document/d/1CqNLjCiojRxKkZopoxBeilkke6Y4yCuz7FYSuefvCMo/edit?tab=t.0)

## Challenge context

Predict a probability vector over payouts **0x, 1x, 2x, 5x, and 20x** for each queried machine at its cutoff.

Key difficulties:

- Short histories provide noisy machine-specific evidence.
- Relevant patterns may involve exact tick delays of up to eight.
- Underlying behavior can change over time.
- Related machines within an episode can experience shared changes.
- Machines of the same type can respond differently.
- Missing ticks are unknown observations, not zero payouts.

Select models with the official probability-error score and episode/condition aggregation. A correct top-category guess is not the same as a good probability forecast.

## Research map

| Papers | Direction | Main question |
| --- | --- | --- |
| 1 | Hierarchical categorical estimation | How much evidence should related machines share? |
| 2–4 | Lag and context models | Which earlier payouts matter, and how much memory is needed? |
| 5–8 | Changing distributions | How do we distinguish gradual drift, abrupt changes, and shared changes? |
| 9–11 | Hidden-state models | Can unobserved modes explain evolving machine behavior? |
| 12–14 | Learned adaptation and missingness | Can a model adapt from machine histories and episode context? |
| 15–16 | Practical supervised baselines | How well do trees and tabular neural models combine historical features? |
| 17–20 | Calibration and ensembles | Can we improve probabilities by correcting or combining predictors? |

## 1. Hierarchical Multinomial–Dirichlet Model for the Estimation of Conditional Probability Tables

**Authors:** Laura Azzimonti, Giorgio Corani, and Marco Zaffalon  
**Year:** 2017  
**Paper:** https://arxiv.org/abs/1708.06935  
**Effort:** Moderate  
**Priority:** High

### Research idea

Estimate related categorical conditional distributions jointly, allowing them to borrow statistical strength rather than fitting each independently.

This extends the earlier smoothing discussion: the shared distribution can itself be learned through a hierarchical model.

### House Edge experiment

Compare fixed-prior smoothing with a learned hierarchical prior for machine payout distributions. Consider learning separate shared structures for machine types while retaining individual differences.

A machine with little relevant history should rely more on the shared structure; strong local evidence should allow it to depart from that structure.

### Limitations

The paper concerns conditional probability tables, not this simulator. The particular hierarchy across types, machines, and contexts is our design choice. Pooling must not erase opposing machine-specific responses.

## 2. The Mixture Transition Distribution Model for High-Order Markov Chains and Non-Gaussian Time Series

**Authors:** André Berchtold and Adrian E. Raftery  
**Year:** 2002  
**Paper:** https://sites.stat.washington.edu/people/raftery/Research/PDF/berchtold2002.pdf  
**Effort:** Moderate  
**Priority:** High

### Research idea

A Mixture Transition Distribution model combines contributions from individual historical lags instead of estimating a separate distribution for every complete lag sequence. The paper reviews the original model and extensions, including different transition matrices for different lags.

### House Edge experiment

Build a smoothed categorical model for exact delays one through eight. A multimatrix form can be written as:

$$
P(Y_t=k\mid H_t)=\sum_{\ell=1}^{8}w_\ell Q_\ell(Y_{t-\ell},k),
$$

where the lag weights are nonnegative and sum to one.

Learn common structure across training sequences and investigate machine-specific adaptation from their histories.

### Limitations

Basic lag mixtures can miss effects requiring particular combinations of two or more earlier payouts. Missing lag values need explicit handling; row offsets are not tick lags.

## 3. Bayesian Context Trees: Modelling and Exact Inference for Discrete Time Series

**Authors:** Ioannis Kontoyiannis, Lambros Mertzanis, Athina Panotopoulou, Ioannis Papageorgiou, and Maria Skoularidou  
**Year:** 2022  
**Paper and published version:** https://www.repository.cam.ac.uk/items/b28c8d82-2c9c-42c9-8ad1-5aaf667e7cfd  
**Effort:** Moderate  
**Priority:** High

### Research idea

Infer variable-memory models for discrete sequences using Bayesian context trees. Different recent patterns can require different memory lengths. The framework supports general finite alphabets rather than only binary observations.

### House Edge experiment

Treat the five payout categories as the alphabet. Compare context depths and posterior predictive forecasts against frequency-only and individual-lag models.

Use contiguous historical segments or a principled missing-data treatment. The paper supplies algorithms through the R package BCT.

### Limitations

A short machine history may support only shallow contexts. More complex pooling, drift adaptation, and missing-tick handling are additional modeling choices.

## 4. Context Tree Switching

**Authors:** Joel Veness, Kee Siong Ng, Marcus Hutter, and Michael Bowling  
**Year:** 2012 conference paper; 2011 preprint  
**Paper:** https://arxiv.org/abs/1111.3182  
**Effort:** Moderate to advanced

### Research idea

Modify Context-Tree Weighting to mix over a larger class of context models without increasing its asymptotic time or space complexity.

### House Edge experiment

Compare fixed reliance on short and long contexts with a switching mixture of context predictors. Investigate whether changing the reliance on different pattern lengths helps forecasting.

### Limitations

The original work concerns binary stationary sources. A categorical implementation requires adaptation. Switching context strategies should not be presented as an established solution to arbitrary distribution drift.

## 5. Dynamic Generalized Linear Models and Bayesian Forecasting

**Authors:** Mike West, P. Jeff Harrison, and Helio S. Migon  
**Year:** 1985  
**Paper:** https://www2.stat.duke.edu/~mw/MWextrapubs/West1985a.pdf  
**Effort:** Advanced

### Research idea

Extend generalized linear models by allowing underlying parameters to evolve over time and updating their uncertainty as observations arrive.

### House Edge experiment

Build a dynamic categorical model with evolving logits. Compare it with constant probabilities, exponentially weighted counts, and abrupt-change models.

An extension could include lag features or machine-specific coefficients, allowing their influence to evolve.

### Limitations

The categorical model and inference procedure require implementation decisions. A smooth evolution assumption may react differently from an abrupt-change model. Validate whether that assumption matches the available evidence.

## 6. Learning from Time-Changing Data with Adaptive Windowing

**Authors:** Albert Bifet and Ricard Gavaldà  
**Year:** 2007 conference paper; linked manuscript dated 2006  
**Paper:** https://www.cs.upc.edu/~gavalda/papers/adwin06.pdf  
**Effort:** Practical to moderate

### Research idea

ADWIN adjusts a history window based on evidence of change. The window grows in apparently stable periods and shrinks when older observations appear inconsistent with newer observations.

### House Edge experiment

Monitor per-payout indicator streams and use change evidence to adjust a smoothed probability estimator or construct features for another model.

Compare against fixed windows and exponential decay.

### Limitations

ADWIN is not a complete five-category forecaster. Short or dependent histories can limit detector reliability. Average payout alone can conceal changes because different categorical distributions can have the same mean.

**Implementation resource:** https://riverml.xyz/dev/api/drift/ADWIN/

## 7. Bayesian On-line Learning of the Hazard Rate in Change-Point Problems

**Authors:** Robert C. Wilson, Matthew R. Nassar, and Joshua I. Gold  
**Year:** 2010  
**Paper:** https://pmc.ncbi.nlm.nih.gov/articles/PMC2966286/  
**Effort:** Advanced

### Research idea

A change-point detector often assumes a fixed rate of change, called the hazard rate. This paper learns that rate and allows it to change.

### House Edge experiment

Compare a fixed change-frequency assumption with a learned or hierarchically pooled assumption. Training episodes could inform a prior for adaptation to new machines.

### Limitations

A single short history may provide little information about change frequency. Additional hierarchy and flexibility introduce uncertainty and inference cost. Start after establishing a basic change-point baseline.

## 8. The Group Fused Lasso for Multiple Change-Point Detection

**Authors:** Kevin Bleakley and Jean-Philippe Vert  
**Year:** 2011 preprint  
**Paper:** https://arxiv.org/abs/1106.4199  
**Effort:** Moderate  
**Priority:** High for shared change

### Research idea

Find change times shared across multiple co-occurring signals using a penalty on multidimensional total variation. Signals can change by different amounts while sharing a boundary.

### House Edge experiment

Represent payouts with category-indicator channels, align them by actual tick, and search for common change times across machines in the same episode.

Use candidate boundaries to construct current-regime probability estimates or episode features.

### Limitations

Missing observations require a mask-aware adaptation. Filling missing channels with zero would falsely assert a known observation. The resulting change boundaries are uncertain and need to improve forecasts, not merely look plausible.

## 9. An HDP-HMM for Systems with State Persistence

**Authors:** Emily B. Fox, Erik B. Sudderth, Michael I. Jordan, and Alan S. Willsky  
**Year:** 2008  
**Paper:** https://people.eecs.berkeley.edu/~jordan/papers/fox-etal-icml08.pdf  
**Effort:** Moderate for a simplified experiment; advanced for the full model

### Research idea

A sticky hierarchical Dirichlet process HMM encourages persistent hidden states and accommodates uncertainty about the number of states. This addresses unrealistic rapid switching among redundant states.

### House Edge experiment

First compare a small categorical HMM with and without a prior favoring self-transitions. Consider nonparametric state-count inference only if the simpler experiment suggests it is useful.

### Limitations

The simplified experiment is inspired by the paper, not a full implementation. Very short histories may not support many regimes. Hidden-state persistence is an assumption that must be evaluated.

## 10. Factorial Hidden Markov Models

**Authors:** Zoubin Ghahramani and Michael I. Jordan  
**Year:** 1997  
**Paper:** https://mlg.eng.cam.ac.uk/pub/pdf/GhaJor97a.pdf  
**Effort:** Advanced

### Research idea

Represent hidden state using multiple factors rather than one discrete variable. Exact inference becomes expensive; the paper develops approximate approaches.

### House Edge experiment

Explore a model with an episode-wide hidden regime and separate machine-level hidden components. Their combined effects would determine each machine's categorical payout probabilities.

### Limitations

The proposed factor structure and categorical observation model are our adaptations. Approximate inference and parameter learning may be substantial work. Do not assume shared and local causes can be uniquely identified from short logs.

## 11. An Input Output HMM Architecture

**Authors:** Yoshua Bengio and Paolo Frasconi  
**Year:** 1994  
**Paper:** https://papers.nips.cc/paper/1994/hash/8065d07da4a77621450aa84fee5656d9-Abstract.html  
**Effort:** Advanced

### Research idea

Allow hidden-state transitions and output distributions to depend on observed inputs. The model supports categorical outputs.

### House Edge experiment

Compare an ordinary categorical HMM with an input-dependent model using available context and lag features. Recent payouts could help predict both a hidden mode and the next outcome distribution.

### Limitations

Only use context known at the relevant time. In particular, do not backfill a query's current signal into historical training examples without evidence that it was available then. Keep model complexity appropriate to the amount of data.

## 12. Deep Sets

**Authors:** Manzil Zaheer et al.  
**Year:** 2017  
**Paper:** https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html  
**Effort:** Moderate to advanced  
**Priority:** High for a neural episode model

### Research idea

Model sets so that permuting their elements does not alter the output.

### House Edge experiment

Encode each machine's history, aggregate representations of the other machines in its episode, and combine that episode summary with the target machine's representation.

Compare learned aggregation with simple means and other handcrafted episode summaries.

### Limitations

Preserve temporal order inside each machine's history. Permutation invariance applies across the collection of machines, not across payout ticks. A target-specific prediction must retain which representation belongs to the target.

**Author implementation:** https://github.com/manzilzaheer/DeepSets

## 13. Conditional Neural Processes

**Authors:** Marta Garnelo et al.  
**Year:** 2018  
**Paper:** https://proceedings.mlr.press/v80/garnelo18a.html  
**Effort:** Advanced

### Research idea

Share learning across tasks while conditioning each prediction on a supplied set of observed examples.

### House Edge experiment

Treat a machine as a task. Construct historical feature–payout pairs as context and predict the next payout with a five-category decoder. Train across machines so that the model learns how to use a new machine's limited context.

### Limitations

An unordered context representation does not inherently preserve sequence relationships. Include timestamps, lag features, or a suitable history encoder. Construct all context examples without future information.

**Research implementation resource:** https://github.com/google-deepmind/neural-processes

## 14. Recurrent Neural Networks for Multivariate Time Series with Missing Values

**Authors:** Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag, and Yan Liu  
**Year:** 2018 journal paper; 2016 preprint  
**Paper:** https://arxiv.org/abs/1606.01865  
**Effort:** Moderate

### Research idea

GRU-D incorporates observation masks, elapsed time, and learned decay into recurrent modeling.

### House Edge experiment

Compare a standard categorical recurrent predictor with one receiving explicit missingness masks and tick gaps.

This directly addresses sampled older histories, where consecutive rows need not represent consecutive ticks.

### Limitations

The original model primarily concerns numerical measurements. Adapt the masking and time-gap ideas to categorical embeddings rather than interpolating payout amounts or assuming missing payouts approach an average value.

## 15. XGBoost: A Scalable Tree Boosting System

**Authors:** Tianqi Chen and Carlos Guestrin  
**Year:** 2016  
**Paper:** https://arxiv.org/abs/1603.02754  
**Effort:** Practical

### Research idea

An efficient tree-boosting system, including techniques for sparse data and approximate tree learning.

### House Edge experiment

Compare XGBoost with CatBoost using identical frequency, exact-lag, missingness, and episode features. Request five-class probability predictions and select settings with the official scorer.

### Limitations

This is a comparison within the feature-based approach, not a fundamentally different source of information. Hold feature construction and evaluation episodes fixed to distinguish model improvements from feature improvements.

## 16. Revisiting Deep Learning Models for Tabular Data

**Authors:** Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko  
**Year:** 2021  
**Paper:** https://papers.neurips.cc/paper_files/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html  
**Effort:** Moderate for a residual network; higher for a Transformer

### Research idea

The paper identifies strong tabular neural baselines, including a ResNet-like model and a feature-tokenizing Transformer, and compares them under consistent protocols.

### House Edge experiment

Use the same historical-summary features as the tree baselines in a small residual network with five probability outputs. This changes the learner while holding the representation constant.

### Limitations

The paper does not establish a universally superior model. Any additional complexity should be justified by held-out probability quality and reproducible inference cost.

**Author implementation:** https://github.com/yandex-research/rtdl

## 17. Beyond Temperature Scaling: Obtaining Well-Calibrated Multiclass Probabilities with Dirichlet Calibration

**Authors:** Meelis Kull et al.  
**Year:** 2019  
**Paper:** https://proceedings.neurips.cc/paper_files/paper/2019/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html  
**Effort:** Practical to moderate after fitting a base model

### Research idea

Learn a multiclass probability correction using log probabilities, a linear transformation, and softmax. It can correct more structured category-specific biases than one global temperature.

### House Edge experiment

Compare unadjusted predictions, temperature scaling, and regularized Dirichlet calibration.

### Limitations

Extra flexibility can overfit. Choose the adjustment using tuning episodes and assess it on separate episodes with the official score. Better calibration diagnostics alone do not guarantee better contest points.

**Authors' project:** https://dirichletcal.github.io/

## 18. Super Learner

**Authors:** Mark J. van der Laan, Eric C. Polley, and Alan E. Hubbard  
**Year:** 2007  
**Paper:** https://biostats.bepress.com/ucbbiostat/paper222/  
**Effort:** Practical to moderate  
**Priority:** High after complementary models exist

### Research idea

Use cross-validation to select or combine candidate predictors according to a loss function.

### House Edge experiment

Generate episode-grouped out-of-fold forecasts from long-memory smoothing, recent smoothing, a lag model, and a tree model. Fit a simple nonnegative blend whose weights sum to one.

For model probability vectors p^(m), a convex blend is:

$$
p_{\mathrm{blend}}=\sum_m w_m p^{(m)}, \qquad w_m\ge 0,\quad \sum_m w_m=1.
$$

### Limitations

Asymptotic theory is not a guarantee of improvement on this finite dataset. Avoid fitting and assessing blend weights on the same episodes. Check both the official total and condition errors.

## 19. Tracking the Best Expert

**Authors:** Mark Herbster and Manfred K. Warmuth  
**Year:** 1998  
**Paper:** https://doi.org/10.1023/A:1007424614876  
**Effort:** Moderate

### Research idea

Combine experts when the best one can change across parts of a sequence. Weight sharing allows an expert that performed poorly earlier to regain influence later.

### House Edge experiment

Treat slow-memory, fast-memory, and lag-based predictors as experts. Update their weights using prequential losses on earlier observed payouts, allowing their influence to change during a machine's history.

### Limitations

Each historical forecast must be generated before seeing its outcome. Do not evaluate an expert on an outcome it already used to fit that forecast. Online-loss guarantees do not automatically imply better error against the simulator's hidden probabilities.

## 20. Adaptive Mixtures of Local Experts

**Authors:** Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, and Geoffrey E. Hinton  
**Year:** 1991  
**Paper:** https://www.cs.utoronto.ca/~hinton/absps/jjnh91.html  
**Effort:** Moderate

### Research idea

Train specialized predictors with a mechanism that assigns cases to appropriate experts.

### House Edge experiment

Use a small gating model to choose probability-blend weights from observable features:

- Number and recency of observations.
- Differences between recent and older distributions.
- Evidence of predictive lag patterns.
- Evidence of changes across other machines.

For example, short histories could favor a prior-based expert while strong change evidence could favor a recent-history expert.

### Limitations

Train the gate on held-out or out-of-fold expert predictions. It cannot depend on unavailable test condition labels. More flexible gating needs enough independent episodes to support it.

## Recommended implementation priorities

These are proposed priorities, not an empirical ranking.

| Priority | Paper | First experiment | Main question |
| --- | --- | --- | --- |
| 1 | 1: Hierarchical categorical estimation | Compare fixed and learned priors | Does adaptive information sharing improve short histories? |
| 2 | 2: Lag-mixture models | Fit exact-lag categorical transitions | Do individual delays add information beyond frequencies? |
| 3 | 3: Bayesian context trees | Compare shallow variable-memory models | Do combinations of recent payouts add value? |
| 4 | 8: Group fused lasso | Construct shared change features | Do related machines reveal useful change times? |
| 5 | 18: Super Learner | Blend complementary probability forecasts | Are different models' errors complementary? |

For a neural research direction, investigate **Deep Sets plus Conditional Neural Processes**: summarize the episode while adapting the target forecast to its own observed history. This is a proposed combination, not an established result of either paper.

## Candidate solution families

### A. Compact statistical baseline

Combine hierarchical categorical smoothing, a lag-mixture model, recent weighting, and a small convex blend.

Relevant papers: **1, 2, 6, 18**.

### B. Shared-change forecasting

Use other-machine category channels to infer common change times, then estimate target-machine probabilities under a recent or dynamic model.

Relevant papers: **5, 8, 10, 12**.

### C. Learned machine adaptation

Learn across training machines while conditioning on a new machine's history and an episode summary.

Relevant papers: **11, 12, 13, 14**.

### D. Observable-feature gating

Build several probability experts and learn when to trust each using available historical evidence.

Relevant papers: **18, 19, 20**.

These families can overlap. Implement simple comparisons and ablations before combining many components.

## Evaluation and implementation requirements

- Use inputs strictly before each query's cutoff, including other-machine inputs.
- Restrict other-machine information to the same episode.
- Group histories by both episode_id and machine_id.
- Preserve actual tick gaps and represent missing observations explicitly.
- Keep entire episodes together in internal evaluation splits.
- Do not use test condition labels or unavailable historical signal values.
- Use observed training payouts as labels; true training probability vectors are not provided.
- Use the official validation scorer and its episode/condition aggregation.
- Reserve independent assessment episodes when choosing model, calibration, or ensemble settings.
- Keep output probabilities in payout order 0, 1, 2, 5, 20.
- Record versions, seeds, settings, runtime, and model files.
- Verify the exact CSV and reproduction package before submission.

## Experiment log

All entries begin as planned. Fill in measurements only after running the corresponding experiments.

| Run | Papers / approach | Hypothesis | Episode split | Official points | Condition errors | Runtime | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pending | Hierarchical smoothing | Learned pooling helps short histories | Pending | — | — | — | Pending |
| Pending | Lag mixture | Exact delays improve on frequency estimates | Pending | — | — | — | Pending |
| Pending | Context tree | Pattern combinations add information | Pending | — | — | — | Pending |
| Pending | Shared change | Other machines improve change inference | Pending | — | — | — | Pending |
| Pending | Probability blend | Complementary errors improve combined forecasts | Pending | — | — | — | Pending |

