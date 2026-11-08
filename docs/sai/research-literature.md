# House Edge: research and background statistics

Written 2026-11-07 by Claude for Sai's TAMU Datathon team. Companion to `solution-playbook.md` (what to build and test) and `scripts/`.

Every item is tagged so you know how much to trust it:
- **[Lit]** published result; link in the reference list at the bottom.
- **[Math]** derivation I checked numerically (a short script, not a paper).
- **[Measured]** computed on this repo's main-round data (`mainFiles/public`) with the scripts in `scripts/`.
- **[Idea]** my inference or untested suggestion. Treat as a hypothesis to test, not a fact.

All results are for the **main round**. Round two changes which relationships recur, so re-measure every [Measured] item on the new kit.

---

## 1. What the metric rewards

**Metric.** Per query, the score is the squared Euclidean distance between your 5-vector and the simulator's true vector q. It is a Brier-type score measured against the true probabilities, not against sampled outcomes. The Brier score is a strictly proper scoring rule, meaning the expected score is optimized only by reporting the true distribution [Lit: Gneiting & Raftery 2007; Waghmare & Ziegel 2025].

**Link to training on labels [Math].** For any forecast p and true q, E over y~q of ||p - e_y||^2 = ||p - q||^2 + 1 - ||q||^2.
- The second term does not depend on p. So minimizing Brier loss on observed outcomes minimizes the contest metric in expectation.
- Log loss is also strictly proper and also consistent, but it weights tiny probabilities heavily. The contest barely cares about p_20 (about 0% of remaining error in the prototype, see playbook), so Brier or MSE-style objectives align better.
- [Measured] On validation, mean (1 - ||q||^2) = 0.378. That is the irreducible part of outcome-based Brier loss, so outcome-based scores look much worse than the contest metric. Do not compare the two.

**Reliability vs resolution [Lit: Murphy 1973; Siegert 2013].** The Brier score splits into reliability (miscalibration) minus resolution (how much forecasts vary in step with real changes in the event probability) plus uncertainty (the score of always forecasting the base rate). Squared distance to q rewards both: be calibrated, and vary the forecast only when the history truly supports it. Sharpening a forecast without real resolution hurts.

**Soft targets [Lit: Hinton et al. 2015].** Soft probability targets carry more information per example and give lower-variance gradients than hard labels.
- [Measured] `public/validation/targets.csv` has the true q for 768 queries. That is a direct soft-label signal.
- [Idea] Use it for a second stage (stacker, blend weights, recalibrator), with out-of-fold fitting by episode. 768 rows is too few to fit the main model on, and it is exactly the data you tune on, so keep an audit split (see section 7).

## 2. Estimating a probability vector from counts

**Shrinkage beats raw frequencies [Lit].** Efron and Morris showed that shrinking individual averages toward a pooled mean predicts better under squared error than the raw averages (the batting-average example); Brown (2008) field-tested empirical Bayes methods on the same problem. This is the partial-pooling idea behind the starter's `pooled` baseline.

**Dirichlet-multinomial form [Lit + Math].** With prior mean m (a 5-vector) and strength a, the posterior mean is (counts + a*m) / (n + a). The weight on the machine's own data is n/(n+a).
- The starter uses a = 20 [`baseline.py`]. With n_eff about 46 (half-life 16), data weight is 0.70.
- Fit m and a by maximum marginal likelihood with Minka's fixed-point or Newton iteration for the Dirichlet-multinomial [Lit: Minka 2000], or just grid-search a against validation squared error.

**Effective sample size of a recency-weighted count [Math, verified].** With weights 2^(-age/H) and lambda = 2^(-1/H): n_eff = (1+lambda)/(1-lambda).

| half-life H | n_eff | sampling-noise floor, sum p(1-p)/n_eff |
|---|---|---|
| 8 | 23 | 0.016 |
| 16 | 46 | 0.008 |
| 32 | 92 | 0.004 |
| 64 | 185 | 0.002 |

[Measured] The floor uses mean(1 - ||q||^2) = 0.378. The best plain EWMA scores 0.183 against the true vectors.

**Takeaway [Measured].** Sampling variance is not the bottleneck: about 0.175 of the 0.183 EWMA error is bias, meaning real structure in q that frequency counts cannot see. Shrinkage matters mainly for short-history queries and for per-machine coefficients (section 4). Memory tuning on its own cannot close the gap, which matches the flat plateau in `EXPERIMENT_LOG.md` entry 3.

## 3. Nonstationarity: forgetting, tracking and change detection

**Exponential smoothing is the optimal filter for a random-walk level plus noise [Lit: Muth 1960].**
- [Math, verified] Steady-state Kalman gain for the local-level model with signal-to-noise q is K = (-q + sqrt(q^2 + 4q)) / 2, and the EWMA keeps weight (1-K) per tick.
- Half-life in ticks: H = ln 2 / (-ln(1-K)). Examples: K = 0.05 gives H = 13.5; K = 0.10 gives 6.6; K = 0.20 gives 3.1.
- [Idea] Use this to sanity-check a chosen half-life: the memory you select implies a drift-to-noise ratio. If the implied drift looks implausibly large, the selection is chasing noise (see playbook: allowing half-lives of 4-12 hurt).

**Forgetting factors and dynamic model averaging [Lit: Raftery, Karny & Ettler 2010; McCormick et al. 2012].** Raftery et al. combine a state-space model for each candidate model's parameters with a Markov chain over which model is right, both written in terms of forgetting factors so no transition matrices need estimating. McCormick et al. extend it to binary outcomes (dynamic logistic regression plus model averaging) and adapt the forgetting level online from predictive performance.
- [Idea] Our per-machine half-life choice is a crude version. A DMA-style upgrade keeps several half-lives alive and weights them by recent predictive likelihood, so the weights move smoothly.

**Tracking the best expert [Lit: Herbster & Warmuth 1998].** When the best predictor changes over time, Fixed Share mixes in a small fraction alpha of uniform weight each round, w_next = (1-alpha) * w_updated + alpha / n. The switching rate is about (number of switches)/(T-1).
- [Idea] Use it over candidate predictors (memory lengths, lag sets, or whole models) instead of a hard holdout argmin.

**Bayesian online changepoint detection [Lit: Adams & MacKay 2007].** It maintains a posterior over the run length (time since the last change) and predicts by averaging the run-length-conditional predictives; conjugate-exponential models make it cheap.
- [Idea] "Shared change" episodes (guide: relationships change together across an episode) suggest pooling changepoint evidence across the 8 machines of an episode, then refitting each machine only on data after the detected change.

**Regime switching [Lit: Hamilton 1989].** Markov-switching models fit an unobserved state with its own parameters.
- [Idea, untested] My data work found drift and shared-change behave like changing coefficients that a short memory tracks well. I did not test hidden-state flips, so treat an HMM as a possible alternative, not a recommendation.

**HMM tutorial [Lit: Rabiner 1989, Proc. IEEE 77(2):257-286, doi:10.1109/5.18626].** The standard reference for hidden Markov models: forward-backward recursions, Viterbi decoding, Baum-Welch (EM) re-estimation, and the scaling needed to keep long sequences numerically stable. Speech is the application; nothing in it is specific to this contest.
- How it could map here [Idea]: each machine's win/payout process is emitted from one of K hidden regimes (K = 2-3). The next-tick forecast is sum over regimes of P(regime now | history) x P(payout | regime), where the regime posterior comes from the forward pass and then the transition matrix.
- Fit constraints from the contest: pure numpy or stdlib, fixed initialization and fixed iteration count so results reproduce to 1e-8 (Baum-Welch has local optima, so random restarts are out unless seeded), and cheap enough for 2 CPUs / 30 min.
- Risks: short histories (short_history machines) cannot support many parameters, so tie emissions across machines of a type or use Dirichlet priors; it does not by itself capture the lag-7 and pair structure in section 4, so it would need lag features inside the emission model (an input-output HMM), which is beyond the tutorial.
- Value: modest. It is a reference for regime-based memory, a competitor to the half-life search, and a candidate stack member. It is not a source of new signal.

## 4. Modeling lag structure

**Mixture transition distribution (MTD) [Lit: Raftery 1985; Raftery & Tavare 1994; Berchtold & Raftery 2002; Heiner & Kottas 2019].** A full order-l Markov chain over m states has m^l (m-1) parameters (for 5 payouts at lag 8 that is 1,562,500 [Math]). MTD makes each lag contribute additively through one transition matrix, so parameters grow linearly. Bayesian and sparse extensions exist for lag selection.
- [Idea] Because MTD lag contributions add, it cannot represent a pure pair interaction (an effect that exists only through the product of two lagged outcomes). The measured structure below is exactly that, so a plain MTD would miss it. It is still a reasonable lag-screening baseline.
- Variable-length and sparse Markov chains are the other parsimonious families named in those papers.

**What the data shows [Measured, `scripts/lag_screen.py`; train histories alone reproduce it].**
- Single lags: only **lag 7** stands out. Excess z^2 (sign-agnostic, mean of z^2 - 1 across machines) is about 13.8 steady, 13.5 balanced, 8.2 jackpot, against about 0.5-1.3 (steady/balanced) or about 1.4-3.5 (jackpot) at lags 9-16. Lag 14 (twice 7) is a mild echo.
- Pairs: **(1,3), (2,5), (4,7)** dominate for steady and balanced (excess z^2 about 36-60 versus null pairs at about 0.8, max 2.9). Weaker: (4,5), (2,6). Jackpot shows the same three at about 15-20.
- The effect is on the win/loss axis (payout >= 1). On the payout >= 2 axis only jackpot shows a notable pair signal.
- Signs vary across machines (for example steady (2,5): 26% of machines have z > 3 and 12% have z < -3). That is the "opposite directions" warning in the guide, so pooling a single coefficient across machines of a type cancels the effect. Per-machine fits are needed.
- The mean signed lag-7 coefficient is positive in every type (about +0.06 steady, +0.05 balanced, +0.03 jackpot) while individual machines vary, which suggests a type-level prior mean plus per-machine deviation (partial pooling) for coefficients.

**Why it matters for round two [Idea].** The guide says adaptation changes which temporal relationships recur. Do not hard-code lags 7 and (1,3),(2,5),(4,7); rerun the screen on each round's train histories and let the null-control lags (9-16) set the bar.

## 5. Combining forecasts

- **Linear pooling is miscalibrated and unsharp [Lit: Ranjan & Gneiting 2010; Gneiting & Ranjan 2013].** Any non-trivial weighted average of distinct calibrated forecasts is itself uncalibrated and lacks sharpness. Fixes: the beta-transformed linear pool and spread-adjusted variants.
- **Under this metric a convex blend never does worse than the weighted average of its members [Math, verified].** The squared distance to q is convex, so ||sum w_i p_i - q||^2 <= sum w_i ||p_i - q||^2. Blending is therefore a safe default; the calibration critique matters mainly when you then try to sharpen the blend.
- **Stacking [Lit: Yao et al. 2018].** Learn blend weights by optimizing a proper score on held-out predictions; the paper generalizes stacking of predictive distributions to any proper scoring rule and argues for it over model-averaging weights when the truth is not among the candidates (the M-open case).
- **Recalibration [Lit: Guo et al. 2017; Kull et al. 2019].** Temperature scaling is one parameter on the logits; Dirichlet calibration (log-transform, one linear layer, softmax) is the multiclass generalization and exposes the model's biases.
  - [Idea] Fit either one on validation by minimizing squared error to **q** (which you have), not NLL on outcomes.
  - Fit with episodes held out. Per-condition parameters cannot be applied at test time because condition labels are withheld.

## 6. Model families for small tabular data

- **Trees remain strong on medium-sized tabular data [Lit: Grinsztajn et al. 2022].** Their benchmark finds tree ensembles still ahead of deep nets on roughly 10K-sample tabular tasks. Your existing HistGradientBoosting prototype (32.99 validation points) fits that picture.
- [Measured, see playbook] A hand-built per-machine pair model reaches 46.07 points, far above generic-feature GB. The pair model's ability comes from structure a generic feature set did not expose, not from model class.
- [Idea] Use GB as a **second-stage learner**: feed it the pair-model prediction, its chosen half-life, EWMA features and episode context, and let it learn residual patterns (payout sizes, extremes).
- [Idea] Neural nets or large pretrained tabular models look a poor fit here: the entry must reproduce to 1e-8 on 2 CPU cores in 30 minutes with no network and with dependencies you must package. Not tested.

## 7. Validation hygiene

- **Block cross-validation wherever dependence exists [Lit: Roberts et al. 2017].** They recommend blocked splits even when no dependence shows up in residuals. Here: keep whole episodes together (the guide says the same), and note the cutoff-tick outcomes of other machines are unavailable inputs.
- **Small validation set [Measured].** 96 episodes (24 per condition). `scoring.py` already has an episode-level bootstrap (`bootstrap_mean_interval`); report intervals and treat differences under about 0.2 points as noise (your own log says the same).
- **Overfitting by repeated tuning.** The guide warns that the held-out half stops being independent once you tune on it. Choose structure (lags, pairs) from histories only, and pick hyperparameters by an inner holdout inside each machine's own history where possible (the prototype does this for memory length).
- Choices in the prototype that touched validation scores: the half-life grid floor (16) and ridge strength. Re-pick them on the round-two validation split.

## 8. Formula cheat-sheet

| Quantity | Formula | Status |
|---|---|---|
| Outcome Brier vs contest metric | E||p - e_y||^2 = ||p - q||^2 + 1 - ||q||^2 | [Math] verified |
| Effective sample size, half-life H | n_eff = (1+lambda)/(1-lambda), lambda = 2^(-1/H) | [Math] verified |
| Sampling-noise floor | sum_k q_k(1-q_k) / n_eff | [Math] |
| Dirichlet posterior mean | (counts + a*m)/(n + a); data weight n/(n+a) | [Lit] |
| Local-level steady gain | K = (-q + sqrt(q^2+4q))/2; half-life = ln2 / -ln(1-K) | [Lit]+[Math] verified |
| Fixed Share | w_next = (1-alpha) w_upd + alpha/n | [Lit] |
| Blend never worse than average | ||sum w_i p_i - q||^2 <= sum w_i ||p_i - q||^2 | [Math] verified |
| Full order-8 chain size over 5 payouts | 5^8 * 4 = 1,562,500 parameters | [Math] |

## 9. What the research does NOT cover

- Outside data: the participant guide says to use organizer-released data only. Papers and methods are fine; external datasets are not.
- I found no published model of this exact simulator. Sections 1-3 and 5-7 are standard theory; section 4's lag findings are from your data.
- Slot-machine payout literature (real casino machines) is not relevant: the guide states these are simulator patterns, not claims about real machines.

## References

- Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. JASA. doi:10.1198/016214506000001437
- Waghmare, K. & Ziegel, J. (2025). Proper scoring rules for estimation and forecast evaluation. https://arxiv.org/abs/2504.01781
- Murphy, A. H. (1973). A new vector partition of the probability score. J. Applied Meteorology 12(4). doi:10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2
- Siegert, S. (2013). Variance estimation for Brier score decomposition. https://arxiv.org/abs/1303.6182
- Hinton, G., Vinyals, O. & Dean, J. (2015). Distilling the knowledge in a neural network. https://arxiv.org/abs/1503.02531
- Efron, B. & Morris, C. (1975, 1977), Stein's paradox in statistics; Brown, L. D. (2008). In-season prediction of batting averages. Ann. Appl. Stat. https://arxiv.org/abs/0803.3697
- Minka, T. P. (2000, rev. 2012). Estimating a Dirichlet distribution. https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf
- Muth, J. F. (1960). Optimal properties of exponentially weighted forecasts. JASA 55(290):299-306. Gain formula as stated in https://arxiv.org/abs/2512.16745
- Raftery, A. E., Karny, M. & Ettler, P. (2010). Online prediction under model uncertainty via dynamic model averaging. Technometrics 52(1). doi:10.1198/TECH.2009.08104
- McCormick, T. H., Raftery, A. E., Madigan, D. & Burd, R. S. (2012). Dynamic logistic regression and dynamic model averaging for binary classification. https://sites.stat.washington.edu/raftery/Research/PDF/McCormick2012.pdf
- Herbster, M. & Warmuth, M. (1998). Tracking the best expert. Machine Learning. Overview and fixed-share form: https://arxiv.org/abs/0802.2015
- Adams, R. P. & MacKay, D. J. C. (2007). Bayesian online changepoint detection. https://arxiv.org/abs/0710.3742
- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. Proc. IEEE 77(2):257-286. https://ieeexplore.ieee.org/document/18626
- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica 57:357-384. See also Hamilton (2005), Regime-switching models.
- Raftery, A. E. (1985); Raftery, A. E. & Tavare, S. (1994); Berchtold, A. & Raftery, A. E. (2002). The mixture transition distribution model for high-order Markov chains and non-Gaussian time series. Statistical Science 17(3). https://sites.stat.washington.edu/raftery/Research/PDF/berchtold2002.pdf
- Heiner, M. & Kottas, A. (2019). Estimation and selection for high-order Markov chains with Bayesian mixture transition distribution models. https://arxiv.org/abs/1906.10781
- Ranjan, R. & Gneiting, T. (2010). Combining probability forecasts. JRSS-B 72(1). doi:10.1111/j.1467-9868.2009.00726.x
- Gneiting, T. & Ranjan, R. (2013). Combining predictive distributions. EJS 7. doi:10.1214/13-EJS823
- Yao, Y., Vehtari, A., Simpson, D. & Gelman, A. (2018). Using stacking to average Bayesian predictive distributions. Bayesian Analysis. https://arxiv.org/abs/1704.02030
- Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). On calibration of modern neural networks. ICML. https://arxiv.org/abs/1706.04599
- Kull, M. et al. (2019). Beyond temperature scaling: well-calibrated multiclass probabilities with Dirichlet calibration. NeurIPS. https://arxiv.org/abs/1910.12656
- Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? NeurIPS. https://arxiv.org/abs/2207.08815
- Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography 40(8). doi:10.1111/ecog.02881
