# House Edge (TAMU Datathon 2026) — Statistical Research & Modeling Guide

## Part 1: Formal Problem Structure & Loss Function Anatomy

Your task is to predict the 5-class probability vector $\hat{\mathbf{p}}_q = (\hat{p}_0, \hat{p}_1, \hat{p}_2, \hat{p}_5, \hat{p}_{20}) \in \Delta^4$ for a target machine's spin at its `cutoff` tick $T \in [192, 256]$, using only ticks $t < T$ from machines within the same `episode_id`.

* **Evaluation Metric (Multi-Class Brier Score):** For each query $q$, the loss is the sum of squared errors against the simulator's true hidden probability vector $\mathbf{p}_q$:
  $$L(\hat{\mathbf{p}}_q, \mathbf{p}_q) = \sum_{k \in \{0,1,2,5,20\}} (\hat{p}_{q,k} - p_{q,k})^2$$
* **Balanced Regime Weighting:** Errors are averaged within each episode, then across episodes within four hidden regimes—**`stable`**, **`short history`**, **`drift`**, and **`shared change`**—and divided by the frozen `pooled` baseline error per condition to compute the average ratio $R$, where:
  $$\text{Points} = \frac{\text{MaxPoints}}{1 + R}$$
* **Why Proper Scoring Rule Theory Matters:** Because the multi-class Brier score is a strictly proper scoring rule that decomposes into **calibration** (reliability) and **resolution** (sharpness), tree models or neural networks trained purely on observed single-outcome cross-entropy (`labels.csv`) will suffer from high variance on rare classes ($5\times, 20\times$). However, in `public/validation/`, you are given the **true target probabilities**—meaning you can directly fit or calibrate a meta-model against the exact Brier loss on validation data and use self-supervised historical windows from `history.csv` for pre-training.

---

## Part 2: Published Statistical Frameworks & Research Literature

Each of the four evaluation conditions in the simulator corresponds to a classic problem in statistical time-series and panel econometrics:

| Challenge Regime / Dynamic | Core Statistical Literature & Models | How It Solves the Simulator Dynamic |
| :--- | :--- | :--- |
| **Short History & Noise** (`short history`: 40 recent consecutive + 24 sampled older ticks) | **Empirical Bayes & Dirichlet-Multinomial Shrinkage** (Efron & Morris, 1973; Good-Turing / Dirichlet smoothing) | Shrinks noisy individual machine frequency estimates toward a hierarchical prior stratified by `machine_type` (`steady`, `balanced`, `jackpot`), `signal` ($-1, 0, 1$), and episode mean. |
| **Lag Interactions (Up to 8 Ticks) & Opposite Machine Responses** | **Multinomial Autoregressive Models & Mixture Transition Distribution (MTD)** (Raftery, 1985; Fokianos & Kedem, 2003 *Regression Theory for Categorical Time Series*) | Models high-order Markov chains ($p \le 8$) without the $5^8$ parameter explosion by expressing $\mathbf{p}_t$ as a linear or logit combination of individual lag transition matrices $\sum_{l=1}^8 w_l Q_l \mathbf{y}_{t-l}$. |
| **Non-Stationary Payouts (`drift`)** | **Bayesian Online Changepoint Detection (BOCPD)** (Adams & MacKay, 2007) with Dirichlet-Multinomial conjugate priors | Maintains a posterior distribution over the "run length" $r_t$ (ticks since the last regime change) and dynamically resets or shortens the memory window when a shift is detected. |
| **Simultaneous Multi-Machine Shifts (`shared change`)** | **Dynamic Panel Models & Factor-Augmented Categorical Time Series** (Juárez & Steel, 2010; Switching State-Space Models) | Extracts a common latent "episode state" by pooling synchronous tick residuals across all machines in the same `episode_id` to detect synchronized probability shifts. |
| **Ensembling Across Regimes** | **Stacked Regression & Probability Calibration** (Breiman, 1996; Multicalibration / Brier Score Optimization) | Combines long-window, short-window, episode-pooled, and autoregressive predictions using a latent condition classifier as a gating network. |

---

## Part 3: Background Statistics & Feature Engineering Blueprint

To capture the simulator rules, compute the following groups of background statistics strictly using ticks $t < \text{cutoff}$:

### A. Multi-Horizon Exponentially Decaying Dirichlet-Multinomial Statistics
Because missing ticks in `short history` are **unknowns rather than zero payouts**, always compute decay using **actual tick distance** $\Delta t = \text{cutoff} - t$, not row index:
1. **Multi-Half-Life Smoothed Frequencies:** For half-lives $h \in \{8, 16, 32, 64, 128, \infty\}$, define exponential weights $w_t(h) = 2^{-(\text{cutoff} - t)/h}$. Compute the smoothed class probability vector for machine $m$ with Dirichlet prior concentration $\alpha_0$ and prior mean $\boldsymbol{\pi}_{\text{type, signal}}$:
   $$\hat{p}_{m,k}(h) = \frac{\sum_{t < \text{cutoff}} w_t(h) \mathbb{I}(y_{m,t} = k) + \alpha_0 \pi_{\text{type, signal}, k}}{\sum_{t < \text{cutoff}} w_t(h) + \alpha_0}$$
2. **Effective Sample Size (ESS) & Missingness Statistics:**
   * Total observed ticks $N_m$, fraction of ticks observed in $[T-40, T-1]$ vs. $[1, T-41]$ (immediately separates `short history` episodes from full-history episodes).
   * Kish's Effective Sample Size:
     $$\text{ESS}_m(h) = \frac{(\sum w_t)^2}{\sum w_t^2}$$

### B. Exact Lag & Autoregressive Statistics (Delays $l \in \{1, \dots, 8\}$)
Past payouts can interact at delays of up to eight ticks, and machines of the same type can respond differently, even in opposite directions:
1. **Exact Tick-Matched Lag Indicators:** Look up exact ticks $t = \text{cutoff} - l$ for $l \in \{1, 2, \dots, 8\}$. Encode one-hot payout indicators, raw payout values, and binary indicator `is_win` ($y > 0$), plus an explicit `is_missing` flag if tick $\text{cutoff} - l$ is absent.
2. **Machine-Specific Lagged Empirical Autocorrelation (Capturing Opposite Directions):** Because two machines of the same `machine_type` can react in opposite directions to past wins, global lag weights alone will cancel out. For each machine $m$ and lag $l \in \{1, \dots, 8\}$, compute its **own historical conditional lift** from its history $t < \text{cutoff}$:
   $$\text{Lift}_{m}(l) = \hat{P}_m(\text{win}_t \mid \text{win}_{t-l}) - \hat{P}_m(\text{win}_t \mid \text{loss}_{t-l})$$
   Then interact $\text{Lift}_{m}(l)$ with whether a win occurred at tick $\text{cutoff} - l$. This allows a global model (like LightGBM, CatBoost, or an MLP) to apply positive feedback to momentum machines and negative feedback (cooldown) to mean-reverting machines.
3. **Rolling Window Summary Moments:** Over exact tick windows $W \in \{4, 8, 16, 40\}$, compute:
   * Mean return $\frac{1}{|W|}\sum y_t$, log-payout mean $\frac{1}{|W|}\sum \log(1 + y_t)$, and payout variance.
   * Time since last nonzero payout ($y > 0$) and time since last high payout ($y \in \{5, 20\}$).

### C. Drift & Changepoint Statistics (`drift` Condition)
To identify whether a machine's odds have shifted prior to `cutoff`:
1. **Short-vs-Long Divergence Statistics:** Compute the **Jensen-Shannon Divergence (JSD)**, **Total Variation Distance**, and $L_2$ norm between the recent window distribution $\hat{\mathbf{p}}_m(h=16)$ and the older historical distribution $\hat{\mathbf{p}}_m(\text{ticks} < \text{cutoff}-40)$:
   $$D_{\text{TV}}(\hat{\mathbf{p}}_{\text{recent}}, \hat{\mathbf{p}}_{\text{old}}) = \frac{1}{2} \sum_{k} |\hat{p}_{\text{recent}, k} - \hat{p}_{\text{old}, k}|$$
2. **CUSUM & Run-Length Posterior:** Track the cumulative sum of standardized payout residuals $S_t = \sum_{\tau \le t} (y_{m,\tau} - \bar{y}_m)$ and its maximum range $\max S_t - \min S_t$ (rescaled range statistic), or run a fast Dirichlet-Multinomial BOCPD filter to output the expected run length $\mathbb{E}[r_{\text{cutoff}}]$.

### D. Cross-Sectional Episode Pooling (`shared change` Condition)
Because episodes group related machines observed at the same cutoff, and win chances and payout sizes can change across several machines together:
1. **Episode-Level Synchronized Drift:** Average the short-vs-long divergence $\bar{D}_{\text{TV, episode}}$ and recent win-rate change $\Delta \bar{p}_{\text{episode}} = \bar{p}_{\text{episode}}(h=16) - \bar{p}_{\text{episode}}(h=128)$ across all machines in `episode_id` (using only $t < \text{cutoff}$).
2. **Episode-Type Cross-Machine Shrinkage:** Compute the leave-one-out episode mean probability vector $\hat{\mathbf{p}}_{-m, \text{episode}}(h)$ both across all machines in the episode and across machines sharing the same `machine_type` and `signal` within the episode.

---

## Part 4: Recommended Two-Stage Architecture

1. **Stage 1 — Latent Condition Classifier (Auxiliary Task):**
   Train a 4-class classifier (e.g., LightGBM / CatBoost or MLP) on `train` + `validation` queries to predict $P(\text{condition} = c \mid \text{features}_q)$ using missingness patterns, machine-level $D_{\text{TV}}$, and episode-level synchronized drift $\Delta \bar{p}_{\text{episode}}$.
2. **Stage 2 — Dense Self-Supervised Pre-Training + Direct Brier Fine-Tuning:**
   * **Expand Training Data via Rolling Cutoffs:** Slice `train/history.csv` at historical ticks $t \in [120, \text{cutoff}-1]$ to create tens of thousands of additional training rows, and combine them with `validation` (where exact target probabilities are known).
   * **Multi-Head / GBDT + Neural Blending:** Train specialist forecasters and blend their probability vectors using weights optimized directly on the competition's **condition-normalized score $R$** via `scipy.optimize.minimize` (SLSQP with simplex constraints $\sum_{k} \hat{p}_k = 1, \hat{p}_k \ge 0$).
3. **Episode-Grouped Cross-Validation:** Always use `GroupKFold(n_splits=5)` grouped by `episode_id` (stratified by `condition`) so machines from the same episode never leak across train and validation splits.

---

## Part 5: 10 Critical Factors That Affect Results & How to Tweak Them

### 1. The Variance vs. Bias Tradeoff Across the Four Regimes
The simulator evaluates your model equally across four distinct hidden regimes (`stable`, `short history`, `drift`, `shared change`), each requiring conflicting memory lengths.

* **How it affects results:**
  * If you use a **long memory window** (or unweighted empirical average), you achieve near-zero variance on `stable` machines, which lowers your squared error there. However, on `drift` and `shared change` machines, old ticks come from a *pre-change distribution*, introducing massive **bias** that ruins your score in those two regimes.
  * Conversely, if you use a **short memory window** (e.g., half-life $h = 8$ or $16$), you adapt quickly to `drift`, but your sample size shrinks so much that sampling noise inflates your error on `stable` machines.
* **How to adjust and tweak:**
  * **Adaptive Memory Weighting (Kalman / BOCPD Gain):** Instead of a fixed half-life $h$, make the half-life a function of the **divergence between the last 24 ticks and all older ticks**. Compute the chi-squared or Jensen-Shannon divergence $D_{\text{JS}}(\hat{\mathbf{p}}_{\text{last 24}}, \hat{\mathbf{p}}_{\text{older}})$.
  * **Tuning rule:** Define an adaptive weight $\lambda_m = \sigma\big(\beta_0 + \beta_1 D_{\text{JS}} + \beta_2 \bar{D}_{\text{JS, episode}}\big)$ where $\sigma(\cdot)$ is the logistic sigmoid, and blend:
    $$\hat{\mathbf{p}}_m = (1 - \lambda_m)\hat{\mathbf{p}}_{\text{long}} + \lambda_m \hat{\mathbf{p}}_{\text{short}}$$
    Tune $(\beta_0, \beta_1, \beta_2)$ directly on the validation set. When a machine is stable, $D_{\text{JS}}$ is small and the model trusts the full history; when a machine or episode drifts, $D_{\text{JS}}$ spikes and the model shifts weight to the short window.

### 2. Missing Ticks & Irregular Sampling in `short history`
In the `short history` condition, you do not get all ticks up to `cutoff`. Instead, you get **40 recent consecutive ticks** ($\text{cutoff}-40$ to $\text{cutoff}-1$) and **24 randomly sampled older ticks**.

* **How it affects results:**
  * **Row-Index Bug (Fatal Error):** If your code uses row indices (e.g., `df.iloc[-2]` for lag 2, or exponential decay based on row count `np.arange(len(df))`), the 24 older ticks will be treated as if they happened immediately before tick $\text{cutoff}-40$, and lags computed inside historical rollouts will align the wrong ticks.
  * **Density Bias in Exponential Decay:** Even if you use exact tick timestamps $w_t = 2^{-(\text{cutoff}-t)/h}$, a `short history` machine has *fewer observed ticks* in the older region $[1, \text{cutoff}-41]$ than a `stable` machine. Summing raw weights $\sum w_t$ without accounting for sampling density causes the recent 40 ticks to dominate the older 24 ticks far more heavily than they would in a fully observed series.
* **How to adjust and tweak:**
  * **Strict Timestamp Indexing:** Always index by exact tick $t$. Reindex or pivot each machine's history onto a dense grid $t \in [0, \text{cutoff}-1]$ filled with `NaN` for unobserved ticks so that lag $l$ is strictly `grid[t - l]`.
  * **Importance-Sampling / Density Correction for Older Ticks:** In `short history`, those 24 older ticks are a random sample of a much larger interval (roughly $\text{cutoff} - 40 \approx 152\text{ to }216$ ticks). When computing the long-term baseline estimate for a `short history` machine, weight each of the 24 older ticks by the **inverse sampling probability** $\frac{\text{cutoff} - 40}{24} \approx 7\times$ if you want an unbiased estimate of the machine's full-history mean before applying shrinkage.
  * **Stronger Prior Shrinkage:** Increase the Dirichlet prior strength $\alpha_0$ (shrinking toward the `machine_type` + `signal` + `episode` mean) whenever total observed ticks $N_m \le 64$.

### 3. Lagged Interactions (Up to 8 Ticks) & Opposite-Sign Machine Responses
Past payouts can interact at delays of up to eight ticks, and machines of the same type can respond differently, even in opposite directions.

* **How it affects results:**
  * If you train a standard pooled model (like a global logistic regression or GBDT) using only `payout_lag_1` through `payout_lag_8` as features, the model tries to learn a single global coefficient for `payout_lag_1`.
  * If Machine A is "hot/momentum" (a win at $t-1$ *increases* win probability at $t$) and Machine B of the same `machine_type` is "cold/compensating" (a win at $t-1$ *decreases* win probability at $t$), the average effect across machines is near **zero**. A global model will conclude lags don't matter and ignore them completely.
* **How to adjust and tweak:**
  * **Per-Machine Autoregressive Lift Features (Target-Encoded Lags):** For each machine $m$, iterate over all valid historical pairs $(t-l, t)$ inside its own history ($t < \text{cutoff}$) for each lag $l \in \{1, \dots, 8\}$. Estimate that specific machine's local conditional shift:
    $$\Delta \hat{\mathbf{p}}_{m, l}(y_{t-l}) = \hat{\mathbf{p}}_m(y_t \mid y_{t-l}) - \hat{\mathbf{p}}_m(y_t)$$
  * **Shrinkage on Local Transition Matrices:** Because 200 ticks is too small to estimate a full $5 \times 5$ transition matrix for each of the 8 lags without overfitting, compress the conditioning event at lag $l$ into **2 or 3 states**: $\text{Loss } (0\times)$, $\text{Small Win } (1\times, 2\times)$, and $\text{Big Win } (5\times, 20\times)$, or use the scalar log-payout $\log(1 + y_{t-l})$ correlated with future outcome indicators.
  * **Interaction Feature for the Cutoff Spin:** Feed the product $\text{LocalAutocorr}_{m, l} \times \text{ObservedState}_{m, \text{cutoff}-l}$ into your model. That way, if Machine A has positive lag-2 correlation and just hit a win at $\text{cutoff}-2$, the feature is positive; if Machine B has negative lag-2 correlation and just hit a win at $\text{cutoff}-2$, the feature is negative.

### 4. Synchronized Multi-Machine Shifts (`shared change`)
Within an `episode_id`, multiple machines are observed simultaneously up to the same `cutoff`, and win chances/payouts can shift across several machines together.

* **How it affects results:**
  * Looking at a single machine's last 10–15 spins has too much variance to distinguish a real parameter change from a lucky/unlucky streak.
  * However, if an episode has $M$ machines (e.g., 8 to 20 machines) and half of them simultaneously show a +10% jump in win rate over the last 20 ticks, the probability that this is random noise drops by $1/\sqrt{M}$.
* **How to adjust and tweak:**
  * **Cross-Sectional Residual Pooling:** For every tick $t < \text{cutoff}$ and every machine $m$ in episode $e$, subtract the machine's historical baseline probability vector $\overline{\mathbf{p}}_{m, \text{old}}$ from its one-hot outcome $\mathbf{e}_{y_{m,t}}$ to get the **innovation/residual vector**:
    $$\mathbf{r}_{m,t} = \mathbf{e}_{y_{m,t}} - \overline{\mathbf{p}}_{m, \text{old}}$$
  * **Episode Common Shock Estimator:** Average those residuals across all other machines $m' \neq m$ in the same episode over a short recent window (e.g., half-life $h = 12$ or last 20 ticks):
    $$\overline{\mathbf{r}}_{-m, e}^{\text{recent}} = \frac{1}{|M_e - 1|} \sum_{m' \neq m} \sum_{t < \text{cutoff}} \tilde{w}_t \mathbf{r}_{m', t}$$
  * Add a scaled version of $\overline{\mathbf{r}}_{-m, e}^{\text{recent}}$ (or stratify it by `machine_type` and `signal`) directly to machine $m$'s predicted probabilities before projecting back onto the probability simplex.
  * **Sub-Group Clustering Within an Episode:** Because shifts can happen across *several* machines together (not necessarily *all* machines in the episode), compute the cosine similarity between machine $m$'s recent residual vector $\mathbf{r}_{m}^{\text{recent}}$ and every other machine $m'$'s recent residual vector in the episode, and pool more heavily with machines moving in the same direction or sharing the same `signal` / `machine_type`.

### 5. Rare Class Estimation ($5\times$ and $20\times$) & Zero-Count Traps
In slot-machine logs, $0\times, 1\times, 2\times$ occur frequently, while $5\times$ and $20\times$ (especially $20\times$ on `steady` vs. `jackpot` machines) are rare events.

* **How it affects results:**
  * In a 64-tick (`short history`) or 200-tick history, a machine with a true $p_{20} = 0.015$ has a $(1 - 0.015)^{64} \approx 38\%$ chance of showing **zero** $20\times$ payouts in its history.
  * An unsmoothed empirical or short-window frequency estimator will output $\hat{p}_{20} = 0.000$. Conversely, if it hits twice by chance in the last 20 ticks, a short-window estimator will output $\hat{p}_{20} = 0.100$, causing a huge squared error spike $(0.100 - 0.015)^2$.
* **How to adjust and tweak:**
  * **Class-Specific Half-Lives & Shrinkage ($\alpha_{0,k}$):** Do **not** use the same half-life $h$ or prior strength $\alpha_0$ for all 5 classes:
    * Frequent outcomes ($0\times, 1\times, 2\times$) have enough statistical power to use shorter half-lives ($h \in [16, 32]$) and weaker prior shrinkage.
    * Rare outcomes ($5\times, 20\times$) have high Poisson variance; give them a **longer half-life** ($h \in [64, 128]$) and **higher shrinkage** toward the `(machine_type, signal)` prior, coupled with the machine's overall recent win-rate shift.
  * **Hierarchical Decomposition (Win Probability $\times$ Conditional Payout Distribution):** Factorize the 5 probabilities into two sub-problems:
    1. **Binary Win Probability:** $P(\text{Win}) = 1 - p_0$ (high sample size, easy to track drift and lags over short windows).
    2. **Conditional Payout Severity given a Win:** $P(y = k \mid y > 0)$ for $k \in \{1, 2, 5, 20\}$, which is strongly governed by `machine_type` (`steady`, `balanced`, `jackpot`) and `signal` ($-1, 0, 1$).
    Then reconstruct $\hat{p}_k = (1 - \hat{p}_0) \cdot \hat{P}(y = k \mid y > 0)$. Blending a factorized model with a direct 5-class model significantly stabilizes rare-class predictions.

### 6. Condition-Normalized Scoring Metric ($R$) & Asymmetric Influence
The contest scoring formula is:
$$R = \frac{1}{4} \sum_{c \in \{\text{stable, short, drift, shared}\}} \frac{\text{MSE}_c}{\text{MSE}_{c, \text{reference}}}$$

* **How it affects results:**
  * Standard training losses (like unweighted MSE or log-loss across all queries in `train` or `validation`) weight every query equally.
  * However, in the competition metric, **1) episodes with fewer machines have more weight per machine** (since errors are averaged *within each episode first*, then across episodes), and **2) conditions where the `pooled` reference model already has a very small error ($\text{MSE}_{c, \text{reference}}$) have a much smaller denominator.**
  * Specifically, in the `stable` condition, the `pooled` baseline is already quite accurate, so $\text{MSE}_{\text{stable, reference}}$ is small. That means a tiny $0.0005$ improvement (or degradation) in $\text{MSE}_{\text{stable}}$ translates into a **much larger change in the ratio $R$** than the exact same $0.0005$ error change in `drift` (where the reference error is large).
* **How to adjust and tweak:**
  * **Sample Weighting During Training / Blending:** When fitting any global model or ensemble blender on validation/training episodes, weight each query $q$ in episode $e$ and condition $c$ by:
    $$w_q = \frac{1}{|M_e| \cdot N_{\text{episodes}, c} \cdot \text{MSE}_{c, \text{reference}}}$$
    Optimizing with these exact weights aligns your loss gradient $100\%$ with the leaderboard score $R$.
  * **Protect the `stable` Condition at All Costs:** Never deploy a global tweak (like shortening the global half-life) that improves `drift` by 2% if it hurts `stable` by 3% in relative ratio terms. Always inspect the 4 individual condition ratios inside `validation_score.json`.

### 7. Training Label Noise (`train` vs. `validation` Asymmetry)
* `public/train/` gives you **observed next-spin payouts (`labels.csv`)** — i.e., a single random draw $y_q \in \{0, 1, 2, 5, 20\}$, **not** the true underlying probabilities.
* `public/validation/` gives you **both** the observed payouts **and the simulator's true target probabilities**.

* **How it affects results:**
  * If you train a complex neural network or GBDT only on `train/labels.csv` (one-hot targets), the targets have enormous irreducible label noise ($\text{Var}(\mathbf{e}_y) = \mathbf{p}(1-\mathbf{p})$). Over-parameterized models will memorize noise in `train` and produce poorly calibrated probabilities.
  * Conversely, `validation` has noiseless ground-truth probability vectors $\mathbf{p}_q$, which provide dramatically higher signal-to-noise ratio per query—but over-tuning directly on `validation` risks overfitting to the validation episodes.
* **How to adjust and tweak:**
  * **Massive Self-Supervised Data Multiplication from `history.csv`:** Every machine in `train/history.csv` (and even `validation/history.csv` and `test/history.csv` prior to cutoff) has up to 256 observed spins. By creating sliding virtual cutoffs at $t \in [100, \text{cutoff}-1]$, you turn a few thousand queries into **hundreds of thousands of self-supervised training rows** to learn the exact lag structures ($l = 1 \dots 8$) and drift mechanics.
  * **Smoothed Pseudo-Targets on Historical Windows:** For any historical tick $t$ in a `stable` episode where you can observe ticks *after* $t$ ($t+1 \dots \text{cutoff}-1$) during training, you can compute the empirical distribution over $[t-20, t+20]$ as a low-noise proxy for the true probability vector $\mathbf{p}_t$ when training your feature extractors (using only features from $< t$ as inputs).
  * **Out-of-Fold (OOF) Fitting on `validation` Target Probabilities:** Use 5-fold Episode-Grouped Cross-Validation on `public/validation/` using the **exact ground-truth target probabilities** as the regression target to fit your final linear/Dirichlet stacking weights and calibration layer, then refit on all validation episodes before generating `test/submission.csv`.

### 8. The Static Context Features (`machine_type` and `signal`)
Every query includes `machine_type` $\in \{\text{steady}, \text{balanced}, \text{jackpot}\}$ and `signal` $\in \{-1, 0, 1\}$.

* **How it affects results:**
  * `machine_type` sets the structural base shape of the 5-class distribution (e.g., `jackpot` has higher $p_0$ and higher $p_{20}$, whereas `steady` concentrates mass on $1\times, 2\times$).
  * `signal` ($-1, 0, 1$) acts as a regime modifier or bias shift on the payout odds.
* **How to adjust and tweak:**
  * **9-Cell Stratified Priors:** Always stratify your hierarchical priors across all $3 \times 3 = 9$ combinations of `(machine_type, signal)`, rather than pooling across `machine_type` alone.
  * **Test Whether `signal` Modulates Drift or Lags:** Check in validation data whether `signal == 1` vs. `signal == -1` represents the direction of recent drift or a covariate shift in the transition matrix. Include interaction terms `signal * recent_win_rate` and `machine_type * lag_payout` in your model.

### 9. Probability Simplex Constraints & Post-Processing Calibration
Your five outputs $(\hat{p}_0, \hat{p}_1, \hat{p}_2, \hat{p}_5, \hat{p}_{20})$ must be non-negative and sum to 1.

* **How it affects results:**
  * If you train 5 separate regression models (e.g., 5 LightGBM regressors predicting $p_0, p_1, p_2, p_5, p_{20}$) and simply normalize them by dividing by their sum ($\hat{p}_k / \sum \hat{p}_j$), negative predictions or noisy shifts in $\hat{p}_0$ distort the small probabilities $\hat{p}_5$ and $\hat{p}_{20}$.
  * On the other hand, if you use a softmax output trained on log-loss (cross-entropy), extreme log-loss penalties near $p=0$ force the model to over-predict rare probabilities compared to the Brier score optimum.
* **How to adjust and tweak:**
  * **Euclidean Projection onto the Probability Simplex:** Because Brier score is squared Euclidean distance $\|\hat{\mathbf{p}} - \mathbf{p}\|_2^2$, the mathematically optimal way to map any unconstrained 5D prediction vector $\mathbf{z} \in \mathbb{R}^5$ onto the valid probability simplex $\Delta^4$ (minimizing Brier error) is **Euclidean Simplex Projection** ($\min_{\mathbf{p} \in \Delta^4} \|\mathbf{p} - \mathbf{z}\|_2^2$), **not** division by the sum or softmax:

```python
import numpy as np

def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Projects a 1D array v onto the probability simplex (L2 optimal for Brier score)."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    return np.maximum(v - theta, 0.0)
```

  * **Dirichlet / Affine Matrix Calibration Layer:** Before projecting to the simplex, apply a learned $5 \times 5$ affine transformation $\hat{\mathbf{p}}_{\text{cal}} = \text{Proj}_{\Delta^4}(W \hat{\mathbf{p}}_{\text{raw}} + \mathbf{b})$ per `(machine_type, signal)` or per predicted condition, fitted via L-BFGS / SLSQP to minimize the exact contest score $R$.

### 10. Round 2 ("Adaptation") Overnight Distribution Shifts & Reproducibility
At 9 PM Saturday, Round 2 releases fresh data where some patterns have changed, worth 20 points, and all submissions must pass an automated reproducibility check (`entry.zip`).

* **How it affects results:**
  * **Hardcoded Constants Break in Round 2:** If you hardcode fixed class priors (e.g., `p_20 = 0.02`) or fixed lag weights from Round 1's training data, and Round 2 changes the base payout table, changes which lag $l \in \{1..8\}$ is active, or alters the drift rate, hardcoded parameters will fail badly.
  * **Reproducibility Mismatch (`mismatch` / `failed` status):** Non-deterministic multi-threading in GBDTs/PyTorch, unseeded random splits, or exceeding the grader's CPU timeout/dependency environment causes the automated code check to reject your score even if your CSV was accepted.
* **How to adjust and tweak:**
  * **Make All Priors & Hyperparameters Data-Driven:** Write your pipeline so that base priors $\boldsymbol{\pi}_{\text{type, signal}}$, lag correlation matrices, and ensemble blending weights are **automatically estimated from whatever `public/train` and `public/validation` folders are passed in**. That way, when Round 2 drops at 9 PM, running your script on the new `public/` directory automatically adapts to the new simulator parameters in minutes.
  * **Compare Round 1 vs. Round 2 Distribution Immediately at 9 PM:** Run a diagnostic script that prints:
    1. Mean probabilities per `(machine_type, signal)` in `validation`,
    2. Average autocorrelation at lags $1 \dots 8$,
    3. Error of each baseline method (`global`, `context`, `empirical`, `recent`, `pooled`, `pooled_recent`) per condition.
  * **Deterministic Execution Check:** Fix all seeds (`random`, `numpy`), sort all dictionaries/DataFrames by `query_id`, `episode_id`, `machine_id`, and `tick` before any groupby or aggregation, and test your `entry.zip` locally using the kit's checker before uploading.

---

## Part 6: Summary Checklist of Highest-ROI Tweaks (In Order of Impact)

| Priority | Tweak / Modification | Target Regime(s) | Expected Impact |
| :--- | :--- | :--- | :--- |
| **1** | **Multi-half-life ensemble** ($h \in \{8, 16, 32, 64, \infty\}$) gated by a **latent condition classifier** ($P(\text{condition} \mid \text{features})$) | All 4 (`stable`, `short`, `drift`, `shared`) | **Massive** — resolves the core memory-length tradeoff across conditions. |
| **2** | **Episode-level recent residual pooling** ($\overline{\mathbf{r}}_{-m, e}^{\text{recent}}$ across peer machines at $t < \text{cutoff}$) | `shared change` & `short history` | **High** — directly captures synchronized multi-machine shifts. |
| **3** | **Per-machine lag-1 to lag-8 conditional lift** interacted with exact tick $\text{cutoff}-l$ outcomes | All 4 (especially machines with lag dynamics) | **High** — captures up to 8-tick delays even when machines respond in opposite directions. |
| **4** | **Direct optimization of blending weights on validation target probabilities** using the exact condition-normalized $R$ objective + L2 simplex projection | All 4 (especially `stable` denominator weight) | **Medium–High** — aligns model outputs with the exact competition scoring rule. |
| **5** | **Factorized Win-Rate $\times$ Conditional Severity** with stronger shrinkage on $5\times, 20\times$ | `short history` & `drift` | **Medium** — eliminates rare-class variance spikes. |