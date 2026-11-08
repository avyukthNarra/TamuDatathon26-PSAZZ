# Research brief: improving win-probability forecasts for "House Edge" (TAMU Datathon 2026)

> Paste this whole file into a deep-research tool. It is self-contained: the tool has no access to our code.

## The problem (what we forecast)

- Simulated slot machines. Each machine has a payout history: one payout per tick, in {0, 1, 2, 5, 20} (0 = loss).
- For each query (one machine at a cutoff tick) we forecast the probability vector of the **next** payout: P(0), P(1), P(2), P(5), P(20).
- **Scoring:** squared error against the **true** generating probabilities (not the sampled outcome). It's computed as an error ratio vs a fixed reference forecast, averaged over four conditions, and points = 60 / (1 + R).
- **Conditions:**
  - stable
  - short_history: about 200 ticks with only 64 observed, scattered with about 20 gaps
  - drift: the rules change over time
  - shared_change: machines in the same episode change at the same times
- **Data:**
  - about 8 machines per episode, of 3 machine types (balanced, jackpot, steady)
  - training data with sampled outcomes only
  - 96 validation episodes with the true probabilities
  - a `signal` column in {-1, 0, 1} per machine
- **Organizer rules:** earlier outcomes interact at lags up to 8 ticks. Machines of the same type can respond differently or *oppositely*. Both win chance and win size can change. Other machines' earlier outcomes may help. A second round (14 hours) changes which lag relationships recur, so methods must re-learn structure from new data quickly.
- **Constraints:**
  - offline and CPU only: 2 cores, 16 GB, 30 minutes total runtime
  - Python, deterministic
  - standard library preferred; numpy/scipy are acceptable

## What we know about the generator (found empirically)

- **Win/loss** depends on pairs of lagged outcomes. The pairs found on train are (1,3), (2,5) and (4,7). What matters is whether the two lagged outcomes **match or differ**, an XOR/parity-like interaction (a Walsh basis term).
- **True win probabilities are U-shaped but continuous.** By tenths of the population: 0.01, 0.02, 0.06, 0.13, 0.32, 0.66, 0.82, 0.91, 0.96. This looks like a logistic of a weighted sum of lag terms, not a two-level switch.
- **Payout size given a win** is one of exactly two mixes per machine type ("small" vs "big" mode). The mode is switched by whether the spin 1, 2, 3 or 4 back was a big win.
- In shared_change episodes, machines share changepoint *times*, but not rules.
- `signal` correlates with a machine's win level (jackpot: 0.17 / 0.24 / 0.31 true win rate for -1 / 0 / +1), but adds nothing once history is used.

## Current best model (dev score 55.34 / 60)

20% rule model + 80% "pair mixer":

- **Rule model:** searches all lag rules per machine.
- **Pair mixer:** an online mixture of experts. Each expert keeps exponentially forgotten win counts (half-lives 12 and 40) in "lags match / differ" cells, smoothed toward a two-component Beta-mixture prior (dead about 0.04, live about 0.90). Experts are weighted by exp(-eta * recent Brier).
- **Payout size:** a Bayes estimate of the two-mode switch.

**Where the remaining error is** (dev points if one part were perfect):

| Part made perfect | Points | Gain |
|---|---|---|
| nothing | 55.34 | - |
| win probability | 58.36 | **+3.0** |
| size given win | 56.76 | +1.4 |

Error ratio by condition: stable .063, short_history .121, drift .070, shared_change .082.

## Already tried, no gain (don't suggest these unless with a clearly new twist)

- Blending/stacking of 11 models, with linear, factorised (win vs size) and geometric (log) pools. Saturated.
- Logit recalibration, temperature, and gated blends by machine type or history length.
- Fixed-share expert switching, HMM, ADWIN, BOCPD alone, JSD gates.
- Train-learned priors over lag pairs, type-level shrinkage of win cells, pooling win cells across episode peers.
- Signal-dependent priors, which hurt.
- Peer-triggered memory shortening.
- For gapped histories: soft expected counts for missing lags, forward imputation, decay per observed tick, single-lag experts.
- Gradient-boosted residual learners, MLP meta-learner, a neural hypernetwork over prefix statistics (about 49 points), and a signed-L1 interaction logistic (about 48).

## Research questions (priority order)

### A. Learning a sparse, continuous win function per machine from 64-250 binary observations (biggest lever, up to +3)
1. Best methods for **online/sequential Bayesian logistic regression** on Boolean (Walsh/Fourier) features of lagged outcomes, with few samples. Compare Laplace, variational, assumed-density filtering and Polya-Gamma.
2. **Learning sparse parities / low-degree Boolean functions with noise:** practical algorithms and sample complexity for degree-2 terms among 8 lags (28 pairs + 8 singles).
3. **Hierarchical priors where effect size is shared across machines of a type but the sign varies per machine.** Examples: sign-flip mixtures, horseshoe or spike-and-slab with shared scale, partial pooling of |beta|.
4. **Bayesian model averaging over which lag terms are active**, done fast enough for about 600 machines in seconds. Context-tree weighting and variable-order Markov models as alternatives.
5. How do these compare with **count-based cell estimates plus Beta-mixture smoothing** (our current approach) when the true function is a logistic of a few terms?

### B. Non-stationarity (drift, shared changepoints)
6. **Dynamic logistic regression:** Kalman-filtered or discounted weights, and forgetting factors chosen online.
7. **Hierarchical / multi-series Bayesian online changepoint detection** where several series share changepoint times but have different parameters.

### C. Payout size
8. Online Bayesian estimation of a **two-regime switching mixture** whose regime depends on recent lagged events, from sparse, gapped histories.

### D. A different paradigm
9. **Simulation-based / amortized inference:** write a simulator matching the facts above, then train a sequence model (transformer, neural process, or TabPFN-style prior-fitted network) on millions of synthetic episodes to output near-Bayes-optimal next-step probabilities. Feasibility on CPU within 30 minutes of inference, how close such models get to Bayes-optimal, and how to make them robust to a changed generator in round two.
10. Any published work, competitions or Kaggle solutions on **forecasting probabilities of synthetic Markov/Boolean-interaction processes**, or on scoring rules against true probabilities.

## Output I want back

For each question:
- the 2-4 most relevant methods or papers, with links
- a short explanation of the method
- **concretely how it would plug into the model above** (replace the pair mixer's win cells, add as an expert, or a new pipeline)
- expected difficulty in hours of implementation
- the risks

End with a ranked shortlist of the **3 experiments** most likely to beat 55.34, each doable in under 4 hours of coding.
