# Rao-Blackwellized filter for next-spin win probability

The filter does not beat the pair mixer. The train-selected model scores **56.06 / 60** on validation. The pair mixer scores **56.29**. One pre-specified ablation, an episode-level changepoint model on the three train pairs, scores **56.30** and is a tie: mean paired episode-error difference −0.00011, 50 of 96 episodes better. That is inside the ±0.4 point noise between halves. `validation_predictions.csv` is the train argmax, not that ablation.

Payout size given a win is taken from the mixer’s validation predictions, so every number below isolates win probability. The size slice of the error is the same for every filter: stable 0.00275, short history 0.00952, drift 0.00272, shared change 0.00510.

## Model

Per machine the discrete hypothesis is a lag pair and a direction (agree is the live cell, or agree is the dead cell), plus a no-rule null. There are 28 pairs. Live and dead win rates are integrated out with a Beta (directed model) or with the mixer’s three-component Beta mixture on each cell (mixture model), so the weight of each hypothesis is a Bernoulli marginal. That is the Rao-Blackwell step: the particles are the discrete hypotheses, not samples of the rates.

Three ways of handling a changing rule, all fit on train:

| Mechanism | What is stored |
| --- | --- |
| No changepoint | One run over the whole history. Counts can be frozen (exact Bayes) or forgotten with half-life γ = 2^(−1/h). |
| Per-machine BOCPD | Top-K run lengths per machine. A new run starts with hazard H and the prior; the old runs keep growing with probability 1−H. |
| Episode BOCPD | One run length shared by all 8 machines. Each machine keeps its own hypothesis. A surprising tick is the product of the eight predictive likelihoods, which is how a shared change shows up. |

Missing outcomes are skipped. A tick whose lag context is missing does not update that hypothesis’s counts; the predictive averages the two cells. Hazard is charged only on observed ticks, so the gap in a short history does not itself force a changepoint. A synthetic check in `pf.py` recovers pair (1, 3) with posterior mass 1, prefers half-life 24 over a static rule after a reversal (recent log score −30.7 vs −40.2), and puts the shared run’s birth tick on a flip at t = 100.

## Fit

Hyperparameters were chosen on train only: the mean, over the four conditions, of each episode’s mean log score on its last 40 observed ticks. Validation was not used to pick the csv. Dead/live levels are the same validation-target mixture the mixer uses (steady 0.149/0.956, balanced 0.039/0.840, jackpot 0.013/0.669, strength scaled from 10). The null sits on the train base rate (steady 0.578, balanced 0.407, jackpot 0.224).

Train discovery puts 216, 212 and 150 machines on (2, 5), (4, 7) and (1, 3). The next pair has 26. Stage 1 gave every pair equal prior. Stage 2 either kept only those three pairs or put 90% of the rule prior on them and 10% on the other 25.

Train argmax, and the csv: directed per-machine BOCPD, hazard 0.005, no within-run forgetting, Beta strength 16, top 8 runs, spiked pair prior (`dirS-mac-s16`). Train objective −0.3377. The episode ablation was second, −0.3384, worse on drift histories and better on the other three conditions.

## Validation against the mixer

Official points use the frozen starter as the denominator, same as `runs/combined-v6`.

| Model | Points | Stable | Short | Drift | Shared | Paired mean (ours − mixer) | Episodes better |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pair mixer | 56.286 | 0.01062 | 0.01857 | 0.00926 | 0.01158 | 0 | — |
| **Train argmax, spiked per-machine CP** | **56.055** | 0.00886 | 0.02221 | 0.00983 | 0.01220 | +0.00077 | 45/96 |
| Episode CP, 3 pairs, half-life 48 | 56.298 | 0.00924 | 0.02090 | 0.00854 | 0.01091 | −0.00011 | 50/96 |
| Mixture forget, 3 pairs, half-lives 12/40/100 | 56.052 | 0.01038 | 0.02316 | 0.00861 | 0.01089 | +0.00075 | 52/96 |
| Mixture forget, all 28 pairs, same half-lives | 55.789 | 0.01035 | 0.02679 | 0.00862 | 0.01082 | +0.00164 | 51/96 |
| Per-machine CP, all 28 pairs, uniform prior | 55.468 | 0.00898 | 0.02876 | 0.01119 | 0.01232 | +0.00280 | 35/96 |
| Static Bayes, 3 pairs, mixture strength 10 | 32.870 | 0.00888 | 0.02366 | 0.33773 | 0.27623 | +0.149 | 22/96 |
| Static Bayes, all 28 pairs | 33.295 | 0.00894 | 0.02752 | 0.33483 | 0.25569 | +0.144 | 19/96 |

Loss ratios for the episode ablation: stable 0.047, short 0.118, drift 0.045, shared 0.053 (mixer 0.054 / 0.105 / 0.049 / 0.056). For the csv model: 0.045 / 0.125 / 0.052 / 0.059.

Win-probability squared error, episode-macro, for the episode ablation is 0.00548 against the mixer’s 0.00552. By condition it is stable 0.00479 (mixer 0.00557), short 0.00848 (mixer 0.00675), drift 0.00432 (mixer 0.00489), shared 0.00434 (mixer 0.00485). The short-history win probability is the remaining gap.

### Per type

Episode-macro distribution error.

| Type | Mixer | Episode CP, 3 pairs | Train argmax |
| --- | ---: | ---: | ---: |
| Balanced | 0.01266 | 0.01229 | 0.01337 |
| Jackpot | 0.01255 | 0.01287 | 0.01464 |
| Steady | 0.01493 | 0.01501 | 0.01413 |

Jackpot win-probability squared error stays about 0.0085 (mixer 0.00825). The filter does not fix the jackpot win-rate problem.

The short-history loss is concentrated on steady machines, and the mixer has the same shape: short/steady error is 0.0399 for the episode model and 0.0348 for the mixer. Short/jackpot is 0.0191 vs 0.0150. Short/balanced is 0.0159 vs 0.0158. On drift the episode model is better on jackpot (0.0114 vs 0.0141) and balanced (0.0097 vs 0.0115). On stable it is better on steady (0.0061 vs 0.0104).

### Paired episodes for the 56.30 ablation

Mean episode error minus the mixer. Negative means the filter is closer to the true probabilities.

| Condition | Mean diff | Median diff | Episodes better |
| --- | ---: | ---: | ---: |
| All | −0.00011 | −0.00003 | 50/96 |
| Stable | −0.00138 | +0.00008 | 11/24 |
| Short history | +0.00233 | +0.00029 | 9/24 |
| Drift | −0.00072 | −0.00018 | 14/24 |
| Shared change | −0.00067 | −0.00054 | 16/24 |

Stable’s mean is better while its median is not: a few episodes move a lot. Shared change is the cleanest gain (16/24, median −0.00054).

## Blends with the mixer

Pre-specified weights on the **train-argmax** probability vector. None of these was chosen by scanning weights on validation.

| Blend | Points | Stable | Short | Drift | Shared | Episodes better |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50/50 | 56.347 | 0.00909 | 0.01969 | 0.00888 | 0.01135 | 50/96 |
| 80% filter / 20% mixer | 56.214 | 0.00879 | 0.02103 | 0.00929 | 0.01173 | 46/96 |
| 20% filter / 80% mixer | 56.353 | 0.00985 | 0.01885 | 0.00895 | 0.01136 | 52/96 |

The 20/80 blend is +0.07 points over the mixer, with paired mean error −0.00026. That is the same noise band. An 80% filter blend is worse than the mixer because the short-history errors are still in the mixture.

## Ablation

**No changepoint, no forgetting.** Exact Beta-Bernoulli (or Beta-mixture) over the whole history scores about 33 points. Stable error is actually the best in the table (0.0089). Drift and shared change go to 0.26–0.34 because the posterior locks onto the early rule and the recent regime cannot catch a likelihood gap of hundreds of nats. This is the static exact-Bayes failure, reproduced.

**Forgetting, no explicit changepoint.** A cell-mixture with half-lives 12, 40 and 100 is the simple version that works: 55.79 with all 28 pairs, 56.05 on the three train pairs. It is better than the mixer on drift (0.0086 vs 0.0093) and shared change (0.0109 vs 0.0116) and worse on short history. A single half-life of 40, and a directed model with forgetting but no reset, both fell apart on train drift (condition log score −0.65 and −0.89). Direction is a hard label with a strong prior; once the rule reverses, the old direction stays confident unless a new run is allowed to start. The cell mixture does not have that problem, because each cell can move from the live component to the dead one as counts are forgotten.

**Per-machine changepoints.** This won the train objective at every stage. With a uniform 28-pair prior it scores 55.47, hurt by short history (0.0288) and by drift (0.0112). Restricting or spiking the pair prior brings it to 55.95–56.06. Hazard 0.005 beat 0.015 and 0.04 on train. Top-4 and top-8 were close. Adding a half-life of 40 inside the run did not win train.

**Episode changepoints.** On the full pair set, episode runs were slightly behind per-machine runs on train and scored 55.27 on validation, with the best short-history error of that stage (0.0246) and a bad drift error (0.0195) when a half-life of 48 was combined with all 28 pairs. On the three train pairs, hazard 0.01, half-life 48, strength 16, top 8 is the 56.30 tie. Shared-change evidence does pool: the synthetic birth tick landed on the true flip, and on validation this is the model that wins 16 of 24 shared-change episodes.

**Pair prior.** Moving from 28 uniform pairs to the three discovered pairs cut short-history error from 0.0268 to 0.0232 (mixture forget) and from 0.0288 to 0.0233 (per-machine CP). The spiked 90/10 prior on all 28 pairs was the train winner and landed at 56.06, a hair ahead of the hard 3-pair machine filter on train and on short history (0.0222 vs 0.0233). Fixed-share on hypothesis weights (α = 0.03) never entered the train top 15.

## Runtime

Validation wall time on 2 CPUs, then ×2 for 192 test episodes on the same 2 CPUs:

| Model | Val, 96 episodes | Test, 192 episodes |
| --- | ---: | ---: |
| Train argmax (28 pairs, per-machine top-8) | 33.8 s | 68 s |
| Episode ablation (3 pairs) | 6.5 s | 13 s |
| 3-pair mixture forget | 3.4 s | 7 s |

Both are far under 30 minutes. The research grid (45 configs, then 12) was about 9 minutes plus 80 seconds of train search on 2 CPUs and is not part of prediction.

## Recommendation

Keep the pair mixer as the submission. The filter ties it only in one ablation, loses under the train selection rule, and does not move jackpot win probability.

If the filter is kept as a second opinion, use the 3-pair episode model (hazard 0.01, half-life 48, strength 16, top 8 runs, pairs (1, 3), (2, 5), (4, 7)). It is the one that matches the mixer, it is faster, and its gains sit on drift and shared change, which is what the changepoint was for. Do not ship static Bayes. Do not ship a uniform prior over 28 pairs. A directed hypothesis needs a reset; a cell-mixture with three half-lives is the version that survives without one, and it still gives up short history.

The short-history steady cell is where both systems are weak (mixer 0.035, filter 0.040). That is the open problem, and it is win probability, not payout size.
