# What sets the payout-size atom

The atom at the query tick is a per-machine lag rule. Each machine uses one lag in {1, 2, 4} and one polarity. Given a win, the conditional mix is atom `s XOR 1{payout at that lag ≥ 2}`. A uniform Bayesian average over the three lags, with the polarity updated from that machine's own winning spins, is the predictor.

Plugged into v6's win probability (v6 `p_0` kept, size mix replaced, probabilities floored at 1e-4), this scores **57.03** against v6's **56.29**. A perfect atom, same `p_0`, scores 57.71. The gain is +0.74 points, about half of the gap that a perfect atom can close.

Argmax accuracy on the 768 validation queries is **87.2%**. The mean posterior on the true atom is 0.841. When at least half the posterior sits on one lag (599 queries), the hard rule `s XOR bit` matches the target on **93.8%** (562/599).

## The two atoms

Every validation target's `(p_1, p_2, p_5, p_20) / p_win` matches one of two vectors per type. The largest within-cluster deviation is 1.3e-14. Atom 0 is the one with the higher chance of paying 1.

| type | atom | n | p(1) | p(2) | p(5) | p(20) |
|---|---|---:|---:|---:|---:|---:|
| steady | 0 | 137 | 0.7539893896 | 0.1968084883 | 0.0482180796 | 0.0009840424 |
| steady | 1 | 132 | 0.4551220076 | 0.4359023939 | 0.1067960865 | 0.0021795120 |
| balanced | 0 | 131 | 0.4998531085 | 0.3479282724 | 0.1304731021 | 0.0217455170 |
| balanced | 1 | 103 | 0.2140661471 | 0.5467365933 | 0.2050262225 | 0.0341710371 |
| jackpot | 0 | 147 | 0.3351460319 | 0.2449461988 | 0.3149308270 | 0.1049769423 |
| jackpot | 1 | 118 | 0.1207867301 | 0.3239206784 | 0.4164694436 | 0.1388231479 |

KL between the two steady atoms is about 0.19 nats. If a machine's history were really one atom, 60 wins would pick the wrong one about 1% of the time. They do not.

## The rule

For machine `m` and lag `L ∈ {1, 2, 4}`:

- `f(t) = 1` when the spin at `t − L` paid at least 2, and `0` when it paid 0 or 1. A missing tick leaves `f(t)` unknown.
- Latent polarity `s ∈ {0, 1}`.
- A win at tick `t` is drawn from atom `s XOR f(t)`.

Losses carry no information about the atom. The prior is uniform on the three lags and on `s`. The posterior uses only wins before the cutoff. The forecast is the posterior mixture of the two atoms. Lags 1, 2 and 4 each take about a third of the machines (MAP counts 265, 251, 252). Polarities are split 350 / 418, so the direction is a per-machine bit, the same way the win rule's direction is.

| slice | accuracy | P(true atom) | n |
|---|---:|---:|---:|
| all | 0.872 | 0.841 | 768 |
| stable | 0.917 | 0.901 | 192 |
| drift | 0.922 | 0.913 | 192 |
| shared_change | 0.932 | 0.892 | 192 |
| short_history | 0.716 | 0.659 | 192 |
| steady | 0.952 | 0.927 | 269 |
| balanced | 0.895 | 0.869 | 234 |
| jackpot | 0.770 | 0.730 | 265 |

Mean posterior mass is 0.32 / 0.35 / 0.33 on lags 1 / 2 / 4. Calibration at the top is tight: 498 queries have posterior confidence ≥ 0.9, accuracy there is 0.976, and the mean probability on the true atom is 0.965. The misses are machines whose lag and polarity are not yet identified, mostly short histories and jackpots (fewer wins, closer atoms).

## Score against v6

`slot_forecasting.contest.score_contest_files(..., max_points=60)`, v6's `p_0` held fixed.

| model | points | stable | short | drift | shared |
|---|---:|---:|---:|---:|---:|
| v6 | 56.29 | 0.0541 | 0.1047 | 0.0488 | 0.0564 |
| full-history atom posterior | 53.88 | 0.1027 | 0.1336 | 0.1052 | 0.1131 |
| three-lag mixture | **57.03** | **0.0436** | **0.0844** | **0.0379** | **0.0424** |
| true atom | 57.71 | 0.0399 | 0.0518 | 0.0351 | 0.0321 |

All four conditions improve. The paired episode squared-error gap (v6 minus this model) averages 0.00265, standard error 0.00057, better on 77 of 96 episodes. The 95% interval is [0.0015, 0.0038]. Even episodes: 56.45 → 57.04. Odd episodes: 56.15 → 57.03.

v6's own size mix is already on the nearer atom 77.9% of the time (weight on the true atom 0.74). The lag mixture raises that to 87.2%. On train histories the three-lag model gains 1,736 nats of online log score over a static atom (−58,636 versus −60,372 on 58,302 wins).

## What does not determine the atom

**A fixed atom per machine.** The posterior from every historical win is 68.7% accurate (mean P(true) 0.676, mean confidence 0.920). On machines with 60 or more wins, accuracy is 0.716 while confidence is 0.971, and 118 of 420 confident calls are wrong. That is the prompt's ~66% and the 53.88 score, reproduced here at 53.88. The history really does look like one atom. The query is often the other one, because the lag bit spends more time on one side and the pooled likelihood locks onto that side.

**A hidden Markov atom.** A two-state HMM with the known atoms as emissions, switch rate chosen by train marginal likelihood, picks a per-tick switch probability of 0.02 (win-step chain: 0.10). That is +335 nats over a frozen atom, far less than the lag model's +1,736. The filtered posterior at the cutoff is 61.3% accurate and scores 54.68. Every switch rate above 0 is worse on the validation labels than rate 0. Early-half and late-half maximum-likelihood atoms disagree on 31% of machines; when they disagree, the late half matches the query 55% of the time. A single changepoint scores 54.49. The atom is not a slowly switching latent, and it does not flip with the shared-change or drift schedule.

**One global lag, or a parity of two lags.** Across win / ≥2 / ≥5 / =20 bits, every lag 1..8, and every pair's XOR, AND and OR (405 features), the per-machine polarity was fit from history only. The best train feature, `payout ≥ 2` at both lags 1 and 2, gains +344 nats and is 69.9% accurate on validation. The best validation association in that list is 71.0% (`≥2` at lags 1 and 4). A single global sign, fit on train, agrees with the query about 52% of the time. XOR of lags 1, 2 and 4 has a negative train gain as a lone feature, and adding it to the three-lag average costs 83 train nats. A free 4-cell map on a pair (the analogue of a non-parity interaction) overfits: the train-best cell scores 55.86, and the validation-best cell (`≥2` at lags 2 and 4, tuned on the labels) scores 56.29, level with v6.

**Live/dead, signal, machine index, neighbours.** The recent win-rule pair's live bit matches the atom on 47.4% of queries (best orientation 52.6%). Signal, within type, moves P(atom 1) only between 0.41 and 0.51. Neighbour bits at lags 1, 2 and 4, best global orientation, reach 55.1%. Giving each machine a polarity on a neighbour's bit loses several hundred train nats. Using the machine's own win-rule pair as the size pair also loses likelihood.

The threshold is 2, not 5 and not "any win". Families built on `payout ≥ 5` or on the win bit gain a few hundred train nats and score about 54.5. `payout ≥ 2` is the bit that separates the atoms.

## How the choice was made

The lag set {1, 2, 4} and the big-win bit are the ones already named by v6's size keys and by the brief. Train online likelihood, not the validation score, then dropped two extras that had been included in the first average: the XOR of all three lags, and a static-atom hypothesis. A static prior of 2% is the train-likelihood peak and scores 57.033, against 57.030 for the uniform three-lag average. The report uses the uniform three-lag average.

Validation labels were used to measure accuracy and the contest score. They were not used to pick the lag, the threshold, or the polarity.

## Recommendation

Replace v6's size split with this three-lag mixture. Keep v6's win probability. That is worth about **+0.74** points, and both episode halves move the same way.

Leave the win model alone. Live/dead, signal and neighbours do not carry the size bit. A full-history atom posterior, and an HMM on top of it, both throw away the lag and cost points.

What is left is identification, not a missing feature. Short history is 71.6% accurate and jackpot is 77.0%, against 95% on steady and about 92% on long stable, drift and shared-change histories. The oracle still has 0.68 points on the table, mostly there. The next size experiment is a sharper prior for the lag and the polarity when the machine has few wins, using the same three lags.

## Code

- `investigate.py` searches the HMM, the windows, the 405 lag features, the cell maps, the neighbour bits, and the pre-specified averages. It writes `results.json` and `predictions/`.
- `characterize.py` decomposes the lag posterior and the paired episode gap.
- The scored file is `predictions/bma_ge2_124_only.csv`.
