# What the size-atom misses are

The rule from g5 is unchanged: on a win, the payout atom is `s XOR 1{payout(t−L) ≥ 2}`, with one lag `L ∈ {1,2,4}` and one polarity `s` per machine. Losses are ignored. The forecast is the posterior mixture of the two atoms. v6's `p_0` is kept, and probabilities are floored at `1e-4`.

Reproduced with `score_contest_files(..., max_points=60)`:

| model | points | stable | short | drift | shared |
|---|---:|---:|---:|---:|---:|
| v6 | 56.286 | 0.0541 | 0.1047 | 0.0488 | 0.0564 |
| uniform prior, validation atoms | **57.030** | 0.0436 | 0.0844 | 0.0379 | 0.0424 |
| uniform prior, train-EM atoms | 57.020 | 0.0438 | 0.0844 | 0.0382 | 0.0426 |
| true atom | 57.708 | 0.0399 | 0.0518 | 0.0351 | 0.0321 |

Within each condition, even/odd episode halves of the uniform rule are 57.040 and 57.031 (v6: 56.447 and 56.153). Mean posterior mass on the true atom is 0.841. Mixture-argmax accuracy is 87.8% (674/768). That is the same predictive distribution as the published 57.03 rule: the four loss ratios match to four decimals, and P(true atom) matches. Short-history accuracy here is 74.0% and jackpot 78.9%, against P(true) of 0.659 and 0.730.

Train EM on lags 1..8 puts mass 0.346 / 0.322 / 0.321 on lags 1 / 2 / 4 and less than 0.01 on each other lag. L1 distance of the EM atoms from the validation clusters is 0.01–0.04 per type. Emissions are not the obstacle.

## Misses are unidentified lags, not a broken bit

Of 94 argmax misses, 82 have posterior confidence below 0.9. The confidence bins:

| confidence | n | accuracy | P(true) | mean wins | misses |
|---|---:|---:|---:|---:|---:|
| <0.6 | 81 | 0.506 | 0.495 | 26.6 | 40 |
| 0.6–0.75 | 90 | 0.633 | 0.546 | 36.0 | 33 |
| 0.75–0.9 | 99 | 0.909 | 0.769 | 41.2 | 9 |
| ≥0.9 | 498 | 0.976 | 0.965 | 96.6 | 12 |

Accuracy by win count is monotone: 0.625 (0–5 wins), 0.639 (6–15), 0.750 (16–30), 0.842 (31–60), 0.957 (61+). Misses average 37 wins, hits sit inside an overall mean of 75. By slice, short history is 0.740 accurate (50 of the 94 misses, mean 22 wins on those misses) and jackpot is 0.789 (56 misses, mean 31 wins). Stable, drift and shared-change are 0.917–0.932. Steady is 0.952.

On every miss, at least one of lags 1, 2, 4 is observed at the query, so some `(L, s)` in the family predicts the true atom. The log-likelihood gap between the best such state and the best state that predicts the wrong atom has median **−0.60 nats** across all 94 misses, and 70% of the gaps are milder than −1 nat. No miss has a gap below −8. A polarity that was simply backwards would cost about 0.2 nats per win, tens of nats on a long history. The history of a typical miss does not rule the true atom out. The posterior is flat, and the query bit then tips a near-tie the wrong way.

The 12 confident misses (confidence ≥ 0.9, 7 of them jackpot) are the only place a rule failure could hide. Their median gap is −2.81 nats (mean −3.05, none below −8) on a median of 60 wins. Refitting on the second half of each machine's wins predicts the query atom on **0/12**; the last 12 wins predict it on 4/12. On confident hits the same late half is correct 96% of the time. So these 12 are not a late switch back to the truth. They are machines whose recent wins and the query disagree by a few nats, which is what a finite sample looks like when the two jackpot atoms are close. Query bits at lags 1, 2 and 4 agree on 7/12 of them and disagree on 5/12, the same split as on confident hits (50% / 50%). The payout at the MAP lag is 0 on 7/12, and the `≥5` bit differs from the `≥2` bit on only 2/12 (16%, against 16% on confident hits). The win bit differs on 1/12.

An alternative feature was counted as a rescue when its best polarity matched the true atom and its likelihood was within 1 nat of the best `≥2` state. Some alternative cleared that bar on 10/12 confident misses. The same bar was cleared on 483/486 confident hits. Other lags, other partitions, `≥5`, the win bit, two-lag Boolean functions and the live bit all fit machines the baseline already gets right. Nothing in that catalog is specific to the misses.

## Richer rules lose on train likelihood

Emissions were held at the validation atoms so the comparison is the feature, not the fit. ΔLL is train marginal likelihood against the three-lag `≥2` mixture. Validation accuracy is the mixture argmax against the exact atom.

| family | train ΔLL | val accuracy | contest points |
|---|---:|---:|---:|
| `payout ≥ 2` at lags 1, 2, 4 | 0 | 0.878 | 57.030 |
| `≥ 2` AND-NOT, pairs inside {1,2,4} | −374 | 0.816 | — |
| `≥ 2` OR, same pairs | −1034 | 0.757 | — |
| `≥ 2` AND, same pairs | −1075 | 0.745 | — |
| `≥ 5` at lags 1, 2, 4 | −1304 | 0.720 | 54.501 |
| best single two-lag Boolean (AND of lags 1 and 2) | −1385 | 0.708 | 54.415 (EM atoms) |
| `≥ 2` XOR, pairs inside {1,2,4} | −1518 | 0.707 | 54.902 (EM atoms) |
| payout = 20 at lags 1, 2, 4 | −1651 | 0.686 | — |
| `≥ 2` at the pair's member, XOR the live bit | −1713 | 0.540 | 53.563 |
| win/loss bit at lags 1, 2, 4 | −1778 | 0.668 | 54.297 |
| live bit = win XOR win on the machine's pair | −2080 | 0.517 | 53.407 |

Every one of the 16 binary partitions of `{0,1,2,5,20}` was scored as a single lag feature (payout 0 fixed at bit 0, so polarity cannot rename the partition). The best is mask 14, which is `payout ≥ 2`, and the best single lag is still 1,399 nats behind the three-lag mixture. Mask 6 (`{2,5}` versus the rest) is the runner-up and is worse. A sticky HMM on the six `(L, s)` states, leave probability per tick, peaks at rate 0; rate `1e-4` is already −0.2 nats and rate 0.01 is −260. A polarity-only chain inside each lag also peaks at rate 0 (−31 nats at rate 0.001). The atom does not drift relative to the bit.

`s` is not the win-rule direction. On 271 train machines with a sharp size posterior and a sharp pair, direction 0 splits 84 / 65 and direction 1 splits 60 / 62. Signal and machine index are flat as well. Adding direction to the prior key loses likelihood at every prior strength.

## The lag is the {1, 2, 4} member of the win pair

Win pairs rediscovered on train, by the likelihood gain of a two-rate model on `win(t−a) XOR win(t−b)`, are the same three the win model already uses:

| pair | train machines |
|---|---:|
| (2, 5) | 218 |
| (1, 3) | 201 |
| (4, 7) | 200 |
| next, (1, 2) | 38 |

On machines where both the size posterior and the pair are sharp, the size lag matches the member of the pair that sits in `{1,2,4}`:

| pair | L=1 | L=2 | L=4 | agree |
|---|---:|---:|---:|---|
| (1, 3) train | 90 | 6 | 3 | 90/99 |
| (2, 5) train | 7 | 83 | 5 | 83/95 |
| (4, 7) train | 3 | 6 | 68 | 68/77 |
| validation, same cut | — | — | — | 251/291 = 86.3% |

The scoring prior does not hard-code that table. It is the average of the train posteriors given the machine's best pair among those three, with a 0.5 pseudocount on each of the six states. Uncertain machines dilute it to about 0.65 on the matching lag and 0.16–0.21 on each other lag. λ in `P = (1−λ)/6 + λ P_train` was chosen by even/odd train cross-fit. λ = 1, keyed by pair only, wins the grid at **+242 nats** over the uniform prior. Keying by pair and direction is worse at every λ.

A hard assignment (all mass on the modal lag of that prior, polarity free) scores **+259 nats** on the same cross-fit, and then fails the contest check. With train-EM atoms it scores 57.057. The episode-paired squared-error change against the uniform EM rule is +0.00008 ± 0.00044, and the interval covers zero. The even half goes 57.032 → 56.920. A wrong pair becomes a confident wrong lag, which is cheap in the likelihood of past wins and expensive in squared error at the query.

The soft λ = 1 prior keeps the other lags alive. Contest scores, same v6 `p_0`:

| model | points | stable | short | drift | shared | even | odd |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform, validation atoms | 57.030 | 0.0436 | 0.0844 | 0.0379 | 0.0424 | 57.040 | 57.031 |
| **pair prior, validation atoms** | **57.208** | 0.0418 | 0.0754 | 0.0367 | 0.0414 | **57.182** | **57.238** |
| uniform, train-EM atoms | 57.020 | 0.0438 | 0.0844 | 0.0382 | 0.0426 | 57.032 | 57.020 |
| **pair prior, train-EM atoms** | **57.201** | 0.0420 | 0.0754 | 0.0369 | 0.0415 | **57.176** | **57.231** |

All four conditions move down in loss ratio. The short-history ratio does the real work, 0.0844 → 0.0754. Argmax accuracy goes 0.878 → 0.897 (validation atoms): short 0.740 → 0.755, jackpot 0.789 → 0.800, steady 0.952 → 0.974, stable 0.917 → 0.953, drift 0.922 → 0.958. Shared-change argmax dips, 0.932 → 0.922, while its loss ratio still falls, 0.0424 → 0.0414. The prior corrects 38 uniform misses and breaks 23 uniform hits.

Paired by episode, uniform minus this prior, validation atoms: mean squared-error drop 0.00060, standard error 0.00017, better on 66 of 96 episodes, 95% interval [0.00026, 0.00093]. Train-EM atoms: drop 0.00061, standard error 0.00017, better on 73 of 96, interval [0.00028, 0.00094]. The half-to-half level of v6 itself moves by about 0.3 points; the paired change is estimated off the difference inside each episode, and that interval does not cover zero.

On short histories the uniform MAP lag matches the pair's member only 56% of the time (107/192). The prior is informative there precisely because the size likelihood is not. It does not finish the job. The oracle with the true atom is still 57.708, so 0.50 points remain, and short history is still the worst ratio (0.075 against an oracle 0.052). Jackpot squared error barely moves (0.01161 → 0.01147) because most of that error is v6's win probability, which this model does not touch.

## Recommendation

Ship the three-lag `≥ 2` mixture with the train-estimated prior `P(L, s | win pair)` at λ = 1, and with atoms fit by train EM. Keep v6's `p_0`. That is the round-two version: pairs, prior and atoms all come from train, λ was chosen by train cross-fit, and it scores **57.201** against the uniform rule's 57.020 and the published 57.030. The same prior with validation atoms scores 57.208, which is the like-for-like gain, **+0.18**. Both episode halves move up.

Leave the bit alone. `≥ 5`, the win bit, two-lag Boolean functions, the live bit, and a sticky HMM on `(L, s)` are all worse on train likelihood, and the ones that were scored land between 53.4 and 55.2. Do not hard-assign the lag. The hard rule wins the train likelihood and loses one validation half.

What is left is still identification on short histories and the jackpot win probability. The size bit is not where those points are.

## Code

`analyze.py` loads the public train and validation logs, fits the EM atoms, decomposes the misses, screens the rule families, cross-fits the prior, and writes `results.json` plus `predictions/`. The scored prior is `predictions/prior_pair_l1.0.csv` (train-EM atoms) and `predictions/prior_pair_l1.0_targetE.csv` (validation atoms, same prior).
