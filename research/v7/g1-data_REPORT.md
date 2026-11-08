# Generative process: what the histories actually are

Read-only forensics on main-round train (96 episodes × 8 machines) and validation targets. Long histories are complete from tick 0 through cutoff−1 (0 gaps). Cutoffs run from 192 to 256 (mean 223). Each condition is exactly 24/96 episodes in both train and validation. Short histories are 64 ticks (40 trailing + 24 older); that is an observation mask, and the kernel below is the same one (validation size mixtures on short-history queries are the same two atoms).

Code: `analyze.py`, `analyze2.py`. Numbers below come from those runs.

A win bit is `payout > 0`. Payout 1 is a win for the parity rule. See §2.

## 1. Win rule

On each stable machine, every lag pair `(a, b)` with `1 ≤ a < b ≤ 8` was scored by the binomial log-likelihood gain of two cells (lags agree vs lags differ) over a single rate. The outcome in that score is also `payout > 0`.

| Feature used on the lags | Machines scored | Mean live−dead gap | Mean LL gain (nats) | Share of pairs in {(1,3),(2,5),(4,7)} |
| --- | ---: | ---: | ---: | ---: |
| `payout > 0` | 182 | 0.728 | 68.5 | 0.989 |
| `payout ≥ 2` (1 counts as a loss) | 181 | 0.334 | 14.1 | 0.680 |
| `payout ≥ 5` | 168 | 0.279 | 5.2 | 0.310 |

Pushes are common: 24,788 of 53,214 positive long-history spins are exactly 1 (46.6%). Recoding them as losses destroys the rule. The parity is on `payout > 0`.

### Which pair, which direction

Of 192 stable machines, 182 have both cells occupied at least 10 times. Their winning pairs:

| Pair | Machines | Share |
| --- | ---: | ---: |
| (1, 3) | 66 | 0.363 |
| (2, 5) | 59 | 0.324 |
| (4, 7) | 55 | 0.302 |
| (1, 4), (3, 8) | 1 each | 0.011 |

The two outsiders are jackpots with LL gain ≈ 9, against a mean gain of 68. Their second-best pair is several nats worse but also weak (separation 0.39). They are unidentified jackpots, not a fourth rule. 23 of 24 stable episodes use all three canonical pairs across their eight machines (one episode has four, because of an outsider). That matches independent draws: with 8 machines and 3 equally likely pairs, the expected number of distinct pairs is about 2.9.

Direction is a per-machine coin flip, not a property of the pair or the type.

| Direction (live cell) | Machines |
| --- | ---: |
| XNOR: live when the two lag-wins agree | 97 (0.533) |
| XOR: live when they differ | 85 (0.467) |

The median gap from the best pair to the second pair is 54 nats. Only 2/182 machines have a gap under 2 nats. One pair is enough.

### Parity, not a 4-cell table, not a third lag

Inside the winning pair, the two “agree” cells (loss/loss vs win/win) differ by 0.076 on average, and the two “differ” cells differ by 0.065. A rate estimated on ~50 spins has a standard error near 0.06, so that is sampling noise. The 4-cell table beats the 2-cell parity by 1.57 nats per machine on average; two extra free rates are worth about 1 nat under the null. Parity is the model.

XOR of three lags is worse than the best pair by 36 nats per machine. The best extra lag, searched over the six lags outside the pair, adds 2.7 nats and a 0.14 rate gap; that is what the search finds under noise. One machine in 182 has a triple that beats its pair by more than 2 nats. No machine needs a 3-lag rule.

The 10 unscored machines are all jackpots with 0–5 wins in ~200 ticks (three of them never win). A live cell at probability 0.3–0.5 cannot hide in a series that short of wins, but a jackpot whose realized rate is ~0.01 produces only a handful of wins, which is not enough to pick a pair. They are the same rule with a small sample, not a no-pair class.

### How clean the two rates are

Mean live and dead rates on train stable histories (sampling noise still in the rows):

| Type | n | Live mean (sd) | Dead mean (sd) | Mean gap | Worst gap |
| --- | ---: | --- | --- | ---: | ---: |
| steady | 72 | 0.938 (0.041) | 0.153 (0.086) | 0.785 | 0.422 |
| balanced | 63 | 0.828 (0.069) | 0.046 (0.029) | 0.783 | 0.623 |
| jackpot | 47 | 0.584 (0.127) | 0.016 (0.009) | 0.569 | 0.284 |

Jackpot’s smaller gap is the level, not a broken rule: dead sits near 0.01 and live near 0.5.

## 2. Levels are per machine, shifted by type and signal

Validation probability vectors are not a small discrete set. All 187 stable targets (the machines whose cell could be scored) have a distinct 5-vector at 1e-8 rounding. Within a discovered cell the win probabilities still spread.

Empirical `p_win = 1 − p_0` on stable validation, cell taken from that machine’s own pair and direction:

| Type | Cell | n | Median | Mean | Sd | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| steady | dead | 31 | 0.157 | 0.171 | 0.077 | 0.053 | 0.378 |
| steady | live | 38 | 0.943 | 0.946 | 0.028 | 0.878 | 0.985 |
| balanced | dead | 29 | 0.049 | 0.063 | 0.049 | 0.010 | 0.186 |
| balanced | live | 36 | 0.889 | 0.864 | 0.077 | 0.635 | 0.957 |
| jackpot | dead | 30 | 0.012 | 0.018 | 0.013 | 0.004 | 0.057 |
| jackpot | live | 23 | 0.502 | 0.518 | 0.178 | 0.269 | 0.855 |

Signal moves the center. Same targets, split by the observed signal:

| Type | Cell | signal −1 | signal 0 | signal +1 |
| --- | --- | --- | --- | --- |
| steady | dead | 0.146 (sd 0.059, n=12) | 0.148 (0.051, 11) | 0.241 (0.092, 8) |
| steady | live | 0.932 (0.027, 15) | 0.943 (0.031, 8) | 0.961 (0.021, 15) |
| balanced | dead | 0.034 (0.022, 11) | 0.040 (0.017, 8) | 0.115 (0.047, 10) |
| balanced | live | 0.782 (0.064, 10) | 0.879 (0.058, 11) | 0.901 (0.059, 15) |
| jackpot | dead | 0.012 (0.005, 10) | 0.018 (0.011, 9) | 0.025 (0.016, 11) |
| jackpot | live | 0.412 (0.138, 4) | 0.440 (0.117, 9) | 0.630 (0.185, 10) |

A logit regression on those targets, `logit(p) = a + b·signal + c·live`, fitted separately by type:

| Type | a | b per signal step | c for live | Residual sd (logit) | R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| steady | −1.627 | +0.318 | +4.638 | 0.52 | 0.956 |
| balanced | −2.964 | +0.598 | +4.850 | 0.56 | 0.955 |
| jackpot | −4.196 | +0.428 | +4.190 | 0.63 | 0.924 |

Fitted probabilities at the three signals (plug-in, no shock):

| Type | signal −1 dead / live | signal 0 dead / live | signal +1 dead / live |
| --- | --- | --- | --- |
| steady | 0.125 / 0.937 | 0.164 / 0.953 | 0.213 / 0.965 |
| balanced | 0.028 / 0.784 | 0.049 / 0.868 | 0.086 / 0.923 |
| jackpot | 0.010 / 0.393 | 0.015 / 0.499 | 0.023 / 0.604 |

After type, signal, and cell, a machine shock of about 0.5–0.6 logit remains. On the train histories the two cells of the same machine are positively correlated on the logit scale once signal is removed: steady 0.38, balanced 0.31, jackpot 0.07. The live−dead logit gap has mean 4.5–4.9 and sd 0.55–0.64. So the gap itself is random across machines; a single shared shock is not enough, and neither is one universal pair of levels.

Between-machine sd with sampling variance removed (train stable) agrees: steady live still has a true sd of about 0.02–0.03 inside a signal, and steady dead at signal +1 has a true sd of about 0.09. Jackpot dead is tight on the probability scale (validation sd 0.013) because the probability is near zero, not because the logit is fixed.

The pair-mixer point levels (steady 0.149/0.956, balanced 0.039/0.840, jackpot 0.013/0.669) match the steady and balanced centers roughly. Jackpot live at 0.669 does not: the cell-conditional median is 0.50, and the mean at signal 0 is 0.44. That point mass sits in the right tail of the jackpot live distribution.

## 3. Drift is a linear swap of the two levels, finished before the cutoff

Early window (ticks 8–70) and late window (last 70 ticks), same scoring:

- The late pair equals the early pair on 73% of drift machines. The misses are windows that straddle the ramp.
- When the pair matches, the direction has flipped on 99.2% of machines.
- A model that keeps the early levels forever has window-rate RMSE 0.36. A probability-linear blend has RMSE 0.153. A logit-linear blend has RMSE 0.154. Per-tick likelihood on 27,255 spins: the linear blend beats a single mid-ramp step by 3,000–6,500 nats and beats “no drift” by about 23,000 nats.

Orient each machine so its early-live cell starts at flip-fraction 0 and the early-dead level is 1. Mean fraction by ticks-before-cutoff (36-tick windows):

| Ticks before cutoff | 200 | 120 | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flip fraction | −0.02 | −0.02 | 0.02 | 0.10 | 0.23 | 0.42 | 0.62 | 0.77 | 0.89 | 0.97 | 0.97 |

The line `u = clip((t − (cutoff − 96)) / 62, 0, 1)` passes through that curve (0 at 96 ticks out, 1 at 34 ticks out). Per episode, the fitted start is `cutoff − k` with `k` mean 96 and sd 8.5 (range 86–128). `k` is almost uncorrelated with cutoff (0.13), while the absolute start tick correlates 0.91 with cutoff. The schedule is anchored to the cutoff, not to a fixed absolute tick and not to a fixed fraction of the episode.

At the validation cutoff the same normalization has mean 1.00 (sd 0.16, median 1.01) for all three types. The reversal is over before the query. The last ~30 ticks are the fully swapped rule.

## 4. Shared change is one abrupt direction flip, on every machine, at one tick

A joint likelihood puts one tick `c` on all machines of an episode, each keeping the pair it showed in the first 45% of the history, and allows both cells’ rates to differ on either side of `c`.

| | Shared change | Drift | Stable |
| --- | ---: | ---: | ---: |
| Episodes | 24 | 24 | 24 |
| Mean LL gain of the best split (nats per machine) | 60.6 | 55.1 | 3.2 |
| Median / p90 of that gain | 62 / 86 | 58 / 77 | 2.8 / 5.8 |
| Fraction of machines whose direction flips | 1.00 | 0.99 | 0.01 |
| Fraction whose pair changes | 0.02 | 0.14 | 0.09 |
| Mean within-episode sd of the per-machine tick | 3.4 | 14.9 | 49.9 |
| Episodes where all 8 machines flip | 24/24 | — | — |

The profile of the pre-change live cell, aligned to the joint tick (mean win rate across episodes):

| Offset (ticks) | −60 | −20 | −5 | 0 | +5 | +10 | +20 | +40 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Win rate | 0.82 | 0.81 | 0.82 | 0.52 | 0.24 | 0.13 | 0.10 | 0.09 |

Flat, then a one-bucket drop, then flat at the dead level. The +5 bucket is a 5-tick bin plus a tick estimate that jitters by a couple of spins. This is a step, not a ramp. Drift’s split gain is just as large because a ramp also wants a break, but the window curve in §3 is a ramp and the shared-change profile is not.

What flips is the direction. The pair stays (left-window counts: (1,3) 62, (2,5) 59, (4,7) 54; right window almost the same). Mean absolute change in the live rate is 0.070 and in the dead rate 0.045, in line with splitting ~200 spins in half. Size total-variation between the two sides averages 0.15, which is multinomial noise on a few dozen wins, and §6 shows the size atoms do not change.

The tick is shared but not a constant fraction. Across 24 joint estimates:

| | Mean | Sd | p10 | p25 | p50 | p75 | p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `c / cutoff` | 0.663 | 0.047 | 0.594 | 0.631 | 0.670 | 0.690 | 0.708 |
| Ticks before cutoff | 74.5 | ~13 | 61 | 66 | 74 | 78 | 94 |

A regression `c = −8.1 + 0.710·cutoff` leaves residual sd 11.5 ticks. Machine-to-machine jitter inside an episode is only ~3 ticks, so the residual is real: the change time varies by about 11 ticks around two-thirds of the way through the episode. It is not `round(0.65·cutoff)` exactly (that guess has mean error +5 ticks and mae 10).

## 5. Stable episodes have no changepoint

The same split search on stable episodes gains 3 nats, the machines do not agree on a tick (within-episode sd 50), and the direction does not flip (1%). Splitting each parity cell into an early half and a late half gives 337 z-statistics with mean −0.09 and sd 1.07; 4.5% have `|z| > 2` and 0.6% have `|z| > 3`, which is the standard normal tail. Rates are constant Bernoulli draws for the whole stable episode.

## 6. Payout size given a win: two exact mixtures, one per machine, constant in time

Conditional on a win and on `p_win > 0.02`, every validation target in every condition lands on one of two mixtures per type. Rounded to 1e-8 there are exactly two distinct vectors per type, and the within-cluster max deviation is 0. The same two vectors appear in stable, drift, shared-change, and short-history.

`P(payout = 1, 2, 5, 20 | win)`:

| Type | Mode | n (stable, p_win>0.02) | 1 | 2 | 5 | 20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| steady | small | 35 | 0.753989 | 0.196808 | 0.048218 | 0.000984 |
| steady | big | 34 | 0.455122 | 0.435902 | 0.106796 | 0.002180 |
| balanced | small | 27 | 0.499853 | 0.347928 | 0.130473 | 0.021746 |
| balanced | big | 33 | 0.214066 | 0.546737 | 0.205026 | 0.034171 |
| jackpot | small | 16 | 0.335146 | 0.244946 | 0.314931 | 0.104977 |
| jackpot | big | 16 | 0.120787 | 0.323921 | 0.416469 | 0.138823 |

Counts are consistent with a fair coin per machine (jackpot 16/16, steady 35/34, balanced 33/27 on this cut).

Nothing at lags 1–8 selects the mode. Within-type rms of the 4-vector is 0.060 (jackpot), 0.089 (balanced), 0.097 (steady). Grouping by the lag’s payout, by `payout>0`, by `payout≥2`, by `payout≥5`, by signal, by machine index, or by a neighbour’s win, leaves rms in the 0.082–0.088 range. The best single lag agrees with the mode on 64–71% of machines, barely above a coin flip, and there is no lag that is best in every type. Live vs dead cells both contain both modes (steady rms stays 0.093 on each side).

Within a train machine the mixture is frozen. Forcing every winning spin onto one of the two global modes, versus letting a lag toggle the mode, loses likelihood for every `payout>0` lag (lag 1 is the least bad, at −291 nats across ~176 machines). A free mixture weight beats the single mode by only 1.2 nats per machine, which is what sampling noise buys you when the truth is a vertex. On validation, the machine’s own history (nearest empirical mix) recovers the cutoff mode on 105/141 machines (0.745). The misses are small samples; the two modes do not alternate inside the episode.

So the lag-1/2/4 big-win story is not in the probabilities. An unconstrained size table at lag 1 does pick up ~2 nats per machine on train, but the targets are identical across machines that disagree on those lags. That train gain is dependence among the sampled spins, not a change in the conditional distribution.

Size is a static per-machine bit. A particle filter should put a 1/2 prior on the two atoms and update it from the machine’s observed winning payouts. It should not recompute the mode from the last few lags.

## 7. Neighbours do not move the win probability

After conditioning on the machine’s own parity cell, every other machine at every lag 1–8 was allowed to split both cells. There are 7 offsets × 8 lags = 56 candidates.

- The best candidate, chosen per machine, adds 4.5 nats on average and a 0.21 gap in win rate. That is the maximum of 56 tests.
- The typical candidate (the best offsets in a pooled ranking) adds about 1.2 nats. A null split of two cells is worth about 1 nat.
- Which offset wins is uniform: counts 13, 27, 26, 34, 25, 23, 31 across offsets 2, 1, 3, 4, 5, 6, 7. No neighbour index.
- On the targets, the median variance drop from adding a neighbour bit inside `(type, cell)` is 0.029. The best of 56 drops 0.21, with a mean absolute level shift of 0.05.

A real neighbour term would also make a single machine over-dispersed, because the neighbour bit fluctuates. Section 8 says it does not. Cross-machine lift to put in a filter: none. An upper bound, after the search, is a couple of probability points, and the data do not ask for it.

## 8. No hidden Markov state

Residual autocorrelation of `win − cell rate` on stable machines, averaged over machines:

| Lag | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mean residual correlation | 0.000 | 0.014 | 0.007 | 0.019 | 0.000 | −0.002 | 0.005 | −0.006 |

The early/late z-test in §5 (sd 1.07, correct tail) is the extra-binomial check. A Markov latent that moved the probability would thicken those tails and leave lag-1 correlation. Neither shows up. The only latent that survives is a machine-level draw: the two win levels, the direction, the pair, and the size mode. All four are constant on a stable episode.

## 9. Generative model

```
condition ~ uniform{stable, drift, shared_change, short_history}   # 24/96 each
# cutoff is observed, in 192..256. short_history only hides ticks.

for each machine i:
    pair      ~ uniform{(1,3), (2,5), (4,7)}
    direction ~ Bernoulli(0.5)          # 1 => agree is live at the start
    size_mode ~ Bernoulli(0.5)          # index into the two atoms of type_i
    (logit p_dead, logit p_live) ~ Normal_2(
        mean = (a[type] + b[type]*signal,
                a[type] + b[type]*signal + c[type]),
        sd = σ[type], correlation = ρ[type])
    # a,b,c,σ,ρ from the table in §2
    #   steady   a=-1.627 b=0.318 c=4.638 σ=0.52 ρ=0.38
    #   balanced a=-2.964 b=0.598 c=4.850 σ=0.56 ρ=0.31
    #   jackpot  a=-4.196 b=0.428 c=4.190 σ=0.63 ρ=0.07

if condition == shared_change:
    c_tick ~ centered near 0.66*cutoff, sd about 11 ticks
             # empirical: mean 0.663*cutoff, IQR of c/cutoff is 0.63–0.69
if condition == drift:
    u(t) = clip((t - (cutoff - 96)) / 62, 0, 1)   # shared by all 8 machines

for tick t = 0, 1, ... synchronously, using only outcomes before t:
    for each machine i with pair (a, b):
        if t < b: draw from the machine's marginal (no rule yet)
        else:
            agree = 1{ (y[i, t-a] > 0) == (y[i, t-b] > 0) }
            live  = agree if direction == 1 else 1 - agree
            if condition == drift:
                # early-live cell slides from p_live to p_dead
                p = (1-u)*p_live + u*p_dead if live else (1-u)*p_dead + u*p_live
            else if condition == shared_change and t >= c_tick:
                p = p_dead if live else p_live      # direction flipped
            else:
                p = p_live if live else p_dead
            W ~ Bernoulli(p)
            if W == 0: payout = 0
            else:      payout ~ Categorical(atom[type, size_mode]) on {1,2,5,20}
```

There is no neighbour input and no time-varying latent.

### Priors a particle filter can use

Static, per machine, updated from that machine’s history:

| Parameter | Prior |
| --- | --- |
| Pair | 1/3 each on (1, 3), (2, 5), (4, 7). Drop the other 25 pairs. |
| Direction | Bernoulli(0.5), independent of pair and type. |
| Levels | Bivariate logit-normal in the table above, given type and the observed signal. |
| Size mode | Bernoulli(0.5) on the two atoms in §6. Update from winning payouts only. The atoms are point masses, not a Dirichlet. |

Episode-level, shared by the eight machines:

| Hypothesis | Probability | What it does |
| --- | --- | --- |
| Stable | 0.25 | Direction and levels fixed. Per-tick hazard of a flip: 0. |
| Drift | 0.25 | Deterministic blend `u(t)` above. Not a jump. By the cutoff, `u = 1`. |
| Shared change | 0.25 | Exactly one jump. Hazard is the density of `c_tick`, about 0.036 per tick at the mode (`1/(11√(2π))`) and ~0 outside roughly 0.55–0.75 of the episode. Unconditionally (×0.25) the peak hazard is about 0.009 per tick. All eight machines flip together; pairs and levels do not. |
| Short history | 0.25 | Same kernel as stable. The missing ticks are unknown, not losses. |

Do not use a constant per-tick hazard. A stable series never jumps, a drift series never jumps either (it slides), and a shared-change series jumps once at a predictable region of the episode.

## 10. What to change in a filter, and what not to

Worth encoding:

- Win bit is `payout > 0`, including pushes.
- Only three pairs, one per machine, direction fair.
- Levels are a logit-normal around the type×signal×cell mean, with sd ~0.55. Jackpot live is centered near 0.50 at signal 0, not 0.67.
- Drift is the linear schedule in §3, tied to the known cutoff. It has already finished at query time, so the last ~30 ticks are a clean swapped rule; a 100-tick half-life still mixes in the old direction.
- Shared change is one common tick near `0.66 × cutoff` (sd 11). The pre-change half of a long memory is the wrong direction.
- Size is two exact atoms and a static coin flip. Lag features for size are not in the conditional distribution.

Not worth encoding:

- A third lag, a 4-cell interaction, or any pair outside the three.
- Neighbour outcomes. Best cross-machine lift after selection is noise.
- A hidden Markov state on top of the cell. Residual correlation is 0.00 at lag 1, and within-cell rates pass a binomial half-split.
- A discrete universal dead/live table with one number per type. The targets are all distinct; the machine shock is part of the process.

The static-rule Bayes filter that lost 6–12 points was missing the drift blend and the shared flip, not the within-regime likelihood. The within-regime likelihood is this simple.
