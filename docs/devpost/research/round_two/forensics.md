# Round-two forensics

Adaptation validation targets are noise-free. The same screens on the main-round validation recover the known rules, so the deltas below are the round-two changes. Train histories (no targets) give the same lag sets.

## What changed

**Win lags.** Each machine is still live or dead by XOR/XNOR of its own win bits (`payout>0`) at one pair. The pair set moved.

| Split | Stable machines whose best rule is a 2-lag XOR | Pair counts |
|---|---|---|
| Main validation (control) | 185/192 | (2,5)=69, (4,7)=59, (1,3)=57 |
| Adaptation validation | 188/192 | (2,6)=67, (3,8)=67, (1,4)=53 |
| Adaptation train | 186/192 | (1,4)=75, (3,8)=60, (2,6)=50 |

New catalogue: **{(1,4), (2,6), (3,8)}** = `(k, 2k+2)` for `k=1,2,3`. All three types use all three pairs. Median likelihood margin of the winning pair over the next own-lag rule: 26.9 nats (main 27.4).

**Payout-size lags.** Given a win, the conditional distribution on `(1, 2, 5, 20)` is still an exact hard switch between two atoms, selected by `sign XOR [payout(t-L) >= 2]`. `L` moved from **{1, 2, 4}** to **{1, 2, 3}**.

Pinned-to-target winners on stable validation (lag, threshold):

- Main: (2, ≥2)=41, (4, ≥2)=38, (1, ≥2)=34. Median margin 1.13 nats (same weakness as before; keep a mixture over the three lags).
- Adaptation: (3, ≥2)=40, (2, ≥2)=34, (1, ≥2)=31. Median margin 1.12 nats.

The six atoms are the same floats as main (max abs deviation of any stable conditional from the nearest atom: `1.3e-14`). Order is `P(1), P(2), P(5), P(20) | win`.

| Type | Atom A | Atom B |
|---|---|---|
| steady | 0.753989389636, 0.196808488291, 0.048218079631, 0.000984042441 | 0.455122007628, 0.435902393897, 0.106796086505, 0.002179511969 |
| balanced | 0.499853108469, 0.347928272369, 0.130473102139, 0.021745517023 | 0.214066147146, 0.546736593290, 0.205026222484, 0.034171037081 |
| jackpot | 0.335146031901, 0.244946198773, 0.314930826994, 0.104976942331 | 0.120786730110, 0.323920678380, 0.416469443632, 0.138823147877 |

## What did not change

**One bit, two levels per machine.** On stable adaptation, the matched cell's historical win rate sits on the target (median abs error 0.027; the other cell is off by median 0.75). Every machine draws its own dead rate and live rate. There is one gap per type and no third mass. Exact target `p_win` on stable, split at that gap:

| Type | Dead mean ± sd (n) | Live mean ± sd (n) | Main dead / live means |
|---|---|---|---|
| steady | 0.156 ± 0.060 (25) | 0.936 ± 0.022 (27) | 0.172 / 0.946 |
| balanced | 0.044 ± 0.030 (31) | 0.827 ± 0.086 (42) | 0.064 / 0.861 |
| jackpot | 0.013 ± 0.010 (42) | 0.594 ± 0.152 (25) | 0.017 / 0.518 |

Signal still shifts both clouds up (adaptation live means, signal −1 / 0 / +1): steady 0.929 / 0.939 / 0.948, balanced 0.783 / 0.836 / 0.857, jackpot 0.533 / 0.557 / 0.683. Same pattern as main. Jackpot live's mean moved by about two standard errors; refit the mixture, do not add a component.

**No 3-lag rule, no AND/OR, no cross-machine term.** Adaptation stable winners outside 2-lag XOR: 3 single-lag, 1 triple. Own-pair loglik beats the best neighbor feature by median 63.8 nats (own better on 99.5% of machines). XOR of the winning pair with another machine's win bit adds median 0.0 nats (gain > 2 on 1.6% of machines).

**Drift** is still a swap of that machine's two levels over a short ramp in the back half of the episode, shared by the eight machines. On adaptation drift episodes, a logit blend fits best (mean history loglik −585, median |cutoff p − target| 0.027), then a hard flip (−611, 0.044), then a probability-linear blend (−629, 0.033). Onset at about 0.62 of the cutoff, duration about 32 ticks. Main ranks the same way (logit 0.029, linear 0.036, flip 0.039; onset 0.61, duration 32).

**Shared change** is still one hard polarity flip for all eight machines. Adaptation: flip mean loglik −540 vs logit −571 vs linear −599; median |p − target| 0.022, matching stable. Flip tick / cutoff median 0.63 (p10 0.55, p90 0.73). Main median 0.65 (0.59–0.74).

**Short history** is unchanged: the last 40 ticks are all present, plus exactly 24 older ticks. The older mask differs across the eight machines in every episode (24/24).

## Changes for the pair-mixer / blend

1. Win experts: drop pairs `(1,3), (2,5), (4,7)`. Use **`(1,4), (2,6), (3,8)`** only. A full `C(8,2)` XOR search on this round's train already returns those three.
2. Payout-size experts: lags **`{1, 2, 3}`**, threshold **`payout >= 2`**, per-machine sign. Keep a mixture; per-machine margins are ~1 nat. Paste the atoms above; they did not move.
3. Win prior: keep two components per type (machine-specific draws). Refit on this round. Steady dead is a bit lower (mean 0.156 vs 0.172). Leave signal as a mean shift inside the two clouds.
4. Leave out triple-lag, AND/OR, and other-machine features.
5. Keep the regime block. Drift: logit interpolation between the machine's two rates, ~32 ticks, starting near 60% of the episode. Shared change: one flip for the whole episode near 63% of the cutoff (roughly 55–73%). Short-history mask: 40 recent + 24 older, per machine.

## How this was measured

`fit_rules.py` screens own-lag rules (singles, pair XOR/AND/OR, triple XOR) by two-cell Bernoulli likelihood on the history, then checks the cutoff target. `fit_more.py` clusters payout atoms, pins the size lag to the exact target atom, compares neighbor features, and grids a shared flip vs a probability-linear blend vs a logit blend. History ticks are `0 .. cutoff-1`. The target is the distribution of the spin at `cutoff`.
