# Do machines in an episode share a win-lag pair?

**Verdict: episode-level pooling of pair evidence is unjustified.** On both train and validation, machines that clear a 3-nat pair margin agree on the lag pair, and on XOR versus XNOR direction, at the rate implied by the pooled frequencies. Grouping by `machine_type` does the same. Estimate the pair and the direction on each machine's own history.

## Result

Agreement is the fraction of unordered machine pairs, inside an episode or inside a type, that carry the same label. The chance rate is \(\sum_k f_k^2\) from the label frequencies of the machines that enter those pairs. A one-sided permutation p-value shuffles labels across machines, keeps the groups, and counts shuffles whose agreement is at least as large as observed (4000 shuffles, plus the observed labeling, seed 7). The null band is the 2.5th–97.5th percentile of that shuffle distribution.

| Split | Label | Grouping | Agreeing pairs | Agreement | Chance \(\sum f^2\) | Excess | Null band | p (greater) |
|---|---|---|---:|---:|---:|---:|---|---:|
| Train | Lag pair | Episode | 425 / 1227 | 0.346 | 0.336 | +0.011 | 0.309–0.362 | 0.18 |
| Train | Lag pair | Machine type | 10292 / 30438 | 0.338 | 0.336 | +0.003 |  | 0.061 |
| Train | XOR direction | Episode | 635 / 1227 | 0.518 | 0.501 | +0.017 | 0.474–0.529 | 0.11 |
| Train | XOR direction | Machine type | 15362 / 30438 | 0.505 | 0.501 | +0.004 |  | 0.045 |
| Validation | Lag pair | Episode | 386 / 1160 | 0.333 | 0.338 | −0.005 | 0.310–0.366 | 0.60 |
| Validation | Lag pair | Machine type | 9212 / 27417 | 0.336 | 0.339 | −0.003 |  | 0.57 |
| Validation | XOR direction | Episode | 547 / 1160 | 0.472 | 0.500 | −0.028 | 0.473–0.529 | 0.98 |
| Validation | XOR direction | Machine type | 13675 / 27417 | 0.499 | 0.500 | −0.001 |  | 0.37 |

The unweighted mean of the per-episode rates tells the same story. Train pair 0.326 across 69 episodes (median 0.333); validation pair 0.335 across 64 episodes (median 0.321). Train direction 0.500; validation direction 0.475. Episodes in which every confident machine shares one pair: 0 of 69 on train, 1 of 64 on validation.

Train's pair excess of about one percentage point, and its direction excess of about two, both sit inside the null band. Validation pair agreement is a half point under chance. Validation direction agreement sits on the low edge of its null band (0.472 versus a lower end of 0.473). That dip is one split and one label; the train direction rate is slightly high. Nothing here is a shared episode factor large enough to pool on.

## Pooled frequencies

Confident machines, all three pairs in play:

| Split | Machines | (1, 4) | (2, 6) | (3, 8) | XOR (differ-live) | XNOR (match-live) |
|---|---:|---:|---:|---:|---:|---:|
| Train | 428 | 133 | 136 | 159 | 206 | 222 |
| Validation | 404 | 119 | 127 | 158 | 201 | 203 |

`(3, 8)` is the largest share on both splits (37% train, 39% validation) and the three pairs stay in a narrow band, so the chance agreement for the pair is about 1/3. Direction is split evenly, so its chance agreement is 1/2.

The same balance holds inside each type. Train pair counts by type: balanced 49 / 35 / 64, jackpot 38 / 52 / 40, steady 46 / 49 / 55, in the order `(1, 4)`, `(2, 6)`, `(3, 8)`. Validation: balanced 44 / 49 / 64, jackpot 45 / 37 / 48, steady 30 / 41 / 46. Within-type chance rates for the pair stay between 0.334 and 0.353. No type owns a pair.

Same-type and cross-type pairs inside an episode agree at the same rate. Train pair: same-type 143/410 = 0.349, cross-type 282/817 = 0.345. Validation pair: 134/390 = 0.344 and 252/770 = 0.327. Shuffling pair labels inside each type, so the type margins stay fixed, gives within-episode pair p-values 0.38 (train) and 0.43 (validation).

`machine_type` itself is mixed inside episodes at the chance rate: train 410/1227 = 0.334 versus \(\sum f^2 = 0.335\); validation 390/1160 = 0.336 versus 0.339. Episode agreement is an ordinary mix of types, and that mix does not carry the lag pair.

## Why the labels are real enough to trust a null

Each split has 768 machines in 96 episodes. History never reaches the query cutoff. The 192 machines with exactly 64 observed ticks are dropped by the 150-tick rule; those 64 ticks are scattered through the episode rather than a contiguous prefix. Of the 576 machines at or above 150 ticks, 428 (train) and 404 (validation) have a best pair at least 3 nats ahead of the second. Three train machines and two validation machines had an empty match cell or an empty differ cell on every pair, so they have no two-state likelihood.

Among machines with two scored pairs, the margin's median is 10.8 nats on train and 10.1 on validation. On the confident subset the match-cell and differ-cell win rates differ by a median of 0.41 (train) and 0.43 (validation); the 10th percentile is still 0.23. The two-state model beats a single rate by a median of 21 nats (train) and 22 nats (validation). No confident machine had equal cell rates, so every one has a direction. Replacing the MLE log-likelihood with Jeffreys smoothing, \(p = (k+0.5)/(n+1)\), keeps the same machines and the same winning pair on every one of them.

Raising the bar to a minimum cell of 20 and a margin of 10 nats, or to a cell of 30 and a margin of 20, leaves episode pair excess at +0.01 to +0.03 on train and +0.01 on validation, and leaves direction excess at or below zero on both splits once the cut is strict. The null is the measurement, not a wash of weak labels.

## Method

Win bit is `payout > 0`. For a candidate pair \((a, b)\) in \(\{(1,4),(2,6),(3,8)\}\) and a tick \(t\) where \(t\), \(t-a\), and \(t-b\) are all observed, the cell is match when the two lag wins are equal and differ when they are not. Each cell gets its own Bernoulli MLE. The pair's log-likelihood is the sum of the two cell log-likelihoods, with the convention \(0 \log 0 = 0\). The best pair is the maximum. A machine is confident when that maximum beats the second pair by at least 3 nats. Direction is XOR when the differ cell has the higher win rate and XNOR when the match cell does.

Train and validation are fit and scored separately. Query outcomes and `targets.csv` are unused. The chance rate uses the machines inside the groups being compared; on train that is all 428 confident machines, and on validation it is 401 of 404 (three confident machines sit in an episode with no second confident machine). The finite-population rate \(\sum_k n_k(n_k-1)/(N(N-1))\) is 0.002 below \(\sum f^2\) and does not move the comparison.

## Decision

Keep lag-pair evidence and XOR/XNOR direction on the machine that produced them. An episode pool and a type pool both reproduce the marginal frequencies and add no agreement beyond that.
