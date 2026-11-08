# Sequence models besides the particle filter

Scored on validation against `runs/combined-v6` (the pair mixer, 56.286 / 60). None of the 62 prototypes, and none of the pre-specified 80/20 or 50/50 blends with that mixer, beat it. The closest is a likelihood mixture of the same three lag pairs, at 54.51.

Code: `predictors.py`, `run_eval.py`. Full grid: `results.json`. Reproduce with `python3 run_eval.py` (about 8 minutes, stdlib only).

## Protocol

Each model walks every machine online: the forecast at tick t uses only ticks before t, and missing ticks stay missing. Win-bit models are turned into a 5-way distribution by keeping the v6 mixer's payout-size split and replacing only P(win). PPM predicts the five symbols itself; a second row swaps in the mixer size split so the win bit can be compared on its own.

The grid (depth, order, half-life, leaf prior, expert set) was fixed before looking at validation scores. Inside each family the pick is the lowest **train** outcome Brier: win-bit Brier for win models, 5-class Brier for PPM, averaged per machine then per episode. Validation points are reported for the whole grid.

Leaf probabilities are sequential Beta-mixture posterior means. The type prior is the same dead/live mixture v6 reads from validation targets (`win_prior_by_type_from_targets`, strength 10): steady dead 0.149 / live 0.956, balanced 0.039 / 0.840, jackpot 0.013 / 0.669, plus a weak middle. KT is Beta(1/2, 1/2). Half-life γ = 2^(−1/h) multiplies counts and log-evidence. A half-life of ∞ is no forgetting.

Blends are linear in probability, then renormalized. **b80** is 0.8 mixer + 0.2 candidate. **b50** is equal weight. Points are the contest score, 60 / (1 + R), R the mean of the four condition loss ratios. Paired diffs are episode error minus the mixer's episode error (positive means the candidate is worse). "Better" counts episodes with a strictly lower error. ± is the standard error of the paired mean.

A synthetic check: a deterministic XOR of lags 1 and 2. Depth-2 CTW tail Brier 9.3e-6, final P(win) 0.998 against truth 1. The pair mixture tail Brier is 4.7e-6. PPM on a constant-0 series gives P(0) = 0.987, and a cycle through all five symbols does not crash. The implementations can represent the rule when the context geometry matches it.

Re-scoring the v6 file reproduced 56.286 points, R = 0.06599. Reattaching the mixer's own win probability to its size split changes probabilities by at most 1e-16.

## Train lag pairs

On train histories the best parity pair per machine is (2,5) on 216 machines, (4,7) on 212, (1,3) on 150, and then a long tail (next is (2,7) on 26). The top three are 578 / 768 machines. `choose_pairs` keeps exactly those three, which is what v6 uses.

Short histories observe about 29% of the ticks in the span (train 0.283, validation 0.293). The other three conditions are complete.

## Baseline decomposition

Official episode-weighted error, and the win/size split (query-mean; size is the error left after replacing P(win) with the truth and keeping the forecast's size mix):

| condition | error | loss ratio | win part | size part |
|---|---:|---:|---:|---:|
| stable | 0.01062 | 0.0541 | 0.00788 | 0.00275 |
| short_history | 0.01857 | 0.1047 | 0.00904 | 0.00952 |
| drift | 0.00926 | 0.0488 | 0.00654 | 0.00272 |
| shared_change | 0.01158 | 0.0564 | 0.00648 | 0.00510 |

Query-mean squared error / win-bit squared error: balanced 0.01216 / 0.00391 (n=234), jackpot 0.01230 / 0.00810 (n=265), steady 0.01302 / 0.00437 (n=269). Jackpot's win-bit error is about twice the other types. On short history the size half of the error is as large as the win half. On stable and drift, size is already the smaller term.

## Headline validation

| model | train Brier | points | Δ | stable | short | drift | shared | better | b80 | b50 | sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v6 pair mixer |  | 56.286 | 0 | 0.0106 | 0.0186 | 0.0093 | 0.0116 |  |  |  |  |
| BMA, top-3 pairs, h=12/40/100, type | 0.1304 | 54.509 | −1.78 | 0.0193 | 0.0240 | 0.0145 | 0.0191 | 13/96 | 56.128 | 55.701 | 6 |
| stack, top-3, h=40, capped, type | 0.1425 | 49.442 | −6.84 | 0.0184 | 0.0232 | 0.0675 | 0.0563 | 7/96 | 55.968 | 54.392 | 2 |
| CTW depth 8, h=100, type (best val in family) | 0.1924 | 36.565 | −19.72 | 0.0711 | 0.1422 | 0.1459 | 0.1298 | 0/96 | 55.137 | 49.648 | 6 |
| CTW depth 8, h=∞, type, fixed-share 0.02 (train pick) | 0.1916 | 35.762 | −20.52 | 0.0765 | 0.1366 | 0.1551 | 0.1509 | 0/96 | 55.045 | 49.251 | 6 |
| PPM order 2, h=∞, own 5-way | 0.6102 | 27.595 | −28.69 | 0.2240 | 0.2457 | 0.2220 | 0.2058 | 0/96 | 54.084 | 44.722 | 1 |
| PPM order 2 win bit + mixer size | 0.2443 win | 29.229 | −27.06 | 0.2057 | 0.2134 | 0.2025 | 0.1836 | 0/96 | 54.265 | 45.696 | 0 |

The best blend in the whole grid is b80 of the top-3 BMA, at 56.128. That is still 0.16 points under the mixer. Its paired episode gap is +0.00058, better on 29/96 episodes. Every other blend is lower. The candidates err in the same direction as a weaker copy of the mixer, so averaging does not cancel mistakes.

## Context-tree weighting

Classic CTW, binary win/lose, suffix contexts, depth 0–8, half-lives ∞ / 40 / 100, KT or type prior, plus fixed-share α=0.02 on the leaf-vs-split odds at depth 8.

Depth is the only knob that moves the score, and it saturates far below the mixer:

| depth | best points in that row | train win-Brier of that row's best |
|---:|---:|---:|
| 0 | 31.23 (h=40, type) | 0.223 |
| 2 | 31.04 (h=40, KT) | 0.218 |
| 4 | 34.84 (h=40, type) | 0.200 |
| 8 | 36.56 (h=100, type) | 0.192 |

Depth 2 ties depth 0. The two biggest train pairs are (2,5) and (4,7); a suffix of length 2 never sees them. Depth 8 does see them, and picks up about 6 points over the base rate, then stops. A suffix tree can only reach lag 7 by splitting on lags 1–6 as well, so each leaf has a handful of counts out of ~200 spins. The XOR self-test passes because lags 1 and 2 are a contiguous suffix. The real rule is not.

Train picks fixed-share, depth 8, no forgetting, type prior (Brier 0.1916, 35.76 points). The validation-best CTW is 0.8 points higher and was not used for selection. The whole depth-8 band is 35.1–36.6. Fixed-share at α=0.02 does not help validation. The type prior beats KT by about 0.7 points at depth 8, half-life 100 (36.56 vs 35.80); it is not the reason the tree loses. Paired gaps are about +0.11 ± 0.006, and 0 of 96 episodes improve.

Per-condition win-part for the validation-best CTW: stable 0.068, short 0.133, drift 0.143, shared 0.125, against the mixer's 0.008 / 0.009 / 0.007 / 0.006. Size part matches the mixer exactly, because the size split was copied. Runtime is 5–6 seconds for train plus validation at depth 8.

## PPM on the 5-symbol alphabet

PPM-C, orders 2 / 4 / 8, half-lives ∞ / 40 / 100, exclusion, uniform order −1. Train and validation agree: order 2 with no forgetting is the least bad, at 27.60 points (5-class train Brier 0.610, win-bit Brier 0.244). Longer context and faster forgetting both hurt; the worst is order 8, half-life 40, at 20.36. Swapping in the mixer size split lifts order 2 from 27.60 to 29.23, so PPM's size model is also weak, and the win bit alone is still around the constant-prior floor (the type-prior mean, ignoring history, scores 29.33 once it is given the mixer size split).

A 5-way context of length 4 has 625 cells. Wins are sparse, and the parity rule is a function of two bits, not of the payout symbol. Escape to the uniform distribution throws away the dead/live shape. 0 of 96 episodes improve.

## Lag-subset Bayesian mixture

This is the non-suffix version of "which lags matter": one expert per parity (or full cell) of a lag subset, Beta-mixture leaves, weights equal to discounted marginal likelihood, and an expert sleeps when any of its lags is missing. Half-lives are separate experts in the same mixture.

Train and validation agree on the winner: the three train pairs, half-lives 12, 40 and 100, type prior. 54.51 points. Paired episode error +0.00672 ± 0.00115, better on 13/96. By condition the paired gaps are all positive: stable +0.0087 ± 0.0032 (1/24), short +0.0054 ± 0.0020 (6/24), drift +0.0053 ± 0.0017 (3/24), shared +0.0075 ± 0.0021 (3/24). Win-part roughly doubles the mixer's (stable 0.0166 vs 0.0079, short 0.0145 vs 0.0090). Jackpot win-bit SE goes from 0.0081 to 0.0132.

Ablations, all with the type prior unless noted:

| variant | train Brier | points | what changed |
|---|---:|---:|---|
| top-3, h=12/40/100 | 0.1304 | 54.51 | the pick |
| all pairs + singles, h=40 | 0.1478 | 51.01 | one memory, every pair |
| full 4-cell + parity, h=40 | 0.1484 | 51.11 | AND/OR cells as well as XOR |
| + neighbor lags ±1..3, h=40 | 0.1484 | 50.43 | other machines' win bits |
| + triple parities, h=40 | 0.1461 | 49.05 | XOR of three lags |
| all pairs + singles, h=12/40/100 | 0.1437 | 50.55 | extra pairs, three memories |
| all pairs + singles, h=40, KT | 0.1554 | 44.60 | flat Beta(1/2,1/2) leaves |
| all pairs + singles, h=∞ | 0.1776 | 34.06 | no forgetting |

Reading that table:

- The same three pairs the mixer uses are the right expert set. The other 25 pairs are a noise tail. Adding them, or adding triples, lowers the score.
- Four-cell experts tie parity experts. The real interaction is XOR/XNOR, which the two-cell parity table already represents.
- Neighbor win bits do not help. On this data, peers are not features of the next spin.
- The dead/live prior is worth about 6 points once the context is a real pair (51.01 vs 44.60). KT shrinks a nearly-deterministic cell toward 1/2.
- No forgetting matches stable (error 0.0089, paired −0.0018 ± 0.0019, 10/24, not distinguishable from noise) and then collapses on drift (error 0.316, loss ratio 1.67, worse than the starter's 0.190) and shared change (0.246). The old likelihood never releases a reversed rule.

The 1.8 point gap to the mixer, on the same three pairs and the same three half-lives, is the aggregator. This contest scores squared probability error. Exponential weights on recent squared error (what v6 does, eta 6) are the aggregating algorithm for that loss. Marginal-likelihood weights are the aggregating algorithm for log loss. Mixing half-lives by likelihood also leaves weight on the fast expert during stable stretches: the single half-life-40 all-pairs model has stable win-part 0.0073, close to the mixer's 0.0079, while the three-half-life likelihood mixture blows stable win-part out to 0.0166. Square-loss weights concentrate; the likelihood mixture hedges.

## Log-odds stack

Each pair contributes gate × (logit(cell) − logit(empty)), with the gate a discounted Bayes factor against the empty model and a 1/(n+1) inclusion prior. If the gates sum to more than 1 they are rescaled, so this can add two real interactions without double-counting a whole extra rule. It is the form the kit generator uses (sum of log-odds). It is worse than picking one pair.

Train pick: the three train pairs, half-life 40, capped, type prior, 49.44 points, paired +0.0289 ± 0.0029, better on 7/96. Drift win-part is 0.065 against the mixer's 0.007. Opening the gate prior to 1/2 scores 35.91. Dropping the cap, adding singles, using KT, or dropping forgetting all score between 33.7 and 48.2. Two stacked interactions are not where the remaining error is. One parity pair per machine, with a fast switch under square loss, already matches the data.

## Meta-learned sequence model (not built)

A small GRU can be shipped as a weight file and evaluated in pure Python. Hidden size 32 is on the order of 10^4 parameters. One test pass is about 192 episodes × 8 machines × 250 ticks × a few thousand flops, around 10^9 operations, which is a few tens of seconds to a couple of minutes in CPython and fits the 30-minute, 2-CPU, stdlib, deterministic box. Training would happen offline; it does not have to be stdlib.

The expected gain is about zero, which is why it was not trained. The data-generating structure that survives on the real logs is one XOR/XNOR pair per machine, drawn from three pairs, with a dead/live rate and occasional reversal. The pair mixer is already an online estimator aimed at that class and at square loss, and it sits at 56.3 / 60 with episode error about 0.012. A net trained on 768 real sequences has to rediscover that estimator and then beat it on calibration. Square loss punishes miscalibration directly. A transformer trained on the kit generator would fit the generator's base rates near one half, its neighbor term, and its hidden state; neighbor features scored 50.4 here against 51.0 for the same mixture without them, and the brief already records that the generator does not match the logs. Building it is feasible. It is not the highest-value use of the remaining time.

## Ranked recommendations

1. **Keep the v6 pair mixer as the win model.** Best alternative 54.51, best blend 56.13, mixer 56.29. The paired gap to the best alternative is about six standard errors (+0.0067 ± 0.0012).

2. **Do not replace square-loss expert weights with a Bayesian likelihood mixture.** Same pairs, same half-lives, same dead/live leaves, likelihood weights: −1.78 points. The contest loss is squared error, and the mixer's aggregator matches that loss.

3. **Do not ship suffix CTW or 5-symbol PPM.** Depth-8 CTW is a 6-point improvement over a Beta base rate and a 20-point loss to the mixer, with 0/96 episodes better. PPM order 2 is at the ignore-history floor once size is held fixed. The universal prior over contiguous contexts is the wrong prior for pairs at lags (2,5) and (4,7).

4. **Do not add triples, 4-cell AND/OR tables, or neighbor lags.** Each one raised train Brier and lost validation points relative to parity pairs. The long tail of train pairs (190 machines outside the top three) behaves like noise.

5. **Forgetting stays mandatory.** A no-forgetting pair mixture is tied with the mixer on stable and scores 0.32 error on drift. Half-lives 12/40/100 are doing real work; the open question is how the weights across those half-lives are updated, and square loss is the update that won.

6. **The next point, if there is one, is not another win-bit sequence model.** Short history is still the weak condition (ratio 0.105), and there the size half of the error (0.0095) is as large as the win half (0.0090). Jackpot's win-bit SE (0.0081) is about twice balanced or steady. Every sequence model here made both of those worse. A change to the size model under missing ticks is a better target than CTW, PPM, or a GRU.
