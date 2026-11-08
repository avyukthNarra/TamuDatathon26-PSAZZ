# e4 size flip

Base is v7c in `mainFiles` (57.37 points, halves 57.31 / 57.39). The size rule uses one `(L, s)` for the whole history: atom `s XOR 1{payout(t−L) ≥ 2}`, lags learned on train. The episode flip swaps win-cell counts at the joint MAP tick when the train gap gate fires. Shared-change size error is about half of total error (balanced 49%, steady 35%) against about 5% in stable episodes.

Hypothesis: at a shared change, `s` or `L` also flips at that tick.

**Not confirmed. Nothing was kept.** `mainFiles` is unchanged. The simulated kits were not re-run.

## How the split was fit

Atoms and lags come from `SizeRule.learn` on all train histories (lags `(1, 2, 4)`, static prior 0.193). The flip tick is the same detector as `episode_flip.py`, with train pairs `((1, 3), (2, 5), (4, 7))`. The train gap gate is 102.67 (width 270.42). Every train shared-change episode is above that gate (24/24).

For each machine, wins before the tick and wins after it each get their own posterior over `(L, s)`. A machine counts only when both sides have at least 6 wins. The log Bayes factor compares two equal-sized models on all of that machine's wins: one polarity the whole time, versus the opposite polarity from the tick onward. The mixture weight `q = P(s flips)` is the train MLE of that factor. Validation targets are scored only as a check, and they do not choose `q`.

Controls: drift at the detector tick and at the win-rate ramp (tick that maximizes the before/after win-rate gap, 20 observed ticks on each side); stable at the detector tick and at a hash-seeded random tick.

## Train histories

| split | episodes | machines scored | P(s differs) | P(L differs) | MAP s differs | MAP L differs | MAP s differs given same L | mean logBF(flip − const) | fraction logBF > 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shared, detector | 24 | 173 | 0.237 | 0.534 | 0.156 | 0.445 | 0.073 | −3.35 | 0.150 |
| drift, detector | 24 | 174 | 0.247 | 0.548 | 0.167 | 0.460 | 0.021 | −3.56 | 0.115 |
| drift, ramp | 24 | 166 | 0.313 | 0.587 | 0.265 | 0.548 | 0.080 | −2.68 | 0.199 |
| stable, detector | 24 | 61 | 0.429 | 0.638 | 0.328 | 0.525 | 0.138 | −0.93 | 0.295 |
| stable, random tick | 24 | 171 | 0.234 | 0.525 | 0.199 | 0.439 | 0.063 | −3.67 | 0.105 |

Shared-change matches the stable random-tick control. Polarity agrees on about 93% of machines whose MAP lag is the same on both sides (7.3% flip vs 6.3% on the random control). Lag disagreement is about 44% in both, which is the posterior spread, not a second regime. Mean logBF is about −3.4 nats: the flipped-polarity model is worse. Train MLE of `q` is **0** on the pooled shared-change machines and on each episode half (104 and 69 scored machines).

The stable detector row is not a counterexample. Only 61 of 192 machines have 6 wins on both sides of that tick, so the tick is usually near an edge. Its logBF is still negative.

## Validation targets (diagnosis only)

Conditional size Brier of the payout mix given a win, at the cutoff. `q = 0` is the current size rule. `q = 1` forces the polarity flip after the tick (detector tick on shared-change episodes that fire the gate; ramp on drift; random tick on stable).

| condition | n | q = 0 | q = 1 |
|---|---:|---:|---:|
| shared_change | 192 | 0.0056 | 0.0566 |
| drift | 192 | 0.0045 | 0.0496 |
| stable | 192 | 0.0047 | 0.0434 |

A hard flip is about ten times worse on the condition it was meant to fix. All 192 shared-change validation queries sit in episodes the train gate would swap, so this is not a “gate never fires” miss.

## Decision

`s` does not flip at the shared-change tick, and `L` does not move more than it does on a random stable split. A train-learned soft weight would sit at 0 and leave forecasts unchanged. A hard flip would spend the size error the hypothesis was trying to remove. No flag, no scorer run, no simulated-kit run.
