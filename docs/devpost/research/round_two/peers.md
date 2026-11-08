# Peer features vs the blend's win residual

**Ship one correction. Leave every other peer feature out.**

Use the other machines' win rate at lag 3 as a logit offset on `P(win)`, and keep the blend's size split. Cross-fit on even vs odd validation episodes:

| Correction | Fit even, score odd | Fit odd, score even |
|---|---:|---:|
| `sigmoid(logit(p) + b · peer_lag3)` | **+0.034** (38.593 → 38.627) | **+0.054** (38.609 → 38.663) |
| `sigmoid(logit(p) + a + b · peer_lag3)` | **+0.084** (38.593 → 38.677) | **+0.072** (38.609 → 38.681) |

Ship the second row. The residual changes sign: peers that just lost should pull `P(win)` down, and a lone slope on a 0–1 rate cannot do that. The two held-out fits agree (`a, b` = −0.217, +0.817 and −0.266, +0.770). Squared win error falls about 8% on both halves (−0.00032 on a base near 0.00385). A block permutation of episodes never matched that no-intercept win-error drop (0/200).

Do not ship a per-machine peer, a same-type mean, an 8-lag peer regression, or a `shared_change`-only level shift in addition to lag 3. Those either miss the other fold or lose points once lag 3 is available.

## Setup

`r = (1 − p_0_true) − (1 − p_0_blend)` on the 768 cross-fitted rows in `blend_val.csv`. A win in history is `payout > 0`. Every machine's history ends at `cutoff − 1` (768/768), and the lag-1..8 window has no missing ticks, so lag `L` is the spin at `cutoff − L`. The scored spin is the one at `cutoff`.

Peer mean at lag `L` is the mean win bit of the other seven machines. Same-type mean uses the other machines of the same `machine_type` (46 of 768 queries have none; those rows are left uncorrected). Episode rate is the mean win bit over all eight machines. Folds are even and odd episode ids, 48 episodes each (team halves A and B). Odd/even condition counts are 11/13 or 13/11.

The scorer matches the kit: within each condition, average the five-class sum of squares inside the episode, average episodes, divide by the same quantity on `validation/reference.csv`, then `points = 40 / (1 + mean of the four ratios)`. Targets score 40 and the reference scores 20. The blend scores **38.5879** overall, **38.6093** even, **38.5931** odd.

Size shares stay `p_k / (1 − p_0)` from the blend. Only `P(win)` moves.

## The leftover error is mostly P(win)

Mean residual is +0.0047 (blend `P(win)` 0.405 vs truth 0.410). RMSE is 0.062. The `p_0` term is 56% of the five-class squared error. Giving the blend the true `P(win)` and keeping its size split scores **39.631** (+1.043). Giving it the true size split and keeping its `P(win)` scores **38.927** (+0.339). A perfect win correction is worth about three times a perfect size correction.

`shared_change` is the biased cell: mean residual **+0.018**, against +0.0015 drift, −0.0031 short history, and +0.0021 stable.

## What predicts the residual

Out-of-fold linear `R²` is fit on one half and scored on the other. Correlations are Pearson, with an episode-bootstrap 95% interval on the pooled correlation.

| Feature | Corr even | Corr odd | OOS R² even→odd | OOS R² odd→even | Pooled corr (95% CI) |
|---|---:|---:|---:|---:|---|
| Peer mean lag 1 | +0.125 | +0.109 | +0.004 | +0.009 | +0.113 (+0.01, +0.22) |
| Peer mean lag 2 | +0.092 | +0.055 | +0.001 | +0.006 | +0.076 (−0.04, +0.19) |
| **Peer mean lag 3** | **+0.174** | **+0.225** | **+0.046** | **+0.026** | **+0.200 (+0.10, +0.28)** |
| Peer mean lag 4 | +0.181 | +0.060 | −0.014 | +0.016 | +0.122 (+0.01, +0.21) |
| Peer mean lag 5 | +0.042 | +0.169 | +0.011 | −0.014 | +0.107 (+0.00, +0.21) |
| Peer mean lag 6 | +0.122 | −0.007 | −0.051 | −0.004 | +0.038 (−0.05, +0.13) |
| Peer mean lag 7 | −0.005 | +0.158 | −0.004 | −0.038 | +0.070 (−0.06, +0.19) |
| Peer mean lag 8 | +0.017 | +0.119 | +0.001 | −0.015 | +0.066 (−0.02, +0.15) |
| Peer mean, lags 1–8 | +0.166 | +0.199 | +0.030 | +0.017 | +0.178 (+0.08, +0.27) |
| Episode rate, lags 1–8 | +0.167 | +0.187 | +0.025 | +0.018 | +0.173 (+0.08, +0.27) |
| Same-type, lags 1–8 | −0.010 | +0.121 | −0.005 | −0.025 | +0.051 (−0.03, +0.14) |
| `shared_change` indicator | +0.044 | +0.211 | +0.014 | −0.030 | +0.126 (+0.04, +0.22) |
| Own win, lag 3 | +0.002 | +0.022 | −0.003 | −0.003 | +0.012 (−0.06, +0.08) |

Lag 3 is the only single lag with positive out-of-fold `R²` on both halves. The probability-scale fit is `r ≈ −0.02 + 0.06 · peer_lag3` on both halves (slope +0.061 and +0.067). Own lag 3 is flat, so this is not leftover own history. Lag 4 and lag 6 look correlated on one half and fail the other.

Inside each condition the lag-3 correlation stays positive on both halves for drift (+0.25 / +0.13), shared_change (+0.16 / +0.17), and stable (+0.16 / +0.16). Short history is +0.01 / +0.24. Condition-by-condition regressions, fit on only 11 episodes, have negative out-of-fold `R²`. The pooled slope is the one that scores.

**Per-peer max |corr| is noise.** The largest full-sample correlation between `r` and one other machine's win bit is 0.142 (machine 4, lag 8). A block permutation of residuals across episodes has median max |corr| 0.148 and 95th percentile 0.196; 62% of nulls are at least as large as 0.142. Picking the machine on the training half and scoring the other half selects machine 3 lag 1, then machine 7 lag 7. Test correlations are +0.008 and −0.113. Test `R²` is −0.056 and −0.105. Points move **−0.058** and **−0.047**.

**Eight lags together overfit.** A no-intercept logit on all eight peer means changes points by **−0.188** and **−0.066**. With an intercept the same eight lags still lose **−0.098** and **−0.014**. Adding the eight own lags as well is worse (−0.172 / −0.130 with an intercept). Given own lags 1–8, the peer block raises squared win error by +0.00029 and +0.00018. Same-type lags 1–8 move points by −0.088 / −0.072.

## Points, for every feature that cut win error on both halves

The gate is the specified correction `sigmoid(logit(p) + b · feature)`, `b` fit by squared win error on one half. Negative `dMSE` is an improvement. The useful rows are lag 3 and, well behind it, the pre-specified recent-rate summaries.

| Feature | dMSE odd / even | Points odd / even | `b` |
|---|---:|---:|---|
| Peer mean lag 3 | −0.000134 / −0.000245 | **+0.034 / +0.054** | +0.35 / +0.25 |
| Peer mean lag 5 | −0.000088 / −0.000096 | +0.023 / +0.021 | +0.21 / +0.18 |
| Peer mean, lags 1–8 | −0.000009 / −0.000140 | +0.002 / +0.031 | +0.30 / +0.16 |
| Episode rate, lags 1–8 | −0.000003 / −0.000131 | +0.001 / +0.030 | +0.29 / +0.15 |
| Same-type lag 1 | −0.000037 / −0.000023 | +0.010 / +0.006 | +0.09 / +0.11 |
| `shared_change` | −0.000188 / −0.000057 | +0.051 / +0.015 | +0.17 / +0.26 |
| Peer-last-8 × shared_change | −0.000226 / −0.000094 | +0.062 / +0.024 | +0.46 / +0.55 |

Lag 5's intercept version fails the even half (−0.007), and its within-condition correlations flip sign. The last-8 raw correction barely moves one half (+0.002). Same-type lag 1 is a +0.01 wobble; with an intercept it goes negative.

The `shared_change` indicator is a real level bias, not a peer outcome. It lifts that cell only, by +0.051 and +0.015, and improves 9 and 5 episodes out of 48. Fitting the same shift with a free intercept on the other conditions does not clear both halves (−0.006 / +0.014).

Adding an intercept to lag 3 is what should ship. Held-out condition ratios, affine lag 3 minus the blend (negative is better):

| Test half | drift | shared_change | short_history | stable | Episodes with lower win MSE |
|---|---:|---:|---:|---:|---:|
| Odd | −0.0014 | −0.0061 | −0.0009 | −0.0007 | 21/48 |
| Even | −0.0051 | −0.0021 | +0.0009 | −0.0014 | 26/48 |

The gain sits in drift and shared_change. Short history on the even half is the one cell that gets slightly worse. About half the episodes improve; the score is the mean, and the mean goes up on both halves.

Inside shared_change alone, a peer-rate slope on top of a shared_change intercept adds about +0.01 points (affine last-8 peer +0.062 / +0.025 versus the intercept's +0.051 / +0.015). Out-of-fold `R²` inside that condition is negative (−0.024 / −0.029). That is a level shift with a small extra, not a second feature to stack on lag 3. A four-feature logit (lag 1, last-8, episode rate, shared_change) with an intercept scores only +0.037 / +0.026, below lag 3 alone.

## What to implement

For each query, `peer_lag3` = mean of `1{payout > 0}` over the other machines at tick `cutoff − 3`.

```
p' = sigmoid(logit(p_win) + a + b * peer_lag3)
p_0' = 1 - p'
p_k' = p' * p_k / (1 - p_0)    for k in {1, 2, 5, 20}
```

Fit `a` and `b` on labeled episodes that are not the ones being scored. On these two halves the fits were (−0.217, +0.817) and (−0.266, +0.770). The one-coefficient form, `b` only, is worth +0.034 and +0.054 if an intercept is refused; its `b` was +0.346 and +0.247.

Count on the worse affine half, about **+0.07 points**, against a ceiling of **+1.04** if `P(win)` were perfect. Lag 3 was the survivor of the pre-set screen of lags 1–8, and the no-intercept error drop for that feature was outside a 200-draw episode permutation. Lags with a similar one-half correlation did not survive the other half.
