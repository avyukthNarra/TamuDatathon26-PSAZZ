# v7c as a gated fourth blend member

**Accepted.** v7c's win probability is a fourth log-odds member. Its weight is a train-only structure share. On main and on K1, K7, and K9 the weight is 0, so those forecasts are the a4 blend. On K10 the weight is 0.966 (24% of the log-odds sum) and the kit rises by 0.684 / 40. No other kit falls.

The base is the a4 blend: equal log-odds of v17-guards with the short and jackpot patches, the exact filter, and sysid. Size stays with v13. Every member still learns its lag pairs from that round's train.

## What the gate measures

On each machine's train history, even ticks and odd ticks take turns as the fit half. The fit half picks one lag-pair parity, one 3-lag parity, and one 2-lag AND (lags 1..8, add-one cell rates). The other half scores them. The gap is held-out log likelihood of the better alternative minus the pair.

Drift and shared-change series flip, so a 3-lag parity wins there on every kit, including main. The reference condition is the train condition with the lowest median gap. That is the calm regime, and on these rounds it is the condition named stable. It is chosen from the gaps, not by name.

The unexplained tail is the mean of the positive gaps on that condition, minus the same mean after each series is redrawn from its own fitted pair rule (one redraw, seed 0, real missingness kept). Dividing by the mean pair win on that condition (mean of the negative gaps, sign flipped) turns the tail into a share `f` of a full pair explanation. v7c's log-odds weight is `3f / (1-f)`, and the other three members stay at weight 1, so v7c's share of the sum is `f`. `f` is capped just below 1 by the machine count. Nothing in that map is a number copied from main's score or from a validation file.

Sai's Walsh screen (`orch/r2screen/walsh_screen.py`) is the same lag-product energy, summed over train, validation, and test. In-sample, triples beat pairs even on pair data, which that screen already saw on the public round. The holdout, and the pair-world redraw, are what keep the weight at 0 when the structure is clean.

## Train gate

| round | ref tail (nats) | pair-world tail | excess | pair-win scale | v7c share | v7c weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| main | 0.077 | 0.125 | 0 | 36.12 | 0 | **0** |
| K1 | 0.155 | 0.355 | 0 | 31.16 | 0 | **0** |
| K2 | 0.142 | 0.139 | 0.003 | 33.26 | 0.0001 | 0.0003 |
| K3 | 0.149 | 0.081 | 0.069 | 33.03 | 0.0021 | 0.0062 |
| K4 | 0.148 | 0.039 | 0.108 | 35.89 | 0.0030 | 0.0091 |
| K5 | 0.430 | 0.196 | 0.234 | 36.55 | 0.0064 | 0.0193 |
| K6 | 0.293 | 0.205 | 0.088 | 33.88 | 0.0026 | 0.0078 |
| K7 | 0.231 | 0.603 | 0 | 33.85 | 0 | **0** |
| K8 | 0.220 | 0.024 | 0.196 | 31.15 | 0.0063 | 0.0190 |
| K9 | 0.610 | 0.767 | 0 | 29.51 | 0 | **0** |
| K10 | 8.468 | 0.412 | 8.056 | 33.08 | 0.2435 | **0.9658** |

K10's calm-condition 90th percentile gap is +40 nats. Main's is −9 and K1's is −7, so on those rounds the worst tenth of calm machines still prefer a pair. K2–K6 and K8 open the gate by less than 0.02, which is under 1% of the log-odds sum.

## Before / after

Kit scores are `s3-final/score_pred.py` on the reblend of the a4 cross-fit member CSVs and the a3 v7c cross-fit CSV. Alphabetical episode halves, out of 40. Where the weight is 0 the line matches a4's `score.txt` (K1, K7, K9 checked to the condition ratios).

| kit | before | after | change | weight |
| --- | ---: | ---: | ---: | ---: |
| K1 main-like | 38.801 | 38.801 | 0 | 0 |
| K2 disjoint pairs | 38.962 | 38.962 | 0 | 0.0003 |
| K3 four pairs | 38.911 | 38.912 | +0.001 | 0.0062 |
| K4 shared lag | 39.264 | 39.264 | 0 | 0.0091 |
| K5 long lags | 39.195 | 39.197 | +0.002 | 0.0193 |
| K6 static size | 39.365 | 39.365 | 0 | 0.0078 |
| K7 size bit ≥ 5 | 39.098 | 39.098 | 0 | 0 |
| K8 late flip | 39.131 | 39.134 | +0.003 | 0.0190 |
| K9 shifted levels | 39.071 | 39.071 | 0 | 0 |
| K10 mixed 3-lag / AND | 32.080 | **32.764** | **+0.684** | 0.9658 |
| mean | 38.388 | 38.457 | +0.069 | |

K10 halves go from 32.02 / 32.38 to 32.83 / 32.94. v7c alone on that kit is 32.654, so the gated blend is ahead of both the old blend and v7c.

Main is weight 0, so it is the a4 forecast: **57.971**, halves **57.98 / 57.96**. Both halves stay above 57.85, and the total stays above 57.90.

## Patch

`entry.patch` applies in `a4-v17member/variant` with `patch -p1` (dry-run succeeded). It gives `blend.py` optional weights, adds `structure_health.py`, and teaches `predict.py` to compute the gate on `--data` and to mix `member-dir/v7c.csv` when the weight is positive. The scored table is that mix applied to the saved cross-fit files. A fresh `predict.py` still needs a v7c run to write `v7c.csv`; the three-member path is unchanged when the weight is 0 or the file is absent.
