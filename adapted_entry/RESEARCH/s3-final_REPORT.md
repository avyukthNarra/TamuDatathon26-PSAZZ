# Final blend

Equal log-odds blend of three win models, with v13's size split and a 1e-4 floor. Built from the s2 entry. The exact filter and the short-history patch are the s1-adapt members, which learn lag pairs from train. The jackpot patch from `jackpot/run_jackpot.py` on `origin/sai/master-test` runs inside the v13 member on that same learned support. There is no shipped list of main-round pairs.

## What changed

A machine votes for the pair `a < b ≤ 8` with the largest two-cell Bernoulli gain, and only if both cells have at least 5 ticks and the leader beats the runner-up by 2 nats. Pairs with at least 5% of train machines are kept. If none qualify, all 28 pairs stay eligible.

| Member | Role |
|---|---|
| v13 + short patch + jackpot patch | Win and size. Short histories use `grid_marginal` and the other fold's 2-means level centres. Other jackpot queries use the active four-cell rate of the MAP pair, shrunk toward that pattern's parity pool. Shrink strength is the train-label Brier choice among map, atom4, atom8, and atom16. |
| exact filter | Win only. Same pair vote. Level centres are the other fold's 2-means fit, and the lower centre is dead. |
| sysid | Unchanged. Train-only rule selection. It ignores the pair override. |

On main the vote is (1, 3), (2, 5), (4, 7), and the jackpot strength is atom16 (194 validation jackpots patched). The kits vote for their own pairs: disjoint (1, 6), (2, 4), (3, 8); four those three plus (5, 7); sharedlag (1, 6), (1, 8), (3, 6).

`PAIRS_OVERRIDE` in `members/hooks.py` still forces a list. Left as `None`, every patched member learns.

## Main cross-fit

Episodes sorted by id within condition, alternating. Official points out of 60.

| Forecast | Points | Halves A / B |
|---|---:|---|
| This blend | **57.961** | **57.97 / 57.95** |
| s2 blend, before jackpot | 57.909 | 57.98 / 57.84 |
| Published jackpot blend | 57.973 | 58.01 / 57.94 |

Ratios: stable 0.0259, short_history 0.0605, drift 0.0242, shared_change 0.0301.

Both halves clear 57.75, and the point is above 57.85. Wall time 247 seconds.

The 4-fold score from the docker suite is a different split (answers for the scored fold hidden): **57.8445 / 60**, 95% CI 57.485–58.192. Ratios: stable 0.0259, short 0.0603, drift 0.0293, shared 0.0336. Jackpot loss ratio on that stitch is 0.0875.

## Round-two kits (out of 40)

Same cross-fit protocol.

| | disjoint | four | sharedlag |
|---|---:|---:|---:|
| This blend | **38.193** | **37.103** | **37.401** |
| Halves A / B | 38.23 / 38.15 | 38.11 / 36.15 | 38.00 / 36.78 |
| v7c | 37.72 | 36.45 | 37.63 |
| Adapted exact filter alone | 38.389 | 37.605 | 37.525 |

The blend beats v7c on disjoint and four. On sharedlag it is 0.23 behind v7c and 0.12 behind the exact filter alone. Short histories are the weak condition there (loss ratio 0.121). Drift is the weak condition on four (0.114). v13's own mixer is still the main-round ensemble; only the short patch, the jackpot patch, and the exact filter use the learned pairs, so the equal average can sit under the exact filter on a shifted kit.

## Runs

| Run | Result |
|---|---|
| Test submission, validation targets used as priors | `submission.csv`, 1536 rows, `check_submission.py --split test` `VALID FORMAT`. SHA-256 `48cbca5d8410c0aa70ea748d147704c2af873fa655f1c6544345391babc8a590`. 224 seconds. |
| No validation directory | `runs/noval.csv`, 1536 rows, valid format. Exact filter kept the train level prior. Short filter used train cell rates. v13 used train frequencies. Max absolute difference versus the submission is 0.063. 165 seconds. |
| Fresh process | The zip was unzipped to `runs/fresh` and rerun. Max absolute difference versus `submission.csv` is 0. `entry.zip` (79 files) was refreshed after that rerun with a comment-only edit in `members/hooks.py`. |
| Docker quick suite | **PASS WITH WARNINGS.** 0 requirement failures, 2 warnings, 37 passes. Kit-test exit 0 in **188.61 seconds** (10.5% of the 1800 second limit). Determinism passed. |

The two warnings are the future-tick probe. v13's history index rejects a tick at or after the cutoff and that member exits 1. The suite records the refusal as a warning. Legal kit files do not contain those rows.

## Risks

- Sharedlag is the kit where this blend does not beat v7c. The short-history slice is the reason.
- Jackpot strength is chosen on train labels, not by a validation fold. On main that choice is atom16 and the cross-fit still lands on 57.961.
- Without validation targets the level priors fall back to train. That file is valid and is not the 57.96 model.
- A history row at the cutoff makes v13 abort. The future-tick probe is the warning above.
