# Pranav: research and building

Drafted by Sai's agent from Pranav's 51 commits (branches `pranav/*`, final `pranav/super`). Scores are main-round validation /60
unless marked /40. Reviewed by Pranav; reasoning added under *Key ideas and reasoning*.

## Path
| When | Step | Score | Commit |
|---|---|---|---|
| Sat 12:48 | Lag-feature forecaster trained on every past spin | | `e2f7884` |
| Sat 13:31 | Lag-pair mixer with a verified entry | 53.35 | `aed7329` |
| Sat 13:37 | Tuned at cutoffs: two memories, size experts | 53.75 | `98493d0` |
| Sat 13:39 | Rediscovers lag pairs and rule type from each round's train | | `77c320b` |
| Sat 14:41 | Payout-size experts keyed on exact earlier payouts | 54.21 | `55afa43` |
| Sat 16:31 | Payout size as a two-mode switch | 54.90 | `b57b8fd` |
| Sat 16:52 | Win cells pulled toward dead/live levels (Beta-mixture prior) | 55.34 | `39e5757` |
| Sat 18:33 | Per-type win prior and learned size lags, honest eval | 56.12 | `27d776d` |
| Sat 19:12 | 100-spin win memory, slower size re-weighting (mixer6) | 56.19 | `814cb60` |
| Sat 19:33-19:58 | Per-type middle level; levels per type and signal | 56.29 → 56.32 | `8ab8586`, `fb5636a` |
| Sat 20:18 | Hidden-rule HMM win chance blended in | 56.51 | `59ad70b` |
| Sat 23:12 | Research harness and log-odds pooled ensemble | 56.54 | `d8ff0a2` |
| Sat 23:55 | Peer correction found by residual analysis | 56.59 | `52bc7d7` |
| Sun 00:31 | v11 plateau average of three near-equivalent win models | 56.61 | `b83b1a6` |
| Sun 00:59 | **v13 strongest build**: win model + size-rule HMM + v7c | 57.52 | `a70abc0` |
| Sun 01:32 | v15: signal term on the blended win chance | 57.55 | `be6db30` |
| Sun 02:07 | v16: v7c's episode flip swap moved into the win model | 57.59 | `7e1c1fc` |
| Sun 02:22 | v17: jackpot sub-cell correction (super branch) | 57.72 | `fd65593` |
| Sun 03:49 | v18: round-adaptive add-on gates, v7c round-two guards | 57.72 | `ee570b1` |
| Sun 04:34 | v19: 81 s entry, build-time validation gates | 57.74 | `15241ec` |
| Sun 05:39 | v20 super entry: Sai's blend + v19 as a 4th member | 58.00 | `0d45f25` |
| Sun 11:07 | Round two: rediscover sub-cell lag pairs per round | | `7b89c64` |
| Sun 11:40 | v21 round-two entry: peer term on the super blend (halves 38.38 / 38.87) | 38.67 / 40 | `7968b04` |

v13 and its successors are the base of the final blend's first member (v17-round2-guards).

## Key ideas and reasoning
- **Win chance is a lag-pair rule.** Residuals showed a machine's own result a fixed number of spins back mattered far more than anything else, and the win chance depends on whether its results at two lags MATCH or DIFFER ((1,3), (2,5), (4,7) on main). The pair mixer learns a match/differ cell rate per pair, weights the pairs by how well each has predicted lately (three memory lengths), and rediscovers the pairs and size lags from each round's train, so round two (pairs (1,4), (2,6), (3,8); size lags 1, 2, 3) needed no code change.
- **Look at what the truth can be before modelling it.** Payout size given a win is exactly one of two mixes per machine type, and the mix switches spin by spin depending on whether a recent payout was big, so P(big mode) per lag cell by Bayes' rule (54.21 → 54.90) beat free size cells. Likewise true win chances are U-shaped (most near ~0.04 or ~0.90), so smoothing cells toward the ~0.5 base rate pulled them the wrong way; a Beta-mixture dead/live prior fixed that (→ 55.34), and per-type levels (jackpot is almost never live) added more (→ 56.12).
- **Rule switches and the episode.** Drift episodes swap a machine's rule, so a hidden-rule HMM (Zhill's idea, our levels) pooled with the mixer in log-odds handles switches (56.51). A residual hunt then found that machines win more when the episode's other machines just won (peer correction, 56.59); neighbour-machine "links" turned out to be small-sample noise at real overlap lengths (~40 spins).
- **Combine with teammates where they are strong.** v13 took payout sizes from Zhill's size-rule HMM and blended Avyukth's v7c (0.7 / 0.3 in log-odds) for 57.52; v14-v17 put back what the blend diluted (the peer term v7c ignores, the signal term, v7c's episode flip swap moved into our win model, and Sai's jackpot sub-cell finding) for 57.72. v20 then made v19 the 4th equal member of Sai's blend (58.00, halves 58.04 / 57.92).
- **Honest evaluation decides.** Validation gains of a few hundredths flipped to losses out of fold twice, so every change was scored on cross-fitted validation halves (priors from the other half) AND out-of-fold train (episode folds, cutoff plus the last 10 history ticks, ~4x tighter intervals), and kept only if it won on both. That rule rejected the regime member, learned stackers and per-lag weights, which all looked better on validation alone.
- **What ships must reproduce.** Each runner zips the entry, reruns it in a fresh process (must match to 1e-8) and read-only without validation answers. This caught two real flaws: v18's run-time gates made the forecast depend on whether answers were mounted (v19 decides gates at build time and saves them), and Sai's members changed by up to 0.047 without answers (v20 bundles the public validation folder, so output is identical with, without, or with no validation folder).
- **Round two.** The jackpot sub-cell still used main-round pairs; it now rediscovers them per round. None of Sai's three members look at other machines, so v21 adds the peer term to the final blend (0.75 x peers' last-5-tick win share minus the split mean). The effect shows clearly in the scored test split's own history (coefficient 1.14 [0.65, 1.58]), less so in validation (0.54) and train (0.27), and it raises both cross-fit halves over v20 (38.34 / 38.81 → 38.38 / 38.87; full 38.62 → 38.67 / 40). Against the team rules blend (38.53 / 38.63) it has the higher mean but not the higher first half, so the team blend stayed the upload.

## Tried, not adopted (from commit messages)
Population priors for expert weights; per-machine residual hunts and surprise tempering; per-lag and short-history peer refinements;
wider residual families, joint pool/peer grid, middle-level HMM; post-v15 short-history gates, regime member, size-flip detection;
per-type blend weights; rejected rule-model blend; round-7 jackpot and size attempts. Also: learned stackers, leave-one-out expert
scoring, gap decay, coarse / pooled and shrunk size cells, hedging sizes toward 50/50 in flip episodes, continuous add-on scales,
HMM-off-on-flip, the sub-cell correction on all types, and in round two episode-shared lag pairs (agreement 0.52 vs ~0.49 by chance)
and a signal term on the blend (lowered one half).

## Final numbers
| Entry | Round | Validation | Cross-fit halves |
|---|---|---|---|
| v19 (stdlib) | main /60 | 57.74 | 57.74 / 57.69 |
| v20 (Sai's blend + v19) | main /60 | 58.00 | 58.04 / 57.92 |
| v20 | round two /40 | 38.62 | 38.34 / 38.81 |
| v21 (v20 + peer term) | round two /40 | 38.67 | 38.38 / 38.87 |

## Notes and code
- `docs/SUPER_BRANCH.md`, `docs/STRONGEST_BUILD.md`, `docs/RESIDUAL_HUNT.md` on `pranav/super`: every version, its evidence, and what was rejected.
- `docs/MODEL_PROMPTS.md` on `pranav/super`: every error-lowering step, the error map, and paste-ready rebuild prompts.
- Entries and harness: `pranav/super` (`run_super.py`, `round_two.py`, `peer_term.py`).

## For Pranav to add
- [x] The reasoning behind the big jumps (see *Key ideas*).
- [x] Anything missing from the "tried" list, and the final v20/v21 numbers (see *Final numbers*).
