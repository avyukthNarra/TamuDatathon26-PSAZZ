# Pranav: research and building (draft for Pranav to correct)

Drafted by Sai's agent from Pranav's 51 commits (branches `pranav/*`, final `pranav/super`). Scores are main-round validation /60
unless marked /40. **Pranav: please fix anything wrong, add the reasoning behind each step, and approve.**

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
| Sun 11:40 | v21 round-two entry: peer term on the super blend | 38.67 / 40 | `7968b04` |

v13 and its successors are the base of the final blend's first member (v17-round2-guards).

## Tried, not adopted (from commit messages)
Population priors for expert weights; per-machine residual hunts and surprise tempering; per-lag and short-history peer refinements;
wider residual families, joint pool/peer grid, middle-level HMM; post-v15 short-history gates, regime member, size-flip detection;
per-type blend weights; rejected rule-model blend; round-7 jackpot and size attempts.

## Notes and code
- `docs/MODEL_PROMPTS.md` on `pranav/*` branches: every error-lowering step, the error map, and paste-ready rebuild prompts.
- Entries and harness: `pranav/super` (`run_super.py`, `round_two.py`, `peer_term.py`).

## For Pranav to add
- [ ] The reasoning behind the big jumps (two-mode sizes, Beta-mixture prior, size-rule HMM, v13).
- [ ] Anything missing from the "tried" list, and the final v20/v21 numbers on your split.
