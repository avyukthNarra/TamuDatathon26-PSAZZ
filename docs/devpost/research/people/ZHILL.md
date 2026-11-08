# Zhill: research and building (draft for Zhill to correct)

Drafted by Sai's agent from Zhill's 18 commits (branches `zhill*`, `latest-update-push-zhill`). Scores are main-round validation /60.
**Zhill: please fix anything wrong, add the reasoning behind each step, and approve.**

## Path
| When | Step | Score | Commit |
|---|---|---|---|
| Sat 14:30 | Per-machine lag-pair rule model | 52.89 | `98e9c50` |
| Sat 14:53 | Rule model blended with the team pair mixer | 54.48 | `232f850` |
| Sat 15:03 | Rule-model blend weight lowered to 0.2 after an independent check | | `4fe7ba8` |
| Sat 17:16 | `auto_tune.py` re-picks the blend per round | | `3bdf336` |
| Sat 17:40 | Pair mixer v3 in the combined entry | 55.58 | `fd8f7dd` |
| Sat 17:59 | Optional live/dead win prior for the rule model | | `4d4dd9b` |
| Sat 20:02 | Hidden-rule HMM, blended 50/50 | 56.28 | `e5ffc49` |
| Sat 22:24 | Entry on mixer6 + rule model + HMM | 56.59 | `6f471e4` |
| Sun 00:12 | Size-rule HMM for payout size | 57.32 | `00d0d9d` |
| Sun 02:17 | Size-rule prior learned from each round's train (EM) | | `00b47d6` |
| Sun 02:55 | **v17-round2-guards**: v17 with Avyukth's v7c round-two guards (the final blend's first member) | | `a2e5d86` |
| Sun 03:34 | v18 backup entry: v17 guards blended with Sai's exact filter | 57.95 | `278efff` |
| Sun 04:08-04:48 | Round-adaptive member gate + bundled validation for the rules blend (`r2/gate_members.py`) | | `1edf5a8`, `8931a6c` |

The 20% rule + 80% pair mixer combination was the team's best model on Saturday afternoon (55.37 → 55.90 on Sai's audit episodes).

## Tried, not adopted
Neighbour-rule and trend-extrapolation attempts (`mainFiles/MODEL_PLAN.md`).

## Notes and code
- `mainFiles/MODEL_PLAN.md`, `make_entry.py`, `auto_tune.py` on `zhill*` branches; v18 on `zhill/v18-exact-blend`; gate on `zhill/blend-member-gate`.

## For Zhill to add
- [ ] How the rule model and the size-rule HMM work, and why the 0.2 weight.
- [ ] Anything missing from the "tried" list.
