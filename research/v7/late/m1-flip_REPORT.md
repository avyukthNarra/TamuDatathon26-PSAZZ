# M1: episode flip on Zhill 57.32

The v7c episode parity swap, added to Zhill's mixer6 (`00d0d9d`), raises full validation from **57.321727** to **57.429823** / 60. Swapping the HMM direction posterior as well reaches **57.431332**. The count swap is the result. The HMM add-on does not win on both halves.

## What was added

The detector is `slot_forecasting/episode_flip.py` (the v7c gate). On train it takes the midpoint of the widest gap in the episode log-Bayes-factors. On the split being forecast, an episode above that midpoint exchanges agree/differ counts at its joint MAP tick, then mixer6 keeps learning. This train: gap −32.538 to 237.882, midpoint **102.672**, width 270.420. Same gate the v7c report measured (102.67).

`--flip` on `predict.py` (default `off`):

| Scope | Behavior |
|---|---|
| `off` | Zhill blend, no swap. |
| `mixer` | Swap inside mixer6's parity counts only. |
| `all` | That swap, plus a match/differ posterior exchange at the same tick in mixer6's rule HMM and the outer HMM. |

The rule model keeps full lag-bit tables, not agree/differ counts, so it is not swapped. Pairs, the tick, and the gate are fit from the round being forecast. `off` is the default, so `make_entry.py` stays on the 57.32 path.

## Scores

Loss ratios are lower-is-better. Even/odd are the kit halves (`episode_folds`: even index = tune, odd index = audit), the same halves as the commit's 57.44 / 57.20.

| Variant | All 96 | Even (tune) | Odd (audit) | stable | short_history | drift | shared_change |
|---|---:|---:|---:|---:|---:|---:|---:|
| Z | **57.321727** | 57.438428 | 57.195772 | 0.034712 | 0.081114 | 0.032855 | 0.038213 |
| Z+flip (`mixer`) | **57.429823** | 57.500088 | 57.354344 | 0.034712 | 0.081114 | 0.030313 | 0.032875 |
| Z+flip+hmm (`all`) | **57.431332** | 57.487898 | 57.371173 | 0.034712 | 0.081114 | 0.030377 | 0.032701 |

Z was scored twice: the unmodified entry (`runs/z_validation.csv`, SHA-256 `72533f4cb700df69800be202707b16247bc0b61ddf66418a738eaf8b33bcbb56`) and the instrumented `off` path. Both give 57.321727. Unit tests: 24 OK.

`mixer` and `all` swap the same **48 / 96** validation episodes: all 24 drift and all 24 shared change, none of the stable or short-history episodes. Stable and short-history ratios are unchanged.

Deltas versus Z: mixer **+0.108** (even +0.062, odd +0.159); mixer+HMM **+0.110**. The HMM posterior swap is **+0.0015** over mixer alone. It helps the odd half (+0.017) and hurts the even half (−0.012), and it slightly worsens drift while slightly helping shared change.

## Recommendation

Use `--flip mixer`. The gain is the count swap on drift and shared change, which is the same episode set the v7c detector selected, and both halves move the same direction. Leave the HMM and the rule model unflipped. `--flip all` is in the code for the comparison and should stay off.

Short history is still the weak condition (ratio 0.081). The gate correctly does not fire there.
