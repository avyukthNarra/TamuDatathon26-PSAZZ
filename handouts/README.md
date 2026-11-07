# Handouts: our House Edge experiment branches

Deep-dive guides for teammates building on our work, one per successful branch. Start with the overview.

All scores use the kit's `episode_folds` split of validation, 48 tuning / 48 audit episodes, out of 60. The kit baseline scores 31.06 tuning / 30.57 audit.

| Handout | Branch | Tuning | Audit | Read it for |
|---|---|---|---|---|
| [AVYUKTH_E6_E7_BLEND.md](AVYUKTH_E6_E7_BLEND.md) | `avyukth/entry-final` | 50.65 | 50.02 | Overview, the final entry, how everything fits together, and transferable ideas |
| [E7_NEURAL_HYPERNETWORK.md](E7_NEURAL_HYPERNETWORK.md) | `avyukth/exp-e7-neural-context` (PR #12) | 50.18 | 48.99 | A learned estimator that maps a machine's statistics to coefficients; best on short history |
| [E6_SIGNED_L1_INTERACTIONS.md](E6_SIGNED_L1_INTERACTIONS.md) | `avyukth/exp-e6-interactions` (PR #11) | 48.54 | 48.19 | Why ±1 coding + L1 finds the XOR/XNOR lag-pair rules; under 1 s |
| [E11_EPISODE_MEMORY_EXPERTS.md](E11_EPISODE_MEMORY_EXPERTS.md) | `avyukth/exp-e11-flex-experts` | 40.26 | 40.90 | Episode-pooled memory selection, short-history shrinkage, shape head, tuning-only scorer |
| [E2_ADAPTIVE_MEMORY.md](E2_ADAPTIVE_MEMORY.md) | `avyukth/exp-e2-adaptive-memory` (PR #8) | 37.15 (C=10) | — | Per-machine memory selection by prequential loss |

**Context:** the team's best model is Zhill's rule + pair-mixer blend (`zhill-branch`, 54.37 / 54.57 on the same split). Our handouts point out which of our ideas could improve it.

**More:** `../HOUSE_EDGE_EXPERIMENT_RESULTS.md` has the full experiment log and lessons.
