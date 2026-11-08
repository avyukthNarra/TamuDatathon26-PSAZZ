# Archived branches (deleted 2026-11-07)

These branches were removed to tidy the repo. Their full history is in `deleted-branches-2026-11-07.bundle` (needs `main` as a prerequisite). Results and conclusions for every one of them are summarised in [../MASTER.md](../MASTER.md).

Restore one: `git fetch research/branch-archive/deleted-branches-2026-11-07.bundle 'refs/remotes/origin/<branch>:refs/heads/<branch>'` (or `refs/heads/<branch>` for the local-only ones).

| Branch | Tip | Last commit |
|---|---|---|
| `remotes/origin/avy-zhill-improvement-1` | `889b3157e` | docs(changelog): link zhill entry to pull request |
| `remotes/origin/avyukth/entry-final` | `588a70b1b` | feat(entry): final entry blends signed-L1 logistic and neural context model |
| `remotes/origin/avyukth/entry-main` | `dbb6e382c` | feat(entry): switch entry to signed-L1 interaction model (48.2 audit) |
| `remotes/origin/avyukth/exp-combo-e2-memory` | `7c506e295` | feat(model): select combo memory independently |
| `remotes/origin/avyukth/exp-combo-trees-blend` | `39168b8e2` | feat(model): combine per-machine trees with short-history shrinkage |
| `remotes/origin/avyukth/exp-e1-pair-logistic` | `60f1daa3f` | fix(model): ignore ticks at or after the cutoff in win_probability |
| `remotes/origin/avyukth/exp-e10-machine-trees` | `281eaa47d` | feat(model): per-machine tree ensembles for the win chance (E10) |
| `remotes/origin/avyukth/exp-e11-flex-experts` | `f35bb67e2` | test(model): E11 rounds 3-4 sweep around the three-expert mix |
| `remotes/origin/avyukth/exp-e2-adaptive-memory` | `6511dfb5f` | docs(model): clarify adaptive memory regularization tie-break |
| `remotes/origin/avyukth/exp-e3-pair-selection` | `56b60d680` | feat(model): rank lag pairs from train machines |
| `remotes/origin/avyukth/exp-e4-episode-memory` | `0b3ff0d2f` | feat(model): pick memory length per episode from pooled prequential loss |
| `remotes/origin/avyukth/exp-e4-shared-change` | `0e33eaa8f` | docs(changelog): link E4 experiment entry |
| `remotes/origin/avyukth/exp-e5-shape-head` | `43f7de914` | fix(model): apply the cutoff guard to the shape head too |
| `remotes/origin/avyukth/exp-e6-interactions` | `c3ba38752` | feat(model): add configurable lag interaction sweep |
| `remotes/origin/avyukth/exp-e7-neural-context` | `af645aa2b` | feat(model): save and restore neural context artifacts |
| `remotes/origin/avyukth/exp-e8-neighbor-inputs` | `5c2bc3cc8` | feat(model): try neighbour-machine lag inputs for the win model (E8) |
| `remotes/origin/avyukth/exp-e9-win-blend` | `f8c0978c5` | feat(model): shrink per-machine win chance toward the baseline (E9) |
| `remotes/origin/avyukth/exp-m4-jackpot-prior` | `1cc88763a` | feat(model): add train-only jackpot win prior option (M4) |
| `remotes/origin/avyukth/exp-r0-baseline` | `68e5bd6fd` | docs(r0): keep report under word limit |
| `remotes/origin/avyukth/exp-r1-bayes-experts` | `981318f2a` | docs(model): report r1 tuning comparison |
| `remotes/origin/avyukth/exp-r2-em-priors` | `27495945a` | docs(model): report R2 windowed EM tuning results |
| `remotes/origin/avyukth/exp-r3-missing-filter` | `192709da8` | docs(model): record R3 missing-window no-op |
| `remotes/origin/avyukth/exp-r8-logpool` | `b72404772` | feat(model): add log pool and temperature to r0 blend |
| `remotes/origin/avyukth/exp-sig-prior` | `e2b7e0649` | feat(model): optional signal-shifted win prior (signal_prior experiment) |
| `remotes/origin/avyukth/exp-zhill-improve` | `889b3157e` | docs(changelog): link zhill entry to pull request |
| `heads/avy-zhill-improvement-1` | `889b3157e` | docs(changelog): link zhill entry to pull request |
| `heads/avyukth/entry-final` | `588a70b1b` | feat(entry): final entry blends signed-L1 logistic and neural context model |
| `heads/avyukth/entry-main` | `dbb6e382c` | feat(entry): switch entry to signed-L1 interaction model (48.2 audit) |
| `heads/avyukth/exp-combo-e2-memory` | `7c506e295` | feat(model): select combo memory independently |
| `heads/avyukth/exp-combo-trees-blend` | `39168b8e2` | feat(model): combine per-machine trees with short-history shrinkage |
| `heads/avyukth/exp-e1-pair-logistic` | `60f1daa3f` | fix(model): ignore ticks at or after the cutoff in win_probability |
| `heads/avyukth/exp-e10-machine-trees` | `281eaa47d` | feat(model): per-machine tree ensembles for the win chance (E10) |
| `heads/avyukth/exp-e11-flex-experts` | `f35bb67e2` | test(model): E11 rounds 3-4 sweep around the three-expert mix |
| `heads/avyukth/exp-e2-adaptive-memory` | `6511dfb5f` | docs(model): clarify adaptive memory regularization tie-break |
| `heads/avyukth/exp-e3-pair-selection` | `56b60d680` | feat(model): rank lag pairs from train machines |
| `heads/avyukth/exp-e4-episode-memory` | `0b3ff0d2f` | feat(model): pick memory length per episode from pooled prequential loss |
| `heads/avyukth/exp-e4-shared-change` | `0e33eaa8f` | docs(changelog): link E4 experiment entry |
| `heads/avyukth/exp-e5-shape-head` | `43f7de914` | fix(model): apply the cutoff guard to the shape head too |
| `heads/avyukth/exp-e6-interactions` | `c3ba38752` | feat(model): add configurable lag interaction sweep |
| `heads/avyukth/exp-e7-neural-context` | `af645aa2b` | feat(model): save and restore neural context artifacts |
| `heads/avyukth/exp-e8-neighbor-inputs` | `5c2bc3cc8` | feat(model): try neighbour-machine lag inputs for the win model (E8) |
| `heads/avyukth/exp-e9-win-blend` | `f8c0978c5` | feat(model): shrink per-machine win chance toward the baseline (E9) |
| `heads/avyukth/exp-m4-jackpot-prior` | `1cc88763a` | feat(model): add train-only jackpot win prior option (M4) |
| `heads/avyukth/exp-r0-baseline` | `68e5bd6fd` | docs(r0): keep report under word limit |
| `heads/avyukth/exp-r1-bayes-experts` | `981318f2a` | docs(model): report r1 tuning comparison |
| `heads/avyukth/exp-r2-em-priors` | `27495945a` | docs(model): report R2 windowed EM tuning results |
| `heads/avyukth/exp-r3-missing-filter` | `192709da8` | docs(model): record R3 missing-window no-op |
| `heads/avyukth/exp-r4-neighbor-experts` | `68e5bd6fd` | docs(r0): keep report under word limit |
| `heads/avyukth/exp-r8-logpool` | `b72404772` | feat(model): add log pool and temperature to r0 blend |
| `heads/avyukth/exp-sig-prior` | `e2b7e0649` | feat(model): optional signal-shifted win prior (signal_prior experiment) |
| `heads/avyukth/exp-x1-pfn-sim` | `68e5bd6fd` | docs(r0): keep report under word limit |
| `heads/avyukth/exp-zhill-improve` | `889b3157e` | docs(changelog): link zhill entry to pull request |
