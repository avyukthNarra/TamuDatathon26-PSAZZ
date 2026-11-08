# m3 win-side port

Base is v7c, with the size head, the flip swap, and the train-only size and cell-mode gates left on. v7c scores **57.369244 / 60**, halves **57.308288 / 57.391178**. Condition ratios: stable 0.041897, short history 0.074864, drift 0.032360, shared change 0.034305.

A piece is kept only when both cross-fit halves are at least as high as the previous kept row and the full score rises. Round-two kits must stay no worse than v7c (raw MSE 0.01199648 / 0.01408449 / 0.01205055, technical points 37.72 / 36.45 / 37.63).

## Main ablation

| Stack | Points | Tuning half | Audit half | Stable | Short | Drift | Shared | MSE | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v7c | 57.369244 | 57.308288 | 57.391178 | 0.041897 | 0.074864 | 0.032360 | 0.034305 | — | base |
| + signal prior, shrink 0.5 | 57.384080 | 57.320697 | 57.409527 | 0.042013 | 0.074519 | 0.031453 | 0.034360 | 0.00862358 | keep |
| + Pranav HMM at 0.3 | 57.536471 | 57.624688 | 57.383069 | 0.035921 | 0.072761 | 0.029133 | 0.033453 | 0.00809001 | drop |
| signal + Zhill 0.1/0.1, no HMM | 57.493274 | 57.458842 | 57.498984 | 0.036542 | 0.072869 | 0.031284 | 0.033706 | 0.00824041 | drop after round two |
| signal + HMM + Zhill | 57.591294 | 57.671855 | 57.455754 | 0.032800 | 0.071559 | 0.029618 | 0.033319 | 0.00789979 | not kept; HMM already failed |

The signal prior is a per-type x signal Beta level (`type|signal`, pulled halfway back to the type). It is the only addition that clears both halves. Drift improves; stable and shared change move by less than 0.0001. The flip still swaps 48 of 96 validation episodes, so the signal prior is not a substitute for the swap.

Pranav's hidden-rule HMM, blended at 0.3 into the win chance only, raises the full score by 0.15 and improves every condition, including drift and shared change. The audit half falls from 57.409527 to 57.383069, which is also under v7c's 57.391178. That fails the keep rule. The HMM and the flip are aimed at the same drift and shared-change episodes; the full-set gain does not survive the held-out half.

Zhill's blend is 0.8 of the current win chance, 0.1 of his rule model, and 0.1 of his HMM. Sizes stay on v7c's size head. On the signal stack both halves rise and the full score is 57.493274. Stacked on the HMM as well, the full score is 57.591294, but that row still contains the HMM piece the audit half rejected.

## Round-two sims

Ungated Zhill on the signal stack is worse than v7c on two kits. A train-only gate then compares Zhill's blend with the mixer on two held-out train episode folds (mean paired Brier delta plus 1.96 SE must be below zero, at least 8 episodes). The gate rejects on main and on every kit, so the delivered forecast matches signal alone.

| Kit | v7c MSE | v7c points | Signal MSE | Signal points | Ungated Zhill MSE | Ungated Zhill points | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| disjoint | 0.01199648 | 37.72 | 0.01191221 | 37.732249 | 0.01193592 | 37.727999 | reject (mean delta +0.000209, upper +0.001566) |
| four | 0.01408449 | 36.45 | 0.01408123 | 36.454854 | 0.01449302 | 36.361192 | reject (mean delta −0.001396, upper +0.000101) |
| sharedlag | 0.01205055 | 37.63 | 0.01199228 | 37.638335 | 0.01342616 | 37.387276 | reject (mean delta −0.002899, upper +0.000081) |

On main the same gate rejects Zhill: 96 episodes, mean delta **+0.000506**, SE 0.000376, upper **+0.001244**. Train out-of-fold does not support the validation gain, so the blend stays off. The size gate still keeps the v6 size head on all three kits and the new size head on main.

## Recommendation

Ship v7c plus the signal prior. Leave the HMM off. Leave Zhill off: ungated it misses the round-two bar on four and sharedlag, and the train gate does not turn it on anywhere measured.

The packaged entry is `mainFiles/runs/m3-final/entry/` and `mainFiles/runs/m3-final/entry.zip`. Fresh-process reproduction max difference is 0. The entry contains `rule_hmm.py`, `hmm_model.py`, `rule_model.py`, `win_blend.py`, and `baseline.py`. Switches in that entry: signal on, HMM off, Zhill off, size, size gate, flip, and round two on. Scored **57.384080 / 60**, episode-weighted MSE 0.00862358, same as the signal row above. 31 unit tests passed.

## Changed files

- `mainFiles/slot_forecasting/pair_mixer.py` — optional `type|signal` win prior, used by the walk
- `mainFiles/slot_forecasting/win_blend.py` — HMM and Zhill win-chance blends, plus the train gate
- `mainFiles/slot_forecasting/rule_hmm.py`, `hmm_model.py`, `rule_model.py` — teammate modules
- `mainFiles/slot_forecasting/forecast.py` — blend runs after the flip and before the size head
- `mainFiles/slot_forecasting/entry_config.py` — `USE_SIGNAL`, `USE_HMM`, `USE_ZHILL`
- `mainFiles/run_pair_mixer.py` — `--signal`, `--hmm`, `--zhill`
- `mainFiles/entry_template/predict.py` — same switches in the packaged entry
- `mainFiles/tests/test_v7_changes.py` — signal prior level check
