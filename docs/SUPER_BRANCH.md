# `pranav/super`: everything from Pranav's branches, one build (v17, 57.72)

This branch contains the full history of all of Pranav's branches (COMBINED_SOLUTIONS, bocpd-model,
stacked-model, population-priors, residual-hunt, peer-plus, deep-hunt, strongest-build, v13-plus,
v14-plus, v15-plus, v15-evalsuite) plus main (the evaluation suite). Teammates' branches were only
read; their ideas are credited where used.

## Build, check, submit
```sh
cd mainFiles
python run_strongest.py --out runs/strongest-v17 --cv-val      # ~7 min: build + cross-fit + entry + rerun
python check_submission.py runs/strongest-v17/submission.csv
python -m unittest discover -s tests                           # 41 tests
```
Upload `runs/strongest-v17/submission.csv` and `runs/strongest-v17/entry.zip` (entry runtime ~2 min,
standard library only, byte-identical on Python 3.13 read-only with or without validation targets).
Round two: `python round_two.py --data <new kit>/public` (see docs/ROUND_TWO.md).

## What v17 is (each line passed validation cross-fit AND held-out train, or is noted)
| Layer | Source | Effect |
|---|---|---|
| Pair mixer: lag-pair match/differ experts, 3 memories, per-type+signal dead/live/middle win prior, learned size lags | ours (COMBINED_SOLUTIONS) | base, 56.29 |
| Hidden-rule HMM (rule switches), pooled with the mixer in log-odds | zhill-56.28-hmm idea, our levels | 56.51 -> 56.54 |
| Peer correction: machines win more when the episode's other machines just won | ours (residual hunt) | 56.59 |
| Plateau average of three near-equal win variants | ours | 56.61 |
| Payout size: size mode set per machine by whether the payout k spins back was >= 2x | zhill size-rule HMM / Avyukth v7c size atoms | 57.36 |
| 0.7 / 0.3 log-odds blend with Avyukth's v7c (vendored unchanged in `v7c_forecasting/`), v7c sizes | v7c | 57.52 |
| Extra peer term (v7c ignores peers), signal term | ours | 57.55 |
| v7c's episode flip swap transferred to our win model | v7c flip gate | 57.59 |
| Jackpot sub-cell correction (parity cells pool sub-cells with different rates) | Sai's jackpot finding, our implementation | **57.72** |

v17 validation 57.72 (57.720965), cross-fit halves 57.71 / 57.68, R 0.0395 (stable 0.029, short
0.066, drift 0.027, shared 0.036). Test SHA-256
6ca96a1f82a4593043a8111b017b2e21e2568f83cce8bf160ea34d9e4a00734d. Full evidence: docs/STRONGEST_BUILD.md,
docs/RESIDUAL_HUNT.md, docs/STACKING_RESEARCH.md, docs/MODEL_PROMPTS.md.

## Flaws found and how they are handled
1. **Choices made while looking at validation inflate validation.** Cross-fitting stops prior leaks
   but not tuning leaks (the regime model, learned stackers and per-lag weights all looked better on
   validation than on held-out train). Rule used throughout: a change is kept only if it raises both
   validation halves AND does not lose on held-out train (8 episode folds, cutoff + 10 ticks).
2. **v7c had no held-out train check inside our blend.** Fixed: `research/v7c_cache.py` /
   `v7c_flip_cache.py` refit it without each fold (in parallel, 3.6 min).
3. **The 50/50 blend silently halved the peer effect** (v7c ignores other machines). Fixed in v14.
4. **Our win model never saw episode flips.** Fixed in v16 (flip transfer).
5. **Parity cells pool sub-cells with different win rates.** Fixed for jackpot in v17; for other types
   it raised validation but not train, so it stays off there.
6. **Round-two robustness.** On the evaluation suite's simulated shifted rounds, the newest parts can
   lose to the simpler v8 (e.g. round2_shift -0.49 for v15; v7c loses on 5 of 7 presets per
   Avyukth). `round_two.py` therefore rebuilds every candidate (v8 ... v17) on the new data and picks by
   cross-fitted validation, ties to the simpler model.
7. **Frozen constants.** The plateau's per-lag peer weights and the blend settings (combo.json) were
   learned on the main round; everything data-derived (pairs, size lags, size modes, priors, peer
   centres, v7c's rules and gates) is re-learned from each round's data.
8. **Packaging slips** (a missing module in a first build) are caught by the fresh-process rerun
   in every runner and by the read-only, no-targets rerun used before staging files.

## Tried and rejected (don't repeat): see the "rejected" rows in docs/STRONGEST_BUILD.md and
docs/RESIDUAL_HUNT.md - regime model member, learned stackers, population priors, size shrinkage,
gap decay, leave-one-out expert scoring, coarse / pooled size cells, size-flip detection (AUC 0.54-0.64),
HMM-off-on-flip, per-type / short-history blend weights, sub-cell on all types.
