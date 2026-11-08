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

## v18: adaptable for part 2 (same main score, 57.72)
- **Avyukth's v7c round-two guard fix** (via zhill's acc3d73, cherry-picked): v7c's round-two path
  lost to v6 on 5 of 7 simulated shifted rounds; fixed (edge_cases 28.82 -> 37.51 in zhill's check).
  Main test submission unchanged (byte-identical).
- **Train-evidence gates** (`combo.gate_terms`): the entry re-checks each add-on (jackpot sub-cell,
  flip transfer, extra peer, signal) on THIS round's train outcomes (cutoff + last 5 observed ticks)
  and switches it off if it doesn't help. Main round: all four kept (train Brier 0.36000 with all;
  off: sub-cell 0.36013, flip 0.36040, peer 0.36023, signal 0.36004).
- Built: validation 57.72 (57.720965), halves 57.71 / 57.68, test SHA-256 6ca96a1f... (= v17).
  Entry runtime 293 s (v17: 122 s), byte-identical on Python 3.13 read-only without targets.
- **Main round: submit v17's files** (identical predictions, faster entry). **Round two: v18**
  (`round_two.py` builds it and v8-v12 and picks by cross-fit on the new data).
- Not adopted from teammates: Sai's exact filter / sysid members (NumPy + SciPy: reproduction within
  1e-8 on the organisers' machines is a risk, and Sai measured our v16 adds only +0.01 to his blend,
  i.e. the members overlap); Avyukth's HMM-off-on-flip (worse in our blend, train +0.0015).

## v19: faster, gated on validation truth at build time (57.74)
Task list from the v18 review, with what the evidence said:
| Task | Result |
|---|---|
| 1. Runtime 293 s -> ~120 s | **Done: 81-88 s.** v7c runs in a forked 2nd process (pure stdlib, pickled results, deterministic); gates are decided at BUILD time on validation truth (768 exact targets instead of 4,608 noisy train outcomes) and saved in combo.json |
| 1. Continuous add-on scales | Tested (`research/continuous_scales.py`): train-chosen scales (v7c weight 0.1, peer 0.625...) 57.68 (57.71 / 57.66) vs 57.69 - not adopted; binary gates kept |
| 2. 5-band quantile win prior, LOO pair weighting | Already tested earlier: quantile prior 56.19 vs 56.26 (tuning half down); LOO 56.04 vs 56.12 - skipped |
| 3. Coarse / LOO size cells in our mixer | v17+ take payout sizes from v7c, so mixer size cells cannot change the forecast; coarse cells failed before - skipped |
| 4. Size flips coupled to win flips | v7c's flip gate fires for ALL drift / shared machines and no others (it identifies the condition, not the 8.8% wrong ones); hedging size toward 50/50 in flip episodes lowers both checks (`research/size_flip_coupling.py`, `size_hedge_flip.py`) - rejected |
| Build-time gates on main | sub-cell keep (0.007477 vs off 0.007903), flip keep (0.007779), signal keep, **extra peer OFF** (0.007398): both validation halves choose off independently, but held-out train says off is worse (window +0.0012 [+0.0003, +0.0022]); 0.25 is in between. A real split between the data sets (~+-0.02 on test) |
| Reproduction flaw found and fixed | Gating at run time made the forecast depend on whether validation answers are mounted (the eval suite mounts none): v19 decides at build time, so it is byte-identical with and without targets |

v19: validation 57.74 (57.742585), cross-fit halves 57.74 / 57.69 (the gate choice uses all of validation,
so these include that selection), R 0.0391 (stable 0.028, short 0.066, drift 0.027, shared 0.035). Entry
81-88 s; byte-identical on Python 3.13 read-only with and without validation targets. Test SHA-256
cd59f8b23599202c2ac6ba7b6c92a663c5cc0759a7176a0082504719b64a4e54.
Main-round upload: v17 (every part supported by both checks) or v19 (+0.02 on validation, peer
decision contradicted by train) - see the summary for the recommendation. Round two: v19 via round_two.py.

## v20: Sai's blend + my v19 as a 4th member (58.00)
Sai's `adapted_entry_v17g_rules` (sai/master-test, read only, copied into
`mainFiles/entry_template_super/`): equal log-odds blend of (1) my v17 with Sai's short-history and
jackpot patches, (2) Sai's exact Bayesian filter (learns pair, 3-lag XOR and AND/OR rule families
from each round's train census), (3) the sysid rule library; sizes from member 1. It ships pinned
NumPy 2.3.4 / SciPy 1.16.3. v20 adds `members/pranav_v19.py`, which runs my v19 entry (vendored in
`vendor/v19/`) as a 4th equal member - the only change to Sai's files is one REGISTRY line in our copy.

| Model | Validation | Cross-fit halves | Ratios stable / short / drift / shared |
|---|---|---|---|
| v19 (mine, stdlib) | 57.74 | 57.74 / 57.69 | 0.028 / 0.066 / 0.027 / 0.035 |
| Sai's 3-member blend (his saved cross-fit forecasts) | - | 58.02 / 57.93 (57.973) | - |
| **v20 = Sai's 3 + v19** | **58.00** (57.998498) | **58.04 / 57.92** | **0.025 / 0.060 / 0.024 / 0.030** |

Flaw found and fixed: without validation answers mounted, Sai's members fall back to train-only
priors and the forecast changed by up to 0.047 (a reproduction mismatch if the runner hides them,
as main's eval suite does). v20 bundles this round's public validation folder
(`bundled_validation/`, 8 MB) and, if /data lacks validation answers or the whole folder, builds a
scratch view in the temp folder; output is identical with full /data, without the answers, and with no
validation folder at all (max diff 0 each). main's eval suite (quick, local, no answers mounted): 0
FAIL, 31 PASS, 2 WARN (the future-ticks probe is refused by the schema's leak guard); its honest
4-fold validation is 57.91 [57.54, 58.25] (v19: 57.51). Build:
`runs/.venv-sai/bin/python run_super.py --out runs/super-v20` (venv with the pinned packages, git-ignored).
Entry: 168-169 s on this machine (members run as parallel processes). Test SHA-256
8a5a1f18a55a8d2684cfc269545ad8e596c264689f7637cdc87a5d918f761e82.
Round two: `runs/.venv-sai/bin/python round_two.py --data <kit>/public` builds v19 first, then v20
vendoring that new-round v19, and picks by cross-fit (v20's members re-learn their rules per round).

## v21: round two (adaptation) - v20 plus a peer term on the blend (38.67 / 40)
None of Sai's three members look at other machines; in v20 the peer effect reaches the blend only
inside the v19 member at 1/4 weight. `entry_template_super/peer_term.py` adds
0.75 x (other machines' win share over the last 5 ticks - the forecast split's mean share) to the
blended win log-odds. It uses history only; `peer.json` (written by `run_super.py --extra-peer`)
switches it on, so v20 builds are unchanged.

Evidence (round-two kit, 40-point scale):
| Check | Result |
|---|---|
| Peer coefficient in each split's own history (logistic vs a lag-pair cell baseline, episode bootstrap) | test 1.14 [0.65, 1.58], validation 0.54 [0.00, 1.13], train 0.27 [-0.53, 0.76]; main-round train 1.03 [0.35, 1.73] |
| Coefficient sweep on v20's saved cross-fit members (centre = split mean) | 0.5: 38.37 / 38.86, **0.75: 38.38 / 38.87**, 1.0: 38.38 / 38.88, 1.5: 38.37 / 38.88 (v20 38.34 / 38.81) |
| Signal term on the blend | lowers the audit half - not used |
| Built v21 | full 38.67 (R 0.0345; v20 38.62, R 0.0358), halves 38.38 / 38.87; ratios stable 0.031, short 0.050, drift 0.031, shared 0.027 |

Train-history support is weak (CI spans 0); the term is kept because the scored test split's own
history shows the effect clearly and it lifts both validation halves. 0.75 sits below the
validation optimum. The team rules blend's cross-fit is 38.53 / 38.63 (mean 38.59): v21 has the higher mean
(38.63) and the higher second half, but not the higher first half.
Build: `runs/.venv-sai/bin/python run_super.py --data <kit>/public --out runs/r2-v21 --extra-peer 0.75
--v19 runs/r2-v19/entry` (the scoring step needs the kit's rounds.py because its metadata still says
20 points; forecasts are written before that). Fresh read-only rerun without validation answers:
byte-identical, 173 s. Kit checker: valid. Test SHA-256
14099d37298a3b44d7285c09e007936b70bac2d438199d1ca4ad24bd93f42b29.
