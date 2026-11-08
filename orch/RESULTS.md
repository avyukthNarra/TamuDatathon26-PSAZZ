# Fresh-start win-probability research: 57.91 cross-fitted (v13: 57.50)

All numbers: main-round validation, official formula (`common.points`, matches check_submission), cross-fitted =
any fitted choice made on the other half of the episodes (halves: episodes sorted by id within condition, alternating).
Reproduce the table: `python blend_final.py` (from this folder; reads `results/*.csv`).

| Forecast | Points | Halves A / B | stable | short | drift | shared |
|---|---:|---|---:|---:|---:|---:|
| v13 (team best, `pranav/strongest-build`) | 57.500 | 57.56 / 57.45 | .0380 | .0719 | .0291 | .0349 |
| v13 + short-history win patch | 57.676 | 57.71 / 57.65 | .0380 | .0591 | .0291 | .0349 |
| exact Bayesian filter | 57.488 | 57.41 / 57.56 | .0340 | .0713 | .0352 | .0343 |
| sysid rule library (train-only selection) | 57.445 | 57.37 / 57.49 | .0299 | .0922 | .0241 | .0317 |
| **blend: v13+short, exact, sysid** | **57.909** | **57.98 / 57.84** | .0285 | .0601 | .0250 | .0307 |

**The blend:** P(win) = sigmoid(mean of the three members' logit P(win)), with equal 1/3 weights (not tuned), and v13's
size split given a win. Weights chosen on one half and scored on the other come out at 0.2-0.4 each and score slightly
lower (57.85), so the equal weights are not overfit. Blending sizes too is worse (57.79).

**Why it gains:** the members' win errors correlate only 0.78-0.84 and each is best somewhere different:
sysid on stable/drift (explicit unchanged / machine-reversal / episode-reversal hypotheses), the exact filter on short
histories (exact sum over missing ticks), v13 on balanced machines. Win MSE: v13 .00473, exact .00438, sysid .00440,
blend .00385. Jackpot win remains the largest gap for every member.

## Members (each on its own branch, local commits pushed)
- `sai/fresh-exact`: exact posterior predictive under the full generative model (pair x direction enumerated,
  logit-normal levels on a 25x25 grid, joint episode inference over condition and one shared change tick, exact sum over
  missing win bits for short histories). Report: `worktrees/_coord/fresh/exact.md` (copied here as `members/exact.md`).
- `sai/fresh-sysid`: sparse rule library learned from train, machine-specific direction/intercept, summed over
  unchanged / independent-reversal / episode-shared-reversal hypotheses. Report `mainFiles/SYSID.md` on that branch.
- `sai/fresh-bayes`: short-history win patch on top of v13 (`mainFiles/short_win.py`): 256-state distribution over the
  last eight win bits so the 24 scattered older ticks still count; level grid from the type x signal prior.
- v13 itself: `origin/pranav/strongest-build`, `run_strongest.py --cv-val`.

## Supporting facts (this branch)
- `priors/pairs_all_histories.py`: 2058 confident long-history machines in train, validation AND test all use
  (1,3), (2,5) or (4,7). Round two can learn its pair set from all three splits.
- `priors/levels_from_truth.py`: validation targets are exact level draws; jackpot live centres on 0.58 at signal 0.
- `heldout.py`: held-out-tick scorer (Brier on real next spins from pseudo-truncated episodes); ranks v13 over mixer6 by
  0.0041 [0.0022, 0.0062], matching their true-error gap.

## Status / next
- A submittable v14 entry for this blend is being packaged (branch `sai/v14-blend`), then the eval suite.
- Not yet verified on a shifted round: the exact filter currently restricts pairs to the three main-round pairs and must
  re-learn them for round two.
