# FINAL: best score found (main round, public validation)

> **UPDATE 6 (2026-11-07 ~8 PM, latest): new best = 20% zhill rule + 80% Pranav mixer5 (`27d776d`, `pkhanna/honest-tuning`: per-type
> dead/live win prior + learned size lags): dev 55.99 in-sample, 55.96 cross-fitted, +0.62 [+0.34,+0.96] over the v4 combo (+1.2 over the 55.90 run).
> Script: `round/score_mixer5.py`. Gemini research (`gemini-code-1794099642432.md`) tested on top: Exp 3 exact 8-hypothesis size switch loses
> 0.29-1.12 (`round/size/`); Exp 1 screened Walsh-logistic BMA has 7.4x v4's win MSE and blend weight 0 (`round/walsh/`); Exp 2 skipped (T3 showed no headroom).

> **UPDATE 5 (2026-11-07 evening, latest): best model = 20% zhill rule + 80% Pranav mixer v4 (`39e5757`): dev 55.34, +0.57 [+0.21,+1.02] over the
> 55.90-audit v3 combo; audit ~56.2 for the tied top-10 variants. Details: `COMBOS_TOP10.md`.**
> - The 55.90 v3 combo reproduces from source (`mainFiles/combo/predict.py`, max diff 2e-16).
> - v3/v4 read size modes / win levels from public validation targets (allowed by RULES.md). Cross-fitting them on half the episodes gives 55.35 vs 55.34: no inflation (`combos/crossfit_priors.py`).
> - Nothing adds to v4 beyond noise: 660-structure team blend search, our C1/C2/M9b/S5, Avyukth E67, gated blends, recalibration (`improve/`),
>   train priors / shrinkage / peer pooling (`improve/model/`), gapped-history handling, signal prior, peer-triggered forgetting (`boost/`).
> - "short_history" machines span ~200 ticks with 64 observed and ~21 gaps; ~45% of observed spins lack a pair lag. Gap handling did not help.
> - Where the rest of the error is (dev, one part made perfect): true win probability -> 58.36 (+3.0); true size given win -> 56.76 (+1.4).
>   True win probabilities are continuous (deciles .01 .02 .06 .13 .32 .66 .82 .91 .96), not two levels, so the next lever is a better win function
>   (e.g. per-machine logistic on Walsh lag terms with sign-flip hierarchical priors), not more blending.

> **UPDATE (after pulling teammates' branches, see `TEAM_COMPARISON.md`): the best model is the teammates' combined model on `zhill-branch`
> (20% zhill rule model + 80% Pranav pair mixer): 55.37 [54.5,56.1] on our 24 audit episodes (dev 54.14), above our stack's 54.09.**
> It is stdlib-only, deterministic and already packaged. Our stack adds only +0.15 (CI -0.05,+0.34) to it on dev. The sections below describe OUR models.

> **UPDATE 4 (more blending, volatility, geometric-motion, extensive combination search):** K1 (324 compositions, 154 blends, nested check), K2 (1681 two-stage structures, gates),
> V1-V3 (volatility), G1-G3 (gated/per-class/MLP blends), Q1-Q3 (random walk on log-odds, volatility-linked, pair-cell) all end within +0.0..+0.25 dev points of the best model
> (20% rule + 80% mixer v3, dev 54.77, audit 55.90), and nested checks put the held-out gain of the searched combos at about +0.007. That model is saturated under blending of the available members.
> The only combination worth considering is 15% of our C2 (+0.08, CI includes 0). Real gains now need a better win-probability component (the largest remaining error, p0 share .52).
>
> **UPDATE 3 (latest teammate push): Pranav's commit 6a8098c adds a two-mode payout-size model. The 20% rule + 80% mixer-v3 blend scores dev 54.77 [53.7,55.7] and
> audit 55.90 [55.1,56.6] through our harness: the best model found. Our C2 adds +0.07 (CI includes 0) to it. Details: `TEAM_COMPARISON.md`.
>
> **UPDATE 2 (blending everything, volatility, gated/meta blends):** the full team stack (our members + teammates' models, simplex pw/shares weights fit on dev) scores
> dev 54.29 (OOF) and **55.59 [54.89,56.17] on the 24 audit episodes**, vs 55.37 for the zhill combined model alone: a gain of about +0.2 that is within noise.
> Volatility-gated blends (V1-V3), observable-evidence gates (G1), per-class/geometric pools (G2) and a tiny MLP meta-learner (G3) all land at 54.3-54.4 dev,
> +0.1 to +0.25 over TM_COMBINED with CIs that mostly include 0. Blending is saturated near 54.3-54.4 dev; a better component (payout-size modelling) is what is needed.

**Best: W5 stack = 54.09 points on the 24 reserved audit episodes (CI 53.08-54.90)**, scored once after all choices were made on the other 72.
For reference, on the same audit set: pooled starter 30.00 by definition, original pair prototype B3 46.78, base B4 52.26 [50.95,53.35], simple composition C2 53.77 [52.71,54.66].

| Model | Dev (72 ep.) | Audit (24 ep., scored once) | Audit vs B4 (CI) |
|---|---|---|---|
| B3 pair prototype (start of this work) | 45.82 | 46.78 | -5.5 |
| B4 = lag 7 + pairs, ridge lambda 0.05 | 50.89 | 52.26 | |
| C2 = M9b win memory + C1 sizes | 52.09 | 53.77 | +1.51 |
| **W5 stack (winner)** | 52.32 (out-of-fold) | **54.09** | **+1.83 (+1.18, +2.60)** |

Stack vs C2 on audit: +0.32 (CI -0.05, +0.67): the stack's edge over the simple C2 is not distinguishable from zero, C2 is the cheaper model with nearly the same score.
Audit per-condition ratios (stack): stable .066, short_history .125, drift .124, shared_change .122 (pooled reference = 1.0).

## What produced the gains (each tested, CI vs its base)
1. **B3 -> B4, +5.1 dev / +5.5 audit.** B3's ridge (lambda 20) over-shrank the real lag effects and its extra single lags 1-5 hurt. Lag 7 plus pairs (1,3),(2,5),(4,7) at lambda 0.05 (near-OLS) is the win model. Lags/pairs are learned from train histories with null controls.
2. **A9 severity model, +0.58 dev.** How much a win pays depends per machine on centred log1p payouts at lags 1, 2, 4 (not on lag 7).
3. **Alpha-64 size-share prior, +0.27 dev.** One effect found by A1/A4/A6.
4. **M9b peer-gated episode changepoints, +0.33 dev.** Pool changepoint evidence across an episode's machines only when >=2 of 7 peers show a regime start in the last 80 ticks; B3-style holdout memory as fallback.
5. **Stacking the members, +0.23 dev over C2** (CI includes 0).
The win model (lag 7 + pairs) and the size model (lags 1,2,4) use disjoint lag structure, which is why their gains add.

## What did not work (tested, reported in RESULTS.md and reports/)
More lag structure (all pairs, triples, one-hot, MTD, per-machine selection), pooling coefficients to the type mean, signal interactions, HMM, ADWIN, BOCPD alone, JSD gates, half-lives 4-12 and weight floors, peer/episode features (E1-E5), condition-gated specialists, MLP, future-window pseudo-targets, rolling-cutoff row multiplication, payout-level features in the win model. Gradient-boosted residual learners (S1, S2, Codex) improved the old B3 but add <= +0.09 to B4.

## Caveats (read before trusting the numbers)
- Every member's settings were chosen on dev (32-336 cells each), so dev numbers are optimistic. The audit set was not used for any choice; its access log is `results/audit_access.log` (selfcheck scored B0/B3 on all 96 earlier; final scoring is the W5_final lines).
- 24 audit episodes is small; CIs are episode-level bootstrap.
- Audit scored higher than dev for B3 as well (46.78 vs 45.82), so the audit set is somewhat easier; compare within a set, not across.
- Codex W3/W4 learners (E1-E5, S1-S8) were built on the old B3; a rebase onto B4 (S1_B4, S2_B4, S5_B4) was requested but is not in these results. Expected value is small (their B3 gains were <= +0.9 and B4 adds <= +0.09 as blend members).
- Nothing here is packaged for submission: models need numpy/scipy; W6 stdlib port was dropped by request.
- Round two changes the structure: rerun `scripts/lag_screen.py` and the severity screen on the new train data first.
