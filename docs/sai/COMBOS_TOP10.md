# Top 10 ways to combine the team's models

## Read this first (plain English)

- **The 55.90 run is reproducible.** Re-running `mainFiles/combo/predict.py` from source matches the saved forecast to 2e-16 (dev 54.77), so its audit score is unchanged at 55.90.
- **All ten top combinations are the same idea:** about **20-26% zhill's rule model + 74-76% Pranav's mixer v4**, sometimes with a 4-8% pinch of our C1/C2. They score dev 55.31-55.36 and audit 56.21-56.31, vs 54.77 / 55.90 for the old run. Their differences from each other are noise.
- **What changed is mixer v4,** whose dead/live win prior fixes short-history machines (short-history error ratio .149 -> .121). Swapping v3 for v4 in the 55.90 recipe gives +0.57 dev [+0.21, +1.02].
- **Simplest entry to ship: COMBO 7 or plain 20/80** (rule + mixer v4). Both are standard-library only. COMBO 7's 24/76 weight is a tie with 20/80 (zhill's dev-only autotune also kept 20/80).
- **The validation-target "hardcode" is not inflating these scores.** v3/v4 read the size modes and win levels from public validation answers (allowed by RULES.md; the code re-reads them every round). Fitting them on half the episodes and scoring the other half gives 55.35 vs 55.34 in-sample (`combos/crossfit_priors.py`).
- **Our own models add nothing on top of v4** (C1/C2 ≤ +0.03, CI includes 0). Neither does anything in "Improvement attempts" below.

## Improvement attempts on the weak spots (dev only, all vs reference with paired CI)

Diagnosis of the 55.90 run: short_history (64 observed spins) has ratio .149, about double the other conditions (halving it is worth ~+0.95 points); jackpot machines have error/base .149 vs ~.08 elsewhere, 69% of it win probability; win-probability tails are slightly off.

| Attempt | Who | Base | Result |
|---|---|---|---|
| Win-prob logit recalibration (global / per type / per type x short) | Haiku | v3 combo | -0.13 to +0.04, noise |
| Blend weights gated on machine type and short-vs-long history (7 members) | Haiku | v3 combo | +0.52, but only because mixer v4 is a member; = COMBINED4 |
| Shrink short-history forecasts to a type prior | Haiku | v3 combo | -0.01 |
| Train-learned lag-pair prior (M1) | Codex | v3 / v4 | +0.06 / +0.05, noise on v4 |
| Type-level shrinkage of win cells (M2) | Codex | v3 / v4 | +0.23 / +0.00 |
| Pool evidence across episode peers (M3) | Codex | v3 / v4 | +0.20 / +0.00 (train check: peers share changepoints, not rules) |
| Train-only jackpot win prior (M4) | Codex | v3 / v4 | +0.23 on v3; on v4 only as a jackpot-only swap: +0.12 [-0.07, +0.36] |
| M1+M2+M4 together (M5) | Codex | v3 / v4 | +0.34 / +0.05 |
| Add our C1, C2, M9b, S5 or Avyukth's E67 to COMBINED4 | Claude | v4 combo | ≤ +0.03, CI includes 0 |

Mixer v4 already captures what these fixes were aiming at. The one lead left is the train-only jackpot prior swap (+0.12, not CI-clear), which would make the jackpot part of v4 independent of validation answers. Code and per-run numbers: `improve/gated.py`, `improve/model/`, `preds/IMP_*`.

## Full search results

The best dev result is COMBO 1 at 55.359 [54.548, 56.105] points out of 60, a paired +0.585 [+0.251, +0.996] against the previous 20% rule + 80% mixer-v3 entry (54.774 dev / 55.90 audit). All ten listed combinations fall in the best model's noise band; their numerical order does not establish a winner. COMBO 7 is the cheaper stdlib linear alternative (24.05% Rule, 75.95% Mixer4). The main change is mixer v4's win prior, which itself reads all validation targets. These are useful public-validation comparisons, not blind evidence of generalisation. The 660-structure search includes every teammate's available member; audit was opened once after freezing this ranking and all weights.

| Rank | Name | Members + full-dev weights | Family | Dev OOF [95% CI] | Δ vs previous entry, dev [paired CI] | Audit [95% CI] | Stdlib? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | COMBO 1 | win: 22.11% Rule + 74.17% Mixer4 + 3.71% C2; size: 70.46% Mixer4 + 29.54% C2 | factorised | 55.359 [54.548, 56.105] | +0.585 [+0.251, +0.996] | 56.298 [55.491, 56.863] | no |
| 2 | COMBO 2 | win: 23.56% Rule + 76.09% Mixer4 + 0.35% C1; size: 70.51% Mixer4 + 29.49% C1 | factorised | 55.354 [54.537, 56.110] | +0.580 [+0.245, +0.992] | 56.308 [55.500, 56.872] | no |
| 3 | COMBO 3 | 19.88% Rule + 72.14% Mixer4 + 7.98% C2 | linear | 55.345 [54.517, 56.105] | +0.571 [+0.245, +0.985] | 56.222 [55.419, 56.793] | no |
| 4 | COMBO 4 | 23.78% Rule + 69.76% Mixer4 + 6.46% C2 | geometric | 55.344 [54.483, 56.131] | +0.570 [+0.256, +0.947] | 56.233 [55.477, 56.789] | no |
| 5 | COMBO 5 | 24.51% Rule + 71.19% Mixer4 + 4.30% C1 | geometric | 55.328 [54.480, 56.124] | +0.554 [+0.248, +0.926] | 56.233 [55.467, 56.793] | no |
| 6 | COMBO 6 | 21.12% Rule + 73.71% Mixer4 + 5.16% C1 | linear | 55.326 [54.494, 56.090] | +0.552 [+0.228, +0.965] | 56.216 [55.407, 56.793] | no |
| 7 | COMBO 7 | 24.05% Rule + 75.95% Mixer4 | linear | 55.326 [54.499, 56.094] | +0.552 [+0.223, +0.962] | 56.213 [55.395, 56.804] | yes |
| 8 | COMBO 8 | 26.51% Rule + 73.49% Mixer4 | geometric | 55.320 [54.476, 56.110] | +0.546 [+0.242, +0.921] | 56.222 [55.432, 56.767] | yes |
| 9 | COMBO 9 | 26.13% Rule + 72.96% Mixer4 + 0.92% M9b_B4 | geometric | 55.318 [54.466, 56.114] | +0.544 [+0.237, +0.914] | 56.215 [55.433, 56.765] | no |
| 10 | COMBO 10 | 26.39% Rule + 73.23% Mixer4 + 0.37% S5_B4 | geometric | 55.312 [54.462, 56.108] | +0.538 [+0.231, +0.913] | 56.218 [55.425, 56.767] | no |

Weights are rounded for reading; exact vectors and per-fold weights are in [combos/results.json](combos/results.json). The delta column compares dev with dev; 55.90 names the reference run’s historical audit score, not the subtraction baseline. Stdlib describes prediction-time source requirements, after omitting components with fitted weight ≤ 1e-8; all selected structures still require their members for reproducing OOF fitting. The search/fitting code uses NumPy/SciPy in every case.

Aliases: Rule = zhill’s TM_RULE; Mixer3/Mixer4 = Pranav’s TM_MIXER3/TM_MIXER4; Zpool = Avyukth’s pooled-memory zhill blend; AvyCombo = Avyukth’s adaptive tree/logistic combo; AvyE67 = his neural E6/E7 blend; Ref = TM_COMBINED3. C1, C2 and M9b_B4 are Sai/Claude models; S5_B4 is Codex’s soft-target learner.

## New teammate members (Phase A, dev only)

| ID | Dev [95% CI] | Validation-target use |
| --- | --- | --- |
| TM_MIXER4 | 55.095 [54.160, 55.927] | YES: size modes per type AND Beta-mixture win prior fit from all 96 validation targets; settings previously validation-tuned. |
| TM_COMBINED4 | 55.345 [54.538, 56.106] | 20% saved Rule + 80% Mixer4; inherits Mixer4 leakage. |
| TM_ZPOOL | 54.438 [53.501, 55.328] | No validation targets read at runtime; inherited blend/settings were selected on validation. EpisodeMemoryMixer gamma=0.5. |
| TM_AUTOTUNE | 54.140 [53.071, 55.115] | YES for blend selection: dev targets used in this scoped wrapper; native auto_tune uses all 96 validation targets, including its own tune/audit halves. Not a Phase B member. |
| TM_AVYCOMBO | 38.923 [37.465, 40.219] | No validation targets read at runtime; fits train priors and selects memory prequentially in pre-cutoff histories; upstream hyperparameters validation-tuned. |

The dev-only autotuner's best candidate was {'rule': 0.4, 'mixer': 0.6}; it kept {'rule': 0.2, 'mixer': 0.8}. Tune gain was 0.085, below the native 0.15 threshold. The native auto_tune.py was deliberately not executed: it scores all 96 validation episodes, violating Phase A’s dev-only scope. The wrapper reuses its exact candidates and 0.1 weight grid with disjoint, condition-stratified 36/36-episode dev halves. That retained selection is an in-sample dev diagnostic, not an OOF score, and TM_AUTOTUNE is excluded from the combination pool. TM_AVYCOMBO completed in about four minutes with the already installed scikit-learn; no member was skipped.

## Search and uncertainty

All subsets of two or three of the eleven nominal members were enumerated: 220 subsets × three families = 660 structures. The reference is an allowed member but already contains Rule and Mixer3, so three nominal members can represent more than three underlying predictors. Linear and geometric pools have one simplex; factorised pools fit separate win and conditional-size simplices over the same subset (union ≤ 3). Geometric pooling uses log(max(p, 1e-12)) and renormalises. Factorised pooling uses the harness’s parts/combine functions.

For each fixed structure, weights are fitted on five folds and evaluated on the sixth, repeating across six deterministic, condition-stratified episode folds (12 episodes held out each time). A machine never crosses its episode’s fold. Condition/reference loss weights are estimated only from the training folds. Full-dev fitted weights produce the saved deployment forecasts. Every table dev number uses held-out weights; it does not use the full-dev weights shown beside it. The 95% CIs use the shared evaluate/diff_ci harness with 1,000 condition-stratified episode bootstraps (seed 314159). Ranking is fixed on dev; numerical copies from zero-weight padding are suppressed only when full-fit dev predictions differ by <1e-5 in every probability. Among deployment-equivalent structures, the highest OOF structure represents the group. All 660 structures, including copies, remain in results.json.

A separate nested selector uses three inner episode folds to choose a structure within each of the six outer training sets, then fits that structure there and scores the untouched outer fold. Its dev result is 55.227 [54.349, 56.048], delta +0.453 [+0.193, +0.764]. This evaluates the selection procedure; it is not substituted for any fixed structure’s OOF number. Inherited member training/tuning is not rerun inside these folds, so neither number removes member-level leakage. Numerical fitting uses SLSQP with analytic gradients, a uniform simplex start, maxiter=200 and ftol=1e-11; any unsuccessful fit aborts the run.

Noise band: 73 of 660 structures have a paired-difference CI against the best that includes zero. Listed ranks in the band: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10. Treat them as ties. This is a descriptive band without a multiple-comparison correction. Bootstrap intervals condition on fitted OOF predictions and do not refit the search within bootstrap replicates.

## The ten combinations

### 1. COMBO 1 — factorised

Search subset: TM_RULE, TM_MIXER4, C2. Full-dev deployment: win: 22.11% Rule + 74.17% Mixer4 + 3.71% C2; size: 70.46% Mixer4 + 29.54% C2. Independent win and size weights allow one model to correct p0 while another supplies the payout mix after a win. Measured on OOF dev, p0 MSE falls by +15.8% and conditional-size squared error at the true win rate falls by +3.3% relative to Ref (negative means an increase). The size diagnostic isolates size error by substituting the true win probability; it is not an additive decomposition of total loss.

Class-error shares (0, 1, 2, 5, 20): 49.9%, 25.5%, 18.0%, 6.0%, 0.5%. Paired delta to COMBO 1: +0.000 [+0.000, +0.000]; tied within noise.

Packaging cost: Package or port the non-stdlib member(s); Sai/Claude/Codex members need NumPy/SciPy and AvyE67 needs its neural environment. Zero-weight components may be omitted only after checking deployment parity; no new entry is packaged here.

### 2. COMBO 2 — factorised

Search subset: TM_RULE, TM_MIXER4, C1. Full-dev deployment: win: 23.56% Rule + 76.09% Mixer4 + 0.35% C1; size: 70.51% Mixer4 + 29.49% C1. Independent win and size weights allow one model to correct p0 while another supplies the payout mix after a win. Measured on OOF dev, p0 MSE falls by +15.7% and conditional-size squared error at the true win rate falls by +3.2% relative to Ref (negative means an increase). The size diagnostic isolates size error by substituting the true win probability; it is not an additive decomposition of total loss.

Class-error shares (0, 1, 2, 5, 20): 50.0%, 25.5%, 18.0%, 6.0%, 0.5%. Paired delta to COMBO 1: -0.005 [-0.019, +0.010]; tied within noise.

Packaging cost: Package or port the non-stdlib member(s); Sai/Claude/Codex members need NumPy/SciPy and AvyE67 needs its neural environment. Zero-weight components may be omitted only after checking deployment parity; no new entry is packaged here.

### 3. COMBO 3 — linear

Search subset: TM_RULE, TM_MIXER4, C2. Full-dev deployment: 19.88% Rule + 72.14% Mixer4 + 7.98% C2. A convex average lets the members offset probability errors directly across all five payout classes. Measured on OOF dev, p0 MSE falls by +15.9% and conditional-size squared error at the true win rate falls by +1.5% relative to Ref (negative means an increase). The size diagnostic isolates size error by substituting the true win probability; it is not an additive decomposition of total loss.

Class-error shares (0, 1, 2, 5, 20): 49.7%, 25.7%, 17.9%, 6.1%, 0.6%. Paired delta to COMBO 1: -0.014 [-0.066, +0.032]; tied within noise.

Packaging cost: Package or port the non-stdlib member(s); Sai/Claude/Codex members need NumPy/SciPy and AvyE67 needs its neural environment. Zero-weight components may be omitted only after checking deployment parity; no new entry is packaged here.

### 4. COMBO 4 — geometric

Search subset: TM_RULE, TM_MIXER4, C2. Full-dev deployment: 23.78% Rule + 69.76% Mixer4 + 6.46% C2. A renormalised geometric average rewards agreement on classes and can damp probabilities supported by only one member. Measured on OOF dev, p0 MSE falls by +15.8% and conditional-size squared error at the true win rate falls by +1.4% relative to Ref (negative means an increase). The size diagnostic isolates size error by substituting the true win probability; it is not an additive decomposition of total loss.

Class-error shares (0, 1, 2, 5, 20): 49.7%, 25.6%, 18.0%, 6.1%, 0.6%. Paired delta to COMBO 1: -0.015 [-0.096, +0.055]; tied within noise.

Packaging cost: Package or port the non-stdlib member(s); Sai/Claude/Codex members need NumPy/SciPy and AvyE67 needs its neural environment. Zero-weight components may be omitted only after checking deployment parity; no new entry is packaged here.

### 5. COMBO 5 — geometric

Search subset: TM_RULE, TM_MIXER4, C1. Full-dev deployment: 24.51% Rule + 71.19% Mixer4 + 4.30% C1. A renormalised geometric average rewards agreement on classes and can damp probabilities supported by only one member. Measured on OOF dev, p0 MSE falls by +15.5% and conditional-size squared error at the true win rate falls by +0.9% relative to Ref (negative means an increase). The size diagnostic isolates size error by substituting the true win probability; it is not an additive decomposition of total loss.

Class-error shares (0, 1, 2, 5, 20): 49.7%, 25.7%, 17.9%, 6.1%, 0.6%. Paired delta to COMBO 1: -0.031 [-0.110, +0.038]; tied within noise.

Packaging cost: Package or port the non-stdlib member(s); Sai/Claude/Codex members need NumPy/SciPy and AvyE67 needs its neural environment. Zero-weight components may be omitted only after checking deployment parity; no new entry is packaged here.

### 6. COMBO 6 — linear

Search subset: TM_RULE, TM_MIXER4, C1. Full-dev deployment: 21.12% Rule + 73.71% Mixer4 + 5.16% C1. A convex average lets the members offset probability errors directly across all five payout classes. Measured on OOF dev, p0 MSE falls by +15.5% and conditional-size squared error at the true win rate falls by +0.9% relative to Ref (negative means an increase). The size diagnostic isolates size error by substituting the true win probability; it is not an additive decomposition of total loss.

Class-error shares (0, 1, 2, 5, 20): 49.7%, 25.7%, 17.9%, 6.1%, 0.6%. Paired delta to COMBO 1: -0.033 [-0.082, +0.011]; tied within noise.

Packaging cost: Package or port the non-stdlib member(s); Sai/Claude/Codex members need NumPy/SciPy and AvyE67 needs its neural environment. Zero-weight components may be omitted only after checking deployment parity; no new entry is packaged here.

### 7. COMBO 7 — linear

Search subset: TM_RULE, TM_MIXER3, TM_MIXER4. Full-dev deployment: 24.05% Rule + 75.95% Mixer4. A convex average lets the members offset probability errors directly across all five payout classes. Measured on OOF dev, p0 MSE falls by +15.9% and conditional-size squared error at the true win rate falls by -0.4% relative to Ref (negative means an increase). The size diagnostic isolates size error by substituting the true win probability; it is not an additive decomposition of total loss.

Class-error shares (0, 1, 2, 5, 20): 49.5%, 25.9%, 18.0%, 6.1%, 0.6%. Paired delta to COMBO 1: -0.033 [-0.090, +0.022]; tied within noise.

Packaging cost: All listed model sources are standard-library code; port only the pooling arithmetic and exact fitted constants. Mixer4 requires its validation-derived size modes and Beta-mixture prior, beyond the currently packaged v3 entry.

### 8. COMBO 8 — geometric

Search subset: TM_RULE, TM_MIXER4, TM_COMBINED3. Full-dev deployment: 26.51% Rule + 73.49% Mixer4. A renormalised geometric average rewards agreement on classes and can damp probabilities supported by only one member. Measured on OOF dev, p0 MSE falls by +15.7% and conditional-size squared error at the true win rate falls by -0.1% relative to Ref (negative means an increase). The size diagnostic isolates size error by substituting the true win probability; it is not an additive decomposition of total loss.

Class-error shares (0, 1, 2, 5, 20): 49.6%, 25.8%, 18.0%, 6.1%, 0.6%. Paired delta to COMBO 1: -0.039 [-0.115, +0.040]; tied within noise.

Packaging cost: All listed model sources are standard-library code; port only the pooling arithmetic and exact fitted constants. Mixer4 requires its validation-derived size modes and Beta-mixture prior, beyond the currently packaged v3 entry.

### 9. COMBO 9 — geometric

Search subset: TM_RULE, TM_MIXER4, M9b_B4. Full-dev deployment: 26.13% Rule + 72.96% Mixer4 + 0.92% M9b_B4. A renormalised geometric average rewards agreement on classes and can damp probabilities supported by only one member. Measured on OOF dev, p0 MSE falls by +15.7% and conditional-size squared error at the true win rate falls by -0.4% relative to Ref (negative means an increase). The size diagnostic isolates size error by substituting the true win probability; it is not an additive decomposition of total loss.

Class-error shares (0, 1, 2, 5, 20): 49.5%, 25.8%, 18.0%, 6.1%, 0.6%. Paired delta to COMBO 1: -0.041 [-0.117, +0.031]; tied within noise.

Packaging cost: Package or port the non-stdlib member(s); Sai/Claude/Codex members need NumPy/SciPy and AvyE67 needs its neural environment. Zero-weight components may be omitted only after checking deployment parity; no new entry is packaged here.

### 10. COMBO 10 — geometric

Search subset: TM_RULE, TM_MIXER4, S5_B4. Full-dev deployment: 26.39% Rule + 73.23% Mixer4 + 0.37% S5_B4. A renormalised geometric average rewards agreement on classes and can damp probabilities supported by only one member. Measured on OOF dev, p0 MSE falls by +15.5% and conditional-size squared error at the true win rate falls by -0.2% relative to Ref (negative means an increase). The size diagnostic isolates size error by substituting the true win probability; it is not an additive decomposition of total loss.

Class-error shares (0, 1, 2, 5, 20): 49.6%, 25.8%, 18.0%, 6.1%, 0.6%. Paired delta to COMBO 1: -0.047 [-0.123, +0.027]; tied within noise.

Packaging cost: Package or port the non-stdlib member(s); Sai/Claude/Codex members need NumPy/SciPy and AvyE67 needs its neural environment. Zero-weight components may be omitted only after checking deployment parity; no new entry is packaged here.

## Reproduce and verify

Run from the repository root with the existing Python environment. Never switch this shared checkout’s branch. Archive the pinned sources into worktrees/_tm; 81d4fff changes only documentation relative to mixer4’s 39e5757 model.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p worktrees/_tm/{mixer4,zpool,autotune,avycombo}
git archive 39e5757 | tar -x -C worktrees/_tm/mixer4
git archive 9573666 | tar -x -C worktrees/_tm/zpool
git archive 3bdf336 | tar -x -C worktrees/_tm/autotune
git archive 7c506e2 | tar -x -C worktrees/_tm/avycombo
nice -n 10 python3 docs/sai/combos/members.py mixer4
nice -n 10 python3 docs/sai/combos/members.py zpool
nice -n 10 python3 docs/sai/combos/members.py autotune
nice -n 10 python3 docs/sai/combos/members.py avycombo
nice -n 10 python3 -m unittest discover -s docs/sai/combos -p 'test_*.py' -v
nice -n 10 python3 docs/sai/combos/search.py dev --out worktrees/_coord/combo-reproduce
```

The original dev run used the default output directory, then search.py audit once. Do not rerun the audit or delete its marker. The audit command verifies source/prediction SHA-256s and exclusively creates audit.once.json before reading the split. Exactly eleven evaluate calls use who=COMBO_TOP10: ten combinations and TM_COMBINED3. audit.once.json records the frozen pre-audit results hash, ranking and forecast hashes. Dev replay uses a new output directory so it cannot rewrite the frozen record. Saved COMBO_<n>_validation.csv files use full-dev weights for all 768 queries; *_oof_validation.csv files use OOF predictions on dev and inert 0.2 placeholders on audit and must never be submitted.

Validation performed: five synthetic search tests; ten existing copied mixer tests; eight existing verification tests; twelve pinned v4 mixer tests; six pinned pooled-memory tests (41 total passed); Python compilation, whitespace and forecast integrity checks. An independent full dev replay reproduced results.json and all ten deployment/OOF CSVs byte for byte. There is no configured lint command or CI. No dependencies or configuration were changed.

## Caveats and next steps

- Teammate models/settings were tuned using all 96 validation episodes. Mixer3 reads all validation size targets; Mixer4 additionally fits a win prior from them. The audit numbers are not blind. The new weights alone are dev-only.
- There are only 24 audit episodes (six per condition). Audit is shown in the frozen dev order; no weights, ranks or candidate membership changed after seeing it. The reference’s one-shot re-score is 55.902 [55.023, 56.616].
- The many structures share most of their predictors. Small paired differences inside the band do not identify a better entry; prefer lower packaging cost among tied models. The table is a list of combinations, not ten independent discoveries.
- Before deployment, reproduce the selected model on test data, verify determinism and the contest contract, and package it separately. This experiment changes no submission predictor. For round two, relearn the priors and lag structure from permitted data and use fresh episode-held-out evaluation. The main remaining opportunity is an independently estimated win-probability component, with payout-size calibration checked separately.
