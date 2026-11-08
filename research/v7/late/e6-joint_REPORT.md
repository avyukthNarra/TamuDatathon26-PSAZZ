# e6 joint pair–size posterior

The reverse coupling is implemented and switchable (`USE_JOINT`, `--joint` / `--no-joint`). Train cross-fit does not turn it on. Full validation and both halves match v7c, and the three static-size kits match v7c because the size gate rejects the size rule and the coupling stays off.

## Decision

Not kept as an active change to the forecast. With the flag on, the fitted strength is zero:

| run | joint | points | v7c |
| --- | --- | ---: | ---: |
| main validation | off, no train gain | **57.369244** / 60 | 57.369244 |
| half, tuning with audit priors | same fit | **57.308288** | 57.308288 |
| half, audit with tuning priors | same fit | **57.391178** | 57.391178 |
| kit disjoint | off, size gate | **37.71726** / 40 | 37.72 |
| kit four | off, size gate | **36.45405** / 40 | 36.45 |
| kit sharedlag | off, size gate | **37.62915** / 40 | 37.63 |

Condition loss ratios on the main set are unchanged: stable 0.041897, short history 0.074864, drift 0.032360, shared change 0.034305.

## What was fit

Per machine, at the forecast tick only, a posterior over the canonical pairs `(1,3)`, `(2,5)`, `(4,7)` is

`P(pair | wins, sizes) ∝ P(wins | pair) · P(pair) · Σ_{L,s} P(sizes | L, s) · P(L, s | pair)^λ`.

`P(wins | pair)` is the Jeffreys parity-cell likelihood ratio on the history window (full, early 45%, or last 40 ticks) with the clearest separation. `P(sizes | L, s)` is the train size-rule atom likelihood, missing lag bits marginalized. `P(L, s | pair)` is learned on train: each machine votes for its best pair with weight equal to the gap over the runner-up, and splits that vote by its size posterior. λ tempers the table (λ = 0 makes it flat, so size cannot move the pair). The posterior then blends the mixer's lag-pair expert masses, keeping each pair's half-life mix. Machines whose size posterior is at least half static get blend weight 0.

The table, from 750 train machines, recovers the structural map:

| pair | P(L=1) | P(L=2) | P(L=4) |
| --- | ---: | ---: | ---: |
| (1, 3) | **0.818** | 0.079 | 0.102 |
| (2, 5) | 0.101 | **0.816** | 0.083 |
| (4, 7) | 0.088 | 0.116 | **0.796** |

λ ∈ {0, 0.5, 1, 2, 4} and blend α ∈ {0.25, 0.5, 1} were scored on even/odd train episodes by cutoff Brier against `labels.csv`, after the same size-head replacement the entry uses. Every cell was slightly worse than α = 0 or tied at the printed noise floor. The least-bad positive setting, λ = 4 and α = 1, changed the two folds by **+3.7e-9** and **+6.0e-7** (higher is worse). Mean dynamic factor was 0.78 and 96.9% of machines were eligible, so the grid did apply the blend. The gate requires both folds to improve, so it selects off.

A separate read of validation histories (not used to choose λ) shows why the hard-label score does not move even though jackpot pair calls do. Forcing λ = 4 and α = 1 would change the mixer's pair mass by mean L1 0.084 overall and **0.202 on jackpot** (argmax disagrees on 9.4% of jackpots). Mean absolute win-probability shift is 0.008 overall and **0.022 on jackpot**, concentrated on jackpot drift (0.036) and jackpot shared change (0.045). Those are the episodes where a windowed pair vote fights the decayed mixer. The shifts are not systematically closer to the observed outcomes, so even/odd Brier does not fall.

## Win error by type

These are the shipped (coupling-off) validation numbers. Win SE is `(p_win − true p_win)^2`. Win part is the contest Brier left after replacing the size split with the truth's size split.

| slice | win SE | win part | n |
| --- | ---: | ---: | ---: |
| steady | 0.00401 | 0.00587 | 269 |
| balanced | 0.00346 | 0.00476 | 234 |
| jackpot | **0.00745** | **0.00972** | 265 |
| jackpot / stable | 0.00861 | 0.01126 | 58 |
| jackpot / short history | **0.00945** | **0.01223** | 71 |
| jackpot / drift | 0.00642 | 0.00835 | 65 |
| jackpot / shared change | 0.00545 | 0.00721 | 71 |

Jackpot, and jackpot with a short history, remain the large win errors. The coupling aims at that slice and does move those probabilities, but the train gate will not pay for the movement.

## Kits

On disjoint, four, and sharedlag the size gate keeps the v6 size head (deltas +0.00100, +0.00075, −0.00161 with upper bounds above zero). The joint fitter returns immediately with reason `size-gate` and does not reweight experts. Technical scores round to the v7c figures 37.72 / 36.45 / 37.63.

## Code

- `mainFiles/slot_forecasting/joint_coupling.py` — table, posterior, even/odd strength, per-machine static factor.
- `pair_mixer.blend_pair_weights` — forecast-tick reweight only. Online expert learning is unchanged. α = 0 returns the same weights.
- `forecast.fit_joint` / `predict_split(..., joint=)`.
- Flag default `USE_JOINT = True`. Off, or a rejected size rule, or no train gain, leaves predictions on the v7c path.
- Standard library only. No RNG. Episode folds are sorted episode ids, even/odd index.
- Tests: `python -m unittest tests.test_joint_coupling -v` (3 passed). Flat λ ignores size; a lag-4 size likelihood puts posterior mass on `(4,7)`; α = 0 preserves weights; a rejected rule blends nothing.

Runs: `mainFiles/runs/e6-joint`, `runs/kit-disjoint`, `runs/kit-four`, `runs/kit-sharedlag` (`--score-only`; main also `--cv-val`).
