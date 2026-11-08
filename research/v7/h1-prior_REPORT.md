# H1: data-derived win-level prior (cross-fitted)

Copy: `/Users/avynarra/Documents/Datathon/research_pf/h1-prior/mainFiles` (only copy edited).
Runs: `/Users/avynarra/Documents/Datathon/research_pf/h1-prior/runs/<variant>/` (`crossfit_validation_score.json`, `crossfit_validation_predictions.csv`, `.log`).
Analysis scripts: `h1-prior/analyze.py`, `h1-prior/stats.py`.

## What changed (copy only)

- `slot_forecasting/pair_mixer.py`: appended `win_prior_by_type_v2(targets, queries, centre_rule, strength_rule, fixed)`.
  Same 0.45 / 0.45 / 0.1 weights, same 0.5 split, same middle component (type median of 0.2-0.8, strength 10).
  Dead/live centre = `mean` or `median` of the cell's win chances. Strength rules on the empirical variance
  (method of moments, k = m(1-m)/var - 1, clamped to [1, 200]):
  `fixed` (10), `mom` (k), `min_mom` (min(10, k): only widens), `shrunk_mom` (0.5*(k+10)).
- `run_pair_mixer.py`: `WIN_VARIANT=centre,strength[,fixed]` env switch; unset = original code path.
- No constant from this dataset is hard-coded; all estimates come from the targets passed in.

## Results (`--cv-val`, honest: each half scored with priors from the other half)

Time: baseline run alone 104 s (not 30 s). Five variants run concurrently: 140 s wall each.

| variant (centre, strength) | tuning half | audit half | cross mean | vs baseline | in-sample val (not honest) |
|---|---|---|---|---|---|
| baseline (median, fixed 10) | 56.13 | 56.43 | **56.277** | - | 56.29 |
| mean, fixed 10 | 55.92 | 56.41 | 56.165 | -0.11 | 56.17 |
| mean, min(10, mom) | 56.03 | 56.47 | 56.251 | -0.03 | 56.26 |
| **mean, mom** | 56.15 | 56.50 | **56.323** | **+0.046** | 56.36 |
| mean, shrunk mom | 56.05 | 56.49 | 56.270 | -0.01 | 56.29 |
| median, min(10, mom) | 56.07 | 56.39 | 56.229 | -0.05 | 56.24 |

Per-condition loss ratios (cross-fitted, tuning / audit halves):

| condition | baseline | mean, mom |
|---|---|---|
| drift | 0.0517 / 0.0459 | 0.0539 / 0.0426 |
| shared_change | 0.0443 / 0.0695 | 0.0425 / 0.0661 |
| short_history | 0.1059 / 0.1052 | 0.1038 / 0.1064 |
| stable | 0.0741 / 0.0326 | 0.0744 / 0.0326 |

Win-part squared error summed over validation targets (cross-fitted; lower is better):

| variant | balanced | jackpot | steady | jackpot total |
|---|---|---|---|---|
| baseline | 1.24 | **2.85** | 1.72 | 3.30 |
| mean, fixed | 1.21 | 3.21 | 1.73 | 3.66 |
| mean, min(10, mom) | 1.21 | 2.96 | 1.73 | 3.40 |
| mean, mom | 1.10 | 3.00 | 1.58 | 3.45 |
| mean, shrunk mom | 1.13 | 3.12 | 1.60 | 3.56 |
| median, min(10, mom) | 1.24 | 2.99 | 1.72 | 3.43 |

Fitted jackpot prior (from full validation targets) for reference: dead 0.087 (strength 10 fixed; 2.4 by min-MoM), live 0.674 (MoM strength 17.1).

## Findings

1. The best honest variant is mean centres with method-of-moments strengths: 56.323 vs 56.277 baseline, +0.046. Both halves move the same way (tuning +0.02, audit +0.07). That is inside the half-to-half spread (about 0.3 points), so treat it as a weak signal, not a confirmed gain.
2. The gain comes from balanced and steady (win-part error down 0.14 and 0.14), not jackpot. Jackpot win error gets slightly worse under every variant that changes its spread.
3. The jackpot-live centre is not off by the amount claimed. In this round, jackpot targets >= 0.5 average 0.674 (sd 0.109; halves 0.683 and 0.666). Moving to the mean changes it by under 0.01. The ~0.58 figure does not match the >= 0.5 split. It may come from the 0.2-0.8 band (jackpot mean 0.552, sd 0.148), but I did not confirm that.
4. Data-derived spreads are wide for jackpot dead (MoM strength 2.4 vs 10), and widening it hurts jackpot slightly. So "too narrow" is not the data's verdict for jackpot.
5. Not tested: variant (d), a signal-dependent centre. The prior function gets no per-query signal, and adding one would be a larger change with more noise (about 130-140 targets per type per half).

## Recommendation

Do not promote a variant from this. The best gain (+0.05) is within noise. If a variant is adopted later, the candidate is mean centres with MoM strengths (`mean,mom`), re-checked on a new round. The jackpot-live hypothesis is not supported by these numbers.

## Caveats

- Method-of-moments strengths use truncated halves (split at 0.5), so they are rough.
- Cross-fit priors come from half the validation targets, so they are noisier than the full-data prior. That affects every variant equally.
- Entry/predict packaging for variants was not checked. Only the cross-fit scores were used.
