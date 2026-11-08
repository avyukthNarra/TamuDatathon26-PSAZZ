# Change-point regime model (experimental branch `pranav/bocpd-model`)

A different approach from the pair mixer, built as new files only (the 56.51 model's code is
imported, not edited): `slot_forecasting/changepoint.py`, `slot_forecasting/ensemble.py`,
`run_ensemble.py`, `entry_template_ensemble/`, `tests/test_changepoint.py`.

## Idea
Bayesian online change-point detection (Adams & MacKay 2007) over each machine's hidden story:
when the current regime began, which lag pair drives it (and in which direction: the "agree" or
"differ" cell is live), and each cell's own win level learned exactly from its counts since the
regime began, under the type/signal Beta-mixture prior (live cell: live + middle components,
dead cell: dead component). Hazard 0.001 per tick, likelihood tempered by 0.5, 24 segments kept.

## Results (cross-fitted validation, halves; payout sizes from the pair mixer)
| Model | Score | Notes |
|---|---|---|
| Regime model alone, uncoupled cells | 55.27 | short_history 0.152 |
| + coupled live/dead cells | 55.85 | |
| + hazard 0.001 | 56.20 | stable 0.045 (best of any model) |
| + tempered likelihood 0.5 | 56.36 (56.34 / 56.41) | beats the pair mixer alone (56.29) |
| 56.51 model (mixer 0.7 + HMM 0.3) | 56.48 (56.47 / 56.50) | |
| Ensemble 1/3 mixer + 1/3 HMM + 1/3 regime | **56.60 (56.63 / 56.58)**; built entry 56.63 | stable 0.044, drift 0.042, shared 0.050, short 0.102 |

## Out-of-fold train check (vs the 56.51 model): NOT passed
| Weights mixer / HMM / regime | Cutoff dR | Window dR (cutoff + 10 spins) |
|---|---|---|
| 1/3 each | +0.0031 [-0.0027, +0.0090] | -0.0002 [-0.0017, +0.0014] |
| 0.35 / 0.30 / 0.35 | +0.0033 | -0.0000 |
| 0.25 / 0.25 / 0.50 | +0.0057 | +0.0009 |

Train shows no gain and is worse on drift (+0.012 at the cutoff): a regime model that never
forgets within a regime handles gradual drift worse than decaying tallies. Validation's +0.12
is the kind of gain that earlier failed out of sample, so the 56.51 model stays the submission.

## Ideas to make it pass
- A drift-aware version: let a regime's level decay slowly (half-life ~100) instead of fixed counts.
- Weight the regime model by an observable drift signal (recent vs older win rate), not condition labels.
- Size modes inside the regime model (mode as a hidden state switched by lags 1/2/4).

Prompt used: "With everything you know and the limitations, create something completely new on a
new branch; don't edit my code, reference it where needed, and build a really good model."
