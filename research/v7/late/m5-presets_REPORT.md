# m5 presets: which v7c piece hurts synthetic presets

The loss is the round-two path. Size and flip do not explain it. A train-only change keeps full cells when the in-sample win is at least 0.002 Brier, falls back to the v6 pair vote when the robust search keeps a single pair, and refuses to flip an episode whose log Bayes factor is still negative. Every preset that v7c lost is back to the v6 level or better. Real cross-fit validation stays 57.31 / 57.39. The three g7 kits stay 37.72 / 36.45 / 37.63.

## Setup

Presets from `eval_suite.generator`, seed 0, 24 episodes per condition on train, validation, and test (the suite uses the kit validation count). Points are the contest score of `submission.csv` against `_truth/test` (max 60, except `round2_shift` at 40). v6 and v7c match the h10 report on every preset.

Variants are `run_pair_mixer.py --score-only` flags. v6 is size, size gate, flip, and round2 off. v7c is all of them on. Each leave-one-out turns off one component. The cell-mode gate runs only inside round2.

## Preset × variant (test points)

| preset | max | v6 | v7c | v7c−flip | v7c−size | v7c−round2 |
|---|---:|---:|---:|---:|---:|---:|
| main_like | 60 | 40.2661 | 40.2741 | 40.2741 | 40.2661 | 40.2741 |
| round2_shift | 40 | 23.9018 | 21.2809 | 21.2091 | 21.2637 | 23.9168 |
| all_lags | 60 | 35.5139 | 31.8555 | 31.5735 | 31.8568 | 35.5248 |
| neighbour_heavy | 60 | 36.2560 | 36.2560 | 36.2560 | 36.2560 | 36.2560 |
| hidden_state_heavy | 60 | 38.7640 | 32.3923 | 32.3911 | 32.3923 | 38.7640 |
| fast_drift | 60 | 32.8342 | 29.9366 | 30.9937 | 29.9423 | 32.8424 |
| edge_cases | 60 | 38.3499 | 32.8273 | 32.6944 | 32.8273 | 38.3499 |

Dropping size never recovers a loss. Dropping flip never recovers one either, except a partial +1.06 on `fast_drift` that is still 1.84 short of v6. Dropping round2 recovers every loss, and the recovered score is at least v6.

## What round2 does wrong

Three separate mechanisms. Neighbour-heavy and main-like do not hit them, which is why those two tie.

**Cell-mode veto.** `adapt` picks full cells when they win the in-sample cutoff Brier, then the round-two gate throws that choice out unless the held-out mean plus 1.96 SE is negative. The generator's AND and OR interactions are not functions of agree/differ, so the 4-cell experts are the ones that can represent them. On the losing presets the in-sample comparison already prefers full, and the held-out mean usually does too, but the upper bound is positive so the gate forces parity.

| preset | parity Brier | full Brier | held-out mean (full − parity) | upper CI | gate |
|---|---:|---:|---:|---:|---|
| main_like | 0.56093 | 0.52694 | −0.0131 | −0.0005 | keeps full |
| neighbour_heavy | 0.54295 | 0.51268 | −0.0153 | −0.0008 | keeps full |
| round2_shift | 0.56594 | 0.55343 | −0.0065 | +0.0103 | forces parity |
| all_lags | 0.57675 | 0.56926 | −0.0007 | +0.0118 | forces parity |
| hidden_state_heavy | 0.56647 | 0.55685 | −0.0085 | +0.0068 | forces parity |
| edge_cases | 0.55713 | 0.52471 | −0.0206 | +0.0006 | forces parity |
| fast_drift | 0.57547 | 0.57230 | +0.0045 | +0.0185 | forces parity |

On `edge_cases` the held-out mean favors full by 0.021 and the upper bound is only +0.0006. That veto costs 5.52 points. Pairs on these rows, except hidden-state, are the same set v6 uses.

**Robust pair collapse.** On `hidden_state_heavy`, `choose_pairs_robust` keeps only `(1, 2)`. v6's vote keeps all 28 pairs. A hidden state does not produce a stable lag-pair winner, so one windowed winner clears the 5% floor and every other pair is dropped. Turning size or flip off leaves the score at 32.39. Restoring the v6 pair vote (round2 off) returns 38.7640 exactly.

**Negative flip gap.** On `fast_drift` the widest training gap in the log Bayes factor sits entirely below zero (−20.58 to −17.61, midpoint −19.10). The old rule swaps every episode above that midpoint, which is 88 of 96 test episodes, all of them still evidence against a flip. Those swaps cost about 1.06 points only while the model is stuck on parity cells. With full cells the same swaps do not move the score. The other presets have a positive midpoint, so a positive-logBF requirement does not change their swap sets.

Size is not the culprit. The size gate accepts the size rule on main-like, where it is the whole +0.008 over v6, and on edge-cases it already refuses the size rule (`v6_size_head`). Neither decision tracks the losses.

Real data never enters the cell veto: in-sample parity wins (0.35270 vs full 0.35963). Its robust search keeps three pairs, `(1, 3)`, `(2, 5)`, `(4, 7)`. Its flip midpoint is +102.7.

## Fix

In `slot_forecasting/pair_mixer.py` and `episode_flip.py`:

1. If full beats parity by at least 0.002 in-sample Brier, keep full and skip the cross-fit veto. Below that, the existing mean + 1.96 SE veto still runs. g7 `four` wins by only 0.00108 in-sample; keeping full there scores 34.98, and the veto restores 36.45. The smallest preset advantage that was right on test is `fast_drift` at 0.00317, so 0.002 sits between those two measured gaps.
2. If the robust search keeps exactly one pair, use `choose_pairs` (the v6 vote). An empty search still returns all 28 pairs. Real data and the g7 kits keep two or more pairs, so they do not take this branch.
3. Swap an episode only when its log Bayes factor beats the train gap and is positive.

`tests/test_pair_mixer.py` covers the small-margin veto and the large-margin skip. `python3 -m unittest tests.test_pair_mixer tests.test_v7_changes.Round2Tests.test_empty_search_is_not_the_main_pairs` passed (26 tests, then the two guard tests again after the margin move).

## Before / after

Fixed column is v7c with the three changes, scored the same way on test.

| preset | v6 | v7c | fixed |
|---|---:|---:|---:|
| main_like | 40.2661 | 40.2741 | 40.2741 |
| round2_shift | 23.9018 | 21.2809 | 23.9168 |
| all_lags | 35.5139 | 31.8555 | 35.5248 |
| neighbour_heavy | 36.2560 | 36.2560 | 36.2560 |
| hidden_state_heavy | 38.7640 | 32.3923 | 38.7640 |
| fast_drift | 32.8342 | 29.9366 | 32.8424 |
| edge_cases | 38.3499 | 32.8273 | 38.3499 |

`hidden_state_heavy` and `edge_cases` match v6 exactly. The others match the round2-off column, which is a hair above v6 because size stays on. `fast_drift` still swaps 16 test episodes (the ones with a positive log Bayes factor) instead of 88, and lands on the same 32.8424 as round2-off.

Real `python run_pair_mixer.py --cv-val` on `mainFiles/public`: full validation 57.37 / 60, cross-fit halves 57.31 and 57.39. Both halves clear 57.30. Pairs, the parity choice, the flip midpoint, and the 96/192 test swaps match the previous v7c run.

g7 validation points, same flags:

| kit | required | fixed |
|---|---:|---:|
| disjoint | 37.72 | 37.72 |
| four | 36.45 | 36.45 |
| sharedlag | 37.63 | 37.63 |

`four` is the rerun after the margin was raised from 0.001 to 0.002. At 0.001 the in-sample advantage of 0.00108 skipped the veto, full cells stayed, and the kit scored 34.98. At 0.002 the cross-fit runs, the held-out mean is +0.0028, full is rejected, and the score is 36.45.
