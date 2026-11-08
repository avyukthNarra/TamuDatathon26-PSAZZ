# m5 presets: which v7c piece hurts synthetic presets

Status: interim. Seed-0 test scores match the h10 report on the finished rows. Ablation still running. Started 2026-11-08 ~01:43 local.

## Question

v7c scores 57.36 on real validation (+1.09 vs v6) but loses to v6 on 5 of 7 eval-suite presets (h10 report, seed 0, test scored against `_truth`):

| preset | max | v7c | v6 | v7c - v6 |
|---|---:|---:|---:|---:|
| main_like | 60 | 40.27 | 40.27 | +0.01 |
| round2_shift | 40 | 21.28 | 23.90 | -2.62 |
| all_lags | 60 | 31.86 | 35.51 | -3.66 |
| neighbour_heavy | 60 | 36.26 | 36.26 | 0.00 |
| hidden_state_heavy | 60 | 32.39 | 38.76 | -6.37 |
| fast_drift | 60 | 29.94 | 32.83 | -2.90 |
| edge_cases | 60 | 32.83 | 38.35 | -5.52 |

## Method (in progress)

Presets: `python3 -m eval_suite.generator --preset <name> --seed 0 --episodes-per-condition 24` from `worktrees/eval-main/mainFiles`. Suite uses the kit's validation count (24) for every split (`eval_suite/cli.py`). Score **test** `submission.csv` against `_truth/test` with `score_contest_files` (same formula as `score_bundle`: points = max_points / (1+R)). `round2_shift` max_points is 40.

Variants via `mainFiles/run_pair_mixer.py --score-only` flags (`--size/--no-size`, `--flip/--no-flip`, `--round2/--no-round2`; size gate follows `--size-gate`, cell-mode gate is inside `adapt` when round2 selects full):

| variant | size | size gate | flip | round2 (includes cell-mode gate) |
|---|---|---|---|---|
| v6 | off | off | off | off |
| v7c | on | on | on | on |
| v7c-flip | on | on | off | on |
| v7c-size | off | off | on | on |
| v7c-round2 | on | on | on | off |

## Hypothesis (not yet measured)

Largest drops are hidden_state_heavy and edge_cases, then all_lags. Neighbour_heavy ties, so the neighbour path is unlikely to be the sole culprit. Candidates:

- flip gate: a hidden-state or fast-drift level shift can look like a parity flip and swap the wrong direction.
- size gate: accepts a size rule whose lag is really a win-indicator lag (generator `size_shift` is tied to interactions), which hurts when the lag set is not the real-data one (`all_lags`, `round2_shift` lags (2,3,5,7)).
- round2 pair picker: `choose_pairs_robust` / AND-OR interactions can lock a bad pair; cell-mode full can overfit sparse edge episodes.

## Partial table (test points vs `_truth/test`)

Finished rows reproduce h10 to 4 decimals. Dash means that variant is still running.

| preset | max | v6 | v7c | v7c-flip | v7c-size | v7c-round2 |
|---|---:|---:|---:|---:|---:|---:|
| main_like | 60 | 40.2661 | 40.2741 | 40.2741 | 40.2661 | 40.2741 |
| round2_shift | 40 | 23.9018 | 21.2809 | 21.2091 | 21.2637 | 23.9168 |
| all_lags | 60 | 35.5139 | 31.8555 | 31.5735 | 31.8568 | 35.5248 |
| neighbour_heavy | 60 | 36.2560 | 36.2560 | 36.2560 | 36.2560 | 36.2560 |
| hidden_state_heavy | 60 | 38.7640 | 32.3923 | 32.3911 | 32.3923 | running |
| fast_drift | 60 | running | running | running | running | running |
| edge_cases | 60 | | | | | |

## Culprit so far: round2 cell-mode gate, not size or flip

On `round2_shift`, dropping flip or dropping size leaves the loss (21.21 and 21.26). Dropping round2 returns 23.92, a hair above v6. Lag pairs are the same six odd-lag pairs in every variant. Size keys match. The round2 path's cell-mode gate is what changes the model: in-sample cutoff Brier prefers full (0.55343 vs parity 0.56594), but the cross-fit mean full-minus-parity is -0.0065 with upper CI +0.0103, so the gate forces parity. v6 has no gate and keeps full.

That matters because the generator draws single, xor, and, and or interactions. And and or are not functions of agree/differ, so the 4-cell (full) experts can represent them and the 2-cell experts cannot.

On `main_like` the same gate keeps full (mean -0.0131, upper -0.0005), and round2 does not move the score. The +0.008 over v6 is the size head. Flip swaps 2 of 96 test episodes and does not change the printed points.

Real data never calls this gate: in-sample parity wins (cutoff Brier 0.35270 vs full 0.35904, from `m3-winport` `m3-final.log`). g7 `four` does call it and rejects full because the cross-fit mean is positive (+0.0028), which is the case the gate was written for. g7 disjoint and sharedlag pick parity in-sample, so the gate does not run.

## Fix to test (not applied yet; ablation processes still import the current module)

Keep full when the train cross-fit mean delta is negative, instead of requiring mean + 1.96 SE < 0. `four` still rejects (mean > 0). Real data and the other two sims still never enter the gate. Then re-score the presets, `run_pair_mixer.py --cv-val` on real public, and the three g7 publics.
