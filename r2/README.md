# Round two: one-command build and ranking

When the adaptation kit arrives, run this from the repo root (the worktree or `main`):

```sh
python3 r2/run_round2.py --kit /ABSOLUTE/PATH/TO/KIT/public --out /ABSOLUTE/PATH/TO/OUT
```

Add `--extra NAME=ENTRY_DIR` to include a folder that has its own `predict.py`
(for example our blend entry). The flag can repeat.

What it does, in order:

1. Checks the kit (`train/`, `validation/`, `test/`, `metadata.json`, and the validation answers) and warns
   loudly if `round` is not `adaptation`.
2. Runs the lag structure screen (`orch/r2screen/walsh_screen.py`) and the pair census. It prints whether the
   structure moved from (1,3),(2,5),(4,7).
3. Builds, at most 3 at once, into `OUT/<name>/`:
   - `v17-guards`: `worktrees/_tm/v17g/mainFiles`, copied, `run_strongest.py --cv-val`
   - `v7c`: `origin/avyukth/v7c-size-flip` `mainFiles/`, extracted with `git archive`, `run_pair_mixer.py --cv-val`
   - `v6`: the same branch's `mainFiles/round2/tooling/fallback-v6`, extracted, `run_pair_mixer.py --cv-val`
   - `mixer6`: `worktrees/_tm/mixer6/mainFiles`, copied, `run_pair_mixer.py --cv-val`
   - `blend`: `/Users/saisreed/Desktop/Coding/TamuDatathon26-PSAZZ/adapted_entry` (default `--blend`, untracked
     in the main checkout; `--blend ""` disables it). Runs `predict.py --data KIT --out OUT/blend/submission.csv`,
     then `--split validation --protocol crossfit` for cross-fitted validation predictions.
   - each other `--extra NAME=ENTRY_DIR`: `predict.py --data KIT --out OUT/NAME/submission.csv`
     (plus `--split validation` if the script supports it)
4. Scores validation with the official formula, using the cross-fitted file if it exists (else in-sample), and
   gives half A and half B. It runs paired bootstrap comparisons against `v17-guards` (`orch/paired.py`).
5. Runs `check_submission.py --split test` on each test CSV and records its sha256.
6. Writes `OUT/RANKING.md` with the table and a recommendation. The safest valid entry is uploaded first. A
   second upload is recommended only if its paired CI vs the safe entry excludes 0.

Nothing is uploaded. Source trees are only read: they are copied or extracted into `OUT/_src/`.

Options: `--v7c-ref` (default `origin/avyukth/v7c-size-flip`; run `git fetch origin` first),
`--jobs` (default 3).

Expected main-kit check (metadata `round: main`, so the warning fires and max_points is 60), in-sample:
v17-guards 57.72, v7c 57.37, mixer6 56.19, blend 57.96 (cross-fit). Runtime about 8 minutes on this laptop.
See `_coord/fresh/runbook.md`.

## Pick rule (2026-11-08, after PR #22)
- Default candidate `blend` = `adapted_entry_v17g/` (adapted blend, v17-guards first; main 57.97, adapt_bench mean 38.39/40).
  `blend_v13` = `adapted_entry/` is added as a challenger automatically.
- Upload the default first. Switch to a challenger only if it beats the default by more than 0.1 on the mean AND on both
  cross-fit halves. Paired deltas in RANKING.md are measured against the default.
- Fallback order if the default's test CSV is invalid: blend -> v17-guards -> v7c -> v6 -> mixer6 (add zhill's v18 with --extra).
- Main-kit rehearsal (8.9 min): blend 57.97 (57.98/57.96), blend_v13 57.96, v17-guards 57.69, v7c 57.34, v6 56.26,
  mixer6 56.18; all test CSVs valid; pick = blend, keep it.
