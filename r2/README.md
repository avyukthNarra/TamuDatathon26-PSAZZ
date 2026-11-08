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
   - `v17-guards`: `origin/zhill/v17-round2-guards` `mainFiles/`, extracted with `git archive`, `run_strongest.py --cv-val`
   - `v7c`: `origin/avyukth/v7c-size-flip` `mainFiles/`, extracted with `git archive`, `run_pair_mixer.py --cv-val`
   - `v6`: the same branch's `mainFiles/round2/tooling/fallback-v6`, extracted, `run_pair_mixer.py --cv-val`
   - `mixer6`: commit `814cb60` `mainFiles/` (frozen snapshot), extracted, `run_pair_mixer.py --cv-val`
   - `blend`: `adapted_entry_v17g_rules/` in this repo (default `--blend`; `--blend ""` disables it); `blend_v13` =
     `adapted_entry/` runs as a challenger. Runs `predict.py --data KIT --out OUT/blend/submission.csv`,
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
`--v17g-ref`, `--mixer6-ref`, `--v17g-src PATH` and `--mixer6-src PATH` (local `mainFiles/` folders to copy
instead of the git refs), `--jobs` (default 3). Candidates whose source is unavailable are skipped and noted in RANKING.md.

Expected main-kit check (metadata `round: main`, so the warning fires and max_points is 60), in-sample:
v17-guards 57.72, v7c 57.37, mixer6 56.19, blend 57.96 (cross-fit). Runtime about 8 minutes on this laptop.
See `_coord/fresh/runbook.md`.

## Pick rule (2026-11-08, after PR #22)
- Default candidate `blend` = `adapted_entry_v17g_rules/` (adapted blend, v17-guards first, exact filter also learns 3-lag XOR and AND/OR rules from train; main 57.97, adapt_bench K1–K9 unchanged, K10 32.08 → 35.38).
  `blend_v13` = `adapted_entry/` is added as a challenger automatically.
- Upload the default first. Switch to a challenger only if it beats the default by more than 0.1 on the mean AND on both
  cross-fit halves. Paired deltas in RANKING.md are measured against the default.
- Fallback order if the default's test CSV is invalid: blend -> blend_v17g -> v17-guards -> v7c -> v6 -> mixer6 (add zhill's v18 with --extra).
- Main-kit rehearsal (8.9 min): blend 57.97 (57.98/57.96), blend_v13 57.96, v17-guards 57.69, v7c 57.34, v6 56.26,
  mixer6 56.18; all test CSVs valid; pick = blend, keep it.

## Kit-day steps (2026-11-08, after PR #27)
Run in order; the gate must finish before run_round2.py starts, because the blend reads `gate.json` at startup.
```bash
KIT=/ABS/PATH/kit/public; OUT=/ABS/PATH/out
python3 r2/gate_members.py --entry adapted_entry_v17g_rules --kit $KIT   # ~4.5 min: writes gate.json + bundled/validation/
python3 r2/run_round2.py --kit $KIT --out $OUT --jobs 6                  # ~8 min: OUT/RANKING.md, OUT/blend/submission.csv
# code ZIP from a clean copy (no __pycache__) plus the two generated files
mkdir -p $OUT/zip && git archive HEAD adapted_entry_v17g_rules | tar -x -C $OUT/zip
cp adapted_entry_v17g_rules/gate.json $OUT/zip/adapted_entry_v17g_rules/
cp -R adapted_entry_v17g_rules/bundled $OUT/zip/adapted_entry_v17g_rules/
python3 mainFiles/build_entry.py --source $OUT/zip/adapted_entry_v17g_rules --out $OUT/entry.zip
python3 mainFiles/check_submission.py $OUT/blend/submission.csv
```
- `bundled/validation/` makes the rerun identical with or without `/data/validation/targets.csv`, or with no `/data/validation/` at all
  (checked on main: all three byte-identical).
- If the gate drops a member, the blend CSV from run_round2.py already uses it (same `gate.json`).
- After uploading, commit `gate.json` and `bundled/` so the repo reproduces the uploaded CSV.
- Main-kit check of this path (Docker eval suite, quick): PASS WITH WARNINGS, 0 FAIL, k-fold 57.86; the 2 WARNs are the
  known future-dated-rows refusal.
