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
- After uploading, commit `gate.json` and record the files' SHA-256 (below). `bundled/validation/` is a copy of the kit's
  `validation/`, so it is not committed; rebuild it with `cp -R $KIT/validation adapted_entry_v17g_rules/bundled/`.
- Main-kit check of this path (Docker eval suite, quick): PASS WITH WARNINGS, 0 FAIL, k-fold 57.86; the 2 WARNs are the
  known future-dated-rows refusal.

## Round-one (main) upload, 2026-11-08
Submitted through the main-round Google Form from `sai/master` (then named `sai/master-test`) @ 65e55a2 (entry `adapted_entry_v17g_rules/`, gate keeps
all three members, main validation 57.97).
- `submission.csv` SHA-256 `076014ef241d8b61b6b608a513e350293a309294c538f0125a35d9e80aaba364` (check_submission: VALID FORMAT, 1536 queries, main, test)
- `entry.zip` SHA-256 `f3dae4e5fdb0a6c83d981f4222bff152a1cf89f6ba091f43a9cdee035250743d` (742 KB; clean `git archive` of the entry + `gate.json` + `bundled/validation/` =
  `mainFiles/public/validation/`)

## Round-two (adaptation) upload, 2026-11-08
Uploaded Avyukth's build of the default `adapted_entry_v17g_rules/` (gate keeps all three members; no challenger cleared the
> 0.1 both-halves rule; validation 38.588/40 cross-fit on the max_points-40 copy of the kit). Files and hashes:
`origin/avyukth/r2-adaptation` @ `a80ddc6`, `r2/adaptation_submission/`.
- `ADAPTATION_submission.csv` SHA-256 `a7ec520dd737c40502bbb277…` (check_submission on the real kit: VALID FORMAT, 1536 queries, adaptation, test)
- `ADAPTATION_entry.zip` SHA-256 `feb33968c7070000891f84547e8a896a3c1e5e0c4538498be840f995d76729d2` (Docker rerun, pinned numpy/scipy, matches to 9.7e-15)
- Cross-check: Sai's independent build (`submission.csv` `108434136029fcf2…`, `entry.zip` `1b467e5d…`) equals it to 3.2e-14, its ZIP
  code is identical, and rerunning that ZIP on the real kit reproduced its CSV byte for byte in 159 s.
