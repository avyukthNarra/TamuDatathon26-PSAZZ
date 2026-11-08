# Round-two runbook (Sunday, 10:30–12:00 Central)

This procedure refits the candidate on the newly released round's own public
data, checks the test forecast, and leaves the two upload files in
`c3-codex/SUBMIT_R2/`. It never uploads or submits anything. The main-round
`public/` bundled with a source checkout is deliberately excluded; every build
receives the explicit new kit path.

## 10:30–10:40 — receive and identify the kit

1. Download and fully unzip the adaptation kit to a new directory. Locate its
   `public/` folder. Confirm it contains `metadata.json` and `train/`,
   `validation/`, and `test/`.
2. Keep the original extracted kit untouched. Note the round/season from its
   `public/metadata.json`.
3. Open a terminal and run `python3 --version` (Python 3.11+). No Python
   packages or external timeout utility are required.

## 10:40–10:55 — build and verify

From any working directory, run:

```sh
/Users/avynarra/Documents/Datathon/research_pf/c3-codex/round2.sh \
  /absolute/path/to/new-kit/public
```

The script copies only the runtime model code and entry templates into a fresh
isolated workspace, uses the candidate source from `c3-codex/v7c-mainFiles`, passes
the new public folder explicitly, and requests cross-fitted validation output.
The runner uses its fixed train-only, two-fold episode cross-fit gate to choose
between the candidate size rule and the round-two v6 size head. It prints the
conditional Brier delta, uncertainty bound, selected head, train-discovered
win pairs and validation score; it also writes those diagnostics to
`size_rule.json`. It then prints size lags and learned atoms from the fit, plus
the public checker result. The fresh process reproduction check must report a
maximum prediction difference no greater than 1e-8.

The g7 stress fixtures have static per-machine payout-size atoms. They contain
no true size lag, so candidate size-lag lists there are spurious fit results,
not discoveries. The train-only gate selected the candidate size rule on the
main train split, but selected the v6 size head on all three g7 train splits
(disjoint, four, and sharedlag). This is a useful stress check, not a forecast
for the real adaptation round. The gate choice is fixed by a predeclared
train-only conditional-size Brier criterion; do not change it based on the
validation score.

The full-cell guard is separate from the size-head gate. On g7 disjoint and
sharedlag, the runner's initial train cutoff comparison selected parity. On
four it slightly preferred full in-sample (0.37534 vs. 0.37642 Brier), but the
fixed whole-episode cross-fit guard rejected full: full-minus-parity episode
Brier +0.002817, SE 0.006580, upper 95% bound +0.015713. All three final
rehearsals therefore used parity and the v6 size head. Their synthetic public
scores were 37.72/40 (disjoint), 36.45/40 (four), and 37.63/40 (sharedlag).
Each emitted an entry package whose fresh-process reproduction max difference
was 0, and the supplied test CSV passed the public checker for all 64 queries.
Detailed logs, predictions, checker outputs, source hashes, and archived
artifact pairs are under `archive/v7c-rehearsals/`. These fixtures are
simulator checks, not performance claims for the real kit.

To force a plain-v6 build manually, use the frozen fallback source explicitly:

```sh
ROOT=/Users/avynarra/Documents/Datathon/research_pf/c3-codex
"$ROOT/round2.sh" /absolute/path/to/new-kit/public \
  --source "$ROOT/tooling/fallback-v6"
```

The target is under ten minutes. The candidate build has a six-minute timeout,
the plain v6 fallback has a three-minute timeout, and each public check has a
45-second timeout. If candidate build, reproduction, artifact creation, or
test checking fails, the script repeats the same procedure with the frozen,
allowlisted plain v6 snapshot at `c3-codex/tooling/fallback-v6/`. That snapshot
was copied from the read-only `/Users/avynarra/Documents/Datathon/worktrees/pranav-5629/mainFiles`
source on 2026-11-07. Its per-file SHA-256 provenance is recorded in
`tooling/fallback-v6/SOURCE_PROVENANCE.json`. It contains runtime code and
templates only; it has no public data, priors, runs, bytecode, or git metadata.
The fallback uses the same new kit path. If both attempts fail, no `SUBMIT_R2`
directory is left behind; the previous pair, if any, and failure logs remain
in `c3-codex/.round2-work/` for diagnosis.

## 10:55–11:10 — inspect the result

Confirm the final line says `READY` and identify whether it says `candidate` or
`fallback-v6`. Check that both files exist and are nonempty:

```sh
ls -lh /Users/avynarra/Documents/Datathon/research_pf/c3-codex/SUBMIT_R2
```

Read the printed values against the round's known data story. Treat any pair or
lag list that looks unfamiliar as a prompt to inspect the logs and train
support, not as a reason to insert main-round constants. The script reports
size-rule lags and atoms learned by the candidate; g7's synthetic stress
simulator has static payout sizes, so its lag list cannot establish true size
lag recovery. Do not select between models by repeatedly tuning on validation.

The fallback protects against a broken candidate build and invalid artifacts;
it is not an automatic score-based model selection. The script never copies a
partially built forecast into `SUBMIT_R2/`.

## 11:10–11:25 — final local checks

Check that `SUBMIT_R2/submission.csv` is from this adaptation kit (the checker
prints its season and query count), and that `entry.zip` accompanies it. Keep
the exact pair together. If there was a fallback, note that the uploaded code
is plain v6 and the candidate's failure reason from its log.

If time allows, repeat the checker directly with the same kit as a final
independent confirmation:

```sh
python3 -B /Users/avynarra/Documents/Datathon/research_pf/c3-codex/v7c-mainFiles/check_submission.py \
  /Users/avynarra/Documents/Datathon/research_pf/c3-codex/SUBMIT_R2/submission.csv \
  --data /absolute/path/to/new-kit/public --split test
```

If this says the query IDs do not match, stop: do not upload. Rerun the script
with the correct kit path and verify its metadata output.

## 11:25–12:00 — upload buffer

Use the adaptation submission form announced by the organizers. Attach exactly
`SUBMIT_R2/submission.csv` and `SUBMIT_R2/entry.zip`, submit before the stated
deadline, and wait for the form confirmation page. Keep the files and
confirmation. The local checker does not upload anything and cannot confirm
form receipt.

## Recovery notes

- `public folder not found` or missing split: unzip the kit again and pass the
  directory that directly contains `train/`, `validation/`, and `test/`.
- Missing candidate runner/checker: candidate source is incomplete. The script
  reports the error and uses the frozen fallback automatically. Use
  `--source /path/to/complete/mainFiles` if a complete candidate lives elsewhere.
- Candidate failure followed by fallback failure: inspect each
  `.round2-work/run.*/{candidate,fallback-v6}/build.log` and `check.log`.
  No new submission has been published.
- Timeout: do not extend beyond the round window without checking remaining
  time. The six-minute candidate plus three-minute fallback budget leaves room
  for manual review.
- No score-based retuning in the final window. Validation is for diagnosis;
  preserve the predeclared candidate unless it cannot build or pass checks.
