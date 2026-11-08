# Combo model verification

The entry combines 20% of the rule forecast with 80% of pair-mixer v3, in payout
order 0, 1, 2, 5, 20. The copied mixer is pinned to teammate commit `6a8098c`;
the later `81d4fff` model is outside this reproduction. Verification adds no
dependencies: the CLI and comparisons use Python's standard library; scoring
imports the existing harness, which already requires NumPy.

## Status: tooling ready, predictor verification pending

On 2026-11-07, from the shared `sai/combo` checkout:

- Copied pair-mixer tests: **10 passed**, 0 failures/errors, 0.115 seconds.
- No tests for `rule_model.py` were found; no rule-model tests were invented.
- Verification tests: **8 passed**, 0 failures/errors, 0.382 seconds. Coverage
  includes shuffled IDs, mismatched IDs, duplicates, invalid probabilities,
  strict tolerance, audit repeat guards, runtime timeout and memory termination.
- Saved forecast compared against itself: **768 queries, exactly 0 maximum
  absolute difference in every probability column**.
- Saved forecast dev score: **54.77398565347036**, 95% episode-bootstrap CI
  **[53.737327140644695, 55.714773653299446]**, 72 episodes.
- Dev ratios: stable **0.06556259302236582**; short_history
  **0.14885998143040835**; drift **0.07603375601643553**; shared_change
  **0.09118574170660068**.
- The bounded subprocess runner reproduced that score in **0.234 seconds**.
- Full predictor runs, submission checks and audit scoring are **pending** the
  coordinator's explicit readiness message. No audit score has been requested.

The initial repo-root unittest invocation produced one import error because
`slot_forecasting` was not on the import path. The documented invocation below,
from `mainFiles`, passed all ten tests without modifying the copied sources.
An initial attempt to cap address space with `RLIMIT_AS` failed on this macOS
host; the runner now monitors resident memory instead, verified by termination
of a synthetic child exceeding a deliberately small memory budget.

## Commands

Run from the repository root unless a command says otherwise:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
(cd mainFiles && nice -n 10 timeout 600 python3 -m unittest discover -s tests -v)
nice -n 10 timeout 600 python3 -m unittest discover -s mainFiles/combo/verify -v
nice -n 10 timeout 600 python3 mainFiles/combo/verify/compare_to_tm.py docs/sai/preds/TM_COMBINED3_validation.csv
nice -n 10 timeout 600 python3 mainFiles/combo/verify/score_dev.py docs/sai/preds/TM_COMBINED3_validation.csv
```

`compare_to_tm.py <csv>` matches rows by query ID, rejects malformed data and
requires the maximum absolute difference to be strictly below `1e-6` in every
column. It rejects missing/extra IDs, duplicate IDs, nonfinite/out-of-range
probabilities and rows whose sum differs from 1 by more than `1e-6`. Optional
`--reference <csv> --tolerance <value>` also supports deterministic comparisons.
It does not renormalize predictions to hide differences.

`score_dev.py <csv>` resolves the input path, changes to `docs/sai`, and calls
`load_split('validation')`, `read_pred_csv(...)`, and
`evaluate(..., 'dev')`. It prints the harness's complete JSON result, including
its default 1,000-bootstrap CI. No scoring formulas are duplicated.

Once the coordinator says the predictor is ready:

```sh
sh mainFiles/combo/verify/run_all.sh --out .pane/combo-verification
```

This chains validation prediction, saved-forecast comparison, dev scoring,
validation submission checking, two independent test predictions, their strict
`<1e-8` comparison, and test submission checking. Any failed command stops the
chain. The official validation checker reports its score over all 96 public
validation episodes; that differs from the harness's 72-episode dev scope.
`run_all.sh` never calls the reserved audit scorer.

For just two independent test runs and their comparison:

```sh
sh mainFiles/combo/verify/determinism.sh --out .pane/combo-determinism
```

Both wrappers run commands serially with niceness 10 and native thread counts
set to 1. This single-process stdlib predictor stays within the two-CPU budget.
Each child has a 600-second wall limit; the determinism pair shares a 1,800-second
wall budget. The macOS runner samples child resident memory every 250 ms and
terminates its process group above 16 GiB; brief allocation spikes between
samples may escape detection. It prints each command's runtime and saves
`determinism.json` with differences and both prediction runtimes. Output
directories must be new, preserving earlier evidence rather than overwriting it.

## Audit: one attempt only

Only after the full pipeline has passed and all model choices are frozen:

```sh
nice -n 10 timeout 600 python3 mainFiles/combo/verify/score_audit.py .pane/combo-verification/validation.csv
```

The script uses `scope='audit', who='combo-sai'`. Before importing/loading the
harness it exclusively creates `.pane/combo-audit.once.json`, recording the
prediction file's SHA-256 and the attempt time. A second attempt fails, including
concurrent calls. It also refuses if `combo-sai` already appears in the harness
access log. A failed attempt consumes the marker; do not delete it to retry.
Successful output is printed and saved as `.pane/combo-audit.result.json`, with
24-episode points, CI, and stable/short_history/drift/shared_change ratios.
The unit tests exercise only synthetic temporary markers and logs, never audit
predictions or targets. Never tune the entry using this score.

## Handoff and infrastructure

Only `mainFiles/combo/verify/` and this document belong to Codex. Claude owns the
predictor, entry metadata, README and archive; copied teammate code is unchanged.
The shared checkout is intentional. No branch switching, rebasing, stashing,
merging, resetting or pushing is permitted. This verification-only commit has
no semver impact and opens no PR. The coordinator owns any aggregate changelog
entry; none was edited here because it is outside the assigned paths.

Existing source/test layout and a remote are present. No CI workflow, lint/format
configuration, dependency manifest or `.env.example` was found. The participant
README is in `mainFiles/`; there is no root README. Branch protection was not
queried because network access is prohibited; the playbook states it is absent.
No issue was supplied. Shell syntax, Python compilation and whitespace checks
supplement unittest; there is no configured lint command.
