# eval_suite

Runs any entry directory (the code ZIP root: `README.md`, `reproduce.json`, optional
`requirements.txt`, plus code) through the documented requirements and writes a report.
It never imports a team model. The suite is built from the kit's documents
(`RULES.md`, `DATA.md`, `README.md`, `SUBMISSION_README.md`, `EXPERIMENTS.md`,
`public/metadata.json`) and kit code. `REQUIREMENTS.md` and `requirements.py` list
every testable requirement with its source line and the check IDs that cover it.

## Run it

From the kit root (the directory that contains `eval_suite/` and `public/`):

```bash
# quick: contract, k-fold real-validation score, determinism, 3 causality probes (about seconds)
python3 -m eval_suite run --entry path/to/entry --suite quick --mode local --out runs/eval/NAME

# full: quick + every perturbation + packaging + dataset checks + generator presets
python3 -m eval_suite run --entry path/to/entry --suite full --mode docker --out runs/eval/NAME

# paired comparison against a second entry on the same datasets
python3 -m eval_suite run --entry path/to/A --compare path/to/B --suite full --mode docker --out runs/eval/A-vs-B
```

Options: `--seed` (default 0), `--k` (validation folds, default 4),
`--bootstrap-repeats` (default 1000), `--kit` (kit `public/` directory, default `public/`),
`--keep-work` (keep mounted dataset copies for debugging). Exit status is 0 for no FAIL,
1 for any FAIL, 2 for a suite error with no FAIL.

Tests: `python3 -m unittest discover -s eval_suite/tests -t . -v` (the protocol and runner
tests use a two-episodes-per-condition kit subset and finish in about 10 seconds).

## What each level runs

| Step | quick | full | Requirement IDs covered (examples) |
|---|---|---|---|
| Entry layout and `reproduce.json` schema, ZIP root, pins | yes | yes | CON-REPRODUCE-SCHEMA, PKG-ZIP-ROOT, PKG-REQUIREMENTS-PINNED |
| Kit test run: `train/` and `test/` mounted, no validation answers. Forecast format, UTF-8, strict CSV, runtime, output file | yes | yes | CON-CSV-*, CON-QUERY-SET, CON-PROB-*, CON-EXIT-ZERO, ROB-RUNTIME-LIMIT |
| k-fold real validation: each fold mounts its visible episodes plus the scored episodes' queries and history, with no answers. Predictions are stitched and scored by `scoring.official_score`. Matches the kit's `score_round` | yes | yes | SCO-FORMULA, SCO-MATCHES-KIT, GDE-07, DAT-12 |
| Determinism: fold 0 rerun, max abs difference 1e-8 after row normalisation | yes | yes | ROB-DETERMINISM, RUL-37 |
| Causality on fold 0: future ticks, row order, answers hidden | yes | yes | CAU-FUTURE-TICKS, ROB-ROW-ORDER, CAU-ANSWERS-REMOVED |
| All perturbations: cutoff-tick rows for other machines, condition removal, ID renaming, missing vs zero, cross-episode (`identical_subset`) | no | yes | CAU-CUTOFF-OTHERS, CAU-CONDITIONS-REMOVED, ROB-ID-RENAME, CAU-MISSING-PROBE, CAU-CROSS-EPISODE |
| Fresh folder: the reproduce command runs from a copy with nothing else in the folder, and matches the kit test run | no | yes | ENT-FRESH-FOLDER, ENT-MATCHES-SUBMITTED |
| Packaging: `package_entry` then `check_zip` (needs `datachecks.py`) | no | yes | PKG-* |
| Dataset checks on the kit and on each generator preset (needs `datachecks.py`) | no | yes | DAT-*, CON-ROUND-METADATA |
| Mount checks: no answers at the mount root or `test/`, no forecast file (needs `datachecks.py`; a suite-internal check always runs) | partial | yes | ROB-MOUNT-NO-ANSWERS, ROB-NO-FORECAST-INPUT |
| Generator presets (`main_like`, `round2_shift`, `all_lags`, `neighbour_heavy`, `hidden_state_heavy`, `fast_drift`, `edge_cases`): generated with `episodes-per-condition` from metadata (24), scored against `_truth` | no | yes (needs `generator.py`) | GEN-* (not emitted by the suite), presets are scores |
| `--compare`: the second entry runs the same datasets, and `paired_compare` reports the points difference | k-fold only | k-fold and presets | paired table |

Docker-only limits (`ROB-OFFLINE`, `ROB-CPU-LIMIT`, `ROB-MEM-LIMIT`, `ROB-READONLY-MOUNTS`,
`ENT-DEPS-PREINSTALL`) are recorded in docker mode from the kit test run. In local mode
they are NOT RUN with the reason.

Scores are never computed on the kit test split, because its answers are private.
Every score comes from validation or from generated `_truth`.

## How to read `report.md`

1. **Verdict line.** FAIL if any requirement or any check failed. PASS WITH WARNINGS if
   a check warned. Local mode is always labelled NON-AUTHORITATIVE.
2. **Requirement coverage.** One row for every requirement in `requirements.py`, so none
   is silently dropped. Status is the worst status of its check IDs:
   - PASS, INFO: verified (INFO is information only, for example ENT-MAIN-DATA-MOUNT).
   - WARN: a soft rule failed (for example PKG-README-SEASON, or a missing-vs-zero
     treatment). A refused variant is also WARN, not PASS.
   - FAIL: the entry violates the requirement. The evidence column names the check.
   - NOT RUN: a check did not run. The reason is always given: quick level, local mode,
     missing module, or rule-level (no file check exists).
   - MANUAL: deadlines, form process, provenance. Not checkable from files.
3. **Scores.** Points = max_points / (1 + R), with R the mean over the four conditions of
   E[c]/B[c]. Lower ratios are better. The CI is a condition-stratified episode bootstrap,
   and the half-split columns show the spread over random halves. Points are out of 60 for
   the main round.
4. **Slices.** Per-condition, machine type, signal, history kind and cutoff bin for the
   real validation. Ratio above 1 means worse than the frozen reference in that slice.
5. **Paired comparison.** Only with `--compare`. A - B with a paired bootstrap CI.
   A CI that excludes zero is a real difference on these episodes.
6. **Runtime and memory.** Wall time per run against the 1800 s limit. Peak memory is
   available in docker mode only.
7. **Caveats.** Read these before trusting a number.

`report.json` has the same content in machine-readable form: `requirements`, every
`checks` record (with `covers`, `status`, `message`, `source`), `scores`, `paired`, `runs`,
`errors` and `caveats`.

`predictions/` holds every entry CSV the suite produced, so a number can be
recomputed without rerunning the entry.

## Runtime expectations

- Local quick on the full kit: about 5 s for a trivial entry. The suite's own work
  (bootstrap, stitching, checks) is a few seconds. Real entries add roughly one run per fold,
  so the cost is about 8 entry runs.
- Local full: about 25 entry runs (folds, rerun, 8 perturbations, fresh folder, generator
  presets and, with `--compare`, the same again for the second entry). Plan on several minutes
  for a fast entry.
- Docker: adds one image build per distinct `requirements.txt` (cached by hash), then
  container start-up per run. The organizer's limit for each run is 1800 s, so a worst-case
  full run with a slow entry can take hours. Use `--suite quick` for iteration.

## Known limits

- Local mode is not the organizer's environment. It has no network block, no read-only mounts,
  no CPU or memory limits, and no requirements install. Use docker mode for authoritative runs.
- `entry.py` maps `"image": "python"` to `python:3.12-slim`, but the rules say the base
  environment is Python 3.13 (GDE-21). ENT-BASE-PY313 is therefore not probed, and docker runs
  use 3.12 until `entry.py` changes.
- The k-fold score is comparable across entries but is not identical to training on all of
  validation. Each fold's entry sees the other folds' validation answers as training data.
- The generator presets are spec-derived approximations, not the organizer's data.
