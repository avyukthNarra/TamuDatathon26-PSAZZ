# eval_suite report: entry

**VERDICT: PASS WITH WARNINGS - 0 requirement(s) FAIL, 3 WARN, 54 PASS, 3 INFO, 19 NOT RUN, 9 MANUAL. Level full, docker run (VM memory and quotas as reported).**

Entry `<team-best entry dir>`. Seed 0, k=4, bootstrap repeats 1000. Wall time 1011.9s. Generated 2026-11-08T00:51:28+00:00.

## 1. Requirement coverage
54 PASS, 0 FAIL, 3 WARN, 3 INFO, 19 NOT RUN, 9 MANUAL of 88 requirements.

| ID | Category | Status | Check IDs | Source | Reason / evidence |
|---|---|---|---|---|---|
| RUL-01 | scoring | PASS | `SCO-MAX-POINTS` | RULES.md:9 | SCO-MAX-POINTS: PASS |
| RUL-02 | scoring | PASS | `SCO-MAX-POINTS` | RULES.md:11 | SCO-MAX-POINTS: PASS |
| RUL-03 | contract | MANUAL | - | RULES.md:11 | Deadline; organizer process, not a file property. Same deadline in README.md:15 and :273. |
| RUL-04 | scoring | NOT RUN | `SCO-NO-IMPROVEMENT-BONUS` | RULES.md:13 | SCO-NO-IMPROVEMENT-BONUS: rule-level: round combination is organizer code (rounds.combine_rounds), not a file check |
| RUL-05 | scoring | PASS | `SCO-FORMULA` | RULES.md:17 | SCO-FORMULA: PASS |
| RUL-06 | scoring | PASS | `SCO-REFERENCE-SOURCE`, `SCO-REFERENCE-REPRO` | RULES.md:17 | SCO-REFERENCE-SOURCE: PASS; SCO-REFERENCE-REPRO: PASS |
| RUL-07 | scoring | NOT RUN | `SCO-FORMULA`, `SCO-REF-FLOOR` | RULES.md:20 | partly verified; the rest did not run; SCO-REF-FLOOR: not checked: needs a synthetic zero-reference case (covered by scoring unit tests) |
| RUL-08 | scoring | NOT RUN | `SCO-FINAL-SUM` | RULES.md:23 | SCO-FINAL-SUM: rule-level: round combination is organizer code (rounds.combine_rounds), not a file check |
| RUL-09 | scoring | PASS | `SCO-CONDITION-EQUAL` | RULES.md:26 | SCO-CONDITION-EQUAL: PASS |
| RUL-10 | scoring | PASS | `SCO-REFERENCE-SOURCE` | RULES.md:26 | SCO-REFERENCE-SOURCE: PASS |
| RUL-11 | scoring | PASS | `SCO-TRUTH-MAX` | RULES.md:26 | SCO-TRUTH-MAX: PASS |
| RUL-12 | scoring | PASS | `SCO-REF-HALF` | RULES.md:26 | SCO-REF-HALF: PASS |
| RUL-13 | scoring | NOT RUN | `SCO-OUTCOME-INDEPENDENT` | RULES.md:28 | SCO-OUTCOME-INDEPENDENT: by construction: scored probabilities never read the outcome column; no perturbation run |
| RUL-14 | scoring | NOT RUN | `SCO-TIE-RANK` | RULES.md:30 | SCO-TIE-RANK: ranking-level: tie scheme is unannounced (RUL-14); no ranking code in this suite |
| RUL-15 | robustness | PASS | `ENT-MATCHES-SUBMITTED`, `ENT-FRESH-FOLDER`, `ROB-ID-RENAME` | RULES.md:30 | ENT-MATCHES-SUBMITTED: PASS; ENT-FRESH-FOLDER: PASS; ROB-ID-RENAME: PASS |
| RUL-16 | robustness | PASS | `ROB-MOUNT-NO-ANSWERS`, `CAU-ANSWERS-REMOVED` | RULES.md:34 | ROB-MOUNT-NO-ANSWERS: PASS x2; CAU-ANSWERS-REMOVED: PASS |
| RUL-17 | contract | MANUAL | - | RULES.md:34 | Provenance of data is not checkable from files; manual review of imports and data paths. |
| RUL-18 | causality | WARN | `CAU-FUTURE-TICKS` | RULES.md:35 | CAU-FUTURE-TICKS: WARN (refused: entry exited 1 on the variant (slot_forecasting.schema.ValidationError: Future data leak: tick 202...) |
| RUL-19 | causality | INFO | `CAU-MISSING-PROBE` | RULES.md:35 | CAU-MISSING-PROBE: INFO |
| RUL-20 | data-fact | NOT RUN | `GEN-LAG-RANGE` | RULES.md:36 | GEN-LAG-RANGE: generator-level checks (GEN-*) are not emitted by the suite; presets are scored only |
| RUL-21 | causality | PASS | `CAU-CONDITIONS-REMOVED` | RULES.md:36 | CAU-CONDITIONS-REMOVED: PASS |
| RUL-22 | data-fact | NOT RUN | `GEN-MECHANISM-FAMILY` | RULES.md:36 | GEN-MECHANISM-FAMILY: generator-level checks (GEN-*) are not emitted by the suite; presets are scored only |
| RUL-23 | robustness | PASS | `ROB-OFFLINE` | RULES.md:38 | ROB-OFFLINE: PASS |
| RUL-24 | contract | PASS | `CON-CSV-COLUMNS`, `CON-CSV-STRICT` | RULES.md:43 | CON-CSV-COLUMNS: PASS x14; CON-CSV-STRICT: PASS x14 |
| RUL-25 | contract | PASS | `CON-QUERY-SET` | RULES.md:49 | CON-QUERY-SET: PASS x14 |
| RUL-26 | contract | PASS | `CON-PROB-VALUES` | RULES.md:49 | CON-PROB-VALUES: PASS x28 |
| RUL-27 | contract | PASS | `CON-PROB-SUM` | RULES.md:49 | CON-PROB-SUM: PASS x14 |
| RUL-28 | packaging | PASS | `CON-CSV-SIZE`, `PKG-ZIP-SIZE` | RULES.md:51 | CON-CSV-SIZE: PASS x14; PKG-ZIP-SIZE: PASS x2 |
| RUL-29 | packaging | PASS | `PKG-ZIP-ROOT` | RULES.md:51 | PKG-ZIP-ROOT: PASS x3 |
| RUL-30 | scoring | NOT RUN | `SCO-MISSING-ZERO` | RULES.md:53 | SCO-MISSING-ZERO: rule-level: a missing round entry scores zero in the organizer's ranking, not in this suite |
| RUL-31 | scoring | NOT RUN | `SCO-MAIN-REQUIRED` | RULES.md:53 | SCO-MAIN-REQUIRED: ranking-level: overall ranking flag is organizer code, not a file check |
| RUL-32 | contract | MANUAL | - | RULES.md:53 | Form-process rule. |
| RUL-33 | contract | MANUAL | - | RULES.md:55 | CONFLICTS with RUL-30 / RULES.md:53 (invalid round-two entry earns zero). See REQUIREMENTS.md contradictions. |
| RUL-34 | contract | PASS | `CON-IMAGE-PYTHON` | RULES.md:55 | CON-IMAGE-PYTHON: PASS x2 |
| RUL-35 | packaging | PASS | `PKG-REQUIREMENTS-PINNED` | RULES.md:55 | PKG-REQUIREMENTS-PINNED: PASS x2 |
| RUL-36 | robustness | PASS | `ROB-RUNTIME-LIMIT`, `ROB-CPU-LIMIT`, `ROB-MEM-LIMIT` | RULES.md:55 | ROB-RUNTIME-LIMIT: PASS x13; ROB-CPU-LIMIT: PASS; ROB-MEM-LIMIT: PASS |
| RUL-37 | robustness | PASS | `ROB-DETERMINISM` | RULES.md:55 | ROB-DETERMINISM: PASS |
| DAT-01 | data-fact | PASS | `DAT-SCHEMA-COLUMNS` | DATA.md:9 | DAT-SCHEMA-COLUMNS: PASS x8 |
| DAT-02 | contract | PASS | `DAT-TEST-SPLIT-FILES` | DATA.md:17 | DAT-TEST-SPLIT-FILES: PASS x8 |
| DAT-03 | data-fact | PASS | `DAT-QUERY-ENUMS` | DATA.md:21 | DAT-QUERY-ENUMS: PASS x8 |
| DAT-04 | data-fact | PASS | `DAT-CUTOFF-RANGE` | DATA.md:23 | DAT-CUTOFF-RANGE: PASS x8 |
| DAT-05 | data-fact | NOT RUN | `GEN-HISTORY-SHAPE`, `DAT-HISTORY-CAUSAL` | DATA.md:23 | partly verified; the rest did not run; GEN-HISTORY-SHAPE: generator-level checks (GEN-*) are not emitted by the suite; presets are scored only |
| DAT-06 | data-fact | PASS | `DAT-PAYOUT-SET` | DATA.md:25 | DAT-PAYOUT-SET: PASS x8 |
| DAT-07 | causality | WARN | `CAU-CUTOFF-OTHERS` | DATA.md:33 | CAU-CUTOFF-OTHERS: WARN (refused: entry exited 1 on the variant (slot_forecasting.schema.ValidationError: Future data leak: tick 202...) |
| DAT-08 | data-fact | NOT RUN | `GEN-SYNCHRONOUS` | DATA.md:33 | GEN-SYNCHRONOUS: generator-level checks (GEN-*) are not emitted by the suite; presets are scored only |
| DAT-09 | data-fact | PASS | `DAT-SPLIT-DISJOINT` | DATA.md:37 | DAT-SPLIT-DISJOINT: PASS x8 |
| DAT-10 | contract | PASS | `CON-ROUND-METADATA` | DATA.md:39 | CON-ROUND-METADATA: PASS x8 |
| DAT-11 | contract | MANUAL | - | DATA.md:39 | Path does not exist in this kit (data is under public/). See contradictions. |
| DAT-12 | scoring | PASS | `SCO-REFERENCE-REPRO` | DATA.md:14 | SCO-REFERENCE-REPRO: PASS |
| DAT-13 | data-fact | NOT RUN | `GEN-CONDITION-SEMANTICS` | DATA.md:31 | GEN-CONDITION-SEMANTICS: generator-level checks (GEN-*) are not emitted by the suite; presets are scored only |
| GDE-01 | contract | PASS | `CON-CSV-COLUMNS` | README.md:35 | CON-CSV-COLUMNS: PASS x14 |
| GDE-02 | causality | WARN | `CAU-FUTURE-TICKS`, `CAU-CUTOFF-OTHERS` | README.md:140 | CAU-FUTURE-TICKS: WARN (refused: entry exited 1 on the variant (slot_forecasting.schema.ValidationError: Future data leak: tick 202...); CAU-CUTOFF-OTHERS: WARN (refused: entry exited 1 on the variant (slot_forecasting.schema.ValidationError: Future data leak: tick 202...) |
| GDE-03 | causality | INFO | `CAU-MISSING-PROBE` | README.md:142 | CAU-MISSING-PROBE: INFO |
| GDE-04 | causality | PASS | `CAU-CROSS-EPISODE` | README.md:143 | CAU-CROSS-EPISODE: PASS |
| GDE-05 | packaging | PASS | `ENT-FRESH-FOLDER` | README.md:204 | ENT-FRESH-FOLDER: PASS |
| GDE-06 | scoring | NOT RUN | `SCO-SCORE-NOT-RANKING` | README.md:241 | SCO-SCORE-NOT-RANKING: diagnostic only: the suite never prints the 'score' field, so there is nothing to check |
| GDE-07 | scoring | PASS | `SCO-MATCHES-KIT` | README.md:241 | SCO-MATCHES-KIT: PASS |
| GDE-08 | scoring | NOT RUN | `SCO-HALF-SPLIT` | README.md:263 | SCO-HALF-SPLIT: not checked: scoring.repeated_half_splits does not expose its partition |
| GDE-09 | contract | PASS | `CON-QUERY-SET` | README.md:296 | CON-QUERY-SET: PASS x14 |
| GDE-10 | contract | MANUAL | - | README.md:310 | Form process; conflicts in spirit with RUL-30 (see contradictions). |
| GDE-11 | packaging | PASS | `PKG-NO-ANSWER-FILES`, `PKG-NO-DATA-COPY` | README.md:329 | PKG-NO-ANSWER-FILES: PASS; PKG-NO-DATA-COPY: PASS |
| GDE-12 | packaging | PASS | `CON-REPRODUCE-SCHEMA` | README.md:333 | CON-REPRODUCE-SCHEMA: PASS x2 |
| GDE-13 | contract | PASS | `CON-REPRODUCE-SCHEMA` | README.md:340 | CON-REPRODUCE-SCHEMA: PASS x2 |
| GDE-14 | robustness | PASS | `ROB-READONLY-MOUNTS` | README.md:340 | ROB-READONLY-MOUNTS: PASS |
| GDE-15 | robustness | PASS | `ROB-READONLY-MOUNTS`, `ROB-MOUNT-NO-ANSWERS` | README.md:342 | ROB-READONLY-MOUNTS: PASS; ROB-MOUNT-NO-ANSWERS: PASS x2 |
| GDE-16 | robustness | INFO | `ENT-MAIN-DATA-MOUNT` | README.md:343 | ENT-MAIN-DATA-MOUNT: INFO |
| GDE-17 | robustness | NOT RUN | `ROB-TMP-QUOTA` | README.md:344 | ROB-TMP-QUOTA: not produced by this run (level full, mode docker); see caveats |
| GDE-18 | robustness | NOT RUN | `ROB-OUTPUT-QUOTA` | README.md:345 | ROB-OUTPUT-QUOTA: not produced by this run (level full, mode docker); see caveats |
| GDE-19 | contract | PASS | `CON-OUTPUT-PATH`, `CON-EXIT-ZERO` | README.md:345 | CON-OUTPUT-PATH: PASS x14; CON-EXIT-ZERO: PASS x13 |
| GDE-20 | robustness | PASS | `ROB-NO-FORECAST-INPUT` | README.md:347 | ROB-NO-FORECAST-INPUT: PASS x2 |
| GDE-21 | contract | NOT RUN | `ENT-BASE-PY313` | README.md:349 | ENT-BASE-PY313: not probed separately: docker mode runs python:3.13-slim by construction (README.md:349) |
| GDE-22 | robustness | PASS | `ENT-DEPS-PREINSTALL` | README.md:349 | ENT-DEPS-PREINSTALL: PASS |
| GDE-24 | packaging | PASS | `PKG-ZIP-LIMITS`, `PKG-ZIP-SIZE` | README.md:365 | PKG-ZIP-LIMITS: PASS; PKG-ZIP-SIZE: PASS x2 |
| GDE-25 | packaging | PASS | `PKG-ZIP-SAFE` | README.md:365 | PKG-ZIP-SAFE: PASS x3 |
| GDE-26 | robustness | PASS | `ROB-DETERMINISM` | README.md:367 | ROB-DETERMINISM: PASS |
| GDE-28 | contract | PASS | `CON-CSV-STRICT`, `CON-CSV-COLUMNS` | README.md:388 | CON-CSV-STRICT: PASS x14; CON-CSV-COLUMNS: PASS x14 |
| GDE-29 | contract | PASS | `CON-CSV-UTF8` | README.md:390 | CON-CSV-UTF8: PASS x14 |
| GDE-33 | contract | PASS | `ROB-ROW-ORDER`, `CON-QUERY-SET` | README.md:387 | ROB-ROW-ORDER: PASS; CON-QUERY-SET: PASS x14 |
| GDE-34 | contract | MANUAL | - | README.md:403 | Form timestamp deadline; not a file property. |
| SUB-01 | packaging | PASS | `PKG-README-SEASON` | SUBMISSION_README.md:9 | PKG-README-SEASON: PASS |
| SUB-02 | robustness | PASS | `ROB-DETERMINISM` | SUBMISSION_README.md:41 | ROB-DETERMINISM: PASS |
| EXP-01 | scoring | NOT RUN | `SCO-FORECAST-ONLY` | EXPERIMENTS.md:23 | SCO-FORECAST-ONLY: by construction: points are a pure function of the CSV and targets (see ROB-DETERMINISM) |
| EXP-03 | contract | MANUAL | - | EXPERIMENTS.md:47 | The suite must not compute any test score; enforced by design, not by a file check. |
| MET-01 | data-fact | PASS | `DAT-CONDITION-SET` | public/metadata.json:2 | DAT-CONDITION-SET: PASS x8 |
| MET-02 | data-fact | PASS | `DAT-COUNTS-PER-CONDITION` | public/metadata.json:8 | DAT-COUNTS-PER-CONDITION: PASS x8 |
| MET-03 | data-fact | PASS | `DAT-MACHINES-PER-EPISODE` | public/metadata.json:14 | DAT-MACHINES-PER-EPISODE: PASS x8 |
| MET-04 | data-fact | NOT RUN | `GEN-LAG-RANGE`, `GEN-MAIN-LIKE-CALIBRATION` | public/metadata.json:13 | GEN-LAG-RANGE: generator-level checks (GEN-*) are not emitted by the suite; presets are scored only; GEN-MAIN-LIKE-CALIBRATION: generator-level checks (GEN-*) are not emitted by the suite; presets are scored only |
| MET-05 | robustness | MANUAL | - | public/metadata.json:23 | Entries must not rely on event seeds; a code-review item. |

## 2. Scores
Points = max_points / (1 + R), where R is the mean over conditions of E[c] / B[c]. Lower ratios are better. The CI is a condition-stratified episode bootstrap. Half-split rows show points on random halves of the episodes.

### Real validation (k-fold, each episode scored with answers hidden from its entry run)
| Dataset | Points | R | 95% bootstrap CI | Half-split A mean (sd, range) | Half-split B mean (sd, range) | ratio stable | ratio short_history | ratio drift | ratio shared_change |
|---|---|---|---|---|---|---|---|---|---|
| validation-kfold | 55.5743 / 60 | 0.0796 | [54.903, 56.169] | 55.52 (0.38, 54.63-56.04) | 55.62 (0.37, 55.11-56.48) | 0.0589 | 0.1204 | 0.0658 | 0.0734 |

### Generator presets (spec-derived approximations, scored against _truth)
| Dataset | Points | R | 95% bootstrap CI | Half-split A mean (sd, range) | Half-split B mean (sd, range) | ratio stable | ratio short_history | ratio drift | ratio shared_change |
|---|---|---|---|---|---|---|---|---|---|
| main_like | 40.0695 / 60 | 0.4974 | [38.592, 41.617] | 40.04 (0.73, 38.79-41.72) | 40.05 (0.73, 38.42-41.42) | 0.3449 | 0.6826 | 0.6341 | 0.3280 |
| round2_shift | 23.6573 / 40 | 0.6908 | [22.504, 24.657] | 23.78 (0.57, 22.83-24.76) | 23.50 (0.52, 22.69-24.32) | 0.4239 | 1.0269 | 0.9017 | 0.4107 |
| all_lags | 35.5546 / 60 | 0.6875 | [34.061, 37.089] | 35.41 (0.82, 33.84-37.49) | 35.66 (0.77, 33.80-37.20) | 0.5478 | 0.8645 | 0.8557 | 0.4822 |
| neighbour_heavy | 36.6553 / 60 | 0.6369 | [35.211, 38.131] | 36.53 (0.80, 35.12-37.90) | 36.74 (0.78, 35.50-38.12) | 0.5192 | 0.8218 | 0.7091 | 0.4974 |
| hidden_state_heavy | 39.3606 / 60 | 0.5244 | [37.904, 40.827] | 39.46 (0.79, 38.23-41.10) | 39.21 (0.81, 37.50-40.36) | 0.4260 | 0.5971 | 0.7060 | 0.3684 |
| fast_drift | 32.9360 / 60 | 0.8217 | [31.651, 34.087] | 32.93 (0.69, 31.95-34.20) | 32.90 (0.69, 31.53-33.83) | 0.4557 | 0.9217 | 1.3705 | 0.5389 |
| edge_cases | 38.7333 / 60 | 0.5491 | [37.084, 40.539] | 38.52 (0.95, 36.91-40.17) | 38.84 (0.93, 37.05-40.53) | 0.3538 | 0.8443 | 0.6140 | 0.3841 |

## 3. Slices (real validation, k-fold stitched)

**By condition**

| Value | Queries | MSE | Reference MSE | Ratio | Win-prob error |
|---|---|---|---|---|---|
| drift | 192 | 0.01249 | 0.18983 | 0.0658 | 0.05890 |
| shared_change | 192 | 0.01509 | 0.20556 | 0.0734 | 0.05904 |
| short_history | 192 | 0.02136 | 0.17734 | 0.1204 | 0.05930 |
| stable | 192 | 0.01156 | 0.19625 | 0.0589 | 0.04353 |

**By machine_type**

| Value | Queries | MSE | Reference MSE | Ratio | Win-prob error |
|---|---|---|---|---|---|
| balanced | 234 | 0.01393 | 0.22101 | 0.0630 | 0.05230 |
| jackpot | 265 | 0.01406 | 0.11359 | 0.1238 | 0.05872 |
| steady | 269 | 0.01721 | 0.24471 | 0.0703 | 0.05422 |

**By signal**

| Value | Queries | MSE | Reference MSE | Ratio | Win-prob error |
|---|---|---|---|---|---|
| -1 | 247 | 0.01299 | 0.18297 | 0.0710 | 0.04525 |
| 0 | 259 | 0.01638 | 0.19165 | 0.0855 | 0.06071 |
| 1 | 262 | 0.01589 | 0.20159 | 0.0788 | 0.05910 |

**By history_kind**

| Value | Queries | MSE | Reference MSE | Ratio | Win-prob error |
|---|---|---|---|---|---|
| long | 576 | 0.01305 | 0.19722 | 0.0662 | 0.05382 |
| short | 192 | 0.02136 | 0.17734 | 0.1204 | 0.05930 |

**By cutoff_bin**

| Value | Queries | MSE | Reference MSE | Ratio | Win-prob error |
|---|---|---|---|---|---|
| 192-207 | 184 | 0.01956 | 0.20346 | 0.0961 | 0.06292 |
| 208-223 | 200 | 0.01388 | 0.17435 | 0.0796 | 0.04945 |
| 224-239 | 240 | 0.01173 | 0.20227 | 0.0580 | 0.05116 |
| 240-256 | 144 | 0.01684 | 0.18608 | 0.0905 | 0.06000 |

## 4. Paired comparison
No --compare entry was given.

## 5. Runtime and memory
| Run | Entry | Mode | Exit | Wall s | Limit s | Peak MB | Output |
|---|---|---|---|---|---|---|---|
| kit-test | A | docker | 0 | 56.75 | 1800 | 164.0 | csv |
| A:fold0 | A | docker | 0 | 20.14 | 1800 | 98.7 | csv |
| A:fold1 | A | docker | 0 | 20.52 | 1800 | 96.3 | csv |
| A:fold2 | A | docker | 0 | 19.08 | 1800 | 96.9 | csv |
| A:fold3 | A | docker | 0 | 18.79 | 1800 | 95.7 | csv |
| A:fold0-rerun | A | docker | 0 | 18.98 | 1800 | 97.3 | csv |
| A:CAU-FUTURE-TICKS | A | docker | 1 | 13.72 | 1800 | 99.6 | none |
| A:CAU-SAME-TICK-OTHERS | A | docker | 1 | 13.98 | 1800 | 103.1 | none |
| A:CAU-CONDITIONS-HIDDEN | A | docker | 0 | 18.93 | 1800 | 103.5 | csv |
| A:CAU-ANSWERS-HIDDEN | A | docker | 0 | 18.51 | 1800 | 97.0 | csv |
| A:ROB-ROW-ORDER | A | docker | 0 | 18.67 | 1800 | 107.8 | csv |
| A:ROB-ID-RENAME | A | docker | 0 | 17.74 | 1800 | 98.7 | csv |
| A:INFO-MISSING-VS-ZERO-missing | A | docker | 0 | 17.69 | 1800 | 98.9 | csv |
| A:INFO-MISSING-VS-ZERO-zero | A | docker | 0 | 17.84 | 1800 | 97.7 | csv |
| A:CAU-CROSS-EPISODE | A | docker | 0 | 18.72 | 1800 | 103.9 | csv |
| A:fresh-folder | B | docker | 0 | 52.41 | 1800 | 164.8 | csv |
| A:preset:main_like | A | docker | 0 | 82.39 | 1800 | 106.6 | csv |
| A:preset:round2_shift | A | docker | 0 | 40.44 | 1800 | 87.8 | csv |
| A:preset:all_lags | A | docker | 0 | 84.81 | 1800 | 107.1 | csv |
| A:preset:neighbour_heavy | A | docker | 0 | 85.43 | 1800 | 110.1 | csv |
| A:preset:hidden_state_heavy | A | docker | 0 | 88.04 | 1800 | 92.5 | csv |
| A:preset:fast_drift | A | docker | 0 | 87.09 | 1800 | 109.9 | csv |
| A:preset:edge_cases | A | docker | 0 | 77.04 | 1800 | 96.0 | csv |

Slowest run: A:preset:hidden_state_heavy at 88.04s (4.89% of the 1800s limit).

## 6. Caveats
- Docker mode is the organizer emulation. Peak memory is sampled with `docker stats` every 0.5 s, so it is an estimate, and Docker VM memory may differ from the host. The /tmp and /output quotas are not enforced by entry.py.
- Reproducibility: the suite runs each entry with `python -m eval_suite run`; every number is recomputed from the CSVs it writes, and predictions are kept under predictions/.
- Real-validation score is a k-fold stitch: each fold's entry sees the other folds' validation answers as training data and never its own scored episodes. This is comparable across entries but is not identical to training on all of validation.
- The kit test run mounts train and test only, with no validation answers. It checks format and runtime; it never scores anything (EXP-03: test answers are private).
- Fold datasets are partial by design (validation holds only visible episodes, test holds the scored ones), so check_dataset is run on the kit and on generator presets, not on fold datasets.
- The scored answers live in <fold>-truth/_truth, outside the mounted dataset, so ROB-MOUNT-NO-ANSWERS can hold for the mounts the entry sees.
- CAU-ANSWERS-REMOVED and CAU-CONDITIONS-REMOVED change the scored split, which has no answer or condition files in the kit, so those variants are equal to the base by construction.
- CAU-CUTOFF-OTHERS (perturb CAU-SAME-TICK-OTHERS) also adds a tick-at-cutoff row for the query's own machine. That is stricter than the rule's wording (other machines only).
- CAU-CROSS-EPISODE randomises payouts of the other scored episodes' history in the scored split only. Train and validation labels are not touched, since changing them would legitimately change every prediction.
- CAU-MISSING-PROBE is information only. A WARN means the entry treats a missing tick as a zero payout.
- A variant the entry refuses (non-zero exit) is recorded as WARN ('refused'), not PASS and not FAIL.
- Requirements marked MANUAL (deadlines, form process, provenance) have no file check; see REQUIREMENTS.md.
- Generator presets are spec-derived approximations of the organizer's data, not the organizer's data.
- Docker mode maps the image 'python' to python:3.13-slim (README.md:349). The Docker Desktop VM on this machine may have less than 16 GiB, so the memory limit is applied but not fully reproducible.
- No --compare entry was given, so no paired comparison was made.
