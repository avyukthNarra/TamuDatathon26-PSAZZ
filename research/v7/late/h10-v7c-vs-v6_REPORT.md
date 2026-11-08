# eval_suite report: entry

**VERDICT: PASS WITH WARNINGS - 0 requirement(s) FAIL, 3 WARN, 54 PASS, 3 INFO, 19 NOT RUN, 9 MANUAL. Level full, docker run (VM memory and quotas as reported).**

Entry `/Users/avynarra/Documents/Datathon/research_pf/c3-codex/v7c/entry` compared with `/Users/avynarra/Documents/Datathon/worktrees/pranav-5629/mainFiles/runs/combined-v6/entry`. Seed 0, k=4, bootstrap repeats 1000. Wall time 954.1s. Generated 2026-11-08T07:26:31+00:00.

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
| RUL-24 | contract | PASS | `CON-CSV-COLUMNS`, `CON-CSV-STRICT` | RULES.md:43 | CON-CSV-COLUMNS: PASS x26; CON-CSV-STRICT: PASS x26 |
| RUL-25 | contract | PASS | `CON-QUERY-SET` | RULES.md:49 | CON-QUERY-SET: PASS x26 |
| RUL-26 | contract | PASS | `CON-PROB-VALUES` | RULES.md:49 | CON-PROB-VALUES: PASS x52 |
| RUL-27 | contract | PASS | `CON-PROB-SUM` | RULES.md:49 | CON-PROB-SUM: PASS x26 |
| RUL-28 | packaging | PASS | `CON-CSV-SIZE`, `PKG-ZIP-SIZE` | RULES.md:51 | CON-CSV-SIZE: PASS x26; PKG-ZIP-SIZE: PASS x2 |
| RUL-29 | packaging | PASS | `PKG-ZIP-ROOT` | RULES.md:51 | PKG-ZIP-ROOT: PASS x3 |
| RUL-30 | scoring | NOT RUN | `SCO-MISSING-ZERO` | RULES.md:53 | SCO-MISSING-ZERO: rule-level: a missing round entry scores zero in the organizer's ranking, not in this suite |
| RUL-31 | scoring | NOT RUN | `SCO-MAIN-REQUIRED` | RULES.md:53 | SCO-MAIN-REQUIRED: ranking-level: overall ranking flag is organizer code, not a file check |
| RUL-32 | contract | MANUAL | - | RULES.md:53 | Form-process rule. |
| RUL-33 | contract | MANUAL | - | RULES.md:55 | CONFLICTS with RUL-30 / RULES.md:53 (invalid round-two entry earns zero). See REQUIREMENTS.md contradictions. |
| RUL-34 | contract | PASS | `CON-IMAGE-PYTHON` | RULES.md:55 | CON-IMAGE-PYTHON: PASS x2 |
| RUL-35 | packaging | PASS | `PKG-REQUIREMENTS-PINNED` | RULES.md:55 | PKG-REQUIREMENTS-PINNED: PASS x2 |
| RUL-36 | robustness | PASS | `ROB-RUNTIME-LIMIT`, `ROB-CPU-LIMIT`, `ROB-MEM-LIMIT` | RULES.md:55 | ROB-RUNTIME-LIMIT: PASS x24; ROB-CPU-LIMIT: PASS; ROB-MEM-LIMIT: PASS |
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
| GDE-01 | contract | PASS | `CON-CSV-COLUMNS` | README.md:35 | CON-CSV-COLUMNS: PASS x26 |
| GDE-02 | causality | WARN | `CAU-FUTURE-TICKS`, `CAU-CUTOFF-OTHERS` | README.md:140 | CAU-FUTURE-TICKS: WARN (refused: entry exited 1 on the variant (slot_forecasting.schema.ValidationError: Future data leak: tick 202...); CAU-CUTOFF-OTHERS: WARN (refused: entry exited 1 on the variant (slot_forecasting.schema.ValidationError: Future data leak: tick 202...) |
| GDE-03 | causality | INFO | `CAU-MISSING-PROBE` | README.md:142 | CAU-MISSING-PROBE: INFO |
| GDE-04 | causality | PASS | `CAU-CROSS-EPISODE` | README.md:143 | CAU-CROSS-EPISODE: PASS |
| GDE-05 | packaging | PASS | `ENT-FRESH-FOLDER` | README.md:204 | ENT-FRESH-FOLDER: PASS |
| GDE-06 | scoring | NOT RUN | `SCO-SCORE-NOT-RANKING` | README.md:241 | SCO-SCORE-NOT-RANKING: diagnostic only: the suite never prints the 'score' field, so there is nothing to check |
| GDE-07 | scoring | PASS | `SCO-MATCHES-KIT` | README.md:241 | SCO-MATCHES-KIT: PASS |
| GDE-08 | scoring | NOT RUN | `SCO-HALF-SPLIT` | README.md:263 | SCO-HALF-SPLIT: not checked: scoring.repeated_half_splits does not expose its partition |
| GDE-09 | contract | PASS | `CON-QUERY-SET` | README.md:296 | CON-QUERY-SET: PASS x26 |
| GDE-10 | contract | MANUAL | - | README.md:310 | Form process; conflicts in spirit with RUL-30 (see contradictions). |
| GDE-11 | packaging | PASS | `PKG-NO-ANSWER-FILES`, `PKG-NO-DATA-COPY` | README.md:329 | PKG-NO-ANSWER-FILES: PASS; PKG-NO-DATA-COPY: PASS |
| GDE-12 | packaging | PASS | `CON-REPRODUCE-SCHEMA` | README.md:333 | CON-REPRODUCE-SCHEMA: PASS x2 |
| GDE-13 | contract | PASS | `CON-REPRODUCE-SCHEMA` | README.md:340 | CON-REPRODUCE-SCHEMA: PASS x2 |
| GDE-14 | robustness | PASS | `ROB-READONLY-MOUNTS` | README.md:340 | ROB-READONLY-MOUNTS: PASS |
| GDE-15 | robustness | PASS | `ROB-READONLY-MOUNTS`, `ROB-MOUNT-NO-ANSWERS` | README.md:342 | ROB-READONLY-MOUNTS: PASS; ROB-MOUNT-NO-ANSWERS: PASS x2 |
| GDE-16 | robustness | INFO | `ENT-MAIN-DATA-MOUNT` | README.md:343 | ENT-MAIN-DATA-MOUNT: INFO |
| GDE-17 | robustness | NOT RUN | `ROB-TMP-QUOTA` | README.md:344 | ROB-TMP-QUOTA: not produced by this run (level full, mode docker); see caveats |
| GDE-18 | robustness | NOT RUN | `ROB-OUTPUT-QUOTA` | README.md:345 | ROB-OUTPUT-QUOTA: not produced by this run (level full, mode docker); see caveats |
| GDE-19 | contract | PASS | `CON-OUTPUT-PATH`, `CON-EXIT-ZERO` | README.md:345 | CON-OUTPUT-PATH: PASS x25; CON-EXIT-ZERO: PASS x24 |
| GDE-20 | robustness | PASS | `ROB-NO-FORECAST-INPUT` | README.md:347 | ROB-NO-FORECAST-INPUT: PASS x2 |
| GDE-21 | contract | NOT RUN | `ENT-BASE-PY313` | README.md:349 | ENT-BASE-PY313: not probed separately: docker mode runs python:3.13-slim by construction (README.md:349) |
| GDE-22 | robustness | PASS | `ENT-DEPS-PREINSTALL` | README.md:349 | ENT-DEPS-PREINSTALL: PASS |
| GDE-24 | packaging | PASS | `PKG-ZIP-LIMITS`, `PKG-ZIP-SIZE` | README.md:365 | PKG-ZIP-LIMITS: PASS; PKG-ZIP-SIZE: PASS x2 |
| GDE-25 | packaging | PASS | `PKG-ZIP-SAFE` | README.md:365 | PKG-ZIP-SAFE: PASS x3 |
| GDE-26 | robustness | PASS | `ROB-DETERMINISM` | README.md:367 | ROB-DETERMINISM: PASS |
| GDE-28 | contract | PASS | `CON-CSV-STRICT`, `CON-CSV-COLUMNS` | README.md:388 | CON-CSV-STRICT: PASS x26; CON-CSV-COLUMNS: PASS x26 |
| GDE-29 | contract | PASS | `CON-CSV-UTF8` | README.md:390 | CON-CSV-UTF8: PASS x26 |
| GDE-33 | contract | PASS | `ROB-ROW-ORDER`, `CON-QUERY-SET` | README.md:387 | ROB-ROW-ORDER: PASS; CON-QUERY-SET: PASS x26 |
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
| validation-kfold | 57.3565 / 60 | 0.0461 | [56.783, 57.877] | 57.33 (0.29, 56.66-57.81) | 57.38 (0.29, 56.92-58.01) | 0.0419 | 0.0753 | 0.0327 | 0.0345 |

### Generator presets (spec-derived approximations, scored against _truth)
| Dataset | Points | R | 95% bootstrap CI | Half-split A mean (sd, range) | Half-split B mean (sd, range) | ratio stable | ratio short_history | ratio drift | ratio shared_change |
|---|---|---|---|---|---|---|---|---|---|
| main_like | 40.2741 / 60 | 0.4898 | [38.482, 41.843] | 40.26 (0.76, 38.76-41.71) | 40.25 (0.77, 38.87-41.90) | 0.3322 | 0.6761 | 0.6501 | 0.3008 |
| round2_shift | 21.2809 / 40 | 0.8796 | [20.339, 22.139] | 21.41 (0.53, 20.40-22.18) | 21.13 (0.52, 20.39-22.27) | 0.7195 | 1.2349 | 0.8615 | 0.7026 |
| all_lags | 31.8555 / 60 | 0.8835 | [30.700, 33.115] | 31.93 (0.57, 31.16-33.46) | 31.75 (0.52, 30.51-32.46) | 0.8133 | 1.0462 | 0.9110 | 0.7636 |
| neighbour_heavy | 36.2560 / 60 | 0.6549 | [34.665, 37.877] | 36.12 (0.84, 34.66-37.46) | 36.34 (0.84, 35.15-37.82) | 0.5181 | 0.8458 | 0.7657 | 0.4900 |
| hidden_state_heavy | 32.3923 / 60 | 0.8523 | [31.162, 33.591] | 32.56 (0.72, 30.87-34.07) | 32.23 (0.71, 30.72-33.75) | 0.8201 | 1.0025 | 0.8083 | 0.7783 |
| fast_drift | 29.9364 / 60 | 1.0042 | [28.593, 31.266] | 29.74 (0.82, 28.14-31.03) | 30.09 (0.85, 28.55-31.92) | 1.0425 | 1.1829 | 0.8277 | 0.9639 |
| edge_cases | 32.8273 / 60 | 0.8277 | [31.333, 34.392] | 32.55 (1.03, 30.49-34.13) | 33.06 (1.13, 31.53-35.53) | 0.6757 | 1.0938 | 0.7160 | 0.8255 |

## 3. Slices (real validation, k-fold stitched)

**By condition**

| Value | Queries | MSE | Reference MSE | Ratio | Win-prob error |
|---|---|---|---|---|---|
| drift | 192 | 0.00621 | 0.18983 | 0.0327 | 0.04406 |
| shared_change | 192 | 0.00709 | 0.20556 | 0.0345 | 0.03791 |
| short_history | 192 | 0.01335 | 0.17734 | 0.0753 | 0.05264 |
| stable | 192 | 0.00821 | 0.19625 | 0.0419 | 0.04418 |

**By machine_type**

| Value | Queries | MSE | Reference MSE | Ratio | Win-prob error |
|---|---|---|---|---|---|
| balanced | 234 | 0.00753 | 0.22101 | 0.0341 | 0.04360 |
| jackpot | 265 | 0.01078 | 0.11359 | 0.0949 | 0.05102 |
| steady | 269 | 0.00772 | 0.24471 | 0.0315 | 0.03942 |

**By signal**

| Value | Queries | MSE | Reference MSE | Ratio | Win-prob error |
|---|---|---|---|---|---|
| -1 | 247 | 0.00709 | 0.18297 | 0.0387 | 0.03675 |
| 0 | 259 | 0.01031 | 0.19165 | 0.0538 | 0.04880 |
| 1 | 262 | 0.00868 | 0.20159 | 0.0431 | 0.04813 |

**By history_kind**

| Value | Queries | MSE | Reference MSE | Ratio | Win-prob error |
|---|---|---|---|---|---|
| long | 576 | 0.00717 | 0.19722 | 0.0364 | 0.04205 |
| short | 192 | 0.01335 | 0.17734 | 0.0753 | 0.05264 |

**By cutoff_bin**

| Value | Queries | MSE | Reference MSE | Ratio | Win-prob error |
|---|---|---|---|---|---|
| 192-207 | 184 | 0.01036 | 0.20346 | 0.0509 | 0.04903 |
| 208-223 | 200 | 0.00862 | 0.17435 | 0.0494 | 0.04352 |
| 224-239 | 240 | 0.00703 | 0.20227 | 0.0347 | 0.04031 |
| 240-256 | 144 | 0.00957 | 0.18608 | 0.0514 | 0.04810 |

## 4. Paired comparison
| Dataset | Points A | Points B | A - B | 95% CI of difference | Share of resamples A better |
|---|---|---|---|---|---|
| validation-kfold | 57.3565 | 56.2659 | +1.0905 | [+0.799, +1.430] | 1.000 |
| preset:main_like | 40.2741 | 40.2661 | +0.0080 | [-0.036, +0.053] | 0.671 |
| preset:round2_shift | 21.2809 | 23.9018 | -2.6209 | [-3.645, -1.535] | 0.000 |
| preset:all_lags | 31.8555 | 35.5139 | -3.6584 | [-5.173, -2.087] | 0.000 |
| preset:neighbour_heavy | 36.2560 | 36.2560 | +0.0000 | [+0.000, +0.000] | 0.000 |
| preset:hidden_state_heavy | 32.3923 | 38.7640 | -6.3717 | [-7.851, -4.859] | 0.000 |
| preset:fast_drift | 29.9364 | 32.8342 | -2.8978 | [-4.614, -1.285] | 0.000 |
| preset:edge_cases | 32.8273 | 38.3499 | -5.5226 | [-7.184, -3.934] | 0.000 |

## 5. Runtime and memory
| Run | Entry | Mode | Exit | Wall s | Limit s | Peak MB | Output |
|---|---|---|---|---|---|---|---|
| kit-test | A | docker | 0 | 51.10 | 1800 | 198.6 | csv |
| A:fold0 | A | docker | 0 | 38.72 | 1800 | 96.9 | csv |
| A:fold1 | A | docker | 0 | 38.86 | 1800 | 98.8 | csv |
| A:fold2 | A | docker | 0 | 39.32 | 1800 | 96.3 | csv |
| A:fold3 | A | docker | 0 | 39.24 | 1800 | 96.8 | csv |
| B:fold0 | B | docker | 0 | 24.93 | 1800 | 103.1 | csv |
| B:fold1 | B | docker | 0 | 25.00 | 1800 | 96.5 | csv |
| B:fold2 | B | docker | 0 | 25.08 | 1800 | 99.5 | csv |
| B:fold3 | B | docker | 0 | 24.64 | 1800 | 95.9 | csv |
| A:fold0-rerun | A | docker | 0 | 39.38 | 1800 | 97.4 | csv |
| A:CAU-FUTURE-TICKS | A | docker | 1 | 45.14 | 1800 | 96.9 | none |
| A:CAU-SAME-TICK-OTHERS | A | docker | 1 | 45.44 | 1800 | 96.9 | none |
| A:CAU-CONDITIONS-HIDDEN | A | docker | 0 | 47.86 | 1800 | 97.0 | csv |
| A:CAU-ANSWERS-HIDDEN | A | docker | 0 | 49.07 | 1800 | 97.3 | csv |
| A:ROB-ROW-ORDER | A | docker | 0 | 43.75 | 1800 | 96.8 | csv |
| A:ROB-ID-RENAME | A | docker | 0 | 44.21 | 1800 | 97.3 | csv |
| A:INFO-MISSING-VS-ZERO-missing | A | docker | 0 | 44.06 | 1800 | 96.9 | csv |
| A:INFO-MISSING-VS-ZERO-zero | A | docker | 0 | 44.06 | 1800 | 97.0 | csv |
| A:CAU-CROSS-EPISODE | A | docker | 0 | 43.96 | 1800 | 98.6 | csv |
| A:fresh-folder | B | docker | 0 | 58.56 | 1800 | 198.9 | csv |
| A:preset:main_like | A | docker | 0 | 189.66 | 1800 | 122.4 | csv |
| B:preset:main_like | B | docker | 0 | 115.61 | 1800 | 120.8 | csv |
| A:preset:round2_shift | A | docker | 0 | 107.91 | 1800 | 122.6 | csv |
| B:preset:round2_shift | B | docker | 0 | 35.51 | 1800 | 120.2 | csv |
| A:preset:all_lags | A | docker | 0 | 199.82 | 1800 | 122.4 | csv |
| B:preset:all_lags | B | docker | 0 | 123.68 | 1800 | 121.2 | csv |
| A:preset:neighbour_heavy | A | docker | 0 | 203.82 | 1800 | 121.4 | csv |
| B:preset:neighbour_heavy | B | docker | 0 | 131.84 | 1800 | 120.3 | csv |
| A:preset:hidden_state_heavy | A | docker | 0 | 65.57 | 1800 | 121.6 | csv |
| B:preset:hidden_state_heavy | B | docker | 0 | 128.37 | 1800 | 120.6 | csv |
| A:preset:fast_drift | A | docker | 0 | 222.61 | 1800 | 121.9 | csv |
| B:preset:fast_drift | B | docker | 0 | 123.49 | 1800 | 120.9 | csv |
| A:preset:edge_cases | A | docker | 0 | 198.23 | 1800 | 109.7 | csv |
| B:preset:edge_cases | B | docker | 0 | 104.75 | 1800 | 108.1 | csv |

Slowest run: A:preset:fast_drift at 222.61s (12.37% of the 1800s limit).

## 6. Caveats
- Docker mode is the organizer emulation. Peak memory is sampled with `docker stats` every 0.5 s, so it is an estimate, and Docker VM memory may differ from the host. The 128 MiB /tmp and 32 MiB /output tmpfs quotas are applied and recorded by docker inspect.
- Reproducibility: the suite runs each entry with `python -m eval_suite run`; every number is recomputed from the CSVs it writes, and predictions are kept under predictions/.
- Real-validation score is a k-fold stitch: each fold's entry sees the other folds' validation answers as training data and does not mount answer files for its own scored episodes. This is comparable across entries but is not identical to training on all of validation.
- Scored-fold answer files are withheld from each mounted dataset, but packaged weights or priors may already encode validation information; training provenance and prior exposure are not verified.
- The kit test run mounts train and test only, with no validation answers. It checks format and runtime; it never scores anything (EXP-03: test answers are private).
- Fold datasets are partial by design (validation holds only visible episodes, test holds the scored ones), so check_dataset is run on the kit and on generator presets, not on fold datasets.
- The scored answers live in <fold>-truth/_truth, outside the mounted dataset, so ROB-MOUNT-NO-ANSWERS can hold for the mounts the entry sees.
- In docker mode, CAU-ANSWERS-REMOVED and CAU-CONDITIONS-REMOVED change the scored split, which has no answer or condition files in the kit, so those variants are equal to the base by construction.
- CAU-CUTOFF-OTHERS (perturb CAU-SAME-TICK-OTHERS) also adds a tick-at-cutoff row for the query's own machine. That is stricter than the rule's wording (other machines only).
- CAU-CROSS-EPISODE randomises payouts of the other scored episodes' history in the scored split only. Train and validation labels are not touched, since changing them would legitimately change every prediction.
- CAU-MISSING-PROBE is information only. Identical forecasts after deleting versus zeroing sampled rows are inconclusive: sampled outcomes may already be zero, or the entry may ignore those rows.
- A variant the entry refuses (non-zero exit) is recorded as WARN ('refused'), not PASS and not FAIL.
- Requirements marked MANUAL (deadlines, form process, provenance) have no file check; see REQUIREMENTS.md.
- Generator presets are spec-derived approximations of the organizer's data, not the organizer's data.
- Docker mode maps the image 'python' to python:3.13-slim (README.md:349). The Docker Desktop VM on this machine may have less than 16 GiB, so the memory limit is applied but not fully reproducible.
