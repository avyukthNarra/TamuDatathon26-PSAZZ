# eval_suite report: reference

**VERDICT: PASS WITH WARNINGS - 0 requirement(s) FAIL, 3 WARN, 56 PASS, 1 INFO, 19 NOT RUN, 9 MANUAL. Level full, docker run (VM memory and quotas as reported).**

Entry `eval_suite/fixtures/reference`. Seed 0, k=4, bootstrap repeats 1000. Wall time 117.2s. Generated 2026-11-08T00:51:28+00:00.

## 1. Requirement coverage
56 PASS, 0 FAIL, 3 WARN, 1 INFO, 19 NOT RUN, 9 MANUAL of 88 requirements.

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
| RUL-18 | causality | PASS | `CAU-FUTURE-TICKS` | RULES.md:35 | CAU-FUTURE-TICKS: PASS |
| RUL-19 | causality | WARN | `CAU-MISSING-PROBE` | RULES.md:35 | CAU-MISSING-PROBE: WARN (entry treats a missing tick the same as a zero payout; docs say missing is unknown, not a loss) |
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
| DAT-07 | causality | PASS | `CAU-CUTOFF-OTHERS` | DATA.md:33 | CAU-CUTOFF-OTHERS: PASS |
| DAT-08 | data-fact | NOT RUN | `GEN-SYNCHRONOUS` | DATA.md:33 | GEN-SYNCHRONOUS: generator-level checks (GEN-*) are not emitted by the suite; presets are scored only |
| DAT-09 | data-fact | PASS | `DAT-SPLIT-DISJOINT` | DATA.md:37 | DAT-SPLIT-DISJOINT: PASS x8 |
| DAT-10 | contract | PASS | `CON-ROUND-METADATA` | DATA.md:39 | CON-ROUND-METADATA: PASS x8 |
| DAT-11 | contract | MANUAL | - | DATA.md:39 | Path does not exist in this kit (data is under public/). See contradictions. |
| DAT-12 | scoring | PASS | `SCO-REFERENCE-REPRO` | DATA.md:14 | SCO-REFERENCE-REPRO: PASS |
| DAT-13 | data-fact | NOT RUN | `GEN-CONDITION-SEMANTICS` | DATA.md:31 | GEN-CONDITION-SEMANTICS: generator-level checks (GEN-*) are not emitted by the suite; presets are scored only |
| GDE-01 | contract | PASS | `CON-CSV-COLUMNS` | README.md:35 | CON-CSV-COLUMNS: PASS x14 |
| GDE-02 | causality | PASS | `CAU-FUTURE-TICKS`, `CAU-CUTOFF-OTHERS` | README.md:140 | CAU-FUTURE-TICKS: PASS; CAU-CUTOFF-OTHERS: PASS |
| GDE-03 | causality | WARN | `CAU-MISSING-PROBE` | README.md:142 | CAU-MISSING-PROBE: WARN (entry treats a missing tick the same as a zero payout; docs say missing is unknown, not a loss) |
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
| SUB-01 | packaging | WARN | `PKG-README-SEASON` | SUBMISSION_README.md:9 | PKG-README-SEASON: WARN (season_id 5fffdc4d0c700bbf61c3f911 not in README.md) |
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
| validation-kfold | 29.5328 / 60 | 1.0316 | [29.055, 30.024] | 29.52 (0.27, 29.05-29.98) | 29.53 (0.27, 29.08-29.99) | 1.0218 | 1.1121 | 1.0263 | 0.9664 |

### Generator presets (spec-derived approximations, scored against _truth)
| Dataset | Points | R | 95% bootstrap CI | Half-split A mean (sd, range) | Half-split B mean (sd, range) | ratio stable | ratio short_history | ratio drift | ratio shared_change |
|---|---|---|---|---|---|---|---|---|---|
| main_like | 27.9336 / 60 | 1.1480 | [27.146, 28.653] | 27.86 (0.40, 27.06-28.48) | 27.95 (0.41, 27.20-28.64) | 1.2082 | 1.2517 | 0.8422 | 1.2898 |
| round2_shift | 18.3773 / 40 | 1.1766 | [17.907, 18.803] | 18.28 (0.29, 17.68-18.87) | 18.45 (0.29, 17.86-19.02) | 1.2497 | 1.2696 | 0.8794 | 1.3077 |
| all_lags | 28.0945 / 60 | 1.1356 | [27.354, 28.892] | 28.15 (0.38, 27.51-29.06) | 27.99 (0.36, 27.16-28.64) | 1.4063 | 1.2019 | 0.8302 | 1.1041 |
| neighbour_heavy | 27.6858 / 60 | 1.1672 | [26.923, 28.459] | 27.66 (0.43, 26.95-28.44) | 27.67 (0.42, 26.90-28.32) | 1.2936 | 1.2174 | 0.8515 | 1.3062 |
| hidden_state_heavy | 28.3087 / 60 | 1.1195 | [27.619, 29.005] | 28.33 (0.35, 27.78-29.25) | 28.25 (0.37, 27.29-28.82) | 1.1238 | 1.2442 | 0.9754 | 1.1346 |
| fast_drift | 28.4739 / 60 | 1.1072 | [27.701, 29.134] | 28.35 (0.46, 27.17-29.14) | 28.58 (0.46, 27.81-29.78) | 1.2488 | 1.2000 | 0.7540 | 1.2260 |
| edge_cases | 27.6083 / 60 | 1.1733 | [26.653, 28.439] | 27.62 (0.65, 26.43-29.19) | 27.54 (0.66, 26.06-28.89) | 1.3082 | 1.2917 | 0.8721 | 1.2211 |

## 3. Slices (real validation, k-fold stitched)

**By condition**

| Value | Queries | MSE | Reference MSE | Ratio | Win-prob error |
|---|---|---|---|---|---|
| drift | 192 | 0.19482 | 0.18983 | 1.0263 | 0.35060 |
| shared_change | 192 | 0.19866 | 0.20556 | 0.9664 | 0.34999 |
| short_history | 192 | 0.19722 | 0.17734 | 1.1121 | 0.34888 |
| stable | 192 | 0.20052 | 0.19625 | 1.0218 | 0.34842 |

**By machine_type**

| Value | Queries | MSE | Reference MSE | Ratio | Win-prob error |
|---|---|---|---|---|---|
| balanced | 234 | 0.22989 | 0.22101 | 1.0402 | 0.39315 |
| jackpot | 265 | 0.11692 | 0.11359 | 1.0293 | 0.26990 |
| steady | 269 | 0.24957 | 0.24471 | 1.0199 | 0.38987 |

**By signal**

| Value | Queries | MSE | Reference MSE | Ratio | Win-prob error |
|---|---|---|---|---|---|
| -1 | 247 | 0.18762 | 0.18297 | 1.0254 | 0.34115 |
| 0 | 259 | 0.19451 | 0.19165 | 1.0149 | 0.34763 |
| 1 | 262 | 0.21067 | 0.20159 | 1.0451 | 0.35913 |

**By history_kind**

| Value | Queries | MSE | Reference MSE | Ratio | Win-prob error |
|---|---|---|---|---|---|
| long | 576 | 0.19800 | 0.19722 | 1.0040 | 0.34967 |
| short | 192 | 0.19722 | 0.17734 | 1.1121 | 0.34888 |

**By cutoff_bin**

| Value | Queries | MSE | Reference MSE | Ratio | Win-prob error |
|---|---|---|---|---|---|
| 192-207 | 184 | 0.20774 | 0.20346 | 1.0210 | 0.35828 |
| 208-223 | 200 | 0.17765 | 0.17435 | 1.0190 | 0.33181 |
| 224-239 | 240 | 0.20453 | 0.20227 | 1.0112 | 0.35555 |
| 240-256 | 144 | 0.20191 | 0.18608 | 1.0851 | 0.35263 |

## 4. Paired comparison
No --compare entry was given.

## 5. Runtime and memory
| Run | Entry | Mode | Exit | Wall s | Limit s | Peak MB | Output |
|---|---|---|---|---|---|---|---|
| kit-test | A | docker | 0 | 0.17 | 1800 | 1.0 | csv |
| A:fold0 | A | docker | 0 | 0.17 | 1800 | 1.0 | csv |
| A:fold1 | A | docker | 0 | 0.15 | 1800 | 0.9 | csv |
| A:fold2 | A | docker | 0 | 0.16 | 1800 | 0.9 | csv |
| A:fold3 | A | docker | 0 | 0.17 | 1800 | 1.2 | csv |
| A:fold0-rerun | A | docker | 0 | 0.14 | 1800 | 1.0 | csv |
| A:CAU-FUTURE-TICKS | A | docker | 0 | 0.16 | 1800 | 1.0 | csv |
| A:CAU-SAME-TICK-OTHERS | A | docker | 0 | 0.17 | 1800 | 1.0 | csv |
| A:CAU-CONDITIONS-HIDDEN | A | docker | 0 | 0.16 | 1800 | 1.1 | csv |
| A:CAU-ANSWERS-HIDDEN | A | docker | 0 | 0.15 | 1800 | 1.4 | csv |
| A:ROB-ROW-ORDER | A | docker | 0 | 0.15 | 1800 | 1.0 | csv |
| A:ROB-ID-RENAME | A | docker | 0 | 0.15 | 1800 | 0.9 | csv |
| A:INFO-MISSING-VS-ZERO-missing | A | docker | 0 | 0.13 | 1800 | 1.0 | csv |
| A:INFO-MISSING-VS-ZERO-zero | A | docker | 0 | 0.15 | 1800 | 1.1 | csv |
| A:CAU-CROSS-EPISODE | A | docker | 0 | 0.15 | 1800 | 0.8 | csv |
| A:fresh-folder | B | docker | 0 | 0.15 | 1800 | 1.2 | csv |
| A:preset:main_like | A | docker | 0 | 0.16 | 1800 | 1.4 | csv |
| A:preset:round2_shift | A | docker | 0 | 0.18 | 1800 | 1.0 | csv |
| A:preset:all_lags | A | docker | 0 | 0.21 | 1800 | 1.4 | csv |
| A:preset:neighbour_heavy | A | docker | 0 | 0.15 | 1800 | 1.0 | csv |
| A:preset:hidden_state_heavy | A | docker | 0 | 0.17 | 1800 | 1.3 | csv |
| A:preset:fast_drift | A | docker | 0 | 0.21 | 1800 | 1.6 | csv |
| A:preset:edge_cases | A | docker | 0 | 0.15 | 1800 | 1.2 | csv |

Slowest run: A:preset:fast_drift at 0.21s (0.01% of the 1800s limit).

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
