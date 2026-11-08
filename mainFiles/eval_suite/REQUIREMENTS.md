# eval_suite: requirements traceability

Generated from `eval_suite/requirements.py` (the source of truth; this table mirrors it). Built only from README.md (participant guide), RULES.md, DATA.md, SUBMISSION_README.md, EXPERIMENTS.md, public/metadata.json and the kit code (check_submission.py, build_entry.py, slot_forecasting/{submission_contract,scoring,contest,schema,io,rounds}.py). Source = `file:line` (line numbers refer to the files in this worktree). Quotes are verbatim or shortened substrings. Rows with `manual` have no automated check and a note.

Units: MB = 1,000,000 bytes; MiB = 2^20; GiB = 2^30. "Identical" predictions = max |delta p| <= 1e-8 after row normalisation (p_k / sum(p)).

## Counts

- Requirements: **88** rows; distinct check IDs: **77**.
- By category: contract 23, packaging 9, robustness 15, causality 7, scoring 20, data-fact 14.
- By testable_by: contract.py 27, entry.py 17, generator.py 7, manual 9, perturb.py 8, scoring.py 20.

## Requirements by category

### contract (23)

| ID | Source | Quote | Testable by | Check IDs | Note |
| --- | --- | --- | --- | --- | --- |
| RUL-03 | RULES.md:11 | Round two closes Sunday, November 8 at 11 AM Central, giving everyone 14 hours. | manual | - | Deadline; organizer process, not a file property. Same deadline in README.md:15 and :273. |
| RUL-17 | RULES.md:34 | Use organizer-released data. | manual | - | Provenance of data is not checkable from files; manual review of imports and data paths. |
| RUL-24 | RULES.md:43 | submit one CSV with exactly these columns | contract.py | `CON-CSV-COLUMNS`, `CON-CSV-STRICT` |  |
| RUL-25 | RULES.md:49 | Include every test query once. | contract.py | `CON-QUERY-SET` |  |
| RUL-26 | RULES.md:49 | Probabilities must be finite, between zero and one | contract.py | `CON-PROB-VALUES` |  |
| RUL-27 | RULES.md:49 | sum to one within `1e-6` | contract.py | `CON-PROB-SUM` |  |
| RUL-32 | RULES.md:53 | Main entries can't be replaced after round two opens. | manual | - | Form-process rule. |
| RUL-33 | RULES.md:55 | A replacement still counts if validation or its code check later fails | manual | - | CONFLICTS with RUL-30 / RULES.md:53 (invalid round-two entry earns zero). See REQUIREMENTS.md contradictions. |
| RUL-34 | RULES.md:55 | Keep `"image": "python"` | contract.py | `CON-IMAGE-PYTHON` |  |
| DAT-02 | DATA.md:17 | Test data contains queries and observed histories. | contract.py | `DAT-TEST-SPLIT-FILES` |  |
| DAT-10 | DATA.md:39 | The kit's data version is `experimentation-suite-v3`. | contract.py | `CON-ROUND-METADATA` |  |
| DAT-11 | DATA.md:39 | Main lives in `season/main/public` | manual | - | Path does not exist in this kit (data is under public/). See contradictions. |
| GDE-01 | README.md:35 | Return five probabilities in this order: `0, 1, 2, 5, 20`. | contract.py | `CON-CSV-COLUMNS` |  |
| GDE-09 | README.md:296 | Your main CSV won't work for round two because the query IDs are different. | contract.py | `CON-QUERY-SET` |  |
| GDE-10 | README.md:310 | Your team's latest Form response received before the deadline counts, even if its files are invalid or its code later fails. | manual | - | Form process; conflicts in spirit with RUL-30 (see contradictions). |
| GDE-13 | README.md:340 | The command is an argument list, not a shell string. | contract.py | `CON-REPRODUCE-SCHEMA` |  |
| GDE-19 | README.md:345 | Write the final CSV to `/output/submission.csv`. | entry.py | `CON-OUTPUT-PATH`, `CON-EXIT-ZERO` |  |
| GDE-21 | README.md:349 | The base environment is Python 3.13 with the standard library. | entry.py | `ENT-BASE-PY313` |  |
| GDE-28 | README.md:388 | Use exactly these six columns, without a dataframe index or extra columns. | contract.py | `CON-CSV-STRICT`, `CON-CSV-COLUMNS` |  |
| GDE-29 | README.md:390 | Save as UTF-8 CSV. | contract.py | `CON-CSV-UTF8` |  |
| GDE-33 | README.md:387 | Include every test query exactly once, with no extras. Row order doesn't matter. | perturb.py | `ROB-ROW-ORDER`, `CON-QUERY-SET` |  |
| GDE-34 | README.md:403 | Main responses must arrive before 9 PM Saturday; adaptation responses before 11 AM Sunday. | manual | - | Form timestamp deadline; not a file property. |
| EXP-03 | EXPERIMENTS.md:47 | test answers are private, so that checks format only. | manual | - | The suite must not compute any test score; enforced by design, not by a file check. |

### packaging (9)

| ID | Source | Quote | Testable by | Check IDs | Note |
| --- | --- | --- | --- | --- | --- |
| RUL-28 | RULES.md:51 | Submit the CSV (up to 10 MB) and code ZIP (up to 100 MB) | contract.py | `CON-CSV-SIZE`, `PKG-ZIP-SIZE` |  |
| RUL-29 | RULES.md:51 | The ZIP must open directly to `README.md` and `reproduce.json` | contract.py | `PKG-ZIP-ROOT` |  |
| RUL-35 | RULES.md:55 | pin extra packages in a root-level `requirements.txt` | contract.py | `PKG-REQUIREMENTS-PINNED` |  |
| GDE-05 | README.md:204 | Include any project modules it imports in your ZIP. | entry.py | `ENT-FRESH-FOLDER` |  |
| GDE-11 | README.md:329 | Leave out unrelated data, secrets, and temporary outputs. | contract.py | `PKG-NO-ANSWER-FILES`, `PKG-NO-DATA-COPY` |  |
| GDE-12 | README.md:333 | "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"] | contract.py | `CON-REPRODUCE-SCHEMA` |  |
| GDE-24 | README.md:365 | Limits are 5,000 entries, 100 MB compressed, and 512 MiB unpacked. | contract.py | `PKG-ZIP-LIMITS`, `PKG-ZIP-SIZE` |  |
| GDE-25 | README.md:365 | Symlinks, encrypted files, unsafe paths, and duplicate paths are rejected. | contract.py | `PKG-ZIP-SAFE` |  |
| SUB-01 | SUBMISSION_README.md:9 | Season ID: [copy `season_id` from this round's `public/metadata.json`] | contract.py | `PKG-README-SEASON` |  |

### robustness (15)

| ID | Source | Quote | Testable by | Check IDs | Note |
| --- | --- | --- | --- | --- | --- |
| RUL-15 | RULES.md:30 | your code must recreate your predictions and follow the data rules | entry.py | `ENT-MATCHES-SUBMITTED`, `ENT-FRESH-FOLDER`, `ROB-ID-RENAME` | ROB-ID-RENAME is a suite-derived supporting check (predictions must not depend on ID strings). |
| RUL-16 | RULES.md:34 | Test answers and private organizer files are off limits. | entry.py | `ROB-MOUNT-NO-ANSWERS`, `CAU-ANSWERS-REMOVED` |  |
| RUL-23 | RULES.md:38 | Your submitted code must make its predictions without internet access or external APIs. | entry.py | `ROB-OFFLINE` |  |
| RUL-36 | RULES.md:55 | Runtime limits are 30 minutes, two CPU cores, and 16 GiB RAM | entry.py | `ROB-RUNTIME-LIMIT`, `ROB-CPU-LIMIT`, `ROB-MEM-LIMIT` |  |
| RUL-37 | RULES.md:55 | a maximum probability difference of `1e-8` after normalization | entry.py | `ROB-DETERMINISM` |  |
| GDE-14 | README.md:340 | It runs from `/submission`, where your code is read-only. | entry.py | `ROB-READONLY-MOUNTS` |  |
| GDE-15 | README.md:342 | `/data`: the current round's public data, read-only. | entry.py | `ROB-READONLY-MOUNTS`, `ROB-MOUNT-NO-ANSWERS` |  |
| GDE-16 | README.md:343 | `/main_data`: main-round public data, available during adaptation. | entry.py | `ENT-MAIN-DATA-MOUNT` |  |
| GDE-17 | README.md:344 | `/tmp`: writable scratch space, limited to 128 MiB. | entry.py | `ROB-TMP-QUOTA` |  |
| GDE-18 | README.md:345 | `/output`: writable output space, limited to 32 MiB. | entry.py | `ROB-OUTPUT-QUOTA` |  |
| GDE-20 | README.md:347 | It doesn't receive your uploaded forecast CSV as an input. | entry.py | `ROB-NO-FORECAST-INPUT` |  |
| GDE-22 | README.md:349 | Organizers prepare those dependencies in an isolated environment before the offline prediction run. | entry.py | `ENT-DEPS-PREINSTALL` |  |
| GDE-26 | README.md:367 | The largest allowed difference per probability is `1e-8` after CSV normalization. | entry.py | `ROB-DETERMINISM` |  |
| SUB-02 | SUBMISSION_README.md:41 | If reruns are not deterministic, describe the source of variation and how the submitted predictions were produced. | entry.py | `ROB-DETERMINISM` | Permits non-determinism if documented, which conflicts with the 1e-8 rule. See contradictions. |
| MET-05 | public/metadata.json:23 | Public demo seeds are reconstructable; event seeds are private. | manual | - | Entries must not rely on event seeds; a code-review item. |

### causality (7)

| ID | Source | Quote | Testable by | Check IDs | Note |
| --- | --- | --- | --- | --- | --- |
| RUL-18 | RULES.md:35 | Use observations strictly before the prediction tick, including when making training examples from earlier history. | perturb.py | `CAU-FUTURE-TICKS` |  |
| RUL-19 | RULES.md:35 | Missing ticks are unobserved, not zero payouts. | perturb.py | `CAU-MISSING-PROBE` | Info-only probe; the doc states the rule but the suite cannot prove internal treatment. |
| RUL-21 | RULES.md:36 | Condition labels are supplied for training/validation only. | perturb.py | `CAU-CONDITIONS-REMOVED` |  |
| DAT-07 | DATA.md:33 | Another machine's result at the tick you're predicting is not an available input. | perturb.py | `CAU-CUTOFF-OTHERS` |  |
| GDE-02 | README.md:140 | For cutoff 200, use ticks below 200. Another machine's result at tick 200 is unavailable too. | perturb.py | `CAU-FUTURE-TICKS`, `CAU-CUTOFF-OTHERS` |  |
| GDE-03 | README.md:142 | A two-tick lag isn't necessarily the second-most-recent row. | perturb.py | `CAU-MISSING-PROBE` | Lags are by tick value, not row offset; probe only (info). |
| GDE-04 | README.md:143 | Use other machines' earlier results only within the same episode. | perturb.py | `CAU-CROSS-EPISODE` |  |

### scoring (20)

| ID | Source | Quote | Testable by | Check IDs | Note |
| --- | --- | --- | --- | --- | --- |
| RUL-01 | RULES.md:9 | Main round — 60 points. | scoring.py | `SCO-MAX-POINTS` |  |
| RUL-02 | RULES.md:11 | Round two (adaptation) — 40 points. | scoring.py | `SCO-MAX-POINTS` |  |
| RUL-04 | RULES.md:13 | Round two doesn't give a bonus for improving on your main score. | scoring.py | `SCO-NO-IMPROVEMENT-BONUS` |  |
| RUL-05 | RULES.md:17 | average squared probability error within each episode, then average the episodes | scoring.py | `SCO-FORMULA` |  |
| RUL-06 | RULES.md:17 | Calculate `B[c]` the same way for the frozen pooled starter fitted on that round's training data. | scoring.py | `SCO-REFERENCE-SOURCE`, `SCO-REFERENCE-REPRO` |  |
| RUL-07 | RULES.md:20 | R = mean over four conditions of E[c] / max(B[c], 1e-9) | scoring.py | `SCO-FORMULA`, `SCO-REF-FLOOR` |  |
| RUL-08 | RULES.md:23 | final points = main points + adaptation points | scoring.py | `SCO-FINAL-SUM` |  |
| RUL-09 | RULES.md:26 | Each counts equally within its round. | scoring.py | `SCO-CONDITION-EQUAL` |  |
| RUL-10 | RULES.md:26 | Main and adaptation have separate frozen references. | scoring.py | `SCO-REFERENCE-SOURCE` |  |
| RUL-11 | RULES.md:26 | exact knowledge of all probabilities scores 100 | scoring.py | `SCO-TRUTH-MAX` |  |
| RUL-12 | RULES.md:26 | The pooled starter normally scores 30 + 20 = 50 | scoring.py | `SCO-REF-HALF` |  |
| RUL-13 | RULES.md:28 | sampled jackpots do not decide the ranking | scoring.py | `SCO-OUTCOME-INDEPENDENT` |  |
| RUL-14 | RULES.md:30 | Equal totals share a rank; organizers will announce how prize ties are handled before kickoff. | scoring.py | `SCO-TIE-RANK` | Tie-breaking scheme is unannounced; the check only pins equal-total sharing. |
| RUL-30 | RULES.md:53 | A missing or invalid round-two entry earns zero for that round. | scoring.py | `SCO-MISSING-ZERO` |  |
| RUL-31 | RULES.md:53 | You need a valid main entry for an overall ranking. | scoring.py | `SCO-MAIN-REQUIRED` |  |
| DAT-12 | DATA.md:14 | Frozen pooled starter forecasts | scoring.py | `SCO-REFERENCE-REPRO` | Table row: reference.csv (validation) = frozen pooled starter forecasts. |
| GDE-06 | README.md:241 | The older generic `score` field is a different diagnostic and isn't used for ranking. | scoring.py | `SCO-SCORE-NOT-RANKING` |  |
| GDE-07 | README.md:241 | Use the checker's validation points, or `validation.contest.technical_points` in `run.json` | scoring.py | `SCO-MATCHES-KIT` |  |
| GDE-08 | README.md:263 | It chooses a model using half the validation episodes in each condition | scoring.py | `SCO-HALF-SPLIT` |  |
| EXP-01 | EXPERIMENTS.md:23 | Model size and architecture don't earn points by themselves. | scoring.py | `SCO-FORECAST-ONLY` |  |

### data-fact (14)

| ID | Source | Quote | Testable by | Check IDs | Note |
| --- | --- | --- | --- | --- | --- |
| RUL-20 | RULES.md:36 | Earlier outcomes can interact at lags up to eight ticks. | generator.py | `GEN-LAG-RANGE` |  |
| RUL-22 | RULES.md:36 | Machines of the same type can respond differently or oppositely | generator.py | `GEN-MECHANISM-FAMILY` |  |
| DAT-01 | DATA.md:9 | query_id, episode_id, machine_id, machine_type, signal, cutoff | contract.py | `DAT-SCHEMA-COLUMNS` |  |
| DAT-03 | DATA.md:21 | Types are `steady`, `balanced`, and `jackpot`; the observable `signal` is -1, 0, or 1. | contract.py | `DAT-QUERY-ENUMS` |  |
| DAT-04 | DATA.md:23 | Cutoffs range from 192 to 256 ticks. | generator.py | `DAT-CUTOFF-RANGE` |  |
| DAT-05 | DATA.md:23 | Long histories include every earlier tick. Short histories include 40 recent consecutive ticks and 24 sampled older ticks. | generator.py | `GEN-HISTORY-SHAPE`, `DAT-HISTORY-CAUSAL` |  |
| DAT-06 | DATA.md:25 | 0 loses the chip, 1 returns it, and 2, 5, or 20 returns that many chips. | contract.py | `DAT-PAYOUT-SET` | Payout set {0,1,2,5,20} is the testable part; gross-return semantics are not file-checkable. |
| DAT-08 | DATA.md:33 | Ticks are generated synchronously from earlier outcomes. | generator.py | `GEN-SYNCHRONOUS` |  |
| DAT-09 | DATA.md:37 | Main and adaptation have separate training, validation, and test episodes with disjoint IDs. | contract.py | `DAT-SPLIT-DISJOINT` |  |
| DAT-13 | DATA.md:31 | Both the chance of winning and the sizes of positive payouts can change. | generator.py | `GEN-CONDITION-SEMANTICS` |  |
| MET-01 | public/metadata.json:2 | "conditions": ["stable", "short_history", "drift", "shared_change"] | contract.py | `DAT-CONDITION-SET` | Array spans lines 2-7 of the file. |
| MET-02 | public/metadata.json:8 | "counts_per_condition": {"test": 48, "train": 24, "validation": 24} | contract.py | `DAT-COUNTS-PER-CONDITION` | Object spans lines 8-12. Verified: train and validation have 24 episodes per condition, test has 192 episodes = 4 x 48. |
| MET-03 | public/metadata.json:14 | "machines_per_episode": 8 | contract.py | `DAT-MACHINES-PER-EPISODE` |  |
| MET-04 | public/metadata.json:13 | Machine-specific interactions, lags up to 8, missing observations and changing regimes. | generator.py | `GEN-LAG-RANGE`, `GEN-MAIN-LIKE-CALIBRATION` |  |

## Check IDs (implement exactly these)

| Check ID | Definition | Module |
| --- | --- | --- |
| `CON-CSV-COLUMNS` | Header row equals exactly 'query_id,p_0,p_1,p_2,p_5,p_20' in that order; no index column, no extra, missing or duplicate column. Fail otherwise. (Kit io.py accepts any order; suite is stricter, see REQUIREMENTS.md.) | contract.py |
| `CON-CSV-STRICT` | Parsed with Python csv (strict): every row has exactly 6 fields and no empty cell. Fail on any ragged or empty row. | contract.py |
| `CON-CSV-UTF8` | File bytes decode as UTF-8. A leading UTF-8 BOM is accepted with severity warn; any other encoding (UTF-16, cp1252, Excel workbook) fails. | contract.py |
| `CON-CSV-SIZE` | Submission CSV size <= 10,000,000 bytes (10 MB, MB = 1e6 bytes). Fail if larger. (MAX_CSV = 16 MiB in submission_contract.py is weaker and unused; doc value governs.) | contract.py |
| `CON-QUERY-SET` | Set of query_id values equals exactly the query_id set of <round>/test/queries.csv: no missing, no extra, no duplicate rows. Row order ignored. Report up to 5 missing and 5 extra IDs. | contract.py |
| `CON-PROB-VALUES` | Each p_0..p_20 cell parses as a finite decimal (no NaN, inf, empty, '%' or thousands separator) with 0 <= p <= 1 inclusive. Fail otherwise. | contract.py |
| `CON-PROB-SUM` | Per row, \|fsum(p_0..p_20) - 1\| <= 1e-6. Fail otherwise. Normalisation used elsewhere is p_k / sum(p) (schema.probabilities), applied only after this check passes. | contract.py |
| `CON-ROUND-METADATA` | <data>/metadata.json is a JSON object with suite_version in {temporal-suite-v2, experimentation-suite-v3}; round='main' implies max_points=60, round='adaptation' implies max_points=40; season_id matches ^[0-9a-f]{24}$; conditions list is the four names. | contract.py |
| `CON-REPRODUCE-SCHEMA` | reproduce.json decodes to a JSON object with exactly the keys {image, command}; command is a JSON array of 1-64 strings, each non-empty, <= 4096 characters, no NUL, executed as argv (no shell). File <= 65,536 bytes (kit limit). Command must be the argv form in README.md:340. | contract.py |
| `CON-IMAGE-PYTHON` | reproduce.json 'image' equals exactly the string 'python'. (Kit regex submission_contract.py:59 is broader; the documented value governs.) | contract.py |
| `CON-OUTPUT-PATH` | After the offline run, /output/submission.csv exists as a regular file produced during the run (an empty or pre-existing file does not count). | contract.py |
| `CON-EXIT-ZERO` | The reproduce command exits with status 0 within the runtime limit (ROB-RUNTIME-LIMIT). | contract.py |
| `PKG-ZIP-SIZE` | Code ZIP file size <= 100,000,000 bytes (100 MB). Fail if larger. (MAX_ZIP = 128 MiB in submission_contract.py is weaker; doc value governs.) | contract.py |
| `PKG-ZIP-LIMITS` | ZIP has <= 5,000 entries (directory entries included) and sum of declared uncompressed sizes <= 536,870,912 bytes (512 MiB). | contract.py |
| `PKG-ZIP-SAFE` | Reject: symlinks or non-regular entries; encrypted members (flag bit 0); case-folded duplicate paths; absolute paths, '..' or '.' components, backslashes, ':' in a component, Windows device names (con, prn, aux, nul, com0-9, lpt0-9), control characters; a path used as both file and directory; normalised or truncated names; member CRC or streamed size mismatch. Implement by reusing submission_contract.inspect_archive. | contract.py |
| `PKG-ZIP-ROOT` | README.md and reproduce.json are top-level members (no enclosing folder); each <= 65,536 bytes; README.md is non-empty after strip. | contract.py |
| `PKG-REQUIREMENTS-PINNED` | If root requirements.txt exists, every line that is not blank or a comment matches ^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9._,-]+\])?==[A-Za-z0-9.!+_-]+$ (exact == pin). Lines starting with '-', containing a URL, 'git+' or ' @ ' fail. Absent file passes. | contract.py |
| `PKG-NO-ANSWER-FILES` | ZIP contains no member whose basename is targets.csv, labels.csv, conditions.csv or reference.csv, and no path component named _truth. Fail if any found. | contract.py |
| `PKG-NO-DATA-COPY` | Warn (not fail) if ZIP contains basenames queries.csv, history.csv or sample_submission.csv, or any member under a public/ directory. | contract.py |
| `PKG-README-SEASON` | ZIP README.md contains the season_id string from the round's metadata.json and the round name ('main' or 'adaptation') matching metadata round. Severity warn. | contract.py |
| `CAU-FUTURE-TICKS` | Append synthetic history rows with tick >= cutoff for every machine in each scored episode (same machine and others, random payouts from {0,1,2,5,20}). Predictions must stay identical. Fail on any change. | perturb.py |
| `CAU-CUTOFF-OTHERS` | Append only other machines' rows at tick == cutoff (random payouts) to the scored episode. Predictions must stay identical. Fail on any change. | perturb.py |
| `CAU-CONDITIONS-REMOVED` | Scored split with conditions.csv removed (or condition labels shuffled across queries). Predictions must stay identical. Fail on any change. | perturb.py |
| `CAU-ANSWERS-REMOVED` | Scored split with labels, targets, reference and conditions files removed versus the full copy. Predictions must stay identical. Fail on any change. | perturb.py |
| `CAU-CROSS-EPISODE` | Replace the history and labels of every episode other than the scored one with random payouts (cutoffs and tick structure kept). Predictions for the scored episode's queries must stay identical. Not in DESIGN.md; proposed addition. | perturb.py |
| `CAU-MISSING-PROBE` | Info only, never fails. Compare (a) a missing tick left absent versus (b) the same tick inserted with payout 0. Report max \|delta p\| and the affected query_ids. Severity info. | perturb.py |
| `ROB-ROW-ORDER` | Shuffle row order of queries.csv and history.csv (fixed seed). Predictions must stay identical by query_id. Fail on any change. | perturb.py / entry.py |
| `ROB-ID-RENAME` | Apply a consistent injective renaming of query_id, episode_id and machine_id (new IDs matching the kit identifier regex). After mapping old to new IDs, predictions must stay identical. Fail on any change. | perturb.py / entry.py |
| `ROB-DETERMINISM` | Two independent runs in fresh output dirs with identical settings: max \|delta p\| <= 1e-8 after row normalisation (p_k / sum(p)) over all rows, with identical query_id sets. Fail otherwise. | perturb.py / entry.py |
| `ROB-OFFLINE` | The reproduce command completes with the container network disabled (--network none); any network attempt that makes the run fail is a fail. Local mode is non-authoritative and does not enforce this. | perturb.py / entry.py |
| `ROB-RUNTIME-LIMIT` | Wall-clock runtime of the offline run <= 1800 seconds. Killed at the limit is a fail. | perturb.py / entry.py |
| `ROB-CPU-LIMIT` | Run is executed under a 2-CPU quota (--cpus=2). Pass when it completes successfully under that quota. Report reported CPU count as info. | perturb.py / entry.py |
| `ROB-MEM-LIMIT` | Run is executed under a 16 GiB memory limit (17,179,869,184 bytes, no extra swap). OOM kill or non-zero exit due to memory is a fail. Report peak memory as info. | perturb.py / entry.py |
| `ROB-READONLY-MOUNTS` | /data and /submission are mounted read-only (mount flags contain 'ro'); the working directory is /submission. Any writable mount for those paths is a fail. /tmp and /output must be writable. | perturb.py / entry.py |
| `ROB-TMP-QUOTA` | /tmp is a writable scratch mount limited to 134,217,728 bytes (128 MiB). Run failing with ENOSPC or killed by the quota is a fail. | perturb.py / entry.py |
| `ROB-OUTPUT-QUOTA` | /output is a writable mount limited to 33,554,432 bytes (32 MiB). Run failing with ENOSPC or killed by the quota is a fail. | perturb.py / entry.py |
| `ROB-MOUNT-NO-ANSWERS` | Mounted /data (and /data/test) contains no file named targets.csv, labels.csv, conditions.csv or reference.csv and no _truth directory. sample_submission.csv is allowed. Fail if any answer file is present. | perturb.py / entry.py |
| `ROB-NO-FORECAST-INPUT` | Before the run, /output is empty and no file named submission.csv or validation_predictions.csv exists in the entry folder or any mounted path (sample_submission.csv is allowed). The run must still produce /output/submission.csv itself. | perturb.py / entry.py |
| `ENT-FRESH-FOLDER` | Extract the code ZIP into an empty folder and run the reproduce command from that folder only (plus mounted data). Success with no reliance on kit files or host paths. Fail otherwise. | entry.py |
| `ENT-MATCHES-SUBMITTED` | The fresh-run CSV matches the uploaded forecast CSV: identical query_id set and max \|delta p\| <= 1e-8 after row normalisation. Fail otherwise. | entry.py |
| `ENT-DEPS-PREINSTALL` | Pinned packages from root requirements.txt are installed (with network) into a derived image before the offline run. An install failure is a fail. A failed offline import is a fail. | entry.py |
| `ENT-MAIN-DATA-MOUNT` | Round 'adaptation': /main_data is mounted read-only and contains the main-round public tree. Round 'main': no /main_data mount is expected (info only if present). | entry.py |
| `ENT-BASE-PY313` | Base image reports Python 3.13 (sys.version_info[:2] == (3, 13)). Mismatch is info; below 3.11 is fail. | entry.py |
| `SCO-FORMULA` | Independent reimplementation equals the scorer: per query e = sum_k (p_k - q_k)^2 over the five payouts (q = target probabilities normalised to sum 1); episode error = mean over its queries; E[c] = mean over episodes of condition c with equal episode weight; R = mean over the four conditions of E[c]/max(B[c],1e-9); points = max_points/(1+R). Tolerance \|delta points\| <= 1e-9. | scoring.py |
| `SCO-MATCHES-KIT` | The suite's technical_points on the kit validation split equal check_submission.py's technical_points (rounds.score_round) within 1e-9; the printed 6-decimal value equals round(points, 6). | scoring.py |
| `SCO-MAX-POINTS` | max_points is 60 for round 'main' and 40 for round 'adaptation' (from metadata); points lie in [0, max_points]. contest.py's default 70 and its 0..70 review range are legacy and must not be used. | scoring.py |
| `SCO-REF-FLOOR` | Denominator is max(B[c], 1e-9) exactly: a zero reference error uses 1e-9 (no division by zero), and E[c] = 0 gives ratio 0. | scoring.py |
| `SCO-REFERENCE-SOURCE` | B[c] uses the frozen reference of the same round only (validation/reference.csv for kit scoring; the pooled starter fitted on that dataset's train for synthetic data). Main and adaptation never share a reference. | scoring.py |
| `SCO-REFERENCE-REPRO` | Refit the kit pooled starter (strength 20) on the round's training observations and compare with reference.csv: max \|delta p\| <= 1e-9 passes; otherwise warn (reference not reproducible from train). | scoring.py |
| `SCO-TRUTH-MAX` | Scoring the target probabilities as the submission gives points == max_points within 1e-9 (R = 0, score 100 in the diagnostic). | scoring.py |
| `SCO-REF-HALF` | Scoring the frozen reference as the submission gives R = 1 in every condition and points == max_points/2 within 1e-9. | scoring.py |
| `SCO-OUTCOME-INDEPENDENT` | Replacing each target 'outcome' with a different payout while keeping the probabilities leaves points unchanged (within 1e-12). | scoring.py |
| `SCO-CONDITION-EQUAL` | R is the unweighted arithmetic mean of the four condition ratios; each condition's error uses only its own queries (perturbing one condition's predictions changes only that ratio). | scoring.py |
| `SCO-FINAL-SUM` | Final points = main points + adaptation points, with each round's points in [0, max_points] (rounds.combine_rounds); a round with no valid entry contributes 0. | scoring.py |
| `SCO-MISSING-ZERO` | A missing or invalid round entry (fails any CON-*/PKG-* fail check, or is absent) contributes exactly 0 points for that round. | scoring.py |
| `SCO-MAIN-REQUIRED` | Overall ranking requires a valid main entry; a team without one is flagged as unranked (ranking-level flag, not a file check). | scoring.py |
| `SCO-NO-IMPROVEMENT-BONUS` | Adaptation points depend only on the adaptation CSV: changing the main-round entry or main score leaves adaptation points unchanged (within 1e-12). | scoring.py |
| `SCO-HALF-SPLIT` | Validation episodes are split into two halves stratified by condition: within each condition the halves differ in size by at most one, and the split is deterministic for a fixed seed. Used by the kit experiment runner and any held-out evaluation. | scoring.py |
| `SCO-FORECAST-ONLY` | Points depend only on the submitted CSV and the target file: identical CSVs produced by different code or runtime give identical points (exact). | scoring.py |
| `SCO-SCORE-NOT-RANKING` | The diagnostic 'score' field (100*(1 - error/2)) is reported at most as diagnostic; no ranking, pass/fail or points output may use it. | scoring.py |
| `SCO-TIE-RANK` | Entries whose totals are equal (\|delta\| <= 1e-9) share one rank; the next distinct total takes its position (competition ranking). Severity info until organizers announce the tie scheme. | scoring.py |
| `DAT-SCHEMA-COLUMNS` | Input files have exactly the documented columns: queries.csv (query_id, episode_id, machine_id, machine_type, signal, cutoff); history.csv (episode_id, machine_id, tick, payout); labels.csv (query_id, outcome); conditions.csv (query_id, condition); targets.csv (query_id, episode_id, p_0..p_20, outcome); reference.csv (query_id, p_0..p_20). | contract.py |
| `DAT-TEST-SPLIT-FILES` | The test split contains queries.csv and history.csv (and sample_submission.csv); it contains no labels.csv, conditions.csv, targets.csv or reference.csv. Fail otherwise. | contract.py |
| `DAT-QUERY-ENUMS` | machine_type is in {steady, balanced, jackpot}; signal is an integer in {-1, 0, 1}. | contract.py |
| `DAT-CUTOFF-RANGE` | Every cutoff is an integer in [192, 256]. | contract.py |
| `DAT-PAYOUT-SET` | Every payout in history.csv and every outcome in labels.csv / targets.csv is an integer in {0, 1, 2, 5, 20}. | contract.py |
| `DAT-CONDITION-SET` | Condition labels are a subset of {stable, short_history, drift, shared_change}; every announced condition occurs; each episode maps to exactly one condition. | contract.py |
| `DAT-COUNTS-PER-CONDITION` | Episodes per condition equal metadata counts_per_condition for splits with labels (train 24, validation 24). For test (no labels), total episodes equal 4 x counts_per_condition.test (= 192). Counts are episodes, not queries. | contract.py |
| `DAT-MACHINES-PER-EPISODE` | Each episode has exactly 8 distinct machine_id values (metadata machines_per_episode = 8); (episode_id, machine_id) is unique in queries.csv. | contract.py |
| `DAT-SPLIT-DISJOINT` | episode_id and query_id sets are disjoint across train, validation and test within a round, and across main and adaptation when both are present. | contract.py |
| `DAT-HISTORY-CAUSAL` | No history row has tick >= the cutoff of its machine's query (ticks are integers >= 0); no duplicate (episode_id, machine_id, tick). Fail otherwise. Verified: 0 violations in the public train, validation and test files. | contract.py |
| `GEN-LAG-RANGE` | Every generated interaction uses an integer lag in [1, 8]; each machine has 1 to 3 lag interactions. | generator.py |
| `GEN-MECHANISM-FAMILY` | The sampled family includes pairwise boolean interactions (XOR, AND, OR), single-lag effects with random sign, and optional neighbour-machine lag terms; two machines of the same type can get opposite signs (checked over a sample). | generator.py |
| `GEN-HISTORY-SHAPE` | Long histories contain every tick < cutoff. Short histories contain exactly the 40 ticks cutoff-40..cutoff-1 plus 24 distinct older ticks (64 total), and no tick >= cutoff. | generator.py |
| `GEN-SYNCHRONOUS` | Outcome at (machine m, tick t) is drawn using only state and outcomes at ticks < t; no draw for (m, t) reads any outcome at the same tick t (instrumented generator). | generator.py |
| `GEN-CONDITION-SEMANTICS` | stable: no drift or changepoint; short_history: only short-history queries; drift: relations interpolate gradually across the episode; shared_change: one changepoint shared by all machines of the episode. Payout-size and win-rate changes are allowed in drift and shared_change. | generator.py |
| `GEN-MAIN-LIKE-CALIBRATION` | For preset main_like, generated win rate by type x signal and payout share by type, and lag-k autocorrelations (k = 1..8), differ from public/train by at most 0.02 absolute (rates) and 0.03 (autocorrelations). Warn when exceeded. | generator.py |

## Contradictions and ambiguities between documents

Each item quotes both sides. The suite's resolution is what the check definitions above implement; organizers should confirm the ones marked **decision needed**.

- **C1 Column order (contract).** RULES.md:43-47 "submit one CSV with exactly these columns" with header `query_id,p_0,p_1,p_2,p_5,p_20`; vs slot_forecasting/io.py:22-23 (`set(header) != set(columns)`, order not enforced). Resolution: suite enforces the documented order (CON-CSV-COLUMNS, as DESIGN.md:52-53 says).
- **C2 Code ZIP size (decision needed).** RULES.md:51 and README.md:365 "100 MB compressed" vs slot_forecasting/submission_contract.py:11 `MAX_ZIP = 128 * 1024 * 1024` with error text "at most 128 MiB". Resolution: 100,000,000 bytes (PKG-ZIP-SIZE). The kit's check would accept up to 134,217,728 bytes.
- **C3 CSV size constant (decision needed).** README.md:367 "The final CSV may be at most 10 MB." vs submission_contract.py:14 `MAX_CSV = 16 * 1024 * 1024` (defined, never referenced). Resolution: 10,000,000 bytes (CON-CSV-SIZE).
- **C4 Invalid round-two entry (decision needed).** RULES.md:53 "A missing or invalid round-two entry earns zero for that round." vs RULES.md:55 "A replacement still counts if validation or its code check later fails" and README.md:310 "even if its files are invalid or its code later fails". Unclear whether an entry that fails only the code rerun (ENT-MATCHES-SUBMITTED) earns zero. Suite: CSV/ZIP format fails map to zero (SCO-MISSING-ZERO); code-rerun failures are reported, not zeroed automatically.
- **C5 Determinism vs allowance.** RULES.md:55 and README.md:367 require max difference `1e-8`; SUBMISSION_README.md:41 "If reruns are not deterministic, describe the source of variation and how the submitted predictions were produced." Suite treats >1e-8 as a fail (ROB-DETERMINISM).
- **C6 "after normalization" undefined.** RULES.md:55 "after normalization"; README.md:367 "after CSV normalization"; schema.py:73 `return tuple(v / total for v in result)`. Resolution: row-sum division (p_k / sum(p)) in CON-PROB-SUM and ROB-DETERMINISM.
- **C7 Data path.** DATA.md:39 "Main lives in `season/main/public`" vs the kit tree (public/ at the root; check_submission.py default `public/`). No season/ folder exists in this worktree. Suite takes the data root as an argument.
- **C8 Legacy contest code (not used).** slot_forecasting/contest.py:15 `max_points=70`, contest.py:65 `0 <= ... <= 70`, REVIEW_MAX human review (contest.py:10) and CONTEST_VERSION "evidence-contest-v1" vs RULES.md:9-11 (60/40, no human review). rounds.py:12 uses 60/40, so the event path is consistent; the contest.py defaults are not. Suite never uses them (SCO-MAX-POINTS).
- **C9 reproduce.json image (decision needed).** RULES.md:55 "Keep `\"image\": \"python\"`" (exact) vs submission_contract.py:59 regex `[a-z][a-z0-9_-]{0,31}` with message "organizer-approved environment". Resolution: exact "python" (CON-IMAGE-PYTHON).
- **C10 Scratch and output sizes (confirm).** README.md:344-345 "limited to 128 MiB" (/tmp) and "limited to 32 MiB" (/output) vs SUBMISSION_README.md:23 and DESIGN.md:19-20 (no sizes given). Suite uses the README values (ROB-TMP-QUOTA, ROB-OUTPUT-QUOTA).
- **C11 Missing reproduce template.** DESIGN.md:14-16 and the task refer to `entry_template/reproduce.json`; that file does not exist in this worktree (it exists only in sibling worktrees). The schema is taken from README.md:333-338 and build_entry.py:49.
- **C12 Gaps in long histories.** DATA.md:23 "Long histories include every earlier tick." vs DATA.md:23 "A missing tick is unknown, not a loss." The docs do not say whether a long history may still have gaps. Public data: long-history machines have no gaps (all 576 train machines with long histories are complete up to cutoff); only short-history machines have gaps. Suite allows gaps in any history (CAU-MISSING-PROBE is info only).
- **C13 counts_per_condition unit (confirm).** public/metadata.json:8-12 `"counts_per_condition": {"test": 48, "train": 24, "validation": 24}` does not say episodes. Data matches episodes (train/validation 24 episodes per condition = 192 queries; test 192 episodes = 4 x 48). Suite uses episodes (DAT-COUNTS-PER-CONDITION).
- **C14 Tie ranking (announce).** RULES.md:30 "Equal totals share a rank; organizers will announce how prize ties are handled before kickoff." Rank scheme is not stated; contest.py:90-96 uses competition ranking (1,1,3). SCO-TIE-RANK is info until announced.
- **C15 "score" vs points (no conflict, naming risk).** README.md:241 says the `score` field "isn't used for ranking"; scoring.py:99 `score = 100*(1-error/2)` is a different scale. Suite reports points only (SCO-SCORE-NOT-RANKING).
- **C16 Python version (not a conflict).** README.md:47 "You'll need Python 3.11 or newer" (local kit) vs README.md:349 "The base environment is Python 3.13" (entry image). Different environments; ENT-BASE-PY313 checks the image.
- **C17 CSV encoding (minor).** README.md:390 "Save as UTF-8 CSV." vs io.py:17 reads with `utf-8-sig`, silently accepting a BOM. Resolution: BOM is warn (CON-CSV-UTF8).

No conflicts were found between the documents on deadlines: 9 PM / 11 AM Central and the 14-hour window are consistent across RULES.md:11, README.md:15 and :271-275, and README.md:403.

## Data facts verified against public/ (this worktree)

- train: 768 queries, 96 episodes, 8 machines each; validation: 768 queries, 96 episodes; test: 1536 queries, 192 episodes.
- All cutoffs in [192, 256]; types and signals in their enumerations; payouts only in {0, 1, 2, 5, 20}.
- Condition counts: 192 queries (24 episodes) per condition in train and validation.
- No history row has tick >= cutoff and no negative tick, in any split (DAT-HISTORY-CAUSAL).
- Short-history machines: last 40 ticks before cutoff are all present for every machine; about 25% of pre-cutoff ticks are present. The exact 24-older-tick count is not verified.
- test/ contains queries.csv, history.csv and sample_submission.csv only (no answers, no conditions).

## Notes for implementers

- Every `manual` row has `check_ids` = [] and a note; `CAU-MISSING-PROBE` is info-only and never fails.
- `ROB-ID-RENAME` has no direct doc sentence; it supports RUL-15 (suite-derived).
- `CON-QUERY-SET` covers the cross-round case (GDE-09): a main CSV fails on adaptation query IDs.
- Byte limits in the check definitions are the documented values; where kit code is looser, the conflict is listed in C2, C3 and C9.
