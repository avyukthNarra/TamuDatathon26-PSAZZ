# eval_suite — method-agnostic evaluation derived from the kit docs

Goal: any entry (any model family) can be run through this suite and get a
definitive, reproducible result: pass/fail on every documented requirement,
plus the official score with confidence intervals and slices. The suite is
built ONLY from the documents (README.md = participant guide, RULES.md,
DATA.md, SUBMISSION_README.md, EXPERIMENTS.md, public/metadata.json) and the
kit code (check_submission.py, build_entry.py, slot_forecasting/{scoring,
contest,submission_contract,starter,io,schema}.py). It must never import or
assume anything about a team model (pair mixer, rule model, etc.).

## The entry under test

An "entry" is a directory laid out like the code ZIP root: `README.md`,
`reproduce.json` (`{"image": "python", "command": [...]}`), optional
`requirements.txt`, plus code. The organizer contract (SUBMISSION_README):
current round mounted at `/data` (read-only), code at `/submission`
(read-only), writable `/tmp` and `/output`, adaptation also mounts
`/main_data`; command must write `/output/submission.csv`, offline;
limits 30 min, 2 CPUs, 16 GiB; determinism within 1e-8 after normalisation.

## Datasets

A dataset is a directory in the kit's public layout:
`<root>/{train,validation,test}/` with `queries.csv`, `history.csv`, and for
train/validation `labels.csv`, `conditions.csv`, (validation) `targets.csv`,
`reference.csv`, plus `<root>/metadata.json`. Synthetic datasets ALSO write a
private `<root>/_truth/test/{targets.csv,conditions.csv,reference.csv}` that
is NEVER mounted into the entry. For the real kit, test truth is unknown, so
real-data scoring uses `validation` as the scored split: the entry runs on a
dataset copy whose `test/` holds the real validation queries+history with all
answer files removed, and whose `validation/` is replaced by a held-out part
(see scoring.py: k-fold over validation episodes so the entry never sees the
answers of the episodes it is scored on).

## Modules (one owner each; keep these interfaces exactly)

- `requirements.py` + `REQUIREMENTS.md` — traceability matrix: every
  testable requirement with ID (e.g. `RUL-07`), source `file:line`, quote,
  category (contract | causality | scoring | robustness | packaging), and
  the check ID(s) that cover it. `requirements.py` exposes
  `REQUIREMENTS: list[dict]` with the same fields.
- `entry.py` — `run_entry(entry_dir, data_root, *, main_data_root=None,
  mode="docker"|"local", timeout_s=1800, cpus=2, memory_gb=16) -> RunResult`
  (dataclass: `csv_path, wall_s, exit_code, stdout_tail, stderr_tail,
  peak_mem_mb|None, mode`). Docker mode = organizer emulation
  (`--network none`, `--cpus`, `--memory`, read-only mounts, image from
  reproduce.json; pinned requirements installed into a cached derived image
  BEFORE the offline run). Local mode = same command with the five mount
  paths rewritten to temp dirs (fast; reported as non-authoritative).
- `contract.py` — checks returning `CheckResult(id, passed, severity
  ("fail"|"warn"|"info"), message, details)`: CSV columns exactly
  `query_id,p_0,p_1,p_2,p_5,p_20`; every query exactly once, no extras;
  finite, in [0,1], sums within 1e-6; size ≤ 10 MB (10,000,000 bytes); ZIP
  opens to README.md and reproduce.json at root (reuse
  `submission_contract.inspect_archive`), ≤ 100 MB, image == "python",
  extra packages pinned in root requirements.txt; runtime ≤ 1800 s;
  determinism (two runs, max abs diff ≤ 1e-8 after normalisation).
- `perturb.py` — black-box causality/invariance variants of a dataset,
  `make_variants(data_root, out_root) -> list[Variant(name, data_root,
  relation, id_map|None, requirement_ids)]`: future ticks (>= cutoff)
  appended → identical; other machines' outcome AT the cutoff tick appended
  → identical; condition files removed/shuffled → identical; answer files
  removed from scored split → identical; query and history row order
  shuffled → identical by query_id; consistent id renaming → identical
  after mapping; "missing tick vs zero payout" probe (info only).
  `compare(base_csv, variant_csv, variant) -> CheckResult`.
- `scoring.py` — official metric from RULES.md (per-episode mean squared
  error over the 5 classes, averaged over episodes within each condition,
  E[c]/max(B[c],1e-9), R = mean over conditions, points =
  max_points/(1+R), 60 main / 40 adaptation); MUST equal check_submission.py
  on the kit validation; reference B from `reference.csv` or the kit's pooled
  starter fitted on that dataset's train; condition-stratified episode
  bootstrap CIs; repeated stratified half-splits as in EXPERIMENTS.md;
  slices by condition, machine_type, signal, history kind (long/short,
  inferred from history density), cutoff bin, per-episode; paired
  comparison of two prediction sets (paired bootstrap of the points
  difference).
- `generator.py` — spec-driven simulator (GENERATOR section) writing full
  datasets including `_truth`.
- `cli.py` — `python -m eval_suite run --entry DIR [--compare DIR2]
  [--suite quick|full] [--mode docker|local] --out runs/eval/<name>` →
  `report.md` + `report.json`: requirement table (pass/fail per
  REQUIREMENTS id), contract/causality results, scores on real validation
  and on every synthetic scenario with CIs and slices, runtime/memory.
- `tests/` — tests OF THE SUITE ITSELF, never of a team method: the metric
  equals check_submission.py; truth scores max_points; the frozen reference
  scores max/2; uniform forecasts score below reference; contract checks
  catch deliberately broken CSVs/ZIPs; perturbations catch a deliberately
  leaky dummy entry. Dummy entries live in `eval_suite/fixtures/`
  (reference-only, uniform, future-leaky, broken-format).

## GENERATOR (documented facts only; broader than any one model)

Facts from DATA.md/RULES.md/metadata: 8 machines per episode; types steady,
balanced, jackpot; observable signal in {-1,0,1}; payouts {0,1,2,5,20};
cutoffs 192–256; long histories = every earlier tick, short = 40 recent
consecutive + 24 sampled older ticks; missing ticks are unknown; outcomes
interact at lags up to 8; same-type machines can react differently or
oppositely; useful patterns repeat across machines with differing
direction; win chance and payout sizes can both change; some relationships
gradually reverse, others change together across an episode; earlier
outcomes on neighbouring machines can help; hidden state leaves
uncertainty; ticks generated synchronously from earlier outcomes; four
conditions with equal counts; round two changes which temporal
relationships recur.

Sample from a FAMILY wide enough to contain many mechanisms, not one team's
hypothesis: per machine 1–3 lag interactions from lags 1..8 (pairwise
boolean functions incl. XOR/AND/OR and single-lag effects, random sign),
optional neighbour-machine lag terms, a hidden Markov state, type- and
signal-dependent base rates, payout-size distributions with lag-dependent
size shifts; drift = gradual interpolation to a changed/reversed relation;
shared change = one changepoint for all machines of an episode. Presets:
`main_like` (calibrated so generated train statistics match real
public/train: win rate by type and signal, payout mix by type, lag-k
autocorrelations, sampling), `round2_shift`, `all_lags`, `neighbour_heavy`,
`hidden_state_heavy`, `fast_drift`, `edge_cases` (cutoff 192 and 256,
extreme sparsity, all-loss and rare-jackpot machines). Each preset writes
train/validation/test with exact true conditional probabilities and a
reference from the kit's pooled starter fitted on that dataset's train.
Outputs must load with the kit's loaders and pass check_submission.py.
