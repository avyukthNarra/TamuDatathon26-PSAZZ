# v7: size atoms, episode flip, round-two discovery

Submission candidate. v6's pair mixer, plus the train-learned size-atom rule from g10, the episode parity swap from g6, and the round-two discovery patch from g7. The patch stays in because on main validation it does not change the score.

Full validation is **57.2045 / 60** (R = 0.04887). A fresh-process rerun matches `submission.csv` with max absolute difference 0 (limit 1e-8). The package run, including cross-fit and the rerun, took **135 s**. `check_submission.py` on the test file reports VALID FORMAT, 1536 queries.

## What is on

`entry_config.py` in the archive:

- `USE_SIZE`. EM on `/data/train` fits two payout atoms per machine type. Lags in 1..8 whose share of posterior mass exceeds 1/8 are kept. This train keeps `(1, 2, 4)`. Train EM log-likelihood −58619.85, the same fit g10 reported.
- `USE_FLIP`. The gate is the midpoint of the widest gap in the train episodes' flip log-Bayes-factors. This train: gap −32.54 to 237.88, midpoint **102.67**. An episode above the gate exchanges agree and differ counts at its joint MAP tick, then the mixture keeps learning. Validation swaps **48 / 96** episodes: all 24 drift and all 24 shared change, and none of the stable or short-history episodes.
- `USE_ROUND2`. Pair votes use each machine's clearest window (full history, first 45% of ticks, or last 40). Pairs used by at least 5% of scored machines are kept. An empty search returns all 28 pairs, not `(1, 3), (4, 7), (2, 5)`. A type that does not split at 0.5 gets a 2-means dead/live prior. If no size lag has positive gain, the size key is the machine-level key only.

On this train the round-two path selects the same pairs `(1, 3), (2, 5), (4, 7)`, the same size keys `((), (1,), (2,), (4,))`, and parity still wins the train Brier (0.35270 vs 0.35963). Jackpot still has 20 targets on each side of 0.5, so 2-means does not replace it. Validation and test predictions are byte-identical to size + flip without the patch.

## Ablation (main validation)

`python run_pair_mixer.py --cv-val --score-only` from `c2-combine/mainFiles`. Points are the official full-validation score. Ratios are loss versus the pooled reference. Cross-fit refits only the win prior, on the opposite episode half. The size atoms and the flip gate stay the train-only fit.

| | Points | stable | short_history | drift | shared_change | cross-fit halves |
|---|---:|---:|---:|---:|---:|---|
| v6 | 56.2856 | 0.05412 | 0.10470 | 0.04879 | 0.05635 | 56.13 / 56.43 |
| + size | 57.0203 | 0.04382 | 0.08438 | 0.03819 | 0.04263 | 57.03 / 56.97 |
| + size + flip | 57.2045 | 0.04382 | 0.08438 | 0.03306 | 0.03421 | 57.14 / 57.23 |
| + size + flip + round-two | **57.2045** | 0.04382 | 0.08438 | 0.03306 | 0.03421 | 57.14 / 57.23 |

The size rule is +0.735 against v6. The flip is another +0.184, on drift and shared change only. Stable and short history do not move. The round-two patch changes the score by 0.000000, which is inside the 0.05 keep threshold, so it stays.

v6 reproduces the brief's 56.29 (exact 56.2856). The size row reproduces g10's 57.0203. The flip gate and the 48 swapped episodes match g6's train gap and its validation mask.

## Simulated rounds

g7's three generators (`disjoint`, `four`, `sharedlag`), seeds already baked into `g7-round2/data`. Error is the episode-weighted mean squared error on the five probabilities, the same metric as g7. v6 here matches g7's published v6 to four digits (0.0156, 0.0207, 0.0308).

| | disjoint | four | sharedlag |
|---|---:|---:|---:|
| v6 | 0.0156 | 0.0207 | 0.0308 |
| flip only | 0.0133 | 0.0182 | 0.0146 |
| round-two patch only | 0.0135 | 0.0210 | 0.0272 |
| size only | 0.0580 | 0.0422 | 0.0609 |
| size + flip | 0.0551 | 0.0408 | 0.0471 |
| **final (size + flip + patch)** | **0.0521** | **0.0399** | **0.0434** |

The patch alone matches g7's published patched errors on disjoint (0.0135) and sharedlag (0.0272). On `four` it does not: the 2-means jackpot prior moves the train Brier so full cells win by 0.001 (0.37534 vs parity 0.37642). g7's published 0.0157 forced parity. With full cells the patch-only error is 0.0210, a hair above v6.

The flip alone improves all three rounds. On sharedlag, where the new direction is only about 23 ticks old, it cuts error from 0.0308 to 0.0146.

The size rule is the regression. These generators draw a static size atom. The rule replaces the target-derived size mixture with a lag bit, and the lags it keeps are not a stable size story (`(1, 3, 4, 6)`, `(1, 2, 6, 7)`, `(5, 6, 7, 8)`). That raises error by roughly 0.02 to 0.04, more than the flip and the patch give back. The 60-point figures on each round's own reference are 55.64 → 47.53, 52.50 → 47.02, and 51.63 → 48.80. Those points are not on main's scale.

## Eval suite

From `worktrees/eval-main/mainFiles`, Docker mode (`python:3.13-slim`), suite `quick`, output `c2-combine/eval-quick`. Wall time 246 s. Exit 0.

**Verdict: PASS WITH WARNINGS. 0 FAIL**, 2 WARN, 37 PASS, 1 INFO, 39 NOT RUN, 9 MANUAL.

Honest 4-fold score (each episode scored with its own answers unmounted): **57.1918 / 60**, R = 0.0491, 95% bootstrap CI [56.653, 57.723].

| stable | short_history | drift | shared_change |
|---:|---:|---:|---:|
| 0.0438 | 0.0848 | 0.0334 | 0.0344 |

About 0.01 under the full-validation 57.20, because each fold's win prior sees three quarters of the validation targets. The size rule and the flip gate still see only train.

Determinism (fold 0 rerun) PASS. Row-order probe PASS. Answers-hidden probe PASS.

The two WARNs are the same event: `CAU-FUTURE-TICKS` exits 1 with `ValidationError: Future data leak`. That is `index_history` refusing ticks at or after the cutoff. The suite records a refusal as WARN. g10's report has the same warning. `CAU-CUTOFF-OTHERS` is not part of the quick suite.

Slowest container run is the kit test at 39.4 s (2.2% of the 1800 s limit). Fold runs are about 24 s. Peak memory about 199 MB on the kit test and about 100 MB on a fold.

## Files

| path | role |
|---|---|
| `entry/`, `entry.zip` | packaged code. `predict.py` refits from `/data`. No fitted atoms in the zip |
| `submission.csv` | test forecast. SHA-256 `06ac7557d9f791fa29ff1e75bb7f7882a250b98217656be504b5c95a55829b95` |
| `mainFiles/` | copy of g10, plus the flip, the patch, and `run_pair_mixer.py` flags |
| `mainFiles/runs/ablation-*` | the four rows above |
| `mainFiles/runs/v7/` | packaged run, reproduction diff 0 |
| `rounds/` | simulated-round scores |
| `eval-quick/` | suite report |

Unit tests: `tests.test_v7_changes` (6) and `tests.test_size_rule` (6) and `tests.test_pair_mixer` (21), all passing.

## Risks

- The size rule is the main-round gain and the round-two loss on these three generators. They keep size static. A later round whose train lags do not separate the way `(1, 2, 4)` did here should turn `USE_SIZE` off and rerun. The switch is one assignment in `entry_config.py`.
- The round-two patch is idle on this train. It is in the entry so an empty pair search cannot fall back to `(1, 3), (4, 7), (2, 5)`, and so a type that fails the 0.5 split does not inherit another type's live level. On `four`, that prior change also flips the cell-mode choice to full, and the parity swap then has no counts to relabel.
- The flip gate is the widest gap in the train log-Bayes-factor list. On this train the gap is 270 nats wide. A round whose log-Bayes-factors form one cloud would cut that cloud at its largest adjacent gap.
- The win prior is still read from validation targets, as in v6. `priors.json` freezes this round's levels for a rerun with no targets file. Size atoms are relearned. A later round with different win levels and no targets file would keep a stale win prior.
- Short history (ratio 0.084) and jackpot (k-fold ratio 0.095) are still the weak slices. The flip correctly leaves short history alone. The size rule does not touch win probability, which is most of the jackpot error.
- Refusing future ticks is why the causality probe warns. Organizer data does not contain those ticks.
