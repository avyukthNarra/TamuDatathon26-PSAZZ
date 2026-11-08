# v7c final report

The guarded v7c entry is packaged and format-checked. Docker quick evaluation
and all three synthetic wrapper rehearsals are complete. The previous v7b
report and evidence are preserved under [`archive/v7b-round2/`](archive/v7b-round2/).

## Main result

The main run kept the requested size, size gate, flip, and round-two flags on;
train-only gates decide which size and win-cell heads are effective. It scored
**57.369244 / 60** (mean loss ratio
0.04585655). Condition ratios were stable **0.041897**, short history
**0.074864**, drift **0.032360**, and shared change **0.034305**. Opposite-half
cross-fit scores were **57.308288** and **57.391178**. The fitted static
size-state prior was **0.192708**; the selected size lags were 1, 2, and 4.
The train-only size gate accepted the candidate: conditional Brier **0.574505**
versus **0.583958** for the v6 size head, mean paired delta **−0.008714**,
episode SE **0.000428**, upper 95% bound **−0.007875** (96 episodes, 58,639
wins). The fixed rule requires at least eight episodes and mean delta plus
1.96 SE below zero.

The size model represents a payout distribution as either a static atom
`(None, s)` or a lag-conditioned dynamic state `(L, s)`. A joint train-only EM
fit learns both normalized four-class atoms together, which lets static rounds
retain distinct atoms without forcing a lag rule. The dynamic prior
`P(L, s | pair)` is learned from train histories with fixed `lambda=1`. The
reported static prior is the fraction of training machines whose normalized
static marginal evidence exceeds their normalized dynamic marginal evidence, clipped to
`[0.01, 0.99]`. Both confidence gates compare causal predictions on held-out
whole TRAIN episodes; each forecast precedes its held-out payout.

The verified canonical [submission CSV](v7c/submission.csv), [entry ZIP](v7c/entry.zip),
and [entry folder](v7c/entry/) are copied from the guarded main run. The CSV
SHA-256 is `8d6e00557ff56bc7dbb380f35f981ca84d67d4f6c5721d5d978cf792eaecfb1f`,
matching the guarded source output. The entry ZIP SHA-256 is
`73041aa7793271ab141fae5647c9cd859ffa92c6a62952579358e113de62ee08`. The
fresh-process reproduction matched exactly (maximum forecast difference 0),
and the copied submission passed `check_submission.py`: **1536 queries | main
| test**. The complete 50-test suite passed.

## Docker quick evaluation

The quick suite completed in **348.9 seconds** with **PASS WITH WARNINGS**:
37 PASS, 0 FAIL, 2 WARN requirement rows, 1 INFO, 39 NOT RUN, and 9 MANUAL.
The four-fold estimate was **57.356462 / 60**, 95% bootstrap CI
**56.78317–57.87727**. This is a separate four-fold estimate from the local
all-validation score above. Docker condition ratios were stable **0.041857**,
short history **0.075271**, drift **0.032728**, and shared change **0.034503**.
The kit test took **51.09 seconds** and peaked at **211.9 MB**; fold containers
took about 35 seconds each. Determinism, hidden-answer, and row-order probes
passed.

The two WARN rows, `RUL-18` and `GDE-02`, refer to the same
`CAU-FUTURE-TICKS` mutation: the entry refused a deliberately future-dated row
with a `ValidationError`. This is a schema refusal, not a second independent
warning event. `CAU-CUTOFF-OTHERS` was not run in quick; `GDE-02` records that
as NOT RUN. Quick also did not test the synthetic `round2_shift` generator.
There is no full-suite claim. See [`eval-v7c-quick/report.md`](eval-v7c-quick/report.md)
and [`report.json`](eval-v7c-quick/report.json) for coverage, results, and
caveats. The report notes that packaged priors may encode validation
information; this evaluation does not verify their provenance. It is an
independent harness result, distinct from the local full-validation score.

## Round-two acceptance

The table compares official episode-weighted probability MSEs against exact
frozen v6 references. These MSEs are distinct from the fixture's 40-point
technical score.

| Kit | v6 raw MSE | Guarded v7c raw MSE (95% episode-bootstrap CI) | Change | Technical score |
| --- | ---: | ---: | ---: | ---: |
| disjoint | 0.01561816 | 0.01199648 [0.00970940, 0.01434125] | −0.00362168 | 37.72 / 40 |
| four | 0.02069269 | 0.01408449 [0.00937866, 0.01930058] | −0.00660819 | 36.45 / 40 |
| sharedlag | 0.03079214 | 0.01205055 [0.00902778, 0.01503756] | −0.01874160 | 37.63 / 40 |

The size gate rejected the new size rule on all three synthetic fixtures and
kept the v6 size head. Its train-only candidate-minus-v6 Brier deltas were
disjoint **+0.000999** (upper bound +0.002658), four **+0.000748** (upper
bound +0.002149), and sharedlag **−0.001611** (upper +0.000142). This is consistent
with these fixtures having static payout-size states and does not claim
size-lag recovery.

## Full-cell correction

The initial four score was raw MSE **0.02103140**, 0.00033871 above v6. The
existing cutoff chooser selected full cells there, although the independent
whole-episode TRAIN cross-fit had not shown a reliable full-cell benefit. A
precommitted train-only guard was added: when round2 is enabled and the
cutoff chooser selects full, full mode is retained only with at least eight
held-out TRAIN episodes and mean paired full-minus-parity Brier plus 1.96 SE
below zero. The four-kit guard rejected full (24 episodes, 192 forecasts;
mean delta **+0.002817**, SE **0.006580**, upper bound **+0.015713**) and used
parity. Its guarded raw MSE fell to **0.01408449**, below the frozen v6
reference. The initial four validation regression motivated checking this
interaction. The guard criterion, minimum support, and 1.96 z value were fixed
before guarded rescoring. Each per-kit guard decision uses only that kit's
TRAIN queries, histories, and labels, never its validation targets.

The same train-only gate architecture was also used for payout-size selection.
The main size rule passed the gate; all three static-size kits retained the v6
size head. The detailed fold results are preserved in
[`V7C_TRAIN_GATE.json`](V7C_TRAIN_GATE.json),
[`V7C_CELL_MODE_TRAIN.json`](V7C_CELL_MODE_TRAIN.json), and the actual final
four-kit [`cell_mode_gate.json`](v7c-mainFiles/runs/v7c-four-guarded/cell_mode_gate.json).
The frozen raw-MSE baselines and dataset/result hashes are in
[`ROUND2_BASELINES.md`](ROUND2_BASELINES.md). Source review is in
[`V7C_REVIEW.md`](V7C_REVIEW.md).

## Wrapper rehearsals

The three sequential `round2.sh` rehearsals passed in **29s** (disjoint),
**52s** (four), and **38s** (sharedlag). All used the same wrapper SHA
`5e33e339ad7d7e92598c4b4fd1b173dd254403a91fc54f11a4ea8199dfda7461` and the
same allowlisted source SHA
`dabbd99efd71a7085ee95b3b88502beb3a6202d96ac45ae75419328245d62bb4`; all
candidate builds passed, used no fallback, reproduced with maximum difference
0, and passed the 64-query test-format checker. Their CSV/ZIP pairs are
archived under [`archive/v7c-rehearsals/`](archive/v7c-rehearsals/), with full
details in [`REHEARSALS-v7c.json`](REHEARSALS-v7c.json). No live `SUBMIT_R2/`
directory remains; these are synthetic simulator rehearsals, not a real kit
submission. The current adaptation instructions are in [`RUNBOOK.md`](RUNBOOK.md)
and wrapper code in [`round2.sh`](round2.sh).

The Docker quick report is available at [`eval-v7c-quick/report.md`](eval-v7c-quick/report.md),
with machine-readable results at [`eval-v7c-quick/report.json`](eval-v7c-quick/report.json).
