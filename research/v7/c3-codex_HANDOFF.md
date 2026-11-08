# c3 v7c handoff — 2026-11-08

The verified guarded v7c main submission is staged at [`v7c/submission.csv`](v7c/submission.csv),
with its packaged entry at [`v7c/entry.zip`](v7c/entry.zip) and unpacked files
at [`v7c/entry/`](v7c/entry/). Source and per-kit score outputs remain under
`v7c-mainFiles/`. The previous v7b report, review, implementation, evaluation,
and rehearsal evidence are preserved under [`archive/v7b-round2/`](archive/v7b-round2/).

The main local run scored **57.369244 / 60**; cross-fit halves scored
57.308288 and 57.391178. The candidate size head passed its train-only
whole-episode gate on main (conditional Brier 0.574505 vs v6 0.583958; paired
delta −0.008714, SE 0.000428, upper bound −0.007875; 96 episodes, 58,639 wins).
The fitted static size prior was 0.192708, with learned lags 1, 2, and 4. The
test-format check passed for 1536 main queries. Fresh-process reproduction was
exact (max prediction difference 0), and all **50 tests passed**.

Round-two raw episode-weighted MSEs beat frozen v6 references on all three
synthetic kits:

| Kit | v6 raw MSE | guarded v7c MSE | Difference | v7c contest score |
| --- | ---: | ---: | ---: | ---: |
| disjoint | 0.01561816 | 0.01199648 | −0.00362168 | 37.72 / 40 |
| four | 0.02069269 | 0.01408449 | −0.00660819 | 36.45 / 40 |
| sharedlag | 0.03079214 | 0.01205055 | −0.01874160 | 37.63 / 40 |

The static-size fixtures rejected the new size head and retained the v6 size
head. An initial four-kit raw-MSE regression (+0.00033871) led to a
train-only, whole-episode guard for full payout-win cells. It uses fixed
minimum support 8 and the rule mean paired full-minus-parity Brier plus 1.96
SE < 0. On four it rejected full (mean +0.002817, SE 0.006580, upper bound
+0.015713), selecting parity; guarded score then beat the frozen v6 reference.
The initial validation residual prompted checking the interaction, but the
guard's criterion and per-kit decisions used TRAIN data only.

The Docker quick evaluation in [`eval-v7c-quick/report.md`](eval-v7c-quick/report.md)
completed in 348.9 seconds: **PASS WITH WARNINGS**, 37 PASS, 0 FAIL, 2 WARN
requirement rows, 1 INFO, 39 NOT RUN, 9 MANUAL. Its four-fold estimate was
57.356462 / 60 (95% bootstrap interval 56.78317–57.87727; ratios stable
0.041857, short history 0.075271, drift 0.032728, shared change 0.034503).
The kit test took 51.09 seconds and peaked at 211.9 MB; folds took about 35
seconds each. Determinism, hidden-answer, and row-order probes passed. WARN
rows RUL-18 and GDE-02 refer to the same future-tick mutation refused by the
schema with `ValidationError`; this is a deliberate schema refusal, not a
separate prediction-leak event. GDE-02 also records CAU-CUTOFF-OTHERS as NOT
RUN, not as a second warning. Quick did not run the full suite or the synthetic
`round2_shift` probe. The evaluation caveat says packaged priors may encode
validation information; their provenance is not independently verified.

All three sequential synthetic [`round2.sh`](round2.sh) wrapper rehearsals
passed: disjoint 29s, four 52s, sharedlag 38s. Each candidate build and
64-query format check passed, reproduction max difference was 0, and no
fallback was used. All used wrapper SHA
`5e33e339ad7d7e92598c4b4fd1b173dd254403a91fc54f11a4ea8199dfda7461` and
allowlisted source SHA
`dabbd99efd71a7085ee95b3b88502beb3a6202d96ac45ae75419328245d62bb4`.
Archived artifacts and details are in [`REHEARSALS-v7c.json`](REHEARSALS-v7c.json)
and `archive/v7c-rehearsals/`. No live `SUBMIT_R2/` directory remains; these
were synthetic simulator rehearsals, not a real kit submission. Current
operator instructions are in [`RUNBOOK.md`](RUNBOOK.md).

The canonical artifact hashes match the guarded source outputs:

- `submission.csv`: `8d6e00557ff56bc7dbb380f35f981ca84d67d4f6c5721d5d978cf792eaecfb1f`
- `entry.zip`: `73041aa7793271ab141fae5647c9cd859ffa92c6a62952579358e113de62ee08`

Detailed score and confidence-gate evidence is in [`REPORT.md`](REPORT.md),
[`ROUND2_BASELINES.md`](ROUND2_BASELINES.md), [`V7C_TRAIN_GATE.json`](V7C_TRAIN_GATE.json),
and [`V7C_CELL_MODE_TRAIN.json`](V7C_CELL_MODE_TRAIN.json). The review is
[`V7C_REVIEW.md`](V7C_REVIEW.md). These results establish the recorded local
checks and synthetic-kit rehearsals; they do not establish organizer receipt
or a score on an unseen real adaptation kit.
