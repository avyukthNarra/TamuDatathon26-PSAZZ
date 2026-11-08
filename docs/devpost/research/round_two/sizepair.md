# Size-channel evidence for win-pair identification

Status: complete, 2026-11-08 11:19 CST. Started 11:10 CST; hard stop 11:33 CST.

## Current verdict

**DO NOT SHIP. Keep `adapted_entry_v17g_rules` unchanged.** The size-channel candidate scores **38.588518/40**, versus the reproduced baseline **38.587938/40**: **+0.000580**, far below the required **>+0.1**. Both condition-balanced halves improve, but that is insufficient to pass the pick rule. The candidate remains a local experiment; nothing was pushed or uploaded.

## Scope and safeguards

- Read the supplied SHORT_HISTORY_TASK.md using the requested read-only git show; no git mutations, pushes, or uploads.
- All retained experiment files and copied entries are in this directory; an initial temporary-directory exception is documented below.
- The added size evidence learns only from train and each query's observable history. The baseline's existing crossfit protocol still uses opposite-half validation target priors; this experiment does not add any validation-target fitting.
- Three GPT-6-Luna workers performed the training check, baseline/candidate scoring, and implementation; the parent managed decisions and reviewed results.

## Train check

The corrected diagnostic uses only winning ticks with all three lag contexts observed; candidate matching requires the full pair to be one of `(1,4), (2,6), (3,8)`.

- 768 train machines; 569 have a confident win pair under the existing two-nat / five-ticks-per-cell rule. Of those, 40 vote outside the three supported pairs.
- On the remaining 529 machines, the best size lag matches the pair's smaller member on **409/529 (77.3%)**, versus a 35.0% majority-lag baseline. Nine machines have tied size-lag likelihoods.
- By condition: stable **161/185 (87.0%)**; short_history **84/157 (53.5%)**; drift **90/107 (84.1%)**; shared_change **74/80 (92.5%)**.
- On 199 machines without a confident win pair (35 short), the exact model's static pair-posterior mean top mass changes **0.916 → 0.922**, and entropy **0.192 → 0.177 nats**. The top pair changes on 12 machines. Short-history mean top mass decreases **0.774 → 0.767**; the extra channel does not consistently sharpen that group.

These are train diagnostics using train-fitted atoms and level priors, not held-out accuracy estimates. The static-posterior diagnostic does not replicate episode-level drift/shared-change inference. See [train_check.md](train_check.md) and [train_check.json](train_check.json) for details. Earlier preliminary counts used a less careful missing-history treatment and are superseded.

## Candidate design

Reuse the exact model's existing train-only `fit_size` EM to learn two categorical payout distributions per machine type. For every candidate win pair, map its smaller lag to the size channel. Accumulate likelihood over observed winning ticks, marginalize the machine's two XOR directions with equal prior, and add the resulting evidence to that pair's win likelihood with weight 1. Missing lag observations use an equal mixture of the two size distributions. The size evidence is centered across pairs, and non-parity rules receive zero added evidence. Apply this consistently to static, masked short-history, drift, and shared-change inference. Existing payout-size forecasts remain unchanged.

## Scores

| Entry | Overall /40 | Half A | Half B | stable | short_history | drift | shared_change |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reproduced baseline | 38.587938 | 38.530600 | 38.632794 | 38.813872 | 38.133126 | 38.623540 | 38.789029 |
| Size-channel candidate | 38.588518 | 38.530605 | 38.633977 | 38.813872 | 38.135342 | 38.623592 | 38.789029 |
| Delta | +0.000580 | +0.000005 | +0.001183 | ≈0 | +0.002216 | +0.000051 | ≈0 |

Halves must come from `R2/orch/common.py:folds`.

Each split contains 48 episodes; the full evaluation has 96 episodes / 768 queries. The scorer was checked against `orch/common.py` and matches its formula and fold assignment. Per-condition points above are each condition scored individually on the 40-point scale; the overall score uses the mean of condition error ratios.

The short-history change is **−0.000083 points on half A** and **+0.004525 on half B**. Thus the main target condition does not improve consistently across halves, even though both aggregate halves increase. All full-precision scores, per-condition ratios, per-half condition results, and paired deltas are in [baseline_scores.json](baseline_scores.json) and [candidate_scores.json](candidate_scores.json).

## Verification and reproduction

Both full crossfit predictions completed successfully with 768 rows: baseline in **155.8 s**, candidate in **149.2 s**. The added evidence passed checks for weight-zero baseline equivalence, no-win neutrality, identical-atom neutrality, correct encoded `payout>=2` threshold, and exclusion of the cutoff outcome. Test script: [check_size_pair.py](check_size_pair.py); exact executed command, runtime, and test record: [IMPLEMENTATION.md](IMPLEMENTATION.md). Independent Luna source review found no leakage, threshold mismatch, missed inference path, or unintended change to the existing win-level/schedule fitting.

Model limits: missing size contexts use an equal-mixture approximation; the size direction and categorical distributions are assumed stationary through drift/shared-change phases. The kit's learned family census is parity=1 and extras=0; neutral evidence for dormant non-parity families is a heuristic if those families become active in a different dataset. No held-out train-lag accuracy estimate or hidden-test score is claimed.

The candidate was run through `entry/predict.py` with the requested data, split, and protocol; its output was saved at workspace-root `val.csv`, then copied identically to `entry/val.csv`. Run log: [predict.log](predict.log). Equivalent canonical command, keeping temporary files inside this directory:

```sh
cd research_pf/r2-sizepair/entry
TMPDIR=research_pf/r2-sizepair/tmp PYTHONDONTWRITEBYTECODE=1 python3 -B predict.py --data R2_kit40/public --split validation --protocol crossfit --out val.csv
```

Scoring utility: [score_predictions.py](score_predictions.py), using the condition-balanced split from `orch/common.py:folds`.

```sh
cd research_pf/r2-sizepair
python3 -B score_predictions.py entry/val.csv --public R2_kit40/public --compare baseline_entry/val.csv --out candidate_scores.json
```

## Changed files and diff

Full source-only unified diff: [change.diff](change.diff) (141 lines), against the immutable baseline copy.

- `entry/members/exact_filter.py`: fit categorical size atoms on train and include them in model parameters.
- `entry/vendor/exact/model.py`: calculate free-direction size likelihood for each pair and add it consistently to parity rows in static, short-history masked, drift, and shared-change inference; extra families receive neutral added evidence.

No new dependencies, changes to blending, or changes to the existing conditional payout-size forecast. The original entry was not edited. The changed entry should **not** replace the baseline.

## Execution notes

The baseline runner initially inherited the entry's default temporary directory setting. That attempt was interrupted before producing predictions; it may have created an ephemeral `blend-members-*` directory outside this workspace. The restarted run sets `TMPDIR` inside this workspace, keeps member predictions here, and disables bytecode writes. No source/data outside this workspace was edited.
