# Current result: retain TM_COMBINED4

The orchestrator's update supersedes the v3 experiments. **M1–M3 on v4 add no
convincing improvement.** The best exploratory fit is 55.396854 versus
55.344762 for TM_COMBINED4, difference +.052092 [−.015930,+.173876]. Reselecting
all component parameters and the blend weight on opposite dev folds scores
55.295621, difference −.049141 [−.134683,+.039090]. Keep the existing v4 entry.

All additional population statistics use **train histories only**, and all
new selection/scoring uses **72 dev episodes only**. No audit/all scorer was
called. **The inherited v4 Beta prior AND size modes are fitted on all 96
validation targets, including the reserved audit episodes.** Consequently the
baseline is not independent of those targets, even when its score is dev-only.

Evidence: `pranav_v4.py::win_prior_from_targets` computes median true win chances
below/above .5 from the targets passed to it. The existing
`docs/sai/combos/members.py` v4 runner passes the entire validation targets file.
The resulting fixed prior is (.45,.0435,6), (.45,.898,6), (.10,.5,2). We preserve
it unchanged; no additional M4 port, estimation or tuning was performed.

Run from repo root, using the already installed NumPy/SciPy:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
nice -n 10 python3 -m unittest discover -s docs/sai/improve/model -v
nice -n 10 python3 docs/sai/improve/model/v4_experiments.py
nice -n 10 python3 docs/sai/improve/model/verify_model_v4.py
```

`v4_results.json` records all 93 grid rows, per-condition ratios, 1,000-replicate
bootstrap CIs, paired differences, and opposite-fold selection. `IMP_M0V4` is
a weight-only control; its selected 20/80 weights exactly recover TM_COMBINED4.
Predictions `IMP_M1V4`, `IMP_M2V4`, `IMP_M3V4`, `IMP_M5V4` are saved under
`docs/sai/preds/`. The `V4` suffix distinguishes them from the earlier work.

M1 learns type-specific pair frequencies on train. M2 adds type-level cell
pseudo-observations **inside** the v4 Beta posterior, before both component
likelihoods and posterior means are evaluated. M3 adjusts that type prior with
peer rate residuals; train null checks are in `train_structure.json` and the
v4 result. Short-history shrinkage and positive peer pooling hurt on v4.

The reusable `ResearchPairMixerV4` in `improved_mixer_v4.py` preserves v4 with
`win_options={}`. Passing a result row's `params` enables that research variant.
Its `fit` method uses train histories. Callers supply the inherited size modes
and Beta prior explicitly. Rule blending remains external.

Verification compares the adapter to saved M5V4 predictions, checks two-run
exact determinism, and validates IDs/probabilities in all five 768-row CSVs.
The raw independent size head is unchanged between v3/v4; the runner reuses
`v3_raw.npy` and asserts exact reproduction of the saved v4 baseline before
searching. This cache contains forecasts, not truth; the baseline size-mode
parameters used to produce it were validation-derived. Retain it for reruns.

Earlier `results.json`, `experiments.py`, `improved_mixer.py`, and `IMP_M1`–`M5`
refer to v3 and are superseded. Their report and verification are preserved in
`v3_archive/`. No existing teammate source or Git state was modified. Packaging
and stdlib conversion remain outside this research task.
