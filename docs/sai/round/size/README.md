# R5 exact size switch: negative result

R5_BASE is a newly fitted 20% rule + 80% mixer5 (27d776d) forecast. It uses
`win_prior_by_type_from_targets`, `size_modes_from_targets`, and `adapt` from the
copied source; `fit_rules` is the original combo implementation. Win/size priors
retain the requested all-validation-target dependency. Lag/cell structure and
the missing-trigger fallback population rate are fitted on train histories.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
nice -n 10 python3 docs/sai/round/size/baseline.py
nice -n 10 python3 docs/sai/round/size/run.py
nice -n 10 python3 -m unittest discover -s docs/sai/round/size -v
nice -n 10 python3 docs/sai/round/size/verify.py
```

All scoring and parameter selection use the 72 dev episodes only. No audit/all
scoring or validation-target fitting of the new switch is performed.

There are nine hypotheses per machine: lag1+/− through lag4+/− plus a static
50/50 mode. A big win is a 5×/20× payout. Positive polarity selects the big-size
mode following a big win; negative polarity reverses that mapping. The two
modes are ordered by expected payout size. Hypothesis priors are .11 each for
the eight switches and .12 for the static expert.

Only observed positive payout sizes add likelihood. Calendar-time discounted
log evidence uses half-lives 12, 40, infinity; each forecast is computed before
absorbing that tick. The fixed hypothesis prior is retained under discounting.
A missing trigger uses the machine's running observed big-payout probability,
equivalently P(win)×P(big|win), initialized with two train-population
pseudo-observations. Missing observations remain unknown, including long gaps.

The grid tests each memory, equal mixtures 12/40, 40/infinity, and all three,
then either full replacement or a 50/50 average with mixer5's conditional size
shares. The saved mixer win probability is preserved, followed by the fixed
20% rule blend. Both opposite-half dev selections choose the baseline.

`results.json` contains every score, 95% episode-bootstrap interval, paired
comparison, four condition ratios, and short-history size-only errors.
Size-only SSE replaces the forecast win probability with the true win rate
while retaining forecast conditional size shares; it isolates size error.
`exact_sizes.npz` contains switch forecasts and hypothesis posteriors.

`test_exact_size.py` checks the recurrence against a direct discounted
likelihood calculation with missing ticks and different cutoffs, recovers a
known simulated switch, and verifies prediction independence from targets.
`verification.json` records deterministic reruns, strict probability/ID checks,
and unchanged p0 for all CSVs. The recommendation is to keep R5_BASE.

The canonical comparison file is `docs/sai/round/mixer5/COMBINED_v5_insample_validation.csv`. The independently fitted R5_BASE matches within 2.22e−16; `run.py` checks this before using that file directly for every paired comparison.
