# s2 blend entry

One runnable entry for the equal log-odds blend of v13+short, the exact filter, and sysid. Size given a win is v13's. Each probability is floored at 1e-4 and the row is renormalized.

## Structure

`entry/` is the ZIP root (`entry.zip`, 69 files, README and `reproduce.json` at the top).

```
python predict.py --data /data --out /output/submission.csv
```

`predict.py` starts the three members as separate processes (so their packages do not collide) and blends them in `blend.py`.

| Member | Script | What it fits |
|---|---|---|
| v13 + short-history patch | `members/v13_short.py` | v13 from `a70abc0` (`origin/pranav/strongest-build`). Short histories use `grid_marginal` from `short_win.py`. |
| exact filter | `members/exact_filter.py` | Train level grid and change schedule. Level prior from validation targets when that file exists. |
| sysid | `members/sysid_rule.py` | Train-only BIC and reversal likelihood. Selected `strongest_peer/dyn01`, the published train-only choice. |

`members/hooks.py` is the swap point. Edit `REGISTRY` to replace a script. Set `PAIRS_OVERRIDE` to a list of `(lag_a, lag_b)` and the exact filter and the short-history patch will use it. Sysid ignores that override because it learns its rules from train; replace `members/sysid_rule.py` to drop in another library.

Validation targets are priors only. v13 reads size modes and the win prior. The exact filter reads the logit level prior (same two-means fit as `wt/orch/priors/levels_from_truth.py`, rounded to 3 decimals). Sysid never reads them. If `/data/validation` or `targets.csv` is missing, v13 uses train-only frequencies and the exact filter keeps the train level prior. That run completed and `check_submission.py --split test` accepted it.

Short histories are the `short_history` condition when `conditions.csv` is present. Otherwise a machine is short when fewer than 75% of its ticks were observed. That rule matched all 768 validation queries.

Pinned packages: `numpy==2.3.4`, `scipy==1.16.3`. The members are numerical code. The contest base image is the standard library; pinned requirements are how those packages are installed before the offline run.

## Scores

Cross-fitted validation, produced by this code (`python entry/predict.py --protocol crossfit`), scored with the official ratio formula. Halves are episodes sorted by id within condition, alternating.

| Forecast | Points | Halves A / B |
|---|---:|---|
| Blend, before the 1e-4 floor | 57.909 | 57.98 / 57.84 |
| Blend, after the floor | 57.909 | 57.98 / 57.84 |

Ratios: stable 0.0285, short_history 0.0601, drift 0.0250, shared_change 0.0307.

Max absolute difference in P(win) versus the saved research CSVs: v13+short 6e-16, exact 2e-11, sysid 2e-15.

`submission.csv` is the test forecast (1536 rows) with validation targets used as priors. `check_submission.py --split test` reported `VALID FORMAT`. SHA-256 `1a4bf9ae08a17deb244fbbac7b462a00de2d8132f479e8f2069e1efb5d07e8bb`.

A second process, started from the unzipped `entry.zip`, matched `submission.csv` with max absolute difference 0.

## Runtime

Test submission wall time: **463 seconds** on this 11-core machine. The three members run in parallel with one BLAS thread. v13 and the exact filter both ran the whole time; sysid finished first. Two CPUs are enough to keep those two busy, so the same run stays well under 30 minutes. Docker kit-test (no validation targets) was 160 seconds. Each validation fold was about 145 seconds.

The no-validation run finished in 407 seconds and wrote 1536 valid rows.

## Eval

`python3 -m eval_suite run --entry entry --suite quick --jobs 4` (docker).

**Verdict: PASS WITH WARNINGS.** 0 requirement failures, 2 warnings, 37 passes.

**4-fold score: 57.9044 / 60** (95% CI 57.555–58.246). Ratios: stable 0.0286, short 0.0601, drift 0.0251, shared 0.0309.

Determinism passed. The two warnings are the future-tick probe: v13's `index_history` rejects a tick at or after the cutoff and the member exits 1. The suite records that as a refused variant (WARN), not a score failure.

## Risks

- The exact filter and the short-history patch still use pairs (1,3), (2,5), (4,7) unless `PAIRS_OVERRIDE` is set. A shifted round needs the s1-adapt pairs dropped into that hook.
- Sysid contributes only P(win). Its size model is not fit. The blend's sizes are v13's.
- Without validation targets the priors fall back to train. That forecast is valid but is not the 57.91 model.
- Jackpot remains the weak slice on the 4-fold score (loss ratio 0.0814 versus 0.0190 for steady).
- A history row at the cutoff makes v13 abort. Legal kit files do not contain those rows. The future-tick probe does, which is why that check is a warning.
