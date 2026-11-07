# House Edge: solution playbook

Written 2026-11-07 by Claude. Read `README.md` first. Theory and citations are in `research-literature.md`. Tags: **[Measured]** = computed on `mainFiles/public` main-round data; **[Idea]** = untested.

## 1. Scoreboard (validation split, 96 episodes, 768 queries)

| Model | Points | Notes |
|---|---|---|
| pooled (starter, strength 20) | 30.00 | the frozen reference |
| pooled_recent | about 30.6-31 | single recency half-life |
| GB (HL 4,16,64,128 + LAGRESP) | 32.99 | generic features |
| **pair model prototype** | **46.07** (R = 0.302) | `scripts/pair_model_prototype.py` |

Points = max / (1 + R), with R the normalized squared-error ratio against the reference. Lower R is better.

Prototype error shares by payout class [Measured]: p0 58%, p1 24%, p2 14%, p5 4%, p20 about 0%. The win/no-win split (p0) is where most remaining error lives.

## 2. What the prototype does

1. Per machine, flag lags 1,2,3,4,5,7 as +1 (payout > 0), -1 (zero), 0 (missing, which is unknown and not a loss).
2. Add pair products (1,3), (2,5), (4,7).
3. Fit a ridge regression (lambda 20) of "win" on those features with exponential time weights. The half-life is chosen per machine from {inf,128,64,32,16} by an inner holdout of the last 40 ticks.
4. Clip p_win to [0.005, 0.995]; split it over payout sizes using recency-weighted type-shrunk shares (half-life 64, prior 8).
5. Use only ticks strictly below the cutoff.

Run from `mainFiles/`:
`python3 ../docs/sai/scripts/pair_model_prototype.py --data public --split validation`
(add `--split test --out file.csv` to write predictions; about 2 s).

Caveats: the half-life floor of 16 was picked after looking at validation, and 96 episodes is small, so use `scoring.bootstrap_mean_interval`. Differences under about 0.2 points are noise.

## 3. Background statistics

| Fact | Value |
|---|---|
| Mean irreducible term, 1 - ||q||^2 | 0.378 (outcome Brier only; not in the contest metric) |
| Best plain EWMA error vs true q | about 0.183 |
| Sampling floor at half-life 16 | about 0.008 |
| Share of EWMA error that is structure, not noise | about 95% |
| Recurring single lag | 7 |
| Recurring pairs | (1,3), (2,5), (4,7); weaker (4,5), (2,6) |
| Null-control lags/pairs | 9-16 (excess z^2 about 0.5-3) |
| Memory preference | short_history: none needed; drift and shared_change: half-life about 16 |

Effects are on the win/loss axis, and signs differ across machines, so fit per machine, not per type.

## 4. Ranked experiment plan

1. **Payout-size model.** p1+p2+p5 hold about 42% of the remaining error. Try per-machine recency-weighted shares with partial pooling, or a small regression of size on lag features. [Idea]
2. **Memory selection.** Replace the hard holdout argmin with fixed-share or DMA over the half-life grid; consider sharing changepoints across an episode's machines. [Idea]
3. **Partial pooling of coefficients.** Shrink per-machine ridge coefficients toward the type mean (mean lag-7 sign is positive in every type) instead of toward zero. [Idea]
4. **Stacking.** Blend pair model, GB and pooled by minimizing squared error to q on validation, with episode-grouped CV. A convex blend cannot be worse than the average of its members. [Math]
5. **GB second stage** on the pair-model output plus EWMA features. [Idea]
6. **Recalibration** (temperature or Dirichlet), fit to q with episodes held out. [Idea]
7. **Regime/HMM memory (optional, Rabiner 1989).** Fit a 2-3 state HMM per machine (or pooled by episode for shared_change) with fixed init, forecast via forward pass, and stack it with the pair model; see `research-literature.md` section 3. Only worth a try if the half-life approach plateaus. [Idea]
8. Re-tune ridge lambda, lag set, holdout length and the half-life floor on whatever split you are *not* scoring.

Adopt a change only if its episode-bootstrap interval clearly beats the prototype, and log it in `EXPERIMENT_LOG.md`.

## 5. Round two

Round two changes which relationships recur. Do not hard-code lag 7 or the three pairs.
1. Run `python3 ../docs/sai/scripts/lag_screen.py --data public --splits train validation [--thr 1] [--top 6]` on the new kit. It reports per-machine, per-type excess z^2 for lags 1-16 and pairs, with lags/pairs in 9-16 as null controls and a min-cell-count guard.
2. Take lags and pairs whose excess clearly exceeds the null maximum, and edit `LAGS` and `PAIRS` in the prototype.
3. Re-pick the half-life floor, ridge strength and holdout on the new validation split.
4. Check timeline in the guide: main closes Saturday 9 PM CT; round two Sunday 11 AM CT.

## 6. Pitfalls

- Missing ticks are unknown, not losses.
- Use ticks strictly below the cutoff; never other machines' outcomes at or after it.
- Condition labels are not available at test time, so no per-condition parameters at inference.
- Keep whole episodes together in any CV.
- Screening structure on train-only histories avoids leaking validation.
- Repeated tuning on validation inflates scores; keep an audit split.

## 7. Reproducibility and packaging

- The run must reproduce within 1e-8, finish in 30 min on 2 CPUs / 16 GiB, run offline, and use organizer-released data only.
- The packaging contract is `reproduce.json` plus `predict.py --data --out` with `/data`, `/main_data`, `/output`.
- The prototype uses numpy and pandas only (no scipy or sklearn). If the rules require stdlib-only or offline wheels, port the ridge solve (8x8 normal equations) to pure Python; keep a fixed iteration order for determinism.
- Seed anything random; avoid multithreaded BLAS nondeterminism (set thread counts to 1).

## 8. Discrepancies to confirm with organizers

- Point split: 80/20 versus 60/40 appears in different places.
- Dependencies: stdlib/offline wheels versus a pinned `requirements.txt`.
