# v7: pair mixer plus a train-learned size-atom rule

Submission candidate. v6's win probability is unchanged. Given a win, the payout mix is a posterior over two atoms and a lag, both learned from `/data/train` at run time. Nothing from the main round is hard-coded: no atom vectors and no lag set `{1, 2, 4}`.

Full validation is **57.02 / 60** (R = 0.0523). The same train EM that h2 scored at 57.0203 reproduces that figure here to the printed digits. A fresh-process rerun matches the submission with max abs difference 0 (limit 1e-8). Wall time for train, validation, test, cross-fit, packaging, and the rerun is **106 s**.

## What the module does

`slot_forecasting/size_rule.py` (standard library only):

1. EM, six fixed seeds `0..5`, fits two payout atoms per machine type on training wins. A shared lag prior is fit at the same time. Atom 0 is the one with the higher chance of paying 1. Best train log-likelihood is −58619.849. The other five seeds land within 0.005 nats, so the fit is not a lucky restart.
2. With those atoms and a uniform prior, each training machine gets a posterior over lags 1..8. A lag is kept when its share of that mass is above the uniform share `1/8`. If none qualify, every lag 1..8 is kept.
3. At a query, the machine's own wins before the cutoff update a uniform posterior over the kept lags and a polarity bit. The feature is `1{payout at t−L ≥ 2}`. A missing lag tick is unknown and is skipped, not treated as a loss. The forecast size mix is the posterior mixture of the two atoms.
4. `p_0` stays the pair mixer's. The five probabilities are floored at 1e-4 and renormalized.

On this train the mass shares are 0.224, 0.209, 0.205 on lags 1, 2, 4 and 0.069–0.076 on the other five. The uniform share is 0.125, so the rule keeps `(1, 2, 4)`. The gap under the cut is about 0.13, not a near-tie.

Learned atoms (pay 1, 2, 5, 20), which match h2's train-EM atoms:

| type | atom | p(1) | p(2) | p(5) | p(20) |
|---|---:|---:|---:|---:|---:|
| steady | 0 | 0.7478 | 0.1994 | 0.0516 | 0.0011 |
| steady | 1 | 0.4623 | 0.4249 | 0.1106 | 0.0022 |
| balanced | 0 | 0.4943 | 0.3507 | 0.1333 | 0.0216 |
| balanced | 1 | 0.2075 | 0.5461 | 0.2095 | 0.0370 |
| jackpot | 0 | 0.3171 | 0.2591 | 0.3117 | 0.1121 |
| jackpot | 1 | 0.1201 | 0.3290 | 0.4080 | 0.1429 |

## Files changed

Copied `worktrees/pranav-5629/mainFiles` to `g10-build/mainFiles`, then:

| file | change |
|---|---|
| `slot_forecasting/size_rule.py` | new, 264 lines. EM, lag selection, posterior, `apply` |
| `tests/test_size_rule.py` | new, 114 lines. Six tests, all passing |
| `run_pair_mixer.py` | +38 / −14. Learns the rule on train, replaces the size split on validation, test, and both cross-fit halves, writes `runs/v7-size/size_rule.json`, packages `size_rule.py` |
| `entry_template/predict.py` | +13 / −8. Same learn-and-apply path, from `/data/train`, whenever the entry is rerun |

`runs/v7-size/entry/` and `entry.zip` contain `slot_forecasting/size_rule.py` and a `predict.py` that imports it. No fitted atoms are stored in the archive. `priors.json` is still only the v6 win-chance fallback for when validation targets are absent.

## Scores

`python run_pair_mixer.py --out runs/v7-size --cv-val` from `g10-build/mainFiles`.

| | points | stable | short_history | drift | shared_change |
|---|---:|---:|---:|---:|---:|
| v6 (brief) | 56.29 | 0.0541 | 0.1047 | 0.0488 | 0.0564 |
| v7, full validation | **57.0203** | 0.0438 | 0.0844 | 0.0382 | 0.0426 |
| cross-fit, tuning with audit priors | 57.0265 | 0.0600 | 0.0725 | 0.0407 | 0.0354 |
| cross-fit, audit with tuning priors | 56.9722 | 0.0265 | 0.0996 | 0.0357 | 0.0508 |

Raw errors on the full set: stable 0.00860, short_history 0.01496, drift 0.00725, shared_change 0.00876. R = 0.052256.

The cross-fit refits only the win-chance prior, on the opposite episode half. The size atoms and lags stay the train-only fit. Both halves stay within 0.05 points of 57.02.

Reproduction check: fresh `python predict.py` against `submission.csv`, max probability difference **0** (limit 1e-8). SHA-256 of `submission.csv` is `085b324d4e23dcfde212faaaf12cbd93e55de877b0e47be16709410cded39358`.

## Unit tests

`python -m unittest tests.test_size_rule -v` — 6 tests, 0.065 s, OK.

- Lag selection keeps mass above the uniform share, and a flat mass falls back to all eight lags.
- A missing lag tick is not a zero payout. Ticks at or after the cutoff do not change the posterior.
- On a synthetic two-atom process whose machines use lags in {1, 2, 4}, EM recovers both atoms within 0.05 and the lag set `(1, 2, 4)`. Two learns are identical, including the log-likelihood.
- `apply` floors a tiny `p_20` and renormalizes. Classes that were already above the floor keep their ratios.

## Eval suite

From `worktrees/eval-main/mainFiles`, Docker mode (`python:3.13-slim`), suite `quick`, output `g10-build/eval-quick`. Suite wall time 221 s. Exit 0.

**Verdict: PASS WITH WARNINGS. 0 FAIL**, 2 WARN, 37 PASS, 1 INFO, 39 NOT RUN, 9 MANUAL.

Honest 4-fold score (each episode scored with its own answers unmounted): **57.0002 / 60**, R = 0.0526, 95% bootstrap CI [56.460, 57.530].

| stable | short_history | drift | shared_change |
|---:|---:|---:|---:|
| 0.0438 | 0.0848 | 0.0389 | 0.0431 |

About 0.02 points under the full-validation 57.02, because each fold's win prior sees three quarters of the validation targets. The size rule still sees only train.

Determinism (fold 0 rerun) PASS. Row-order probe PASS. Answers-hidden probe PASS.

The two WARNs are the same event: `CAU-FUTURE-TICKS` exits 1 with `ValidationError: Future data leak`. That is v6's `index_history` refusing ticks at or after the cutoff. The suite records a refusal as WARN, not FAIL. The published team-best report has the same warning. `CAU-CUTOFF-OTHERS` is not part of the quick suite.

Slowest container run is the kit test at 33.0 s (1.8% of the 1800 s limit). Fold runs are about 21 s. Peak memory about 211 MB on the kit test and about 100 MB on a fold.

## Risks

- The lag cut is "above a uniform share," not a threshold fit on validation. On this train it is comfortable. A round whose posterior is flat keeps all eight lags. h2 scored that wider set at 56.72, still above v6 and below 57.02.
- The win prior is still read from validation targets, as in v6. `priors.json` freezes this round's win levels for a rerun that has no targets file. Size atoms are not in that file; they are relearned. A later round with different win levels and no targets file would keep a stale win prior and a fresh size rule.
- Short history (ratio 0.084) and jackpot (k-fold ratio 0.103) are still the weak slices. The size rule does not touch win probability, which is most of the jackpot error.
- Refusing future ticks is safe on organizer data, which does not contain them, and it is why the causality probe warns instead of passing.
