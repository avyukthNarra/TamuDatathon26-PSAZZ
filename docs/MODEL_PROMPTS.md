# Prompts that made the House Edge model better

A running log of the requests (to Claude Code or any coding agent) that led to
real score gains, and the ones that didn't. Use it to steer the next agent fast,
especially in round two. Validation points are out of 60; the pooled baseline is 30.

## What moved the score (in order)

| # | Prompt / instruction that worked | What it led to | Validation |
|---|---|---|---|
| 0 | "Run the starter and explain each score" | Understood that points = 60/(1+R) and 30 = baseline | 30.00 |
| 1 | "Train on every past spin (history.csv), not just the 768 labelled queries, and make sure no input reads the tick being predicted" | Lag-feature logistic model + no-lookahead tests | 31.96 |
| 2 | **"Before changing the model, split the validation error into the part from the win chance (p_0) vs the part from payout size"** | Found ~93% of error was the win chance; true win chance swings 0.04 to 0.99 while ours barely moved | (diagnosis) |
| 3 | **"Read the kit's own docs (EXPERIMENTS.md, transition.py) for hints about how the simulator works"** | EXPERIMENTS.md: "a useful pair of earlier spins may differ between machines" | (diagnosis) |
| 4 | **"For each machine, find which PAIR of lags best predicts its next win; count which pairs come up most"** | Only three pairs matter: (1,3), (4,7), (2,5), ~200 machines each | (diagnosis) |
| 5 | **"The pair's effect changes over time, so test it online: per-machine decayed win table per candidate pair, mixed by recent accuracy (Fixed-Share style)"** | Win-chance error vs outcomes 0.214 -> 0.121 on train | **50.29** |
| 6 | **"Print the 4-cell win rates of each machine's best pair and look for a shape"** | Cells are HLLH or LHHL: the machine wins when the two lags *match* (or *differ*), 96-100% vs 2-4%. Experts switched to 2 cells (match/differ) | **52.92** |
| 7 | **"Train and validation disagree on the best half-life; find out why before trusting either"** | Ticks *inside* the history prefer short memory (6), but the *cutoff tick* prefers longer (10-14), on train labels.csv and validation truth alike. Tune on train labels at the cutoff | **53.35** |
| 8 | "Split the remaining error into win part vs size part per condition" | Now ~50/50. Payout size has exactly two shapes per machine type and the mode isn't predictable from history, so sizes are near their floor | (diagnosis) |
| 9 | **"Build a tuning harness that scores on train labels at the cutoff, per condition, as an estimated change in R with an episode-bootstrap interval; keep a change only if it ALSO improves validation"** | Two memory lengths per pair (half-lives 12 and 40) | 53.56 |
| 10 | **"Re-test ideas you rejected on history ticks, now at the cutoff"** | Payout-size experts keyed on lag 1 and lags (1,2), rejected earlier, actually help at cutoffs | **53.75** |

## Prompt patterns that work

- **Diagnose before you model.** Ask the agent to decompose the error (which
  probability, which condition, which machine type) before proposing fixes. One
  decomposition prompt found the problem that a day of feature tuning would have missed.
- **Make it read the organizers' hints.** "Read every .md and .py in the kit and list
  every sentence that hints at how the data was generated."
- **Ask for the simplest test of a hypothesis first** (counts and tables on train,
  early-vs-late holdout) before building it into the real model.
- **Tune on train outcomes, confirm on validation once.** Brier score on observed
  outcomes ranks models the same way as error vs the true probabilities (in
  expectation), so a grid on train is honest; validation is only the final check.
- **When two honest scores disagree, investigate before choosing.** The train-vs-validation
  disagreement exposed that the query (cutoff) tick behaves differently from history ticks.
- **Score where the test is.** Evaluate at the cutoff tick (train `labels.csv`), not on
  arbitrary history ticks.
- **Package the safe model before improving it.** Main entries can't be replaced after 9 PM.
- **Say what can't change:** standard-library Python, no lookahead, missing tick ≠ 0,
  30-minute runtime, a test that proves no future data leaks in.

## Prompts that did NOT help much

- A 2-state hidden "size mode" tracker: no real gain. (Lag-keyed size tables looked useless
  on history ticks but DID help at cutoffs; see row 10.)
- Fixed-Share switching (alpha > 0) between pair experts: much worse (+0.10 to +0.14 in R).
  Machines don't switch pairs, so any weight left on a confidently wrong expert costs.
- A short extra half-life (8) and "don't decay on missing ticks": better on train, worse on
  validation, so not kept. A win-chance recalibration for streaks: predictions were already
  calibrated by bin and by streak length, so it was skipped.

- "Tune the learning rate / blend weight" on the logistic model: +0.2 to +0.5 points.
- "Add neighbour-machine lag features": neighbour lagged effects were noise.
- Single-lag per-machine logistic regression: worse than a flat rate on later ticks,
  because the real signal is a pair interaction (like XOR), which straight-line models can't see.

## For round two (the relationships change)

The entry already adapts: `predict.py` rediscovers the lag pairs on `/data/train`
and picks match/differ vs 4-cell experts by train cutoff Brier. At 9 PM:

```sh
cd mainFiles
python run_pair_mixer.py --data /path/to/round2/public --out runs/round2
```

It prints the pairs it found, the per-condition validation table, and builds a
verified `runs/round2/entry.zip` plus `submission.csv`. Then prompt:

- "Re-run the pair-count table on round-two train: is there still a clear gap after the
  top pairs? Are the cells still match/differ (HLLH / LHHL)?"
- "Use tune_pair_mixer.py on round-two data to re-check half_life and the size experts;
  keep a change only if train cutoff dR and validation both improve."
- "Split round-two validation error into win vs size part per condition before changing anything."
