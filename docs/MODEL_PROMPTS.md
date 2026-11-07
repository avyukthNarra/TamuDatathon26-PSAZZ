# Prompts that made the House Edge model better

A running log of the requests (to Claude Code or any coding agent) that led to
real score gains, and the ones that didn't. Use it to steer the next agent fast,
especially in round two. Validation points are out of 60; the pooled baseline is 30.

## Paste-ready prompt (rebuilds the 54.21 model from the starter kit)

Copy everything inside the box into a fresh coding agent opened in the repo root.
It describes the finished model exactly, so the agent doesn't have to rediscover it.

````text
You are working in the House Edge (TAMU Datathon 2026) starter kit. The kit is in
mainFiles/: data in mainFiles/public/{train,validation,test}, helpers in
mainFiles/slot_forecasting/ (schema.py, io.py, contest.py, rounds.py, starter.py,
submission_contract.py), and the organizers' baseline in run_starter.py (it always
scores ~30/60 because "pooled" IS the reference model; points = 60/(1+R)).
Read AGENTS.md first and follow it: work on a feature branch, never commit to main.

GOAL: build a forecaster that scores ~54.21/60 on validation, plus a verified entry ZIP.

HARD CONSTRAINTS
- Python standard library only. Deterministic, no randomness.
- No lookahead: the forecast for tick t may only use ticks < t. A missing tick is
  "unknown" (None), never a loss.
- The entry runs as: python predict.py --data /data --out /output/submission.csv
  (offline, 30 min, 2 CPUs). /data has train/, validation/, test/.

WHAT THE DATA ACTUALLY DOES (already established; don't re-research)
- ~93% of the starter's error is the WIN chance p(payout>0), not the payout size.
- Each machine's next win is decided by whether its own results at TWO earlier lags
  MATCH or DIFFER (XNOR/XOR). On main-round train only three pairs matter:
  (1,3), (4,7), (2,5), each used by ~1/3 of machines. Cell win rates are 96-100% vs 2-4%.
- Neighbour machines carry no lagged signal. "shared_change" = episode-wide level shifts.
- Payout size given a win: two shapes per machine type; the exact size of the payouts
  1 and 2 spins back helps (win/loss alone at those lags does not carry it).
- Tune at the QUERY CUTOFF tick (train/labels.csv), NOT on ticks inside history.csv;
  history ticks rank settings differently. Confirm on validation once.

BUILD mainFiles/slot_forecasting/pair_mixer.py with:
1. class PairMixer, one online "walk" per (episode, machine) over own[t] = payout class
   index (0 = loss) or None. At each t: EMIT the forecast, THEN absorb tick t.
   - Win experts: one per (pair, half_life) for half_life in (12, 40). Each keeps
     exponentially decayed [wins, n] per cell. Default cell_mode "parity" = 2 cells
     (lags agree / disagree); also support "full" = 4 cells (won/lost at a x at b).
     Expert prediction = (wins + k_win*base) / (n + k_win), k_win = 0.25, where
     base = (decayed machine wins + 1) / (decayed n + 2) with decay half-life = the first
     half_life. If a lag is missing, the expert predicts base.
   - Mix win experts with exponential weights: log_w -= eta*(p - won)^2, eta = 16,
     uniform prior, NO Fixed-Share (alpha = 0; it hurt by +0.10-0.14 R).
   - Size given win (4 classes): experts keyed on () machine-level, (1,), (1,2), (2,)
     and (2,3). A size cell is the EXACT earlier payout class at those lags (0/1x/2x/5x/20x),
     not just won/lost. Each keeps decayed counts per cell (size half-life 80), smoothed toward the
     machine's own size mix with k_size = 16; the machine mix is smoothed toward a
     per-machine-type prior with k_machine = 32. Mix size experts with
     log_w -= size_eta * squared error (size_eta = 8), only on winning ticks.
   - Decay all tables every tick, including missing ticks (decay_missing = True).
   - Output [1 - pw] + [pw * s_k], floor each prob at 1e-4, renormalize.
2. fit(): per machine_type size prior = counts of each winning payout size over train
   history (+0.5 smoothing), normalized.
3. discover_pairs(): for each machine, over all lag pairs from 1..8, pick the pair whose
   4 cells best separate its win rate (weighted between-cell variance, need >= 20 obs);
   count winners. choose_pairs(): keep pairs with count >= 0.3 x the top count
   (fallback (1,3),(4,7),(2,5)).
4. adapt(train_queries, train_history, labels): pairs = choose_pairs(train); build
   "parity" and "full" models; pick the one with the lower 5-class Brier vs
   train/labels.csv at the cutoff. Print the pairs and both Briers.

ALSO CREATE
- mainFiles/entry_template/predict.py: loads train, calls adapt(), predicts --split
  (default test), writes the CSV. Insert its own folder on sys.path first.
  entry_template/reproduce.json:
  {"image": "python", "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"]}
- mainFiles/run_pair_mixer.py --out runs/pair-mixer: adapt on train, write
  validation_predictions.csv and submission.csv, score validation with
  contest.score_contest_files (print per-condition table, R and points), copy
  __init__.py, schema.py, io.py, pair_mixer.py + predict.py + reproduce.json + a README
  into entry/, zip it with submission_contract.package_code, then RE-RUN entry/predict.py
  in a fresh process and fail if any probability differs by > 1e-8.
- mainFiles/tests/test_pair_mixer.py (unittest): (a) mutating or deleting any rows at
  tick >= t never changes predictions at t; (b) walk emits before absorbing;
  (c) on synthetic data whose wins follow lags (2,6) with XNOR, adapt() finds (2,6)
  and predicts confidently.
- Make sure mainFiles/runs/ is git-ignored.

VERIFY (and report the actual output)
  cd mainFiles && python -m unittest discover -s tests
  python run_pair_mixer.py --out runs/pair-mixer
  python check_submission.py runs/pair-mixer/submission.csv
  python check_submission.py runs/pair-mixer/validation_predictions.csv --split validation
Expect validation ~54.21/60 (R ~0.107: stable ~0.083, short_history ~0.151,
drift ~0.093, shared_change ~0.100). The reference build's test submission SHA-256 is
068259d6be6d47b05c10eebdfebc7ecec1291fedb0615239e5ada54218b37b90. A rebuild will
only match it byte for byte if the arithmetic order is identical, so treat the score
as the check. Upload runs/pair-mixer/submission.csv and runs/pair-mixer/entry.zip.
````

The shortcut, if the branch `COMBINED_SOLUTIONS` is available, is to check it out and
run the VERIFY commands. The prompt above is for rebuilding without it.

## What changed on `COMBINED_SOLUTIONS` (53.75 -> 54.21)

Source: another team's write-up of their "rule model v4" (52.89 / 60). It scored worse than ours
overall but better on **stable** (0.068 vs our 0.096). Every idea it had that our model didn't
was scored with `tune_pair_mixer.py --val`. An idea was kept only if it improved BOTH
train labels at the cutoff AND validation.

**Kept: payout-size experts read the EXACT earlier payout, not just win/loss.**
- Before: size experts keyed on `()`, `(1,)`, `(1,2)`, each cell = won/lost at those lags.
- After: size experts keyed on `()`, `(1,)`, `(1,2)`, `(2,)`, `(2,3)`, each cell = the payout
  class (0 / 1x / 2x / 5x / 20x) at those lags. New `PairMixer(size_cell=...)` option:
  `"win"`, `"big"` (lost / 1-2x / 5-20x) or `"class"` (the default), or one per size key.
- Why it works: once a machine wins, the SIZE of the win tends to follow the sizes of its
  last one or two wins. Won/lost cells throw that information away.

| Condition | Before | After | Their model |
|---|---|---|---|
| stable | 0.096 | **0.083** | 0.068 |
| short_history | 0.151 | 0.151 | 0.208 |
| drift | 0.104 | **0.093** | 0.130 |
| shared_change | 0.114 | **0.100** | 0.132 |
| **Points** | 53.75 | **54.21** | 52.89 |

The steps that got there (train cutoff dR, negative = better; then validation):

| Setting tried | Train dR [95% interval] | Validation |
|---|---|---|
| size cells = lost/small/big | +0.0082 [+0.001, +0.016] | 53.80 (rejected) |
| size cells = exact payout | -0.0053 [-0.018, +0.009] | 54.07 |
| + extra size key (2,) | -0.0119 [-0.026, +0.001] | 54.18 |
| + extra size key (2,3) | **-0.0145 [-0.029, -0.001]** | **54.21** (kept: first interval clear of zero) |

Not kept (no gain on both checks): single-lag experts (lag 7 / 7+8 / all 8), never-forget
memory (half-life 1e9 or 200), blending 2-cell + 4-cell win experts, a stronger size prior
(`k_machine` 80), size half-life 200. About 10 settings were scored on validation in this
round, so 54.21 is slightly optimistic. The train-cutoff interval is the stronger evidence.

Rebuild and check it (from `mainFiles/`):
```sh
python run_pair_mixer.py --out runs/combined
python check_submission.py runs/combined/validation_predictions.csv --split validation   # 54.21
python check_submission.py runs/combined/submission.csv   # SHA-256 068259d6...
```

### Reusable prompt: learn from another solution

````text
Here is another team's write-up of their solution: <path>. Compare it with our model
(mainFiles/slot_forecasting/pair_mixer.py, current validation <score>). First list, per
condition, where theirs beats ours. Then list every idea they use that we don't. Test each
one with mainFiles/tune_pair_mixer.py '<json settings>' --val, adding an option to PairMixer
if needed. Keep a change only if it improves BOTH the train-cutoff dR (ideally with the 95%
interval below zero) AND validation. Stop when the next tweaks are within noise. Don't copy
ideas that only help on validation. Work on a new branch, never main. Then update
docs/MODEL_PROMPTS.md, CHANGELOG.md and AGENTS.md section 12, rebuild with run_pair_mixer.py,
run the tests and check_submission.py, and commit and push.
````

## What moved the score (in order)

| # | Prompt / instruction that worked | What it led to | Validation |
|---|---|---|---|
| 0 | "Run the starter and explain each score" | Understood that points = 60/(1+R) and 30 = baseline | 30.00 |
| 1 | "Train on every past spin (history.csv), not just the 768 labelled queries, and make sure no input reads the tick being predicted" | Lag-feature logistic model + no-lookahead tests (since removed; superseded by the pair mixer) | 31.96 |
| 2 | **"Before changing the model, split the validation error into the part from the win chance (p_0) vs the part from payout size"** | Found ~93% of error was the win chance; true win chance swings 0.04 to 0.99 while ours barely moved | (diagnosis) |
| 3 | **"Read the kit's own docs (EXPERIMENTS.md, transition.py) for hints about how the simulator works"** | EXPERIMENTS.md: "a useful pair of earlier spins may differ between machines" | (diagnosis) |
| 4 | **"For each machine, find which PAIR of lags best predicts its next win; count which pairs come up most"** | Only three pairs matter: (1,3), (4,7), (2,5), ~200 machines each | (diagnosis) |
| 5 | **"The pair's effect changes over time, so test it online: per-machine decayed win table per candidate pair, mixed by recent accuracy (Fixed-Share style)"** | Win-chance error vs outcomes 0.214 -> 0.121 on train | **50.29** |
| 6 | **"Print the 4-cell win rates of each machine's best pair and look for a shape"** | Cells are HLLH or LHHL: the machine wins when the two lags *match* (or *differ*), 96-100% vs 2-4%. Experts switched to 2 cells (match/differ) | **52.92** |
| 7 | **"Train and validation disagree on the best half-life; find out why before trusting either"** | Ticks *inside* the history prefer short memory (6), but the *cutoff tick* prefers longer (10-14), on train labels.csv and validation truth alike. Tune on train labels at the cutoff | **53.35** |
| 8 | "Split the remaining error into win part vs size part per condition" | Now ~50/50. Payout size has exactly two shapes per machine type and the mode isn't predictable from history, so sizes are near their floor | (diagnosis) |
| 9 | **"Build a tuning harness that scores on train labels at the cutoff, per condition, as an estimated change in R with an episode-bootstrap interval; keep a change only if it ALSO improves validation"** | Two memory lengths per pair (half-lives 12 and 40) | 53.56 |
| 10 | **"Re-test ideas you rejected on history ticks, now at the cutoff"** | Payout-size experts keyed on lag 1 and lags (1,2), rejected earlier, actually help at cutoffs | **53.75** |
| 11 | **"Compare with another team's write-up (rule model v4, 52.89) and test each idea it has that we don't"** | Their size rules look at earlier payout SIZE, ours only at win/loss. Keying size experts on the exact earlier payout, with lags (), 1, (1,2), 2, (2,3): train cutoff dR -0.0145 [-0.029,-0.001] | **54.21** |

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

- From the other team's rule model, tested on train cutoffs + validation and NOT kept:
  single-lag experts (lag 7, lags 7+8, all 8: validation 53.65 / 53.59 / 53.36), a
  never-forget memory (half-life 1e9 or 200: dR +0.001), blending parity + 4-cell experts
  (53.44), size prior strength 80 (noise), lost/small/big size cells (dR +0.008, worse).

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
