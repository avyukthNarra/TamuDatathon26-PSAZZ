# Prompts that made the House Edge model better

A running log of the requests (to Claude Code or any coding agent) that led to
real score gains, and the ones that didn't. Use it to steer the next agent fast,
especially in round two. Validation points are out of 60; the pooled baseline is 30.

## Paste-ready prompt (rebuilds the 55.34 model from the starter kit)

Copy everything inside the box into a fresh coding agent opened in the repo root.
It describes the finished model exactly, so the agent doesn't have to rediscover it.

````text
You are working in the House Edge (TAMU Datathon 2026) starter kit. The kit is in
mainFiles/: data in mainFiles/public/{train,validation,test}, helpers in
mainFiles/slot_forecasting/ (schema.py, io.py, contest.py, rounds.py, starter.py,
submission_contract.py), and the organizers' baseline in run_starter.py (it always
scores ~30/60 because "pooled" IS the reference model; points = 60/(1+R)).
Read AGENTS.md first and follow it: work on a feature branch, never commit to main.

GOAL: build a forecaster that scores ~55.34/60 on validation, plus a verified entry ZIP.

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
- Payout size given a win is EXACTLY one of two fixed mixes per machine type (a "small" and
  a "big" mode), readable from validation/targets.csv (RULES.md: learn from training AND
  validation). The mode switches spin to spin: for many machines it is set by whether the
  spin 1, 2 or 4 back was a big win. Mode is NOT persistent (a machine's whole past size
  record predicts it no better than a coin flip), and the win chance is continuous (no modes).
- True win chances are U-shaped: most sit near a "dead" level (median ~0.044) or a "live"
  level (median ~0.898), only ~22% between 0.2 and 0.8. Neighbour machines carry no usable
  lagged signal (apparent links are small-sample noise in short-history episodes).
- Tune at the QUERY CUTOFF tick (train/labels.csv), NOT on ticks inside history.csv;
  history ticks rank settings differently. Confirm on validation once.

BUILD mainFiles/slot_forecasting/pair_mixer.py with:
1. class PairMixer, one online "walk" per (episode, machine) over own[t] = payout class
   index (0 = loss) or None. At each t: EMIT the forecast, THEN absorb tick t.
   - Win experts: one per (pair, half_life) for half_life in (12, 40). Each keeps
     exponentially decayed [wins, n] per cell. Default cell_mode "parity" = 2 cells
     (lags agree / disagree); also support "full" = 4 cells (won/lost at a x at b).
     Expert prediction for a cell with decayed (w wins, n spins) = posterior mean under a
     Beta-MIXTURE prior: components (weight, mean, strength) = (0.45, dead, 6),
     (0.45, live, 6), (0.1, 0.5, 2), Beta(a = mean*strength, b = (1-mean)*strength).
     post_k ∝ weight_k * B(a_k + w, b_k + n - w) / B(a_k, b_k) (use math.lgamma);
     rate = sum_k post_k * (a_k + w) / (a_k + b_k + n). dead/live = median true win chance
     below / above 0.5 in validation targets.csv. Without targets: fall back to
     (w + k_win*base) / (n + k_win), k_win = 0.25, base = (decayed machine wins + 1) /
     (decayed n + 2) at the first half_life. If a lag is missing, the expert predicts base.
   - Mix win experts with exponential weights: log_w -= eta*(p - won)^2, eta = 8 with the
     prior (16 without), uniform prior, NO Fixed-Share (alpha = 0; it hurt by +0.10-0.14 R).
   - Size given win (4 classes): experts keyed on () machine-level, (1,), (2,), (3,), (4,).
     A size cell is the EXACT earlier payout class at that lag (0/1x/2x/5x/20x). Each keeps
     decayed counts of the 4 sizes per cell (size half-life 80). TWO-MODE prediction: with
     the type's (small, big) mixes, llr = sum_k count_k * log(big_k / small_k) (floor 1e-3),
     P(big) = sigmoid(llr) (clip llr to +-50), cell forecast = (1-P)*small + P*big. An empty
     cell uses the machine-level counts the same way. Mix size experts with
     log_w -= size_eta * squared error (size_eta = 8), only on winning ticks.
     Fallback when no modes are available: keys (), (1,), (1,2), (2,), (2,3) with free
     frequencies, smoothed toward the machine mix (k_size 16) and a type prior (k_machine 32).
   - Decay all tables every tick, including missing ticks (decay_missing = True).
   - Output [1 - pw] + [pw * s_k], floor each prob at 1e-4, renormalize.
2. fit(): per machine_type size prior = counts of each winning payout size over train
   history (+0.5 smoothing), normalized.
3. discover_pairs(): for each machine, over all lag pairs from 1..8, pick the pair whose
   4 cells best separate its win rate (weighted between-cell variance, need >= 20 obs);
   count winners. choose_pairs(): keep pairs with count >= 0.3 x the top count
   (fallback (1,3),(4,7),(2,5)).
4. adapt(train_queries, train_history, labels, size_modes): pairs = choose_pairs(train);
   build "parity" and "full" models; pick the one with the lower 5-class Brier vs
   train/labels.csv at the cutoff. Print the pairs and both Briers.
5. size_modes_from_targets(targets, queries): per type, the true size-given-win vectors
   p[1:]/(1-p[0]) (skip p_win < 0.05), rounded and merged within 0.02; return the two biggest
   groups as (small = more 1x, big). Return {} unless each type shows exactly two clear modes.
6. win_prior_from_targets(targets): the Beta-mixture above from the medians of true win
   chances below / above 0.5; None if either side has < 20 targets.

ALSO CREATE
- mainFiles/entry_template/predict.py: loads train, reads size modes AND the win prior from
  <data>/validation/targets.csv (else from priors.json saved next to it), calls adapt(),
  predicts --split (default test), writes the CSV. Insert its own folder on sys.path first.
  entry_template/reproduce.json:
  {"image": "python", "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"]}
- mainFiles/run_pair_mixer.py --out runs/pair-mixer: adapt on train, write
  validation_predictions.csv and submission.csv, score validation with
  contest.score_contest_files (print per-condition table, R and points), copy
  __init__.py, schema.py, io.py, pair_mixer.py + predict.py + reproduce.json + a README
  + priors.json ({"size_modes", "win_prior"}) into entry/, zip it with submission_contract.package_code, then RE-RUN entry/predict.py
  in a fresh process and fail if any probability differs by > 1e-8.
- mainFiles/tests/test_pair_mixer.py (unittest): (a) mutating or deleting any rows at
  tick >= t never changes predictions at t; (b) walk emits before absorbing;
  (c) on synthetic data whose wins follow lags (2,6) with XNOR, adapt() finds (2,6)
  and predicts confidently; (d) size_modes_from_targets recovers two known mixes;
  (e) with modes, size experts learn "big mode iff the spin 2 back paid 5x+" exactly;
  (f) win_prior_from_targets reads the two levels; (g) a 3-of-3 cell reads > 0.85 and a
  0-of-3 cell < 0.1 under the prior.
- Make sure mainFiles/runs/ is git-ignored.

VERIFY (and report the actual output)
  cd mainFiles && python -m unittest discover -s tests
  python run_pair_mixer.py --out runs/combined-v3
  python check_submission.py runs/combined-v3/submission.csv
  python check_submission.py runs/combined-v3/validation_predictions.csv --split validation
Expect validation ~55.34/60 (R ~0.084: stable ~0.069, short_history ~0.130,
drift ~0.065, shared_change ~0.074). The reference build's test submission SHA-256 is
63d7fc565002b14be12d8b8ccda0cece4eeee958894cbedc7f8e2cb735d57d5d. A rebuild will
only match it byte for byte if the arithmetic order is identical, so treat the score
as the check. Upload runs/combined-v3/submission.csv and runs/combined-v3/entry.zip.
````

The shortcut, if the branch `COMBINED_SOLUTIONS` is available, is to check it out and
run the VERIFY commands (current reference: 55.34). The prompt above is for rebuilding without it.

## What changed on `COMBINED_SOLUTIONS`, round 3 (54.90 -> 55.34)

Goal: push the score higher while following `mainFiles/RULES.md`. The rules that shaped the
work: use only organizer-released data (train AND validation are allowed, "Public validation
is available for model selection"; test answers are off limits); use observations strictly
before the prediction tick; missing ticks are unknown; the code must recreate the CSV offline
within 30 minutes (this entry: ~50 s for train + validation + test, standard library only).

**Where the error was (oracle bounds on the 54.90 model):** a perfect payout size would give
~56.4; a perfect win chance ~58.3. So the win chance was the bigger lever.

**Kept: a two-level prior for each win cell.**
- Validation truth shows true win chances are U-shaped: most near a "dead" level or a "live"
  level (histogram: 220 of 768 below 0.05, 172 above 0.90, only ~22% between 0.2 and 0.8).
- Each match/differ cell sees only a few recent spins (half-lives 12/40), so its estimate is
  noisy, and it used to be smoothed toward the machine's overall rate (~0.5): the wrong place.
- Now each cell's rate is its posterior mean under a mixture of Beta priors (dead level,
  live level, weak middle), so a noisy cell is pulled toward the level its counts favour.
  The levels are the medians of true win chances below/above 0.5 in the round's validation
  `targets.csv` (main: 0.044 / 0.898), so round two re-derives them. Expert weighting eta
  16 -> 8 fits better with sharper cells.
- Entry: `predict.py` reads both priors (size modes + win levels) from
  `/data/validation/targets.csv`, with a saved copy in `priors.json`. Verified: unzipped entry
  with `targets.csv` deleted reproduces the CSV byte for byte.

| Condition | 54.90 | **55.34 (now)** | Win part before -> after |
|---|---|---|---|
| stable | 0.073 | **0.069** | 0.0105 -> 0.0098 |
| short_history | 0.147 | **0.130** | 0.0156 -> 0.0126 |
| drift | 0.070 | **0.065** | 0.0104 -> 0.0093 |
| shared_change | 0.082 | **0.074** | 0.0116 -> 0.0098 |
| **Points** | 54.90 | **55.34** | |

**Testing process change:** `tune_pair_mixer.py --base '<json>'` compares candidates against
the CURRENT model (not the old defaults), so each change is measured on its own.

Every idea tested this round (vs the 54.90 model; train cutoff dR, negative = better):

| Idea | Train dR [95% interval] | Validation (halves) | Kept? |
|---|---|---|---|
| Neighbour machines' earlier wins (RULES.md hint) | residual check: strong links only in short-history episodes, matching chance for ~40-spin overlaps | not built | no, noise |
| Win prior, levels 0.03 / 0.93, strength 6 | -0.0143 [-0.029, +0.002] | 55.26 (54.93 / 55.66) | close |
| ...strength 2 / 4 / 10 | (all negative) | 55.17 / 55.14 / 54.57 | 10 too strong |
| ...levels 0.02/0.95, 0.04/0.90, middle 0.2 | -0.0102 / -0.0088 / -0.0087 | 55.20 / 55.32 / 55.28 | plateau |
| ...with half-lives (8,30) / (16,50) / (10,24,60) | -0.011 / -0.004 / -0.015 | 54.81 / 55.30 / 55.14 | no gain |
| ...with eta 8 | **-0.0144 [-0.027, -0.0005]** | 55.26 (54.94 / 55.65) | yes (eta) |
| ...with 4-cell experts | +0.032 | 53.81 | no |
| **Win prior from validation medians (0.044 / 0.898), strength 6, eta 8** | -0.0125 [-0.026, +0.002] | **55.34 (55.03 / 55.71)** | **YES**: adapts to round two |
| + size_eta 4 / mode_temp 1.3 / size half-life 120 / k_size 8 | +0.0009 / -0.0015 / +0.0008 / 0 | 55.41 / 55.28 / 55.30 / 55.34 | no (train not better) |

All ~12 win-prior variants had negative train dR and raised BOTH validation halves; the gain
is a plateau, not a lucky setting. The chosen one adapts its levels to the round's data.

Rebuild and check it (from `mainFiles/`):
```sh
python run_pair_mixer.py --out runs/combined-v3
python check_submission.py runs/combined-v3/validation_predictions.csv --split validation   # 55.34
python check_submission.py runs/combined-v3/submission.csv   # SHA-256 63d7fc56...
```

### Prompts used this round (paste these to repeat it)

1. Start from the rules and the error budget:
````text
Read mainFiles/RULES.md and list the constraints that limit what we can do (data allowed,
no lookahead, runtime, reproducibility). Then, on the current model, compute oracle bounds:
the score if payout size were perfect, and if the win chance were perfect. Work on whichever
part has more to gain.
````
2. Test a rules hint properly before building it:
````text
RULES.md says "other machines' earlier outcomes may help". Check it per machine PAIR, not
pooled: correlate our model's win residuals with each neighbour's win at lags 1-8, compare
with what chance gives for the actual overlap length, and split by condition and sign.
````
3. Look at the shape of the truth:
````text
Plot a histogram of the true win chances in validation targets.csv. If it is U-shaped,
replace each win cell's smoothing toward the machine's base rate with the posterior mean
under a mixture of Beta priors (dead level, live level, weak middle), read the levels from
validation targets at run time, and tune strength and eta with tune_pair_mixer.py --base
'<current model json>' --val.
````
4. Lock it in reliably:
````text
Keep the version that improves train-cutoff dR and BOTH validation halves and adapts to new
data; check a few more settings on top and stop if none clearly stacks. Rebuild, run tests
and check_submission.py, prove the unzipped entry reproduces the CSV with validation targets
deleted, update docs/MODEL_PROMPTS.md, CHANGELOG.md and AGENTS.md section 12, commit, push.
````

## What changed on `COMBINED_SOLUTIONS`, round 2 (54.21 -> 54.90)

Source: a second team's handoff notes, `HOUSE_EDGE_EXPERIMENT_RESULTS.md` (E0-E11 plus a
final E6+E7 blend; best 50.02 audit). Their best is below ours, so nothing was copied
wholesale. Their *processes* were tested on our model instead, and one of them led to the gain.

**Kept: payout size is a two-mode switch, and each size cell estimates which mode is on.**
- Found by checking validation `targets.csv`: every true size-given-win vector is exactly one
  of **two** mixes per machine type (e.g. steady: small `[.75 .20 .05 .00]`,
  big `[.46 .44 .11 .00]`).
- The mode is **not** a per-machine constant: a machine's whole past size record predicts it
  no better than 50/50. It switches spin to spin; a per-machine rule search on train found
  it is set by whether the spin **1, 2 or 4 back was a big win** for many machines.
- Model change: `PairMixer(size_modes=...)`. Each size cell now holds one number, P(big
  mode), computed by Bayes' rule from its decayed size counts, instead of 4 free
  frequencies. Size keys are now `()`, `(1,)`, `(2,)`, `(3,)`, `(4,)` (lag 4 was missing).
- Where the modes come from: `size_modes_from_targets()` reads them from the round's
  `validation/targets.csv` at run time (RULES.md: "learn from the main training and
  validation data"). The entry also saves a copy as `size_modes.json` as a fallback. Without
  either, the model falls back to the old free-frequency size experts.
- Verified: the unzipped entry gives a byte-identical CSV even with `targets.csv` deleted.

| Condition | 53.75 | 54.21 | **54.90 (now)** | Their best (E6+E7, audit) |
|---|---|---|---|---|
| stable | 0.096 | 0.083 | **0.073** | 0.148 |
| short_history | 0.151 | 0.151 | **0.147** | 0.235 |
| drift | 0.104 | 0.093 | **0.070** | 0.212 |
| shared_change | 0.114 | 0.100 | **0.082** | 0.203 |
| **Points** | 53.75 | 54.21 | **54.90** | 50.02 |

**More reliable testing (process change):** `tune_pair_mixer.py --val` now also prints the
kit's **tuning / audit halves** of validation (48 episodes each; same split as their
`episode_folds`). A change is kept only if the train-cutoff dR improves (95% interval below
zero), validation improves, and **both halves** improve.

Every idea tested this round (train cutoff dR, negative = better; validation; halves):

| Idea (source) | Train dR [95% interval] | Validation (halves) | Kept? |
|---|---|---|---|
| Size prior by type + signal (their E5) | -0.0007 [-0.0013, -0.0002] | 54.21 (54.07 / 54.38) | no, too small |
| Size memory doesn't fade on missing ticks | -0.0022 [-0.0046, +0.0002] | 54.22 | no, noise |
| Short-history win calibration x0.8 / x1.2 / x1.5 (their E9) | +0.002 / +0.002 / +0.007 | 54.10 / 54.16 / 53.98 | no, already calibrated |
| Pair x lag-8 experts (their "lag 7/8 alone") | +0.0037 | 54.16 | no |
| Two-pair experts (pair x pair) | +0.0117 | 53.84 | no |
| ...with a head start for simple rules (their prior_weight) | +0.0095 | 54.04 | no |
| Episode-pooled memory choice (their E4) | worse on train | 53.68-53.90 | no, our fast/slow experts already do it |
| Per-machine memory choice (their E2) | worse on train | 53.74-54.04 | no |
| Size key lag 4 added (free frequencies) | -0.0058 [-0.013, +0.003] | 54.45 (54.15 / 54.80) | led to the next row |
| **Two-mode size experts, lags 1,2,4** | -0.0192 [-0.040, +0.005] | 55.00 (54.55 / 55.53) | close; interval touches 0 |
| **Two-mode size experts, lags 1,2,3,4** | **-0.0233 [-0.042, -0.001]** | **54.90 (54.44 / 55.43)** | **YES**: interval clear of 0 |
| Two-mode, other strengths/memories (temp 0.7/1.3, size half-life 40/160, size_eta 4/16) | -0.016 to -0.021 | 54.82-55.04 | plateau: the gain is robust |
| Two-mode, lags 1-5 / 1-8 | -0.019 / -0.010 | 54.92 / 54.59 | no gain |

Rebuild and check it (from `mainFiles/`):
```sh
python run_pair_mixer.py --out runs/combined-v2
python check_submission.py runs/combined-v2/validation_predictions.csv --split validation   # 54.90
python check_submission.py runs/combined-v2/submission.csv   # SHA-256 1fbdda9b...
```

### Prompts used this round (paste these to repeat it)

1. Compare with another solution:
````text
Read <path to the other team's notes>. Compare it with our model on branch
COMBINED_SOLUTIONS (mainFiles/slot_forecasting/pair_mixer.py, validation <score>). Our model
is better overall, so don't copy it: list each PROCESS or idea they used that we haven't
(memory selection, shrinkage, priors, calibration, evaluation protocol) and test each one on
our model with mainFiles/tune_pair_mixer.py '<json>' --val. Keep a change only if the
train-cutoff dR interval is below zero AND validation AND both validation halves improve.
````
2. Find where the error is before building anything:
````text
Using validation targets.csv, break the remaining error down: win part vs size part per
condition, and for the win part, queries whose true win chance is in the middle (0.15-0.85)
vs near 0/1. Then check whether the true win chance and the true size mix take only a few
distinct values per machine type.
````
3. Turn a discrete structure into a model:
````text
The true size-given-win mix is exactly one of two vectors per machine type. Check whether
the mode is a persistent per-machine state (does the machine's past size record predict
it?) or switches by a rule on earlier spins (search per machine on train history: which
lag or lag pair, on won/lost or big/small, best explains the size of its wins). Then make
the size experts predict P(big mode) per cell by Bayes' rule from the two known mixes,
read the mixes from validation targets.csv at run time with a saved fallback, and test it.
````
4. Lock it in reliably:
````text
Check the settings around the best one (strength, memory, keys). If they all score about
the same, pick the one whose train-cutoff interval clears zero, not the top validation
number. Revert the model file and re-apply only the change that worked (no dead options).
Rebuild, run the tests and check_submission.py, prove the unzipped entry reproduces the CSV
exactly even if validation/targets.csv is missing, update docs/MODEL_PROMPTS.md,
CHANGELOG.md and AGENTS.md section 12, then commit and push to the same branch.
````

## What changed on `COMBINED_SOLUTIONS`, round 1 (53.75 -> 54.21)

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
| 8 | "Split the remaining error into win part vs size part per condition" | Now ~50/50. Payout size has exactly two shapes per machine type. (Concluded the mode wasn't predictable; row 12 shows it is, from lags 1-4) | (diagnosis) |
| 9 | **"Build a tuning harness that scores on train labels at the cutoff, per condition, as an estimated change in R with an episode-bootstrap interval; keep a change only if it ALSO improves validation"** | Two memory lengths per pair (half-lives 12 and 40) | 53.56 |
| 10 | **"Re-test ideas you rejected on history ticks, now at the cutoff"** | Payout-size experts keyed on lag 1 and lags (1,2), rejected earlier, actually help at cutoffs | **53.75** |
| 11 | **"Compare with another team's write-up (rule model v4, 52.89) and test each idea it has that we don't"** | Their size rules look at earlier payout SIZE, ours only at win/loss. Keying size experts on the exact earlier payout, with lags (), 1, (1,2), 2, (2,3): train cutoff dR -0.0145 [-0.029,-0.001] | **54.21** |
| 12 | **"Compare with a second team's experiment log; test their processes on our model; find where the error is; check if the truth takes few distinct values"** | Payout size is exactly one of two mixes per type, switched by whether the spin 1, 2 or 4 back was a big win. Size cells now estimate P(big mode) by Bayes' rule with the two mixes read from validation targets. Train cutoff dR -0.0233 [-0.042,-0.001]; both validation halves up | **54.90** |
| 13 | **"Read RULES.md; compute oracle bounds (perfect size vs perfect win); test the neighbour hint per machine pair; histogram the true win chances"** | Win chance had more headroom; neighbour links were small-sample noise; true win chances are U-shaped, so each win cell now uses a dead/live Beta-mixture prior read from validation targets (eta 16 -> 8). Both validation halves up | **55.34** |

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
- **Check whether the truth is discrete.** Looking at the distinct true size vectors per type
  showed exactly two, which turned a 4-number-per-cell estimate into 1 number per cell (row 12).
- **Compute oracle bounds first.** "What would we score if X were perfect?" says which part
  of the model is worth the next hour (row 13: win 58.3 vs size 56.4).
- **Check a hint against chance with the right sample size.** The neighbour signal looked
  25x stronger than chance until the short-history overlap length was used (row 13).
- **Shrink toward where the truth actually lives.** Smoothing toward an average (0.5) is wrong
  when the truth is U-shaped; a mixture prior pulls each estimate to the nearer level.
- **Use both halves of validation.** A gain must show on the kit's tuning AND audit halves,
  not just the total; pick from a plateau, not a single best point.
- **Package the safe model before improving it.** Main entries can't be replaced after 9 PM.
- **Say what can't change:** standard-library Python, no lookahead, missing tick ≠ 0,
  30-minute runtime, a test that proves no future data leaks in.

## Prompts that did NOT help much

- Round 3, NOT kept: neighbour-machine inputs (per-pair links were small-sample noise), a
  too-strong win prior (strength 10), 4-cell experts with the prior, re-tuned half-lives,
  and size tweaks on top (size_eta 4, mode_temp 1.3, size half-life 120).
- From the second team's experiment log (round 2), NOT kept: size prior by type+signal,
  short-history shrinkage/calibration, episode-pooled or per-machine memory selection,
  pair x lag-8 and two-pair experts (with or without a head start). Details in the round 2 table.
- From the other team's rule model, tested on train cutoffs + validation and NOT kept:
  single-lag experts (lag 7, lags 7+8, all 8: validation 53.65 / 53.59 / 53.36), a
  never-forget memory (half-life 1e9 or 200: dR +0.001), blending parity + 4-cell experts
  (53.44), size prior strength 80 (noise), lost/small/big size cells (dR +0.008, worse).

- A 2-state hidden "size mode" tracker as a persistent per-machine state: no real gain,
  because the mode isn't persistent; it follows earlier spins (row 12). (Lag-keyed size tables looked useless
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

The entry already adapts: `predict.py` rediscovers the lag pairs on `/data/train`,
picks match/differ vs 4-cell experts by train cutoff Brier, and reads the two payout-size
mixes per type AND the dead/live win levels from the round's `/data/validation/targets.csv`
(it prints both; it falls back to free size frequencies or base-rate smoothing if round two's
data doesn't show those shapes). At 9 PM:

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
- "Do the true size-given-win vectors in round-two validation still take exactly two values
  per type? If not, how many? Which lags set the mode now (rerun the per-machine size-rule search)?"
- "Histogram round-two true win chances: still U-shaped? Are the printed dead/live levels sensible?"
