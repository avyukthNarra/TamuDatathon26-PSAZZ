# E2: Per-machine adaptive memory by prequential loss

Teammate handout. Prepared 2026-11-07 from read-only git (`origin` fetched, no checkouts).

Sources:
- Experiment branch: `origin/avyukth/exp-e2-adaptive-memory` (PR #8, draft, by the Codex agent), head `6511dfb`.
- Follow-up branch: `origin/avyukth/exp-combo-e2-memory` (PR #14), head `7c506e2`.
- Results: `HOUSE_EDGE_EXPERIMENT_RESULTS.md` (E2 section), `worktrees/_coord/CODEX_REPORT.md`, `worktrees/_coord/RESULTS.md`, `worktrees/_coord/E2_C10_CHECK.md`.

---

## 1. TL;DR

- Each machine's win model uses recency weighting with a half-life. No single half-life suits every condition (stable wants long or no decay, drift and shared change want about 32), and the test set does not say which condition a machine is in.
- E2 picks the memory per machine by replaying its own past spins. For each candidate half-life it asks, "how well would this have predicted my earlier spins, using only the spins before each one?" It then mixes the candidates with soft weights.
- Tuning result: a real but modest gain. Against the fixed E1 model at C=3 and half-life 32 it is +0.81 (CI [+0.23, +1.38]). The fair check, at matched C=10, is +0.93 (37.15 vs 36.22). The gain is mostly on stable and drift machines. Short-history is unchanged.
- Not yet applied to E6, E7, or the pair mixer. It has not been audited. All numbers are tuning-only, on the fixed tuning episodes.
- PR #14 (combo version) has code and tests but **no results yet**. Its tuning scoring has not been run.
- **Check before porting:** the code's soft-weight formula is `exp(-T * (loss - best))`, not the `exp(-loss / T)` written in the summary. See section 9.

---

## 2. The problem it solves

- E1 (pair-logistic, `pair_logistic.py`) fits one L2-regularised logistic regression per machine for P(win). It weights recent spins with a single half-life, fixed at 32 in the comparison.
- Conditions differ in how fast history goes stale:
  - Stable machines: long memory or no decay is best. In the E1 analysis, the p0 error on stable is lowest with no decay (0.028), while drift and shared are lowest near half-life 32 (`HOUSE_EDGE_EXPERIMENT_RESULTS.md`, condition table).
  - Drift and shared-change machines: half-life around 32 is best.
  - Short-history machines: little data, so memory choice matters less.
- The test queries carry no condition labels. A fixed half-life is a compromise. A per-machine choice has to be learned from the machine's own data, and it must not peek at the labels it is predicting.
- Idea: score the candidates on the machine's own past, in the order the spins happened. That is "prequential" evaluation: each outcome is predicted before its label is used.

---

## 3. The method in detail

### 3.1 Pseudo-code

```
predict_win(X, y, ticks, query, cutoff):              # adaptive_memory.py:120-230
    eligible  = rows with tick < cutoff                # line 145
    eval_idx  = eligible[-max_eval_rows :: eval_stride]  # line 148; about 8 spins

    for i in eval_idx:                                 # prequential loop, 154-177
        prefix = rows with tick < tick[i]              # strictly earlier only
        if len(prefix) < min_train: skip
        for cand in candidates:                        # half_life x C
            p = fit_prob(prefix, X[i], cutoff=tick[i], cand)  # prior if 1-class or short
            p = clip(p, 1e-6, 1-1e-6)
            loss[cand].append((p - y[i])**2)           # Brier; y[i] used only as target

    mean_loss[cand] = mean(loss[cand])                 # lines 179-182

    soft: w[cand] = exp(-T * (mean_loss[cand] - min_loss)), normalised   # 191-198
    hard: w = 1 for first cand with mean_loss <= min_loss + tie_tolerance # 184-190
    no evaluated losses: w = 1 for candidates[0] (no decay, lowest C)    # 199-201

    final: for each cand with w > 0:
        p_cand = fit_prob(eligible rows, X_query, cutoff, cand)          # 203-213
    return sum_cand  w[cand] * p_cand                                    # 214-216
```

`fit_prob` (lines 102-118): if there are fewer than `min_train` rows or only one class, return the smoothed prior `(sum(y) + 2*prior_win) / (n + 2)`. Otherwise fit `LogisticRegression(C, max_iter=500)` with sample weights `2^(-(cutoff-1-tick)/half_life)`, or weight 1 when half-life is None (lines 96-100). Ages are measured in **tick numbers**, so gaps in the tick sequence stay gaps.

### 3.2 Candidate grid

Candidates are the product of half-lives and Cs, sorted for stable tie-breaks (lines 72-82): no decay first, then longer finite half-lives, then lower C. With the tuned settings, half-lives (None, 16, 32, 64) times Cs (3,) gives 4 candidates. With Cs (3, 10) it gives 8.

### 3.3 Wrapper

`AdaptiveMemoryForecaster` (lines 233-273) subclasses E1's `PairLogisticForecaster`. Its `win_probability(seq, cutoff)`:
1. Builds the E1 query lag vector from the ticks before the cutoff.
2. Builds training rows only from ticks `< cutoff`, skipping ticks with missing lags.
3. Sets the prior to `1 - base.global_prior[0]`, or 0.4 if that is unset (lines 265-267). The global prior is fit on train.
4. Calls the selector and stores diagnostics keyed by `(cutoff, sorted history)` (lines 271-272).

The payout-shape path is inherited unchanged from E1.

### 3.4 Leakage-safety argument

1. **Future ticks are dropped.** The wrapper skips `tick >= cutoff` (line 253-255), and the selector's `eligible` mask does the same (line 145). Test `test_future_labels_do_not_change_cutoff_prediction` checks that flipping labels at and after the cutoff leaves the prediction unchanged.
2. **Each selection score uses only the past.** Outcome `i` is predicted from `prefix = {tick < tick[i]}` (line 155). Its label is read only after the prediction, as the scoring target (line 167). Test `test_prequential_prediction_does_not_use_its_current_label` flips one label and checks that its own prediction is unchanged.
3. **Weights come only from scored past outcomes.** The soft or hard choice uses only `mean_loss`, which is built from step 2.
4. **The final forecast uses only pre-cutoff rows.** The refit at line 209 uses `train_X`, which is restricted to `eligible`.
5. **The prior uses training labels only**, through `global_prior`.

Residual risk, not leakage: the candidate set, the temperature, and the C=3 vs C=10 choice were all compared on the same tuning episodes. The earlier `diag.py` and `diag2.py` diagnostics looked at all validation targets. The audit has not been scored for E2.

---

## 4. Parameters

| Parameter | Code default | Tuned value (Codex) | Meaning |
|---|---|---|---|
| `half_lives` | `(None, 16, 32, 64)` | same | Recency half-lives in ticks. `None` means no decay. |
| `Cs` | `(1.0, 3.0)` | `(3,)`, or `(10,)` for the matched check | LogisticRegression inverse regularisation. Lower C is stronger regularisation. |
| `mode` | `"hard"` | `"soft"` | Hard picks one candidate. Soft mixes them by weight. |
| `temperature` | `10.0` | `128` | Sharpness of soft weights. Applied as `exp(-T * gap)`, see section 9. |
| `loss` | `"brier"` | `"brier"` | Scoring rule for prequential predictions. Log-loss is also supported. |
| `max_eval_rows` | `32` | `32` | How many of the most recent eligible rows are considered. |
| `eval_stride` | `4` | `4` | Every n-th of those rows is scored. |
| `min_train` | `16` | `16` | Minimum prefix size before a row is scored, and before a logistic fit is used. |
| `tie_tolerance` | `0.0` | `0.0` | Hard mode: candidates within this loss gap of the best count as tied. |
| `prior_win` | from `global_prior`, else 0.4 | from train | Fallback win rate for short or one-class histories. |
| `clip` | 1e-6 | 1e-6 | Probability clipping, lines 165, 212, 230. |

The PR #14 combo variant differs: `max_eval_rows=24`, `selection_temperature=128`, `selector_tree_estimators=8`, `mix=0.5`, `short_alpha=0.6`.

---

## 5. Results

All results are tuning-only on the kit's `episode_folds` split, scored on the fixed tuning episodes. No audit predictions or scores were produced for E2.

### 5.1 Headline comparisons

| Comparison | E2 | Fixed | Difference | Paired 95% CI |
|---|---:|---:|---:|---|
| vs fixed E1, C=3, half-life 32 | 36.22 | 35.41 | +0.81 | [+0.23, +1.38] |
| **Matched C=10**, E2 `Cs=(10,)` vs fixed E1 C=10 HL32 | **37.15** | 36.22 | **+0.93** | [+0.14, +1.71] |
| E2 `Cs=(3,10)` vs fixed E1 C=10 HL32 | 36.97 | 36.22 | +0.76 | [-0.07, +1.48] |

The C=3 comparison does not isolate the memory-selection gain, because E1 at C=3 is a weaker base. The matched C=10 row is the fair test.

CIs are paired 95% episode-stratified bootstrap intervals (1,000 resamples, seed 314159, 12 episodes per condition). They are exploratory, because the same tuning episodes were used to choose the candidates.

### 5.2 C=10 fairness check, per-condition error

| Model | Points /60 | Stable | Short history | Drift | Shared change |
|---|---:|---:|---:|---:|---:|
| Fixed E1 C=10, HL32 | 36.217 | .066 | .186 | .168 | .103 |
| E2 adaptive, `Cs=(10,)`, soft T=128 | **37.146** | **.045** | .186 | **.153** | .106 |
| E2 adaptive, `Cs=(3,10)`, soft T=128 | 36.973 | .047 | .183 | .156 | .111 |

Per-condition differences for E2 `Cs=(10,)` minus fixed: stable -.021, short +.000, drift -.016, shared +.003.

So the gain is mostly on stable and drift. Shared change is slightly worse.

### 5.3 Tuning sweep (C=3 unless noted)

| Variant | Points /60 | Notes |
|---|---:|---|
| hard, Brier, n32 | 36.106 | |
| soft T=32, Brier, n32 | 36.072 | |
| **soft T=128, Brier, n32** | **36.223** | the C=3 headline |
| hard, log-loss, n32 | 36.099 | |
| soft T=32, log-loss, n32 | 36.120 | |
| hard, Brier, n64 | 35.782 | more eval rows did not help |
| hard, Brier, C=1 | 34.677 | |
| soft T=128, Brier, C=1 | 34.940 | |
| hard, Brier, `Cs=(1,3)` | 35.996 | |
| soft T=128, Brier, `Cs=(1,3)` | 35.952 | |
| Fixed E1 C=3, HL32 (anchor) | 35.408 | |
| Fixed E1 C=3, HL none | 30.608 | |

Log-loss and temperature choices moved the score by at most about 0.15, so the result is not very sensitive to them. C matters more.

Per-condition errors, E2 C=3 soft T=128: stable .052, short .182, drift .165, shared .126.

### 5.4 Earlier-run caveat

Early E2 runs at commit `ecc6e93` logged results with a bookkeeping KeyError that was later fixed. Their tuning predictions were not saved. Use the `6511dfb` numbers above, which have saved CSVs and SHA256 values listed in `CODEX_REPORT.md` and `E2_C10_CHECK.md`.

---

## 6. Code map

### 6.1 `origin/avyukth/exp-e2-adaptive-memory`

File: `mainFiles/slot_forecasting/adaptive_memory.py` (273 lines).

| Lines | What |
|---|---|
| 15-23 | `_Candidate` dataclass, name format `half_life=<n|none>,C=<c>` |
| 26-31 | `AdaptiveMemorySelector` class docstring. `ticks` are real tick numbers. |
| 33-45 | `__init__` signature and defaults |
| 46-71 | Argument validation |
| 72-82 | Candidate ordering and product of half-lives and Cs |
| 91-94 | `_prior`: smoothed fallback win rate |
| 96-100 | `_weights`: tick-based exponential recency |
| 102-118 | `_fit_probability`: logistic fit, or prior when short or one-class |
| 120-143 | `predict_win` signature and input validation |
| 145-148 | Eligibility mask and evaluation indices (`[-max_eval_rows::eval_stride]`) |
| 154-177 | Prequential loop: prefix, predict, score |
| 179-201 | Hard, soft, and fallback selection |
| 203-216 | Final refit on all eligible rows and weighted mix |
| 217-230 | Diagnostics dict and return |
| 233-245 | `AdaptiveMemoryForecaster` constructor, wraps E1 |
| 247-273 | `win_probability`: builds E1 rows, prior, calls selector, stores diagnostics |

### 6.2 Tests

File: `tests/test_adaptive_memory.py` (86 lines), 5 tests, all using `unittest`.

| Line | Test | Checks |
|---|---|---|
| 9 | `test_future_labels_do_not_change_cutoff_prediction` | Labels at and after the cutoff do not change the prediction. `eligible_rows == 28`. |
| 24 | `test_prequential_prediction_does_not_use_its_current_label` | Flipping one label leaves its own prequential prediction unchanged. |
| 41 | `test_recency_weights_use_tick_distance_across_gaps` | Weights use tick distance across gaps, `2^(-age/2)`. |
| 54 | `test_single_class_short_history_uses_finite_smoothed_fallback` | One-class short history returns the smoothed prior and sets `fallback`. |
| 66 | `test_forecaster_uses_train_prior_and_filters_future_ticks` | The wrapper filters ticks at or after the cutoff and uses the train prior. |

### 6.3 `origin/avyukth/exp-combo-e2-memory` (PR #14)

Files: `mainFiles/slot_forecasting/combo_e2_memory.py` (`ComboAdaptiveMemoryForecaster`, subclass of `ComboForecaster` from `combo_trees_blend.py`), and `tests/test_combo_e2_memory.py` (5 tests, per the PR).

The combo selector keeps separate soft selection for the ExtraTrees and logistic win components. Prequential fits use the most recent 24 rows (`max_eval_rows=24`) with 8-tree forests (`selector_tree_estimators=8`). The final refit uses the combo's full tree count. The `_estimator` factory (lines 92-101) and `_select_family` (lines 134+) are the relevant parts.

Base combo: `exp-combo-trees-blend` at `39168b8`, tuning 37.964/60 (from `RESULTS.md`).

---

## 7. How to run

Tests (PR #8 reports this command passed, 5 tests):

```bash
git worktree add ../e2-check origin/avyukth/exp-e2-adaptive-memory   # or any checkout of the branch
cd ../e2-check
python3 -m unittest discover -s tests -v
```

The test file imports `mainFiles.slot_forecasting...`, so run from the repository root. If discovery cannot import the package, add `-t .`. That is untested here.

Tuning run: the harness is `worktrees/_coord/codex_tune.py`. It is local to the coordination folder and is not in git. It runs on the fixed tuning episodes only and never scores audit. Key arguments:

```bash
python3 worktrees/_coord/codex_tune.py \
  --worktree <checkout of exp-e2-adaptive-memory> \
  --module mainFiles.slot_forecasting.adaptive_memory \
  --class-name AdaptiveMemoryForecaster \
  --grid-json worktrees/_coord/e2-c10check-adaptive-grid.json \
  --out <new unique .json path> \
  --exp E2
```

The grid JSON gives each candidate's `kwargs`. The matched C=10 grid is `worktrees/_coord/e2-c10check-adaptive-grid.json`. Pass `--no-log` to skip appending rows to `RESULTS.md`. Output is incremental JSON, and it refuses to overwrite an existing file.

Important: pass the tuned kwargs explicitly (`mode="soft"`, `temperature=128`, `Cs=...`). The constructor defaults are `hard`, `T=10`, and `Cs=(1,3)`, which are not the tuned settings.

The C=10 check is in `worktrees/_coord/E2_C10_CHECK.md`. Saved prediction CSVs and SHA256 values are listed there and in `CODEX_REPORT.md`.

---

## 8. How to port it to other models

General recipe: the selector only needs a `fit_prob(rows, weights, query)` function and a candidate list. It is not tied to logistic regression in principle. Keep the prequential loop, the eligibility mask, and the final refit unchanged. The model being ported must be refittable on any prefix, so the cost grows with `eval rows x candidates`.

Cost check before any port: with 4 half-lives and about 8 scored spins per machine, a single query costs roughly 8 x 4 prefix fits plus 4 final fits. Measured runtimes were 11-15 s per tuning run for E2 at 1 C value, and 27 s for 2 C values (`RESULTS.md`).

### 8.1 E6 (signed-L1 logistic, pair selection)

- E6: signed lag interactions with per-machine L1 pair selection. Winner C=0.5, HL=16, tuning 48.5376/60. Its winning candidate used 375 fits, 9 fallbacks, and no max-iteration hits (`RESULTS.md`, E6 row).
- Currently E6 uses a fixed half-life of 16 (`HOUSE_EDGE_EXPERIMENT_RESULTS.md`, E6 section).
- Needed change: `AdaptiveMemorySelector` hard-codes `LogisticRegression(C=c, max_iter=500)`, with no penalty argument. Add a model factory hook so that `penalty="l1"`, `solver` and the signed features are kept. The L1 pair selection must be redone inside each prefix, or it leaks future selection into the past score. Do not reuse the full-history pair list across prefixes.
- **First experiment:** `Cs=(0.5,)`, `half_lives=(None,16,32,64)`, soft T=128, Brier, `max_eval_rows=32`, `eval_stride=4`, `min_train=16`. Score on the tuning episodes with the E6 settings otherwise unchanged. Compare to 48.5376. Also run the fixed HL16 control in the same harness. Expected size: the HOUSE_EDGE doc suggests about +0.5 to +1.5, but that is an estimate, not a result.
- Alternative cheaper first try: the E4 episode-pooled half-life choice (section 8.4).

### 8.2 E7 (contextual neural model)

- E7: contextual neural model, recent HL16, seed 2026, tuning 50.1839/60. Its paired gap over E6 is +1.65 with CI [-0.06, +3.27], which crosses zero. Treat E7 as a tuning candidate, not a confirmed winner (`CODEX_REPORT.md`).
- The E7 code was not read for this handout. Confirm its model structure before porting.
- Neural fits are much more expensive than logistic fits. Per-machine prequential selection over 4 candidates, each needing a fresh train, may be too slow for the time budget.
- **First experiment:** do the cheapest version first. Use the E4-style episode-pooled choice (one selection per episode, 4 candidates, last 48 spins every 3rd), not per-machine. Confirm runtime is acceptable, then test per-machine soft selection with `max_eval_rows=16` only if the cost allows it.

### 8.3 Pair mixer

- The handout does not cover the pair mixer's code. The facts say only that E2 has not been applied to it. Read its win head before porting.
- Porting is simplest if the mixer's per-machine win head is a logistic model over lag features. Then the same `AdaptiveMemoryForecaster` pattern applies: a wrapper that builds rows from ticks `< cutoff`, a prior from train, and a selector call.
- **First experiment:** fixed-C check first. Use the mixer's own C, then run the selector with `Cs=(its C,)`, soft T=128, the four half-lives. Compare to the mixer's fixed half-life. Use the same matched-C logic as section 5.2, because a C mismatch confounded the C=3 comparison.

### 8.4 Pooled episode choice (E4, Claude)

- E4 pools the half-life decision across the episode's 8 machines. Shared changes hit all machines, so there is 8 times as much evidence. Candidates {None, 64, 32, 16}, last 48 spins, every 3rd, Brier summed over the episode, hard pick (`episode_memory.py`, `_episode_losses`).
- On the combo model it gave tuning 39.037 vs 37.964 for the combo base, +1.1 (`RESULTS.md`).
- Compare with E2: E2 is per machine with soft weights. E4 is per episode with a hard pick. Both use prequential scoring, so they share the leakage argument in section 3.4.

---

## 9. Gotchas

1. **Soft-weight formula.** The summary and `HOUSE_EDGE_EXPERIMENT_RESULTS.md` write the weights as `exp(-loss / T)`. The code on the branch computes `exp(-(loss - best_loss) * T)` (`adaptive_memory.py:194`). In the code, a larger `T` means sharper selection. At T=128 a Brier gap of 0.01 gives weight exp(-1.28), about 0.28. Under the written form, T=128 would be almost uniform. The 36.22 and 37.15 results come from the code as written, so the code is the reference. Confirm this before writing it into any doc or port.
2. **Defaults are not the tuned settings.** `mode="hard"`, `temperature=10`, `Cs=(1,3)`. Always pass the tuned values explicitly.
3. **Only about 8 scored spins per machine.** `eval_indices[-32::4]` keeps about 8 points. Selection is therefore noisy, and that may explain the modest gain. Larger `max_eval_rows` (64) scored worse (35.78 hard). Do not assume more eval rows is better without a test.
4. **Fallback when nothing is scored.** If no evaluated prefix reaches `min_train`, selection defaults to the first candidate, which is no decay with the lowest C. The `fallback` flag only marks short or one-class fits. It does not mark this case.
5. **Diagnostics memory.** `selection_diagnostics` is keyed by the full sorted history, so it grows with the number of queries. Turn it off or clear it for long runs.
6. **Hard mode is brittle.** Hard selection can flip between candidates on small loss gaps. The sweep showed soft and hard within about 0.15 points, so prefer soft unless there is a reason not to.
7. **C matters more than the selector.** Fixed E1 at C=10 is already 36.22. Comparing E2 at C=3 against fixed E1 at C=3 was not a fair test. Always compare at matched C.
8. **Not audited.** Every number in this handout is tuning-only. The earlier diagnostics saw all validation. Treat the CIs as exploratory.
9. **Combo PR #14 has no results.** The PR says tuning scoring "remains pending". Do not quote a combo E2 score.
10. **Process notes.** PR #8 is a draft, and its PR body says AGENTS.md §12 and §13 were not updated. The E2 branch does not change `CHANGELOG.md`. PR #14 contains a changelog line.
11. **Wrapper coverage.** Only one test (`test_forecaster_uses_train_prior_and_filters_future_ticks`) exercises `AdaptiveMemoryForecaster`. The other four test the selector directly. Add wrapper tests before porting.
12. **Runtime.** E2 at C=10 took about 15 s for the tuning run. Two C values took 27 s. Neural or L1 models will be much slower. Measure first.
