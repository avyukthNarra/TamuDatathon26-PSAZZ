# E6: Signed-L1 Interaction Logistic (teammate handout)

**Branch:** `avyukth/exp-e6-interactions` @ `c3ba387` · **PR:** #11 (draft, opened by Codex) · **Test suite:** 7 tests
**Main file:** `mainFiles/slot_forecasting/interaction_sweep.py`, class `InteractionSweepForecaster`
**Packaged entry:** `avyukth/entry-main` @ `dbb6e38` (E6 alone, no torch)

---

## 1. TL;DR

- E6 is our best **single linear model**. It scores **48.538 on tuning** and **48.194 on audit**, against a baseline of 31.06 / 30.57.
- It fits one L1-regularised logistic regression **per machine**, on **signed (±1)** win indicators at the last 8 tick lags, plus all 28 pairwise products of those lags.
- Two choices carry almost all of the gain: **signed coding** and **L1 penalty**. Turn off either one and the score falls to the 30s or 40s.
- It was the winner of a 96-config sweep. The chosen setting is `degree=2, encoding="signed", penalty="l1", C=0.5, half_life=16`.
- It runs in under a second for all of validation. It is 50% of our final entry, which is a 50/50 probability blend with E7 (50.65 tuning / 50.02 audit).
- The main open question is short history, which is the weakest condition (audit loss ratio .358). The next experiment is to restrict the pair search to the 3 global pairs the notes flag.

Read this section and then §3 and §8. Those two cover the method and how to run it.

---

## 2. Motivation: how E6 grew out of E1

**The task.** For each query (one machine in one episode at a cutoff tick) we predict a 5-vector over payouts `0, 1, 2, 5, 20`. Most of the error is in `p0 = P(payout = 0)`, the win chance. An oracle check in the Codex work showed about 94% of the error is the win chance. Given a win, the split across 1/2/5/20 is close to fixed per machine type. So the work is in predicting whether the next spin wins.

**What E1 established.** E1 (`pair_logistic.py`, `PairLogisticForecaster`) fits a per-machine logistic regression for P(win) on that machine's own pre-cutoff history. Its features are 0/1 win indicators at lags 1–8, plus all 28 pairwise products. Each row is weighted by recency. E1 at C=10 scored 36.22. Dropping the pairs dropped it to 29.46, below the baseline. So interactions matter.

**What E6 changed.** E6 keeps E1's shape and history contract, and changes two things:

1. **Coding.** The lags are mapped to ±1 (`2x − 1`) before the products are formed. Products of ±1 values encode "do these two lags match?" directly.
2. **Penalty.** The per-machine fit is L1 (sparse) rather than L2 (dense). Sparsity finds the few rules each machine actually uses.

The Codex sweep then tuned degree, C and half-life around that design. The lesson the team took from E1 through E11 was that **0/1-coded learners plateau around 40**, whatever the learner family. The ceiling was the representation. E6 moved the representation, and gained about 8 points over the best 0/1 model (E11 at 40.26 tuning).

**What E6 is not.** It does not pool across machines. Each query's machine gets its own small fit at run time. The notes record that pooled models fail here, because machines respond in opposite directions to the same lag.

---

## 3. The method in full detail

### 3.1 Inputs and history format

- `history.csv` has columns `episode_id, machine_id, tick, payout`. Queries come from `queries.csv`.
- The forecaster sees a per-machine dict `seq = {tick: payout}`. The base class builds it for each query from history that has `tick < cutoff` (the cutoff guard, §10).
- **Lags are tick lags, not row offsets.** The lag-`k` value for a cutoff `c` is `seq[c - k]`. If any of the 8 ticks is missing, that row is unusable. Missing ticks are unknown, never a loss. The short-history machines have many missing ticks, so this matters.

### 3.2 Features: `_lag_vector` and `feature_matrix`

**Step 1, binary lag vector.** `_lag_vector` (in `pair_logistic.py`, line 27) returns an 8-vector with `1.0` if the payout at that lag is positive (a win), else `0.0`. It returns `None` if any of the 8 lag ticks is missing.

**Step 2, signed encoding.** Inside `feature_matrix` (`interaction_sweep.py`, lines 48–59), when `encoding == "signed"`, the 0/1 vector is mapped to `2x − 1`, so win = +1 and lose = −1. This happens *inside* `feature_matrix`. The input to `win_probability` is always 0/1. Do not pre-sign it, or the encoding gets applied twice.

**Step 3, interaction blocks.** For `size` from 2 to `degree`, the code takes every `size`-combination of the 8 columns (`itertools.combinations`, precomputed in `__init__` as `_interaction_indices`) and multiplies the columns together, vectorised over rows.

The feature vector is the concatenation of:

| Block | Count (8 lags) | Meaning |
|---|---|---|
| main lags | 8 | each lag's (signed) value |
| pairs (size 2) | 28 | product of two lags |
| triples (size 3) | 56 | product of three lags. Only when `degree=3` |

So feature dimension is 8 for degree 1, 36 for degree 2, and 92 for degree 3. The unit tests check those counts.

### 3.3 Encodings

- **`binary`** (the constructor default). Lag values are 0/1. A pair product `x_a · x_b` is 1 only when both are wins.
- **`signed`**. Lag values are ±1. A pair product `z_a · z_b` is +1 when the two lags **match** (both win or both lose) and −1 when they differ. This is an XOR/XNOR feature. See §6 for a worked example.

### 3.4 Degrees

- `degree=1`: main lags only. Equivalent in spirit to a per-machine linear model on the lags.
- `degree=2`: main lags plus all 28 pairs. **This is the winner.**
- `degree=3`: adds all 56 triples. Swept, and did not win. The notes flag a hierarchical or group-L1 version as a possible next step (§9).

### 3.5 Penalties

Both penalties use `sklearn.linear_model.LogisticRegression`. The branches in `win_probability` are at lines 99–105.

- **`penalty="l1"`** (the winner): `penalty='l1', solver='liblinear', intercept_scaling=100, random_state=0, max_iter=1000`.
  - liblinear handles the L1 penalty here. Its intercept is penalised unless you scale it. `intercept_scaling=100` makes the intercept large enough that the penalty does not drive it to zero.
  - `random_state=0` makes runs deterministic.
- **`penalty="l2"`** (the constructor default): `C=C, max_iter=1000`, using sklearn's default lbfgs solver.

`C` is the inverse regularisation strength, so a smaller `C` means a stronger penalty. The winner uses **C = 0.5**, which is much stronger than E1's C = 10.

### 3.6 Recency weights

Each training row at tick `t` is weighted by a half-life decay (lines 90–92):

```
age    = cutoff - 1 - tick          # 0 for the most recent pre-cutoff tick
weight = 2 ** (-age / half_life)    # 1.0 if half_life is None
```

The winner uses **half_life = 16**. That makes a spin 16 ticks back count half as much as the latest spin. Half-life `None` means every row counts equally.

### 3.7 Training rows and the fallback

For each query (lines 75–110, `win_probability`):

1. Build the query's own lag vector `xq` from `seq` at `cutoff`. If any lag is missing, **fall back**.
2. Loop over every tick in `seq` that is `< cutoff`. Keep the rows whose 8 lags are all observed. Label `y = int(payout > 0)`.
3. **Fall back** (return `None`) if there are fewer than `min_rows` rows (default 12), or if all labels are the same class.
4. Otherwise expand the training rows with `feature_matrix`, fit the regression, and predict the query's win probability.

A fallback returns `None`. The base class then uses `pooled_recent` (the kit's frequency baseline) for that query. `diagnostics['fallbacks']` counts these. The winner had **9 fallbacks across 375 fits** on tuning, and no `max_iter` hits.

### 3.8 How the win probability becomes a 5-vector

This happens in the inherited `PairLogisticForecaster.predict` (`pair_logistic.py`, lines 77–93):

```
pw    = clip(win_probability, clip, 1 - clip)     # clip = 1e-4 by default
pos   = pooled_recent positive part, normalised    # shape of payouts 1, 2, 5, 20
probs = [1 - pw, pw * pos]                         # then renormalised
```

So E6 only changes the win chance. The payout shape is still the kit's `pooled_recent`. E5 explored replacing it (§9).

### 3.9 The `win_probability` override

`InteractionSweepForecaster` overrides exactly one method from the base class, `win_probability(seq, cutoff)`. It returns `float` or `None`. The base `predict` calls it through `self`, so the override is used automatically.

It also overrides `predict` (lines 66–73). That override resets `self.diagnostics` to zero and then calls `super().predict`. So **diagnostics describe only the most recent `predict` call**. They are not reset by `fit`.

`fit` is inherited unchanged. It only fits the `pooled_recent` fallback baseline on the labelled train data. The per-machine regression is refit inside `predict`, once per query, from that machine's history.

---

## 4. Constructor parameters

Signature: `InteractionSweepForecaster(degree=2, encoding='binary', family='all', lags=8, C=3.0, half_life=32.0, min_rows=12, base_half_life=24.0, clip=1e-4, penalty='l2')`

**Important:** the defaults are **not** the winning config. `encoding` defaults to `'binary'` and `penalty` to `'l2'`. Calling `InteractionSweepForecaster(degree=2)` gives the 0/1 L2 model, which is one of the weak ablations in §5. Always pass `encoding`, `penalty`, `C` and `half_life` explicitly.

| Parameter | Default | Winner | Allowed values | Effect |
|---|---|---|---|---|
| `degree` | `2` | `2` | `1`, `2`, `3` (else `ValueError`) | Interaction order. 1 = main lags, 2 = + pairs, 3 = + triples. |
| `encoding` | `'binary'` | `'signed'` | `'binary'`, `'signed'` (else `ValueError`) | 0/1 vs ±1 lag coding before products. **Critical.** |
| `family` | `'all'` | `'all'` | only `'all'` (else `ValueError`) | Reserved. Only the full product family is implemented. |
| `lags` | `8` | `8` | any int (base class sets `self.lags = int(lags)`) | Number of tick lags. The interaction indices are built from this. |
| `C` | `3.0` | `0.5` | any float > 0 | Inverse regularisation strength. Smaller = stronger penalty. |
| `half_life` | `32.0` | `16` | float, or `None` for equal weights | Recency half-life in ticks. |
| `min_rows` | `12` | `12` | int | Minimum usable training rows before a fit is attempted. Below this, fall back. |
| `base_half_life` | `24.0` | `24.0` | float | Half-life for the `pooled_recent` fallback and payout-shape source. |
| `clip` | `1e-4` | `1e-4` | float | Win probability is clipped to `[clip, 1 - clip]` before shaping. |
| `penalty` | `'l2'` | `'l1'` | `'l1'`, `'l2'` (else `ValueError`) | L1 uses liblinear. L2 uses lbfgs. **Critical.** |

Bad values raise `ValueError` at construction, with messages such as `degree must be 1, 2, or 3` and `penalty must be 'l1' or 'l2'`.

---

## 5. Results and ablations

All numbers are points out of 60, unless marked as a loss ratio. Points come from `points = 60 / (1 + R)`, where `R` is the mean over the 4 conditions of (our error ÷ baseline error). Baseline = 30. Perfect = 60. Lower loss ratio is better.

As a check, the tuning ratios `.228, .259, .264, .194` give `60 / (1 + mean) = 48.53`. The audit ratios `.163, .358, .235, .224` give 48.19. Both agree with the logged scores to rounding.

### 5.1 Headline

| Model | Tuning | Audit | Where |
|---|---|---|---|
| Kit baseline `pooled_recent` | 31.06 | 30.57 | `main` |
| E1 pair logistic, C=10 (0/1, L2) | 36.22 | — | `avyukth/exp-e1-pair-logistic` |
| E11 flexible experts (0/1 learners) | 40.26 | 40.90 | `avyukth/exp-e11-flex-experts` |
| **E6 signed-L1 (this handout)** | **48.54** | **48.19** | `avyukth/exp-e6-interactions` |
| E7 neural hypernetwork | 50.18 | 48.99 | `avyukth/exp-e7-neural-context` |
| Final: 50/50 blend E6 + E7 | 50.65 | 50.02 | `avyukth/entry-final` @ `588a70b` |

### 5.2 Per-condition loss ratios (lower is better)

| Split | stable | short | drift | shared |
|---|---|---|---|---|
| Tuning | .228 | .259 | .264 | .194 |
| Audit | .163 | **.358** | .235 | .224 |

Short history is the weak spot. It is the only condition where audit is far above the rest, and it has only about 30 rows with all 8 lags observed.

### 5.3 Ablations

Same `C=0.5` and `half_life=16`, one thing changed each time.

| Variant | Tuning | Audit | What it shows |
|---|---|---|---|
| **Signed + L1** (winner) | **48.54** | **48.19** | |
| 0/1 coding + L1 | 32.43 | 32.04 | Signed coding is required. |
| Signed + L2 | 38.70 | 40.79 | L1 sparsity is required. |

The conclusion from the ablation: **signed coding AND L1 are both required.** Each one alone recovers only part of the gain. Signed + L2 keeps every feature, so its predictions are blurred. 0/1 + L1 shrinks away the features it needs, because the rule needs several 0/1 coefficients (§6).

### 5.4 The sweep

- **Search space:** encoding (2) × degree (1, 2, 3) × penalty (L1, L2) × C × half-life.
- **Size:** 96 candidate configs. The Codex report splits them into 84 broad-grid candidates and 12 focused L1 refinements.
- **Selection:** on tuning only. The winner was **C=0.5, half-life 16, signed, degree 2, L1**.
- **Fit diagnostics for the winner:** 375 fits, 9 fallbacks, 0 `max_iter` hits.
- **Bootstrap (Codex, exploratory):** versus E2's best (36.22), E6 is +12.31 points, 95% episode-bootstrap interval [+10.08, +14.57]. Versus fixed E1 at C=3 / half-life 32 (35.41), E6 is +13.13, interval [+10.97, +15.31]. These intervals are exploratory, because 96 E6 configs were compared on the same tuning episodes.
- **Runtime:** under 1 second for all of validation. The packaged E6-only entry took 4.95 s on its own run, and the checker passed with deterministic output.

### 5.5 Where E6 sits against E7

E6 and E7 make different errors, so the 50/50 blend beats both. Codex measured E7 minus E6 on tuning as +1.65, with a paired interval of [−0.06, +3.27]. That interval crosses zero, so E7's lead is uncertain. E6 is the simpler and more transparent backup.

---

## 6. Why it works: a small worked example

### 6.1 The sign trick

Take two lags, `a` and `b`. Use win = 1 and lose = 0 in binary, and win = +1 and lose = −1 in signed.

| Lag a | Lag b | Binary `x_a·x_b` | Signed `z_a·z_b` | Match? |
|---|---|---|---|---|
| win | win | 1 | +1 | yes |
| lose | lose | 0 | **+1** | yes |
| win | lose | 0 | −1 | no |
| lose | win | 0 | −1 | no |

In the signed column, the product is +1 whenever the two lags **match** and −1 whenever they differ. A single coefficient on that one feature can express "this machine is live when lags a and b match".

In the binary column the product is 1 only for (win, win). It cannot express "lose, lose" by itself. Expressing equality with 0/1 values needs three terms:

```
match(x_a, x_b) = 1 - x_a - x_b + 2 * x_a * x_b
```

Check it: (0,0) gives 1, (0,1) gives 0, (1,0) gives 0, (1,1) gives 1. So the equality rule needs the product plus two main effects. L1 pays a cost for each nonzero coefficient, so it tends to drop some of these terms, and the rule does not come through.

### 6.2 A concrete feature vector

Take a machine's 8 lags, with lag 1 the most recent, in binary: `x = [1, 0, 1, 1, 0, 0, 1, 0]`. The signed version is `z = 2x − 1 = [1, −1, 1, 1, −1, −1, 1, −1]`.

The three global pairs flagged in the notes, written in 1-indexed lag numbers, are (1,3), (2,5) and (4,7). Their signed products are:

- lags 1 and 3: `z1 · z3 = (+1)(+1) = +1`, so the two match.
- lags 2 and 5: `z2 · z5 = (−1)(−1) = +1`, so they match.
- lags 4 and 7: `z4 · z7 = (+1)(+1) = +1`, so they match.

For this window each of those products is a clean ±1 signal. A machine whose live rule is "lags 1 and 3 match" would put a large weight on one feature, and L1 would leave the others at zero. (The code's column index is 0-based, so lag `k` is column `k−1`. Check the indexing before writing any code that uses these pairs. See §10.)

### 6.3 Why L1 and signed together

- Signed alone with L2: every pair gets a small weight, so the model is blurry. Score 38.70 / 40.79.
- 0/1 with L1: the rule needs three coefficients, and L1 shrinks them out. Score 32.43 / 32.04.
- Signed with L1: each machine finds a few sign-sensitive rules, and the others go to zero. Score 48.54 / 48.19.

The notes summarise this as a per-machine sparse search over "these two spins match" rules.

---

## 7. Code map

### 7.1 `mainFiles/slot_forecasting/interaction_sweep.py` (E6 branch)

| Lines | Symbol | What it does |
|---|---|---|
| 1–6 | module docstring | Summary of degrees and encodings. |
| 7–13 | imports | `combinations`, `warnings`, numpy, `LogisticRegression`, `PairLogisticForecaster`, `_lag_vector`. |
| 16 | `class InteractionSweepForecaster(PairLogisticForecaster)` | The E6 forecaster. |
| 19–21 | `__init__` signature | Parameter list (§4). |
| 23–25 | `super().__init__(...)` | Base class gets `pairs=False`, because E6 owns its own feature expansion. |
| 26–29 | attribute assignments | `degree`, `encoding`, `family`, `penalty`. |
| 30–37 | validation | Raises `ValueError` for bad `degree`, `encoding`, `family` or `penalty`. |
| 38–41 | `_interaction_indices` | Precomputed `combinations(range(lags), size)` for each size in 2..degree. |
| 42–46 | `diagnostics` | Counters: `successful_fits`, `max_iter_hits`, `fallbacks`. |
| 48–59 | `feature_matrix(X)` | Vectorised expansion. Applies signed transform, then concatenates main lags and product blocks. |
| 61–64 | `feature_vector(x)` | One row through `feature_matrix`. Used for the query. |
| 66–73 | `predict(queries, history)` | Resets diagnostics, then calls `super().predict`. |
| 75–110 | `win_probability(seq, cutoff)` | The per-machine fit (§3.7). Returns a probability or `None`. |
| 77–79 | query lag check | Fallback if the query's own lags are missing. |
| 82–92 | training row loop | Cutoff guard at line 83. Recency weights at lines 90–92. |
| 93–95 | minimum-data check | Fallback if `< min_rows` or one class only. |
| 97–98 | warnings suppressed | sklearn convergence warnings are silenced inside the fit. |
| 99–105 | penalty branch | liblinear L1 (lines 100–103) or lbfgs L2 (line 105). |
| 106 | `model.fit(...)` | Uses `sample_weight` from recency. |
| 107–109 | iteration diagnostics | Counts successful fits and `max_iter` hits. |
| 110 | return | `predict_proba(...)[0, 1]` on the query's signed-expanded vector. |

### 7.2 `mainFiles/slot_forecasting/pair_logistic.py` (E1 branch, the base class)

| Lines | Symbol | What it does |
|---|---|---|
| 27–31 | `_lag_vector(seq, tick, lags)` | 0/1 win indicators at lags 1..L, or `None` if any is missing. |
| 34–38 | `_expand(x, pairs)` | E1's own pair expansion. Not used by E6, which is built with `pairs=False`. |
| 41 | `class PairLogisticForecaster` | Base class. |
| 42–47 | `__init__` | Sets `lags`, `pairs`, `C`, `half_life`, `min_rows`, `clip`. Creates `self.base = FrequencyForecaster("pooled_recent", half_life=base_half_life)`. |
| 49–51 | `fit` | Fits only the `pooled_recent` base. Per-machine models are not fitted here. |
| 53–75 | `win_probability` (E1 version) | E6 replaces this entirely. |
| 77–93 | `predict` | Calls `self.win_probability` per query. Combines with the payout shape (§3.8). |

### 7.3 `tests/test_interaction_sweep.py` (E6 branch)

| Lines | Test | What it checks |
|---|---|---|
| 13–16 | `setUp` | Synthetic 100-tick seq with a periodic win rate. |
| 18–27 | `test_feature_shapes_for_each_degree` | Dimensions 8 / 36 / 92. Signed degree-1 output equals `2x − 1`. |
| 29–49 | `test_vectorized_features_match_independent_combination_products` | Batch expansion equals a row-by-row product over `combinations`, for all degrees and encodings. |
| 51–58 | `test_degree_two_binary_matches_e1_prediction` | E6 binary degree-2 equals E1's pair logistic at 14 decimal places. Parity check. |
| 60–65 | `test_current_and_future_rows_do_not_change_prediction` | Adding ticks at or after the cutoff does not change the prediction. Cutoff guard. |
| 67–80 | `test_predict_returns_valid_distribution` | Output is 5 finite values in [0,1] summing to 1. One successful fit, no fallbacks. |
| 82–98 | `test_l1_predict_returns_valid_probability_vector` | Same checks for the L1 path. |
| 100–102 | `test_unsupported_penalty_is_rejected` | `penalty='elasticnet'` raises `ValueError`. |

The PR reports 7 tests passing. Run the tests from the repo root, as in §8.

---

## 8. How to run and score it

### 8.1 Environment

- Python: `/opt/miniconda3/bin/python3` (3.13, sklearn 1.7.2, numpy 2.3.3).
- Run scoring from `mainFiles/`. The `public/` data folder is resolved relative to that directory, as `exp_eval.py` does.
- Run the unit tests from the **repo root**, using the PR's command: `python3 -m unittest discover -s tests -v`. The test imports are `mainFiles.slot_forecasting...`, so the repo root must be on the path.

### 8.2 Scoring with the kit helpers

`exp_eval.py` has no E6 entry in its `CANDIDATES` dict. Do not edit that file to add one. Build the model directly, using the same split and scorer the kit uses:

```python
# Run from mainFiles/ with /opt/miniconda3/bin/python3.
from pathlib import Path

from slot_forecasting.experiments import episode_folds, subset_score
from slot_forecasting.interaction_sweep import InteractionSweepForecaster
from slot_forecasting.io import (load_inputs, read_labels, read_rows,
                                 read_submission, read_targets)
from slot_forecasting.rounds import round_metadata

public = Path("public")
max_points = round_metadata(public)["max_points"]
tq, th = load_inputs(public / "train")
vq, vh = load_inputs(public / "validation")
labels = read_labels(public / "train/labels.csv", tq)
targets = read_targets(public / "validation/targets.csv")
groups = {r["query_id"]: r["condition"]
          for r in read_rows(public / "validation/conditions.csv",
                             ("query_id", "condition"))}
reference = read_submission(public / "validation/reference.csv")
tuning, audit = episode_folds(targets, groups)

model = InteractionSweepForecaster(degree=2, encoding="signed", penalty="l1",
                                   C=0.5, half_life=16)
pred = model.fit(tq, th, labels).predict(vq, vh)

result = subset_score(pred, targets, groups, reference, tuning, max_points)
print(result["technical_points"])           # expect 48.538
print({c: v["loss_ratio"] for c, v in result["by_condition"].items()})

# Audit: ONLY after you have decided this is a final look, and log it in
# worktrees/_coord/RESULTS.md. See section 10.
# audit_result = subset_score(pred, targets, groups, reference, audit, max_points)
```

The keys `technical_points` and `by_condition` (with `loss_ratio`) are the ones `exp_eval.py` reads, so this output matches the kit's own printout.

### 8.3 Scoring with `episode_folds` and `subset_score` directly

The snippet above is the direct path. `episode_folds(targets, groups)` returns two sets of episode IDs, `(tuning, audit)`. It sorts each condition's episodes by SHA-256 of the ID, and alternates them. So it gives 12 tuning and 12 audit episodes per condition. `subset_score(pred, targets, groups, reference, episodes, max_points)` restricts to the given episode set, then runs the kit's contest scorer.

### 8.4 Using the predictions directly

```python
# Predict one query from a raw 0/1 lag window.
model = InteractionSweepForecaster(degree=2, encoding="signed", penalty="l1",
                                   C=0.5, half_life=16)
seq = {tick: payout for tick, payout in history_for_one_machine.items()
       if tick < cutoff}                     # the cutoff guard is inside win_probability too
p_win = model.win_probability(seq, cutoff)   # float, or None for fallback
```

`win_probability` expects `seq` to be a `{tick: payout}` dict for a single machine. `predict` builds this dict from the history for you, so most people will not call it directly.

### 8.5 Packaging the entry

The E6-only entry is `avyukth/entry-main` @ `dbb6e38`. Its build files are in `mainFiles/runs/<entry>/`, as `entry.zip` and `submission.csv`. Use `build_entry.py --source <dir>` and `check_submission.py` to rebuild and check it, as described in the entry notes.

---

## 9. Ideas to build on

Each idea has a first experiment. Score on tuning first. Audit is already partly spent (§10).

### 9.1 Restrict to the 3 global pairs, or use them as a prior

**Why.** The later finding is that the true pairs are global: **(1,3), (2,5) and (4,7)**. E6 searches all 28 pairs for every machine, so it has 25 noise features per machine. On short history that noise is expensive.

**First experiment.** Build a variant that keeps the 8 main lags and adds only these 3 signed products. Use the same `C=0.5`, `half_life=16`, `penalty="l1"`. Compare to 48.538 tuning.

**Prior version.** Keep all 28 pairs, but make the 3 global ones cheaper to use. In a liblinear L1 fit, scaling a feature column by `s` multiplies that feature's effective penalty by `1/s`. So scaling the three global-pair columns by `s = 2` halves their penalty (cheaper to use), and leaves the other columns unchanged. This is the simplest way to express "prefer these". It is a derivation, not a tested result, so check the coefficients after fitting.

**Caveat.** Confirm the pair indexing before coding (§10). The notes write these in 1-indexed lags.

### 9.2 Per-episode memory (our E4) and per-machine memory (E2)

**Why.** E6 uses a fixed half-life of 16. E2 showed per-machine half-life choice helps (+0.93 at matched C). E4 showed an episode-pooled half-life choice helps more (combo 37.96 to 39.04). Neither has been applied to E6 yet.

**First experiment.** Reuse E4's selection: for each half-life in `{None, 64, 32, 16}`, run the fast prequential fit on each machine's last 48 spins (every 3rd spin), sum the Brier loss across the episode's machines, and pick the best. Use that half-life for the whole episode. The notes say soft weighting was no better than a hard pick.

**Files.** `episode_memory.py` (`_episode_losses`) on `avyukth/exp-e4-episode-memory`. `adaptive_memory.py` on `avyukth/exp-e2-adaptive-memory`.

**Caveat.** The selection must use only pre-cutoff spins. Check the cutoff guard, as in §10.

### 9.3 C adapted to row count, for short history

**Why.** Short history is the weak spot (audit .358). A machine with about 30 usable rows and 36 features is badly underdetermined. The notes list "adapt E6's C to the row count" as an option.

**First experiment.** For machines with fewer than some row threshold, use a stronger penalty, meaning a smaller C. Start with a grid on tuning only, and check the short-history loss ratio. Keep the stable and drift settings unchanged, so you can see the effect in isolation.

**Caveat.** A stronger penalty on short history may also drop the one or two rules that do exist there. Look at the short-history ratio, not just the total.

### 9.4 Degree 3 with group-sparse or hierarchical L1

**Why.** Plain degree 3 did not win the sweep. But a triple should only be used when its parent pairs are present. That is a group or hierarchical penalty, and plain L1 does not express it.

**First experiment.** Approximate the hierarchy with column scaling (§9.1). Keep the 36 degree-2 columns at the default scale, and scale the 56 triple columns by `s < 1` to make them more expensive. Sweep `s` on tuning. This is an approximation, because it is not a true group penalty.

**Caveat.** Degree-3 search is expensive, and the notes say the winner was degree 2. Keep this to a small grid.

### 9.5 Payout-size head (E5)

**Why.** E6 still uses `pooled_recent`'s positive part for the payout shape. The notes flag that now that the win-chance error is small, the shape share of the remaining error is larger.

**First experiment.** Swap in E5's `ShapeHeadForecaster.shape()` from `shape_head.py` on `avyukth/exp-e5-shape-head` for the positive part. Then decompose the remaining error: oracle `p0` with model shape, versus model `p0` with oracle shape. That shows which half is worth working on.

**Caveat.** E5 alone moved from 36.22 to 36.45 on tuning. It was small, so the gain here may be small too.

### 9.6 Other ideas from the notes

- **E7 retrain for round 2.** E6 refits per machine at run time, so it adapts without retraining. E7 must be retrained on round 2's train split.
- **Learned blend gate.** Choose the E6/E7 weight from observable features, such as history length or chosen half-life, instead of a fixed 50/50. Fit it on train predictions, then confirm on tuning.

---

## 10. Gotchas

- **The cutoff guard.** `win_probability(seq, cutoff)` must ignore every tick `>= cutoff`. The history files stop before the cutoff, but prequential callers pass full prefixes. E6 has the guard at line 83. The test at lines 60–65 checks it. Any new variant must keep this guard, or it will train on the spin it is predicting.
- **Tick lags, not row offsets.** Use `seq.get(t - lag)`, which is what `_lag_vector` does. Rows with any missing lag are skipped. Imputing them did not help in the 0/1 model (E11).
- **Do not pre-sign the input.** `win_probability` and `_lag_vector` produce 0/1. The signed transform is inside `feature_matrix`. Applying it twice gives `2(2x − 1) − 1`, which silently breaks the model.
- **Defaults are not the winner.** `encoding='binary'`, `penalty='l2'`, `C=3.0` and `half_life=32` are the defaults. Always pass the full winning config (§4).
- **Diagnostics reset per `predict`, not per `fit`.** Read `model.diagnostics` right after `predict`, and only then.
- **The audit budget.** The audit half is only worth something as a fresh look. E6's audit has already been looked at once, in the E6 audit row, and the blends were scored on audit too. Log every audit look in `worktrees/_coord/RESULTS.md`. Do not run audit for sweeps. `exp_eval.py` only runs audit with `--audit`.
- **Contamination.** Early diagnostics (`_coord/diag.py`, `diag2.py`) looked at all validation targets, audit included. That informed E1's pairs, C and half-life. Treat audit as strong evidence, not a blind holdout.
- **Pair indexing.** The notes write the global pairs as (1,3), (2,5), (4,7). The code's columns are 0-based (column 0 is lag 1). Confirm which convention the notes use before writing code that selects these columns.
- **Branch naming.** A local branch named `avy` exists, so `avy/...` branch names fail. Use `avyukth/...`. One branch per experiment: `avyukth/exp-<id>-<desc>`.
- **Stash.** The git stash is shared across worktrees. Do not use a bare `git stash`.
- **macOS and torch.** Importing torch after sklearn can crash with `OMP: Error #15` on conda. Set `KMP_DUPLICATE_LIB_OK=TRUE` before the imports. E6 alone does not import torch, but E7 does, and `predict.py` sets it for you.
- **liblinear L1 intercept.** E6 uses `intercept_scaling=100`, so the intercept is not penalised away. Changing it changes the model.
- **Runtime.** E6 takes under a second for all of validation. The final E6+E7 entry takes about 8 s. E11 takes about 4–5 minutes, so E6 is cheap to sweep.
- **Submission limits.** Python 3.13 base with pinned requirements. No network at prediction time. 30 minutes, 2 CPUs, 16 GiB. The ZIP opens to `README.md` and `reproduce.json`. The command is `python predict.py --data /data --out /output/submission.csv`. The CSV must be at most 10 MB and the ZIP at most 100 MB.
- **Draft status.** PR #11 is a draft based on the E1 branch. Nothing in the E1–E11 line is merged to `main`.

---

## Appendix: sources

- Code: `origin/avyukth/exp-e6-interactions:mainFiles/slot_forecasting/interaction_sweep.py` and `:tests/test_interaction_sweep.py`.
- Base class: `origin/avyukth/exp-e1-pair-logistic:mainFiles/slot_forecasting/pair_logistic.py`.
- Scorer and split: `origin/avyukth/exp-e6-interactions:mainFiles/slot_forecasting/experiments.py` (`episode_folds`, `subset_score`) and `mainFiles/exp_eval.py`.
- Results: `HOUSE_EDGE_EXPERIMENT_RESULTS.md` (E6 sections, §5–§8), `worktrees/_coord/CODEX_REPORT.md`, `worktrees/_coord/RESULTS.md` (E6 sweep rows, ablation rows and the audit row).
- PR: `gh pr view 11` (title and body).
