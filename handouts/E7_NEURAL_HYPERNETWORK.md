# E7: Neural Hypernetwork Win Head (teammate handout)

**Branch:** `avyukth/exp-e7-neural-context` · **PR:** #12 (draft, based on E1) · **Scored at:** `4e0267a`, artifact commit `af645aa`
**Code:** `mainFiles/slot_forecasting/neural_context.py` (368 lines) · **Tests:** `tests/test_neural_context.py` (6 tests)
**Shipped model:** `origin/avyukth/entry-final:mainFiles/entry/models/state.pt`
**Local artifact:** `worktrees/codex-exp/mainFiles/runs/e7-best-artifact/` (`state.pt`, `state.pt.json`, `config.json`)

Read this with `HOUSE_EDGE_EXPERIMENT_RESULTS.md` (§1 bottom line, §3 data facts, §4 protocol). This handout covers E7 only.

---

## 1. TL;DR

- E7 predicts each machine's next-spin probabilities over payouts {0, 1, 2, 5, 20}. Almost all of the error is the **win chance** (the probability that payout > 0). E7 changes only that part.
- The win chance is modelled as a logistic over 36 signed lag features (8 lags and 28 pairwise products). The **coefficients are not fitted per machine at run time**. A small MLP reads each machine's causal history summary and outputs the coefficients. It is trained once, on TRAIN machines, and then frozen.
- **Score:** 50.18 tuning, 48.99 audit (points out of 60; 60 is perfect, 30 is the frozen baseline).
- **Best single model we have.** Its tuning lead over E6 (48.54) is **+1.65 with a paired 95% CI of [−0.06, +3.27]**. That interval crosses zero, so the lead is not established. E7 is clearly better than E6 on the short-history condition (audit .238 vs .358).
- **The final entry is a 50/50 blend of E6 and E7** (50.65 tuning / 50.02 audit). E7 alone is not the submitted model.
- The trained weights are a file: `NeuralContextForecaster.load_artifact(path)` restores them with **no refit** (fresh-process replay max diff 2.2e-16).
- **Round 2 requires retraining E7** on the round-2 TRAIN split. Do not reuse the shipped weights blindly.
- Most promising next experiment: give E7 only the statistics of the 3 global parity pairs (1,3), (2,5), (4,7), or add them as inputs (see §10).

---

## 2. Intuition: a "learned estimator"

**The problem in one line.** For one machine, we have a few hundred past spins. Each spin's win depends on a short rule over the last 8 outcomes, such as "lag 1 and lag 3 matched". Each machine has its own rules, sometimes with the opposite sign to other machines. So we cannot pool the machines, and a single machine has too few rows to fit many coefficients reliably. This is especially bad for short-history machines (about 30 usable rows).

**E6's approach.** For each machine, run an L1-penalised logistic regression at query time. That works well with enough data, but on a short-history machine it has too few rows, and the sparse search is noisy.

**E7's approach.** Instead of fitting a logistic per machine, **learn the fitting procedure itself**. Think of it like this:

1. Take a machine's past rows and summarise them as a few numbers: how often each feature is on, and how strongly each feature is correlated with winning. These are "sufficient statistics" of the history so far.
2. Show the network thousands of (machine history summary, next spin) pairs from TRAIN machines. Each pair has a known answer.
3. The network learns a function from "this summary" to "the coefficients that would predict this machine's next spin well". Across many train machines it discovers that a machine whose feature/outcome correlation is strong should get sharp coefficients, and a noisy one should get small coefficients.
4. At test time there is no fitting. The network reads the summary of a new machine's history and emits coefficients in one forward pass.

This is called amortised or meta-learned estimation. The network is the shared "estimator", and each machine's history is the input. Because the mapping is learned, short histories get sensible, shrunk coefficients rather than noisy ones. That is why E7 beats E6 most on short history.

**What it does not do.** It has no machine ID, episode ID, signal, or condition inputs. Renaming a machine does not change its prediction (checked by a test). It also does not learn the payout shape. The payout-shape part is inherited from the E1 base (§3.6).

---

## 3. Architecture in full

### 3.1 Per-row features (`_query_features`, L61-65; `_signed_lags`, L57-59)

For each usable row (a tick whose 8 lags are all observed) we form:

| Piece | Definition | Count |
|---|---|---|
| Signed lags | `2·x − 1` where `x_k = 1` if the spin `k` ticks ago paid out > 0, else 0 | 8 |
| Signed pairs | `signed[a]·signed[b]` for each `a < b` in `combinations(range(8), 2)` | 28 |
| **z (total)** | concatenation of the above | **36** |

Signed coding matters. A product of two signed lags is +1 when the two lags match and −1 when they differ, so one coefficient can express an equality rule. With 0/1 coding it takes several coefficients, and L1 shrinks those away. This is the core reason E6 beat 0/1 models by about 8 points, and E7 uses the same encoding.

The target is `y = 1` if that row's spin paid out > 0, else 0.

Lag indexing: `_lag_vector` (in `pair_logistic.py`) builds `x[k]` from `seq[tick − (k+1)]`, so **`x[0]` is lag 1**. The feature `signed[a]` is therefore lag `a+1`.

### 3.2 Row extraction (`_sequence_rows`, L67-80)

Takes a `{tick: payout}` dict, sorts the ticks (optionally keeps only ticks `< cutoff`), and keeps only ticks where `_lag_vector` returns a full 8-lag vector. Missing ticks are unknown, not losses, so any row with a missing lag is dropped. It returns `(Z, y, kept_ticks)`.

### 3.3 Prefix sufficient statistics (`_prefix_summaries`, L82-124)

For a sequence of `n` usable rows, the function returns one statistics vector per **prefix**. Prefix `i` is "the first `i` rows", so it never contains row `i` itself. The statistics are:

| Block | Formula | Width |
|---|---|---|
| mean of z | `mean(z)` over the prefix | 36 |
| signed correlation | `mean((2y−1)·z)` over the prefix | 36 |
| win rate | `mean(y)` over the prefix | 1 |
| log count | `log1p(count) / log(257)` (scaled to about [0, 1] for typical counts) | 1 |
| **All-history block** | | **74** |

If `recent_half_life` is set, a second 74-wide block is appended. It uses the same four quantities but **recency-weighted** with weight `2^((tick − anchor)/half_life)`. Weights are anchored at the first tick of the sequence, and the final "log count" uses the mass relative to each prefix's own query tick. This keeps every prefix independent of rows after it (see §4).

With `recent_half_life = 16` (the shipped setting), the statistics vector is **74 + 74 = 148** wide.

### 3.4 The hypernetwork (`fit`, L187-193)

```
input:  stats  (148)            # all-history 74 + recent 74
Linear  148 -> 64
ReLU
Linear  64  -> 37               # 36 coefficients + 1 bias
```

- The **last layer is zero-initialised** (L192-193), so at the start every machine gets logit 0 (p = 0.5 on the win head) and training starts from a neutral point.
- The hidden width of 64 is hard-coded, not a constructor argument.
- Parameter count, derived from the shapes: 148·64 + 64 + 64·37 + 37 = **11,941**.

### 3.5 The prediction

```
coef, bias = network(stats)            # coef: 36 values, bias: scalar
logit      = coef · z_query + bias     # z_query: the 36 features of the current query
p_win      = sigmoid(logit)
```

`win_probability` (L267-283) clips `p_win` to [`clip`, 1 − `clip`] (`clip = 1e-4`).

### 3.6 Shape table

| Object | Shape / value | Where |
|---|---|---|
| query z | 36 | `_query_features` |
| prefix stats per machine | 148 (74 all-history + 74 recent) | `_prefix_summaries` |
| network input | (batch, 148) | L211 |
| network output | (batch, 37): `output[:, :-1]` = coefs (36), `output[:, -1]` = bias | L212 |
| logit | (batch,) = `(output[:, :-1] * z).sum(1) + output[:, -1]` | L212 |
| win probability | scalar per query | L282 |
| 5-vector payout distribution | win head gives P(win); the rest comes from the inherited E1 payout-shape head | inherited (`pair_logistic.py`) |

**Payout shape is not learned by E7.** The inherited `PairLogisticForecaster` base supplies the conditional shape of payouts given a win. The artifact stores its priors (`pooled_recent` method, prior strength 20, keyed by `machine_type|signal` such as `balanced|-1`). Note that the **shape priors are keyed by `signal`**, which is per query only (see §11). E7's win head ignores `signal` and `machine_type`.

---

## 4. Training data construction and leakage safety

### 4.1 How the training examples are built (`_make_training_examples`, L139-160)

1. For each unique query (one per episode and machine, via `unique_queries`), take that machine's **TRAIN history**, the full `{tick: payout}` dict for the machine.
2. Extract the usable rows `Z, y, ticks` with `_sequence_rows` (no cutoff, so all TRAIN ticks).
3. Build the prefix statistics for every row with `_prefix_summaries`, anchored at the query's cutoff for the final prefix (`terminal_query_tick=query.cutoff`).
4. Candidate example indices are `min_context` (16) up to `len(Z) − 1`, so each example has at least 16 prior rows. If there are more than `examples_per_machine` (24) candidates, pick 24 **at random without replacement** (`np.random.default_rng(seed)`), then sort them.
5. Each example is `(prefix stats at i, z_i, y_i)`: the statistics before row `i`, the features of row `i`, and its outcome.
6. Machines with `len(Z) <= min_context` are skipped.

Result on the shipped run: **12,561 train examples and 4,365 holdout examples.** Examples from the same machine overlap heavily, so they are not independent observations (CODEX_REPORT notes this too).

### 4.2 Leakage rules and how they are enforced

| Rule | How it is enforced | Where |
|---|---|---|
| A statistic for row `i` uses only rows before `i` | prefix `i` is built from the first `i` rows only; test checks row-level equality | `_prefix_summaries` L82-124; test L55-72 |
| Recency weights do not depend on later rows | weights anchored at `ticks[0]`, and each prefix's query-scaling uses its own tick | L99-123 |
| Query prediction ignores current and future rows | `win_probability` drops ticks `>= cutoff` before doing anything | L268; test L42-53 |
| Machine and episode names are not inputs | features come only from payouts; renamed machines give identical output | test L74-88 |
| Episode-hash holdout, 1 in 5 | `sha256(episode_id)[:4]` as a big-endian int, `% 5 == 0` means holdout | `_episode_is_holdout` L134-137 |
| Only TRAIN data is used | `fit` receives only the train queries and history; the holdout is a slice of TRAIN episodes | `fit` L162-180 |
| No validation or audit data | nothing in the code path reads validation or audit files; those splits were used only for scoring | — |

The holdout is **internal**. It is used only for early stopping and model selection among training epochs. It is not the kit's tuning or audit split. The holdout episodes are never trained on.

### 4.3 The fallback holdout (`fit`, L169-176)

If the training set has one episode, or if either the train or holdout list is empty, `fit` switches to a different deterministic split. It sorts episodes by `sha256(episode).hexdigest()` and holds out only the first one. This is **not** the 1-in-5 rule. It rarely triggers on the real data, but it matters for small synthetic tests (§11).

---

## 5. Training procedure and hyperparameters

### 5.1 Training loop (`fit`, L182-242)

- Sets `torch.set_num_threads(min(2, n))`, `torch.manual_seed(seed)`, and `np.random.seed(seed)` (L182-185). **These are global side effects.**
- Loss is `BCEWithLogitsLoss` on the logit above.
- Optimiser: Adam with `lr = 1e-3`.
- Each epoch: shuffle the train rows, run mini-batches of 256, then evaluate the holdout BCE with no gradients (L206-221).
- Keep the state dict with the best holdout BCE (`deepcopy`, L224). An improvement must beat the best by more than `1e-6`.
- Stop after `patience` (6) epochs with no improvement, or after `max_epochs` (50).
- Restore the best state (L229) and record diagnostics (L232-241).

Shipped run: **best epoch 13, stopped at epoch 19** (13 + patience 6), **holdout BCE 0.409** (0.408979).

### 5.2 Every constructor parameter

Defaults are from the `__init__` signature (L31-35). The "chosen" column is the shipped artifact, from `config.json` (`configuration` block).

| Parameter | Default | Chosen | Meaning |
|---|---|---|---|
| `recent_half_life` | 32.0 | **16.0** | half-life of the recency-weighted statistics block. `None` disables the block (stats width 74). **The class default is not what we shipped. You must pass 16.0.** |
| `min_context` | 16 | 16 | minimum usable rows before a prefix or query is eligible for E7. Below this, fall back to E1 |
| `examples_per_machine` | 24 | 24 | training prefixes sampled per machine |
| `max_epochs` | 50 | 50 | epoch cap |
| `patience` | 6 | 6 | early-stopping patience in epochs |
| `batch_size` | 256 | 256 | mini-batch size |
| `learning_rate` | 1e-3 | 1e-3 | Adam learning rate |
| `seed` | 2026 | 2026 | seeds torch, numpy, and the example sampler |
| `lags` | 8 | 8 | number of tick lags. Also sets the feature width (36) |
| `C` | 3.0 | 3.0 | passed to the inherited E1 base (`super().__init__`, L36-38). Meaning is defined in `pair_logistic.py`, which this handout does not cover |
| `half_life` | 32.0 | 32.0 | passed to the inherited E1 base; stored separately from `base_half_life` in the config |
| `min_rows` | 12 | 12 | passed to the inherited E1 base |
| `base_half_life` | 24.0 | 24.0 | passed to the base as its `half_life`; the artifact's `base_forecaster.half_life` is 24.0 |
| `clip` | 1e-4 | 1e-4 | probability clip in `win_probability` |

Fixed in code, not constructor parameters: hidden width 64, ReLU, zero-initialised last layer, BCE loss, Adam, holdout = episode hash mod 5, torch threads capped at 2.

### 5.3 Runtime

- Training (`fit` only) recorded **`training_seconds` = 4.16 s** in the shipped diagnostics.
- Loading the artifact and predicting the full validation set takes **about 2 s**.
- HOUSE_EDGE §7 says "retraining takes minutes on CPU". That is a rougher estimate that includes data loading and multiple seeds. Plan for seconds-to-minutes and measure it yourself.

---

## 6. Results

### 6.1 Headline

| Metric | Tuning (48 episodes) | Audit (48 episodes) |
|---|---|---|
| **E7 score (/60)** | **50.18** | **48.99** |
| Loss ratio, stable | .206 | .181 |
| Loss ratio, short_history | .198 | .238 |
| Loss ratio, drift | .220 | .233 |
| Loss ratio, shared_change | .158 | .246 |

Loss ratio is our error divided by the frozen baseline's error for that condition. Lower is better, and 1.0 is baseline. Score = 60 / (1 + mean ratio).

### 6.2 Comparisons

| Comparison | Result | Source |
|---|---|---|
| E7 vs E6 (tuning) | **+1.65**, paired 95% CI **[−0.06, +3.27]** | HOUSE_EDGE §5 and CODEX_REPORT (final paired comparison, 1,000 episode-bootstrap resamples) |
| E7 vs E6 (audit) | E6 audit 48.19 vs E7 audit 48.99 | HOUSE_EDGE §1 |
| 50/50 blend E6 + E7 | **50.65** tuning, **50.02** audit | HOUSE_EDGE §5 |
| 70/30 blend E6/E7 | 50.10 tuning, 49.62 audit | HOUSE_EDGE §5 |
| E7 vs E2 best (tuning) | +13.96, CI [+11.79, +15.92] | CODEX_REPORT |

What the numbers say:

- **E7 is the best single model on both halves**, but the tuning margin over E6 is within noise. Treat "E7 > E6" as a likely but unproven claim.
- **Short history is where E7 earns its place.** Audit short_history: E7 .238 vs E6 .358. E6 has too few rows for its per-machine L1 fit. E7's learned mapping does not.
- **E7 is weaker than E6 on audit for shared_change**, which is the condition that needs the episode view (E7 .246 audit). Neither model uses episode information yet.
- **Blending helps.** E6 and E7 make different errors, and the blend beats both on both halves. The final entry uses the blend.

### 6.3 Caveats on these numbers

- Tuning and audit were both seen by earlier diagnostic scripts (`diag.py`, `diag2.py`) that looked at all validation targets (HOUSE_EDGE §4). The audit score is strong evidence, not a blind holdout.
- E7's four settings and E6's 96 configurations were chosen on the same tuning episodes. Intervals are exploratory.
- The PR #12 body quotes an earlier E6 number of 48.214 and PyTorch 2.9.0. The final E6 score is **48.54** (CODEX_REPORT), and the requirements pin **torch 2.9.1**. Use the final values.

---

## 7. Saving and loading the artifact, and the fallback path

### 7.1 Saving (`save_artifact`, L303-330)

`save_artifact(path, metadata=None)` requires a fitted model and writes:

- `path` (a `torch.save` payload) with keys: `format_version` (= 1), `configuration`, `network_state`, `training_diagnostics`, `base_forecaster` (method, prior strength, half-life, global prior, and context priors), and `metadata`.
- `path + '.json'`: the same payload minus the weights, pretty-printed. This is the human-readable manifest (`state.pt.json`).

The `metadata` argument is free-form. The shipped artifact records `loader_commit`, `scored_training_commit`, input file SHA256s, and the tuning score.

### 7.2 Loading (`load_artifact`, L332-368)

```python
from mainFiles.slot_forecasting.neural_context import NeuralContextForecaster
model = NeuralContextForecaster.load_artifact("…/e7-best-artifact/state.pt")
probs = model.predict(queries, history)   # no refit
```

What it does:

1. `torch.load(..., map_location='cpu', weights_only=True)`. It falls back to the older `torch.load` signature only if `weights_only` is not supported.
2. Checks `format_version == 1`, and that `configuration` and `base_forecaster` are dicts.
3. Rebuilds the model with `cls(**configuration)`, recomputes the stats width (74, doubled to 148 if `recent_half_life` is set), rebuilds the network, and loads the weights.
4. Restores the base's `method`, `prior_strength`, `half_life`, `global_prior`, and `context_priors`.

Verification: a fresh-process replay reproduces the 384 saved tuning predictions with maximum absolute difference **2.2e-16**.

### 7.3 Fallback to E1

`win_probability` (L267-283) falls back to the inherited E1 pair-logistic (`super().win_probability`) in two cases:

1. The query's own 8-lag vector is incomplete (missing ticks), so `xq is None`.
2. The network is not loaded, or the usable row count is below `min_context` (16).

Note that the fallback is **per query**. A machine with 20 usable rows uses E7. A machine with 15 falls back to E1 for every query.

In the final entry, `predict.py` also falls back to E6 alone if torch fails to import (HOUSE_EDGE §5).

---

## 8. Retraining recipe (round 2)

Round 2 changes "which temporal relationships recur", so the round-2 TRAIN split may need a fresh fit. Retrain on the new TRAIN split and then save. Do not run this on validation or audit.

```python
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # macOS conda OMP fix; set BEFORE importing torch
from mainFiles.slot_forecasting.neural_context import NeuralContextForecaster

# Load TRAIN queries, history and labels the same way exp_eval.py does.
# (Not reproduced here. Check that file for the exact loader.)
train_queries, train_history, train_labels = load_train_split()

model = NeuralContextForecaster(
    recent_half_life=16.0,      # NOT the class default (32.0). This matches the shipped artifact.
    min_context=16, examples_per_machine=24, max_epochs=50, patience=6,
    batch_size=256, learning_rate=1e-3, seed=2026,
    lags=8, C=3.0, half_life=32.0, min_rows=12, base_half_life=24.0, clip=1e-4,
)
model.fit(train_queries, train_history, train_labels)
print(model.training_diagnostics)   # check best_epoch, holdout BCE, and example counts

model.save_artifact('mainFiles/runs/e7-round2/state.pt',
                    metadata={'experiment': 'E7', 'split': 'round2-train'})
```

Checklist for the retrain:

- [ ] Train inputs only. Confirm no validation or audit file is passed to `fit`.
- [ ] `training_diagnostics` shows `train_examples` and `holdout_examples` in the same ballpark as the shipped 12,561 and 4,365. Large differences mean the data changed.
- [ ] Check `best_epoch` against `epochs`. If `best_epoch` is at or near `max_epochs`, the run did not converge; raise `max_epochs`.
- [ ] Score the artifact on the round-2 **tuning** split with `exp_eval.py`. Log the audit look in `RESULTS.md` if you run it.
- [ ] Re-check the blend weight on round-2 validation (HOUSE_EDGE §7).
- [ ] Keep the old artifact. Do not overwrite `e7-best-artifact/`.

Runtime: `training_seconds` was 4.16 s for the shipped run (§5.3).

---

## 9. Code map (`mainFiles/slot_forecasting/neural_context.py`, 368 lines)

| Lines | Item | Notes |
|---|---|---|
| 1-7 | module docstring | "optional CPU experiment, not a required runtime dependency" |
| 8-19 | imports | `PairLogisticForecaster`, `_lag_vector` from `pair_logistic`; `index_history`, `unique_queries` from `schema` |
| 22 | `class NeuralContextForecaster(PairLogisticForecaster)` | inherits the E1 base |
| 31-35 | `__init__` signature | all 14 constructor parameters (§5.2) |
| 36-38 | `super().__init__(lags=lags, pairs=True, ...)` | base is configured with pairs on |
| 39-40 | `recent_half_life` coercion | `None` or a float |
| 41-45 | other settings | |
| 46-52 | validation | raises on non-positive values |
| 53-55 | `network`, `training_diagnostics`, `_torch` | all `None` until `fit` or `load_artifact` |
| 57-59 | `_signed_lags` | `2x − 1` |
| 61-65 | `_query_features` | 8 signed lags + 28 signed pairs = 36 |
| 67-80 | `_sequence_rows` | valid exact-lag rows, optional `cutoff` |
| 82-124 | `_prefix_summaries` | 74-wide all-history block, optional 74-wide recent block; prefix `i` excludes row `i` |
| 126-132 | `context_statistics(seq, cutoff)` | final prefix, `None` if below `min_context` |
| 134-137 | `_episode_is_holdout` | sha256 of episode id, mod 5 == 0 |
| 139-160 | `_make_training_examples` | per-machine sampling; returns `{False: train, True: holdout}` |
| 162-242 | `fit` | torch setup 182-185; network 187-193; optimiser 194-195; epoch loop 206-228; diagnostics 232-241 |
| 169-176 | fallback episode holdout | triggers only with one episode or an empty split |
| 244-265 | `_make_training_examples_with_episode_map` | duplicate of 139-160 that also returns episode IDs, used only by the fallback |
| 267-283 | `win_probability(seq, cutoff)` | filters ticks `>= cutoff`; falls back to E1; clips |
| 285-301 | `_artifact_configuration` | all constructor values, including `base_half_life` read from `self.base.half_life` |
| 303-330 | `save_artifact` | `.pt` payload plus `.json` manifest |
| 332-368 | `load_artifact` (classmethod) | no training; rebuilds the network from the config |

Test file `tests/test_neural_context.py` (132 lines):

| Lines | Item |
|---|---|
| 12-30 | `synthetic_training_data`: 10 episodes × 2 machines × 96 ticks, outcomes depend on the lag-1×lag-2 product |
| 34-40 | `setUpClass`: fits a small model (`max_epochs=4`, `patience=2`, `examples_per_machine=8`) |
| 42-53 | `test_prefix_statistics_and_prediction_ignore_current_and_future_rows` |
| 55-72 | `test_training_prefix_summary_ignores_current_and_future_outcomes` |
| 74-88 | `test_machine_identity_is_not_an_input` |
| 90-103 | `test_training_is_deterministic_for_a_fixed_seed` |
| 105-119 | `test_saved_artifact_round_trips_predictions_without_refitting` |
| 121-128 | `test_predict_returns_valid_five_probability_distribution` |

Run the tests from the repo root with `python -m unittest discover -s tests -v`, the command the PR body uses. The PR reports 6 passing. The test imports `mainFiles.slot_forecasting`, so run from the repo root.

---

## 10. Ideas to build on, each with a first experiment

Each idea is a change to `neural_context.py` plus a tuning-only run through `exp_eval.py`. Log every audit look in `RESULTS.md`. The expected outcomes below are hypotheses, not measured results.

### 10.1 Parity-cell statistics for the 3 global pairs (highest priority)

**Why.** Later analysis found that the true pairs are **global**, namely (1,3), (2,5) and (4,7), with XOR/XNOR rules. Across machines the same three pairs matter. E7 currently learns from all 36 features. Restricting it to these three cells could give a cleaner estimator, especially for short history.

**Indexing.** Lag numbers are 1-based (§3.1). The 36-column `z` vector puts the pair `(a, b)` (1-based lags) at column `8 + combinations(range(8), 2).index((a−1, b−1))`. The verified columns are:

| Pair (lags) | `z` column | All-history stats columns (`mean z` / signed) | Recent block (`mean z` / signed) |
|---|---|---|---|
| (1,3) | 9 | 9 / 45 | 83 / 119 |
| (2,5) | 17 | 17 / 53 | 91 / 127 |
| (4,7) | 28 | 28 / 64 | 102 / 138 |

Computed as `8 + i` for pair index `i` (the list is `combinations(range(8), 2)`). The recent-block offset is +74.

**First experiment (ablation, same seed).**
- Variant A: current 148-wide stats (control).
- Variant B: stats restricted to the 6 columns above (3 cells × mean and signed), plus the win rate and log count for both blocks.
- Variant C: keep 148 stats and add the 3 parity-cell features as extra input columns (no restriction).
- Compare holdout BCE and tuning score. Hypothesis: B helps short history more than the other conditions.

Check the lag indexing against `_lag_vector` before trusting the table. A wrong offset would silently test the wrong pairs.

### 10.2 Several recency half-lives

**Why.** Each condition prefers a different memory length (HOUSE_EDGE §3, §6). E7 only has one recent block (half-life 16). Adding more blocks lets the network choose.

**Suggested set.** Half-lives 8, 16, 32, 64, and all-history (the existing block).

**First experiment.** Generalise `_prefix_summaries` to a list of half-lives, and run {16} (control) against {8, 16, 32} on tuning. Compare the holdout BCE to see whether the extra blocks overfit.

**Gotcha.** Stats width changes, so `load_artifact` must rebuild the same width from the config. Bump `format_version` to 2 when you change the layout, and keep loading version 1 for the shipped artifact.

### 10.3 Episode summary input for shared_change

**Why.** `shared_change` is the weakest condition on audit (.246 for E7). Shared changes hit all machines in an episode at the same time, so the other machines' recent statistics carry information that a single machine's history does not.

**Design.** For each example at tick `t` in episode `e`, compute the mean of the 74-wide all-history stats across the other machines in `e`, using only ticks `< t`. This is the **same causal rule** as §4.2 applied across machines. Do not use the full episode history, or the future leaks in.

**First experiment.** Append the episode summary (74 more inputs) to the stats. Run on tuning, and compare `shared_change` loss ratio .158 (tuning) and .246 (audit) against the control. Check the holdout BCE separately, since the holdout is also episode-hashed and the episode summary is per episode.

### 10.4 Seed ensembling

**Why.** The shipped run is a single seed (2026). The network's variance across seeds is unknown. HOUSE_EDGE §7 suggests 3–5 seeds.

**First experiment.** Train seeds 2026, 2027, 2028, 2029, 2030 with the same config. Average the **probabilities** (or the logits) per query, and renormalise the 5-vector as HOUSE_EDGE does for the blend. Compare the ensemble's tuning score with the single-seed 50.18. If the spread across seeds is large, the ensemble is worth the runtime.

**Note.** `load_artifact` loads one model. An ensemble needs either one artifact per seed or a small wrapper that loads several and averages them.

### 10.5 A size head

**Why.** Given a win, the payout split across 1, 2, 5 and 20 is close to fixed per machine type, and only about 6% of the original error (HOUSE_EDGE §6, item 7). The share of error is larger now that the win head is better.

**Design.** Add a second output head, trained on won spins only, that predicts the conditional payout distribution from the same 148-wide stats. Use it in place of the inherited E1 shape head.

**First experiment.** Follow HOUSE_EDGE §7 item 5: decompose the remaining error into (oracle p0 + model shape) vs (model p0 + oracle shape). This tells you whether a size head can gain anything before you build one. Use `ShapeHeadForecaster.shape()` (E5, `shape_head.py`) as the comparison.

---

## 11. Gotchas

1. **`recent_half_life` defaults to 32.0. The shipped model uses 16.0.** Constructing `NeuralContextForecaster()` with no arguments gives a different model. Pass `recent_half_life=16.0` explicitly when retraining, or it silently changes the result.
2. **Use `load_artifact`, not `fit`, to reproduce the shipped predictions.** Refitting with the same seed is deterministic (test L90-103), but the shipped artifact is the reference for the score. Replay with `predict` on the 384 tuning queries to check predictions bit-for-bit.
3. **macOS + conda: OpenMP crash.** Importing torch after sklearn crashes with `OMP: Error #15`. Set `KMP_DUPLICATE_LIB_OK=TRUE` before any import. The entry runs with torch 2.9.1 CPU (pinned in `requirements.txt`).
4. **Torch is a global dependency.** `fit` changes `torch.set_num_threads`, `torch.manual_seed` and `np.random.seed` for the whole process. Don't train inside a loop that depends on those globals.
5. **`load_artifact` imports torch unconditionally** (L335), so a missing torch raises an error instead of returning a model. Only the entry's `predict.py` catches that and falls back to E6 (§7.3). Your own scripts must handle it.
6. **Fallback is per query, not per machine.** The same machine can use E7 for one cutoff and E1 for another. When comparing machines, check how many queries hit the fallback.
7. **`fit` can switch holdout rules silently** (§4.3). With one training episode, or an empty split, the holdout is the first episode in sha256 hexdigest order. This is not the 1-in-5 rule, so your holdout BCE is not comparable across these runs.
8. **`signal` is per query, not per tick.** E7's win head ignores it. The inherited payout-shape priors are keyed by `machine_type|signal`. Do not backfill `signal` into historical rows (HOUSE_EDGE §8).
9. **Tick lags, not row offsets.** Lags are looked up by tick (`seq[tick − k]`). A row with any missing lag is dropped, not imputed (HOUSE_EDGE §8).
10. **Cutoff guard.** `win_probability` drops ticks `>= cutoff` on entry (L268). Keep that line when you refactor, because prequential callers pass prefixes.
11. **Holdout examples are not independent.** The 4,365 holdout examples come from about 20% of TRAIN episodes, with up to 24 examples per machine. Early stopping on this holdout is noisy. Compare runs by the tuning score, not by small holdout BCE differences.
12. **Round 2 needs a retrain** (§8). The shipped weights encode round-1 regularities.
13. **The loader only accepts `format_version == 1`.** Any layout change (§10.2, §10.3, §10.5) needs a new version number, or old artifacts will load with the wrong width.
14. **The statistics are scaled.** `log1p(count) / log(257)` is roughly in [0, 1] for typical counts. A machine with much more than 256 usable rows gets a value above 1. This has not been checked on the real data.
15. **Paired comparisons are exploratory.** The E6-vs-E7 interval crosses zero (§6.2), and both were picked on tuning episodes that earlier scripts had seen. Do not claim E7 > E6 without a new unseen split.
16. **The PR body is stale on two numbers.** It says PyTorch 2.9.0 and E6 48.214. Use torch 2.9.1 and E6 48.54.
