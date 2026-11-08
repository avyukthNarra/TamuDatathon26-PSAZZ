# Avyukth's Team Handout: E6 + E7 Blend (Main Round)

**Written:** 2026-11-07
**Team:** Avyukth's team. Claude (Opus 5.5, with Haiku subagents) and Codex (GPT-6.1-Sol, later GPT-6-Luna).
**Final model:** a 50/50 probability blend of **E6** (signed-L1 interaction logistic) and **E7** (neural hypernetwork over prefix statistics).
**Entry branch:** `origin/avyukth/entry-final` @ `588a70b`, folder `mainFiles/entry/`.

**Primary sources**
- `/Users/avynarra/Documents/Datathon/HOUSE_EDGE_EXPERIMENT_RESULTS.md` (the full write-up; this handout summarises it and cites it for every number)
- `/Users/avynarra/Documents/Datathon/worktrees/_coord/CODEX_REPORT.md` (Codex's bootstrap CIs and test counts)
- Branch code, read with `git show origin/<branch>:<path>` after `git fetch origin`

**Provenance.** Numbers tagged **(lead)** are verified facts supplied by the lead reviewer. They were not re-run for this handout. Everything else is quoted from the two source documents or read from branch code. Nothing in this handout was executed. Reproduction snippets are sketches, marked as such.

**Scale.** Points = 60 / (1 + R), where R is the mean over four conditions of (our error / baseline error). 30 points is the baseline. 60 is perfect.

**Conditions.** `stable`, `short_history`, `drift`, `shared_change`, weighted equally. Test has no condition labels.

---

## Contents

1. TL;DR and comparison with the team's best
2. The path of experiments
3. E6 in depth
4. E7 in depth
5. The final entry (`entry-final`)
6. How to reproduce
7. Transferable ideas for the team's best model
8. Gotchas
9. Code index

---

## 1. TL;DR

- Our final model is a **50/50 average of two per-query win-probability forecasts**, renormalised over the five payouts (0, 1, 2, 5, 20).
  - **E6** fits a per-machine L1 logistic on **signed (±1)** lag features at run time.
  - **E7** is a tiny MLP, trained once on TRAIN, that maps each machine's prefix statistics to that machine's logit coefficients.
- Its score is **50.65 tuning / 50.02 audit**. E6 alone is 48.54 / 48.19. E7 alone is 50.18 / 48.99.
- **It is not the team's best model.** Zhill's rule + mixer blend and Pranav's pair mixer both score higher on the same split (lead).
- The main lesson transfers to every model: the signal is **per machine, sparse, and sign-sensitive**. A "did lag a match lag b?" feature, found with L1 or learned by a hypernetwork, beat every 0/1-coded learner by 8+ points.

### Honest comparison with the team's best (kit `episode_folds` split, 48 tuning / 48 audit)

| Model | Tuning /60 | Audit /60 | Where / source |
|---|---|---|---|
| **Zhill rule + pair-mixer blend** (team's best) | **54.37** | **54.57** | `origin/zhill-branch` @ `4fe7ba8`, `mainFiles/predict.py` (lead) |
| **Pranav pair mixer** | **54.08** | **54.39** | `origin/pranav/adaptive-mixer` @ `49baf63`, `mainFiles/slot_forecasting/pair_mixer.py` (lead) |
| **FINAL: 50/50 E6 + E7 (this entry)** | 50.65 | 50.02 | `origin/avyukth/entry-final` @ `588a70b` (MD §5) |
| E7 alone | 50.18 | 48.99 | `origin/avyukth/exp-e7-neural-context` (MD §5) |
| E6 alone | 48.54 | 48.19 | `origin/avyukth/exp-e6-interactions` (MD §5) |
| Kit baseline `pooled_recent` | 31.06 | 30.57 | `main` (MD §1) |

Gap from the final entry to the team's best:

| Comparison | Tuning gap | Audit gap |
|---|---|---|
| Final vs Zhill blend | 54.37 - 50.65 = **3.72** | 54.57 - 50.02 = **4.55** |
| Final vs Pranav mixer | 54.08 - 50.65 = **3.43** | 54.39 - 50.02 = **4.37** |

**Blending our models into the team's best does not help.** Adding E6+E7 to Zhill's model at 30% or 50% scores 53.3 to 54.0 (lead). Do not assume a blend with E6/E7 improves the best model without testing it. Section 7 lists narrower ways to test it.

**Why the others win (lead's analysis).** The simulator appears to use **three global lag pairs**: (1,3), (2,5), and (4,7). Each pair has an XOR or XNOR rule (match vs differ), and the direction is per machine. E6 instead rediscovers sparse pairs **per machine** from all 28 signed products with L1, which is noisier. The earlier conclusion that "a global pair shortlist doesn't work" (E3) was likely an artefact of 0/1 coding. With match/differ coding, a global pair set appears to work well. This has not been tested yet. Section 7 gives the first experiment.

---

## 2. The path of experiments

Scores are tuning /60 unless marked. "—" means the source documents do not record an audit score for that row.

| ID | What | Branch @ SHA | Tuning | Audit | One-line lesson |
|---|---|---|---|---|---|
| E0 | Kit baseline `pooled_recent` (Dirichlet shrinkage, half-life 24, strength 20) | `main` | 31.06 | 30.57 | Fallback and payout-shape source for most later models |
| E1 | Per-machine pair logistic, 0/1 win lags 1–8 plus 28 products | `avyukth/exp-e1-pair-logistic` @ `60f1daa` | 36.22 (C=10) | — | Pairs are essential (no pairs: 29.46); less regularisation helps |
| E2 | Per-machine prequential memory selection (Codex) | `avyukth/exp-e2-adaptive-memory` @ `6511dfb` | 37.15 (matched C=10) | — | Real but modest: +0.93 over fixed E1 at C=10 |
| E3 | Global lag-pair shortlist from TRAIN (Codex) | `avyukth/exp-e3-pair-selection` @ `56b60d6` | 33.96 (best restricted) | — | Negative. Likely a 0/1-coding artefact (lead). Untested with match/differ coding |
| E4a | Shared change-point detector (Codex) | `avyukth/exp-e4-shared-change` @ `0e33eaa` | 36.06 | — | Weak. Hard cuts are brittle: 6/12 false boundaries on stable episodes |
| E4b | Episode-pooled memory selection (Claude) | `avyukth/exp-e4-episode-memory` @ `0b3ff0d` | combo 37.96 to **39.04** | — | Pool the memory choice over the episode's 8 machines. Robust; used in E11 |
| E5 | Payout-shape head (Claude) | `avyukth/exp-e5-shape-head` @ `43f7de9` | 36.45 | 38.08 | Shape is a small share of error. Big-win lag inputs overfit (34.51) |
| E6 | **Signed-L1 interaction logistic** (Codex) | `avyukth/exp-e6-interactions` @ `c3ba387` | **48.54** | **48.19** | Representation beats learner: +8 over the 0/1 ensembles |
| E7 | **Neural hypernetwork over prefix statistics** (Codex) | `avyukth/exp-e7-neural-context` @ `af645aa` (scored at `4e0267a`) | **50.18** | **48.99** | Learns stats to coefficients. Best on short history |
| E8 | Neighbour-machine inputs (Claude) | `avyukth/exp-e8-neighbor-inputs` @ `5c2bc3c` | 36.51 | — | No gain in 0/1 model. Signed or episode-summary version untested |
| E9 | Short-history shrinkage (Claude) | `avyukth/exp-e9-win-blend` @ `f8c0978` | 37.49 (α=0.6) | — | Shrink machines with fewer than 100 spins toward baseline. Sharpening (T=0.8) hurt |
| E10 | ExtraTrees + logistic (Claude) | `avyukth/exp-e10-machine-trees` @ `281eaa4` | 37.79 | — | Trees capture higher-order rules. ExtraTrees beats RF |
| E11 | Flex experts: logistic 0.25 / ExtraTrees 0.25 / HGB 0.5 (Claude) | `avyukth/exp-e11-flex-experts` @ `f35bb67` | 40.26 | 40.90 | Strong 0/1 ensemble that capped near 40. The representation was the ceiling |
| **Final** | **50/50 E6 + E7** | `avyukth/entry-final` @ `588a70b` | **50.65** | **50.02** | Errors differ. Blend is better on both halves |

Other blend result: 70/30 E6/E7 scores 50.10 tuning / 49.62 audit (MD §5).

### Condition-level loss ratios (MD §5; lower is better; 1.0 = baseline)

Order is stable, short_history, drift, shared_change.

| Model | Tuning ratios | Audit ratios |
|---|---|---|
| E6 | .228 / .259 / .264 / .194 | .163 / .358 / .235 / .224 |
| E7 | .206 / .198 / .220 / .158 | .181 / .238 / .233 / .246 |
| Final blend | not listed | .148 / .235 / .212 / .203 |

E6's weak spot is short history on audit (.358). E7 is best there (.238). This is why the blend helps.

### Key lessons (MD §6)

1. **Representation beats model family.** 0/1 to signed lag coding with L1 was worth about +8 points. Five 0/1 learner families capped near 40.
2. **Signal is per machine, sparse, sign-sensitive.** Pooled models fail because machines respond in opposite directions.
3. **Amortised estimation helps where data is thin.** E7 turns small-sample statistics into coefficients.
4. **Memory must adapt without condition labels.** Prequential selection works per machine (E2) and per episode (E4b). Not yet applied to E6 or E7.
5. **Hard change-point cuts are brittle** (Codex E4). Neighbour inputs did not help the 0/1 model, which is not conclusive for signed models.

---

## 3. E6 in depth: signed-L1 interaction logistic

**Branch:** `origin/avyukth/exp-e6-interactions` @ `c3ba387`
**Files:** `mainFiles/slot_forecasting/interaction_sweep.py` (class `InteractionSweepForecaster`, 110 lines), `tests/test_interaction_sweep.py` (7 tests per CODEX_REPORT)
**Base class:** `PairLogisticForecaster` in `mainFiles/slot_forecasting/pair_logistic.py` (E1 branch)

### 3.1 What E6 is

E6 subclasses the E1 base. It overrides `win_probability(seq, cutoff)`, which returns P(win) for one query. Everything else (the pooled_recent shape head, the `[1 - pw, pw * positive_shape]` composition) is inherited.

**Important clarification for new readers.** `fit(train_queries, train_history, train_labels)` only fits the base `pooled_recent` shape priors on TRAIN. The per-machine logistic is **not** fit on train. It is fit at predict time on **each query's own machine history** (ticks strictly before the cutoff). So the entry's "E6 refits on train" means "reads the train-derived shape priors", and the per-machine fits use the history you pass to `predict`.

### 3.2 Features

For each query's machine, every row is a tick `t < cutoff` where lags 1 to 8 are all observed.

| Block | Count | Definition |
|---|---|---|
| Main lags | 8 | signed lag value, `2*x - 1`, where `x = 1` if payout at `t - lag` > 0 |
| Pair products | 28 | `s_a * s_b` for all `a < b` in lags 1..8 (`itertools.combinations(range(8), 2)`) |
| **Total (degree 2)** | **36** | |

Degree 3 adds triple products (`degree` parameter accepts 1, 2, 3). The sweep found **degree 2** won. The source documents do not record a degree-3 score.

Code, from `feature_matrix`:

```python
values = np.asarray(X, dtype=float)
if self.encoding == 'signed':
    values = 2.0 * values - 1.0          # win -> +1, lose -> -1
blocks = [values]
for size in range(2, self.degree + 1):
    indices = self._interaction_indices[size]
    blocks.append(values[:, indices].prod(axis=2))
return np.concatenate(blocks, axis=1)
```

### 3.3 Why signed coding matters

With ±1 coding, `s_a * s_b` is **+1 when lags a and b match** (both win or both lose) and **-1 when they differ**. That is an XNOR / XOR feature, one sign-flip away.

With 0/1 coding, the same rule needs a main effect plus a product, with coefficients that interact. L1 penalises that, so it gets shrunk away. The numbers in §3.6 show this.

With L2, the right features are kept, but so is everything else, so predictions blur.

**Signed + L1 is a sparse search for a few "these two spins match" rules per machine.** The sign of the fitted coefficient gives the direction (XOR vs XNOR), so direction is per machine automatically.

### 3.4 Liblinear settings and recency weights

```python
LogisticRegression(
    C=self.C,                 # 0.5 in the final entry
    penalty='l1',
    solver='liblinear',
    intercept_scaling=100,    # keeps the intercept from being penalised away
    random_state=0,
    max_iter=1000,
)
model.fit(features, ys, sample_weight=np.asarray(weights))
```

- **Recency weight:** `weight = 2 ** (-age / half_life)`, where `age = cutoff - 1 - tick`. Final half-life is **16**.
- **Minimum data:** fewer than `min_rows` (default 12) usable rows, or only one class in `ys`, returns `None`. `predict` then falls back to `pooled_recent` for that query.
- **Diagnostics** (in `self.diagnostics`): `successful_fits`, `max_iter_hits`, `fallbacks`. The winning candidate had 375 fits, 9 fallbacks, and no max-iteration hits (CODEX_REPORT).
- **Runtime:** under 1 s for all of validation (MD §5).

### 3.5 Sweep

E6 was a **96-configuration sweep** over encoding, degree, penalty, C, and half-life. Winning setting:

```python
InteractionSweepForecaster(degree=2, encoding="signed", penalty="l1", C=0.5, half_life=16)
```

Codex's narrow refinement is also recorded in CODEX_REPORT. Its final tuning score is 48.5376.

### 3.6 Ablations

| Variant | Tuning | Audit | Note (source) |
|---|---|---|---|
| **Signed + L1** (final) | **48.54** | **48.19** | MD §5 |
| 0/1 coding + L1 (same C, half-life) | 32.43 | 32.04 | MD §5 |
| Signed + L2 | 38.70 | 40.79 | MD §5 |
| Degree 3 (triples) | not recorded | not recorded | MD: "the winner was degree 2" |

Tuning condition ratios for signed + L1 are .228 / .259 / .264 / .194. Audit ratios are .163 / .358 / .235 / .224.

Reading the ablations:
- **Coding (0/1 to signed):** about +16 points at the same C and half-life. This is the largest single effect in the project.
- **Penalty (L2 to L1, signed):** about +10 tuning and +7 audit. L1 selects the few useful rules.

### 3.7 Known weak spot

**Short history on audit** (ratio .358). Only about 30 rows per short-history machine have all 8 lags observed, so the sparse selection has little to work with. E7 is much better here (.238).

Candidate fixes (MD §7.3): weight E7 more when the machine has fewer than 100 spins (observable at test time), shrink the blend toward baseline for short histories (E9-style), or scale C by row count.

### 3.8 Where the code lives

`win_probability` is the only override that matters (cutoff guard, tick-lag rows, recency weights, liblinear fit, then `predict_proba` on the query's signed feature vector). `predict` resets `self.diagnostics` each call and then uses the inherited `PairLogisticForecaster.predict`.

---

## 4. E7 in depth: neural hypernetwork over prefix statistics

**Branch:** `origin/avyukth/exp-e7-neural-context` @ `af645aa` (scored at `4e0267a`)
**Files:** `mainFiles/slot_forecasting/neural_context.py` (class `NeuralContextForecaster`, 368 lines), `tests/test_neural_context.py` (6 tests per CODEX_REPORT)
**Artifact:** `mainFiles/entry/models/state.pt` (on `entry-final`)

### 4.1 Idea

E7 is a **learned estimator** (meta-learning). It does not fit anything per machine at run time. Instead, a shared network reads a summary of each machine's history and outputs that machine's **logistic coefficients** for the 36 signed features from §3.2. The network is trained once, on TRAIN, across many machines and many prefixes.

Because the mapping is learned, it can turn a few observations into sharp coefficients, which is what short-history machines need.

The model has **no machine, episode, signal, or condition embeddings**. It reads statistics only (CODEX_REPORT).

### 4.2 Prefix sufficient statistics

For a machine's exact-lag rows before the cutoff, the code builds:

- `Z`: the 36 signed features `z` (8 main lags + 28 pairs), one row per usable tick
- `y`: win indicator, `int(payout > 0)`

Then it computes **prefix** statistics, meaning statistics of the first `k` rows, which excludes row `k` itself (no leakage into its own target). Two copies are computed:

| Block | Width | Definition |
|---|---|---|
| All-history mean of z | 36 | `mean(z_0..z_{k-1})` |
| All-history signed correlation | 36 | `mean((2y - 1) * z)`: how strongly each feature co-moves with winning |
| All-history win rate | 1 | `mean(y)` |
| All-history log count | 1 | `log1p(k) / log(257)` |
| **Subtotal, all-history** | **74** | |
| Recency-weighted copy (half-life 16) | 74 | same four blocks with weights `2 ** ((t - t0) / 16)`, anchored at the first tick |
| **Total stats input** | **148** | `recent_half_life=16` |

The recency-weighted block gives the network a view of the current regime. The all-history block gives stable context. The `log count` lets it know how much data it has.

`context_statistics(seq, cutoff)` returns the full-history summary as of the cutoff. Returns `None` if fewer than `min_context` (16) rows exist.

### 4.3 The hypernetwork

```python
self.network = torch.nn.Sequential(
    torch.nn.Linear(stats_width, 64),          # stats_width = 148
    torch.nn.ReLU(),
    torch.nn.Linear(64, 36 + 1),               # 36 coefficients + 1 bias
)
torch.nn.init.zeros_(self.network[-1].weight)  # last layer zero-init: starts at p = 0.5
torch.nn.init.zeros_(self.network[-1].bias)

# forward pass, per example:
output = self.network(stats)                   # shape (n, 37)
logit = (output[:, :-1] * z_query).sum(dim=1) + output[:, -1]
p = sigmoid(logit)
```

So the network outputs **per-machine coefficients** `coef`, and the logit is `coef . z_query + bias`. The coefficients are conditioned on the machine's statistics.

### 4.4 Training data and procedure

Source: TRAIN history and TRAIN labels only. Procedure (from `fit`):

1. **Build examples.** For each TRAIN query's machine, take its exact-lag rows. If there are more than `min_context` (16) rows, pick up to **24** random prefix positions from index 16 onward (`examples_per_machine=24`, `rng` seeded with 2026).
2. **Split by episode hash.** `sha256(episode_id)` first 4 bytes, mod 5 == 0 means internal holdout (about 1 in 5 episodes). Counts: **12,561 train** and **4,365 holdout** examples (CODEX_REPORT, `config.json`).
3. **Loss and optimiser.** `BCEWithLogitsLoss`, **Adam lr 1e-3**, batch **256**, `torch.manual_seed(2026)`, threads capped at 2.
4. **Early stopping.** Max 50 epochs, patience 6, on internal holdout BCE. Stopped after 19 epochs, best epoch **13**, holdout BCE **0.408979**. Training took about 4.2 s (`config.json`).
5. **Restore** the best state dict.

Codex's warning: the 12,561 examples overlap within machines and episodes. They are not independent observations.

### 4.5 Fallback inside E7

`win_probability(seq, cutoff)` in E7:

```
if the last 8 lags are not all observed, or the network is missing:
    -> E1-style PairLogisticForecaster.win_probability (0/1 coding, C=3, half-life 32)
if fewer than min_context (16) valid context rows:
    -> same E1 pair-logistic fallback
if that also returns None:
    -> predict() uses pooled_recent for the shape and the query probability
```

So E7 degrades to E1, not to E6. Its fallback is the 0/1 model. That matters: E7's short-history answers, where context is thin, come partly from the E1-style fit.

### 4.6 Configuration

From `config.json` (matches CODEX_REPORT): `recent_half_life=16.0`, `half_life=32.0` and `C=3.0` (E1 fallback), `base_half_life=24.0`, `min_context=16`, `min_rows=12`, `examples_per_machine=24`, `max_epochs=50`, `patience=6`, `batch_size=256`, `learning_rate=0.001`, `seed=2026`, `lags=8`, `clip=0.0001`. Metadata candidate name: `e7-recent-16-seed2026`.

### 4.7 Artifact loading (no refit)

`save_artifact` writes two files:

- `state.pt`: a torch dict with `format_version: 1`, `configuration`, `network_state`, `training_diagnostics`, `base_forecaster` (the pooled_recent priors), and `metadata`.
- `state.pt.json`: the same dict minus `network_state`, for humans.

`load_artifact(path)`:

`load_artifact(path)` runs `torch.load(..., weights_only=True)`, rebuilds the model from `configuration` (`cls(**configuration)`), recreates the same `Linear(148, 64) -> ReLU -> Linear(64, 37)` network, loads `network_state`, and restores the pooled_recent priors. It checks `format_version == 1` and raises `ValueError` otherwise. No refit happens.

Entry-level metadata in `config.json` records the train input SHA-256 hashes (`history.csv`, `labels.csv`, `queries.csv`), the scored module hash, and the scored training commit `4e0267a`. Use these to confirm you have the same training data.

### 4.8 Results

| Split | Tuning | Audit | Condition ratios (tuning) | Condition ratios (audit) |
|---|---|---|---|---|
| E7 | **50.18** | **48.99** | .206 / .198 / .220 / .158 | .181 / .238 / .233 / .246 |

- Versus E6 on tuning: **+1.65**, paired episode-bootstrap 95% interval [-0.06, +3.27] (CODEX_REPORT). The interval crosses zero, so E7's lead over E6 is uncertain.
- Versus E2 best (36.22): +13.96, interval [+11.79, +15.92].
- Prediction from the artifact takes about 2 s on validation.
- Replay from the artifact reproduced the winning 384-row tuning predictions to `2.2e-16` (CODEX_REPORT).

### 4.9 Why it works

E7 learns the mapping from **"this machine's feature/outcome correlations so far"** to **sharp coefficients**, using all train machines. That is most useful on short history, where a per-machine L1 fit has too few rows. E7's short-history audit ratio is .238, against E6's .358.

---

## 5. The final entry (`entry-final`)

**Branch:** `origin/avyukth/entry-final` @ `588a70b`
**Commit:** `feat(entry): final entry blends signed-L1 logistic and neural context model`
**Files:** `mainFiles/entry/predict.py`, `README.md`, `requirements.txt`, `reproduce.json`, `models/state.pt`, `models/state.pt.json`, `models/config.json`, plus a full copy of `slot_forecasting/`

### 5.1 `predict.py`, step by step

```python
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"     # must be set before numpy/sklearn/torch import
```

Then:

1. **Load inputs.** `load_inputs(D / "train")` and `read_labels(D / "train/labels.csv", tq)`.
2. **Fit E6.** `InteractionSweepForecaster(degree=2, encoding="signed", penalty="l1", C=0.5, half_life=16).fit(tq, th, labels)`. This sets the pooled_recent priors from train.
3. **Predict E6 on test.** `e6 = model.predict(q, h)`, using the test histories `h`.
4. **Try E7.**
   ```python
   from slot_forecasting.neural_context import NeuralContextForecaster
   e7 = NeuralContextForecaster.load_artifact(str(<entry folder>/"models"/"state.pt")).predict(q, h)
   final = {k: tuple(0.5*a + 0.5*b for a, b in zip(e6[k], e7[k])) for k in e6}
   used = "blend"
   ```
5. **Fallback.** Any `Exception` in step 4 prints the error to stderr and sets `final = e6`, `used = "e6-fallback"`.
6. **Renormalise** each row to sum to 1, then `write_submission(args.out, final)`.
7. **Print** `blend` or `e6-fallback` to stderr. This is how you know which path ran.

Command: `python predict.py --data <dir with train/ and test/> --out <submission.csv>`.

### 5.2 The torch fallback

The fallback is a `try/except Exception`. It covers:

- `torch` import failure (ImportError)
- missing or malformed `models/state.pt` (ValueError, FileNotFoundError)
- shape or key mismatch in the blend (KeyError)

It does **not** cover a hard process crash. If the OpenMP runtime aborts the process (the OMP Error #15 issue below), there is no exception to catch. This is why `KMP_DUPLICATE_LIB_OK` is set at the top of `predict.py`. (This is a reading of the code, not something I reproduced.)

**To check which path ran:** stderr must say `blend`. If it says `e6-fallback`, the blend did not run and the forecast is E6 alone (about 48.5 tuning).

### 5.3 Requirements

`requirements.txt` (exact pins):

```
numpy==2.3.3
scikit-learn==1.7.2
scipy==1.16.2
joblib==1.5.2
threadpoolctl==3.6.0
torch==2.9.1
```

- Python: **3.13.1** (`README.md`, `reproduce.json` image `python`). MD and CODEX_REPORT say "Python 3.13".
- Torch is CPU only. The README records torch 2.9.1 on CPU.
- No network access at prediction time.
- Organiser limits: 30 min, 2 CPUs, 16 GiB. CSV at most 10 MB, ZIP at most 100 MB (MD §8).

### 5.4 Runtime and determinism

- **8.20 s** wall clock (7.31 s user, 0.45 s sys, 94% CPU) on Apple M3 Pro, macOS, Python 3.13.1, torch 2.9.1 CPU (`README.md`).
- Two consecutive runs from a fresh copy give **byte-identical** CSVs (`cmp`).
- Entry checker: `VALID FORMAT: 1536 queries | main | test`, exit 0.
- Forecast SHA-256 recorded in README: `934dd61dba3a6f9015dbd7b523f81b9ba2c759cafb9c99507a8297747328b943`.

### 5.5 The OMP / KMP gotcha

**Symptom:** on macOS with conda, importing torch after sklearn in the same process crashes with `OMP: Error #15` (duplicate OpenMP runtime).
**Fix:** set `KMP_DUPLICATE_LIB_OK=TRUE` **before** the imports. `predict.py` does this. Any script or notebook that loads both must do the same.

This also applies to `exp_eval.py` if you add E7 to it. Set the variable in the shell, or put it at the top of the script.

### 5.6 README housekeeping still open

- `Team ID: TBD` in `README.md`. It must be filled from the team's Form list before submission.
- The README's example command uses a local path. The organiser form is `python predict.py --data /data --out /output/submission.csv`.

---

## 6. How to reproduce

### 6.1 Environment

```bash
# Use Python 3.13 with the pinned requirements.
python3 -m pip install -r mainFiles/entry/requirements.txt

# macOS + conda: set this before any run that imports torch and sklearn together.
export KMP_DUPLICATE_LIB_OK=TRUE
```

Data: place the released data at `mainFiles/public/` (`train/`, `validation/`, `test/`).

To read the code, check out the branch in a **separate** worktree. Do not switch your own branch:

```bash
git fetch origin
git worktree add ../e6-check origin/avyukth/exp-e6-interactions
git worktree add ../e7-check origin/avyukth/exp-e7-neural-context
git worktree add ../entry-check origin/avyukth/entry-final
```

### 6.2 The scorer: `exp_eval.py`

`mainFiles/exp_eval.py` scores candidates on the **tuning** episodes only. It uses the same split and scorer as the kit's `select_model`:

```bash
cd <worktree>/mainFiles
python exp_eval.py pooled_recent e5_s50          # tuning only (safe)
python exp_eval.py <name> --audit                # ALSO scores audit: log it in RESULTS.md first
python exp_eval.py <name> --json out.json        # write per-candidate results
```

**Do not** use `select_model` or `run_experiments.py` for sweeps. They always score audit for the chosen model.

Internals: `episode_folds(targets, groups)` returns `(tuning, audit)`. `subset_score(pred, targets, groups, reference, subset, max_points)` returns the points, the loss ratios, and per-condition values.

**Important:** the `exp_eval.py` on `avyukth/exp-e11-flex-experts`, `avyukth/exp-e6-interactions`, `avyukth/exp-e7-neural-context`, and `avyukth/entry-final` **does not register E6 or E7**. Its `CANDIDATES` dict has `pooled_recent`, E1, and E5 entries only (`exp_eval.py` on `exp-e6-interactions` and `exp-e7-neural-context` lists E1 entries only). You have to add them.

**Sketch (not run by the author of this handout):** add to `CANDIDATES` in `mainFiles/exp_eval.py` on the E6 branch:

```python
from slot_forecasting.interaction_sweep import InteractionSweepForecaster
CANDIDATES["e6_signed_l1"] = lambda: InteractionSweepForecaster(
    degree=2, encoding="signed", penalty="l1", C=0.5, half_life=16)
```

And on the E7 branch:

```python
from slot_forecasting.neural_context import NeuralContextForecaster
CANDIDATES["e7_recent16"] = lambda: NeuralContextForecaster(recent_half_life=16.0)
```

The E7 constructor defaults match `config.json` apart from `recent_half_life`. The constructor default is 32.0, so override it to 16.0. Retraining takes seconds (4.2 s in `config.json`).

### 6.3 The model interface

Every forecaster follows the same interface:

```python
model = SomeForecaster(**params).fit(train_queries, train_history, train_labels)
pred = model.predict(queries, history)      # {query_id: (p0, p1, p2, p5, p20)}
```

- `queries` and `history` come from `load_inputs(dir)` in `slot_forecasting/io.py`.
- `train_labels` comes from `read_labels(dir/"labels.csv", train_queries)`.
- `predict` returns a 5-tuple per query, summing to 1.

Extension point (PairLogisticForecaster subclasses):

```python
def win_probability(self, seq, cutoff) -> float | None:
    # seq: {tick: payout} for one machine. Ignore ticks >= cutoff.
    # Return P(win), or None to fall back to pooled_recent.
```

`predict` composes the output as `[1 - pw, pw * positive_shape]`, where the positive shape is pooled_recent's normalised positive part (or E5's shape head, if used).

### 6.4 Scoring the final blend on validation

`predict.py` writes only the **test** submission. To score the blend on **validation**, repeat the entry's blend on validation inputs, then call `subset_score` on the `tuning` episodes (the same calls `exp_eval.py` makes; the `load_inputs`, `read_labels`, `episode_folds` and `subset_score` names are the ones in `slot_forecasting/`). Sketch, not run by the author:

```python
e6 = InteractionSweepForecaster(degree=2, encoding="signed", penalty="l1", C=0.5, half_life=16).fit(tq, th, labels).predict(vq, vh)
e7 = NeuralContextForecaster.load_artifact("mainFiles/entry/models/state.pt").predict(vq, vh)
blend = {k: tuple(0.5*a + 0.5*b for a, b in zip(e6[k], e7[k])) for k in e6}
# subset_score(blend, targets, groups, reference, tuning, max_points)["technical_points"]  -> expect about 50.65
```

Score the **audit** side only when you are logging an audit look (see §8.4).

### 6.5 Running the entry

```bash
# From mainFiles/ of the entry checkout (predict.py imports its own entry/slot_forecasting/ and loads entry/models/)
cd <worktree>/mainFiles
python3 entry/predict.py --data public --out runs/entry-final/submission.csv
python3 check_submission.py runs/entry-final/submission.csv --data public
```

The README's form runs from inside the entry folder (`python3 predict.py --data <dir> --out <csv>`), which works the same way if `--data` points at the public folder.

Expected stderr: `blend`. Expected wall clock: about 8 s.

### 6.6 Tests

`tests/test_interaction_sweep.py` (E6, 7 tests) and `tests/test_neural_context.py` (E7, 6 tests), per CODEX_REPORT. I did not run them. Pattern: `PYTHONPATH=mainFiles python -m unittest discover -s tests -p 'test_<name>.py' -v`.

---

## 7. Transferable ideas for improving the team's best model

The team's best is **Zhill's rule + pair-mixer blend** (`origin/zhill-branch`, 54.37 / 54.57, lead) and **Pranav's pair mixer** (`origin/pranav/adaptive-mixer`, 54.08 / 54.39, lead). Both already use the global pairs. Ideas here are ranked by expected value per hour. Each has a first experiment. Score every experiment **tuning only** unless you are logging an audit look.

### 7.1 Zhill's rule model: what it does now

From `mainFiles/rule_model.py` (class `RuleForecaster`):

- Per machine, keeps **contingency tallies** of next payout, indexed by the win/lose situation at a set of lag rules (`()`, singles, and all pairs over lags 1..8).
- Scores each rule prequentially on the machine's history, with trust fade `trust_forget`.
- Final forecast = trust-weighted average of all rules.
- Docstring says common pairs are **(1,3), (2,5), (4,7)**.
- `SETTINGS` has a `forget` tuple. The docstring says that with several values, **each machine picks its memory**.
- `predict.py` blends `RULE_WEIGHT = 0.2` of the rule model with 0.8 of the pair mixer, using `Weighted`.

### 7.2 Ideas and first experiments

**Idea A: Test the global-pair hypothesis with signed sparse L1 (fastest, under an hour).**
- Why: the lead's analysis says the simulator uses three global pairs, and per-machine L1 over all 28 products is noisy. If that is true, a sparse model restricted to the three global pairs should match E6 or do better.
- First experiment: subclass `InteractionSweepForecaster` and override `feature_matrix` to keep only the 8 signed main lags plus the products for pairs **(1,3), (2,5), (4,7)**. In 0-based indices these are lag indices (0,2), (1,4), (3,6). Find their columns in the 28-product block with `list(combinations(range(8), 2)).index((a, b))`. Keep C=0.5, half-life 16. Score on tuning and compare with 48.54.
- If it matches or beats 48.54, the global pairs are the structure. Then try it as an expert inside the pair mixer.

**Idea B: Re-test the E3 negative with match/differ coding.**
- Why: the lead thinks E3's conclusion was a 0/1 artefact. E3 used 0/1 win indicators.
- First experiment: on `avyukth/exp-e3-pair-selection`, change the pair features to signed products and rerun the global shortlist (it used all 28 pairs as candidates). Compare with the 0/1 result (33.96 best restricted, 35.41 all-pairs E1 baseline).

**Idea C: E4-style episode-pooled memory choice inside Zhill's rule model.**
- Why: E4b gave +1.1 by pooling the memory choice over an episode's 8 machines. Shocks hit all machines together (MD §5). Zhill's `forget` tuple lets each machine pick its own memory, which is the per-machine version.
- First experiment: in `RuleForecaster`, for each episode, compute the prequential Brier score of each `forget` value summed over that episode's 8 machines (last 48 spins, every 3rd, as in `episode_memory.py`). Use the single best value for all 8 machines. Compare tuning with the per-machine choice.
- Caveat: the MD expects "about +0.5 to +1.5" for memory selection applied to a new base. That is an estimate, not a result.

**Idea D: E7-style hypernetwork on the three parity cells.**
- Why: E7 learns a map from prefix statistics to coefficients. Using only three global pairs gives a much smaller output (about 4 numbers: 3 pair coefficients plus bias, optionally plus 8 main lags), so it should overfit less.
- First experiment: copy `neural_context.py`. Change `_query_features` to return only the three signed pair products (plus the main lags if you want them). Change the output head width to match. Keep the E7 protocol (24 prefixes per machine, episode-hash holdout, BCE, Adam 1e-3, patience 6). Score on tuning against 50.18 (E7) and against Zhill's 54.37.
- The stats block should be **per global pair**: the match rate and signed correlation with winning for each pair, plus recency-weighted copies.
- Note: `load_artifact` rebuilds the network from `lag_width`, so the loader must change with the feature set.

**Idea E: Gate the E6/E7 blend to short histories only (for Zhill's model).**
- Why: full blends of E6/E7 into Zhill's model at 30% or 50% scored 53.3 to 54.0 (lead). E6/E7 are strongest where history is short (E6 .358 on audit, E7 .238). E9 showed that shrinking short-history machines helps.
- First experiment: keep Zhill's prediction for machines with at least 100 observed spins. For machines below 100, use a blend with weight alpha on E7 (try alpha = 0.3 and 0.5). Score tuning. Ideally this does not hurt the long-history conditions.
- The gate uses only observable history length, which is available at test time.

**Idea F: Frozen global pairs in Pranav's pair mixer.**
- Why: Pranav's commit `77c320b` "rediscover lag pairs and rule type from each round's train" rediscovers pairs each round. Discovery might pick the wrong pairs when a round's train data is small or changed. This is a hypothesis to test, not a finding.
- First experiment: in `mainFiles/slot_forecasting/pair_mixer.py`, replace the output of `discover_pairs` and `choose_pairs` with the fixed list (1,3), (2,5), (4,7). Score tuning and compare with 54.08 (lead). I read only the function names of this file, not its internals, so check `choose_pairs` before editing.

**Idea G: Round 2 (adaptation) checks.**
- E7 must be **retrained on round 2 train**. Its learned mapping may encode main-round regularities (MD §7).
- Validate the blend weight again on round 2 validation. E6 may deserve more weight if E7 transfers poorly.
- Main-round data is mounted at `/main_data` during adaptation. Pooling it into E7 training is a reasonable experiment (MD §7).
- E6 refits per machine at run time, so it adapts without retraining.

**Idea H: Signed sparse neighbour features (open).**
- E8 tested neighbour inputs only in the 0/1 pair logistic and found no gain (36.51). The rules say earlier outcomes on neighbouring machines can help.
- First experiment: L1 over products of the own-lag sign with a neighbour's lag-1 sign, in the E6 framework. Or feed an episode summary into E7's statistics. This targets `shared_change` (MD §7.6).

### 7.3 Ideas that are probably not worth it first

- **Shape head swap (E5).** The payout shape is a small share of error, but check the decomposition first (MD §7.5: oracle p0 with model shape vs model p0 with oracle shape). Do it after the win model improves.
- **Degree 3 features in E6.** Degree 2 won the sweep. A group or hierarchical L1 is a better next step (MD §7.7).
- **A never-forget expert in the pair mixer.** Lead's finding: stable machines want infinite memory, and adding a never-forget expert to the pair mixer gives +0.10. Small, but cheap to keep.

---

## 8. Gotchas

### 8.1 The cutoff guard

- `win_probability(seq, cutoff)` must **ignore ticks >= cutoff**. History files stop before the cutoff, but prequential callers pass longer prefixes.
- E1 and E5 add the guard explicitly (commits `60f1daa`, `43f7de9`).
- E6 checks `if tick >= cutoff: continue`.
- E7 filters the sequence first: `clean_seq = {t: p for t, p in seq.items() if t < cutoff}`.
- If you write a new model, copy this check.

### 8.2 Tick lags are not row offsets

- Lags are **tick** lags: `seq.get(tick - lag)`. Use `_lag_vector(seq, tick, lags)`.
- Short-history machines have **missing ticks** (about 70% of their span). A missing tick is unknown, not a loss.
- Rows with any missing lag are skipped. Imputing missing lags did not help (E11).

### 8.3 `signal` is per query, not per tick

- `queries.csv` has `signal` (-1/0/1) per query only.
- Do not backfill it into historical training rows.

### 8.4 Audit budget and contamination

This is the most important gotcha for judging the 50.02.

- **The audit has been looked at more than once.** MD §4 lists audit looks for `pooled_recent`, E5, E11, E6 (plus two ablations), E7, and the E6+E7 blends, all logged in `worktrees/_coord/RESULTS.md`.
- **The blend weight was compared on audit.** MD §5 lists both the 50/50 (50.02 audit) and 70/30 (49.62 audit) blends. The 50/50 choice was therefore made with both audit numbers visible.
- **Early diagnostics saw everything.** `worktrees/_coord/diag.py` and `diag2.py` looked at **all** validation targets, audit included. They informed E1's design: pairs, C, and half-life.
- **Codex's runs used tuning only.** CODEX_REPORT says there were zero new audit predictions or scores for the Codex runs. The audit numbers for E6 and E7 come from Claude's logs in RESULTS.md.
- Treat audit as strong evidence, not a blind holdout. The honest statement is "audit 50.02, after several audit looks".

**Discipline.** Use `exp_eval.py` without `--audit` for sweeps. Use `--audit` only for a candidate you have already chosen, and log it in RESULTS.md.

### 8.5 Noise level

- With 12 episodes per condition per half, differences under about ±0.3 points are noise (MD §4).
- Use the paired episode-bootstrap for close calls. CODEX_REPORT used 1,000 resamples, seed 314159. Those intervals are exploratory, because 96 E6 configurations and four E7 settings were compared on the same tuning episodes.

### 8.6 Two unit systems in the write-ups

- **MD tables** give **loss ratios** per condition (1.0 = baseline; lower is better). Example: E6 tuning stable .228.
- **CODEX_REPORT** gives **absolute five-class probability error** per condition. Example: E6 stable .047251.
- These are different numbers. Do not compare them across the two documents.

### 8.7 Name collisions

- **Two "E4"s.** E4a is Codex's shared change-point detector (`exp-e4-shared-change`). E4b is Claude's episode memory (`exp-e4-episode-memory`), which feeds E11.
- **E2 vs E1.** E2 is Codex's adaptive memory. It is not an extension of E1's code.

### 8.8 Branch naming

- A local branch named `avy` exists, so `avy/...` branch names fail. Use `avyukth/...`.
- One branch per experiment: `avyukth/exp-<id>-<desc>`.
- Branch names on the team's best model: `zhill-branch` and `pranav/adaptive-mixer`.

### 8.9 Smaller items

- **Git stash is shared** across worktrees. Never use bare `git stash` or `git stash pop`. Use a WIP commit or `git stash push -u -m "<tag>"`.
- **liblinear L1 intercept:** E6 uses `intercept_scaling=100`. Without it, the intercept is penalised like the coefficients.
- **Submission limits:** Python 3.13 with pinned requirements, no network, 30 min, 2 CPUs, 16 GiB, CSV at most 10 MB, ZIP at most 100 MB. Use `build_entry.py --source <dir>` and `check_submission.py`.
- **Local-only artifacts** (git-excluded, not on the branch): `worktrees/codex-exp/mainFiles/runs/e7-best-artifact/`, `worktrees/_coord/RESULTS.md`, `CODEX_REPORT.md`. The committed `mainFiles/entry/models/state.pt` on `avyukth/entry-final` is the one to use.

---

## 9. Code index

### 9.1 Branches

| Branch | Tip | Model | Status in sources |
|---|---|---|---|
| `avyukth/entry-final` | `588a70b` | 50/50 E6 + E7 final entry | Entry built, checked (MD §5) |
| `avyukth/entry-main` | `dbb6e38` | E6 alone (entry #3, audit 48.19) | Entry built |
| `avyukth/exp-e6-interactions` | `c3ba387` | E6 signed-L1 | Draft PR #11 |
| `avyukth/exp-e7-neural-context` | `af645aa` (scored `4e0267a`) | E7 hypernetwork | Draft PR #12 |
| `avyukth/exp-e1-pair-logistic` | `60f1daa` | E1 base | Base for PRs #8–#12 |
| `avyukth/exp-e11-flex-experts` | `f35bb67` | E11 flex experts | Scorer `exp_eval.py` |
| `avyukth/exp-e2-adaptive-memory` | `6511dfb` | E2 | Draft PR #8 |
| `avyukth/exp-e3-pair-selection` | `56b60d6` | E3 | Draft PR #9 |
| `avyukth/exp-e4-shared-change` | `0e33eaa` | E4a (Codex) | Draft PR #10 |
| `avyukth/exp-e4-episode-memory` | `0b3ff0d` | E4b (Claude) | Used in E11 |
| `avyukth/exp-e5-shape-head` | `43f7de9` | E5 | |
| `avyukth/exp-e8-neighbor-inputs` | `5c2bc3c` | E8 | |
| `avyukth/exp-e9-win-blend` | `f8c0978` | E9 | |
| `avyukth/exp-e10-machine-trees` | `281eaa4` | E10 | |
| `avyukth/exp-combo-trees-blend` | `39168b8` | E9 + E10 combo | |
| `zhill-branch` | `4fe7ba8` | Team best: rule model + pair mixer (`RULE_WEIGHT = 0.2`) | Lead: 54.37 / 54.57 |
| `pranav/adaptive-mixer` | `49baf63` | Team pair mixer | Lead: 54.08 / 54.39 |

None of these are merged to `main`. PRs #8 to #12 are drafts based on the E1 branch.

### 9.2 Files and classes

| File | Class or function | Branch | Role |
|---|---|---|---|
| `mainFiles/slot_forecasting/pair_logistic.py` | `PairLogisticForecaster`, `_lag_vector`, `_expand` | `exp-e1-pair-logistic` | Base class; extension point `win_probability` |
| `mainFiles/slot_forecasting/interaction_sweep.py` | `InteractionSweepForecaster` | `exp-e6-interactions` | E6 signed-L1 interaction logistic |
| `mainFiles/slot_forecasting/neural_context.py` | `NeuralContextForecaster` (`fit`, `win_probability`, `context_statistics`, `_prefix_summaries`, `_query_features`, `save_artifact`, `load_artifact`) | `exp-e7-neural-context` | E7 hypernetwork |
| `mainFiles/entry/predict.py` | `main()` | `entry-final` | Final blend, env setup, fallback |
| `mainFiles/entry/models/state.pt` | torch payload, `format_version: 1` | `entry-final` | E7 trained weights |
| `mainFiles/entry/models/state.pt.json` | manifest | `entry-final` | Human-readable copy, no weights |
| `mainFiles/entry/models/config.json` | config and metadata | `entry-final` | Constructor settings, training diagnostics, input hashes |
| `mainFiles/entry/README.md` | entry docs | `entry-final` | Team ID TBD; forecast SHA-256 |
| `mainFiles/entry/requirements.txt` | pins | `entry-final` | numpy, sklearn, scipy, joblib, threadpoolctl, torch |
| `mainFiles/entry/reproduce.json` | organiser command | `entry-final` | `python predict.py --data /data --out /output/submission.csv` |
| `mainFiles/exp_eval.py` | `CANDIDATES`, `main()` | `exp-e11-flex-experts` (E6/E7 branches have copies without E6/E7) | Tuning-only scorer |
| `mainFiles/slot_forecasting/shape_head.py` | `ShapeHeadForecaster` | `exp-e5-shape-head` | E5 |
| `mainFiles/slot_forecasting/episode_memory.py` | `EpisodeMemoryForecaster` | `exp-e4-episode-memory` | E4b, reuse for Idea C |
| `mainFiles/slot_forecasting/flex_experts.py` | `FlexEpisodeForecaster` | `exp-e11-flex-experts` | E11 |
| `mainFiles/slot_forecasting/adaptive_memory.py` | `AdaptiveMemoryForecaster` | `exp-e2-adaptive-memory` | E2 |
| `mainFiles/slot_forecasting/win_blend.py` | `WinBlendForecaster` | `exp-e9-win-blend` | E9 short-history shrinkage |
| `mainFiles/slot_forecasting/machine_trees.py` | `MachineTreeForecaster` | `exp-e10-machine-trees` | E10 |
| `mainFiles/slot_forecasting/combo_trees_blend.py` | `ComboForecaster` | `exp-combo-trees-blend` | E9 + E10 |
| `mainFiles/slot_forecasting/neighbor_inputs.py` | `NeighborForecaster` | `exp-e8-neighbor-inputs` | E8 |
| `mainFiles/rule_model.py` | `RuleForecaster`, `default_rules` | `zhill-branch` | Team best, rule model |
| `mainFiles/predict.py` | `Blend`, `Weighted`, `RULE_WEIGHT` | `zhill-branch` | Team best, combined predict |
| `mainFiles/slot_forecasting/pair_mixer.py` | `PairMixer`, `discover_pairs`, `choose_pairs`, `adapt` | `pranav/adaptive-mixer` and `zhill-branch` | Team pair mixer |
| `tests/test_interaction_sweep.py` | 7 tests | `exp-e6-interactions` | E6 |
| `tests/test_neural_context.py` | 6 tests | `exp-e7-neural-context` | E7 |

Test counts from CODEX_REPORT (not re-run): E2 5, E3 4, E4 10, E6 7, E7 6. Each is a separate suite.

---

*End of handout.*
