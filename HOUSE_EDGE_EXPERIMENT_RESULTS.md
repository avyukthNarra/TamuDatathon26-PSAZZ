# House Edge: Experiment Results and Handoff Notes

**Event:** TAMU Datathon 2026, main round · **Written:** November 7, 2026
**Authors:** Claude (Opus 5.5, Haiku subagents) and Codex (GPT-6.1-Sol, later GPT-6-Luna)
**Audience:** agents and teammates developing better methods. It records what was tried, how it works, what it scored, why things worked or failed, and where to look next.

Related files:
- `HOUSE_EDGE_BRAINSTORMING.md`: the original research map.
- `HOUSE_EDGE_ADDITIONAL_RESEARCH.md`: 20 more papers; untracked in the main checkout.
- `worktrees/_coord/RESULTS.md`: the append-only log of every run, with branch@SHA. `worktrees/` is git-excluded and local only.
- `worktrees/_coord/CODEX_REPORT.md`: Codex's detailed report, with bootstrap CIs and test counts.

---

## 1. Bottom line

| Model | Tuning /60 | Audit /60 | Where |
|---|---|---|---|
| Kit baseline `pooled_recent` | 31.06 | 30.57 | `main` |
| E5: per-machine pair-logistic + shape head (Claude) | 36.45 | 38.08 | `avyukth/exp-e5-shape-head` |
| E11: episode memory + 3 tree/linear experts (Claude) | 40.26 | 40.90 | `avyukth/exp-e11-flex-experts` |
| **E6: signed-L1 interaction logistic (Codex)** | **48.54** | **48.19** | `avyukth/exp-e6-interactions` |
| **E7: neural hypernetwork over prefix statistics (Codex)** | **50.18** | **48.99** | `avyukth/exp-e7-neural-context` |
| **FINAL: 50/50 probability blend of E6 + E7** | **50.65** | **50.02** | `avyukth/entry-final` @ `588a70b` |

**Score scale:** points = 60 / (1 + R). R is the mean over the 4 conditions of (our error / baseline error). 30 = baseline, 60 = perfect.

**The one-sentence lesson:** the next spin's win chance is driven by each machine's own small set of sign-sensitive two-spin rules ("did lags a and b match?"). Models that can find a few such rules per machine (L1 on ±1 products, or a network that maps match statistics to coefficients) beat everything else by 8+ points.

---

## 2. The problem in brief

- **Task:** for each query (one machine in one episode, at a cutoff tick), predict the next-spin probability over payouts **0, 1, 2, 5, 20**.
- **Loss:** the squared error between our 5-vector and the simulator's true 5-vector, averaged within each episode, then within each condition, then divided by the frozen baseline's error for that condition.
- **Conditions,** with equal weight: `stable`, `short_history`, `drift`, `shared_change`. Labels exist for train and validation only. **Test has no condition labels**, so any condition-specific behaviour must come from observable evidence.
- **Points:** main round 60 points, adaptation round 40 (opens 9 PM CT Nov 7, closes 11 AM CT Nov 8). The brainstorm file's "80/20" is wrong; RULES.md and the scorer say **60/40**.
- **Data:** see §3. `public/{train,validation,test}/history.csv` has columns `episode_id, machine_id, tick, payout`. `queries.csv` adds `machine_type` (steady/balanced/jackpot), `signal` (−1/0/1, per query only, **not** per tick) and `cutoff`.

---

## 3. Data facts that shaped every decision

| Fact | Number | Consequence |
|---|---|---|
| Episodes / machines / queries | train and validation: 96 episodes × 8 machines, 768 queries each. Test: 192 episodes, 1,536 queries | Only 768 labelled queries per split. Learn from every **history tick** as a next-spin example instead (about 140k per split). |
| Spins per machine | 64 for short-history machines (40 recent consecutive + 24 sampled older), otherwise about 192–256 | Short histories have only about 30 rows with all 8 lags observed. |
| Missing ticks | Only in short_history machines (about 70% of their span) | Lags are **tick** lags, not row offsets. A missing tick is unknown, never a loss. |
| True p0 = P(payout = 0) | Mean 0.56, **std 0.39**, ranging from about 0.04 to 0.99 | The truth swings hard from spin to spin. The baseline's p0 std is only 0.18, far too flat. |
| Error split (Codex oracle) | True p0 + type-average positive shape → error **0.011**; baseline → 0.192 | **About 94% of the error is the win chance.** Given a win, the 1/2/5/20 split is close to fixed per type. |
| Pooled lag dependence | Last-8 win indicators explain **R² = 0.20** of true p0; correlations −0.21 to −0.26 at every lag | Machines respond in opposite directions, so pooling cancels the signal. **Fit per machine.** |
| Condition vs memory | Pair-logistic p0 error: stable is best with no decay (0.028), drift and shared are best with half-life about 32 | No single memory length works for every condition, so memory must adapt. |

Validation `targets.csv` contains the true probability vectors. Train has only sampled outcomes.

---

## 4. Evaluation protocol (keep using it)

- **Scorer:** `mainFiles/slot_forecasting/experiments.py`, using `episode_folds()` and `subset_score()`. This is the same split and scorer the kit's `select_model` uses.
- **Split:** `episode_folds` divides the 96 validation episodes into **48 tuning / 48 audit**, alternating within each condition, so 12 + 12 per condition.
- **Don't call `select_model` or `run_experiments.py` for sweeps.** They always score audit for the chosen model, which spends the audit half. Use `mainFiles/exp_eval.py` (tuning only by default, `--audit` is explicit). It's on any `avyukth/exp-*` branch from E1 onward.
- **Data rules:**
  - Fit global pieces (priors, networks) on **train** only.
  - Each machine's per-query fit uses only that machine's ticks **< cutoff**.
  - Test histories were never pooled into training.
- **Audit looks so far:** all are logged in RESULTS.md. They cover `pooled_recent`, E5, E11, E6 (plus 2 ablations), E7, and the E6+E7 blends.
- **⚠️ Contamination:** the very first diagnostics (`_coord/diag.py`, `diag2.py`) looked at **all** validation targets, audit included, which informed E1's design (pairs, C, half-life). Treat audit as strong evidence, not a blind holdout.
- **Noise level:** with 12 episodes per condition per half, differences under about ±0.3 points are noise. Codex's paired episode bootstraps (in CODEX_REPORT.md) are the right tool for close calls.

### Model interface

Every forecaster has the same interface:

```python
model = SomeForecaster(**params).fit(train_queries, train_history, train_labels)
pred = model.predict(queries, history)   # {query_id: (p0, p1, p2, p5, p20)}
```

Most models subclass `PairLogisticForecaster` (`slot_forecasting/pair_logistic.py`) and override **`win_probability(seq, cutoff) -> float | None`**:
- `seq` is a `{tick: payout}` dict for one machine.
- It returns P(win), or `None` to fall back to `pooled_recent`.

`predict` combines the result as `[1 − pw, pw × positive_shape]`.

### Running things

All commands below run from `mainFiles/`, with Python `/opt/miniconda3/bin/python3` (3.13, sklearn 1.7.2, numpy 2.3.3, torch 2.9.1).

```bash
# Claude's E1–E11 candidates (branch avyukth/exp-e11-flex-experts):
python exp_eval.py pooled_recent e5_s50            # tuning only
python exp_e11.py r3_h50_leaf2_ml3                 # E11 final setting
python exp_e11.py r3_h50_leaf2_ml3 --audit         # ⚠️ logs an audit look
```

```python
# E6 (branch avyukth/exp-e6-interactions):
from slot_forecasting.interaction_sweep import InteractionSweepForecaster
m = InteractionSweepForecaster(degree=2, encoding="signed", penalty="l1", C=0.5, half_life=16)

# E7 (branch avyukth/exp-e7-neural-context). Load the trained artifact, no refit:
from slot_forecasting.neural_context import NeuralContextForecaster
m = NeuralContextForecaster.load_artifact("<...>/e7-best-artifact/state.pt")
```

The E7 artifact lives at `worktrees/codex-exp/mainFiles/runs/e7-best-artifact/` and is also shipped at `worktrees/entry-final/mainFiles/entry/models/state.pt`.

**macOS + conda gotcha:** importing torch after sklearn crashes with `OMP: Error #15`. Set `KMP_DUPLICATE_LIB_OK=TRUE` before the imports; `predict.py` does this.

---

## 5. Experiments in detail

Scores are tuning /60 unless marked. Condition values are **loss ratios** (lower is better; 1.0 = baseline).

### E0: Kit baseline (`pooled_recent`)
- **Method:** Dirichlet shrinkage of the machine's recency-weighted (half-life 24) payout counts toward a `type|signal` prior, strength 20.
- **Result:** 31.06 tuning / 30.57 audit. The frozen reference (`pooled`, no recency) is 30.0 by definition.
- **Role:** the fallback and payout-shape source in most later models.

### E1: Per-machine lag-pair logistic (Claude)
- **Branch:** `avyukth/exp-e1-pair-logistic` @ `60f1daa` (cutoff guard added after `3f4ba83`). File: `pair_logistic.py`.
- **Hypothesis:** the win chance depends on machine-specific lag interactions.
- **Method:**
  - For each queried machine, fit `LogisticRegression(C)` for win/lose on its own pre-cutoff rows.
  - Features: 0/1 win indicators at tick lags 1–8, plus all 28 pairwise products.
  - Rows are weighted 2^(−age/half_life).
  - The shape given a win comes from `pooled_recent`'s positive part.
- **Results:**

  | Variant | Tuning | Note |
  |---|---|---|
  | C=3, half-life 32 | 35.41 | |
  | **C=10** | **36.22** | stable .319, short .986, drift .834, shared .488 |
  | No pairs | **29.46** | Worse than baseline |
  | Half-life 64 | | stable .285, drift .993 |
  | Half-life 16 | | stable .715 |

- **Takeaways:**
  - Pairs are essential.
  - Less regularisation helps, because the rules are strong.
  - The best memory length depends on the condition.
  - Short history barely improved (about 30 usable rows).

### E2: Per-machine adaptive memory (Codex)
- **Branch:** `avyukth/exp-e2-adaptive-memory` @ `6511dfb`, draft PR #8. File: `adaptive_memory.py`, 5 tests.
- **Method:**
  - For each machine, score candidate half-lives {None, 16, 32, 64} prequentially on its own trailing spins.
  - Each trailing spin t is predicted from spins < t only, with Brier loss, every `eval_stride`-th of the last `max_eval_rows`.
  - The candidates are combined with soft weights. The code computes exp(−(loss − best_loss) × T) with T = 128 (`adaptive_memory.py:194`), so a LARGER T means SHARPER selection, and T = 128 is close to a hard pick.
- **Results:**

  | Comparison | E2 | Fixed | Gain |
  |---|---|---|---|
  | vs fixed E1 at C=3, half-life 32 | 36.22 | 35.41 | +0.81, CI [+0.23, +1.38] |
  | **Matched at C=10** | **37.15** | 36.22 | **+0.93** |

  The gain is in stable (error .045 vs .066) and drift (.153 vs .168); short is unchanged.
- **Takeaway:** a real but modest gain. **Not yet applied to E6 or E7**, which is an open direction.

### E3: Learned lag-pair selection (Codex): negative
- **Branch:** `avyukth/exp-e3-pair-selection` @ `56b60d6`, draft PR #9.
- **Method:** screen which of the 28 pairs matter using TRAIN machines (top-K, short-support-only), then fit each machine on that sparse set.
- **Result:** every restriction was worse. The best restricted-8 setting scored 33.96 vs 35.41 with all pairs.
- **Why it failed:** **which** pairs matter differs by machine. A global shortlist throws away each machine's own rule. Per-machine sparsity (E6's L1) is the right version of this idea.

### E4 (Codex): Shared change-point detector: weak
- **Branch:** `avyukth/exp-e4-shared-change` @ `0e33eaa`, draft PR #10, 10 tests.
- **Method:**
  - A masked Bernoulli GLR finds a common change time across the episode's win streams, with machine-specific directions.
  - Rows before the change are down-weighted (before_weight 0.1).
- **Result:** best 36.06 vs 35.41, CI [−0.70, +2.01].
- **Detector behaviour:** it accepts boundaries in 11/12 shared and 10/12 drift episodes, but also 6/12 stable ones. The individual-machine detector gave no gain.
- **Why weak:** hard boundaries are brittle. False positives on stable episodes hurt.

### E4 (Claude): Episode-level memory selection: worked
- **Branch:** `avyukth/exp-e4-episode-memory` @ `0b3ff0d`. File: `episode_memory.py`.
- **Idea:** shared changes hit all 8 machines, so **pool the memory decision across the episode** for 8× more evidence than E2's per-machine choice.
- **Method:**
  - For each half-life in {None, 64, 32, 16}, run the fast pair-logistic prequentially on each machine's last 48 spins (every 3rd).
  - Sum Brier over the episode, pick the best half-life, and use it for the whole model in that episode.
- **Result:** combo 37.96 → **39.04** (stable .32 → .20, drift .73 → .68). Soft weighting was no better than a hard pick.
- **Takeaway:** pooled memory selection is robust. Applying it to E6/E7 is an open direction.

### E5: Separate payout-shape head (Claude)
- **Branch:** `avyukth/exp-e5-shape-head` @ `43f7de9`. File: `shape_head.py`.
- **Method:** fit P(payout | win) separately, as a `type|signal` prior over {1, 2, 5, 20} from train (strength 50), blended with the machine's own positive spins.
- **Results:**
  - **36.45** vs 36.22 for E1 at C=10 (small).
  - Audit **38.08**.
  - Adding "big win" (payout ≥ 2) lag inputs to the win model **hurt** (34.51), by overfitting.
- **Note:** E6 and E7 still use `pooled_recent`'s positive part as their shape. Now that the p0 error is much smaller, the shape share of the remaining error is larger, so recheck this on the blend.

### E8: Neighbour-machine inputs (Claude): no gain
- **Branch:** `avyukth/exp-e8-neighbor-inputs` @ `5c2bc3c`. File: `neighbor_inputs.py`.
- **Method:** add the other machines' earlier spins (mean win at lags 1..k, or each neighbour's lag 1) as features.
- **Result:** 36.51 (+0.06, noise). The per-neighbour version overfit (34.62).
- **Caveat:** tested only inside the 0/1 pair-logistic. The rules say "earlier outcomes on neighbouring machines can help". A sparse signed version (E6-style L1 over neighbour × own-lag products) or an episode summary fed to E7 is **untested**.

### E9: Short-history shrinkage (Claude)
- **Branch:** `avyukth/exp-e9-win-blend` @ `f8c0978`. File: `win_blend.py`.
- **Method:** for machines with fewer than 100 observed spins (visible at test time), pw = α·model + (1−α)·baseline pw.
- **Results:**
  - α = 0.6: **37.49**, with short going .97 → .79.
  - A global α of 0.7–0.85 also helped (about 37.1–37.2).
  - Temperature sharpening (T = 0.8) hurt.

### E10: Per-machine tree ensembles (Claude)
- **Branch:** `avyukth/exp-e10-machine-trees` @ `281eaa4`. File: `machine_trees.py`.
- **Method:** per-machine ExtraTrees or RandomForest on the raw 8 win indicators, recency-weighted. Optionally averaged with the logistic.

| Variant | Tuning |
|---|---|
| ExtraTrees + logistic 50/50 | **37.79** |
| ExtraTrees alone | 37.27 (best drift .690) |
| RandomForest + logistic | 37.29 |
| RandomForest alone | 35.56 |

- **Takeaway:** trees capture higher-order rules. ExtraTrees beats RF (more randomised splits suit tiny samples).

### Combo, then E11: Flexible expert mix (Claude)
- **Branches:** `avyukth/exp-combo-trees-blend` @ `39168b8`, then `avyukth/exp-e11-flex-experts` @ `f35bb67`. Files: `combo_trees_blend.py`, `flex_experts.py`.
- **E11 method:** E4 episode memory + E9 short shrinkage + a per-machine mix of three experts:
  - logistic (pairs, C=10)
  - ExtraTrees (100 trees, min_leaf 2)
  - HistGradientBoosting (80 iterations, learning rate 0.1, 8 leaves, min_leaf 3)
- **Results:**
  - Combo (E9 + E10): 37.96.
  - E11 equal thirds: 39.64.
  - **Weights 0.25 / 0.25 / 0.5: 40.26 tuning, 40.90 audit.**
  - Tuning plateaued at 40.15–40.32 across round-4 variants.
- **No gain from:**
  - imputing missing lags (rows with ≥4 or ≥6 of 8 lags)
  - 300 trees
  - train-fitted logit calibration (a = 1.016, already calibrated)
  - half-life 8 or 128
  - C = 30
- **Takeaway:** a strong ensemble of *0/1-coded* learners still capped at about 40. The ceiling was the representation, not the learner (see E6).

### E6: Signed-L1 interaction logistic (Codex): the breakthrough
- **Branch:** `avyukth/exp-e6-interactions` @ `c3ba387`, draft PR #11, 7 tests. File: `interaction_sweep.py`.
- **Method:**
  - Per machine, L1 logistic (`liblinear`, `intercept_scaling=100`, `random_state=0`) on **signed** (±1) win indicators at lags 1–8 plus all 28 pairwise products. Degree 3 adds triples.
  - Recency half-life 16, C = 0.5.
  - It's a 96-config sweep over encoding, degree, penalty, C and memory.
- **Results:**

  | Variant | Tuning | Audit | Note |
  |---|---|---|---|
  | **Signed + L1** | **48.54** | **48.19** | Tuning ratios stable .228, short .259, drift .264, shared .194. Audit .163 / .358 / .235 / .224 |
  | 0/1 coding + L1 (same C, half-life) | 32.43 | 32.04 | |
  | Signed + L2 | 38.70 | 40.79 | |

  Runtime is under 1 s for all of validation.
- **Why it works:**
  - With ±1 coding, the product x_a·x_b is +1 when lags a and b **match** and −1 when they differ: an XOR/equality feature.
  - With 0/1 coding the same rule needs several coefficients (main effects + product) to express. L1 penalises that, so it gets shrunk away.
  - With L2, the right features are kept but so is everything else, so predictions stay blurred.
  - **Signed + L1 = a sparse search for a few "these two spins match" rules per machine.** That looks like the simulator's actual structure.
- **Weak spot:** short history on audit (.358), with only about 30 rows to select sparse rules from.

### E7: Neural hypernetwork over prefix statistics (Codex)
- **Branch:** `avyukth/exp-e7-neural-context` @ `af645aa` (scored at `4e0267a`), draft PR #12, 6 tests. File: `neural_context.py`.
- **Architecture:** a *learned estimator*, i.e. meta-learning. It doesn't fit per machine at run time.
  - For a machine's exact-lag rows before the cutoff, build the 36 signed features z (8 lags + 28 pairs) and the target y ∈ {0, 1}.
  - Compute **prefix sufficient statistics**, all-history plus a recency-weighted copy (half-life 16):
    - mean(z), 36 values
    - mean((2y−1)·z), 36 values: the signed correlation of each feature with winning
    - the win rate
    - log(count)
  - An MLP (stats → 64 → ReLU → 37) outputs **per-machine coefficients** for the 36 signed features plus a bias.
  - logit = coef·z_query + bias.
  - The last layer is zero-initialised.
- **Training:**
  - Data: TRAIN history only, about 24 random prefixes per machine.
  - Split: 12,561 train examples and 4,365 holdout examples, by episode hash.
  - Loss and optimiser: BCE, Adam with lr 1e-3, batch 256.
  - Early stopping: patience 6, best epoch 13 of 19, holdout BCE .409.
  - Fewer than 16 context rows falls back to the pair-logistic.
- **Results:**
  - **50.18 tuning** (ratios .206 / .198 / .220 / .158) and **48.99 audit** (.181 / .238 / .233 / .246).
  - vs E6 on tuning: +1.65, CI [−0.06, +3.27].
  - Predicting validation takes about 2 s from the artifact.
- **Why it works:** it learns the mapping from "this machine's feature-outcome correlations" to sharp coefficients across all train machines. That's especially valuable for short histories (audit .238 vs E6's .358), where a per-machine L1 fit has too few rows.

### Final: 50/50 blend of E6 + E7 (Claude)
- **Method:** average the two 5-vectors and renormalise.
- **Results:**

  | Blend | Tuning | Audit | Audit ratios |
  |---|---|---|---|
  | **50/50** | **50.65** | **50.02** | .148 / .235 / .212 / .203 |
  | 70/30 (E6/E7) | 50.10 | 49.62 | |

- **Takeaway:** E6 and E7 make different errors. The blend is better on both halves.
- **Entry:** `avyukth/entry-final` @ `588a70b`, `mainFiles/entry/`.
  - `predict.py` refits E6 on `/data/train`, loads E7 from `models/state.pt`, blends them, and falls back to E6 alone if torch fails.
  - Runtime 8.2 s, deterministic.
  - `requirements.txt` pins numpy, scikit-learn, scipy, joblib, threadpoolctl and torch==2.9.1.

### Other entries built
| Entry | Branch @ SHA | Model | Audit |
|---|---|---|---|
| #1 | `avyukth/entry-main` @ `d3c5cbd` | E5 | 38.08 |
| #2 | `avyukth/entry-main` @ `3ef911b` | E4 (Claude) episode memory | (tuning 39.04) |
| #3 | `avyukth/entry-main` @ `dbb6e38` | E6 alone, no torch | 48.19 |
| **final** | `avyukth/entry-final` @ `588a70b` | E6 + E7 blend | **50.02** |

Build files live in each worktree's `mainFiles/runs/<entry>/`, as `entry.zip` + `submission.csv`.

---

## 6. What we learned (ranked by importance)

1. **Representation beats model family.** Going from 0/1 to ±1 lag coding with L1 was worth +8 points. Five learner families (logistic, RF, ExtraTrees, HGB, blends) on 0/1 features all capped at about 40.
2. **The signal is per machine, sparse and sign-sensitive.** Pooled models fail because machines respond oppositely. Global pair shortlists fail (E3) because each machine has its own rules.
3. **Amortised (meta-learned) estimation helps where data is thin.** E7 learns how to turn small-sample statistics into coefficients, and it improves short history most.
4. **Memory must adapt without condition labels.** Prequential memory selection works, per machine (E2, +0.9) and especially per episode (Claude's E4, +1.1 on the combo). Neither has been applied to E6/E7 yet.
5. **Hard change-point cuts are brittle** (Codex E4). Soft or pooled memory choices are safer.
6. **Neighbour inputs didn't help in the 0/1 model.** That's not conclusive for signed/sparse or E7-style models.
7. **The payout shape matters little:** about 6% of the original error, though a larger share now.
8. **Calibration is not the issue.** A train-fitted logit scale came out ≈ 1.

---

## 7. Most promising next directions

These are ranked by expected value per hour, and each includes the experiment to run first.

1. **Memory selection for E6 and E7.** Apply Claude's E4 episode-pooled prequential half-life choice ({None, 64, 32, 16}, last 48 spins, every 3rd) to the signed-L1 model, and/or E2's per-machine soft selection. E6 currently uses a fixed half-life of 16. Given E2/E4's gains on weaker bases, expect about +0.5–1.5. Code to reuse: `episode_memory.py` (`_episode_losses`) and `adaptive_memory.py`.
2. **Richer E7 inputs and ensembling.** Ideas:
   - several recency half-lives as extra statistic blocks (e.g. 8, 16, 32, 64, all)
   - degree-3 signed features
   - payout-size statistics
   - an **episode summary** of the other machines' statistics, aimed at shared_change
   - 3–5 seeds averaged

   Retraining takes minutes on CPU. Keep TRAIN-only fitting, and check for overfit on the hash holdout.
3. **Better short-history handling.** It's the weakest condition for E6 (audit .358). Options:
   - weight E7 more when the machine has fewer than 100 spins (an observable gate)
   - E9-style shrinkage on the blend
   - adapt E6's C to the row count
4. **Learned blend gate.** Instead of a fixed 50/50, choose the E6/E7 weight from observable features (history length, missing fraction, chosen half-life). Fit it on TRAIN prequential predictions, then confirm on tuning.
5. **Shape head on the blend.** Swap `pooled_recent`'s positive part for E5's `ShapeHeadForecaster.shape()`, then re-decompose the remaining error: oracle p0 + model shape vs model p0 + oracle shape.
6. **Signed sparse neighbour features.** Use L1 over products of the own-lag sign × a neighbour's lag-1 sign, or episode statistics fed to E7. This targets shared_change.
7. **Degree 3 and group-sparse penalties.** E6 swept degree 3, and the winner was degree 2. A hierarchical or group L1 (pairs only if their parents are present) might help short history.

**For round 2 (adaptation):** the rules say round 2 "changes which temporal relationships recur".
- E6 refits per machine at run time, so it adapts automatically.
- **E7 must be retrained on round 2's train split.** Its learned mapping may encode main-round regularities.
- Validate the blend weight again on round 2's validation, since E6 may deserve more weight if E7 transfers poorly.
- Main-round data is mounted at `/main_data` during adaptation. Pooling it into E7 training is a reasonable experiment.

---

## 8. Pitfalls and gotchas

- **Cutoff guard.** `win_probability(seq, cutoff)` must ignore ticks ≥ cutoff. History files stop before the cutoff, but prequential callers pass prefixes. E1/E5 add the guard explicitly (`60f1daa`, `43f7de9`).
- **Tick lags ≠ row offsets.** Use `seq.get(t − lag)`. Rows with any missing lag are skipped (imputing them didn't help).
- **`signal` is per query, not per tick.** Don't backfill it into historical training rows.
- **Audit budget.** `select_model` and `run_experiments.py` score audit automatically, so use `exp_eval.py`. Log every audit look in RESULTS.md.
- **Branches.**
  - A local branch named `avy` exists, so `avy/...` branch names fail. Use `avyukth/...`.
  - One branch per experiment: `avyukth/exp-<id>-<desc>`.
  - The git stash is shared across worktrees, so don't use bare `git stash`.
- **Torch + sklearn on conda/macOS.** You need `KMP_DUPLICATE_LIB_OK=TRUE` (OMP Error #15).
- **liblinear L1.** E6 uses `intercept_scaling=100` so the intercept isn't penalised away.
- **Submission constraints.**
  - Python 3.13 base with pinned `requirements.txt`; no network at prediction time.
  - 30 min, 2 CPUs, 16 GiB.
  - The ZIP opens to `README.md` + `reproduce.json`; command `python predict.py --data /data --out /output/submission.csv`.
  - CSV ≤ 10 MB, ZIP ≤ 100 MB. Use `build_entry.py --source <dir>` and `check_submission.py`.
- **Runtime.** E11 takes about 4–5 min on validation single-threaded, while E6 and E7 take seconds. The final entry runs in 8 s.

---

## 9. Code index

| File (`mainFiles/slot_forecasting/`) | Class | Branch |
|---|---|---|
| `pair_logistic.py` | `PairLogisticForecaster` (base, extension point `win_probability`) | `avyukth/exp-e1-pair-logistic` |
| `shape_head.py` | `ShapeHeadForecaster` | `avyukth/exp-e5-shape-head` |
| `neighbor_inputs.py` | `NeighborForecaster` | `avyukth/exp-e8-neighbor-inputs` |
| `win_blend.py` | `WinBlendForecaster` | `avyukth/exp-e9-win-blend` |
| `machine_trees.py` | `MachineTreeForecaster` | `avyukth/exp-e10-machine-trees` |
| `combo_trees_blend.py` | `ComboForecaster` | `avyukth/exp-combo-trees-blend` |
| `episode_memory.py` | `EpisodeMemoryForecaster` | `avyukth/exp-e4-episode-memory` |
| `flex_experts.py` | `FlexEpisodeForecaster` | `avyukth/exp-e11-flex-experts` |
| `adaptive_memory.py` | `AdaptiveMemoryForecaster` | `avyukth/exp-e2-adaptive-memory` (PR #8) |
| (pair selection) | see branch | `avyukth/exp-e3-pair-selection` (PR #9) |
| (shared change) | see branch | `avyukth/exp-e4-shared-change` (PR #10) |
| `interaction_sweep.py` | `InteractionSweepForecaster` | `avyukth/exp-e6-interactions` (PR #11) |
| `neural_context.py` | `NeuralContextForecaster` | `avyukth/exp-e7-neural-context` (PR #12) |
| `mainFiles/exp_eval.py` | tuning-only scorer | E1 onward |
| `mainFiles/entry/predict.py` | final blend entry | `avyukth/entry-final` |

None of these are merged to `main`. PRs #8–#12 are drafts based on the E1 branch.
