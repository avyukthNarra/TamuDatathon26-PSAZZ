# E11: Episode Memory + Three Per-Machine Experts (Claude's best tree/linear model)

**Branch:** `avyukth/exp-e11-flex-experts` @ `f35bb67`
**Files:** `mainFiles/slot_forecasting/{pair_logistic,shape_head,win_blend,machine_trees,combo_trees_blend,episode_memory,flex_experts}.py`, `mainFiles/exp_eval.py`, `mainFiles/exp_e11.py`, `mainFiles/exp_e11_calib.py`
**Scores:** 40.256 tuning / 40.897 audit (kit `episode_folds` split, 48 tuning / 48 audit episodes)
**Sources:** branch files at `origin/avyukth/exp-e11-flex-experts`, commit bodies from `origin/main..origin/avyukth/exp-e11-flex-experts`, `HOUSE_EDGE_EXPERIMENT_RESULTS.md`, `worktrees/_coord/RESULTS.md`

---

## 1. TL;DR

- E11 is Claude's final tree/linear model. It is a stack of small layers, each one a subclass, and the final class is `FlexEpisodeForecaster`.
- It predicts P(win) per machine as a weighted mix of three learners (logistic with lag pairs, ExtraTrees, HistGradientBoosting). It shrinks short-history machines toward `pooled_recent`. It picks the memory length (recency half-life) once per episode, not per machine.
- Final setting: `FlexEpisodeForecaster(w_lr=0.25, w_et=0.25, w_hgb=0.5, et_leaf=2, hgb_min_leaf=3)`. Tuning 40.256, audit 40.897.
- Audit loss ratios: stable .179, short .696, drift .549, shared .443.
- Progression on tuning: pooled_recent 31.06, then E1 36.22, E5 36.45, E9 37.49, E10 37.79, combo 37.96, E4 39.04, E11 equal thirds 39.64, E11 final 40.26.
- **Honest context:** E11 is well below E6 (48.19 audit), E7 (48.99 audit), and the team's best (about 54.5). Its ceiling came from 0/1 feature coding. Five learner families on 0/1 lag features all stopped near 40. The representation, not the learner, was the limit (see section 6).
- **What still matters for other models:** episode-pooled memory selection, the short-history shrinkage gate, the separate shape head, the tuning-only scorer `exp_eval.py`, and an expert-mix pattern. Section 9 says how to port each one.

---

## 2. The layered design

### 2.1 Class hierarchy and MRO

`FlexEpisodeForecaster` inherits from `EpisodeMemoryForecaster` and `FlexTrees`. The resolved MRO is:

```
FlexEpisodeForecaster          flex_experts.py:80      defaults: mixing weights, leaf sizes
 -> EpisodeMemoryForecaster    episode_memory.py:23    per-episode half-life selection (E4)
 -> ComboForecaster            combo_trees_blend.py:11 pure composition, no body
 -> WinBlendForecaster        win_blend.py:15         short-history shrink (E9), defaults C=10
 -> FlexTrees                  flex_experts.py:28      3-expert mix, replaces win_probability
 -> MachineTreeForecaster      machine_trees.py:15     ExtraTrees kind/mix (mostly dead in E11)
 -> ShapeHeadForecaster        shape_head.py:23        type|signal shape prior (E5), shape()
 -> PairLogisticForecaster     pair_logistic.py:41     lag vectors, base pooled_recent, predict()
 -> object
```

Two things to notice:

1. **`win_probability` resolves to `FlexTrees` first.** `WinBlend.win_probability` calls `super()`, which goes to `FlexTrees`. `FlexTrees` does not call `super()`, so the `MachineTree` logic (ExtraTrees with `kind`, the `mix` blend) is never reached on the E11 path. `PairLogistic.win_probability` is reached only through the separate fast scorer that `_episode_losses` builds (a `PairLogisticForecaster` instance, not `self`).
2. **`predict()` resolves to `EpisodeMemoryForecaster.predict` first**, which loops over episodes and calls `super().predict(...)`. From there it runs `WinBlendForecaster.predict` (`win_blend.py:37`), which sets `self._base_pw` and then calls `self.win_probability`.

`pooled_recent` is not in the MRO. It is composed as `self.base = FrequencyForecaster("pooled_recent", half_life=24)` in `PairLogisticForecaster.__init__`.

### 2.2 Data flow for one query

```
predict(queries, history)                                  [EpisodeMemory]
 for each episode:
   seqs per machine (ticks < cutoff only)
   losses[hl] for hl in (None, 64, 32, 16)                 [_episode_losses, fast LR scorer]
   self.half_life = argmin(losses)
   super().predict(episode queries, episode history)       [WinBlend.predict]
     base = pooled_recent P(win) -> self._base_pw
     for each machine:
       pw = FlexTrees.win_probability(seq, cutoff)         [LR + ET + HGB weighted mean]
       if machine has < 100 observed spins: pw = 0.6*pw + 0.4*_base_pw
       clip pw to [1e-4, 1-1e-4]
       probs = [1 - pw, pw * shape]                        [shape from ShapeHead.shape]
```

---

## 3. Each layer in detail

Gains are tuning /60 unless noted. The layer order matches the progression.

### Layer 0: `pooled_recent` (kit baseline, `FrequencyForecaster`)

- Dirichlet shrinkage of each machine's recency-weighted (half-life 24) payout counts toward a `type|signal` prior, strength 20.
- Score: **31.06** tuning / 30.57 audit.
- Role: the fallback when a machine cannot support a fit, and the source of `_base_pw`.

### Layer 1: E1, per-machine lag-pair logistic (`pair_logistic.py`)

- **Hypothesis:** about 94% of the starter's error is the win chance `p0 = P(payout == 0)`. That depends on machine-specific interactions among the last 8 spins.
- **Method:** each queried machine gets its own L2 logistic regression for P(win). Features: win indicators (`payout > 0`) at tick lags 1..8, plus all 28 pairwise products. Rows are weighted `2^(-age/half_life)`. Shape given a win comes from `pooled_recent`'s positive part. A machine that cannot support a fit (too few rows, one class, or missing lags) falls back to `pooled_recent`.
- **Gain:** 31.06 to **36.22** at C=10 (C=3 default gave 35.41). Pairs are essential: without them the score falls to 29.46.

### Layer 2: E5, separate payout-shape head (`shape_head.py`)

- **Hypothesis:** `pooled_recent`'s positive part mixes win-rate evidence into the payout shape. Estimate the shape separately.
- **Method:** `ShapeHeadForecaster.fit` builds a `type|signal` prior over the four positive payouts (1, 2, 5, 20) from train history plus train labels, with pseudo-count 0.5. `shape()` mixes that prior with the machine's own positive spins, with prior strength `shape_strength` (50 is the chosen value).
- **Gain:** 36.22 to **36.45** (audit **38.08**). Small on tuning, but the audit number is the reason it was kept as the safe entry.
- **Rejected:** "big win" lag inputs (`size_inputs=True`) overfit: 34.51.

### Layer 3: E9, short-history shrinkage (`win_blend.py`)

- **Hypothesis:** a per-machine fit on 30 to 200 rows is noisy. Machines with fewer than 100 observed spins should lean on the baseline.
- **Method:** `pw = alpha * model + (1 - alpha) * base_pw`, with `alpha = short_alpha` (0.6) when `< 100` spins are observed before the cutoff, and `alpha = 1.0` otherwise. The count is observable at test time, so no condition label is used.
- **Gain:** 36.45 to **37.49** with short-history loss ratio .97 to .79.
- **Rejected:** global alpha 0.7 (37.17, helps less), 0.85 (37.09), and temperature sharpening `temp=0.8` (35.74). `temp=1.25` gave 36.94.

### Layer 4: E10, per-machine trees (`machine_trees.py`)

- **Hypothesis:** trees can represent higher-order lag rules that the pairwise logistic cannot.
- **Method:** per-machine ExtraTrees on raw 0/1 lags 1..8, with recency weights, then averaged 50/50 with the logistic.
- **Gain:** 36.45 to **37.79** (ET + LR 50/50). ExtraTrees alone gave 37.27 with the best drift score (.690). RandomForest was clearly worse (alone 35.56, mixed 37.29).
- **Reason ExtraTrees beats RF:** more randomised splits suit tiny samples.

### Layer 5: combo (E9 + E10, `combo_trees_blend.py`)

- `ComboForecaster(WinBlendForecaster, MachineTreeForecaster)` with no body. Just the MRO.
- **Gain:** **37.96** (mix 0.5, short alpha 0.6). The gains only partly stack: the trees already lift short history (.842 to .821), so the shrinkage adds about +0.17.

### Layer 6: E4, episode memory selection (`episode_memory.py`)

- **Hypothesis:** shared changes hit all 8 machines together, so memory length should be chosen per episode, using 8 times as much evidence as a per-machine choice.
- **Gain:** combo 37.96 to **39.04**. Stable .32 to .20, drift .73 to .68.
- Full details are in section 4.

### Layer 7: E11, three-expert mix (`flex_experts.py`)

- **Hypothesis:** the logistic, ExtraTrees, and HistGradientBoosting learners make different errors. A weighted mean should beat any single one.
- **Method:** `FlexTrees.win_probability` builds the training rows, fits each enabled expert, and returns the weighted mean `np.dot(preds, weights) / sum(weights)`. Optional logit calibration (`calib_a`, `calib_b`) is applied after the mean, but is off in the final setting.
- **Gain:** E4 39.04, equal thirds 39.64, then **40.26** with weights 0.25/0.25/0.5 (LR/ET/HGB), ET leaf 2, HGB min_leaf 3.

**Expert details (final setting):**

| Expert | Input | Key settings | Notes |
|---|---|---|---|
| Logistic | 0/1 lags 1..8 plus 28 pairs | `C` = 10 (from WinBlend default), `max_iter` 1000 | Weighted by half-life |
| ExtraTrees | 0/1 lags 1..8 (raw) | 100 trees, `min_samples_leaf` = 2, `max_features` = sqrt, seed 0, `n_jobs` 1 | Weighted by half-life |
| HistGradientBoosting | 0/1 lags 1..8 (raw) | `max_iter` 80, `learning_rate` 0.1, `max_leaf_nodes` 8, `min_samples_leaf` 3, `l2_regularization` 1.0, seed 0 | Weighted by half-life |

---

## 4. E4 episode memory in depth

### 4.1 The idea

Memory length (how far back training rows reach, set by the recency half-life) matters. Per machine it is noisy, because a machine has only about 200 spins. Shared changes move all machines in an episode at the same time, so the decision can be pooled across the episode.

### 4.2 Method

For each episode (`episode_memory.py:52-82`):

1. Candidate half-lives: `(None, 64, 32, 16)`, where `None` means no decay.
2. For each candidate `hl`, build a fast scorer: `PairLogisticForecaster(C=self.C, half_life=hl, min_rows=self.min_rows)`. This is the cheap logistic, not the full combo.
3. Prequential scoring, per machine: take the last `n_eval` = 48 observed ticks before the cutoff, sorted ascending, then every `stride` = 3rd one. For each such tick `t`, build a prefix containing only ticks `< t`, predict P(win) at `t`, and accumulate the squared error against the real outcome.
4. Sum across all machines in the episode. Loss = mean Brier over all scored (machine, tick) pairs.
5. Pick `hl` with the lowest loss. Ties go to the earlier candidate in the tuple, so `None` wins ties.
6. Set `self.half_life = hl` and run the full combo (here FlexTrees plus shrinkage) for every machine in that episode.

Only pre-cutoff spins are used. Each prequential target `t` is predicted from ticks strictly earlier than `t`.

### 4.3 Pseudo-code

```
EpisodeMemory.predict(queries, history):
    indexed = index history by (episode, machine)
    for each episode ep with cutoff c:
        seqs = [ {tick: payout for machine m in ep} ]           # 8 machines
        for hl in (None, 64, 32, 16):                           # losses[hl]
            scorer = PairLogistic(C=self.C, half_life=hl)       # fast 0/1 pair logistic
            total, n = 0.0, 0
            for seq in seqs:
                ticks = sorted(t for t in seq if t < c)[-48:][::3]
                for t in ticks:
                    prefix = {k: v for k, v in seq.items() if k < t}
                    pw = scorer.win_probability(prefix, t)     # None -> skip
                    if pw is None: continue
                    total += (pw - 1[seq[t] > 0]) ** 2
                    n += 1
            losses[hl] = total / n if n else inf
        self.half_life = half_lives[argmin(losses)]             # hard pick
        out.update( WinBlend.predict(ep queries, ep history) )  # full model
    return out
```

A `soft=True` option exists (`episode_memory.py:69-81`). It mixes the four half-lives with weights `exp(-(loss - min)/temperature)`. In the sweep it was no better than the hard pick (38.78 to 38.84 vs 39.04), so the final setting uses hard.

### 4.4 Why it works, and what it measured

- The per-episode pick lets one memory length serve all 8 machines when a shift is shared. The logs do not break down which half-life wins in which condition, so do not assume a mapping until you have printed `chosen` per episode.
- The hard pick costs one extra pass of the fast scorer per half-life (4 scorers per episode). Runtime is discussed in section 10.
- Stable loss dropped from .32 to .20 on tuning. Short and drift moved only a little (.80 and .68). Most of the 1.1-point gain comes from stable machines.

### 4.5 Mutable state

`predict` writes `self.half_life` and `self.chosen`. After a call, `self.half_life` holds the value of the last episode. Anything that reads it afterwards gets that stale value (see gotchas).

---

## 5. Full results tables

All numbers are tuning /60 unless marked. Loss ratios are stable / short / drift / shared, lower is better, 1.0 = baseline. Entries marked "n/a" were not logged as loss ratios.

### 5.1 Progression (the headline chain)

| Stage | Branch | Tuning /60 | Audit /60 |
|---|---|---|---|
| pooled_recent (kit) | `main` | 31.06 | 30.57 |
| E1 pair logistic, C=10 | `avyukth/exp-e1-pair-logistic` | 36.22 | n/a |
| E5 shape head, s=50 | `avyukth/exp-e5-shape-head` | 36.45 | 38.08 |
| E9 short shrink, alpha 0.6 | `avyukth/exp-e9-win-blend` | 37.49 | n/a |
| E10 ET + LR 50/50 | `avyukth/exp-e10-machine-trees` | 37.79 | n/a |
| Combo (E9 + E10) | `avyukth/exp-combo-trees-blend` | 37.96 | n/a |
| E4 episode memory, hard, n_eval 48 | `avyukth/exp-e4-episode-memory` | 39.04 | n/a |
| E11 equal thirds | `avyukth/exp-e11-flex-experts` | 39.64 | n/a |
| **E11 final (r3_h50_leaf2_ml3)** | `avyukth/exp-e11-flex-experts` @ `f35bb67` | **40.26** | **40.90** |

### 5.2 E1: pair logistic (`RESULTS.md` rows, `3f4ba83`)

| Variant | Tuning | stable | short | drift | shared | Note |
|---|---|---|---|---|---|---|
| C=3, hl 32 (default) | 35.408 | .442 | .951 | .821 | .564 | |
| No pairs | 29.458 | 1.010 | 1.188 | 1.000 | .950 | pairs essential |
| hl 16 | 33.482 | .715 | .964 | .835 | .655 | |
| hl 64 | 34.484 | .285 | .948 | .993 | .734 | stable wants long memory |
| C=1 | 33.997 | .616 | .929 | .835 | .679 | |
| **C=10** | **36.217** | .319 | .986 | .834 | .488 | less regularisation helps |

### 5.3 E5: shape head (`11e2d5e`)

| Variant | Tuning | stable | short | drift | shared | Note |
|---|---|---|---|---|---|---|
| C=10, shape prior only | 36.309 | .326 | .967 | .827 | .489 | |
| C=10, shape strength 50 | **36.451** | .307 | .972 | .821 | .484 | best E5 |
| C=10, s20 + size inputs | 34.512 | .383 | 1.093 | .845 | .633 | size inputs overfit |

Audit for `43f7de9`: **38.076** (stable .218, short .707, drift .835, shared .544).

### 5.4 E9 and E10 (`f8c0978`, `281eaa4`)

| Variant | Tuning | stable | short | drift | shared | Note |
|---|---|---|---|---|---|---|
| E9: short alpha 0.6 | **37.493** | .307 | .789 | .821 | .484 | short fixed |
| E9: global alpha 0.7 | 37.174 | .368 | .812 | .755 | .521 | |
| E10: ET + LR 50/50 | **37.791** | .321 | .821 | .729 | .480 | best single |
| E10: ET alone | 37.274 | .385 | .842 | .690 | .521 | best drift at the time |
| E10: RF alone | 35.560 | .458 | .929 | .729 | .633 | RF < ET |

Other readings from the commit bodies: E9 short-history alpha 0.3 gave 37.35, global alpha 0.85 gave 37.09, temp 1.25 gave 36.94, temp 0.8 gave 35.74. E10 RF + LR 50/50 gave 37.29. The commit bodies are the source for these; `RESULTS.md` logs only the rows in the tables above.

### 5.5 Combo (`39168b8`)

| Variant | Tuning | stable | short | drift | shared |
|---|---|---|---|---|---|
| **ET + LR 50/50, short alpha 0.6** | **37.964** | .321 | .792 | .729 | .480 |
| ET + LR 50/50, short alpha 0.8 | 37.969 | | | | |
| mix 0.5, short 0.6, global alpha 0.9 | 37.865 | | | | |
| mix 0.3, short 0.6 | 37.800 | | | | |
| mix 0.5, short 0.4 | 37.775 | | | | |
| mix 0.5, short 0.6, min_leaf 8 | 37.446 | | | | |

### 5.6 E4 episode memory (`0b3ff0d`)

| Variant | Tuning | stable | short | drift | shared | Time |
|---|---|---|---|---|---|---|
| **Hard, n_eval 48** | **39.037** | .202 | .796 | .681 | .470 | 234 s val |
| Hard, n_eval 24 | 38.825 | .238 | .796 | .681 | .467 | 153 s |
| Soft, temperature 0.005 | 38.840 | .246 | .794 | .681 | .458 | |
| Soft, temperature 0.02 | 38.782 | | | | | |

Run on an M3 Pro. The commit body estimates about 8 minutes on test (2 times the queries). See gotchas.

### 5.7 E11 round 1 (`c12a3ee`, `RESULTS.md` rows at 15:50)

| Variant | Tuning | stable | short | drift | shared | Note |
|---|---|---|---|---|---|---|
| **Equal thirds (LR/ET/HGB)** | **39.637** | .181 | .772 | .612 | .490 | new best at the time |
| E4 + HGB 0.2 weight | 39.443 | .188 | .781 | .638 | .479 | |
| E4 + ET min_leaf 2 | 39.355 | .180 | .784 | .688 | .446 | |
| Imputed lags (min obs 4 or 6), ET 300, short 0.4 or 0.8, LR .3/ET .7, ET max_features all | 38.46 to 39.03 | | | | | no gain vs 39.037 |

### 5.8 E11 round 2 (`c12a3ee`, `RESULTS.md` rows at 16:05)

| Variant | Tuning | stable | short | drift | shared | Note |
|---|---|---|---|---|---|---|
| **Equal thirds, ET leaf 2, HGB min_leaf 3** | **40.061** | .167 | .749 | .612 | .462 | best at the time |
| 0.25/0.25/0.5, ET leaf 2 | 40.028 | .166 | .757 | .584 | .489 | |
| HGB only | 39.634 | .174 | .752 | .525 | .605 | best drift |
| Train-fitted logit calibration | 39.073 | | | | | a = 1.016: already calibrated |

The calibration row does not name its base config in the log, so its 39.07 is not directly comparable to 40.06. The useful reading is a = 1.016, which says calibration is not the lever.

### 5.9 E11 round 3 (`f35bb67`, `RESULTS.md` rows at 16:15)

| Variant | Tuning | stable | short | drift | shared | Note |
|---|---|---|---|---|---|---|
| **0.25/0.25/0.5, ET leaf 2, HGB min_leaf 3 (final)** | **40.256** | .164 | .735 | .581 | .482 | chosen |
| Thirds, leaf 2, min_leaf 3, half-life 128 option | 40.088 | .162 | .751 | .612 | .462 | within noise |
| Thirds, leaf 2, min_leaf 3, C=30 | 40.042 | .220 | .736 | .609 | .429 | best shared |
| Half-life 8 / n_eval 64 / leaf 1 / min_leaf 2 / short 0.8 | 39.30 to 40.01 | | | | | no gain |

### 5.10 E11 round 4 (`f35bb67`, `RESULTS.md` row at 16:22)

Seven variants around the best (half-life 128, weights 0.2/0.2/0.6, C=30, HGB iterations and leaves) scored 39.86 to 40.32. That is a plateau within tuning noise. The 0.2/0.2/0.6 variant is marginally higher (40.32 is the top of the range) but more extreme, so the team kept the round-3 setting. The commit says so explicitly.

### 5.11 Audit (one look, logged)

| Model | Branch @ SHA | Audit /60 | stable | short | drift | shared |
|---|---|---|---|---|---|---|
| pooled_recent | `main` | 30.565 | .970 | 1.023 | .981 | .878 |
| E5 (safe entry #1) | `exp-e5-shape-head` @ `43f7de9` | 38.076 | .218 | .707 | .835 | .544 |
| **E11 final** | `exp-e11-flex-experts` @ `f35bb67` | **40.897** | .179 | .696 | .549 | .443 |

Audit was looked at for these rows only. See section 7.3.

---

## 6. What didn't help, and why

| Idea | Result | Why it failed (as far as the logs say) |
|---|---|---|
| Drop the lag pairs (E1) | 29.46 | Pairs carry the machine-specific rules. Pooled lags explain about 20% of the p0 variance. |
| "Big win" size inputs (E5) | 34.51 | Overfit: short history 0.98 to 1.09, shared 0.49 to 0.63. |
| Neighbour-machine inputs (E8) | 36.51 (+0.06) | Noise-level, and per-neighbour features overfit (34.62). Tested only in the 0/1 model. |
| Half-life 8 or 128 as a fixed choice (E11) | 40.09 vs 40.06 at 128 | Within noise. The 128 option is kept in the sweep, not the final. |
| Soft half-life weighting (E4) | 38.78 to 38.84 | No better than the hard pick. Hard pick is simpler and faster. |
| n_eval 24 instead of 48 (E4) | 38.83 | Less evidence per episode, and it lost 0.2. |
| Imputed missing lags (E11) | 38.46 to 39.03 | Filling unknown lags with 0.5 added noise rows. Helped nothing. |
| 300 trees (E11) | No gain | The ET is already stable at 100 trees. |
| Train-fitted logit calibration (E11) | a = 1.016 | The model is already calibrated. Nothing to fix. |
| C = 30 (E11) | 40.04 vs 40.06 | Within noise. Shared improved (.429 vs .462) but stable got worse (.220 vs .167). |
| ET max_features = all (E11) | No gain | Sqrt features give the trees more randomisation. |
| Short alpha 0.4 or 0.8 (E11) | No gain | 0.6 is the flat optimum. |
| Temperature sharpening (E9) | 35.74 (temp 0.8), 36.94 (temp 1.25) | Miscalibrated either way. Only the shrink toward base worked. |
| Global alpha instead of short-only (E9) | 37.17 (alpha 0.7), 37.09 (alpha 0.85) | Shrinks machines with long history too. The logs do not isolate why, but the short-only gate scored higher. |
| Random forest instead of ExtraTrees (E10) | 35.56 alone | Less randomised splits overfit tiny samples. |
| Min_leaf 8 for trees (combo) | 37.45 | Too smooth for 30 to 200 rows. |
| Codex E3 (global pair shortlist) | Negative (best 33.96 vs 35.41) | Which pairs matter differs per machine. A global shortlist throws away each machine's own rule. |
| Codex E4 (shared change-point detector) | Weak (36.06 vs 35.41, CI spans zero) | Hard boundaries are brittle. False positives on stable episodes hurt. |

**Overall lesson from the failures:** the tree/linear family on 0/1 features saturates near 40. E6 showed that the signed encoding with L1 is worth about 8 points (see section 9).

---

## 7. How to run

### 7.1 Setup

Run from `mainFiles/`, with `/opt/miniconda3/bin/python3` (Python 3.13, sklearn 1.7.2, numpy 2.3.3). On macOS with conda, importing torch after sklearn crashes with `OMP: Error #15`. E11 does not import torch, but if your shell does, export `KMP_DUPLICATE_LIB_OK=TRUE` first.

The scripts expect the kit data in `mainFiles/public/` by default (`exp_eval.py:45`). Use `--data <path>` to point elsewhere.

### 7.2 `exp_eval.py`: the tuning-only scorer

```bash
cd mainFiles
python exp_eval.py pooled_recent e5_s50              # tuning only (default)
python exp_eval.py e1_default --json out.json        # write per-candidate results
python exp_eval.py e1_default --audit                # ALSO score audit (log it)
```

Output per candidate: tuning points and per-condition loss ratios, plus the seconds taken. The scorer is `subset_score` on `episode_folds`, the same split and scorer the kit's `select_model` uses (`exp_eval.py:18`, `57`, `63`).

### 7.3 `exp_e11.py`: the E11 candidate set

```bash
cd mainFiles
python exp_e11.py r3_h50_leaf2_ml3                   # E11 final setting, tuning only
python exp_e11.py r3_h50_leaf2_ml3 r3_h33_leaf2_ml3  # compare two candidates
python exp_e11.py r3_h50_leaf2_ml3 --audit           # AUDIT LOOK: log it
```

`exp_e11.py` imports `exp_eval`, adds the E11 candidates to `CANDIDATES`, and calls `exp_eval.main()`. So every `exp_eval` flag works here too.

### 7.4 `exp_e11_calib.py`: train-fitted calibration

```bash
cd mainFiles
python exp_e11_calib.py <out.json>
```

It fits the model on train, predicts train queries, then fits `win ~ a * logit(pw) + b` with `LogisticRegression(C=1e6)` (`exp_e11_calib.py:25-29`). It writes `{a, b}` to the JSON file and scores two candidates on tuning. The run on this branch gave a = 1.016.

### 7.5 `--audit` policy

- Audit is 48 episodes that the tuning sweep must not see. Each audit look spends it a little.
- Default to tuning only. Pass `--audit` only for a candidate you have already chosen on tuning.
- Log every audit look in `worktrees/_coord/RESULTS.md` (branch @ SHA, candidate name, numbers). Those rows are the audit record.
- Do not use `select_model` or `run_experiments.py` for sweeps. They score audit automatically.
- Contamination note: the first diagnostics looked at all validation targets, audit included, and informed E1's design. Treat the audit as strong evidence, not a blind holdout.

### 7.6 Adding a candidate

Two ways, both zero-argument factories:

```python
# in exp_e11.py, inside the exp_eval.CANDIDATES.update({...}) block:
"my_variant": lambda: F(w_lr=0.25, w_et=0.25, w_hgb=0.5, et_leaf=2, hgb_min_leaf=3, C=10.0),

# or, in exp_eval.py, CANDIDATES dict (lines 24-39)
```

Name it with the round prefix (`r5_...`), run it on tuning, and log the result. Run `--audit` only after the candidate is chosen.

---

## 8. Code map

All line numbers refer to the `f35bb67` versions.

### `mainFiles/slot_forecasting/pair_logistic.py` (93 lines)

| Line | Symbol | What it is |
|---|---|---|
| 27-31 | `_lag_vector(seq, tick, lags)` | 0/1 win indicators at tick lags 1..L, `None` if any lag missing |
| 34-38 | `_expand(x, pairs)` | Appends the lag-pair products |
| 41 | `class PairLogisticForecaster` | Base class. Extension point is `win_probability` |
| 42-47 | `__init__` | Defaults: `lags=8`, `pairs=True`, `C=3.0`, `half_life=32`, `min_rows=12`, `base_half_life=24`, `clip=1e-4` |
| 53-75 | `win_probability(seq, cutoff)` | Per-machine LR. Cutoff guard at 60-61 |
| 77-93 | `predict(queries, history)` | Combines `[1 - pw, pw * shape]`. Falls back to pooled_recent at 86-87 |

### `mainFiles/slot_forecasting/shape_head.py` (103 lines)

| Line | Symbol | What it is |
|---|---|---|
| 20 | `POSITIVE = (1, 2, 5, 20)` | The four positive payout categories |
| 23 | `class ShapeHeadForecaster` | Subclasses PairLogistic |
| 24-29 | `__init__` | `shape_strength=20`, `shape_half_life=None`, `size_inputs=False` |
| 31-44 | `fit` | Builds the `type|signal` prior from train history and labels (prior at line 43) |
| 46-54 | `_features` | Lag features, with optional size inputs |
| 56-79 | `win_probability` | Only overrides when `size_inputs` is on |
| 81-88 | `shape(query, seq)` | Mixes the prior with the machine's positive spins |
| 90-103 | `predict` | Uses `shape()` for the positive part |

### `mainFiles/slot_forecasting/win_blend.py` (54 lines)

| Line | Symbol | What it is |
|---|---|---|
| 15 | `class WinBlendForecaster` | E9 shrinkage. Subclasses ShapeHead |
| 16-22 | `__init__` | `alpha=1`, `temp=1`, `short_alpha=None`. Defaults `C=10`, `shape_strength=50` |
| 24-35 | `win_probability` | Calls `super()`, applies temp (28-31), short gate (33-34), blend (35) |
| 33-34 | short gate | `< 100` observed spins before cutoff uses `short_alpha` |
| 37-54 | `predict` | Sets `self._base_pw` at 47, then calls `self.win_probability` |

### `mainFiles/slot_forecasting/machine_trees.py` (46 lines)

| Line | Symbol | What it is |
|---|---|---|
| 15 | `class MachineTreeForecaster` | E10. Subclasses ShapeHead |
| 16-21 | `__init__` | `kind="extra"`, `n_estimators=100`, `min_leaf=4`, `mix=0.0`, `seed=0` |
| 23-45 | `win_probability` | ExtraTrees or RF (39-41). Mix blend (43-45) is not reached in E11 |

### `mainFiles/slot_forecasting/combo_trees_blend.py` (12 lines)

| Line | Symbol | What it is |
|---|---|---|
| 11-12 | `class ComboForecaster(WinBlendForecaster, MachineTreeForecaster)` | Composition only |

### `mainFiles/slot_forecasting/episode_memory.py` (82 lines)

| Line | Symbol | What it is |
|---|---|---|
| 23 | `class EpisodeMemoryForecaster(ComboForecaster)` | E4 |
| 24-33 | `__init__` | `half_lives=(None,64,32,16)`, `n_eval=24`, `stride=3`, `soft=False`, `temperature=0.02`. Defaults `kind="extra"`, `mix=0.5`, `short_alpha=0.6` |
| 35-50 | `_episode_losses(seqs, cutoff)` | Prequential Brier per half-life, summed across the episode |
| 52-82 | `predict` | Per-episode loop |
| 64-68 | hard pick | `self.half_life = half_lives[argmin(losses)]` |
| 69-81 | soft branch | Exp-weighted mix. Not used in final |

### `mainFiles/slot_forecasting/flex_experts.py` (92 lines)

| Line | Symbol | What it is |
|---|---|---|
| 28 | `class FlexTrees(MachineTreeForecaster)` | The 3-expert mix. Replaces `win_probability` |
| 29-35 | `_fx(seq, tick)` | Lag features. Missing lags become 0.5 only if `impute_min_obs` is set |
| 37-77 | `win_probability` | Builds rows, fits enabled experts |
| 57-60 | LR expert | Pair-expanded features, `C=self.C` |
| 61-65 | ET expert | `n_estimators=et_trees`, `min_samples_leaf=et_leaf`, `max_features=et_max_features` |
| 66-71 | HGB expert | `max_iter=hgb_iter`, `learning_rate=hgb_lr`, `max_leaf_nodes=hgb_leaves`, `min_samples_leaf=hgb_min_leaf`, `l2_regularization=1.0` |
| 72 | weighted mean | `np.dot(preds, weights) / sum(weights)` |
| 73-76 | calibration | Only if `calib_a` or `calib_b` is non-default |
| 80 | `class FlexEpisodeForecaster(EpisodeMemoryForecaster, FlexTrees)` | Final class |
| 81-83 | `__init__` defaults | `w_lr=0.5`, `w_et=0.5`, `w_hgb=0.0`, `et_leaf=4`, `hgb_min_leaf=6`. **Not the final setting** |
| 84-85 | kw defaults | `n_eval=48`, `stride=3` |

### `mainFiles/exp_eval.py` (78 lines)

| Line | Symbol | What it is |
|---|---|---|
| 24-39 | `CANDIDATES` | Name to zero-arg factory. E1, E5 entries |
| 42-75 | `main()` | Loads data, fits, scores tuning, optional audit |
| 45 | `--data` default | `mainFiles/public` |
| 46 | `--audit` flag | Explicit audit look |
| 63-65 | scoring | `subset_score` on tuning, then audit if requested |
| 73-74 | `--json` | Writes results |

### `mainFiles/exp_e11.py` (41 lines)

| Line | Symbol | What it is |
|---|---|---|
| 1-2 | imports | `exp_eval`, `FlexEpisodeForecaster as F` |
| 3-40 | `CANDIDATES.update({...})` | All E11 candidate names |
| 24 | `r3_h50_leaf2_ml3` | **The final setting** |
| 33 | `r4_best_hl128` | Round-4 variant with half-lives including 128 |

### `mainFiles/exp_e11_calib.py` (35 lines)

| Line | Symbol | What it is |
|---|---|---|
| 22-23 | fit and predict on train | Uses `F()` with defaults, not the final setting |
| 25-29 | calibration fit | `LogisticRegression(C=1e6)` on logit(pw) |
| 32-33 | candidates | `e11_calib`, `e11_calib_half` |
| 34 | argv reset | Drops any flags passed on the command line |

---

## 9. Reusable ideas for other models, and how to port each one

### 9.1 Episode-pooled memory selection (E4)

- **What it is:** pick the recency half-life once per episode by pooled prequential Brier across all machines.
- **Why it transfers:** any model with a recency parameter can use this. Shared changes are the common failure, and pooling fixes it.
- **How to port:**
  1. Write a fast scorer with the same `win_probability(prefix, t)` signature. The pair logistic is the reference.
  2. Copy `_episode_losses` from `episode_memory.py:35-50` and point it at your scorer.
  3. Copy the hard-pick block (`episode_memory.py:64-68`).
  4. Keep `n_eval=48`, `stride=3`, and the candidate set `(None, 64, 32, 16)` as starting values.
- **Porting to E6 or E7:** HOUSE_EDGE section 7 proposes this for the signed-L1 model. E6 currently uses a fixed half-life of 16. The scorer would need to be a fast version of the signed model, or its pair-logistic substitute.

### 9.2 Short-history shrinkage gate (E9)

- **What it is:** for machines with fewer than 100 observed spins, pull P(win) toward the pooled baseline with alpha 0.6.
- **Why it transfers:** the gate uses an observable count, so it is legal at test time. Short history is the weakest condition in most models.
- **How to port:** wrap any model's per-machine P(win) with the 4-line block in `win_blend.py:33-35`. Use your model's own count of observed spins before the cutoff.
- **Check:** short loss ratio should drop. Here it went from .972 (E5) to .789 (E9 alone), and the E11 final reads .735 on tuning.

### 9.3 Separate payout-shape head (E5)

- **What it is:** estimate P(payout | win) from a `type|signal` prior over the four positive payouts, blended with the machine's own positive spins. Do not borrow the shape from a win-rate model.
- **Why it transfers:** it stops win-rate evidence leaking into the shape. The gain was small but the audit confirmed it.
- **How to port:** copy `ShapeHeadForecaster.fit` (`shape_head.py:31-44`) and `shape()` (`81-88`). Use strength 50.

### 9.4 Tuning-only scorer (`exp_eval.py`)

- **What it is:** a registry of named candidates scored only on the 48 tuning episodes. Audit is opt-in and logged.
- **Why it transfers:** it prevents accidental audit spending, which is the most common way to fool yourself on a small validation set.
- **How to port:** copy `exp_eval.py` and the `CANDIDATES` dict pattern. Keep the `episode_folds` split and the `subset_score` call.

### 9.5 Expert mix with weights (E11)

- **What it is:** several learners fitted on the same rows, with a fixed weighted mean of their P(win). Weights chosen on tuning. Final 0.25/0.25/0.5.
- **Why it transfers:** the experts make different errors. HGB alone gave the best drift score (.525) and the E4 model with ET min_leaf 2 gave the best shared score in round 1 (.446). The mix aims to keep both strengths.
- **How to port:** copy the loop in `flex_experts.py:57-72`. Keep the weights as constructor arguments so a sweep can change them.

### 9.6 Never-forget expert for stable machines

This line of work found that stable machines want infinite memory. E1's half-life sweep showed it (half-life 64 lowered stable to .285, against .442 at half-life 32), and so did E4's choices, since stable episodes usually picked "None". The lead ported the idea to the team's pair mixer (`origin/zhill-branch`, `slot_forecasting/pair_mixer.py`) in a scratch test on the tuning half only. The run isn't on a branch:

| PairMixer(pairs=((1,3),(2,5),(4,7)), …) | Tuning /60 | stable ratio |
|---|---|---|
| `half_life=(12, 40)` (default) | 54.075 | .107 |
| `half_life=(12, 40, 1e9)` (adds a never-forget expert) | **54.176** | **.099** |

The gain is +0.10, within noise and not yet audited. Codex is testing it inside the full rule + mixer blend on `avyukth/exp-zhill-improve`.

### 9.7 Per-machine tree on raw lags (E10)

- **What it is:** ExtraTrees on the raw 0/1 lag vector, per machine. The trees find higher-order rules without being told which pairs matter.
- **Caveat:** the representation caps it near 40. Only use trees as a complement to a signed or sparse model, not as the main model.

---

## 10. Gotchas

1. **Class defaults are not the final setting.** `FlexEpisodeForecaster()` with no arguments gives weights 0.5/0.5/0.0, `et_leaf=4`, `hgb_min_leaf=6`. That is the older LR+ET mix. Always pass the final values, or use `exp_e11.py r3_h50_leaf2_ml3`. `exp_e11_calib.py` uses the bare `F()` defaults, so its calibration was fitted on the older setting.
2. **Dead parameters in E11.** `kind`, `mix` (in `MachineTreeForecaster`), and `size_inputs` (in `ShapeHeadForecaster`) have no effect on the E11 path. `FlexTrees.win_probability` does not call `super()`, so the ExtraTrees-or-RF branch and the mix blend never run. Do not tune them expecting a result.
3. **`half_life` is mutated state.** `EpisodeMemoryForecaster.predict` sets `self.half_life` per episode and leaves the last value behind. Build a fresh model per run, or do not read `half_life` after `predict`.
4. **The cutoff guard.** `win_probability(seq, cutoff)` must ignore ticks at or after the cutoff. History files stop before the cutoff, but prequential callers pass prefixes. `pair_logistic.py:60-61` and `shape_head.py:85` implement it. `flex_experts.py:42-43` does too (the loop skips `tick >= cutoff`).
5. **Tick lags are not row offsets.** Use `seq.get(tick - lag)`. Rows with missing lags are skipped. Imputing them did not help.
6. **Short-history count is strict.** The gate counts observed ticks before the cutoff. A machine with 99 observed spins gets shrunk. One with 100 does not.
7. **Runtime differs by source.** `RESULTS.md` logs E11 at about 310 s per validation run on one thread (commit `c12a3ee`). The HOUSE_EDGE doc says about 4 to 5 minutes. E4 alone took 234 s on an M3 Pro (`0b3ff0d`).
8. **Packaged entry is the E4 stage, not E11.** `avyukth/entry-main` @ `3ef911b` packages `EpisodeMemoryForecaster(n_eval=48, stride=3)` with ET (100 trees, min leaf 4) and LR (C=10). It is not the E11 final. Team notes quote about 5 minutes on test for that entry, but the `0b3ff0d` commit body estimates about 8 minutes, since test has twice the queries. Measure before quoting a number.
9. **Two cores at grading time.** The entry sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` to 2 before importing numpy or sklearn (`3ef911b`). Keep that order.
10. **`e1_C10_ref` is a duplicate** of `e1_C10` in `exp_eval.py:31-32`. Harmless. Leave it or remove it, but not in a combined change.
11. **Audit is not pristine.** See section 7.5. Log every look.
12. **Branch names.** A local branch named `avy` exists, so `avy/...` names fail. Use `avyukth/...`. The git stash is shared across worktrees, so do not use bare `git stash`.
13. **Training labels in the shape prior.** `ShapeHeadForecaster.fit` adds train labels to the `type|signal` prior when `labels` is given. Predictions on train queries are therefore not fully out-of-sample for that part. This matters for calibration on train (`exp_e11_calib.py`), though the fitted a = 1.016 says it is small.
14. **Do not invent new half-life candidates without a sweep.** The `(None, 64, 32, 16)` set was chosen on tuning. Adding 8 or 128 gave no gain. A new set needs its own tuning run and an audit log.
15. **Data files.** `exp_eval.py` expects the kit data under `mainFiles/public/`. Check it exists before running, or pass `--data`.

---

## Appendix: Quick reference

```bash
# Read the final code without checking out the branch
git show origin/avyukth/exp-e11-flex-experts:mainFiles/slot_forecasting/flex_experts.py

# Commit bodies with every result table
git log --format='%h %s%n%b' origin/main..origin/avyukth/exp-e11-flex-experts

# Run the final setting on tuning (from mainFiles/)
python exp_e11.py r3_h50_leaf2_ml3
```

Final setting, in one line: `FlexEpisodeForecaster(w_lr=0.25, w_et=0.25, w_hgb=0.5, et_leaf=2, hgb_min_leaf=3)`. Tuning 40.256, audit 40.897.
