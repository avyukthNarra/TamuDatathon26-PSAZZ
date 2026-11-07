# Experiment log (House Edge, main round)

Running history, newest at the bottom. Not committed yet; we're iterating.
Scores are validation technical points out of 60 (reference pooled = 30.0).
"tune/audit" = the two halves from `slot_forecasting.experiments.episode_folds`.
Per-condition numbers are loss ratios vs the reference (lower is better).
Scripts live in `mainFiles/experiments/`; run them from `mainFiles/`.

## 2026-11-07

### 1. Starter, 6 baselines (`run_starter.py`)
global 26.73, context 29.69, empirical 29.92, **pooled 30.00** (default output),
recent 30.74, pooled_recent 30.83. Deterministic: two runs gave identical CSVs.

### 2. Included experiment loop (`run_experiments.py`)
12 candidates (pooled/recent/pooled_recent + lag-k local count models). Chose
pooled_recent, audit 30.56. Local lag-count models did not beat plain recency.

### 3. Grid on H (half_life) x alpha (prior_strength) (`experiments/tune_recency.py`)
Best H=16, alpha=10: tune 31.15, audit 30.62 (default H=24, alpha=20 audit 30.56).
Flat plateau, all cells within ~0.1 pt. **Null result: knob tuning is not the lever.**
short_history stays worse than reference (1.03).
Note: condition labels are not in test, so per-condition H/alpha can't be used at
test time (short_history is detectable from tick gaps though).

### 4. Data look
Per-machine lag correlations are weak (mean |r| 0.08-0.13 vs ~0.05 noise floor);
jackpot machines lean positive at lags 2-4, steady/balanced mixed in sign.

### 5. Gradient boosting on lag + EWMA + episode features (`experiments/gb_proto.py`)
sklearn HistGradientBoostingClassifier (lr 0.03, 200 iters, leaf 80, depth 4, l2 5).
Train rows = every 2nd tick (after 24 prior obs) of train histories, label = payout
at that tick, features use ticks strictly before it. Features: type/signal one-hot,
EWMA payout class freqs at the half-lives in `HL`, lags 1-8 (present, win, >=2, >=5),
other machines' recent win rate and mean payout in the episode.

| HL (EWMA half-lives) | all | tune | audit | stable | short | drift | shared |
|---|---|---|---|---|---|---|---|
| pooled_recent(10,16) baseline | 30.90 | 31.15 | 30.62 | 0.963 | 0.997 | 0.944 | 0.863 |
| 8,32 | 32.04 | 32.52 | 31.50 | 0.855 | 0.917 | 0.969 | 0.751 |
| **4,16,64,128** | **32.72** | 33.28 | 32.09 | 0.819 | 0.893 | 0.904 | 0.719 |
| 4,8,16,32,64,128 | 32.67 | 33.19 | 32.10 | 0.826 | 0.885 | 0.911 | 0.726 |

Takeaways: GB beats baseline by ~1.8 pts on all conditions once multiple memory
lengths are given (short vs long EWMA acts as a drift signal). Blending GB with
pooled_recent was worse than GB alone. Only 3 HL sets tried, picked looking at
validation, so audit is not fully independent.
Needs scikit-learn + numpy (not stdlib), so a pinned requirements.txt and the
team's OK before this can be submitted.

## Ideas queue
- Add explicit short-minus-long EWMA difference and recent-variance features
- Train on train+validation for the final fit (keep validation held out while tuning)
- More training rows (stride 1), per-class hyperparameters, seeds/bagging
- Per-type models, or type x lag interaction features
- Check test-time features for short-history gaps (missing ticks) match training
- Packaging: predict.py + reproduce.json + saved model (stdlib fallback?)

### 6. Error diagnosis of GB (HL 4,16,64,128) vs TRUE probs (`experiments/diagnose_gb.py`)
Mean sq err 0.160 (ref 0.192; floor if we only knew the global true mean 0.239).
- 67% of error is in p_0, 19% p_1, 11% p_2, 3% p_5, 0.3% p_20. The jackpot tail is irrelevant to the score.
- Pred std of p_0 is 0.21 vs true 0.39; corr(pred,true) only 0.54-0.65 per class. Big hidden state left.
- By type: steady 0.209, balanced 0.193, jackpot 0.080 (jackpot is easy because p_0 is high and stable).
- p_0 over-predicted vs truth by ~0.02-0.04 in every type (steady .432/.403, balanced .565/.526,
  jackpot .773/.754): systematic bias. Training rows span all ticks; queries are at cutoff 192-256.
- Training has no tick/cutoff feature and no row weighting toward late ticks.
- Worst episodes are stable/drift/short_history (~0.23-0.27 each), best are shared_change (~0.07).

### 7. Tick feature + late-row weighting, and per-machine lag response (`experiments/gb_proto.py`, env flags)
Flags: TICK=1 (cutoff as feature), WEIGHT=1 (rows with tick>=150 weighted 3x), LAGRESP=1
(per machine, win rate after win vs after loss at lag k for k=1..8, shrunken to the machine's
own win rate with m=10, plus one summed "expected effect" feature). HL=4,16,64,128 throughout.

| variant | all | tune | audit | stable | short | drift | shared |
|---|---|---|---|---|---|---|---|
| entry 5 best (none) | 32.72 | 33.28 | 32.09 | 0.819 | 0.893 | 0.904 | 0.719 |
| TICK + WEIGHT | 32.66 | 33.16 | 32.10 | 0.817 | 0.902 | 0.904 | 0.726 |
| **LAGRESP** | **32.99** | 33.45 | **32.47** | 0.804 | 0.858 | 0.905 | 0.709 |
| TICK + WEIGHT + LAGRESP | 32.64 | 32.95 | 32.28 | 0.830 | 0.880 | 0.911 | 0.732 |

Takeaways: LAGRESP helps (+0.27 all, +0.37 audit), mostly short_history and stable. TICK/WEIGHT did
not help, so the late-tick bias hypothesis from entry 6 is NOT confirmed (p_0 bias not re-measured;
may be episode-level base-rate shift between train and validation). Combining all three was worse
than LAGRESP alone, likely noise-level (~0.2) or the weighting hurting. Keep LAGRESP, drop TICK/WEIGHT.
Not checked: multiple seeds. Differences under ~0.2 should be treated as noise.

## 2026-11-07 master test (docs/sai, branch sai/master-test). Scope dev (72 episodes); B3 = 45.82 [44.75,46.90]

### W2 memory (M1-M10), worker report reviewed, M2/M3/M8/M9 re-scored by the lead through the harness
- M2 fixed-share and M3 DMA: 46.08 / 46.09, +0.26 [+0.02,+0.69] vs B3, all of it in stable (0.118 -> 0.090). The gain
  comes from a 180-tick window (almost the whole history), i.e. picking hl=inf more often. Verdict helps, marginal
  (best cell picked on dev). M2 and M3 are one effect.
- Nothing else beats B3's holdout memory. Hard change-point windows (BOCPD MAP 36.8, ADWIN 34.6) are far worse; they
  only reach B3 when multiplied onto its weights. HMM hurts every condition (-0.94). Half-lives 4-12 hurt again at every holdout.
- M9 (episode-shared BOCPD): shared_change 0.375 -> 0.350 (0.303 in pure-survival cells) but drift 0.40 -> 0.63 in those
  cells. Alive; follow-up is a peer-agreement gate.
- M10 Kalman half-life disagrees with B3's choice (corr 0.00) yet B3's choice is what scores well; no corroboration.
- Takeaway: B3's memory is near a plateau. Verdicts: helps M2,M3; alive M9; neutral M1,M4,M5,M7; hurts M6,M8,M10.

### W1-A frequency and shrinkage (A1-A8), worker report reviewed
Each A method tested (1) as B3's payout-size split / win prior with the pair-ridge win model kept, (2) standalone.
- The only lever found is a stronger prior on the size shares, alpha 24-64: +0.14 to +0.16 [CI ~+0.03..+0.27]
  (A1, A2, A3, A5, A6, A8 all rediscover the same effect). Verdict: A1 helps (tiny), rest neutral.
- p1 and p2 error shares do not move (0.229, 0.137): the size split is NOT where the p1/p2 error lives. Only p5 shrinks.
  So the remaining p1+p2 error is probably conditional structure (lags/state affecting severity), not shrinkage.
- Signal moves win rate ~7 points but barely moves size shares; with/without-signal cross is a null.
- Shrinking the ridge win probability toward a type/cell rate hurts badly (-2.9 at 50 pseudo-ticks): per-machine win
  rates are far from type priors.
- Standalone frequency models score 29.9-30.8 (~15 below B3); blend weight 0.00, zero stack value.
- Implication: look at severity-given-win with lag features (L5 log levels / 3-state) rather than more shrinkage.

### M9b peer-agreement-gated episode changepoints (follow-up to M9), worker report reviewed, lead re-scored: 46.07
- No gate keeps M9's shared_change gain without the drift loss. Ungated pure-survival pooling: shared_change -0.074
  [-0.110,-0.042] (real) but drift +0.236 [+0.052,+0.453]. The h=1/200 cell that looked best (0.303) is not CI-significant.
- Common-shift z statistic cannot discriminate shared change from independent drift (both give large C).
- Best cell: pool only if >=2 of 7 peers' own posteriors show a regime start in the last 80 ticks (W=40 never fires),
  fall back to B3's holdout half-life. 46.07 [44.98,47.15], +0.25 [+0.12,+0.40] vs B3; drift 0.399->0.390, shared_change
  0.377->0.357, stable unchanged. 298 cells searched, so expect ~0.25 selection optimism; neighbours k=3 +0.21.
- Falling back to a machine's OWN BOCPD weights is terrible (drift 1.15); B3's holdout memory is the right fallback.
- Verdicts: M9 helps (one condition, superseded), M9b helps (marginal, to validate on audit).

### W1-L lag structure (L1-L9), worker report reviewed, L2/L5/L7/L4/L8 re-scored by the lead through the harness
**Headline: L2 = 50.89 [49.8,51.9] on dev, +5.07 [+4.22,+5.89] over B3 (45.82).** Engine checked: reads only grid[:cutoff],
no truth/ref/condition/outcome. Lags and pairs are learned from train histories with null controls (9-16), they coincide
with B3's pairs (1,3),(2,5),(4,7) and single lag 7.
- B3's extra single lags 1-5 hurt, and lam=20 over-shrinks the real pair effects: lower lambda was monotonically better
  and plateaus at 0.05-0.5 (near-OLS). Conditions: stable .106, short .216, drift .211, shared .183 (B3 .118/.344/.399/.377).
- More structure does not help: all 28 pairs 41.9, all triples 41.4, null pairs 9-16 each cost several points, one-hot
  lags without pairs 31.3, logistic worse than linear, per-machine selection (L7) 49.3, MTD 43.1, conditional-lift NB 46.7.
- Pooling coefficients to the type mean (L6) and signal/type x lag priors (L9) add nothing: effect signs differ per machine.
- L5: centred log1p of the lag-7 payout with flag pairs 51.09, +0.20 over L2 (marginal).
- Error shares on L2: p0 0.49, p1 0.25, p2 0.19, p5 0.07: the SIZE side is now ~half the remaining error, so size/severity
  work (A methods, lag effects on severity) is worth much more on this base than it was on B3.
- All earlier M and A results were measured on the weak B3 lam-20 model and must be re-tested on L2 (= "B4").

### Codex W3/W4 first batch (E1-E5, S1), merged and re-scored by the lead through the harness (values reproduce)
- E1-E5 (peer residual pooling, cosine/type/signal pooling, episode drift features, null-controlled peer lag screen,
  latent factor): 45.75-45.81 on dev, within noise of B3; blend over B4 gains 0.00. Verdict neutral (built on old B3).
- S1 HistGB residual on B3: 46.59, +0.77 [+0.07,+1.51] vs B3; drift .399->.339, shared .377->.297, stable worse
  (.118->.162). Against B4 it adds only +0.06 [-0.06,+0.19]. Needs rebase onto B4 (note in docs/sai/CODEX_REBASE.md).

### Codex W3/W4 first batch (E1-E5, S1), merged and re-scored by the lead through the harness (values reproduce)
- E1-E5 (peer residual pooling, cosine/type/signal pooling, episode drift features, null-controlled peer lag screen,
  latent factor): 45.75-45.81 on dev, within noise of B3; blend over B4 gains 0.00. Verdict neutral (built on old B3).
- S1 HistGB residual on B3: 46.59, +0.77 [+0.07,+1.51] vs B3; drift .399->.339, shared .377->.297, stable worse
  (.118->.162). Against B4 it adds only +0.06 [-0.06,+0.19]. Needs rebase onto B4 (note in docs/sai/CODEX_REBASE.md).

### Codex W4 complete (S1-S8, 2464 declared cells), merged and re-scored by the lead (values reproduce). ALL on the old B3 base.
Dev points: S1 46.59, S2 46.77 (CatBoost/LightGBM, best), S3 41.91, S4 45.02, S5 46.40, S6 34.74, S7 44.73, S8 39.92 (B3 45.82, B4 50.89).
- As blend members over B4 none adds anything: S1 +0.06, S2 +0.01, S3 +0.03, S4 -0.02, S5 0.00, S6 -0.02, S7 +0.09, S8 +0.05 (all CIs include 0).
- S2 fixes stable (.162->.115 vs S1) and S1/S2 improve drift/shared vs B3, but B4 already beats them in every condition.
- S3 (condition classifier gating), S6 (future-window targets average through drift), S8 (MLP) clearly hurt; S4 row multiplication did nothing.
- Learners on B3 output cannot rescue a weak base; the question is whether residual learning on B4 keeps any gain. Rebase pending in Codex.

### B4 size/severity rebase (A1_B4, A4_B4, A6_B4, A9), worker report reviewed, all four re-scored by the lead (match)
- A9 (severity given win): 51.47 [50.50,52.38], +0.58 [+0.21,+0.92] over B4 (50.89). Per-machine ridge adjusting P(payout>=2 | win)
  on centred log1p lagged payouts at lags 1, 2, 4 (learned from a train severity screen with null lags 9-16). Lag 7 carries NO
  severity signal; win lags and severity lags differ. Conditions: stable .090, short .203, drift .203, shared .167. p1/p2 error
  fell. Effect is per-machine: a pooled train beta alone equals B4. 292 cells, so some selection optimism. Null lags 9-12: -0.2..-0.34.
- A1_B4/A4_B4/A6_B4 (alpha ~64 size-split prior): ~51.16, +0.27 [+0.10,+0.44]. One effect (p5/p20 shrinkage), not three.
- A9 touches p1/p2 and alpha touches p5/p20, so they should add; combination queued as C1.

### B4 memory rebase (M1_B4, M2_B4, M3_B4, M9b_B4), worker report reviewed, lead re-scored (match)
- M9b_B4: 51.22, +0.33 [+0.14,+0.53] over B4; drift .211->.197, shared .183->.168, stable unchanged. Survives the rebase and is robust to neighbours (k=1-4, lam 0.5). Helps.
- M2_B4 +0.22 [-0.01,+0.53] (one cell, window 190 = grid max) and M3_B4 +0.20 [-0.03,+0.51]: the B3 "helps" does not hold up on B4 (CI includes 0). Neutral.
- M1_B4 = B4 exactly: holdout-40 B3-grid memory is at a plateau; half-lives 4-12, floors, longer holdouts still hurt with the near-OLS ridge.
- Lambda 0.05/0.5/2 does not reopen gains for any memory method.

### C1 (B4 + A9 severity + alpha-64 size prior) and C2 (composition with M9b_B4), lead re-scored
- C1 = 51.72 [50.78,52.67], +0.83 [+0.53,+1.16] vs B4, +0.25 [+0.09,+0.45] vs A9 alone: the two gains add (predicted +0.85, got +0.82).
  Plain pre-declared combo (alpha 64, A9 lam 20) 51.715. Re-screening severity with lag 7 / pairs adds <= 0.006. 15 cells.
- C2 = win probability of M9b_B4 + size shares of C1: 52.09 [51.16,52.97], +1.19 vs B4, +0.36 [+0.16,+0.57] vs C1.
  Conditions: stable .090, short .191, drift .184, shared .142 (B4 .106/.216/.211/.183). No new fitted parameters; the
  win model and the size model are independent so their gains compose. Members were each selected on dev (optimism carries over).
