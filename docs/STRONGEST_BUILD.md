# Strongest build: v13 (57.52) on branch `pranav/strongest-build`

Combines the three lines of work that each passed honest checks:
- **Ours, win chance (v11, 56.61):** log-odds pool of pair mixer + hidden-rule HMM, peer correction
  (machines win more when the episode's other machines just won), plateau average.
- **zhill, payout size (size-rule HMM):** size given a win is one of two exact mixes per type, and
  which applies is set per machine by whether the payout k spins back was >= 2x (k in {1, 2, 4},
  either direction). `slot_forecasting/size_rule_hmm.py` (adapted verbatim) infers that rule per
  machine; core rules are re-found on each round's train.
- **Avyukth, v7c (57.37):** train-learned size-atom rule + episode flip swap, vendored unchanged as
  `v7c_forecasting/` (see `v7c_forecasting/VENDORED.md`).

## Steps and checks
| Build | Validation | Cross-fit halves | Other evidence |
|---|---|---|---|
| v11 (ours) | 56.61 | 56.55 / 56.62 | held-out train passes |
| v12 = v11 win + size-rule HMM | 57.36 | 57.47 / 57.18 | held-out train window dR -0.0162 [-0.0236, -0.0090] |
| v7c (Avyukth, built from its branch) | 57.37 | 57.31 / 57.39 | its own honest 4-fold: 57.36 |
| **v13 = 50/50 log-odds win of v12 and v7c, v7c sizes** | **57.52** | **57.55 / 57.44** | beats both parents on both halves |

Combination grid (cross-fit validation, `research/combine_v12_v7c.py`): v12 win + v7c sizes 57.49
(57.64 / 57.34, below v7c on one half); 50/50 win + v7c sizes 57.50 (57.55 / 57.44, above both
parents on both halves, chosen: equal weights, no tuning); 70/30 win + v7c sizes 57.52 (tuned, not used).

Per condition (v13): stable 0.037, short_history 0.071, drift 0.029, shared_change 0.035 (R 0.0430).
Win error is now ~0.005-0.008 per condition and size error 0.0004-0.0042: size went from the
largest error in short_history to well below win.

Not done for v13 as a whole: our held-out train check (it would need v7c refit per train fold).
Each part has its own: v12 passes ours; v7c passes Avyukth's 4-fold; the blend beats both parents
on both validation halves.

## Build and verify
```sh
cd mainFiles && python run_strongest.py --out runs/strongest-v13 --cv-val
```
Entry: 29 files, 109 s, reproduces byte for byte on Python 3.13 read-only WITHOUT validation
targets (both priors fallbacks). Test SHA-256
0c7ef70490144d6b7215dfda5e17688efcf2cde38b723a866f454f3988ec6286. `round_two.py` now builds v13
and v12 too (`--quick` builds v13 only, ~6 minutes).

## v14 (branch `pranav/v13-plus`): v13 with our full peer effect back, now checked on held-out train
`research/v7c_cache.py` refits the vendored v7c WITHOUT each train fold (8 folds, cutoff + 10 ticks),
so v13 variants can finally be judged on held-out train too (`research/v13_eval.py`, `v13_variants.py`).

| vs v13 | Validation (halves) | Train cutoff dR | Train window dR |
|---|---|---|---|
| v12 alone | 57.33 (57.47 / 57.18) | +0.0078 | +0.0026 [-0.0007, +0.0064] |
| v7c alone | 57.34 (57.31 / 57.39) | +0.0038 | +0.0042 [+0.0024, +0.0060] |
| v13 (0.5 / 0.5, v7c sizes) | 57.50 (57.55 / 57.44) | 0 | 0 |
| win 0.4 on v7c | 57.51 (57.58 / 57.44) | -0.0004 | -0.0006 [-0.0009, -0.0002] |
| extra peer 0.25 | 57.50 (57.55 / 57.45) | -0.0009 | -0.0008 [-0.0013, -0.0003] |
| **v14 = win 0.4 + extra peer 0.5** | **57.50 (57.56 / 57.44)** | -0.0020 [-0.0049, +0.0007] | **-0.0019 [-0.0031, -0.0007]** |
| more v7c win (0.6 / 0.7) | 57.48 / 57.46 | worse | worse |
| size-rule HMM mixed into v7c sizes | 57.48-57.49 | worse | worse |

v7c ignores other machines, so v13's 50/50 blend halved the peer effect; v14 restores it. Same
validation (built entry 57.52, 57.519755; halves 57.56 / 57.44) and better held-out train.
Entry 106 s, byte-identical on Python 3.13 read-only without validation targets. Test SHA-256
996be60fb51f4e76b0e9423a0ff659fb31317c246cbd2e10fa782876e1283a19. `round_two.py` builds v14.

## v15 (branch `pranav/v14-plus`): signal term (57.52 -> 57.55)
Residual hunt on v14 (`research/v14_residuals.py`): payout size is clean (no signal at |z| > 2 on
both splits); the win chance still rises with the signal column on BOTH validation (z +3.4) and
train (z +3.1). `research/v14_signal.py` (vs v14):
| Signal coefficient (log-odds per step) | Validation (halves) | Train cutoff / window dR |
|---|---|---|
| 0 (v14) | 57.50 (57.56 / 57.44) | 0 / 0 |
| **0.05 (chosen: half the validation optimum)** | **57.52 (57.58 / 57.47)** | -0.0008 / -0.0004 [-0.0010, +0.0001] |
| 0.1 | 57.53 (57.58 / 57.48) | -0.0012 / -0.0006 |
Built entry: validation 57.55 (cross-fit halves 57.58 / 57.47), R 0.0426 (stable 0.036, short
0.070, drift 0.029, shared 0.036); settings in combo.json; byte-identical on Python 3.13 read-only
without validation targets. Test SHA-256 2c4a968b1580fcdceaa16f8c5041a537705d46eb22b5e1500a278279a13cb2d1.
`run_strongest.py` and round two (candidate "v14" = the run_strongest default) now build v15.

## After v15 (branch `pranav/v15-plus`, no change)
- Short-history-specific blend weights / peer strength (`research/v15_gates.py`): validation flat
  (+-0.01), held-out train at most -0.0006: a plateau.
- Regime model as a third win member (`research/v15_regime.py`): validation +0.04-0.06 but held-out
  train worse (window +0.0005 to +0.0013): its settings were chosen on validation. Rejected.
- Payout size in shared_change: 8.8% of machines have the size mode confidently wrong (2-3% in
  stable / drift), clustered in a few episodes (3 of 7, 2 of 6) - an episode-wide size-rule flip.
  Detecting it from each machine's last K wins vs its own inverted rule (`research/size_flip.py`)
  misfires (mostly on short histories) and lowers both checks (validation down to 57.01-57.50).
v15 (57.55) stays the best verified build.
- Per-type blend weights (`research/v15_type_weights.py`): best (steady 0.2) +0.03 validation,
  train mixed (window -0.0005, cutoff +0.0006) after 6 tries: noise. Not adopted.
- Are shared_change size flips detectable from recent wins at all? (`research/size_flip_auc.py`):
  AUC 0.54-0.64 for "the inverted rule fits the last K wins better" (K = 4-15). Not usable.

## v16 (branch `pranav/v15-evalsuite`): v7c's flip swap transferred to our win model (57.55 -> 57.59)
v7c detects episode-wide win-rule flips (its gate changes v7c's forecast on 50% of rows); our win
model (60% of the blend) never saw them. `research/v7c_flip_cache.py` caches v7c with and without
the swap (validation cross-fit + 8 held-out train folds, run in parallel: 3.6 min vs ~15).
`research/flip_transfer.py`: our win log-odds += lam x (logit v7c with swap - without).
| vs v15 | Validation (halves) | Train cutoff dR | Train window dR |
|---|---|---|---|
| v15 | 57.52 (57.58 / 57.47) | 0 | 0 |
| transfer 0.5 | 57.56 (57.60 / 57.51) | -0.0010 | -0.0007 [-0.0014, -0.0000] |
| transfer 1.0 | 57.56 (57.60 / 57.53) | -0.0015 | -0.0009 |
| **transfer 1.0, v7c win weight 0.3 (v16)** | **57.57 (57.62 / 57.52)** | -0.0020 | **-0.0015 [-0.0029, -0.0002]** |
Built entry: validation 57.59 (57.594120), cross-fit halves 57.62 / 57.52, R 0.0418 (stable 0.036,
short 0.070, drift 0.028, shared 0.034); 124 s; byte-identical on Python 3.13 read-only without
validation targets. Test SHA-256 8d42dc08bc06b8f67895879630d00c88907b74c0e0a4888d5ab39211b29e68a9.

## Evaluation suite from main (`eval_suite`, full level, local mode, --jobs 8): v15 vs v8
- Verdict PASS WITH WARNINGS: 0 FAIL, 47 PASS, 3 WARN (soft), 26 NOT RUN (Docker-only / rule-level).
- Its honest 4-fold real validation: v15 57.51 [57.00, 57.99] vs v8 56.46: **+1.05 [+0.74, +1.39]**.
- Generator presets (a teammate's approximation of the simulator, not the real one): v15 better on
  main_like (+0.42) and fast_drift (+0.62), worse on round2_shift (-0.49), all_lags (-0.74),
  hidden_state_heavy (-1.13), edge_cases (-0.93). Round two must therefore pick by cross-fit on the
  new data (round_two.py does), not assume v15/v16. Slowest preset run: 514 s (28.6% of the limit).

## Residual hunt on v16 (`research/v16_residuals.py`)
Win and size residuals vs every signal used before (own lags 1-8 won / >=2x / >=5x, peers' lags
1-8 won / >=2x, signal, history length, forecast level, own recent win share, v7c-minus-ours,
flip swap active, flip effect size): **nothing** has |z| > 2 with the same sign on both validation
and held-out train. The signal-column and peer effects found earlier are fully absorbed.
