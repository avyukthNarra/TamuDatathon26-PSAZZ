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
