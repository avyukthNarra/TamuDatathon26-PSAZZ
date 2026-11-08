# v17-guards as the blend's first member

**Swap.** On the 10-kit benchmark the variant mean is 38.388 vs the baseline 38.371 (**+0.017** / 40). No kit drops by more than 0.2. The largest drop is K3, **−0.020**. Main stays a tie, as Sai measured.

The benchmark's own note is that a gap under 1 point on one kit is noise. This swap clears the pre-registered bar and does not show a gain you could see above that noise. K10 (mixed 3-lag / AND rules) is the only move larger than 0.1 (**+0.122**) and is still the worst kit for both builds, at about 32 / 40.

## Main cross-fit (out of 60)

Same protocol as `s3-final`: validation split, sha256 episode folds for the priors, then `score_pred.py` (episodes sorted by id within condition, alternating). Data: `worktrees/eval-main/mainFiles`.

| Forecast | Points | Halves A / B | Time |
|---|---:|---|---:|
| Baseline, s3 blend (from `s3-final/REPORT.md`) | 57.961 | 57.97 / 57.95 | 247s there |
| Variant, member 1 = v17-guards + same patches | **57.971** | **57.98 / 57.96** | 145s here |

Variant ratios: stable 0.0254, short_history 0.0605, drift 0.0240, shared_change 0.0300. Learned pairs on main were (1, 3), (2, 5), (4, 7). Jackpot strength atom16, 194 queries patched. That matches the s3 blend's jackpot line.

## 10-kit benchmark (out of 40)

Kits regenerated with `adapt_bench/regenerate.sh` into `kits/`. Train history, validation targets, and metadata match `a3-saiadapt/kits` file for file, so the baseline numbers below are that run's blend scores (`runs/blend/K*/scores.txt`), not a second build. Variant outputs are `runs/kits/K*/blend.csv`, scored with `s3-final/score_pred.py`. Five kits at a time.

| Kit | Baseline | Variant | Diff | Variant seconds |
|---|---:|---:|---:|---:|
| K1 main-like | 38.795 | 38.801 | +0.006 | 138 |
| K2 disjoint pairs | 38.963 | 38.962 | −0.002 | 134 |
| K3 four pairs | 38.931 | 38.911 | −0.020 | 137 |
| K4 shared lag | 39.265 | 39.264 | −0.001 | 119 |
| K5 long lags | 39.198 | 39.195 | −0.003 | 131 |
| K6 static size | 39.371 | 39.365 | −0.006 | 187 |
| K7 size bit ≥ 5 | 39.096 | 39.098 | +0.001 | 175 |
| K8 late flip, fast drift | 39.092 | 39.131 | +0.040 | 157 |
| K9 shifted levels | 39.036 | 39.071 | +0.035 | 156 |
| K10 mixed 3-lag / AND | 31.958 | 32.080 | +0.122 | 168 |
| **Mean** | **38.371** | **38.388** | **+0.017** | |

Worst kit: **K10** at 32.080 (baseline 31.958). Worst drop: **K3 −0.020**.

Runtime: variant wall clock **325s** for all ten (five at a time). Sum of the ten builds **1503s**. The reused baseline builds summed to 1003s in the other folder (83–135s each); those clocks are not paired with this run.

Learned pair support followed the kit: K2 (1, 6), (2, 4), (3, 8); K3 those plus (5, 7); K4 (1, 6), (1, 8), (3, 6); K5 (3, 8), (5, 8), (6, 7). K1 and K6–K10 kept the main three pairs. `PAIRS_OVERRIDE` stayed `None`.

## What the variant is

`variant/` is the s3 entry. `vendor/v13/slot_forecasting` and `vendor/v13/v7c_forecasting` are from `origin/zhill/v17-round2-guards` (`acc3d73`, worktree `z17/`). The guard fix is Avyukth's change to the vendored v7c `pair_mixer.py` and `episode_flip.py`. `short_win.py`, `jackpot_model.py`, the exact filter, and sysid are the s3 copies.

`members/v13_short.py` still owns that registry slot. It calls `combine` with the v17 `run_strongest.py` defaults: v7c win weight 0.3, extra peer 0.5, signal coefficient 0.05, flip transfer 1, jackpot sub-cell on. The peer centre is `train_center` on that dataset's train split (0.4187 on main). Short and jackpot patches still run after that, on the learned pair list.
