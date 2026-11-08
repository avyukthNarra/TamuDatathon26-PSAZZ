# e7 peer-level offset

Kept. Flag `USE_PEER_LEVEL` defaults on. `--no-peer-level` turns it off. Stdlib only. No git.

## What changed

For each machine, using only ticks before its cutoff, the best lag pair among the mixer's pairs supplies a live cell and a dead cell (smoothed rate on the far side of the type prior's midpoint, at least 8 spins). Each cell's logit is residualized against the precision-weighted mean of the same machine type × signal × side in other episodes. The machine's offset is the precision-weighted mean of the other machines in its episode (`n·p·(1-p)`). That offset, times a per-type `w`, is added to the logit of every Beta-mixture component centre inside `walk`, before the experts mix.

`w` is chosen on train by prequential next-tick win Brier after a 40-tick warmup. A value is kept only when both episode halves are at least as good as `w = 0`; ties go to the smaller `w`. Jackpot's ICC was ~0, and its learned `w` is 0.

## Main validation (paired, same code)

| | points | stable | short | drift | shared | halves |
|---|---:|---:|---:|---:|---:|---|
| `--no-peer-level` | 57.369 | 0.0419 | 0.0749 | 0.0324 | 0.0343 | (v7c stated 57.31 / 57.39) |
| `--peer-level` | 57.376 | 0.0417 | 0.0741 | 0.0327 | 0.0344 | 57.310 / 57.398 |

Full points rise by 0.007. Neither cross-fitted half is down by more than 0.02 versus the stated v7c halves (tuning 57.310 vs 57.31, audit 57.398 vs 57.39). Drift and shared ratios move up by under 0.001; short and stable move down.

Win squared error `(p_win − true p_win)²`:

| type | off | on | `w` |
|---|---:|---:|---:|
| balanced | 0.003456 | 0.003414 | 0.50 |
| jackpot | 0.007450 | 0.007450 | 0.00 |
| steady | 0.004011 | 0.004011 | 0.50 |

Train win-Brier gains are small (balanced 0.118422 → 0.118408, steady 0.114205 → 0.114179, jackpot stays at 0). Offsets have mean near 0 and sd about 0.15–0.17 logits. The prior strength is 10, so only a fraction of `w × offset` reaches a cell that already has decayed counts. That is why the score moves by hundredths of a point.

## Simulated kits

Levels are drawn independently per machine. Learned `w` and points (`--peer-level`, no half split):

| kit | points | v7c | weights (balanced, jackpot, steady) |
|---|---:|---:|---|
| disjoint | 37.717 | 37.72 | 0, 0, 0 |
| four | 36.454 | 36.45 | 0, 0, 0 |
| sharedlag | 37.631 | 37.63 | 0, 0, 0.75 |

None is worse than v7c by more than 0.02. Disjoint and four learn `w = 0`. Sharedlag steady learns 0.75: both train halves improve by about 0.00005 Brier on 24 episodes, and the validation score still rounds to 37.63. That kit's `w` is not near 0. The half-sample rule can accept a tiny train dip when the episode count is small.

## Checks

`python -m unittest tests.test_episode_level tests.test_pair_mixer` — 30 tests, OK. The new tests cover leave-one-out (own residual excluded, centre from other episodes, precision weights), rejection of a tick at the cutoff, and a zero logit shift matching the unshifted walk.

Runs are under `runs/peer-off`, `runs/peer-on` (with `--cv-val`), `runs/kit-disjoint`, `runs/kit-four`, `runs/kit-sharedlag`. Scored with `--score-only`, so no entry zip was built. The scored path is the one `predict.py` runs when the flag is on.
