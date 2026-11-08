# Population priors (empirical Bayes) for expert weights: branch `pranav/population-priors`

Idea: every machine starts the pair mixer with uniform expert weights and must learn which lag
pair / memory and which size key matter from its own spins (~40 spins, ~15 wins in a short
history). Train has ~900 machines, so learn each type's average final expert log-weights on train
and start every machine there (scaled by a strength). New files only:
`slot_forecasting/population.py` (PopulationMixer subclasses PairMixer; strength 0 reproduces it
exactly), `research/eval_population.py`.

Learned on train: size experts favour lag 1 and disfavour the machine-level key (steady
-1.85 / +1.06 / +0.48 / +0.31 for (), 1, 2, 4); win experts favour the half-life-12 memory by
3-4 log units.

Results (win chance = the 56.54 log-odds blend; validation priors from the other half, population
learned on train; train: 8 folds, mixer AND population learned without the fold; dR vs strength 0):

| win / size strength | Validation (halves) | short ratio | Train cutoff dR | Train window dR |
|---|---|---|---|---|
| 0 / 0 (the 56.54 model) | 56.51 (56.51 / 56.53) | 0.1038 | 0 | 0 |
| 0 / 0.5 | 56.57 (56.58 / 56.56) | 0.1008 | +0.0004 [-0.0018, +0.0025] | -0.0004 [-0.0013, +0.0005] |
| 0 / 1 | 56.59 (56.62 / 56.57) | 0.0992 | +0.0012 [-0.0025, +0.0049] | -0.0006 [-0.0021, +0.0009] |
| 0.5 / 0 | 56.51 | 0.1040 | +0.0004 | +0.0002 |
| 1 / 0 | 56.51 | 0.1045 | +0.0008 | +0.0005 |
| 0.5 / 0.5 | 56.57 | 0.1010 | +0.0008 | -0.0002 |
| 1 / 1 | 56.59 | 0.0998 | +0.0020 | -0.0001 |

The size prior lowers short_history error on validation (0.104 -> 0.099) and raises both halves,
but held-out train shows no reliable gain (all intervals cross zero), so it is not adopted. The
win prior does nothing. Also checked: shared changes are only weakly visible across machines
(0.79 machines per shared_change episode shift sharply vs 0.42 in stable; 2 of 24 episodes have
>= 3), so episode-wide change sharing has little room.
