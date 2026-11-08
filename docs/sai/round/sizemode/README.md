# S1 size-mode rule discovery: no additional size head justified

All rule selection, held-out prediction checks, lag tallies, and peer null tests
use train histories only. The two size-mode emission vectors are inherited from
validation targets, as authorized. The canonical mixer6 forecast is scored on
72 dev episodes only. No audit/all scoring occurs.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
nice -n 10 python3 docs/sai/round/sizemode/discover.py
nice -n 10 python3 docs/sai/round/sizemode/summarize.py
nice -n 10 python3 -m unittest discover -s docs/sai/round/sizemode -v
nice -n 10 python3 docs/sai/round/sizemode/verify.py
```

The deterministic screen includes 368 rules: single lags1–8 and all28 lag
pairs, with XOR/AND/OR, for both win and big-win(payout≥5) features, with both
polarities. A missing feature uses a causal running observed frequency with
two type-population pseudo-observations; missing pair features are marginalized
independently. Actual missing payouts are never replaced by losses.

Static alternatives are a fixed 50/50 mixture and a per-machine fitted mixture.
The two-state HMM has known size emissions and symmetric switch probability
selected from .005/.02/.08/.25/.5. Its state evolves every calendar tick,
including gaps/losses; only observed winning sizes provide emissions. Swapping
its state labels/polarity gives the same model with uniform initialization.

Raw BIC=-2logL+dlog(n_wins), where deterministic/fixed-static hypotheses have
zero fitted continuous parameters and fitted-static/HMM models have one. The
large discrete rule search is separately penalized by 2log(number hypotheses
in its family); this additional criterion is labeled search-penalized, not BIC.
Tallies/peer tests require at least12 observed wins:730 of768 train machines.
The raw and adjusted per-type family counts, complete winning rules, and lag
participation tallies are in `train_discovery.json`.

To check overfitting, each machine's first70% calendar ticks select rule/
parameters, and its last30% score predictive log-likelihood. The existing
mixer6 size head is evaluated online on the identical tail observations.
Episode bootstrap confidence intervals compare per-machine tail loglik/win.
Every family loses globally; neither target short-history type has a family
with a positive lower CI bound. The tiny steady/short HMM gain is uncertain.
Thus the conditional request to build a new size model is not activated; no
new dev variant or prediction CSV is produced, and no opposite-half dev tuning
is applicable. `summary.json` records this gate and canonical baseline score.

Peer checks compare exact-rule agreement, family agreement, simultaneous
win-size mode log-likelihood-ratio evidence, and HMM-filtered mode probability
co-movement. Nulls shuffle entire machines within train(type,condition) strata,
preserving feature distributions, missingness and within-machine dependence.
P-values are exploratory and not adjusted for the four peer checks. Filtered
states/rule choices here are train-side diagnostics, not validated generator
identification. These noisy null results do not exclude weak mode sharing.

Tests independently enumerate HMM hidden paths, check rule polarity/causality,
and compare the vectorized baseline size head with the copied original mixer6.
Verification reruns the full train screen with fixed seeds and compares every
score array and statistic. Existing files and Git state are left to the
orchestrator. No packages were installed.
