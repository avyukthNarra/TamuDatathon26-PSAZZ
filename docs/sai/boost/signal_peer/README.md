# T2 signal priors and T3 peer-triggered forgetting

All new fitting uses train histories. The v4 Beta component levels/strengths and
size modes remain the inherited, validation-derived baseline parameters. Every
new score and choice uses 72 dev episodes; audit/all scopes are never called.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
nice -n 10 python3 -m unittest discover -s docs/sai/boost/signal_peer -v
nice -n 10 python3 docs/sai/boost/signal_peer/run.py
```

`models.py` follows the vectorized v4 adapter from `improve/model/`. Baseline
win and size walks reproduce the original pinned source. T2 estimates latent
Beta-component weights from each train machine's best-pair parity cells, with
EM regularized toward type weights. Train size-mode EM uses observed positive
payouts. Both signal evidence checks fit weights on one train episode half and
score the other; confidence intervals bootstrap episodes. The dev grid tests
signal-only log weight tilts plus absolute type/signal and type-only controls.
No validation residual shift, calibration, or gated blend is fitted.

T2 size intervention is deliberately skipped because signal's train-held-out
size loglikelihood gain is negative and its CI includes zero.

T3 originally used EWMA excess Brier (results retained in
`excess_brier_results.json`). Train discrimination was poor. A second detector,
EWMA slow-minus-fast observed Brier, is selected on TRAIN alone: it flags shared
change/drift more than stable train episodes. Each detector threshold is the
99th percentile on stable train observations after tick40, separately by type.

A detector event caused by observing spin t is available at t+1. Peer triggering
counts OTHER machines with events within W calendar ticks; no event beyond the
query cutoff is used. k=2/3, W=12/32, fast-memory weight multiplier=4/16.
Per-machine-only controls use the same W, multipliers and thresholds.

Only the fast expert mixture weights are temporarily multiplied. Individual
expert counts and loss-weight updates remain the baseline's; the detector is
computed from the unmodified baseline. Therefore applying the transient weight
boost at earlier ticks cannot change the final individual expert state, and
using the cached final experts is exactly equivalent to an online transient
boost with this frozen reference detector. No win cells are pooled.

`results.json` contains the current contrast-detector grid, every variant's
score/CI, paired difference/CI, condition ratios, train evidence, opposite-fold
selection, and comparisons with type-only/per-machine controls. Original
surprise-detector variants retain their unsuffixed T3 names; contrast-detector
variants add C after T3. All prediction filenames start BOOST_T2 or BOOST_T3.
