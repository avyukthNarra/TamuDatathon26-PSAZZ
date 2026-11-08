# v7c train-only size-gate review

Reviewed `v7c-mainFiles/slot_forecasting/size_gate.py`, `forecast.py`,
`size_rule.py`, `pair_mixer.py`, `entry_config.py`, `entry_template/predict.py`,
and `run_pair_mixer.py`. This is a source review only; no model runs or scores
were produced here.

- `fit_size_gate` assigns sorted episode IDs to alternating folds, then
  excludes each held fold from both `SizeRule.learn` and the v6 `adapt` fit
  (`size_gate.py:117-132`). The v6 comparator gets fold-fitted atoms and
  `labels=None`, avoiding cutoff-label selection (`:38-42`, `:129-131`).
- On held episodes, prior history is truncated to ticks before cutoff.
  Candidate and baseline forecasts are emitted before each payout is absorbed;
  only observed wins contribute conditional-size scores. Optional cutoff
  labels are train labels, used only after both forecasts are produced
  (`:63-92`). The size rule's stream also updates its pair assignment from
  prefix counts after emitting the current forecast (`size_rule.py:538-605`).
- Gate inputs are `queries`, `history`, and optional `labels`; no validation
  targets or validation-derived priors are passed (`forecast.py:13-20`). In
  both runner and packaged prediction, `USE_SIZE_GATE=True` causes the gate
  to be fit at runtime; the config stores the switch, not a prior gate choice
  (`entry_config.py:9-12`, `run_pair_mixer.py:65`, `predict.py:49`). It selects
  the candidate only with at least eight scored episodes and a negative
  episode-mean Brier delta plus 1.96 standard errors (`size_gate.py:155-182`).
- Validation targets are still read by the main forecast path to build the
  PairMixer's size and win priors (`run_pair_mixer.py:57-65`,
  `predict.py:36-48`). This is separate from the gate; the gate's own
  comparator reconstructs its size modes from each fit fold. The production
  fallback remains the PairMixer size head configured by the main path.

No episode-split leakage, scored-label-before-forecast leakage, or validation
input into the gate was found in these paths. This review does not establish
that the candidate improves performance; that is for the separate evaluation
and gate diagnostics.
