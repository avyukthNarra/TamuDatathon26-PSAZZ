# Round-two stress test of the v6 pair mixer

v6 does not crash on a new pair set, and it does not keep the main-round pairs when train has a readable rule. The two failures that move the score are the jackpot win prior (it copies a live level near 0.91 whenever a type does not split at 0.5) and a half-life of 12 when the regime flip sits inside the last ~25 ticks. Pair discovery's candidate list followed the new pairs in all three rounds. A pooled per-machine vote did not: on a cancelling drift it named the true pair on 2 of 48 machines, and the mixer still matched an oracle that was handed the true pair list, because the stable machines had already put those pairs in the list.

The patch in `fixes/pair_mixer_round2.diff` (code in `fixes/round2_adapt.py`) is windowed pair voting, an empty-search fallback that is not the main-round pairs, and 2-means levels for a type the 0.5 split cannot see. Validation error:

| Round | v6 | Patched | What changed |
| --- | ---: | ---: | --- |
| disjoint pairs, main timing | 0.0156 | 0.0135 | jackpot prior, drop one spurious pair |
| four pairs, shifted levels, cancelling drift | 0.0207 | 0.0157 | jackpot prior only (the pair list was already right) |
| shared lags, flip ~23 ticks before the cutoff | 0.0308 | 0.0272 | jackpot prior and one extra pair; the late flip remains |

Main-round raw errors were 0.0106 / 0.0186 / 0.0093 / 0.0116 (stable / short / drift / shared). Disjoint is in that neighbourhood. The late-flip round is about 2.5× main's average. Adding half-life 6 on top of the patch takes that round from 0.0272 to 0.0191 and makes the other two worse, so it is a second pass, not a default.

## What was run

Simulator: `simulator.py`, the g1 kernel (one parity pair, fair direction, logit-normal levels, two main-round size atoms, linear drift anchored at the cutoff, one shared flip). Short history is the eval-suite mask: last 40 ticks plus 24 older ones. Targets use the true past, including hidden ticks. Reference is the pooled starter, strength 20, fit on that round's train. Each split is 6 episodes per condition in train and validation (48 machines per condition) and 2 in test. Real main is 24 per condition, so a hit rate here moves by several points of noise. Seeds 701, 702, 703.

| Config | Pairs | Levels | Drift | Flip fraction realized |
| --- | --- | --- | --- | --- |
| `disjoint` | (1,6), (3,8), (2,4), equal | g1 | anchor 96, duration 62, finished 34 ticks out, u(cutoff)=1 | mean 0.676 (range 0.57–0.78) |
| `four` | (1,6), (2,4), (3,8), (5,7) at weights 0.50 / 0.27 / 0.15 / 0.08 | shifted; jackpot live logit stays negative | anchor 120, duration 30, so the early half and the late half have opposite directions; finished 90 ticks out | mean 0.439 (0.39–0.54) |
| `sharedlag` | (1,6), (1,8), (3,6) | g1 | finishes 15 ticks out | mean 0.899 (0.88–0.93), about 23 ticks of the new direction |

Official `run_pair_mixer.py` on each tree: exit 0, 10.4 s / 12.3 s / 11.8 s, reproduction diff 0, 0, and 1e-16. In-process `adapt` matched the runner's validation file. No main-round pair was in the fitted model. `(4,7)` and `(2,5)` showed up as 1–5 stray votes and were dropped.

`eval_suite/generator.py` preset `round2_shift` is a different generator: `recurring_lags=(2,3,5,7)`, interaction strength 0.85, shared change at 0.55 of the episode, and each machine draws 1–3 interactions from {single, xor, and, or} plus a neighbour term (probability 0.30), a hidden state, and a size tilt from one lag. That is not one parity pair. v6 can express the xor piece and will miss the rest. It was not scored. If the real round's train separation is weak (stable machines not landing on one pair), assume that family and do not spend the hour retuning half-lives.

## Pair discovery

`choose_pairs` returns every pair with at least 0.3× the leading vote. Votes are the per-machine winner of a pooled 4-cell variance on the whole history.

| Config | Truth (train machines) | Votes for those pairs | Chosen | Dropped true pair |
| --- | --- | --- | --- | --- |
| disjoint | 61, 75, 56 | 48, 29, 48 | those three, plus (6,8) with 15 votes | none |
| four | 97, 45, 31, 19 | 46, 21, 17, 15 | all four | none, by 1 vote (15 ≥ 0.3×46 = 13.8) |
| sharedlag | 61, 55, 76 | 52, 44, 65 | exactly those three | none |

Per-machine hit rate of that pooled vote, against the true pair:

| Config | Stable | Short | Drift | Shared | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| disjoint | 0.94 | 0.75 | 0.48 | 0.31 | 0.62 |
| four | 0.98 | 0.69 | 0.04 | 0.15 | 0.46 |
| sharedlag | 0.92 | 0.69 | 0.88 | 0.83 | 0.83 |

On `four`, drift is finished but the two directions occupy similar amounts of the history, so the pooled cells average out and the vote is noise (2/48). Windowed voting (full history, first 45%, last 40 ticks; keep the window with the biggest separation) raises that drift hit rate to 0.79 and shared-change from 0.15 to 0.77. On `disjoint` the same switch raises drift from 0.48 to 0.90 and shared-change from 0.31 to 0.92. On `sharedlag` the flip is so late that 90% of the history is still one regime, and windowing does not change the hit rate (0.83).

The mixer never reads that per-machine label. It gets the candidate list and reweights experts online. On `four` and `sharedlag` the list was already the true set, and v6's error equals the oracle that was given that set (0.0207 and 0.0308). On `disjoint` the extra (6,8) cost 0.0002. A bad per-machine vote is not a score bug while stable machines still elect the true pairs. It becomes one if the round has no stable episodes, or if the leader's count falls and a real rare pair drops under the 0.3 cut. The rare pair here survived on a 1-vote margin.

`cell_mode` was chosen by train cutoff Brier. Parity beat full on every round (0.368 vs 0.383, 0.376 vs 0.379, 0.393 vs 0.403).

## Priors refit, except jackpot

Size atoms were recovered to within 1e-6, two modes per type, on every round. Steady and balanced dead/live levels track the new medians. Jackpot does not, because `win_prior_from_targets` requires 20 targets on each side of 0.5 and otherwise the type inherits the round-wide prior.

| Config | Type | Latent median dead / live | Stock prior dead / live | Targets with p(win) ≥ 0.5 |
| --- | --- | --- | --- | --- |
| disjoint | steady | 0.170 / 0.955 | 0.190 / 0.954 | 29/60 |
| disjoint | balanced | 0.054 / 0.880 | 0.054 / 0.885 | 30/63 |
| disjoint | jackpot | 0.013 / 0.524 | 0.054 / 0.912 | 18/69 |
| four | steady | 0.232 / 0.932 | 0.254 / 0.930 | 30/52 |
| four | balanced | 0.099 / 0.883 | 0.098 / 0.887 | 35/64 |
| four | jackpot | 0.030 / 0.339 | 0.098 / 0.909 | 0/76 |
| sharedlag | jackpot | 0.015 / 0.481 | 0.062 / 0.910 | 10/58 |

The logged prior is these numbers, not the main-round 0.149/0.956, 0.039/0.840, 0.013/0.669. Those literals are not in the predict path. `predict.py` reads `validation/targets.csv` when it is present. 2-means on the jackpot targets lands on 0.036/0.581, 0.031/0.341, and 0.017/0.510. Replacing only the types that fail the 0.5 split (steady and balanced keep the stock prior) is the whole of the `four` gain: jackpot query error 0.0221 → 0.0094, round error 0.0207 → 0.0157.

Discovered size keys were `(6,)`, `(2, 6)`, and `(3, 4, 5, 6)`. Gains were 0.78, 0.43, and 0.30 nats, all noise relative to a static size mode, and none of them is the main-round story of lags 1, 2, and 4. Deleting them and keeping only the machine-level key raised error on every round (0.0169, 0.0246, 0.0332). Leave the discovered keys in.

## Scores against the truth

Error is the contest's episode-weighted mean squared error on the five probabilities. "Oracle set" is v6's half-lives, size keys, cell mode, and prior, with the true pair list swapped in. "Oracle machine" uses only that machine's pair. The patched column is windowed pairs plus the hybrid prior, official size keys, half-lives (12, 40, 100).

| | disjoint | four | sharedlag |
| --- | ---: | ---: | ---: |
| v6 | 0.0156 | 0.0207 | 0.0308 |
| oracle set | 0.0154 | 0.0207 | 0.0308 |
| oracle machine | 0.0140 | 0.0159 | 0.0304 |
| patched | 0.0135 | 0.0157 | 0.0272 |
| patched + half-life 6 added | 0.0141 | 0.0186 | 0.0191 |

v6 by condition (raw error, and the loss ratio against this round's pooled reference):

| | stable | short | drift | shared | R | 60/(1+R) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| disjoint | 0.0080 / 0.043 | 0.0188 / 0.093 | 0.0159 / 0.081 | 0.0197 / 0.098 | 0.078 | 55.6 |
| four | 0.0284 / 0.200 | 0.0188 / 0.134 | 0.0258 / 0.165 | 0.0097 / 0.073 | 0.143 | 52.5 |
| sharedlag | 0.0112 / 0.063 | 0.0232 / 0.119 | 0.0330 / 0.157 | 0.0557 / 0.309 | 0.162 | 51.6 |

The 60-point figure uses this round's reference, so it is not a comparison with main's 56.29. On `four`, the ugly stable ratio is the jackpot prior (patched stable error is 0.0107). On `sharedlag`, shared-change win-error is 0.048 of the 0.056 total: the fast expert (half-life 12) still carries the pre-flip regime when only ~23 ticks have passed. Adding a half-life of 6 to the existing (12, 40, 100), and changing nothing else, moves shared-change from 0.0557 to 0.0321. The same addition on top of the patch moves it to 0.0290. Eta 16 instead of 6 did nothing (sharedlag stayed 0.0308). Half-life 6 alone, with 12 and 100 removed, blows up stable error (disjoint 0.0080 → 0.0142, four 0.0284 → 0.0442).

Knowing the machine's pair does not fix a late flip (sharedlag oracle-machine 0.0304 vs v6 0.0308). It does matter when several pairs share the weight and the prior is wrong: on `four` the machine oracle cuts jackpot error from 0.0221 to 0.0099 even with the bad prior, because one expert's counts can outvote strength 10 and a mixture of four pairs cannot. The hybrid prior gets the same jackpot number without being told the pair.

## Hard-coded constants

Grep of `pair_mixer.py` and `entry_template/predict.py`. "Fired" means this run actually took that branch.

| Literal | Where | Rating |
| --- | --- | --- |
| `DEFAULT_PAIRS = ((1,3),(4,7),(2,5))` | default argument, and `choose_pairs` when no machine has 20 scored ticks | Risky. Did not fire on these trains. A 15-tick all-loss history returns exactly those pairs (`results/leak_demo.json`). |
| `MODE_SIZE_KEYS = ((),(1,),(2,),(3,),(4,))` | `discover_size_keys` when every lag gain is ≤ 0 | Risky, same demo. Did not fire here: noise lags had positive gain, so the keys moved off 1–4. |
| `DEFAULT_SIZE_KEYS` lags 1, (1,2), 2, (2,3) | only if `PairMixer` is built with no size modes and no size keys | Harmless on the `adapt` path, which always passes discovered keys. |
| split at 0.5, ≥20 per side, then copy the global prior; weights 0.45/0.45/0.1 | `win_prior_from_targets`, `win_prior_by_type_from_targets` | Risky. Fired for jackpot on all three rounds. This is the largest score bug. |
| `half_life=(12, 40, 100)` | `PairMixer` default, used by `adapt` | Risky when the new regime is younger than ~40 ticks. Harmless when the flip is as old as main's (~70 ticks). Adding 6 helps only the young-flip case. |
| `min_share=0.3` in `choose_pairs` | relative to the leading vote | Mild. Kept spurious (6,8) at 15/48 and kept the rare pair by one vote. |
| `WIN_PRIOR_ETA = 6` | used whenever a win prior is set | Harmless here. Eta 16 did not help. |
| `discover_size_keys` `min_share=0.5`, `top=4`, `min_wins=12` | which lags survive | Harmless. The lags it kept are not 1/2/4, and removing them hurt. |
| size-mode cluster gap 0.025, mass 0.05, skip `p(win)<0.05` | `size_modes_from_targets` | Harmless at the main atom spacing. Atoms came back exactly. |
| middle band 0.2–0.8, `middle_strength=10`, `strength=10` | per-type prior | Harmless for steady and balanced, which did split. Jackpot never got this far. |
| `k_win=0.25`, `eta=16`, `size_eta=4`, `k_size=16`, `k_machine=32`, `rate_prior=(1,2)`, `size_half_life=80`, `MIN_PROBABILITY=1e-4`, `n<20` | smoothing and the pair-search floor | Harmless as main-round facts. `size_half_life=80` is risky only if size itself starts drifting; these rounds keep size static. The `n<20` floor still passes a 64-tick short history. |
| `decay_missing=True`, `alpha=0` | short-history gaps; no fixed-share | Not a leaked pair. Short-history error on the main-timing round is 0.0188, the same as main's 0.0186. |
| `predict.py` | no numeric pair or level constants | Harmless if `validation/targets.csv` exists. Risky if you submit an old zip: `priors.json` is the fallback when targets are missing, and a main-round zip has main levels in it. |

## Fixes

`fixes/round2_adapt.py` is what was scored. `fixes/pair_mixer_round2.diff` is the same change against `choose_pairs`, the empty size-key fallback, and the `own is None` branch of `win_prior_by_type_from_targets`.

1. `choose_pairs`: vote with `robust_assignment` (full / early 45% / last 40), keep pairs used by at least 5% of scored machines, and return `ALL_PAIRS` when the search is empty. Never return `DEFAULT_PAIRS`.
2. `discover_size_keys`: when the best lag gain is ≤ 0, return `((),)` instead of lags 1–4. Do not add a 1-nat minimum. That cut was tested (`size_static`) and lost on every round.
3. When `win_prior_from_targets` returns None for a type, fit `two_level_fit` on that type's targets instead of copying the global prior.

Not in the patch, because the measurement went the other way: replacing (12, 40, 100) with (6, 12, 40), using eta 16, forcing `cell_mode=full`, or deleting size lags.

## Ninety-minute runbook

The entry already rediscovers pairs and refits both priors from `/data`. Do not paste main pairs or main levels into it. Do not reuse main's `priors.json`.

**0–15 min.** From `mainFiles`, on the new bundle:

```sh
python run_pair_mixer.py --data /data --out runs/r2
```

Read the log lines `lag pairs found`, `payout-size keys`, `win-cell prior`, and the two cell-mode Briers. Confirm the fresh-process diff is under 1e-8. Budget: a set the size of main is a few times these 12-second runs.

**15–30 min.** Three checks on that log, before any edit.

- If the pair list is exactly `(1,3), (4,7), (2,5)`, the empty-search fallback fired. Switch it to `ALL_PAIRS` and rerun. That list did not appear in any of these rounds; it appears when no machine has 20 scored ticks.
- If one type's live level in the log sits on top of another type's (jackpot live near 0.9 next to steady live near 0.9), the 0.5 split failed. Apply `two_level_fit` for that type only. This fired on every round here, including g1 levels, because jackpot live sits near 0.5 and 18 targets above 0.5 is not 20.
- Parity should win the train Brier if the rule is still XOR/XNOR. It did here. If full wins by a clear margin, keep full; the round is not a parity rule (see `round2_shift`).

**30–50 min.** Apply the diff. Rerun. The pair list should be the windowed one. The jackpot (or whichever type failed) live level should move to that type's own two centers. Expect on the order of 0.002–0.005 raw error if the bug was the borrowed live level, and about 0 if the levels already split. Do not delete the discovered size lags to "clean up" a gain of 0.3 nats.

**50–70 min.** Only if shared-change or drift is still several times stable after the patch. On a handful of train episodes, compare the win rate in the last 20 ticks with ticks 40–80 back inside the same parity cell. If those two windows disagree, the new regime is younger than the half-life of 12: rerun with `half_life=(6, 12, 40, 100)` and keep it only if train cutoff Brier on the drift and shared-change episodes improves. Decide that on train labels, not by picking the better validation number. Measured validation effect, which you should not tune to: sharedlag 0.0272 → 0.0191, disjoint 0.0135 → 0.0141, four 0.0157 → 0.0186. A young flip is the only case that paid.

**70–90 min.** Rerun `run_pair_mixer.py`, check the reproduction diff, submit that zip. If train machines do not concentrate on a few pairs (stable hit rate far below the ~0.9 seen here), stop retuning and treat the round as `round2_shift`: mixed operators, neighbours, a hidden state. The parity mixer will not reach main's error on that family, and the hour is better spent confirming the submission runs than searching half-lives.

## Caveats

Train and validation are 6 episodes per condition, not 24. The jackpot-prior gap and the late-flip gap repeat across rounds; a 0.0002 pair-list gap does not. Both v6 and the patch read the validation targets they are scored on, which is the entry's real protocol (`predict.py` does this), not a second fit. The half-life-6 numbers were run after shared-change came out at 0.056, so they are a diagnosis, and the runbook keeps them off the default. The simulator's size mode is static, matching g1; a round that makes size depend on a lag would reverse the "don't delete size keys" result, and the discovered keys would be the thing to trust.
