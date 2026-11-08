# E2 C=10 adaptive-memory check

This is a tuning-only follow-up to compare E2 memory selection under the same C=10 setting as the fixed E1 anchor. The earlier E2 C=3 point score (36.2228) was essentially tied with fixed E1 C=10/HL32 (36.2173); this run provides the matched C=10 adaptive comparison. No model code, branches, dependencies, or audit predictions/scores were changed or produced. Both E2 variants are from PR #8's source branch `avyukth/exp-e2-adaptive-memory` at `6511dfb5fce56d3c40019325953c7b399cf0cbb3`.

The saved-CSV verifier required exact equality with the fixed tuning query IDs: each candidate and the anchor contained all 384 tuning queries and no extra rows. The official kit scorer reproduced each saved score. Paired intervals use 1,000 episode-stratified resamples, seed 314159, 12 episodes per condition, shared draws across the two models, and recomputed reference denominators. These intervals are exploratory because the candidates were selected using the same tuning episodes.

| Candidate | Exact settings | Points / 60 | Stable error | Short-history error | Drift error | Shared-change error | Fit/predict runtime |
|---|---|---:|---:|---:|---:|---:|---:|
| E2 adaptive C=10 | `Cs=[10], eval_stride=4, half_lives=[null,16,32,64], loss=brier, max_eval_rows=32, min_train=16, mode=soft, temperature=128` | 37.145854 | .044991 | .186229 | .152900 | .105933 | 14.96 s |
| E2 adaptive C=3/10 | `Cs=[3,10], eval_stride=4, half_lives=[null,16,32,64], loss=brier, max_eval_rows=32, min_train=16, mode=soft, temperature=128` | 36.972673 | .046573 | .182589 | .156405 | .111127 | 27.10 s |
| Fixed E1 anchor | `PairLogisticForecaster(C=10, half_life=32)` | 36.217264 | .066251 | .186109 | .168433 | .103176 | 0.99 s |

Paired comparison to the fixed E1 C=10/HL32 anchor:

| Candidate minus anchor | Point difference | Paired 95% interval | Condition-error differences (stable, short, drift, shared) |
|---|---:|---:|---|
| E2 adaptive C=10 | +0.928590 | [+0.144172, +1.711445] | [-.021259, +.000119, -.015533, +.002757] |
| E2 adaptive C=3/10 | +0.755409 | [-.067366, +1.478699] | [-.019678, -.003521, -.012028, +.007951] |

The C=10-only adaptive candidate has a positive point difference and its paired interval is above zero; the C=3/10 variant is also higher by point estimate, but its interval crosses zero. Both improve stable and drift errors, leave short-history nearly unchanged or slightly better, and give up a small amount on shared-change error. This establishes a tuning gain over fixed C=10/HL32, not that prequential selection beats every fixed half-life or a static memory mixture; those controls were not run. E2 C=10 remains below E10 (37.791/60) and current E6/E7, so this check does not motivate a blend port or additional sweep. These results remain tuning evidence rather than independent confirmation.

Provenance:

- E2 source: `mainFiles/slot_forecasting/adaptive_memory.py`, SHA256 `fef56dc236d1df49ef3cfc39d72947f516f13be65b23ab2172f762c662681253`.
- Fixed E1 anchor source: `mainFiles/slot_forecasting/pair_logistic.py`, SHA256 `834868f4b2cd3d4691c015f1dc1e3c5d707bb4f7cefedc6a18350f2d2a88e58d`.
- C=10 E2 predictions SHA256 `fad051af016b36d0377bc7bfd29dec316e117e5b49e516516096c72e6c964931`; C=3/10 E2 predictions SHA256 `61a9858d08f63f16a6a244ec236ce07c4db24ee60869a4fe223636090b922013`; E1 anchor predictions SHA256 `2ba48d5e9e3c681d8c86fe329d16f577979c2633d46341df0e9beafb7086aafc`.
- Raw run records: `/Users/avynarra/Documents/Datathon/worktrees/codex-exp/mainFiles/runs/e2-c10check-20261107.json` and `/Users/avynarra/Documents/Datathon/worktrees/codex-exp/mainFiles/runs/e2-c10check-e1-anchor-20261107.json`.
- Bootstrap outputs: `/Users/avynarra/Documents/Datathon/worktrees/_coord/e2-c10-adaptive-vs-e1c10-20261107.json` and `/Users/avynarra/Documents/Datathon/worktrees/_coord/e2-c3-c10-adaptive-vs-e1c10-20261107.json`.
