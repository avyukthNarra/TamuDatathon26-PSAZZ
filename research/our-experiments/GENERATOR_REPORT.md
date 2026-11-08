# Synthetic generator calibration

Calibration used only `public/train` (768 queries, 96 episodes, 139,944 observed history plays). `main_like` uses train-derived type×signal win rates and positive payout proportions (orders `[-1,0,1]` and `[1,2,5,20]`), plus `interaction_strength=1.3`; other mechanisms remain broad. Generated comparison: seed 2407, N=24 episodes per condition. Rate tuples are win probabilities; payout TV is type-level total variation from real. Lag7 reports real/generated phi.

| Type | Real rates | Generated rates | Payout TV | Lag7 phi R/G |
|---|---|---|---:|---|
| steady | .527/.564/.641 | .506/.543/.584 | .0211 | .1885/.0900 |
| balanced | .366/.400/.458 | .392/.424/.471 | .0259 | .2022/.0685 |
| jackpot | .170/.209/.296 | .230/.248/.304 | .0204 | .2026/.0980 |

Overall max type×signal rate gap .0600≤.08; max payout-mix TV .0259≤.10; machine win-rate SD .1737/.1814 (gap .0076≤.06); mean cutoff 222.67/222.09 (gap .57≤8); short fraction .25/.25 with exact condition-specific dense/40+24 pattern. The lag-phi max gap is .1337, above .10: observed lag7 is stronger in real data, especially balanced (gap .1337). A fixed sweep of interaction strengths .7, 1.0 and 1.3 selected 1.3 by smallest maximum lag gap. The permitted final trial added hidden strength .9/persistence .95; it slightly lowered max gap in the pre-freeze comparison but spread substantial false lag correlation across steady-machine lags, so was rejected. Do not claim complete lag calibration.

Generation wall time, seed 2407, N=24 episodes/condition, final config where applicable:

| Preset | Seconds |
|---|---:|
| main_like | 7.03 |
| round2_shift | 7.04 |
| all_lags | 7.74 |
| neighbour_heavy | 7.57 |
| hidden_state_heavy | 7.21 |
| fast_drift | 7.34 |
| edge_cases | 7.01 |

Tests: all 12 focused generator/calibration tests passed. The required full suite, `python -m unittest discover -s eval_suite/tests -t . -v`, passed 158 tests in 140.201s (`OK`). The only calibration band still missed is lag correlation, concentrated at lag7; all other reported bands pass.


CLI integration check after the quota reset (seed 73, N=1/condition): from `mainFiles`, `python -m eval_suite.generator --preset main_like --seed 73 --episodes-per-condition 1 --out /tmp/eval-cli-mainlike-73` and the same command with `round2_shift` and `/tmp/eval-cli-round2-73` both succeeded. Kit loaders found 32 test queries each; public test held only queries/history; all three `_truth/test` files loaded with IDs matching those queries. Truth submissions passed `check_submission.py` (main/60 and adaptation/40). `protocol.generate_preset` also succeeded for main_like seed 74 at `/tmp/eval-protocol-mainlike-74`.
