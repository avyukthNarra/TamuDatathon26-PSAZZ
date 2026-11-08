# m4 residual

Base is v7c in `mainFiles`. Fresh run: **57.37 / 60** (R = 0.0459). Cross-fit halves: tuning 57.31, audit 57.39. Ratios stable 0.042 / short 0.075 / drift 0.032 / shared 0.034.

Nothing was kept. `mainFiles` source is v7c again. Score dumps are in `runs/v7c` and `runs/jackpot-strength`. The three simulated kits were not re-run, because no candidate passed the keep rule.

## Error decomposition

Squared error on the 768 validation queries, split the way `run_pair_mixer.py` splits it: win part is the error removed by plugging in the true win probability and keeping our size mix; size part is what remains. Each episode has 8 machines, so the query mean equals the episode mean.

| condition | type | n | win | size | total | size share |
|---|---|---:|---:|---:|---:|---:|
| stable | balanced | 65 | 0.0038 | 0.0005 | 0.0042 | 12% |
| stable | jackpot | 58 | 0.0113 | 0.0005 | 0.0118 | 5% |
| stable | steady | 69 | 0.0088 | 0.0001 | 0.0090 | 2% |
| stable | all | 192 | 0.0078 | 0.0004 | 0.0082 | 5% |
| short_history | balanced | 60 | 0.0054 | 0.0057 | 0.0111 | 51% |
| short_history | jackpot | 71 | 0.0122 | 0.0015 | 0.0137 | 11% |
| short_history | steady | 61 | 0.0089 | 0.0060 | 0.0149 | 40% |
| short_history | all | 192 | 0.0090 | 0.0042 | 0.0133 | 32% |
| drift | balanced | 58 | 0.0051 | 0.0002 | 0.0053 | 4% |
| drift | jackpot | 65 | 0.0083 | 0.0007 | 0.0091 | 8% |
| drift | steady | 69 | 0.0038 | 0.0004 | 0.0041 | 10% |
| drift | all | 192 | 0.0057 | 0.0004 | 0.0061 | 7% |
| shared_change | balanced | 51 | 0.0050 | 0.0048 | 0.0097 | 49% |
| shared_change | jackpot | 71 | 0.0072 | 0.0013 | 0.0085 | 15% |
| shared_change | steady | 70 | 0.0024 | 0.0013 | 0.0037 | 35% |
| shared_change | all | 192 | 0.0049 | 0.0022 | 0.0071 | 31% |

Jackpot win probability is the largest win slice in every condition. The short-history gap versus the other conditions is mostly size, and that size error is on balanced and steady, not jackpot. Shared-change balanced is the other size pocket (0.0048).

## What was tried

Constants were chosen on train only. A change was eligible for the full scorer only when both train halves were no worse.

**1. Stronger (L, s) prior on a short-history mask.** Train histories were masked as the last 40 ticks plus 24 sampled older ticks. The size-rule likelihood was scaled by 1, 0.5, 0.25, and 0 (0 ignores the machine's wins and uses the pair prior only). Conditional size Brier on wins in the recent window, even/odd episodes:

| scale | half 0 | half 1 |
|---:|---:|---:|
| 1 (current) | 0.60717 | 0.61070 |
| 0.5 | 0.60899 | 0.61237 |
| 0.25 | 0.61177 | 0.61507 |
| 0 | 0.61728 | 0.62050 |
| 2 | 0.60779 | 0.61170 |
| 4 | 0.61034 | 0.61443 |

Scale 1 wins both halves. A softmax over the three win-pair gains, mixed into the same prior, was 0.60719 / 0.61078, a tie with the hard pair. Giving the masked likelihood the pair identified from the full history (an oracle key) reaches 0.60565 / 0.60941, about 0.0014. That is the ceiling of better pair identification, and the soft mixture does not reach it. Not scored on validation.

**2. `decay_missing=False` on the same mask.** Cutoff Brier of the pair mixer: True 0.33520 / 0.38926, False 0.33477 / 0.38983. The halves disagree. Jackpot is worse on both halves under False (0.25411 / 0.34635 versus 0.25349 / 0.34583). Not shipped.

**3. Jackpot live shrinkage, strength chosen by train cross-fit.** Grid 10, 30, 60, 120, 300, 1000 on the highest-mean jackpot Beta component. Means stayed at the round's target-derived levels. Both train halves improved at every step, so the rule kept **1000** (train halves 0.33047 / 0.37720 versus 0.33072 / 0.37864 at strength 10).

Validation rejected it.

| | v7c | strength 1000 |
|---|---:|---:|
| points | 57.37 | 57.26 |
| stable error (ratio) | 0.0082 (0.042) | 0.0083 (0.042) |
| short_history | 0.0133 (0.075) | 0.0145 (0.082) |
| drift | 0.0061 (0.032) | 0.0063 (0.033) |
| shared_change | 0.0071 (0.034) | 0.0070 (0.034) |
| tuning half | 57.31 | 57.30 |
| audit half | 57.39 | 57.12 |

Short-history win error went from 0.0090 to 0.0103. The train cutoff Brier liked pulling jackpot-live forecasts toward 0.669; the validation distribution did not. Code reverted.

## Kept

No change. No source files differ from v7c.
