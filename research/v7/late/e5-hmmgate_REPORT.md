# HMM blend, episode-aware

When the train flip gate fires, that episode's machines use `w_flip` instead of the flat HMM weight 0.3. Stable and short-history rows never fired, so their loss ratios stay on the `--signal --hmm` numbers. The blend path matches `predict_split` to 0. The flat run reproduces the prompt: main 57.54 (57.62 / 57.38), kits 37.75 / 36.70 / 37.26.

Acceptance, all required: main ≥ 57.45, halves ≥ 57.29 and ≥ 57.37, kits ≥ 37.70 / 36.43 / 37.61.

## Result

**No variant is accepted.** (a) is the closest. It passes main, both halves, disjoint, and four. Sharedlag is 37.5998, short of 37.61 by 0.010.

| variant | main | halves (tuning / audit) | disjoint | four | sharedlag | accepted |
|---|---:|---:|---:|---:|---:|---|
| flat 0.3 | 57.54 | 57.62 / 57.38 | 37.75 | 36.70 | 37.26 | no (baseline) |
| (a) w_flip=0, stable 0.3 | 57.49 | 57.53 / 57.39 | 37.80 | 36.62 | 37.60 | no |
| (b) w_flip=0.1, stable 0.3 | 57.52 | 57.57 / 57.40 | 37.80 | 36.65 | 37.53 | no |
| (c) train cross-fit, per kit | 57.46 | 57.56 / 57.25 | 37.61 | 36.84 | 37.04 | no |

(c) weights from each kit's own train, not from main: main fired/stable **0.0 / 0.65** (48 and 48 train episodes), disjoint **0.5 / 0.1**, four **1.0 / 1.0**, sharedlag **0.0 / 1.0**. The audit half and two kits miss. Train one-hot Brier is a weak stand-in for the contest probability loss, and it pushes the stable bucket away from 0.3.

## Per-condition loss ratios

Gate fired on 48/96 main validation episodes and 12/24 on every simulated kit. Stable and short_history ratios do not move for (a) or (b).

Main, flat → (a) → (b) → (c):

| condition | flat 0.3 | (a) | (b) | (c) |
|---|---:|---:|---:|---:|
| stable | 0.036 | 0.036 | 0.036 | 0.035 |
| short_history | 0.073 | 0.073 | 0.073 | 0.076 |
| drift | 0.029 | 0.031 | 0.030 | 0.031 |
| shared_change | 0.033 | 0.034 | 0.034 | 0.034 |

Sharedlag, the late-flip kit, flat → (a): drift 0.093 → 0.061, shared_change 0.059 → 0.053. That is most of the recovery from 37.26 toward v7c's 37.63, and it stops at 37.5998. Episodes the gate does not fire still keep the 0.3 HMM blend, and the signal prior stays on, so this is not a return to v7c.

Disjoint ratios, (a): stable 0.047, short_history 0.061, drift 0.072, shared_change 0.052.
Four ratios, (a): stable 0.063, short_history 0.136, drift 0.121, shared_change 0.050.

## Reproduce

From `mainFiles/`, variant (a), which is the closest and is **not** accepted:

```sh
python run_pair_mixer.py --data public --signal --hmm --hmm-flip-weight 0 --cv-val --score-only
python run_pair_mixer.py --data /Users/avynarra/Documents/Datathon/research_pf/g7-round2/data/sharedlag/public --signal --hmm --hmm-flip-weight 0 --score-only
```

Same flags with `disjoint` or `four` in place of `sharedlag`. Config equivalent: `USE_SIGNAL = True`, `USE_HMM = True`, `HMM_CROSSFIT = False`, `HMM_FLIP_WEIGHT = 0.0`, `HMM_STABLE_WEIGHT = None` (that None means 0.3). (b) is `--hmm-flip-weight 0.1`. (c) is `--signal --hmm --hmm-crossfit`, which refits both bucket weights on that kit's train.
