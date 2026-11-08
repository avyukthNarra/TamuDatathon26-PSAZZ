# M4 jackpot train prior: tuning results

Branch `avyukth/exp-m4-jackpot-prior`, commit `1cc8876` (pushed). Option: `jackpot_prior="train"` (default `"validation"`, unchanged). Only the jackpot win prior changes; it is Sai's train-only levels (0.121 / 0.612) ported exactly. Tuning = 48 of 96 validation episodes (harness split). Audit numbers were not used.

| Config | Tuning /60 | Err stable | Err short | Err drift | Err shared | Jackpot err | Paired Δ vs baseline [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---|
| Blend, validation prior (baseline) | 55.256 | .0165 | .0227 | .0165 | .0131 | .014759 | — |
| Blend + jackpot train prior | 55.279 | .0160 | .0225 | .0165 | .0134 | .014491 | +0.022 [-0.125, +0.192] |
| Mixer, validation prior (baseline) | 55.026 | .0198 | .0231 | .0168 | .0128 | .016795 | — |
| Mixer + jackpot train prior | 55.113 | .0187 | .0228 | .0166 | .0131 | .015715 | +0.087 [-0.107, +0.315] |

Jackpot error: mean over 123 tuning jackpot queries, squared error summed over 5 classes.

Findings:
- Baseline reproduces R0 (tuning 55.256 blend, 55.026 mixer).
- Jackpot-only error falls in both variants, but tuning point gains are within CI noise.
- Sai's +0.12 used 72 dev episodes and a batch re-implementation, so it is not directly comparable.

Commit body wrongly says "36 of 72" (correct: 48 of 96). A force-push fix was blocked by a hook, so it was not amended.
