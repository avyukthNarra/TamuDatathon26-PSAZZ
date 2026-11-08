# Research findings: new approaches to House Edge (2026-11-07, 21:58–22:52)

Agents:
- Grok 4.7 High ×5 (g1–g5)
- Codex GPT-6.1-Sol with Luna subagents (c1)
- Claude Haiku (verification)
- Claude: coordination and a size check

Every folder is under `~/Documents/Datathon/research_pf/` and holds its PROMPT, code and REPORT. Bar to beat: Pranav's pair mixer v6 at **56.29**, with loss ratios stable .054, short .105, drift .049, shared .056.

## 1. What the simulator actually is (g1, verified by Haiku)
| Piece | Finding | Status |
|---|---|---|
| Win bit | `payout > 0` (pushes count as wins). Recoding pushes as losses destroys the rule | confirmed |
| Win rule | Each machine uses one pair from {(1,3), (2,5), (4,7)} (≈⅓ each) and a coin-flip XOR/XNOR. Live/dead is set by the parity of its own win bits. No third lag, no neighbours, no hidden state (residual lag-1 correlation 0.00) | confirmed: the query state is right for 95% of stable machines |
| Win levels | Per machine: logit-normal around a type×signal centre (residual sd ≈ 0.55). Jackpot live is centred near **0.50–0.58**, not the 0.67 v6 assumes | confirmed (g1, g4) |
| Stable | Never changes | confirmed |
| Drift | Live/dead swap, roughly linear over ~96→34 ticks before the cutoff, finished at the query | reversal confirmed (94%); shape not provable |
| Shared change | One abrupt direction flip on all 8 machines, near 0.64 of the cutoff. Pair, levels and size are unchanged | confirmed (16/24 episodes flip all 8) |
| Payout size | Given a win, exactly **two atoms per type** (exact to 1e-9 in the targets) | confirmed |
| Size atom choice | g1 and Haiku said "fixed per machine, not lags 1/2/4". **Claude refuted the fixed part:** a full-history posterior picks the query atom only ~66%, and plugging it in scores 53.88 (−2.4). Something switches the atom; lags 1/2/4 alone don't explain it | **open → g5 investigating** |

## 2. Where the remaining error is (g4)
- v6 squared error: **60% win probability, 40% payout size**.
- Perfect win probability with v6's sizes scores **58.44 (+2.15)**. The last ~1.6 points to 60 are size.
- **Jackpot live** is about half of all win-probability error: truth 0.58 ± 0.16, v6 explains only 30% of the variance (RMSE 0.137). Its prior is too tight and too high.
- **Short-history steady** is mostly a size problem (62% of its error).
- v6 is already well calibrated (deciles within ±0.02). Signal, neighbours, extra lags and size-state features add at most 3.6% of win-probability error. Don't add them.

## 3. Alternative methods tried (g3): all lose
62 prototypes:
- **Best:** a likelihood mixture of the same pairs, 54.51.
- **Context Tree Weighting:** 36.57. It only sees contiguous suffixes, and the pairs skip lags.
- **PPM:** 27.60.
- **Stacked log-odds:** 49.44.
- **GRU:** judged to add about 0.
- **Blends:** every 80/20 or 50/50 blend with v6 is worse than v6.

**Generic sequence learners are a dead end.** The loss is squared error, and v6's expert weights already target it.

## 4. Simulator inversion: particle filter / exact Bayes (g2, c1)
| Model | Points | stable | short | drift | shared | Note |
|---|---:|---:|---:|---:|---:|---|
| v6 (bar) | **56.29** | .054 | .105 | .049 | .056 | |
| g2 Rao-Blackwellized filter, exact rule enumeration + per-machine changepoints (hyperparameters fitted on train) | 55.47 | **.046** | .162 | .059 | .060 | **beats v6 on stable**; loses on short and drift |
| c1 exact Bayes inversion (Codex) | 53.96 | .093 | .141 | .109 | .105 | its size posterior is confidently wrong; it reads every drift episode as a shared flip |
| c1 win probability + v6 sizes | 56.36 | | | | | +0.07 [−0.48, +0.58], noise |
| 80% v6 + 20% c1 | 56.34 | | | | | +0.05, noise |

**Reading:** simulator inversion is *competitive* on win probability and *better* when the episode is stable. It is worse when data is scarce (short history) or the change is gradual (drift). Neither replaces v6 alone. The upside is in **choosing per episode**, which g6 is testing.

## 4b. Other checks (Claude)
- Size atom vs live/dead state: weak link (jackpot dead leans to the small atom, 125/68). It doesn't explain atom switching.
- Still running: **g5** (what selects the size atom), **g6** (episode-level regime pooling and a per-episode blend of v6 and the filter), **g7** (round-two stress test of v6 on simulated data with shifted pairs, plus a hard-coded-constant audit).

## 5. Conclusions
1. The only fundamentally different approach with real upside is **model-based inversion of the known simulator**. It's exact over 6 rule hypotheses × level grid × episode scenario (stable / drift / shared flip), with episode-coupled flips. It targets the 2.15-point win ceiling directly.
2. The cheapest high-value fixes inside v6:
   - (a) a wider, lower jackpot-live prior;
   - (b) hard-coded exact size atoms (v6 learns approximate ones), once g5 finds what selects the atom.
3. Don't pursue CTW/PPM/GRU, neighbour or signal features, or blending.

## 6. g2 final (particle filter)
- **Best variants:**
  - 3-pair episode-changepoint filter (hazard .01, half-life 48): **56.30**, a tie. Ratios .047 / .118 / .045 / .053.
  - 20% filter + 80% v6: **56.35** (+0.07, noise).
- **Per condition, the filter beats v6 on stable, drift and shared, and loses on short history.**
  - Oracle per-condition pick: about .045 / .105 / .045 / .053, giving ≈ **56.5 (+0.2)**, but only if the episode condition is identified. That's g6's job.
- Short-history steady is weak in both systems (mixer .035, filter .040). That's the open problem.
