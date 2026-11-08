# Implementation plan (2026-11-07, 22:50)

Main closes Sun 10:30 AM. Round two is Sun 10:30 AM–12 PM, a 90-minute window.
Rule for every item: **round-two safe.** Every constant is re-derived from that round's train and validation data, with no main-round literals. Items are judged by cross-fitted validation plus the eval-suite `round2_shift` preset.
The upload stays the 56.29 entry (`SUBMIT_MAIN/`) unless a candidate beats it by more than about 0.4 points or by a paired CI above 0.

| # | Item | Expected gain | Evidence so far | Round-two safe? | Owner | Time |
|---|---|---|---|---|---|---|
| 0 | **Round-two hardening of v6**: fix whatever g7 finds (hard-coded fallback pairs or constants; pair discovery on shifted pairs); one-command rebuild script; dry run on simulated shifted data | protects the round-two score | g7 running | this *is* the safety work | Grok g7 → Codex after its reset (23:35) | 1–2 h |
| 1 | **Data-derived level prior**: component centres and spreads estimated by moments from the targets, not a fixed strength of 10 (fixes jackpot-live) | **+0.05 (noise), done** | h1: mean centres + moment-matched strengths give a cross-fit mean of 56.32 vs 56.28. Both halves are positive, but inside the noise; jackpot got slightly worse. Jackpot live isn't miscentred (mean 0.674 above 0.5) | yes | Haiku h1 | **dropped**, optional add-on |
| 2 | **Episode-level selector/blend: v6 vs simulator filter**, using an episode stability posterior from all 8 machines; the filter where the episode looks stable and long, v6 elsewhere | +0.2–0.8 | filter stable .046 vs .054; c1 win ties overall | yes, if the selector is learned or cross-fit | Grok g6 | running |
| 3 | **Size-atom rule** (what switches the atom). Then exact atoms: re-derived per round from targets, with the selector rule inside v6's size head | up to +1.6 (largest pot) | v6's size head is 40% of error; the "fixed per machine" idea is refuted (−2.4) | atoms re-read per round; rule must be learned | Grok g5 | running |
| 4 | Fix the filter's drift handling: a parametric ramp whose timing is learned on train histories; better short-history marginalisation | enables #2 | c1 confused drift with shared flips; the filter loses on short and drift | yes, if timing is learned from train | Grok (next wave) | 1 h |

**Don't pursue:** CTW/PPM/GRU (g3), neighbour/signal/extra-lag features (g4, g3), whole-model replacement by exact Bayes (c1), the fixed-per-machine size atom (refuted).

## Sequence
1. **Tonight, to about 00:30.** Collect h1, g5, g6 and g7.
   - Promote any item with a cross-fit gain into a single candidate entry on a new branch, e.g. `avyukth/v7-candidate`, built from Pranav's `COMBINED_SOLUTIONS` code.
   - Codex (after its reset, using Luna subagents) implements and packages it.
   - The eval suite (`--suite quick --mode docker`) gives the honest 4-fold score plus `round2_shift`.
2. **Morning, by 9:30 AM.** Upload the best entry that beats 56.29 honestly, or keep `SUBMIT_MAIN`. Freeze.
3. **10:30 AM, round two.**
   - Run the one-command rebuild (item 0) on the new kit.
   - Check the discovered pairs and the refit priors.
   - Run the eval quick check, then upload by about 11:15. That leaves 45 minutes of buffer.

## Update, 23:10: g5 breakthrough plus the web sweeps (g8 academic, g9 practitioner)
- **#3 size rule FOUND (g5).** Atom = s XOR [payout at t−L ≥ 2], with a per-machine L ∈ {1,2,4} and bit s, from a Bayesian posterior over (L, s). Plugged into v6: **57.03 (+0.74)**. Haiku h2 is verifying it, including round-two safety (L ∈ 1..8, atoms learned from train only, even/odd halves). **Top priority.**
- **New #5 short-history abstention (g8 #4, g9 #1).** `decay_missing=False`, plus skipping expert updates whose lag is missing, on both the win and size heads. Haiku h3 is testing the flag now. Next step: tune on train histories masked like short_history.
- **New #6 two-lag size rule search (g8 #1)** for g5's residual 13% misses.
- **New #7 round-two hardening (g8 #7, #8; g9 #6):**
  - AdaHedge or a train-picked eta in place of fixed eta 6/16/4;
  - half-lives picked on each round's train;
  - a power prior for main→round-two borrowing;
  - the rehearse / freeze / refit / bail-out runbook.
- Confirmed don'ts: tuning blend weights on validation, recalibration or stacking, Fixed-Share.
- 23:20: #5 step 1 (`decay_missing=False`): **no effect** (56.286 → 56.285), dropped. The size rule already lifts short history from .105 to .084.
- 23:25: **#2 replaced by the g6 flip swap: +0.18 [+0.08, +0.29]**, both halves positive.
  - It swaps v6's parity counts at the joint MAP flip tick, with a flat tau prior, using the train-discovered pairs.
  - Gains are on drift and shared change; win side only, so it's complementary to the size rule.
  - Half-life pooling and the c1 blend are about 0 (dropped).
  - **Next:** after g10 lands, add the g6 swap to the v7 entry, then score the combination (target ≈ 57.2).
- 23:35: **g11, size prior P(L, s | win pair) learned on train (λ = 1, chosen by train cross-fit): 57.20 vs 57.02 (+0.18)**. Halves 57.18 / 57.23; short .084 → .075. Round-two safe.
  - **Next:** after g12 (combination) lands, add it to the combined entry.
  - Remaining size headroom to the oracle: 0.50.
- 23:30: **v7 (size rule) packaged in `SUBMIT_MAIN_v7/`**: honest 57.00, eval suite 0 FAIL. **The user says don't update the submission until they say so.**
- 23:57: **c2 combination (g12):** size + flip + round-two patch = **57.20** on main (halves 57.14 / 57.23).
  - **Risk:** on g7's simulated rounds (static size atoms) the size rule is 2–3.5× WORSE than v6 (e.g. 0.0156 → 0.0521). The flip alone helps on all three simulated rounds.
  - **Fix (sent to Codex):** a per-machine 'static' size hypothesis, plus a train-level gate (rule vs v6 size head, by prequential score on train), merged into ONE entry v7c.
  - Acceptance: main ≥ 57.2 AND no worse than v6 on any simulated round.
