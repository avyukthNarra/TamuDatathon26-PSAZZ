# Sai's iterations and thinking (raw material for the Devpost post)

A chronological dump of what Sai (working with AI coding agents) tried, why, what the numbers said, and what changed as a result.
Scores are main-round validation /60 unless marked /40 (round two). Raw notes live in `docs/sai/` (`FINAL.md`, `RESULTS.md`,
`COMBOS_TOP10.md`, `TEAM_COMPARISON.md`), `orch/RESULTS.md`, `adapted_entry/ADAPTED.md` and `r2/README.md` on `sai/master`/`main`.

## 0. Setup: make the team (and agents) fast and honest (Sat 11:00-14:00)
- Wrote the team playbook (`AGENTS.md`): branch-per-feature, Conventional Commits, changelog, PR flow, merge rules,
  so humans and coding agents (Claude Code, Codex) could work in parallel without breaking `main`.
- Built a shared experiment harness: fixed dev/audit split (72 dev episodes, 24 audit episodes scored once at the end),
  the official scoring formula, bootstrap CIs, a smoke split. Rule: choose everything on dev, touch audit once.
- **Thinking:** with 96 validation episodes, noise is the main enemy. Every claimed gain needs a paired CI, not a point score.

## 1. Classical baselines and the first big jump (Sat 14:00-15:00): 45.8 -> 50.9
- Started from the team's pair prototype B3 (46.8 audit). Ran four worker tracks in parallel: W1 frequency/shrinkage (A1-A8),
  W1 lag structure (L1-L10), W2 memory/changepoint methods (M1-M10), W3/W4 episode/peer features and residual learners (E1-E5, S1-S8).
- **B3 -> B4, +5.1 dev / +5.5 audit:** B3's ridge penalty (lambda 20) over-shrank real lag effects. Lag 7 + pairs (1,3),(2,5),(4,7)
  at lambda 0.05 (near-OLS) was the real win model. Lags/pairs learned from train with null controls.
- Size model: how much a win pays depends on lags 1, 2, 4 (A9 severity model, +0.58), plus an alpha-64 size-share prior (+0.27).
- M9b peer-gated episode changepoints (+0.33): pool changepoint evidence across an episode only when at least 2 of 7 peers show a regime start.
- **Insight:** the win model and the size model use disjoint lag structure, which is why their gains added up.
- Did not work: more lag structure (all pairs, triples, one-hot, MTD), HMM, ADWIN, BOCPD, condition specialists, MLPs, peer features,
  gradient-boosted residuals on top of B4 (<= +0.09).

## 2. Stacking, then admitting the teammates' model was better (Sat 15:00-17:00): 54.1 audit
- W5 stack of all members: 54.09 audit [53.1, 54.9]. But the simple C2 composition was within noise of it.
- Pulled and re-scored teammates' branches through the same harness: Zhill's rule model + Pranav's pair mixer (20/80) scored 55.37 audit,
  above our stack. Pranav's two-mode size mixer pushed it to 55.90 audit.
- **Decision:** switch to the team's best model and spend our effort testing additions to it, rather than defending our own stack.
- Exhaustive blending: K1 (324 compositions), K2 (1681 two-stage structures), volatility gates, MLP meta-learner: all within +0.25 dev,
  nested checks put the real held-out gain at about +0.007. **Blending was saturated.**

## 3. Find where the error actually is (Sat 17:00-20:00): 55.3 -> 56.0
- Swap-in-truth diagnosis on the v4 mixer: perfect win probability -> +3.0; perfect sizes -> +1.4.
  True win probabilities are continuous (deciles .01 ... .96), not two levels.
- **Thinking:** stop blending; build a better win-probability function.
- Tried Bayesian Walsh-logistic win heads, a screened Walsh-logistic BMA expert (7.4x the win MSE, blend weight 0),
  an exact 8-hypothesis size switch (loses 0.3-1.1), jackpot win diagnostics. Mostly negative, but they mapped the problem.
- Mixer5/mixer6 from Pranav (+0.6 over v4) became the base; also wrote a self-contained research brief and ran Gemini research on it.

## 4. Fresh-start win models overnight (Sun 00:45-02:15): 57.50 -> 57.97
Ran four independent "fresh" research tracks with agents, each forbidden to copy the existing model:
- **Exact Bayesian filter** (`sai/fresh-exact`): exact posterior predictive under the generative model: pair x direction enumerated,
  logit-normal levels on a 25x25 grid, joint episode inference over condition and one shared change tick, exact sum over missing
  win bits for short histories.
- **Sysid** (`sai/fresh-sysid`): sparse rule library from train; unchanged / machine-reversal / episode-reversal hypotheses.
- **Short-history win patch** (`sai/fresh-bayes`): a 256-state distribution over the last eight win bits so scattered older ticks still count.
- **Jackpot patch** (`sai/fresh-sbi`): jackpot win from the active four-cell rate of the machine's pair, shrunk to the parity pool.
- Self-supervised HGB with causal per-machine summaries: 52.60 vs 34.47 without them (lesson: expose local evidence; don't assume shared signs).
- **Blend:** equal log-odds average of v13+short+jackpot, exact, sysid = **57.97** (halves 58.01 / 57.94), +0.47 [+0.21, +0.77] over v13.
  Why it works: members' win errors correlate only 0.78-0.84 and each is best somewhere different (sysid on stable/drift, exact on short
  histories, v13 on balanced machines). Equal weights were not tuned; fitted weights scored lower (57.85).

## 5. Make it adapt for round two (Sun 02:15-05:20)
- Risk: the blend hard-coded the main-round pairs; on simulated shifted rounds it scored about 18/40.
- Sai: made the exact filter learn its pair set from the kit's histories (all 28 pairs, census + floor; main 57.478 vs 57.49
  hard-coded) and wrote the round-two structure screen (`orch/r2screen/`: single lags, pairs, triples).
- Avyukth: adapted the whole blend so every member learns its lag pairs per round (`2730597`), swapped member 1
  to Zhill's v17-round2-guards on Sai's suggestion (`3083c87`), made the exact filter learn 3-lag XOR and AND/OR rule families from
  the train census (`979d2d9`), and built the 10-kit simulated adaptability benchmark (K1-K10). Adapted blend: about 38/40 on shifted kits
  vs about 18 locked; main unchanged (57.97); the rules version lifted the hardest kit K10 from 32.1 to 35.4.
- Sai: built `r2/run_round2.py`: one command screens the new kit's structure, builds every candidate, scores cross-fitted validation,
  runs paired bootstraps, checks the CSVs, writes RANKING.md with a recommendation. Pick rule (team decision): a challenger replaces the
  default only if it wins by > 0.1 overall AND on both halves.
- Zhill: the round-adaptive member gate (`r2/gate_members.py`) and bundled validation so the entry runs identically with or without
  validation mounted.
- Docker eval suite: PASS WITH WARNINGS, 0 FAIL, honest 4-fold 57.86.

## 6. Round two day (Sun 10:35-11:40): 38.59 / 40
- Kit arrived 10:35. Pipeline ran end to end: pairs moved to (3,8), (1,4), (2,6), size lags to {1,2,3}; every member relearned them
  (votes 185/180/164 vs 5 for the next best; 0% of machines on old pairs). Default blend: **38.59/40** cross-fit, best of all candidates.
- Orchestrated three Codex workers on short-history ideas plus Avyukth's bots, with a shared brief and a skip list to avoid duplicate work.
- Diagnosis: true P(win) -> 39.63, true sizes -> 38.93; short history's unusual gap is sizes (true sizes cut its ratio 0.049 -> 0.0315,
  about 3x any other condition); only 529/768 machines get a confident per-machine pair. Ceiling for perfect short-history predictions: +0.46.
- Final hour, all below the bar: pair-identity prior (+0.022, one half negative), short-history level pooling (+0.008),
  recalibration (-0.022; already calibrated), size shrink (worse), 3-lag priors (-0.001), blend weights/temperature (-0.09).
- Caught and corrected our own mistakes in public: the Walsh screen's 3-lag "rules" were artefacts (the rule census said parity only),
  and the size member is v17-guards, not v13 as the file name suggests.
- Uploaded Avyukth's build of the default (Docker-verified); Sai's independent build matched it to 3.2e-14.

## Lessons (candidates for "What we learned")
1. Measure where the error is before building: swap-in-truth told us win probability, not blending, was the lever (twice).
2. Blending saturates fast; diversity of members (errors correlated 0.8, not 0.95) is what made the final blend work.
3. Learn structure from each round's data instead of hard-coding it; that one change was worth about 20/40 on shifted rounds.
4. Honest evaluation beats point scores: cross-fitting, condition-balanced halves, paired bootstraps and a pre-agreed margin stopped
   us from shipping noise at least ten times.
5. Agents scale research, but only with a written playbook, a shared harness, and an orchestrator who keeps a skip list.

## Timeline of scores
| When | Model | Score |
|---|---|---|
| Sat 14:00 | B3 pair prototype | 46.8 audit (/60) |
| Sat 14:45 | B4 lag 7 + pairs, near-OLS | 52.3 audit |
| Sat 15:00 | W5 stack | 54.1 audit |
| Sat 16:40 | rule + mixer v3 (team) | 55.9 audit |
| Sat 20:00 | rule + mixer5 | 55.96 cross-fit |
| Sun 01:40 | v13 alone | 57.50 |
| Sun 02:00 | 3-member blend + patches | 57.97 |
| Sun 05:00 | adapted rules blend (final entry) | 57.97 main; ~38/40 simulated shifts |
| Sun 11:00 | same entry, real round-two kit | 38.59 / 40 |
