# Late-night findings (2026-11-08, 00:40–02:05)

All scores:
- main validation /60, with cross-fit halves;
- g7 simulated round-two kits /40, listed disjoint / four / sharedlag.

Baseline v7c: 57.37 (57.31 / 57.39); sims 37.72 / 36.45 / 37.63.

## Worth taking
- The size-lag prior P(L,s | win pair) also works on Zhill's base: 57.32 → 57.52 (m2). Train-EM size atoms cost nothing against validation atoms (57.51).
- Flip swap on Zhill's base: +0.11 (m1).
- Pranav's HMM blend, switched OFF in episodes where the flip detector fires (e5):
  - 57.49 (57.53 / 57.39); sims 37.80 / 36.62 / 37.60;
  - the always-on HMM loses 0.37 on the late-flip sim.
- Episodes share a win-level offset beyond type × signal × state (h11, research_pf/h11-peer):
  - ICC 0.27 (steady 0.56, balanced 0.36, jackpot ~0);
  - as a peer prior on v7c it adds only +0.007 (e7).
- Signal prior +0.015 and missing-lag fractional size evidence +0.003 (m3, e3).

## Dead ends (don't repeat)
- AdaHedge: +0.07 main, −0.07 on the late-flip sim. Train-picked half-lives: −0.15 (e2).
- Win-level pooling toward the type centre: −0.04 to −0.11 (e1).
- Size polarity flip at shared change: refuted. A forced flip makes size error ~10× worse (e4).
- Joint pair/size-lag posterior: the learned coupling is 0 (e6).
- decay_missing=False: 0 (h3).

## Round-two warnings
- Zhill's 57.32 build is WORSE than plain v6 on all 3 g7 sims (h4). Pranav's v15 (57.55) beats v6 but trails v7c by 0.3–1.0 on the sims.
- On the eval-suite generator presets, v7c beats v6 on real data (+1.09) but LOSES on 5 of 7 presets (h10). Under investigation.
  - round2_shift −2.6
  - hidden_state_heavy −6.4
  - edge_cases −5.5
  - all_lags −3.7
  - fast_drift −2.9
- The remaining error is ~all win probability: a perfect win would remove 77.5% of v15's error. The biggest slices are short history and jackpot (h13).

## Update 02:15 — adaptability
- **Flip-switched v15 blend (e8, patch: `e8-v15switch.diff` against Pranav's v15 checkout).** In episodes where v7c's flip gate fires, the win blend uses v7c's win chance with weight w; elsewhere it keeps v15's weights.
  | w | main | halves | sim disjoint | sim four | sim sharedlag |
  |---|---:|---|---:|---:|---:|
  | v15 | 57.55 | 57.58 / 57.47 | 37.44 | 35.62 | 36.64 |
  | 0.8 | 57.55 | 57.55 / 57.50 | 37.62 | 35.82 | 37.31 |
  | 1.0 | 57.53 | 57.51 / 57.50 | 37.65 | 35.88 | 37.46 |

  Main is unchanged, and the shifted rounds gain +0.2 / +0.2 / +0.7. It should transfer to v16, whose blend weight is not episode-aware.
- **The cause of v7c's eval-preset losses is v7c's own round-two patch (`USE_ROUND2`: windowed pair voting, 2-means levels, cell-mode guard), not the size rule or the flip (m5).** With `USE_ROUND2` off, v7c matches v6 on round2_shift (23.92 vs 23.90) and all_lags (35.52 vs 35.51). It is being narrowed down to the responsible part.
