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
