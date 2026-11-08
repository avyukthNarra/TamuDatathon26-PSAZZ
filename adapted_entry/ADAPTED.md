# Sai's blend, adapted for round two

Same model as Sai's 57.97 blend: an equal log-odds average of three win models with v13's size split:
- v13 with the short-history patch and the jackpot patch;
- the exact Bayesian episode filter;
- the sysid rule library.

The difference: every member now LEARNS its lag-pair support from the round's train histories, instead of using the main-round pairs (1,3), (2,5), (4,7). As shipped, the exact filter and the short-history patch scored ~18/40 on simulated shifted rounds.

| | main validation /60 (cross-fit halves) | sim disjoint /40 | sim four | sim sharedlag |
|---|---|---:|---:|---:|
| this adapted blend | 57.961 (57.97 / 57.95) | 38.19 | 37.10 | 37.40 |
| published blend (pairs locked) | 57.973 (58.01 / 57.94) | ~18 | ~19 | ~18 |
| v7c | 57.37 (57.31 / 57.39) | 37.72 | 36.45 | 37.63 |

Docker eval suite (quick): PASS WITH WARNINGS, 0 FAIL. Honest 4-fold 57.84 (CI 57.49–58.19). Kit-test 189 s (limit 1800 s). Determinism passes; the no-validation run works. The two WARNs are the "refuses future-dated rows" probe.

Dependencies: requirements.txt pins numpy==2.3.4 and scipy==1.16.3 (allowed by the rules: pinned root-level requirements.txt, prepared by organizers before an offline run).

Run it from adapted_entry/: `python predict.py --data <public dir> --out submission.csv`.

Details: RESEARCH/s1-adapt_REPORT.md (pair learning), s2-package_REPORT.md (packaging), s3-final_REPORT.md (final integration and scores).

## Update: member 1 = v17-guards (adapted_entry_v17g/)
Per Sai's suggestion, member 1 is swapped from v13 to Zhill's v17-round2-guards (Pranav's v17 plus the v7c guard fix), with the same short-history and jackpot patches; the pairs are still learned.
- Main 57.971 (halves 57.98 / 57.96), against 57.961.
- 10-kit adapt_bench mean 38.388, against 38.371 (+0.017); no kit drops by more than 0.02.
- Main weak spot for both: K10 (30% of machines on 3-lag XOR or AND rules), ~32/40 against 38.8–39.4 on the other kits.
Recommended version: adapted_entry_v17g/.

## Update: gated v7c member for mixed-rule rounds (adapted_entry_v17g_gate/)
A train-only structure statistic (how much of the machines' win signal a single lag pair fails to explain) sets the weight of v7c's win chance as a 4th log-odds member.
- The weight is 0 on main, K1, K7 and K9; under 0.02 on K2–K6 and K8; 0.97 on K10.
- K10 (30% AND / 3-lag machines): 32.08 → 32.76. No other kit falls. adapt_bench mean 38.388 → 38.457.
- On main the forecasts equal adapted_entry_v17g/ (57.971).
Recommended version: adapted_entry_v17g_gate/.
