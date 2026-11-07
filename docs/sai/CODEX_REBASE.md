# Note for Codex (from the lead): base changed, rebase your E/S work onto B4

The lead merged `codex/w3-w4` (E1-E5, S1) into `sai/master-test` and re-scored your prediction CSVs through the harness
(your numbers reproduce). Findings:
- E1-E5: all within noise of B3 (45.75-45.81) and add 0.00 to B4 as a blend member -> neutral.
- S1 (HistGB residual on B3): 46.59, +0.77 [+0.07,+1.51] vs B3. It helps drift/shared_change but hurts stable.

**New base B4** (the old B3 with its win model fixed, `results/B4.json`, `preds/B4_validation.csv`): lag 7 + pairs
(1,3),(2,5),(4,7), ridge lambda 0.05 (near-OLS; B3's lam=20 over-shrank, its lags 1-5 hurt). Dev 50.89 [49.8,51.9] vs B3 45.82.
Reference code: `methods/_core.py` (`predict(split, cfg)`), `reports/L2.md`. Error shares on B4: p0 .49, p1 .25, p2 .19, p5 .07.

Please (all comparisons and blend_gain now against B4: `P_B4 = read_pred_csv("preds/B4_validation.csv", v)`):
1. Rebuild the base column of your feature cache from B4 (not B3) and rerun S1 as S1_B4 (residual on B4), then S3, S4, S5, S7
   as you planned. S1's drift/shared_change gain is the thing to test: does anything remain over B4 (B4 conditions: .106/.216/.211/.183)?
2. Contract fixes: predictions to `preds/<ID>_validation.csv` (not `results/preds/`), one small `results/<ID>.json` per ID
   (no 12k-line grids inside; put grids in `grids/<ID>.json`), reports in `reports/<ID>.md` (<=15 lines). Source in normal
   `methods/<ID>.py` files, not embedded in JSON.
3. Same guardrails: 1 BLAS thread, nice + timeout, at most 2 processes, never score audit/all.
