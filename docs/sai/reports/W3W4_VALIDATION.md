# W3/W4 artifact validation
Validated E1–E5, S1–S8, and S5_NESTED; no verdicts assigned.
All14 prediction CSVs contain768 unique IDs in exact harness split order.
Probabilities are finite, within[0,1], and row-normalized within1e-6.
Harness dev scores,four ratios,1000-bootstrap CI95,and class shares reproduce saved JSON.
Nested S5 paired B3 difference and CI also reproduce; all stored determinism diffs<1e-8.
All reports<=15 lines; all stored Python sources compile; each method grid has32 unique section5 configs.
S1/S8 grids128;S2 grid128(64 CatBoost,64 LightGBM);S3–S7 grids64;E grids160–480.
Future-tick perturbation and exact-tick missingness checks pass; condition feature names absent at inference.
Audit was not scored; shared dense cache stayed read-only; harness diff against worktree base7795dc4 is empty.
Validation runtime0.91s;Python3.13.3;NumPy2.3.3;one process;thread caps1;600second timeout.
