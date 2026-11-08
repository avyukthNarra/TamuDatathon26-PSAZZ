# Round one (main, 60 points)

Submitted through the main-round form by Sai from `sai/master-test` @ `65e55a2`.

- `submission.csv` SHA-256 `076014ef241d8b61b6b608a513e350293a309294c538f0125a35d9e80aaba364`
  (`check_submission.py`: VALID FORMAT, 1536 queries, main, test)
- `entry.zip` SHA-256 `f3dae4e5fdb0a6c83d981f4222bff152a1cf89f6ba091f43a9cdee035250743d`
  (742 KB: clean `git archive` of `adapted_entry_v17g_rules/` + main-round `gate.json` + `mainFiles/public/validation/` as `bundled/validation/`)

Model: equal log-odds blend of v17-round2-guards (+ short-history and jackpot patches; supplies sizes), the exact Bayesian filter
and sysid; the member gate kept all three. Main validation 57.97 / 60 cross-fitted (halves 57.98 / 57.96); Docker eval suite
PASS WITH WARNINGS, 0 FAIL, honest 4-fold 57.86.
