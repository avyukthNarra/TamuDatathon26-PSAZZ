# Round-two (adaptation) upload files, 2026-11-08

Upload these two files to the round-two Form (and add them to Devpost with these names). Do not change the main-round files.

- `ADAPTATION_submission.csv`: test forecasts, 1536 queries. `check_submission.py --split test` on the real kit: VALID FORMAT, adaptation, test.
- `ADAPTATION_entry.zip`: code ZIP (`build_entry.py`), entry `adapted_entry_v17g_rules/` from this branch + `gate.json` (all three members) + `bundled/validation/` (= the kit's `validation/`).
- SHA-256 in `SHA256SUMS`.

Model: Sai's adapted blend, relearned on round two (pairs (1,4),(2,6),(3,8), size lags {1,2,3}). Cross-fitted validation 38.588 / 40.
Docker rerun (python:3.13, pinned numpy 2.3.4 / scipy 1.16.3, offline, 2 CPUs, real 20-point kit): matches this CSV, max diff 9.7e-15.
Rebuild: `python3 adapted_entry_v17g_rules/predict.py --data <kit>/public --out submission.csv` (about 5 min).
