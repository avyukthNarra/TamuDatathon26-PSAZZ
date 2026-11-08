# Round two (adaptation): one-command runbook

Branch: `pranav/residual-hunt` (has v8, v9 and v10). Total time: ~7 minutes (`--quick`: ~3).

## When the new kit opens
1. Download and unzip the new kit into its OWN folder (do not overwrite the main kit), e.g.
   `~/HouseEdge-round2/`. Check `~/HouseEdge-round2/public/metadata.json` says `"round": "adaptation"`.
2. From this repo:
   ```sh
   cd mainFiles
   git checkout pranav/residual-hunt
   python round_two.py --data ~/HouseEdge-round2/public
   ```
3. Upload the two files it prints, from `~/Downloads/HouseEdge_ADAPTATION_submission/`, to the
   **Adaptation** form: `submission.csv` and `entry.zip`. Save the receipt.

## What the script does
- Refits everything from the NEW data (lag pairs, size lags, size modes, win priors per type and
  signal, peer centre), so changed patterns are relearned, not assumed.
- Builds v10 (log-odds pool + peer correction), v9 (log-odds pool) and v8 (COMBINED_SOLUTIONS),
  scores each on cross-fitted validation, and picks the best (ties within 0.02 -> simpler model).
  If the new patterns break an idea (e.g. no peer effect any more), a simpler candidate wins.
- Verifies the choice: kit checker on test + validation, and a fresh read-only rerun of the ZIP
  without validation targets must match the CSV byte for byte. Stops with an error otherwise.

## If short on time
`python round_two.py --data ~/HouseEdge-round2/public --quick` builds only v10 (~3 minutes).

## Rehearsal on the main kit (2026-11-08)
`python round_two.py --data public --tag rehearsal`: v10 56.59 (cross-fit 56.56), v9 56.54, v8 56.51;
chose v10; checker OK; byte-identical rerun; 6.7 minutes; SHAs match the earlier builds.
