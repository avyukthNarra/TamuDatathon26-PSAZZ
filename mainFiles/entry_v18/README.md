# v18: v17 (round-two guards) + Sai's exact filter, 50/50 log-odds win, v17 sizes

Backup round-two candidate. Build: run `python run_strongest.py --out runs/v17` from `mainFiles/`, copy
`runs/v17/entry/` to a folder, rename its `predict.py` to `predict_v17.py`, then add this folder's
`predict.py`, `requirements.txt` and `research_exact/`. `reproduce.json` is unchanged.

Verified in Docker (`python:3.13-slim`, 2 CPUs, numpy 2.1.3 / scipy 1.14.1):
- main validation 57.947 (halves 58.00 / 57.89); test file valid, 1536 rows; test run ~14 min (limit 30).
- simulated round-two kits (avyukth simulator.py, /40): disjoint 38.04, four 36.78, sharedlag 37.51.

Recommended instead: avyukth/sai-blend-adapted (57.98 main, 38.18 / 37.13 / 37.45 sims, ~4 min).
