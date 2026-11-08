# Round two (adaptation, 40 points)

Avyukth's build of the default `adapted_entry_v17g_rules/`, relearned from the round-two train data (no code changes).
Files copied from `avyukth/r2-adaptation` @ `a80ddc6` (`r2/adaptation_submission/`).

- `ADAPTATION_submission.csv` SHA-256 `a7ec520dd737c40502bbb2778f1c1973efdf224e494811dd9243c3798033190f`
  (organizer `check_submission.py --split test` on the real kit: VALID FORMAT, 1536 queries, adaptation, test)
- `ADAPTATION_entry.zip` SHA-256 `feb33968c7070000891f84547e8a896a3c1e5e0c4538498be840f995d76729d2`

Verification:
- Docker rerun (python:3.13, numpy 2.3.4, scipy 1.16.3, offline, 2 CPUs, real kit) matches the CSV to 9.7e-15.
- Sai's independent build (`submission.csv` `108434136029fcf2…`, `entry.zip` `1b467e5d…`) agrees to 3.2e-14; its ZIP has identical code,
  and rerunning it on the real kit reproduced its CSV byte for byte in 159 s.

What changed this round: win pairs (1,3),(2,5),(4,7) -> (1,4),(2,6),(3,8), size lags {1,2,4} -> {1,2,3}; all members relearned them
from train. Validation 38.59 / 40 cross-fitted on a copy of the kit with `max_points` 40 (the kit's metadata says 20).
No challenger cleared the pick rule (> 0.1 overall and on both halves); closest was Pranav's v21 peer term (38.67, +0.08).
Experiments: `docs/devpost/research/round_two/` and `docs/devpost/WRITEUP.md`.
