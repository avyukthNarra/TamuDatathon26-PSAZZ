# Submitted files

## Main round (submitted by Sai from `sai/master-test` @ 65e55a2; keep unchanged)
- `submission.csv` SHA-256 `076014ef241d8b61b6b608a513e350293a309294c538f0125a35d9e80aaba364`
- `entry.zip` SHA-256 `f3dae4e5fdb0a6c83d981f4222bff152a1cf89f6ba091f43a9cdee035250743d`
- Originals: Sai. Record: `r2/README.md` on `sai/master`.

## Round two (adaptation)
- `ADAPTATION_submission.csv` SHA-256 `a7ec520dd737c40502bbb2778f1c1973efdf224e494811dd9243c3798033190f`
- `ADAPTATION_entry.zip` SHA-256 `feb33968c7070000891f84547e8a896a3c1e5e0c4538498be840f995d76729d2`
- Files: branch `avyukth/r2-adaptation` @ `a80ddc6`, folder `r2/adaptation_submission/`:
  `git fetch origin && git checkout origin/avyukth/r2-adaptation -- r2/adaptation_submission`
- Checks: `check_submission.py --split test` on the real kit: VALID FORMAT (1536 queries, adaptation, test).
  Docker (python:3.13, numpy 2.3.4, scipy 1.16.3, offline, 2 CPUs): matches to 9.7e-15. Sai's independent build agrees to 3.2e-14.
