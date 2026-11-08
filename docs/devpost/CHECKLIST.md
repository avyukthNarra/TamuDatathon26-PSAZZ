# Devpost submission checklist (House Edge, TAMU Datathon 2026)

Nothing here is uploaded automatically. Each step is done by hand on Devpost.

## Before you start

- [ ] Find the original round-one files from Sai (he uploaded them and has the originals).
- [ ] Check their SHA-256 hashes against the values in `submissions/round1_main_60/README.md`:
  - `submission.csv`: `076014ef241d8b61b6b608a513e350293a309294c538f0125a35d9e80aaba364`
  - `entry.zip`: `f3dae4e5fdb0a6c83d981f4222bff152a1cf89f6ba091f43a9cdee035250743d`
- [ ] Put the verified copies at:
  - `submissions/round1_main_60/submission.csv`
  - `submissions/round1_main_60/entry.zip`
- [ ] Do not substitute any other file for these two. The files in `SUBMIT_MAIN/`, `SUBMIT_MAIN_v7/` and `research_pf/` are different runs.

## Round one (main): submit the project now

1. [ ] Open the Devpost project for House Edge.
2. [ ] Paste the text of `docs/devpost/WRITEUP.md` into the project fields, leaving out the section `## Round two (adaptation)`.
3. [ ] Upload the main-round files:
   - `submissions/round1_main_60/submission.csv`
   - `submissions/round1_main_60/entry.zip`
4. [ ] Confirm the uploaded files match the hashes above, and that the Devpost submission is saved.
5. [ ] Do not edit or re-upload the main files after this point.

## Round two (adaptation): add before noon

1. [x] Round-two entry built and verified: `submissions/round2_adaptation_40/ADAPTATION_submission.csv` (sha256 a7ec520d…) and `ADAPTATION_entry.zip` (sha256 feb33968…); Docker rerun matches.
2. [x] Choose the file to upload from `RANKING.md` (the default `blend` unless a challenger clears the bar). Decided 11:28: the default `blend`; no challenger cleared the bar.
   - Sai's laptop built the same model independently: `submission.csv` sha256 108434136029fcf2… + `entry.zip` 1b467e5db237f345… (both checkers: VALID, 1536 queries, adaptation, test). Either pair is fine, but upload one person's CSV and ZIP together, never one from each.
   - **Upload Avyukth's pair** (committed on `avyukth/r2-adaptation` at `a80ddc6`, `r2/adaptation_submission/`): `ADAPTATION_submission.csv` sha256 a7ec520dd737c405… + `ADAPTATION_entry.zip` feb33968c7070000…. Cross-check 11:38: his CSV is VALID on the real kit and equals Sai's build to 3.2e-14; the ZIP code is identical (only gate.json notes differ: same members). Docker rerun with pinned numpy/scipy matches to 9.7e-15.
3. [ ] Check the test CSV with `mainFiles/check_submission.py --split test`.
4. [ ] Rename the files as they are added to Devpost, using these exact labels:
   - `ADAPTATION_submission.csv` (the test-set predictions)
   - `ADAPTATION_entry.zip` (the code ZIP built with `mainFiles/build_entry.py`)
5. [ ] Add both as new files to the existing Devpost project. Do not replace or rename the main-round files.
6. [x] Fill in the `## Round two (adaptation)` section of `WRITEUP.md` (score, what changed, which candidate was uploaded), remove the TODO lines, and update the Devpost text to match.
7. [x] Round-two lag pairs confirmed: (1,3),(2,5),(4,7) moved to (1,4),(2,6),(3,8) (pair census and forensics).
8. [x] Round-two Form upload recorded (`sai/master` r2/README.md); hashes in `SUBMISSIONS.md`.
9. [ ] Add both ADAPTATION_* files to Devpost before noon.

## Do not

- [ ] Do not change, replace or delete the main-round files on Devpost.
- [ ] Do not upload anything that has not been checked with `check_submission.py`.
- [ ] Do not put validation answers, tokens or any secrets in the submitted ZIP.
