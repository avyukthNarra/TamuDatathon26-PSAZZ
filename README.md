# House Edge: TAMU Datathon 2026 (team PSAZZ)

Forecasts of the next spin of hidden-rule slot machines: P(no win) and P(payout size | win) for every test query,
scored against the organizers' reference (`mainFiles/RULES.md`).

## What we built
- **Final model:** `adapted_entry_v17g_rules/`. An equal log-odds blend of three win models (v17-guards with short-history and
  jackpot patches, an exact Bayesian filter over lag-pair rules, and a sparse system-identification rule library), with v13's
  payout-size split and a 1e-4 floor. Every member learns its lag pairs from the round's own train data; the exact filter
  also learns 3-lag XOR and 2-lag AND/OR rules. A build-time member gate drops a member only if it clearly hurts on that
  round's validation.
- **Score:** main-round validation 57.97 / 60 cross-fitted (halves 57.98 / 57.96); Docker eval suite k-fold 57.86, 0 FAIL.
- **Reproducible:** offline, 2 CPUs, about 3 minutes; byte-identical output with or without validation answers mounted.
- **Round two:** `r2/run_round2.py` screens a new kit's structure, builds every candidate, scores them on the kit's
  validation, and recommends what to upload (a challenger must beat the default by > 0.1 on the mean and both halves).
- **Round-one upload:** recorded in `r2/README.md` (SHA-256 of the CSV and code ZIP).

## Round two in four commands
See `r2/README.md` ("Kit-day steps") for the full version.
```bash
KIT=/ABS/PATH/kit/public; OUT=/ABS/PATH/out
python3 r2/gate_members.py --entry adapted_entry_v17g_rules --kit $KIT   # ~4.5 min: gate.json + bundled/validation/
python3 r2/run_round2.py --kit $KIT --out $OUT --jobs 6                  # ~8 min: OUT/RANKING.md + submission CSVs
# then build the code ZIP from a clean copy (r2/README.md) and check the CSV:
python3 mainFiles/check_submission.py $OUT/blend/submission.csv
```

## Setup
- Python 3.13. The submitted entry needs only `numpy==2.3.4` and `scipy==1.16.3` (`adapted_entry_v17g_rules/requirements.txt`).
  Research scripts also use scikit-learn, LightGBM and pandas.
- Evaluation suite: `cd mainFiles && python3 -m eval_suite run --entry ../adapted_entry_v17g_rules --suite quick --out runs/eval/NAME`
  (Docker if available; with Colima set `TMPDIR` to a folder under your home directory).
- Simulated round-two kits: `cd mainFiles/adapt_bench && ./regenerate.sh`.

## Layout
| Path | What |
|---|---|
| `adapted_entry_v17g_rules/` | final model (default entry) |
| `adapted_entry_v17g/`, `adapted_entry/` | challengers: same blend without the rule learning, and with v13 as first member |
| `adapted_entry/ADAPTED.md`, `adapted_entry/RESEARCH/` | how the blend was adapted for round two, with evidence |
| `r2/` | round-two pipeline, member gate, kit-day steps |
| `orch/` | scoring helpers, paired bootstrap, `RESULTS.md` |
| `mainFiles/` | organizer kit (`public/`, rules, checker, ZIP builder), eval suite, adapt_bench, research code |
| `docs/sai/` | research harness, experiment reports and results |
| `AGENTS.md` | team playbook (branches, commits, PRs, changelog) |
| `CHANGELOG.md` | what changed, per PR |

## Team workflow
Work on `<name>/<short-desc>` branches and open PRs (see `AGENTS.md`). Sai's working branch is `sai/master`; it was merged
into `main` for round two (PR #30).
