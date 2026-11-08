# Archived full evaluation reports

These reports were emitted by eval-suite source commit `21f81c69d937b4c0b8e5d46c02f8613b6a9c4eef` (`21f81c6`). For privacy and portability, the report entry path is the only edited line: the fixture's absolute path is shown as `eval_suite/fixtures/reference`, and the team entry path is shown as `<team-best entry dir>`. All other report text is preserved from the original output. The raw reports remain under `mainFiles/runs/eval/`.

The source test suite passed 158 tests in 125.877 seconds. The archived reports preserve emitted caveats that were later corrected: `entry.py` enforced `/tmp` and `/output` tmpfs quotas, and identical missing-vs-zero predictions on the reference fixture are inconclusive because that fixture ignores history.

- `fixture-reference-full.md`: `python3 -m eval_suite run --entry eval_suite/fixtures/reference --suite full --mode docker --out runs/eval/fixture-reference-full`; generated 2026-11-08T00:51:28Z; elapsed 117.248s.
- `team-best-55.58-full.md`: `python3 -m eval_suite run --entry /Users/avynarra/Documents/Datathon/worktrees/zhill-5558/mainFiles/runs/entry-5558/entry --suite full --mode docker --out runs/eval/zhill-5558-full`; generated 2026-11-08T00:51:28Z; elapsed 1011.897s.

The team run used package worktree HEAD `fd8f7dd2216fe91acb3694fbe837884186164c54`; packaged `predict.py` SHA-256 was `2c1e0d48bf516afc35616d820037e195d328a9557c34b9f32b4d5ef25219fc7d` (the package was ignored and is not captured by the source commit alone).
