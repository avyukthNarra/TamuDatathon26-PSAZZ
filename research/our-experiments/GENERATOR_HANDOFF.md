# Eval-suite generator handoff

## Quota checkpoint — 2026-11-07, 13% remaining

Stopped at Claude's quota alert. No Luna step or background generator task is running. Do not resume work until Claude confirms the quota reset or assigns the next step.

Worktree: `/Users/avynarra/Documents/Datathon/worktrees/eval-suite`; branch: `avyukth/eval-suite`.

What exists: the seven-preset generator, train-only calibration module, and generator/calibration regression tests listed below. The full suite last passed **158/158** in **140.201 seconds**; focused tests passed **12/12**. No tests were rerun for this checkpoint.

What remains: no generator implementation or test fix is outstanding. Claude has integrated the three files in commit `21f81c6` (`feat(eval): add spec-driven data generator and calibration`) and owns any further release work. The known calibration limitation remains the lag-correlation gap **0.1337**, above the **0.10** descriptive band; further tuning is not authorized. The generator files currently have no uncommitted changes in this worktree.

Reports: `_coord/GENERATOR_REPORT.md`; detailed calibration artifacts: `/tmp/eval-runtime-final/main_like` and `/tmp/eval-calibration-selected.json`. Temporary artifacts may need regeneration if removed; commands are below.

Queue boundaries: R3b was skipped. R4/X1 worktrees were closed empty, with their local branches left unpushed; X2 never started. No R-item changes the shipped entry and no real audit scoring is pending.

## Ownership and boundaries

- Calibration worker: `mainFiles/eval_suite/calibration.py` and this handoff/report. Uses public **train only** and kit data readers; does not load team forecasters or validation targets.
- Generator worker: `mainFiles/eval_suite/generator.py`; owns simulator and `main_like` preset. Calibration returns reviewed config overrides to the generator owner; calibration does not edit generator code.
- Test worker: `mainFiles/eval_suite/tests/test_generator.py`.
- Shared scope: no commits, pushes, branch changes, or edits to another owner's files.

## Agreed API

Calibration exports `summarize_dataset(data_root_or_train_dir) -> dict`, `compare_statistics(real, synthetic) -> dict`, and `calibrate_main_like(real_train_dir) -> dict`. Generator exposes `PRESETS`, `preset_config(name, overrides=None)`, `generate_dataset(out_dir, *, preset='main_like', seed=0, episodes_per_condition=None, config=None) -> Path`. `GeneratorConfig.base_win_rates` uses signal order `[-1, 0, 1]`; `payout_mixes` uses positive payout order `[1, 2, 5, 20]`.

## Calibration and verification commands

From `mainFiles/`:

```sh
python -m eval_suite.calibration --real public --generated /tmp/eval-generated --out /tmp/eval-calibration.json --calibrate
python -m eval_suite.generator --preset main_like --seed 0 --out /tmp/eval-main-like
python -m unittest discover -s eval_suite/tests -t . -v
```

Focused generator/calibration checks: `python -m unittest eval_suite.tests.test_generator -v`.

Calibration diagnostics use predeclared descriptive bands: max type×signal win-rate gap .08; payout-mix TV .10; lag-correlation gap .10; machine win-rate SD gap .06; cutoff mean gap 8 ticks; history sampling pattern exact. These are diagnostics, not constraints on the broader scenario presets.

## Status / limitations

Complete: `mainFiles/eval_suite/calibration.py` summarizes only public/train, reports machine-readable stats and fixed-band comparisons, and returns main-like overrides. Generator owner applied observed-history-derived base win rates, payout mixes, and selected `interaction_strength=1.3`; all other broad mechanisms remain. For the fixed same-seed sweep, strengths .7/1.0/1.3 were compared; the final permitted hidden-state trial (.9 strength/.95 persistence) made an undesirable cross-type lag mismatch and was rejected. Selected configuration still misses the lag-correlation band: max absolute phi gap .1337 (real balanced lag7 .2022 vs generated .0685). Other registered bands pass. Results and per-preset wall times are in `GENERATOR_REPORT.md`.

Final table provenance (already generated; no rerun needed): N=24/condition `main_like` dataset is `/tmp/eval-runtime-final/main_like`; its comparison JSON is `/tmp/eval-calibration-selected.json`.

Verification complete: 12 focused generator/calibration tests and the full 158-test suite passed (140.201s, `OK`). From `mainFiles/`, reproduce the full suite with:

```sh
python -m unittest discover -s eval_suite/tests -t . -v
```

For a fresh reproducible main-like diagnostic, generate N=24 episodes/condition and summarize/compare only train data:

```sh
python -m eval_suite.generator --preset main_like --seed 2407 --episodes-per-condition 24 --out /tmp/eval-main-like
python -m eval_suite.calibration --real public --generated /tmp/eval-main-like --calibrate --out /tmp/eval-calibration.json
```

No validation targets, forecasts, or team models informed calibration. Synthetic data intentionally retains the documented broad mechanism family even where train summary statistics cannot all be matched.

Delivered files: `mainFiles/eval_suite/generator.py`, `mainFiles/eval_suite/calibration.py`, and `mainFiles/eval_suite/tests/test_generator.py`. Codex delegated these files to separate Luna workers and reviewed their outputs; Codex and its workers made no commits or pushes. At the quota checkpoint, all three files are tracked and have no uncommitted changes; check current git history for Claude's integration commits.


## Quota reset follow-through — 2026-11-07

The earlier 13% quota pause is superseded by the confirmed reset. No generator implementation or test work remains outstanding. Bounded CLI verification succeeded for `main_like` and `round2_shift` (seed 73, one episode per condition): 32 truth rows matched each public test query set; public test contained only queries/history; test truth files loaded; and serialized truth forecasts passed `check_submission.py` with main/60 and adaptation/40 metadata. The subprocess-based `protocol.generate_preset` caller also succeeded for main_like seed 74. Artifacts: `/tmp/eval-cli-mainlike-73`, `/tmp/eval-cli-round2-73`, `/tmp/eval-protocol-mainlike-74`. No source edits or suite reruns were made in this follow-through.
