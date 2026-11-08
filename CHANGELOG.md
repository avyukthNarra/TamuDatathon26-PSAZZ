# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Versions are computed from [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
by `scripts/next-version.sh`. See AGENTS.md §5–§7.

<!--
How to add an entry (every PR):
- Put it under [Unreleased], in the right subsection (create the subsection if missing).
- One user-facing line, link the PR: "- Weekly checkout trend chart on the dashboard (#14)"
- Breaking changes start with **BREAKING:**
- Merge conflict here? Keep BOTH sides' lines.
- Never add a version header yourself; scripts/release.sh does that.
-->

## [Unreleased]

### Changed
- Round-two script defaults to the rules-learning adapted blend (`adapted_entry_v17g_rules/`, #24); the previous default stays as challenger `blend_v17g` (main kit unchanged, 57.97)

### Added
- Round-adaptive member gate for the adapted blend (`r2/gate_members.py` + `gate.json`): drops a member only if the cross-fitted validation blend without it is better on both halves by >= 0.1; main and sims keep all members (byte-identical), a broken member is dropped (+1.3 to +3 points)
- Shared experiment harness, master test plan, and results leaderboard under docs/sai (reproduces pooled 30.00 and pair prototype 46.07)
- Independent system-identification forecaster with train-only fitting, cross-fitted validation, and reproducible simulator diagnostics
- Causal residual stacking of mixer forecasts with separate prior-fold training and checkpoint-only test prediction
- Independent self-supervised global learning experiment with causal in-context features, train-only calibration, and episode cross-fitted evaluation
- eval_suite: --jobs N runs independent entry executions in parallel (identical results)
- Win-probability form study: cutoff chances are per-machine logit-normal levels on a parity cell that has already finished flipping
- Validation forecast that pairs the train-history win posterior with the v13 size split (`validation_crossfit.csv`, 57.68 points)
- Method-agnostic evaluation suite (`python -m eval_suite run --entry DIR`) that checks any entry against all 88 documented requirements and scores it with the official metric (#19)
- Agent playbook: drop-in protocol, branch and commit conventions, PR template, and release scripts
- Empirical-Bayes latent-state experiment with reproducible cross-fitted validation and uncertainty reporting
- Short-history win filter that marginalizes missing lag bits and replaces only v13's win probability
- Cross-fitted v13 patch and raw short-history win probabilities for the selected grid filter
- adapt_bench: simulated round-two benchmark (10 kits, quick/full tiers) for adaptability checks

### Changed
- Model outputs go to `mainFiles/runs/<name>/`, which is git-ignored, so prediction files are no longer committed; run docs updated to match (#7)
- Playbook requires agents to delete branches with merged PRs (local and origin) after each merge (#6)
- Playbook spells out PR flow: no pushes to `main`, one approval (own is fine), merge only when approved with no conflicts
- Agents no longer flag unfilled project snapshot `TODO`s before kickoff; whoever adds a piece of the stack fills its row

### Deprecated

### Removed

### Fixed
- Local-mode answer-isolation checks are reported as NOT RUN, and omitted mode selects Docker when available (#19)

### Security
