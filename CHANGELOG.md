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

### Added
- Lag-pair mixer forecaster with a verified, reproducible entry ZIP (`mainFiles/run_pair_mixer.py`); validation 53.75 / 60
- Agent playbook: drop-in protocol, branch and commit conventions, PR template, and release scripts
- House Edge rule model (v4): per-machine lag-pair win rules plus payout-size rules, 52.89 / 60 on validation vs 30.83 for the best starter, with a one-command entry builder for main and round two
- Combined entry: 20% rule model + 80% pair mixer, 54.45 / 60 on validation (built by `mainFiles/make_entry.py`)
- `mainFiles/auto_tune.py` re-picks the blend on each round's validation and keeps it only if the audit half confirms a real gain

### Changed
- Pair mixer's payout-size experts now read the exact earlier payouts (lags 1, 2, (1,2), (2,3)); validation 53.75 -> 54.21 / 60
- Model outputs go to `mainFiles/runs/<name>/`, which is git-ignored, so prediction files are no longer committed; run docs updated to match (#7)
- Playbook requires agents to delete branches with merged PRs (local and origin) after each merge (#6)
- Playbook spells out PR flow: no pushes to `main`, one approval (own is fine), merge only when approved with no conflicts
- Agents no longer flag unfilled project snapshot `TODO`s before kickoff; whoever adds a piece of the stack fills its row

### Deprecated

### Removed

### Fixed

### Security
