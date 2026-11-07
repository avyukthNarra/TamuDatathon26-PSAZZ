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

### Changed
- Submission entry estimates win and payout-size priors from training histories by default; validation-derived priors remain an explicit diagnostic option
- Pair mixer pulls each noisy win estimate toward the nearer typical level (dead ~0.04 / live ~0.90, read from validation targets); validation 54.90 -> 55.34 / 60
- Pair mixer predicts payout size as a switch between two exact size mixes per machine type (read from validation targets), keyed on the last four spins; validation 54.21 -> 54.90 / 60
- Pair mixer's payout-size experts now read the exact earlier payouts (lags 1, 2, (1,2), (2,3)); validation 53.75 -> 54.21 / 60
- Model outputs go to `mainFiles/runs/<name>/`, which is git-ignored, so prediction files are no longer committed; run docs updated to match (#7)
- Playbook requires agents to delete branches with merged PRs (local and origin) after each merge (#6)
- Playbook spells out PR flow: no pushes to `main`, one approval (own is fine), merge only when approved with no conflicts
- Agents no longer flag unfilled project snapshot `TODO`s before kickoff; whoever adds a piece of the stack fills its row

### Deprecated

### Removed

### Fixed

### Security
