# Hackathon agent kit

Drop-in instructions so every teammate's AI coding agent (Claude Code, Codex, Cursor,
Copilot) follows the **same** workflow: feature branch → tests → Conventional Commits →
changelog → PR → wait for review, with semantic versions computed by a script.

> Workflow source of truth: [AGENTS.md](../AGENTS.md). This page explains what each file is for.

## What each file is for

| File | Read by | Purpose |
|---|---|---|
| `AGENTS.md` | **Codex** (native), Cursor, Copilot, Jules, Aider, Zed, most agents; also humans | **Single source of truth**: workflow, semver, changelog, rules, living sections |
| `CLAUDE.md` | Claude Code | Imports `AGENTS.md` via `@AGENTS.md` + Claude-specific rules |
| `.cursor/rules/team-workflow.mdc` | Cursor (`alwaysApply: true`) | Condensed rules + pointer to `AGENTS.md` |
| `.github/copilot-instructions.md` | GitHub Copilot | Pointer to `AGENTS.md` |
| `.github/pull_request_template.md` | GitHub PRs | Forces what/why/tests/changelog/semver in every PR |
| `.claude/commands/ship.md` / `changelog.md` / `release.md` | Claude Code (`/ship`, `/changelog`, `/release`) | One-command workflows |
| `CHANGELOG.md` | Everyone | Keep a Changelog template with `[Unreleased]` |
| `scripts/next-version.sh` | Everyone / agents | Computes next semver from commits since last `vX.Y.Z` tag |
| `scripts/release.sh` | Release captain | Rolls `[Unreleased]` into `[X.Y.Z] - date` and commits |

## Setup still to do
1. Fill in **AGENTS.md §0** as the project starts: stack, commands, CI, team table, release captain, deadlines. The `TODO`s are expected until kickoff; whoever adds a piece fills its row in the same PR.
2. Delete **AGENTS.md §10 Databricks** if not relevant, or adjust catalog/schema names.
3. Add a `README.md` (pitch, how to run, "What we built") and `docs/DEMO.md` once there is something to describe. Agents read these first.
4. `main` can't be protected while the repo is private on a free plan, so the PR flow in AGENTS.md §1/§2 is followed by convention. Turn on branch protection (1 approval, squash only) if the repo goes public or the plan is upgraded.
5. After the first merged `feat`, run `scripts/next-version.sh` and tag `v0.1.0`.

## Script quick reference

```bash
scripts/next-version.sh             # explains bump + next version
scripts/next-version.sh --quiet     # just "0.3.0" (exit 3 if no release needed)
scripts/next-version.sh --current   # last tagged version
scripts/release.sh --dry-run        # preview the rolled CHANGELOG
scripts/release.sh                  # on release/vX.Y.Z branch: roll + commit
scripts/release.sh 1.0.0            # force a version (e.g. demo freeze)
```
