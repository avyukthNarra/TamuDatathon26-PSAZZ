# CLAUDE.md

@AGENTS.md

The import above pulls in the full team playbook. **AGENTS.md is the source of truth.** Put shared rules there, not here. This file only adds Claude Code-specific behavior.

## Claude Code specifics

- **Start of every task:** confirm you're on an up-to-date `main`, then create the feature branch (AGENTS.md §2 Steps 0–2) *before* editing any file.
- **Slash commands** in `.claude/commands/`:
  - `/ship <short description>`: runs the full feature workflow (branch → implement → test → changelog → push → PR/MR → stop).
  - `/changelog`: adds the right `[Unreleased]` entries for the current branch's commits.
  - `/release`: release captain only; computes the version and prepares the release PR.
- **Versions:** always run `scripts/next-version.sh`; never compute a semver bump yourself.
- **Commits:** Conventional Commits (AGENTS.md §5). Write the body for the next agent: why, approach, gotchas, how verified, follow-ups. As long as needed, no padding. End each commit message with the team's agreed co-author trailer if one is configured.
- **Branches:** `<name>/<short-desc>`, e.g. `sai/genie-dashboard` (AGENTS.md §2 Step 2).
- **PR/MR:** use `gh` (GitHub) or `glab` (GitLab) per AGENTS.md §0. Fill `.github/pull_request_template.md`. After opening it, report the link + CI status and **stop**. Do not poll CI in a loop, do not merge.
- **Permissions:** pushing a *feature branch* and opening a PR is pre-approved by the team. Merging (only after approval, with no conflicts, on the human's "merge" in chat), tagging, force-pushing, deleting branches, and spending cloud money always need a human's explicit OK in chat. GitHub doesn't enforce branch protection on this private repo, so never push to `main` yourself.
- **Living doc:** when you hit a non-obvious gotcha or the team makes a decision during your task, append one dated line to AGENTS.md §12 or §13 in the same PR.
- **Reporting:** finish with: branch, PR link, actual test output summary, changelog line, semver impact (`major`/`minor`/`patch`/`none`), open questions.
