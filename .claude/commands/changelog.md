---
description: Add the right CHANGELOG [Unreleased] entries for this branch's commits
---

Update `CHANGELOG.md` for the current branch, per AGENTS.md §6:

1. List this branch's commits: `git log --format='%h %s%n%b' origin/main..HEAD`.
2. Map each user-visible commit to a subsection: `feat` → Added (or Changed if it alters existing behavior), `fix` → Fixed, `perf` → Changed, removals → Removed, security → Security. Skip pure docs/chore/ci/test/refactor unless user-visible.
3. Write one concise, user-facing line per change under `## [Unreleased]`, linking the PR number if known. Prefix breaking changes with **BREAKING:**.
4. Don't duplicate lines that already exist; don't touch released sections; don't add a version header.
5. Show the diff of `CHANGELOG.md` and commit it as `docs(changelog): update unreleased entries`.
