---
description: Ship a feature end-to-end (branch, implement, test, changelog, push, PR) per AGENTS.md
argument-hint: <short description of the feature or fix>
---

Ship this change following AGENTS.md §2 exactly: **$ARGUMENTS**

1. **Sync:** `git checkout main && git pull --ff-only origin main`. If the working tree is dirty, stop and ask.
2. **Branch:** create `<name>/<short-desc>` (e.g. `sai/genie-dashboard`; name = the human's first name, 2-4 kebab words). The Conventional type goes in the commit/PR title, not the branch.
3. **Plan:** state a 3–6 bullet plan in chat, then implement. Stay in scope.
4. **Test:** apply the AGENTS.md §2 Step 4 table. Write the required tests, then run the project's lint + test commands from AGENTS.md §0. Fix failures; never weaken tests silently.
5. **Commit:** small Conventional Commits (AGENTS.md §5) with a body giving why, gotchas, and how verified.
6. **Changelog:** add user-facing line(s) under `## [Unreleased]` in `CHANGELOG.md` in the right subsection. Commit as `docs(changelog): ...` or include in the feature commit.
7. **Semver impact:** run `scripts/next-version.sh` and note what the next release would be.
8. **Sync again:** `git fetch origin && git rebase origin/main` (resolve CHANGELOG conflicts by keeping both sides), re-run tests if anything changed.
9. **Push + PR:** `git push -u origin HEAD`, then `gh pr create` with a Conventional-Commit title and the filled `.github/pull_request_template.md`. Request a reviewer from the ownership table.
10. **Stop.** Report: branch, PR link, CI status (`gh pr checks` once), test output summary, changelog line, semver impact, open questions. Do **not** merge or tag.
