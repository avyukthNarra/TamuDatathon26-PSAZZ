---
description: Release captain only. Compute the next semver and prepare the release PR
---

Prepare a release per AGENTS.md §7. **Confirm with the user that they are the release captain before doing anything.**

1. `git checkout main && git pull --ff-only`. Working tree must be clean.
2. Run `scripts/next-version.sh` and show its explanation. If it says no release is needed, stop.
3. Show the `## [Unreleased]` section of `CHANGELOG.md`. If it's missing entries for merged `feat`/`fix`/`perf` commits since the last tag (`git log $(git describe --tags --abbrev=0 2>/dev/null)..HEAD --oneline`), add them first.
4. `V=$(scripts/next-version.sh --quiet)`; `git checkout -b release/v$V`; run `scripts/release.sh`.
5. Push and open a PR titled `chore(release): v$V`. Report the link and stop.
6. After the user says the release PR is merged: `git checkout main && git pull --ff-only && git tag -a v$V -m "v$V"`, then **ask before** `git push origin v$V` and any `gh release create`.
