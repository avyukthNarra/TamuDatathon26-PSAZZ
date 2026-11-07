# Versioning and releases

Moved from AGENTS.md §7 to keep the always-loaded playbook short.
AGENTS.md §7 keeps the rules every agent needs; this file has the details.

Version = `MAJOR.MINOR.PATCH`. **Git tags (`vX.Y.Z`) are the source of truth.** The version is computed from the Conventional Commits since the last tag. Never pick it by hand.

### Bump rules (take the highest that applies)

| Since last tag there is… | If current ≥ 1.0.0 | If current is 0.y.z (pre-1.0) |
|---|---|---|
| any breaking change (`!` / `BREAKING CHANGE:`) | MAJOR (`1.4.2 → 2.0.0`) | MINOR (`0.4.2 → 0.5.0`) |
| any `feat` | MINOR (`1.4.2 → 1.5.0`) | MINOR (`0.4.2 → 0.5.0`) |
| any `fix` / `perf` | PATCH (`1.4.2 → 1.4.3`) | PATCH (`0.4.2 → 0.4.3`) |
| only docs/chore/ci/test/refactor | no release needed | no release needed |

- Bumping MINOR resets PATCH to 0; bumping MAJOR resets MINOR and PATCH to 0.
- First release is `v0.1.0`. Cut **`v1.0.0` for the final demo/submission** (the "public API" is now what the judges saw).
- Pre-releases if needed: `v1.0.0-rc.1`, `v1.0.0-rc.2` → `v1.0.0`.

### Compute it (don't do it in your head)
```bash
scripts/next-version.sh            # prints e.g. 0.5.0 and why
scripts/next-version.sh --quiet    # prints only 0.5.0
```

### Release procedure (release captain)
```bash
git checkout main && git pull --ff-only
V=$(scripts/next-version.sh --quiet)
git checkout -b release/v$V
scripts/release.sh                 # moves [Unreleased] → [V] - YYYY-MM-DD in CHANGELOG.md and commits
git push -u origin HEAD && gh pr create --fill --title "chore(release): v$V"
# after the release PR is merged:
git checkout main && git pull --ff-only
git tag -a v$V -m "v$V" && git push origin v$V
gh release create v$V --notes-from-tag   # optional (GitHub) — or paste the changelog section
```
Suggested cadence during a hackathon: release after each meaningful merge batch, and always right before demo freeze.
