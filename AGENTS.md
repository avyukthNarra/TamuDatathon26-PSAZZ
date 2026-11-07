# AGENTS.md: Team Playbook (living document)

> **Single source of truth** for how humans *and* AI coding agents (Claude Code, Codex,
> Cursor, Copilot…) work in this repo. Tool-specific files (`CLAUDE.md`,
> `.cursor/rules/*.mdc`, …) only point here and add tool quirks.
>
> **Living doc:** if you (human or agent) learn something the next person needs, add it to
> [§12 Learnings](#12-learnings--gotchas-append-only) or [§13 Decision log](#13-decision-log-append-only)
> in the same PR. Keep edits small and dated.

---

## Drop-in protocol (agents: do this first, every time)

You have just been dropped into this repo with a task. The project may be early-stage and
under-documented, so **do not assume anything is set up.** Run this loop before touching code.

### A. Orient (read-only, ~2 minutes)
1. Read this file, then skim: `README*`, `CHANGELOG.md`, the file tree, and whatever manifest exists
   (`requirements.txt`, `pyproject.toml`, `package.json`, `Makefile`, `.github/workflows/`).
2. `git status`, `git branch --show-current`, `git log --oneline -10`. If the working tree is dirty
   with changes you did not make, stop and ask; never stash or discard them.
3. Restate the task in one or two sentences, and note which files/areas it should touch.

### B. Audit and report what's missing
Check the items below and tell the user, briefly, which are missing or unfilled. Don't fix
them unless they block the task or the user asks; infrastructure gaps are their own PRs.

- [ ] §0 `TODO`s: **expected until kickoff, so don't flag them** unless one blocks your task (see the note under §0)
- [ ] No source layout yet (`src/`, `notebooks/`, `pipelines/`, `tests/`)
- [ ] No test runner or tests, so you can't prove the task works (say how you'll verify instead)
- [ ] No lint/format config
- [ ] No CI (`.github/workflows/`)
- [ ] No `README.md`, `.env.example`, or dependency manifest
- [ ] Remote or `main` missing (`git remote -v`, `gh repo view`), or `main` unprotected
- [ ] No issue/ticket for the task (use the one the user gave; otherwise omit the number)

If a missing piece blocks you (e.g. no way to run tests), pick the smallest sensible option, say so,
and put it in the PR. A new dependency or CI change needs the user's OK first (§4).

### C. Do the task (this is the whole job)
1. **Sync:** `git checkout main && git pull --ff-only origin main` (§2 Step 0).
2. **Branch:** `git checkout -b <name>/<short-desc>` (e.g. `sai/genie-dashboard`) before editing any file (§2 Step 2).
3. **Implement with the smallest diff that solves the task.** Minimally invasive means: touch only
   files the task needs; match existing style; no reformatting, renaming, or reorganizing of
   unrelated code; no new dependencies; no drive-by fixes (note them in the PR instead).
4. **Test:** write tests per §2 Step 4, then run lint + tests. If they fail, fix them. Don't skip, delete,
   or weaken tests. Don't open the PR until they pass (or, if there is no test setup, until you've
   verified by running the code and said exactly how).
5. **Changelog + commits:** Conventional Commits with context in the body (§5); one line under `## [Unreleased]` (§6).
6. **Push and open the PR:** `git push -u origin HEAD`, then `gh pr create --base main` with a
   Conventional-Commit title and the filled `.github/pull_request_template.md` (§2 Step 7).

### Agent do's and don'ts (quick reference)
| Do | Don't |
|---|---|
| Ask one concise question when the task is ambiguous or touches something irreversible | Guess, or ask five questions you could answer by reading the code |
| Commit small and often, with a body that explains the why and any gotchas | Squash everything into one giant commit, or leave a bare subject on a non-obvious change |
| Keep the PR description short: what, why, how tested | Paste a wall of text or re-narrate every file |
| Reuse existing helpers, patterns, and naming | Introduce a new framework, style, or folder structure |
| Say plainly if something failed, was skipped, or is untested | Claim "all tests pass" without running them |
| Leave unrelated problems alone and mention them in the PR | Fix them in the same branch |
| Stop and ask if the task needs > 2x the expected effort (§9) | Silently expand scope |

### D. Hand off and stop
The **user reviews, approves, and merges**. You never merge, tag, or force-push. Report: branch,
PR link, actual test output summary, changelog line, semver impact, the "what's missing" list from
step B, and open questions. Then stop. Don't poll CI.

---

## 0. Project snapshot (fill in as the project starts)

> **`TODO` here is expected until kickoff.** Stack, commands, and CI don't exist yet, so don't nag
> about them. Whoever first introduces a piece fills in its row **in that same PR** (adding `pytest`
> also fills Tests and the `TODO_LINT && TODO_TEST` line in §2 Step 4). Pitch, deadline, team table,
> and release captain are the humans' to fill in. Agents only raise a `TODO` if it blocks the task.

| Field | Value |
|---|---|
| Project name | `TamuDatathon26-PSAZZ` |
| One-line pitch | `TODO` |
| Hackathon / deadline | TAMU Datathon 2026. `TODO` demo freeze + judging times |
| Repo host | GitHub ([avyukthNarra/TamuDatathon26-PSAZZ](https://github.com/avyukthNarra/TamuDatathon26-PSAZZ)), PRs via `gh` |
| Default branch | `main` |
| Stack | `TODO` (e.g. Databricks + PySpark + Delta + AI/BI Dashboards) |
| Install | `TODO` (e.g. `pip install -r requirements.txt`) |
| Run locally | `TODO` |
| Lint / format | `TODO` (e.g. `ruff check . && ruff format .`) |
| Tests | `TODO` (e.g. `pytest -q`) |
| CI | `TODO` (e.g. GitHub Actions: lint + tests on every PR) |

### Team & ownership

| Person | Handle | Owns (area / folder) | Reviewer for |
|---|---|---|---|
| `TODO` | `@` | `TODO` (e.g. ingestion / `pipelines/`) | `TODO` |
| `TODO` | `@` | `TODO` (e.g. ML / `models/`) | `TODO` |
| `TODO` | `@` | `TODO` (e.g. dashboard / demo) | `TODO` |

**Release captain** (the only person who tags releases): `TODO`

---

## 1. Golden rules (non-negotiable)

1. **Never commit or push directly to `main`.** Every change lands through a PR/MR.
2. **One branch = one feature/fix = one PR.** No drive-by unrelated changes.
3. **`main` must always be demo-able.** If it is broken, fixing it beats everything else.
4. **Every PR updates `CHANGELOG.md` → `## [Unreleased]`** (unless it is purely `chore`/`ci`/`docs` with no user-visible effect, and even then adding a line is welcome).
5. **Commit messages are Conventional Commits with a useful body** (§5). The version number is *derived* from them. Never invent a version by hand.
6. **Do not merge your own PR** unless the team has explicitly agreed to it for that PR (e.g. 10 minutes before demo freeze). Agents **never** merge.
7. **No secrets in git**: tokens, `.env`, Databricks PATs, connection strings. Use `.env` (git-ignored) or the platform secret store.
8. **No force-push to shared branches** (`main`, `release/*`, a branch with someone else's name on it). `--force-with-lease` on *your own* feature branch only.
9. **Small PRs** (aim < 400 changed lines). Big feature? Split it, or put it behind a flag.
10. **When unsure, ask** in the team channel or the PR. Do not guess on anything irreversible.

---

## 2. The feature workflow (step by step)

Agents: follow these steps **in order** and report which step you are on.

### Step 0: Sync
```bash
git checkout main
git pull --ff-only origin main
```

### Step 1: Claim the work
- Find or create an issue (GitHub Issue / GitLab Issue). Assign yourself.
- If no issue tracker is in use, post in the team channel: "I'm taking X".

### Step 2: Create a branch
Format: `<name>/<short-desc>`: the teammate you're working for, a slash, 2-4 kebab-case words.
Branch names can't contain spaces, so `Sai - genie dashboard` becomes `sai/genie-dashboard`.

```bash
git checkout -b sai/genie-dashboard
git checkout -b sai/fix-null-dates
```

- `<name>` = the human's first name, lowercase. Don't know it? Use `git config user.name`, or ask.
- Keep it simple: the *what*, not the story. ≤ 30 chars after the slash. No dates, no issue numbers.
- The change type (feat/fix/…) lives in the commit message and PR title, not the branch.
- Only exception: `release/vX.Y.Z` (release captain, §7).

### Step 3: Implement
- Stay inside the scope of the issue. Note unrelated problems in the PR description or a new issue; don't fix them here.
- Commit early and often with Conventional Commit messages (§5).
- Match existing code style; run the formatter before committing.

### Step 4: Test (decide using this table)

| Change | Tests required? |
|---|---|
| New feature with logic (transforms, APIs, model code, business rules) | **Yes**: unit tests for the core logic + one happy-path integration/E2E test if the project has them |
| Bug fix | **Yes**: a regression test that fails before the fix and passes after |
| Refactor | Existing tests must pass; add tests if coverage of the touched code is missing |
| Data pipeline / notebook change | **Yes**: at least a data check (row counts, nulls, schema, uniqueness) on a sample |
| UI-only tweak, copy change, styling | No new tests; manually verify and say so in the PR |
| Docs / chore / CI | No |
| Spike / throwaway prototype | No, but mark the PR `spike` and don't merge to `main` |

Then run the **full** local check:
```bash
# replace with the real commands from §0 (still TODO until kickoff: say how you verified instead)
TODO_LINT && TODO_TEST
```
If something fails, fix it. **Never** skip, delete, or weaken a test to make it pass without saying so explicitly in the PR.

### Step 5: Update the changelog
Add one line per user-visible change under `## [Unreleased]` in `CHANGELOG.md`, in the right subsection (§6). Do **not** create a version header; that happens at release time.

### Step 6: Pre-push checklist
- [ ] Branch is up to date with `main` (`git fetch && git rebase origin/main` or merge it in)
- [ ] Lint + tests pass locally
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] No secrets, no large data files, no stray debug prints / `TODO`s you added
- [ ] Commits follow Conventional Commits
- [ ] If you learned something non-obvious → added to §12/§13 of this file

### Step 7: Push and open the PR/MR
```bash
git push -u origin HEAD

# GitHub
gh pr create --base main --fill --title "feat: add Genie dashboard" --body-file .github/pull_request_template.md
# GitLab
glab mr create --target-branch main --fill --title "feat: add Genie dashboard"
```
- **PR title = a Conventional Commit header** (it becomes the squash-merge commit).
- Fill the PR template: what, why, how tested, changelog line, semver impact.
- Link the issue (`Closes #12`).
- Request a reviewer from the ownership table (§0).

### Step 8: Wait (don't merge)
- Wait for **CI green** and **≥ 1 approval** (or whatever §0 says).
- **Agents:** after opening the PR, report the PR/MR link, CI status, and anything left for a human, then **stop**. Do not merge, do not keep polling for minutes. If asked to check later: `gh pr checks` / `gh pr view --comments` (or `glab mr view`).
- Address review comments with **new commits** (don't rewrite history mid-review), reply to each comment, re-request review.
- If CI fails: read the log (`gh run view --log-failed`), fix on the same branch, push.

### Step 9: Merge (human)
- **Squash merge** (default) so `main` gets one Conventional Commit per PR.
- Delete the branch after merge.
- Everyone else: `git pull` on `main` and rebase their branches when convenient.

### Step 10: Release (release captain, §7)

---

## 3. Keeping in sync & resolving conflicts
- Rebase/merge `main` into your branch at least every couple of hours during a hackathon.
- **`CHANGELOG.md` conflicts are expected**: resolve by **keeping both sides' lines** in the right subsection. Never drop someone else's entry.
- Lockfile conflicts: take `main`'s version, then re-run the install command to regenerate.
- Notebook (`.ipynb`) conflicts: prefer Databricks source format (`.py` with `# COMMAND ----------`) or strip outputs before committing to avoid them.
- If a conflict is in code you don't own, ping its owner instead of guessing.

---

## 4. Agent behavior rules
- Read this file before starting any task. Re-read §2 before pushing.
- State the plan briefly, then act. Work on **one** feature branch per task.
- **Allowed without asking:** creating a feature branch, committing to it, running tests/linters, pushing the feature branch, opening a PR/MR.
- **Ask first:** merging anything, tagging/releasing, deleting branches, force-pushing, changing CI, adding a new dependency, touching another person's owned area, anything that costs money (cloud clusters, paid APIs), editing secrets/config.
- **Never:** push to `main`, merge your own PR, commit secrets, disable tests or CI checks, rewrite published history, run destructive commands on shared data (`DROP TABLE`, `rm -rf`, `DELETE` without `WHERE`) without explicit approval.
- When done, report: branch name, PR link, test results (actual output summary), changelog line added, semver impact, open questions.
- If the instructions here conflict with a direct request from a human teammate, follow the human, and suggest updating this file.

---

## 5. Commit messages: Conventional Commits 1.0

```
<type>(<optional scope>)<optional !>: <description in imperative mood, lowercase, no period>
```

**Write for the next agent.** Commit messages are how agents (and humans) rebuild context from
`git log` without re-reading every diff, so give them what the diff can't show.

- **Subject:** one line, ≤ 72 chars, says *what* changed.
- **Body (blank line after the subject):** *why* it changed, the approach and any alternative you
  rejected, constraints or gotchas the next person would trip over, how you verified it, and
  known gaps or follow-ups. Wrap at 72. Use a short bullet list when there are several distinct points.
- **Size to the change:** a typo fix is a subject only; a design decision or non-obvious fix earns a
  real paragraph. Don't under-explain (agents can't ask you) and don't pad: no recap of what the
  diff already shows line by line, no "this commit…", no filler, no emoji.
- **Footers:** `BREAKING CHANGE:`, `Closes #12`, the co-author trailer.

| Type | Meaning | Version bump |
|---|---|---|
| `feat` | New feature | **MINOR** |
| `fix` | Bug fix | **PATCH** |
| `perf` | Performance improvement | **PATCH** |
| `refactor`, `style`, `test`, `docs`, `build`, `ci`, `chore`, `revert` | No user-visible behavior change | none (on its own) |
| any type with `!` after type/scope, or a `BREAKING CHANGE:` footer | Incompatible change | **MAJOR** |

Examples:
```
feat(dashboard): add weekly checkout trend chart
fix(ingest): handle null return dates in checkouts feed
perf(silver): cluster checkouts table by user_id
feat(api)!: rename /books endpoint to /catalog
refactor: extract date helpers into utils
docs: add Databricks setup steps to README
```

Bad subject: `feat: updated the dashboard to add a new chart and also fixed some bugs and cleaned up imports` (three changes in one commit; split them).
Good subject: `feat(dashboard): add weekly trend chart`, with a body such as:
```
Aggregates checkouts by ISO week in gold_weekly_checkouts rather than
in the chart query, because the raw table is too large to scan per
page load. Weeks with no checkouts are filled with 0 so the line
doesn't skip. Verified against the 2025 sample: 52 rows, no nulls.
Follow-up: timezone is UTC; library local time may shift week edges.
```

Scopes: use the area names from the ownership table (e.g. `ingest`, `silver`, `gold`, `model`, `dashboard`, `api`, `ui`).

---

## 6. Changelog: Keep a Changelog 1.1.0

File: `CHANGELOG.md`. Newest version at the top. Sections, in this order, only if non-empty:

| Subsection | Commit types that usually feed it |
|---|---|
| `### Added` | `feat` |
| `### Changed` | `feat`/`refactor`/`perf` that change existing behavior |
| `### Deprecated` | features that will be removed |
| `### Removed` | removed features (often breaking) |
| `### Fixed` | `fix` |
| `### Security` | security fixes |

Entry style: user-facing, past-tense-neutral, one line, link the PR:
```
- Weekly checkout trend chart on the staff dashboard (#14)
```
Breaking changes: prefix with **BREAKING:**.

---

## 7. Semantic Versioning 2.0.0: how versions are decided

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

---

## 8. Definition of Done
A task is done only when **all** are true:
- Code merged to `main` via an approved PR with green CI
- Tests added/updated per §2 Step 4
- `CHANGELOG.md` updated
- Docs/README updated if setup or usage changed
- Demo path still works on `main`
- Issue closed

---

## 9. Hackathon-specific rules
- **Timebox**: if a task blows past 2× its estimate, stop and tell the team. Cut scope before cutting quality of the demo path.
- **Demo freeze** (time in §0): after it, only `fix:` PRs that the team agrees on. Tag `v1.0.0` at freeze.
- **Feature flags over long-lived branches.** Merge half-done work hidden behind a flag rather than letting a branch rot.
- **Demo script** lives in `docs/DEMO.md` (create it): the exact click path + fallback screenshots/video in case live demo breaks.
- Keep a short **"What we built"** section in README updated as features merge. Judges read it.

---

## 10. Project-specific notes: Databricks (delete if not used)
- Use **Databricks Git folders (Repos)** connected to this repo; work on your feature branch inside it, never edit `main` in the workspace.
- Store notebooks in **source format** (`.py` with `# COMMAND ----------` / `# MAGIC %sql`) so diffs and reviews work. Clear outputs before committing `.ipynb`.
- **Always use 3-level names**: `catalog.schema.table`. Agree on one catalog/schema per team (e.g. `hack.team_x`), and per-person dev schemas for experiments (`hack.dev_<name>`).
- **Medallion layout**: `bronze_*` (raw, append-only), `silver_*` (cleaned, typed, deduped), `gold_*` (aggregates for dashboards/ML). Name tables accordingly.
- Never `DROP`/`TRUNCATE`/`OVERWRITE` a shared `silver_`/`gold_` table without telling the team. Delta time travel (`RESTORE TABLE t TO VERSION AS OF n`) is the undo button.
- Avoid `.toPandas()` / `.collect()` on large DataFrames; sample or aggregate first.
- Log ML runs with **MLflow** (`mlflow.autolog()`); put the best run's link in the PR.
- Data quality checks (row counts, nulls, uniqueness) belong in the pipeline/notebook and count as the "test" for data changes (§2 Step 4).
- Never hardcode tokens: use `dbutils.secrets.get(scope, key)` or env vars.
- Shut down/auto-terminate clusters; prefer serverless where available.

---

## 11. Repo layout (update as it grows)
```
.
├── AGENTS.md                 ← this file (all agents + humans)
├── CLAUDE.md                 ← Claude Code entry (imports this)
├── .cursor/rules/            ← Cursor rules (point here)
├── .github/
│   ├── copilot-instructions.md
│   └── pull_request_template.md
├── .claude/commands/         ← /ship, /release, /changelog for Claude Code
├── CHANGELOG.md
├── scripts/
│   ├── next-version.sh       ← computes next semver from commits
│   └── release.sh            ← rolls [Unreleased] into a version
└── TODO: src/, notebooks/, pipelines/, tests/, docs/
```

---

## 12. Learnings & gotchas (append-only)
<!-- Format: - YYYY-MM-DD (@handle): what happened → what to do instead -->
- YYYY-MM-DD (@handle): _example: the shared cluster restarts at 02:00; save work before then._

## 13. Decision log (append-only)
<!-- Format: - YYYY-MM-DD (@handle): DECISION, because REASON. Alternatives: X, Y. -->
- YYYY-MM-DD (@handle): _example: squash-merge only, so `main` history maps 1:1 to changelog lines._
