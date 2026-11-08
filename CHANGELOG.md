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
- Sai's AI-usage report (`docs/devpost/SAI_AI_REPORT.md`) with its stats script (`docs/devpost/tools/ai_stats.py`)
- `submissions/` with the exact files submitted in both rounds (round one 57.97/60, round two 38.59/40) and their hashes; Devpost materials and every round-two research report under `docs/devpost/`; team commit log of all branches (`docs/devpost/research/COMMIT_LOG.md`) (#31)
- Round-two adapted blend: every member learns its lag pairs from the round's train data (`adapted_entry/`, `adapted_entry_v17g/`, main 57.97) (#22)
- Exact filter also learns 3-lag XOR and 2-lag AND/OR rules from each round's train census (`adapted_entry_v17g_rules/`; main unchanged, simulated mixed-rule kit +3.3) (#24)
- One-command round-two build and ranking (`r2/run_round2.py`): structure screen, all candidates, paired comparison, `RANKING.md` (#23)
- Code ZIP bundles the round's public validation folder so the graded rerun is byte-identical with or without validation answers mounted (#27)
- Top-level README with what we built, round-two steps, setup and layout
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
- Round-two script defaults to the rules-learning adapted blend (`adapted_entry_v17g_rules/`, #24); the previous default stays as challenger `blend_v17g` (main kit unchanged, 57.97)
- Round-two pipeline and final model merged into `main` for the whole team (#30)
- Model outputs go to `mainFiles/runs/<name>/`, which is git-ignored, so prediction files are no longer committed; run docs updated to match (#7)
- Playbook requires agents to delete branches with merged PRs (local and origin) after each merge (#6)
- Playbook spells out PR flow: no pushes to `main`, one approval (own is fine), merge only when approved with no conflicts
- Agents no longer flag unfilled project snapshot `TODO`s before kickoff; whoever adds a piece of the stack fills its row

### Deprecated

### Removed

### Fixed
- Blend entries no longer crash when `/data/validation/targets.csv` is absent (#25)
- Entry no longer crashes when the kit has no `validation/` folder at all (organizer-style mount of train/ and test/)
- `blend_v13` challenger builds again (syntax error left by #25) (#28)
- Local-mode answer-isolation checks are reported as NOT RUN, and omitted mode selects Docker when available (#19)

### Security

## Team research history (all branches)

How the model was built, by person, condensed from 249 commits across every branch (full list:
`docs/devpost/research/COMMIT_LOG.md`; Sai's narrative: `docs/devpost/SAI_ITERATIONS.md`). Scores are main-round validation /60
unless marked /40.

### Pranav (`pranav/*`, 51 commits)
- Sat: lag-feature forecaster, then the lag-pair mixer (53.35 → 53.75), size experts keyed on earlier payouts (54.21),
  two-mode payout size switch (54.90), Beta-mixture dead/live win prior (55.34), per-type win prior and learned size lags (56.12).
- Sat night: 100-spin win memory (56.19), per-type and per-signal win levels (56.32), hidden-rule HMM win blend (56.51),
  log-odds pooled ensemble (56.54), peer correction from residual analysis (56.59), v11 plateau average (56.61).
- Sun: v13 strongest build (our win model + size-rule HMM + v7c, 57.52) → v15 (57.55) → v16 episode flip swap (57.59) →
  v17 jackpot sub-cell correction (57.72) → v19 build-time gates (57.74) → v20 super entry with Sai's blend + v19 (58.00);
  round two: v21 peer term on the blend (38.67/40, below the pick margin).

### Zhill (`zhill*`, 18 commits)
- Sat: per-machine lag-pair rule model (52.89), blended with the team pair mixer (54.48), rule weight 0.2 after an independent check,
  auto-tune of the blend per round, pair mixer v3 in the combined entry (55.58), hidden-rule HMM 50/50 (56.28), mixer6 + rule + HMM (56.59).
- Sun: size-rule HMM for payout size (57.32), size-rule prior learned per round by EM, v17-round2-guards (v17 with v7c guards,
  the final blend's first member), v18 backup entry (v17 guards + exact filter, 57.95), round-adaptive member gate (`r2/gate_members.py`).

### Avyukth (`avyukth/*`, `avy`, 38 commits)
- Sat: research and brainstorming notes (20 papers), experiment summaries and handoffs (E1-E7, E6+E7 blend entry), master research branch.
- Sun night: v7c (size-atom rule, episode flip swap, round-two guards) and its fix for shifted rounds; adapted Sai's blend so every
  member learns lag pairs per round (57.96), swapped member 1 to v17-guards, exact filter learns 3-lag XOR and AND/OR rules,
  10-kit adaptability benchmark, portable `run_round2.py`.
- Round two: adaptation gate (38.59/40), blend-tweak, per-type, headroom, calibration, peer lag-3 and Docker checks; built and
  uploaded the round-two files; Devpost write-up, submissions record and research reports.

### Sai (`sai/*`, 142 commits)
- Sat: team playbook (`AGENTS.md`), shared experiment harness with a once-only audit split, four worker tracks (frequency, lag
  structure, memory/changepoints, peers/residual learners): B3 → B4 near-OLS lag 7 + pairs (46.8 → 52.3 audit), severity size
  model, peer-gated changepoints, W5 stack (54.1 audit); re-scored teammates' models and switched to the better team model;
  blending searches showed saturation (held-out gain ~+0.007); swap-in-truth showed win probability was the lever.
- Sun night: fresh-start win models: exact Bayesian filter, sysid rule library, short-history 256-state patch, jackpot four-cell patch;
  equal log-odds blend 57.97 (+0.47 [+0.21, +0.77] over v13); exact filter learns its pair set from the kit; round-two structure screen;
  one-command `r2/run_round2.py` with paired bootstraps and the > 0.1 both-halves pick rule.
- Round two: ran the pipeline on the real kit (38.59/40), orchestrated Codex workers and Avyukth's bots on short-history ideas
  (all below the bar), diagnosed short history's size gap and the small ceilings, cross-checked the upload (3.2e-14).
