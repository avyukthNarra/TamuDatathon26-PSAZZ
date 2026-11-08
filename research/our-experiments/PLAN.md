# House Edge: shared experiment plan (Claude + Codex)

Main round closes **9:00 PM CT today (Nov 7)**. Round 2 (adaptation, 40 pts) opens at 9 PM and closes at 11 AM CT Nov 8.
This folder is the shared coordination area. It is git-excluded. Append your results to `RESULTS.md`, one row per run.

## Established facts (validation, measured)

- About 94% of the starter's error comes from the **win chance** p0 = P(payout = 0), including its knock-on into the other four probabilities. The positive-payout shape is close to fixed per machine type.
- The true p0 varies hugely (var 0.154) compared with the starter's predictions (var 0.031). Pooled lags explain only 20% of it. The effect is **machine-specific**.
- A per-machine logistic regression for win/loss on its own history, with lag-1..8 win indicators plus all lag pairs and C=3, weighted by a recency half-life of 32, gives p0 sq err **0.079 vs 0.130 for the starter**. The no-recency pair version scores 0.028 on stable machines but gets worse on drift and shared change.
- See `diag.py` and `diag2.py`. They were tuned on all of validation, so the numbers are optimistic.

## Evaluation protocol (both agents, no exceptions)

- Use `mainFiles/slot_forecasting/experiments.py` `run_experiments` / `select_model` and its fixed tuning/audit episode split. Models implement `fit(queries, history, labels)` and `predict(queries, history)`, returning a dict from query_id to 5 probabilities in payout order 0, 1, 2, 5, 20.
- **Select on tuning only.** Score the audit half only for a final candidate, and log every audit look in RESULTS.md.
- Fit global things (pair selection, priors, gates) on **train** only. Don't pool test histories into training. Fitting a test machine on its own pre-cutoff history at predict time is fine.
- Only use inputs strictly before the cutoff. Missing ticks are unknown. Lags are tick lags, not row offsets.
- The bar to beat is `pooled_recent` at 31.06/60 on tuning and 30.56 on audit.

## Ownership

| Agent | Worktree / branch | Owns |
|---|---|---|
| Claude (Opus, with Haiku subagents) | `worktrees/claude-exp`, `avyukth/win-head` | E1 forecaster module (per-machine pair-logistic + recency, wired into the kit interface); E5 payout sizes as inputs plus the shape head; **final integration, submission CSV, entry ZIP** |
| Codex (GPT-6.1-Sol, with cheap subagents) | `worktrees/codex-exp`, `avyukth/codex-exp` (note: `avy/...` is impossible because branch `avy` exists) | E2 adaptive memory per machine; E3 learning which lag pairs matter from train machines; E4 shared change time across an episode; E6 big hyperparameter sweeps; E7 optional neural in-context model |

## One branch per experiment (user requirement)

- Name: `avyukth/exp-<id>-<short-desc>`, for example `avyukth/exp-e1-pair-logistic`, `avyukth/exp-e2-adaptive-memory`, `avyukth/exp-e3-pair-selection`, `avyukth/exp-e4-shared-change`, `avyukth/exp-e5-shape-head`, `avyukth/exp-e7-neural`.
- E1 branches from `origin/main`. **E2–E7 branch from the pushed E1 branch**, so every comparison shares the same base model and only the experiment's change differs. Until E1 is pushed, start from `origin/main` and rebase onto E1 after it's pushed. Rebasing your own unpushed branch is fine.
- Each agent keeps one worktree and switches branches inside it. Commit before switching. Don't use `git stash` (the stash is shared between worktrees).
- Every RESULTS.md row must include the **branch and commit SHA** that produced it. Push each experiment branch; that's pre-approved.
- Combinations, such as E2+E3, get their own branch, `avyukth/exp-combo-<desc>`.

## Integration contract

- Codex's improvements must be importable components in `mainFiles/slot_forecasting/`, as new files only. Don't edit `experiments.py` or the starter files. Claude merges the winners.
- Checkpoints, all CT: **5:30 PM** Codex posts its best tuning results. **7:30 PM** the final candidate is frozen. **8:15 PM** files are ready for the user to upload. Submitting the Google Form is the user's job.
- Never push to `main`, merge, force-push or delete branches without the user's OK.
