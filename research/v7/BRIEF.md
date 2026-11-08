# Shared brief for Grok researchers (House Edge, TAMU Datathon 2026)

## The task
For each query (one machine at a cutoff tick), predict the next spin's payout distribution over {0,1,2,5,20}.
- Score: 60 / (1 + R). R is the mean, over 4 conditions (stable, short_history, drift, shared_change), of our error divided by the frozen pooled-starter error.
- Error is a distribution error against the TRUE simulator probabilities (validation/targets.csv). See `slot_forecasting/scoring.py`.
- An organizer reruns the entry in Docker: python:3.13-slim, standard library only (no numpy), no network, 2 CPUs, 30-minute limit. The output must be deterministic within 1e-8.

## Data (read-only)
`/Users/avynarra/Documents/Datathon/worktrees/pranav-5629/mainFiles/`
- `public/{train,validation,test}/`, with docs `DATA.md`, `RULES.md` and `README.md`.
- **Train:** 96 episodes × 8 machines, about 200 ticks each. Gives queries, history, labels (one sampled outcome per query) and conditions.
- **Validation:** the same layout plus `targets.csv` (true probabilities) and `reference.csv` (starter forecasts used for the ratio).
- **Test:** 192 episodes, which we submit.
- **Usable training signal:** the histories themselves, where every observed tick is a label for the prediction made just before it.

## Current best, the bar to beat: Pranav's pair mixer
Code: `slot_forecasting/pair_mixer.py`, runner `run_pair_mixer.py`. Validation outputs are in `runs/combined-v6/`.
- 56.29 / 60 overall.
- Loss ratios: stable .054, short_history .105, drift .049, shared_change .056.
- Raw errors: .0106 / .0186 / .0093 / .0116.

How it works:
- Online exponential-weight experts over lag-pair parity cells (each spin is "match" or "differ" at the pair). Pairs are rediscovered on train; on main they are (1,3), (2,5), (4,7).
- Half-lives 12, 40 and 100.
- A per-type Beta-mixture dead/live win prior, read from the validation targets:
  - steady: dead .149, live .956
  - balanced: dead .039, live .840
  - jackpot: dead .013, live .669
- A two-mode payout-size model keyed on big wins at lags 1, 2 and 4.

## What we already know (team findings)
- Each spin a machine is effectively live (high win probability) or dead (low), set by XOR/XNOR of its own win/lose at a lag pair. The direction is per machine.
- Payout size given a win switches between two modes, driven by big wins at lags 1, 2 and 4.
- Drift: relations gradually reverse.
- Shared change: all 8 machines of an episode change at one tick. Peers share changepoints, not rules.
- Short history: the last 40 consecutive ticks plus 24 sampled older ticks. Missing ticks are unknown.
- Weak spots: short history (ratio .105) and jackpot type (about 2× the other types' error, mostly win-probability error).
- Failed before:
  - Exact Bayes over rule space with a static rule: −6 to −12 points, because it collapses on drift and shared change.
  - Fixed-share weights: hurt.
  - Logistic or tree models: worse.
  - Blending more models: about 0.
  - Small prior tweaks: saturated.
- Noise level: about ±0.4 points between halves. Use paired comparisons per episode.
- The kit spec generator (simulator guess) is at `/Users/avynarra/Documents/Datathon/worktrees/eval-main/mainFiles/eval_suite/generator.py`. It gives exact truth for synthetic tests, but it does not match real data perfectly.

## Rules for you
- Do NOT modify anything under `worktrees/`. Write ONLY inside your own output folder (given in your prompt).
- **Honest evaluation:** fit or tune on train (and on its histories). Report validation scores without tuning hyperparameters on the same validation episodes. If you tune on validation, use cross-fitting (even/odd episodes) and say so.
- Scoring helper: `python -c` with `slot_forecasting.scoring.score_predictions` or `slot_forecasting.contest.score_contest_files`, with `PYTHONPATH` set to `mainFiles`. Compare against `runs/combined-v6/validation_predictions.csv`, per condition and per type.
- **Your final output:** `REPORT.md` in your folder, concrete and with numbers. Include what you ran, results, what works and what doesn't, and a recommendation. Keep any code you wrote in the folder.
