# House Edge — master research summary (as of 2026-11-07, 7 PM CST)

This branch (`avyukth/master-research`) gathers in one place everything the team learned about the
House Edge forecasting task. Start here, then follow the links. Scores are points out of 60 for the
main round unless stated (score = 60 / (1 + R); R = mean over the four conditions of our error divided
by the frozen pooled-starter error).

## 1. Where we stand

| Entry | Validation (all 96 episodes) | Kit tuning / audit (48 / 48) | Independent check |
|---|---:|---:|---|
| **20% Zhill rule model + 80% Pranav pair mixer v4** (team best, `zhill-55.58`) | **55.58** | 55.26 / 55.96 | **55.57** [54.90, 56.17] in our eval suite's 4-fold protocol (never scored on episodes whose answers it saw) |
| Pranav pair mixer v4 alone (`COMBINED_SOLUTIONS@81d4fff`) | 55.34 | 55.03 / 55.70 | — |
| Sai COMBO 1–10 (≈20–26% rule + 74–76% v4, ± ≤8% C1/C2) | dev 55.31–55.36 (Sai split) | audit 56.21–56.31 | all inside each other's noise |
| Same 20/80 blend, **train-only priors** (no validation answers) | 54.24 | 54.19 / 54.31 | R2 EM prior lifts it to 54.36 |
| Our own best before teammates' work (E6+E7 blend) | — | 50.65 / 50.02 | blending it into the team model hurts |

Half-split spread of the team best is about ±0.4 points, so **differences under ~0.4 between two methods
are noise** unless a paired CI says otherwise.

Per-condition error ratios of the team best (lower is better): stable .059, **short_history .120**,
drift .066, shared_change .073. Short history is the weakest spot; jackpot machines are the weakest
type (error/reference ≈ .13–.15, mostly win-probability error).

## 2. How the task really works (what the data shows)

- Each spin a machine is effectively **live (~0.9 win) or dead (~0.05 win)**, set by XOR/XNOR of its own
  win/lose at lag pairs; on main data the recurring pairs are (1,3), (2,5), (4,7). The direction
  (XOR vs XNOR) is per machine. Two machines of the same type can respond oppositely.
- **Payout size given a win** switches between two modes per machine type, driven by big wins at
  lags 1, 2, 4. Modes switch spin by spin with lagged outcomes, not as a slow persistent state.
- **Drift** = relations gradually reverse; **shared change** = all 8 machines of an episode change at
  one tick. Episode peers share *changepoints*, not rules (Sai's train check).
- **Short history** = the last 40 consecutive ticks + 24 sampled older ticks (64 observed). On the kit
  the final 8-tick window is never missing (R3), so gaps only affect older ticks.
- **`signal` (-1/0/1)** shifts a machine's overall win level (train win rate rises ~0.12 from -1 to +1,
  strongest on jackpot) but not direction, size or drift. No current model uses it; adding it as a
  prior gave only +0.03 (the mixer already learns the level). Script: `data-findings/signal_analysis.py`.
- The validation `targets.csv` holds the simulator's **true probabilities**; dead/live levels read from
  it are .0435 / .898. Using them is allowed (RULES: learn from training and validation data) and does
  not overfit episodes (Sai cross-fit: 55.35 vs 55.34 in-sample).
- Round two (adaptation, kit opens Sat 9 PM, due Sun 11 AM) "changes which temporal relationships
  recur" and ships fresh train + validation answers: anything that **re-discovers lag pairs and refits
  priors from the new data** carries over; hard-coded pairs do not. Pranav's mixer re-discovers pairs
  on train (fallback pairs only if discovery finds nothing).

## 3. The team's best model, in one paragraph

Pranav's **pair mixer v4** (`team/PRANAV_MODEL_PROMPTS.md`): for each machine, online exponential-weight
experts over lag-pair parity cells (pairs rediscovered on train, half-lives 12 and 40, parity vs full
4-cell mode picked on train), each cell's win rate a posterior mean under a **Beta-mixture dead/live
prior** (round 3, the big short-history fix .149 → .121), a **two-mode payout-size** model keyed on lags
1–4, outputs floored at 1e-4. **Zhill's rule model** (`team/ZHILL_MODEL_PLAN.md`) is blended at 20%;
`auto_tune.py` re-picks the weight per round but keeps 0.2. Sai's 660-structure combination search
(`team/SAI_COMBOS_TOP10.md`) confirms nothing else adds to this pair.

## 4. What we tried on top of it (all on the v4 base, tuning-only, paired CIs)

| ID | Idea | Result | Verdict |
|---|---|---|---|
| R0 | Reproduce best + train-only priors switch | exact reproduction; train-only costs ~1.3–1.5 | baseline (`our-experiments/R0_REPORT.md`) |
| R1 | Exact Bayesian experts over rule space (Brier or log-likelihood weights) | −6.6 to −12.4, collapses on drift/shared | **rejected** |
| R2 | EM Beta-binomial train-only priors (32/64-tick windows) | +0.17 [+0.03, +0.31] train-only blend; short −4.3% but jackpot +7.4% | keep as **safe fallback** |
| R3 | Forward filter for missing lag windows | 0/96 queries eligible | no-op |
| R8 | Log-linear pool / temperature on the blend | −0.01 best; temperatures hurt | no |
| M4 | Jackpot-only train prior (Sai's lead) | +0.02 [−0.13, +0.19] | noise |
| SIG | Signal-conditioned win prior | +0.03 max, CIs span 0 | noise |
| Sai/Codex | train pair prior, type shrinkage, peer pooling, logit recalibration, gated blends, adding our models | ≈0 on v4 | no (see `our-experiments/SAI_PRIOR_FINDINGS.md`) |
| Earlier (pre-v4) | E1–E11 (pair logistic, adaptive memory, episode memory, signed-L1, hypernetwork, trees, flex experts) | best 54.18 tuning on old base | superseded; see handouts |

Conclusion: **small tweaks to mixer v4 are saturated.** Remaining headroom is in short-history and
jackpot win probability. Untested ideas worth trying if time allows: neighbour-machine lag experts
(DATA.md says earlier outcomes on neighbouring machines can help; never tested on v4), a per-type
train-only prior (R2 for steady/balanced + M4 for jackpot) for the safe fallback, and an
episode-shared changepoint model (peers share changepoints).

## 5. The evaluation suite (PR #19, branch `avyukth/eval-suite`)

A black-box, method-agnostic suite built only from the kit docs. Point it at any entry folder:
`python -m eval_suite run --entry DIR --suite quick|full --mode docker|local [--compare DIR2]`.
It runs `reproduce.json` like the organizers (Docker python:3.13, no network, 2 CPUs, 16 GiB,
read-only mounts, 30 min), checks all **88 documented requirements** (77 checks: format, package,
determinism, causality/leak probes, data schema), scores with the official metric (equal to the kit
scorer within 1e-9) using a 4-fold validation protocol, and runs **7 spec-derived synthetic presets**
(main_like, round2_shift, all_lags, neighbour_heavy, hidden_state_heavy, fast_drift, edge_cases) with
exact truth. 158 tests pass. Docs: `eval-suite/README.md`, `DESIGN.md`, `REQUIREMENTS.md` (including
17 doc-vs-kit conflicts and how they were resolved). Known gap: main_like lag autocorrelation differs
from real train by up to .13.

## 6. Round-two plan (kit opens 9 PM)

1. Run the team best unchanged on the new kit (`make_entry.py` on `zhill-55.58`): it rediscovers pairs
   and reads the new validation targets. Check the discovered pairs differ from (1,3),(2,5),(4,7).
2. Score with the eval suite (`--suite quick --mode docker`, `--kit` pointing at the new public data)
   and compare to the round-one numbers; if pair discovery fails, fall back to `choose_pairs` with a
   lower `min_share` rather than the hard-coded defaults.
3. Re-tune only the rule weight (auto_tune) and leave everything else; upload early, then experiment.
4. Fallback if validation answers are unusable: train-only priors (R2 W32).

## 7. File map

- `../HOUSE_EDGE_EXPERIMENT_RESULTS.md`: our E1–E11 experiments in depth (pre-v4).
- `../HOUSE_EDGE_BRAINSTORMING.md`, `../HOUSE_EDGE_ADDITIONAL_RESEARCH.md`: early ideas and 20-paper research notes.
- `../handouts/`: build-on handouts for our E2, E6, E7, E11 work.
- `our-experiments/`: coordination plan and every R/M/SIG report, Codex's ideas list (`SOL_IDEAS.md`),
  the full experiment log (`RESULTS.md`), generator report and handoff.
- `team/`: teammates' own write-ups (Pranav model prompts, Sai combination search and improvement
  attempts, team comparison, Zhill model plan) copied from their branches for reference.
- `eval-suite/`: the evaluation suite's design, requirements matrix and usage.
- `data-findings/`: the signal analysis script.

Branches with code: `avyukth/exp-r0-baseline` (harness + safe priors), `avyukth/exp-r2-em-priors`,
`avyukth/exp-r1-bayes-experts`, `avyukth/exp-r3-missing-filter`, `avyukth/exp-r8-logpool`,
`avyukth/exp-m4-jackpot-prior`, `avyukth/exp-sig-prior`, `avyukth/eval-suite`; team best on `zhill-55.58`.
