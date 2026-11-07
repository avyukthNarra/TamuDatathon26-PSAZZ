# House Edge: MASTER TEST PLAN

Synthesis of everything in `docs/sai/`: `solution-playbook.md`, `research-literature.md`, `RESEARCH.md` (teammate), `house_edge_research_modeling_guide.md` (teammate). Written 2026-11-07.

## Ground rule
Lead = highest-capability model; subagents = cheapest model that can do the job (Haiku/Sonnet). See section 9 for roles, model tiers, reporting, and the rules for shelving a method.

**No method is ruled out until it has been tested properly** (a real grid, episode-grouped validation, bootstrap CI). Priors below are only the order to run things in, not a verdict. "Impact" ratings in the teammate files are guesses. The only measured results so far are in section 1.

## 1. Known results (main round, validation, 96 episodes)
| ID | Model | Points | Source |
|---|---|---|---|
| B0 | pooled (starter, strength 20) | 30.00 | reference |
| B1 | pooled_recent | ~30.6-31 | EXPERIMENT_LOG |
| B2 | GB (HL 4,16,64,128 + LAGRESP) | 32.99 | experiments/gb_proto.py |
| B3 | pair model prototype | 46.07 (R=0.302) | scripts/pair_model_prototype.py |

Measured facts to build on: ~95% of EWMA error is structure not noise; lag 7 and pairs (1,3),(2,5),(4,7) recur; effect signs differ per machine; p0 holds ~58% of B3's remaining error, then p1 24%, p2 14%, p5 4%, p20 ~0%. One observed negative: allowing half-lives of 4-12 hurt B3. Retest in new combinations before treating as dead.

## 2. Shared protocol (every experiment must follow)
1. **Harness first (one agent, before fan-out).** Build `docs/sai/harness/` with: data loader (dense tick grid, NaN for unobserved), the contest scorer (use `slot_forecasting/scoring.py`; per-condition ratios vs frozen reference; points = max/(1+R)), episode-level bootstrap CI, a result writer, and a determinism check. All workers import it; nobody reimplements scoring.
2. **Leakage rules.** Only ticks strictly below cutoff. Missing ticks are unknown, never 0. Lags by exact tick (`grid[t-l]`), never row index. Never use same-cutoff peer outcomes. No condition labels at inference (a classifier trained on labels is fine).
3. **Splits.** Group by episode always. Reserve an **audit set**: 24 validation episodes (6 per condition) that no one tunes on until final comparison. Tune on remaining validation plus train-history pseudo-cutoffs. Choose structure (lags, pairs, features) from histories, hyperparameters by inner holdouts where possible.
4. **Report per run** (JSON to `docs/sai/results/<ID>.json`): id, description, params, points, 4 condition ratios, bootstrap 95% CI, per-class error shares (p0..p20), runtime, seed, stdlib-only yes/no, determinism diff (run twice; must be <1e-8).
5. **Verdict rule.** Mark `alive` (not yet tested or inconclusive), `helps` (CI over B3 or over its blend), `neutral`, `hurts` (only after a real grid). Differences under ~0.2 points are noise. Also test each method **as a stack member**, not just standalone: a method that loses alone can still help a blend.
6. **No shared-file edits by workers.** Each worker writes only its own files and results. A coordinator appends to `EXPERIMENT_LOG.md`. Follow AGENTS.md (branch per workstream, `<name>/<desc>`, no pushes to main).

## 3. Method catalog (all start `alive`)
Reference model for "combine with B3" tests: `scripts/pair_model_prototype.py`.

### A. Frequency and shrinkage
- A1 type-level Dirichlet smoothing, alpha in {2,8,24,64}
- A2 9-cell (machine_type x signal) priors
- A3 half-life grid {4,8,16,24,32,48,64,128,inf}
- A4 class-specific half-life and alpha (frequent classes short, rare classes long)
- A5 inverse-density weights for short_history older ticks (~7x)
- A6 prior strength scaled by effective sample size / observed-tick count
- A7 factorized win-rate x conditional payout severity, blended with direct 5-class
- A8 leave-one-out episode and episode-type shrinkage

### L. Lag structure
- L1 one-hot lags 1-8 + missing flag, regularized multinomial
- L2 per-machine ridge on lags + pairs (this is B3; vary lambda, lag set, pair set)
- L3 all pairs among lags 1-8 with null controls (lags/pairs 9-16); triples
- L4 per-machine single-lag conditional lift (teammate guide) vs product features
- L5 3-state conditioning (loss / small win / big win) and log(1+y) levels vs win/loss flag
- L6 partial pooling of coefficients to the type mean
- L7 per-machine sparse feature selection from the screen
- L8 mixture transition distribution (MTD) lags
- L9 signal x lag and type x lag interactions

### M. Memory and nonstationarity
- M1 holdout-selected half-life (B3 today); vary holdout length, grid, floor
- M2 fixed-share over half-lives; M3 dynamic model averaging
- M4 BOCPD (Dirichlet-multinomial); M5 ADWIN
- M6 JSD/TV-gated blend of short and long windows, sigmoid gate tuned on validation
- M7 CUSUM / rescaled-range drift features
- M8 HMM, 2-3 regimes, fixed init and iteration count (Rabiner)
- M9 changepoints shared across an episode's machines
- M10 half-life implied by Kalman gain as a sanity check

### E. Episode and peer information
- E1 recent residual pooling across peer machines (common shock)
- E2 cosine-similarity subgroup pooling; also pool by type and signal
- E3 episode-level drift stats (mean TV, win-rate change h=16 vs 128) as features
- E4 peer lagged associations vs null controls
- E5 latent factor model for episode state

### S. Learners and training data
- S1 GB / HistGB second stage on B3 output + features (residual learning)
- S2 CatBoost / LightGBM (only if the runner allows; else skip packaging, still test value)
- S3 latent condition classifier + gating of specialist forecasters
- S4 rolling virtual cutoffs (t from ~100 to cutoff-1) to multiply training rows
- S5 soft-target distillation from validation true q (episode-OOF)
- S6 smoothed pseudo-targets from a future window (train only; note drift bias)
- S7 regularized multinomial logistic on the full feature set
- S8 small MLP (low prior; check determinism)

### P. Post-processing and blending
- P1 Euclidean simplex projection vs divide-by-sum vs softmax
- P2 temperature / Dirichlet calibration fit to q, episodes held out
- P3 affine 5x5 calibration per (type, signal)
- P4 convex blend with R-weighted objective: w_q = 1/(|M_e| x N_ep,c x MSE_c,ref)
- P5 stacking over all `helps` members (episode-OOF)
- P6 shrink ensemble toward the type prior; P7 clip bounds

## 4. Feature catalog (toggle in ablations)
exact-tick lags 1-8 (win flag, payout, log(1+y), missing); pair products; multi-half-life smoothed frequencies; ESS (Kish) and observed-tick fractions (recent 40 vs older); window moments (4,8,16,40: mean, log mean, variance); time since last win / last 5x-20x; short-vs-long JSD/TV/L2; CUSUM range; episode synchronized drift; LOO episode means; machine_type, signal, and interactions; B3 prediction and chosen half-life as stage-2 inputs.

## 5. Variation matrix
For each method that passes a first smoke test, run the cross: {with / without signal} x {with / without pairs} x {half-life grids} x {A1/A2 priors} x {with / without A5}. Log every cell. Also test per-condition behavior: a gain in one condition that hurts another is reported, not hidden.

## 6. Parallel workstreams (one agent each, own branch)
- W0 harness (blocks everything; do first)
- W1 A + L (shrinkage and lag structure)
- W2 M (memory, changepoint, HMM)
- W3 E (episode and peer)
- W4 S (learners, training data multiplication)
- W5 P (post-processing, blending; runs on outputs saved by W1-W4)
- W6 packaging: stdlib port of the best pipeline, determinism, `check_submission.py` / `build_entry.py`
Every worker saves its **validation predictions** (CSV, same format as submission) so W5 can stack without rerunning.

## 7. Integration
1. Collect all `helps` results; rank by points and CI.
2. W5 stacks them episode-OOF on validation q (excluding the audit set), then evaluate on the audit set once.
3. W6 ports the winner: numpy/scipy/sklearn may not exist in the runner (guide says Python 3.13 stdlib-only unless wheels are shipped; confirm with organizers). Fixed seeds, sorted groupbys, single-thread BLAS.
4. Run the kit checker locally; keep 30 min / 2 CPU / 16 GiB.

## 8. Round two (data drops Sat 9 PM CT, deadline Sun 11 AM CT)
Nothing hard-coded. Pipeline estimates priors, lag/pair sets, half-lives and blend weights from whatever `public/` it is given. First 15 minutes: rerun `scripts/lag_screen.py`, print per-(type, signal) mean probabilities and lag-1-8 autocorrelations, and score all baselines per condition. Then rerun the surviving methods with their grids; the screen decides which lags and pairs enter.

## 9. Orchestration, model tiers, and compute discipline (read before spawning anything; section 9)

**Roles.**
- **Lead (highest-capability model):** owns the plan, the shared harness design, verdicts, stacking design, final selection, and documentation. Does not run grids itself and does not read raw outputs, only subagent reports and `results/*.json`.
- **Subagents (cheapest model that can do the job):** each takes one method ID (or a small cluster), implements it, runs its grid, writes results, and reports. Independent of each other; no shared-file edits.

**Model tiers** (set the Agent tool's `model` parameter per subagent; start low, escalate on failure):
| Task | Model |
|---|---|
| Run a predefined grid, ablation toggles, collect numbers, format reports, update docs tables | Haiku |
| Implement a method from the catalog (shrinkage, lag features, EWMA variants, blends, calibration, simplex projection) | Sonnet |
| W0 harness, stdlib port + determinism, BOCPD/HMM/DMA, learners with leakage risk (S1, S3, S4) | Sonnet |
| Choosing between methods, designing the stack, interpreting conflicts, round-two decisions, final sign-off | Lead only |
Escalation: if a subagent fails the same step twice or reports an unresolved leakage/determinism problem, it stops and reports. The lead decides whether to rerun the task on a stronger model. Never start every task on the strongest model.

**Subagent report (<= 15 lines, to `docs/sai/reports/<ID>.md`, plus `results/<ID>.json`):** method ID, what was run (grid actually completed), best cell and its params, points with CI, 4 condition ratios, per-class error shares, stack-member gain on episode-OOF, runtime, determinism diff, stdlib-only yes/no, anything surprising, suggested follow-ups. Facts only; no verdict. The verdict is the lead's.

**Documentation flow.** Subagents write only their own `reports/` and `results/` files. The lead (or a Haiku docs agent under lead instruction) merges them into `docs/sai/RESULTS.md` (leaderboard + status table: every method ID, state, evidence link) and appends to `EXPERIMENT_LOG.md`. Keep this file's section 3 states current: `alive` / `helps` / `neutral` / `shelved`.

**When a method is truly not worth more compute (the lead's call, but only by these rules, so "nothing ruled out until tested" still holds):**
1. *Smoke test first* on a small slice (about 24 validation episodes, 6 per condition). Crashes, leakage and bugs are fixed or escalated, never counted as a negative result.
2. *Pre-declare the grid* before running. Early-stop a grid if the metric gets worse monotonically for 3 consecutive steps beyond the edge in that direction.
3. A method becomes **`shelved`** only if all hold after the declared grid completes: (a) best cell's CI upper bound is below B3 standalone; (b) added as a blend/stack member on episode-OOF it gains under about 0.1 points; (c) it gives no condition a CI-clear gain; (d) at least one variation from the section 5 cross has been tried (e.g., with/without signal, with/without pairs).
4. `shelved` methods keep their report and are reopened if: round-two data changes the structure, a new feature or a fixed bug could plausibly change (a)-(c), or the lead sees a combination argument. Record the reason.
5. A method that **helps only in one condition** or **only as a stack member** is `helps`, not dead.
6. Stop spending on a method when expected gain is below noise (about 0.2 points) relative to compute: a grid cell costing more than a few minutes of wall time must be justified by a pre-declared hypothesis.

**Compute discipline.**
- Check `nproc` first; cap concurrent subagent runs at the core count minus one (the final entry is limited to 2 CPUs / 30 min, so also profile the winner single-threaded).
- Cache dense grids and feature matrices to disk in the harness; subagents read the cache instead of rebuilding.
- No duplicated work: one subagent per method ID; the lead assigns IDs and tracks them in `RESULTS.md`.
- Sequence: W0 harness -> fan out W1-W4 in parallel (Haiku/Sonnet) -> W5 stacking on saved predictions -> W6 packaging. The lead reviews at each gate (after W0, after fan-out reports, before packaging).
- Wall-clock budget: main round closes Saturday 9 PM CT. Reserve the last hour for packaging and the checker; reserve time after round two drops for the section 8 routine.

### Laptop resource guardrails (every agent follows these)
1. **Cap parallelism.** At most `min(4, cores-1)` experiment processes at once (macOS: `sysctl -n hw.ncpu`). Leave a core free so the laptop stays usable.
2. **One thread per process.** Export `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1` before any run. Otherwise numpy fans out on every core and parallel workers fight each other. This also matches the 2-CPU grader and helps determinism.
3. **Hard time limits.** Wrap every run: `nice -n 10 timeout 600 python3 ...` (smoke tests: 120 s). A run that hits the limit is reported as "timed out", then the lead decides. Cheap methods (B3-style ridge, about 2 s per split) should never need more than a minute for a whole grid.
4. **Smoke test before grid.** Run on the small slice first; only launch the full grid if it is correct and fast. Extrapolate runtime from the smoke test, and report the estimate before launching anything over 10 minutes.
5. **Cache once.** Build the dense tick grids and feature matrices a single time into `docs/sai/cache/` (gitignored); everyone reads from it. No worker rebuilds features.
6. **Heavy methods get an explicit budget.** HMM, BOCPD, DMA, GB, CatBoost, MLP, rolling-cutoff data multiplication (S4, which can reach hundreds of thousands of rows): lead approves a time and memory cap before they start; run them one at a time, not alongside each other.
7. **No runaway output.** Results are small JSON/CSV; no per-row logging, no giant intermediates committed. Delete caches you no longer need.
8. **Power and cleanup.** Plug the laptop in for long runs. When a batch ends, no background processes should remain: check `ps` / Activity Monitor, and stop with `pkill -f docs/sai` if anything is left.
9. **Stop rule.** If the laptop is swapping or fans are pegged, drop concurrency by half and tell the lead.

### Splitting work across Claude Code and Codex (or any other agent)
The repo's `AGENTS.md` is the shared rulebook (Claude Code reaches it through `CLAUDE.md`; Codex reads `AGENTS.md` directly), so both follow the same playbook.
1. **Split by workstream or method ID, never by file.** Example: Claude takes W1+W2, Codex takes W3+W4. The lead records who owns each ID in `RESULTS.md` before anyone starts.
2. **Separate working trees.** Each agent/tool uses its own branch and its own `git worktree` (e.g. `../house-edge-codex`). Two tools editing one checkout will clobber each other.
3. **Same contract.** Both use the W0 harness, the same results JSON schema and the same report format (section 9). The source tool does not matter; the numbers must be comparable.
4. **One shared resource budget.** The laptop guardrails apply to the total across all tools (core cap, one thread per process, timeouts), not per tool.
5. **Cross-check once.** Re-run each tool's best result through the harness on the audit set, with the lead's scoring, before it enters a stack. Different tools write different code, so verify the determinism diff (<1e-8) rather than trusting self-reported scores.
6. **Model tiers.** Use each tool's own cheaper tier for grid-running and its stronger tier for hard implementation. The lead stays the single decision-maker for verdicts.

## 10. Open questions for organizers
Point split 80/20 vs 60/40 (RESEARCH.md says 80/20); stdlib/offline wheels vs pinned `requirements.txt`.
