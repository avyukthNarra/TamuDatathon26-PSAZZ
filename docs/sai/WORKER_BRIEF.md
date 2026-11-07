# Worker brief (Claude subagents and Codex): read this, then your method IDs in MASTER_TEST_PLAN.md section 3

Repo `/Users/saisreed/Desktop/Coding/TamuDatathon26-PSAZZ`, branch is already checked out for you; **never run
git commit/push/checkout/stash**, the lead commits. Work from `docs/sai/`. Python 3.13, numpy/pandas/scipy/sklearn present.

## Every run, always
```
cd docs/sai
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
nice -n 10 timeout 600 python3 methods/<ID>.py        # smoke runs: timeout 120
```
- **One experiment process at a time per agent** (the laptop budget is shared with another agent team).
- Smoke test every method on `smoke_split(v)` (24 dev episodes) first; fix crashes/leaks, never count them as results.
  Write your grid down in the method file's docstring BEFORE running the full grid. Any single run >10 min: stop and
  report the estimate instead. Heavy methods (HMM/BOCPD/DMA/GB) get a cap of 10 min per cell; report timeouts as timed out.
- Early-stop a grid if the metric worsens monotonically for 3 steps beyond the edge.

## Harness (`from harness import ...`; do not reimplement scoring)
- `v = load_split("validation")`, `t = load_split("train")`. Per-query arrays aligned to queries.csv: `grid[i,tick]`
  (NaN = unobserved, NEVER 0; only ticks < cutoff), `win` (1/0/NaN), `mtype`, `signal`, `cutoff`, `episode`, `ep_idx`,
  `peers(i)` (same-episode machines; their ticks are also < cutoff), `seq(i)` ({tick: payout}), `frame()` (DataFrame).
  validation/train also have `condition` (training/analysis only, **never as a feature at inference**), `outcome`;
  validation also `truth` (true q, n x 5) and `ref`. Payout order is 0,1,2,5,20.
- `run_method(ID, owner, description, params, predict_fn, v)` where `predict_fn(split) -> (n,5)` probabilities. It times
  it, runs it twice (determinism must be <1e-8), scores scope `dev` (72 episodes), writes `results/<ID>.json` and
  `preds/<ID>_validation.csv`. Save only your BEST cell under the plain ID; write every grid cell to
  `grids/<ID>.json` (list of {params, points, condition_ratios}); only your own files.
- Never score scope `audit` or `all` (that set is reserved). Tune on `dev`. `evaluate(P, v, "dev")` for extra checks.
- Is it a stack member? `blend_gain(P_method, P_B3, v)` (episode-OOF convex blend; `P_B3 = read_pred_csv("preds/B3_validation.csv", v)`)
  returns points and `vs_base` {diff, ci95}. Also `diff_ci(P_method, P_B3, v, "dev")` for standalone vs B3.
- Heavy reusable pieces: `harness.cached(key, builder)` for features. Train-history pseudo-cutoffs: use `t.grid` rows
  (each machine's history < its cutoff) and make rows at tick s from ticks < s only.
- B3 reference code: `scripts/pair_model_prototype.py` (import via `harness.baselines`). `scripts/lag_screen.py` screens lags.

## Rules (leakage and honesty)
Only ticks < cutoff. Missing = unknown. Lags by exact tick (`grid[i, cutoff-l]`), never row index. Never use peer
outcomes at the cutoff tick. No condition labels at inference. Group by episode. Nothing hard-coded: anything
learned (lags, pairs, half-lives, priors) is estimated from the data you load (round two changes it).
Differences < ~0.2 points are noise; judge by CI. Report negatives faithfully with the grid actually completed.

## Deliverables per method ID (facts only; the lead decides verdicts)
`methods/<ID>.py` (importable, has `predict(split)` and a `__main__` that runs smoke then grid),
`results/<ID>.json`, `grids/<ID>.json`, `preds/<ID>_validation.csv`, and **`reports/<ID>.md` of at most 15 lines**:
method ID; what was run (grid actually completed); best cell + params; points with CI; the 4 condition ratios;
per-class error shares; `vs_base` / blend_gain vs B3 with CI; runtime; determinism diff; stdlib-only yes/no
(numpy-only counts as no); anything surprising; suggested follow-ups.
Escalation: if you fail the same step twice, or hit an unresolved leakage/determinism problem, stop and report.
