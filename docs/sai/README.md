# House Edge docs (Claude research + solution)

Written 2026-11-07 for Sai. Nothing here has been committed; no git commands were run. Follow AGENTS.md (branch, PR) before committing.

## TL;DR
- Most of the error in simple models is hidden structure, not sampling noise.
- Per machine, lag 7 and lag pairs (1,3), (2,5), (4,7) predict wins/losses. A ridge model on them scores **46.07** validation points versus 30.00 (pooled reference) and 32.99 (GB).
- The portable asset is the screening procedure, since round two changes which relationships recur.

## Files
- `MASTER_TEST_PLAN.md`: **start here for testing.** Synthesizes all files below into one catalog of methods/features/variations, shared protocol, parallel workstreams, and integration steps. Nothing is ruled out until tested.
- `solution-playbook.md`: results, stats, ranked experiments, round-two plan, pitfalls, packaging.
- `research-literature.md`: theory and papers, each claim tagged [Lit]/[Math]/[Measured]/[Idea].
- `RESEARCH.md`: a separate, teammate-supplied methodology (unmodified). Literature-backed plan with experiment matrix E0-E8; no measured results. Test each of its experiments independently of the pair model, and log results in `EXPERIMENT_LOG.md`. Notes: it uses the `signal` column and peer-machine features (both untested in the pair model); its one-hot lag features need hand-added pair products to see the structure in `solution-playbook.md`; it says the runner is Python 3.13 stdlib-only, so numpy-based code needs a port or vendored wheels; its citations 3 (ADWIN DOI), 5 and 8 are unverified.
- `house_edge_research_modeling_guide.md`: a second teammate-supplied methodology (unmodified). Two-stage plan: multi-half-life ensemble gated by a latent condition classifier, episode residual pooling, per-machine lag lift features, factorized win-rate x severity, simplex projection, rolling virtual cutoffs. Its "Expected Impact" ratings are unmeasured guesses. Test each idea independently and log results. Notes: its per-machine single-lag lift will find lag 7 but not the pair interactions (1,3),(2,5),(4,7); it uses scipy (SLSQP) and numpy, which may not exist in the runner; its inverse-density weighting for short_history older ticks and rolling-cutoff data multiplication are untested here and worth trying.
- `scripts/pair_model_prototype.py`: the 46.07 model; also writes test CSVs.
- `scripts/lag_screen.py`: lag/pair screening with null controls.

## For Claude Code
Start with the playbook sections 4-6. Run scripts from `mainFiles/` (paths in the playbook). Log experiments in `EXPERIMENT_LOG.md`.

## Caveats
Half-life floor was chosen after seeing validation; 96 validation episodes is small. Confirm the open rule questions in playbook section 8.
