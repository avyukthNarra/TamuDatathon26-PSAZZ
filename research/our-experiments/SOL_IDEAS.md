# SOL ideas: ranked experiments

Reviewed COMBINED_SOLUTIONS `81d4fff`, MODEL_PROMPTS, zhill `3bdf336` rules/tuner, PR #15, and R0. Gains below are speculative points above matched R0 controls: validation-prior blend 55.578; train-only blend 54.241. Zero or negative gains remain possible. Runtimes estimate one fit plus validation on two CPUs, excluding implementation.

1. **Coupled Bayesian win experts (a).** Short-history hypothesis: agreeing/disagreeing cells should share evidence about direction. In `PairMixer.walk`, use three pairs × two directions, coupled low/high Beta priors, and Bernoulli log-likelihood weights; retain a weak null expert. Compare identical experts under Brier weighting; start alpha=0. Avoid `sym`'s incorrect complementary-rate assumption. Gain 0–0.6; risk medium: likelihood need not minimize Brier. Runtime 40–90s.

2. **Repair train-only win priors.** Short/drift hypothesis: full-history averages blur regimes, explaining R0's .1398/.7563 centers. Replace median estimates in `train_priors` with a three-component Beta-binomial mixture over 64-tick pair-cell counts, ten EM iterations; retain existing size priors. Gain 0–1.0 on train-only control; risk high: overlapping windows and weak identifiability. Runtime 1–3min.

3. **Marginalize missing spins.** Short-history hypothesis: scattered observations still constrain hidden lag states. Add a binary-history forward filter to `predict/walk` (2^max_lag states: 128 currently, 256 with lag eight), summing missing outcomes instead of treating unknown contexts as base-rate forecasts. Freeze trained rates during filtering; keep rule identity machine-specific. Gain 0–0.5; risk medium: misspecified transition probabilities. Runtime 2–8min, restricted to sparse histories.

4. **Shrunk population rule priors (c).** Short-history hypothesis: training frequencies can initialize uncertain rules. Set `pair_prior`/direction priors from episode-cross-fitted soft rule counts, machine-type frequencies shrunk toward global frequencies with Dirichlet strength 20. Gain 0–0.2; risk high: type association is unproven. Do not pool eight machines into one pair/direction: shared changes do not imply shared rules. Runtime 40–70s.

5. **Context-gated size Bayes (e replacement).** Short-history hypothesis: share scarce wins across interpretable size gates. Replace independent `size_preds` cells with six lag-{1,2,4} × direction gates; infer gate weights online and learn type emissions through training-only conditional EM. Gain 0–0.4; risk medium/high: emission/gate ambiguity. Runtime 1–3min. Existing code already performs size Bayes; a persistent size HMM contradicts documented spin-by-spin switching.

6. **Conservative shared run length (b).** Shared/drift hypothesis: simultaneous conditional forecast failures reveal level shifts. In an episode walk, adapt [BOCPD](https://arxiv.org/abs/0710.3742): retain top-64 run-length hypotheses plus a no-change state using joint lag-conditioned likelihoods; softly favor short memory on resets. Gain 0–0.3; risk high: correlated errors cause false alarms. Preserve local rules and every target cutoff. Runtime 1–4min.

7. **Compatible PR #15 port.** Stable/short hypothesis: shared memory evidence helps the stronger mixer too. Add half-life infinity and gamma=.25 pooled speed losses, preserving current `_cell_rate` and `mode_llr`; require gamma=0 parity with both priors active. Gain 0–0.3; risk medium/high: COMBINED already recorded negative pooling tests. The old copied walk omits new priors; PR #15 barely helped shared-change. Runtime 35–70s.

8. **Log pool plus temperature (d).** Short hypothesis: geometric pooling handles disagreement better. In blend prediction use `normalize(exp((.8*log(mixer)+.2*log(rule))/T))`, T={1,1.2}, clipped probabilities; compare linear pooling. Gain 0–0.15; risk high: correlated forecasts can reinforce mistakes, and earlier calibration failed. [Temperature scaling](https://proceedings.mlr.press/v70/guo17a.html) supplies a method, not evidence of gain here. Runtime 35–50s.

9. **Level-switching HMM before pair switching (a extension).** Drift hypothesis: levels change while pair identity persists. Add transition hazards {0,1e-4,1e-3} to level states in #1, accounting for elapsed ticks; reset Beta sufficient statistics on transitions, keeping pairs fixed. Gain 0–0.2; risk high: [Fixed-Share](https://mwarmuth.bitbucket.io/pubs/J39.pdf) already hurt substantially; changing weights alone cannot reset stale counts. Runtime 45–120s.

Screen on tuning; freeze candidates before Claude's audit. Repeated `auto_tune` audit acceptance is not fresh confirmation. Keep prior-source controls separate and bootstrap whole episodes.


## 2026-11-07 update: Sai v4 prior evidence

See [SAI_PRIOR_FINDINGS.md](SAI_PRIOR_FINDINGS.md), pinned to `origin/sai/combo-top10` @ `a373345749af89612d0d8571ff1657dcca949ad9`. Deprioritize direct repeats of population pair priors, type-cell shrinkage, peer pooling, logit calibration, and gated blending given the reported v4 results; preserve the ranked ideas above as historical context.
