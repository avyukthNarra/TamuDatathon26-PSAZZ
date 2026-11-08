# Avyukth source review (facts; no overall verdict)
All 17 origin/avy and origin/avyukth refs cataloged with source commits in ../results/TEAM_AVY_REVIEW.json.
Final E6 uses signed all-pair L1 logistic C=.5 half-life16; E67 blends its full distribution 50/50 with loaded E7.
E7 uses TRAIN learned causal context; reuse cached predictions/artifact without retraining.
E2 exposes prequential predictions/losses and sample weights for hard/soft per-machine memory selection.
Combo-E2 independently selects ET/LR memories; inverse temperature128, last24 valid rows stride4.
E4 episode-memory pools causal per-row Brier; its selector always remains default eight-lag LR.
E4 shared-change exposes penalized marginal-rate split evidence, per-machine direction, boundary and support.
Marginal-rate directions do not identify balanced parity coefficient sign flips; conditional evidence is required.
E11 adds LR/ET/HGB experts and optional lag imputation=.5; its memory selector still evaluates LR only.
Derive regret, episode agreement, entropy, disagreement and ESS; no calibrated uncertainty gate exists.
No explicit severity-lag model exists here; E5 uses positive counts and optional >=2 WIN covariates.
Fit TRAIN inputs/labels only; predict VALIDATION histories; trim all historical gate prefixes strictly.
Group E4 by episode/cutoff externally; unsupported soft candidates receive worst finite loss.
No experiments or scoring performed; detailed invocations and source caveats are in the JSON.
