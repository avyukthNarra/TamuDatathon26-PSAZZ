W3/W4 complete on codex/w3-w4; development only:72 episodes; B3 45.8225 points, CI95 [44.74847632985979, 46.90060343816345].
Episode/peer dev points: E1 45.7497; E2 45.7676; E3 45.8130; E4 45.7662; E5 45.7813.
Learner dev points: S1 46.5886; S2 46.7653; S3 41.9140; S4 45.0240.
Learner dev points: S5 46.3991; S6 34.7398; S7 44.7341; S8 39.9203.
Completed 2464 unique declared grid cells across method/iteration/family variations; all13 methods completed32 section5 configurations.
S1 selected20 iterations;80-iteration result retained separately. S8 selected200 iterations;35-iteration result retained separately. S2 tested CatBoost and LightGBM.
Per-method JSONs contain full grids, condition ratios, class-error shares, bootstrap CIs, paired B3 differences and episode-OOF blend measurements.
W3W4_CRITERIA.json records section9 numerical criteria and condition-specific hybrid contribution CIs; no method verdicts assigned.
S5_NESTED.json provides a fixed-selected-hyper nested stack diagnostic (46.31431150257826 dev points); global hyperparameter-selection optimism remains.
Every method repeat-run difference=0; all validation CSVs contain768 unique finite normalized probability rows; W3W4_VALIDATION.json contains verification evidence.
Only ticks strictly below cutoff; NaN remains unknown; training/holdouts grouped by episode; no audit scoring or audit-truth fitting.
Sources retained in W3_IMPLEMENTATION.json,W4_IMPLEMENTATION.json,W3W4_COMMON.json; feature metadata/provenance retained without intermediate matrices.
Environment and reproduction details: W3W4_ENVIRONMENT.json; optional dependencies/runtime isolated in /private/tmp; no project dependencies changed.
Limits: S5 original blend is nonnested; S8 reached iteration cap; E ablations freeze baseline; S6 future windows can average through drift.
Harness unchanged; single-condition CI limitation reported using full-dev hybrid CIs. No source/test/lint/CI scaffolding added; lead owns audit and verdicts.
