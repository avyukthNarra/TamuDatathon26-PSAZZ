# B4 learner rebase
DEV only:72 episodes; B4 50.890458, CI95 [49.84724202918918, 51.86744388008814].
ID | grid cells | DEV points | paired gain over B4 (95% CI)
--- | ---: | ---: | ---
S1_B4 | 320 | 50.985684 | +0.095226 [-0.022371,+0.212208]
S2_B4 | 640 | 51.002522 | +0.112064 [+0.013920,+0.216757]
S5_B4 | 320 | 51.048033 | +0.157575 [+0.047270,+0.272711]
S5_B4 nested fixed-selected-hyper blend: 51.039727; vs B4 +0.149269, CI95 [0.04512771352039504, 0.2518358648637654]; outer weights [1.0, 1.0, 1.0, 1.0, 1.0, 0.75].
All1280 grid cells completed without failures; every selected model refit difference=0; S5 nested re-fit difference recorded in result.
All three main prediction CSVs plus the nested diagnostic have768 finite normalized rows; harness DEV scores and bootstrap intervals reproduce.
B4 base columns match the lead reference within3.33e-16; strict cutoff and missing-tick invariance checks pass.
Only S1,S2,S5 rerun; S3,S4,S6,S7,S8 skipped per lead instruction. Shared harness and original checkout were not edited.
Source: methods/*_B4.py and methods/_w4_b4_*.py; grids:grids/<ID>.json; small results:results/<ID>.json; predictions:preds/<ID>_validation.csv.
Global DEV hyperparameter/strength selection makes intervals optimistic, including fixed-selected-hyper nested diagnostic. Audit was neither scored nor used for fitting; no verdicts assigned.
Branch:codex/w3-w4. Lead owns final audit comparison and method decisions.
