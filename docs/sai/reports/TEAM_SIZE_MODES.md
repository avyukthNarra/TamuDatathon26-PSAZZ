# TEAM_SIZE_MODES
12 predeclared cells; TRAIN severity lags (1, 2, 4); six condition-stratified episode folds.
Best {'hl': 1000000000.0, 'temp': 1.0, 'eta': 8.0}; DEV OOF 54.868, CI [53.8502304212314, 55.85896668451689].
vs TM_COMBINED {'diff': 0.7288198966250548, 'ci95': [0.4000939903186591, 1.0663716677494464]}; blend gain {'points': 54.90229742746246, 'ci95': [53.886201501495336, 55.87463131633315], 'vs_base': {'diff': 0.7627474122777613, 'ci95': [0.475814849490887, 1.0551147542293833]}, 'w_mean': 0.8166666666666668}.
Runtime 13.5s; actual selected OOF repeat max diff 0.0.
Two mode centers per type learned only from each outer-training DEV fold; no audit truth fitting.
Win probability frozen from TM_COMBINED; severity inference uses only observed payouts strictly before cutoff.
Centers rounded to 12 digits for learned-parameter cache identity; no hardcoded modes or selected lags.
Canonical CSV contains DEV OOF predictions and unscored audit inference from full-DEV centers.
Centers fit outer-training DEV only; hyperparameter globally selected on OOF DEV (not nested), so selected score and blend gain retain selection optimism. Audit predictions inferred with full DEV centers and never scored.
No overall verdict.
