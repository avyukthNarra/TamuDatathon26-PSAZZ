TEAM_GATE
DEV points 54.29123614222728; CI95 [53.370807552034655, 55.15336841883264]
Vs TM_COMBINED {'diff': 0.15168612704258067, 'ci95': [-0.06116580891580074, 0.3595505768265795]}
Condition ratios {'stable': 0.07301040311840858, 'short_history': 0.14409479553322338, 'drift': 0.09909763194175412, 'shared_change': 0.10440014428214739}
Class error shares {'p_0': 0.4676526422740056, 'p_1': 0.26063679760806663, 'p_2': 0.19600427206684856, 'p_5': 0.06908677246790713, 'p_20': 0.00661951558317201}
Blend {'w_per_fold': [0.75, 0.8, 0.6000000000000001, 0.9, 0.8500000000000001, 0.75], 'w_mean': 0.775, 'points': 54.27927461538926, 'ci95': [53.32126603871299, 55.17667556588137], 'condition_ratios': {'stable': 0.07344195904131957, 'short_history': 0.14634375256390386, 'drift': 0.09734896478425664, 'shared_change': 0.10444246701536529}, 'vs_base': {'diff': 0.13972460020456623, 'ci95': [-0.018696190205741377, 0.2899309974163067]}}
Runtime 1.239s; repeated-fit max difference 0.0
Six outer whole-episode folds; gate uses three inner whole-episode folds.
Gate lambda selected inside each outer fold by inner episode-OOF scoring; S5 q-trained frozen predictions excluded.
Frozen teammates already tuned on some/all validation; their old audit is not a blind holdout. Current fitting/scoring uses DEV only.
Audit predictions inferred using full DEV fits; audit truth never trained or scored.
Sources and outer-fold weights saved; no verdict assigned.
