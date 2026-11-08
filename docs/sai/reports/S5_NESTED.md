S5_NESTED: fixed selected S5 configuration/hyperparameter; nested diagnostic only.
Outer/inner episode folds: 6/6; blend weights selected by harness grid of 21; weight bootstrap 100.
Smoke: 24 dev episodes, 6 folds; points=45.478611; runtime=0.120s.
Full dev points=46.314312; CI95=[45.16477691632841, 47.429690959335296]; paired vs B3={'diff': 0.4918268849866152, 'ci95': [0.2312350200079994, 0.7262517081803169]}.
Condition ratios={"drift": 0.3763676280118063, "shared_change": 0.35313141747329224, "short_history": 0.3322137371149647, "stable": 0.12027076014653439}.
Class error shares={"p_0": 0.5815847904977042, "p_1": 0.2332848477229891, "p_2": 0.13977044345444925, "p_20": 0.003737020159968715, "p_5": 0.04162289816488873}.
Outer weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; mean=1.000000; determinism max difference=0.00e+00.
Runtime=0.413s; Python 3.13.3; NumPy 2.3.3; stdlib-only=no.
Audit forecast uses full dev fit blended at mean outer weight; audit truth not read or scored.
Selected S5 configuration/hyperparameter were chosen globally on dev and remain optimistic; this diagnostic fixes them and nests member/weight fitting only.
Full-validation predictions saved in results/preds/S5_NESTED_validation.csv.
