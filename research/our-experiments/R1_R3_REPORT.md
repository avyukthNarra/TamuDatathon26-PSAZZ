# R1–R3 fixed-tuning screen

48 fixed tuning episodes (384 queries); no audit. Paired CIs use 1,000 stratified episode draws (seed 314159). V/T=validation/train; M/B=mixer/20% rule blend; Br/LL=Brier/log-loss. S/H/D/C are full-population raw errors (stable/short-history/drift/shared-change). J averages 123 jackpot queries within episode, then equally across 45 eligible episodes (3 omitted); J/ref=candidate/reference J.

## R1 — rejected

Rejected: drift/shared errors worsen; LL does not beat Br for V (paired Δ: M −.115 [−.324,+.084], B −.099 [−.263,+.057]). All R1 candidates have worse J than matched R0.

| Setting | Points | S/H/D/C/J raw errors | J/ref | Δ vs matched R0 [95% CI] |
|---|---:|---|---:|---|
| V-Br-M | 42.7590 | .015979/.022132/.151512/.141344/.059399 | .453940 | −12.267 [−13.571,−11.035] |
| V-Br-B | 46.1206 | .013196/.021010/.110551/.101818/.044799 | .342363 | −9.136 [−10.224,−8.103] |
| V-LL-M | 42.6444 | .016206/.021749/.153183/.142982/.060053 | .458936 | −12.382 [−13.673,−11.131] |
| V-LL-B | 46.0216 | .013324/.020892/.111714/.102969/.045325 | .346389 | −9.234 [−10.330,−8.204] |
| T-Br-M | 44.8818 | .021636/.022160/.120282/.112114/.048110 | .367669 | −8.923 [−9.864,−8.056] |
| T-Br-B | 47.6032 | .017911/.022544/.090002/.082540/.037720 | .288264 | −6.582 [−7.367,−5.903] |
| T-LL-M | 44.5659 | .021844/.021498/.122980/.117835/.050480 | .385781 | −9.239 [−10.194,−8.371] |
| T-LL-B | 47.3610 | .018075/.021965/.091729/.086667/.039312 | .300432 | −6.824 [−7.593,−6.119] |

R1: `avyukth/exp-r1-bayes-experts`, code `2da3b54`, pushed HEAD `981318f`.

## R2 — modest overall gain, worse jackpot error

W32 train blend improves short-history error 4.33% but worsens J 7.43%; W64 worsens J 6.78%. Both windows worsen jackpot error in all four conditions. Centers W32 .1054/.4566/.8645; W64 .1185/.4732/.8474 (historical low/high .0435/.898).

| Setting | Points | S/H/D/C/J raw errors | J/ref | Δ vs matched R0 [95% CI] |
|---|---:|---|---:|---|
| V-M-W32/64-reused | 55.0259 | .019842/.023117/.016751/.012820/.019703 | .150577 | 0.000 [0.000,0.000] |
| V-B-W32/64-reused | 55.2561 | .016539/.022659/.016546/.013071/.017015 | .130033 | 0.000 [0.000,0.000] |
| T-M-W32 | 53.9916 | .024269/.020784/.024508/.020470/.021136 | .161524 | +.187 [+.009,+.364] |
| T-B-W32 | 54.3598 | .020019/.021445/.022797/.019484/.018548 | .141750 | +.174 [+.029,+.313] |
| T-M-W64 | 53.9572 | .024080/.021041/.024762/.020707/.020914 | .159827 | +.153 [+.015,+.288] |
| T-B-W64 | 54.3200 | .019918/.021699/.023056/.019714/.018436 | .140889 | +.135 [+.024,+.241] |

R2: `avyukth/exp-r2-em-priors` candidate `14835f5`, scorer `95b4b61`, HEAD `2749594`; V reuses historical controls, T uses safe R0.

## R3 — no-op

No last-seven/eight gaps (0/96 per condition); short-history gaps are older. Exact R0 controls reused; no model fit or query change.

| Setting | Points | S/H/D/C/J raw errors | J/ref | Touched / condition | Δ vs matched R0 [95% CI] |
|---|---:|---|---:|---:|---|
| V-M-reused | 55.0259 | .019842/.023117/.016751/.012820/.019703 | .150577 | 0/96 | 0.000 [0.000,0.000] |
| V-B-reused | 55.2561 | .016539/.022659/.016546/.013071/.017015 | .130033 | 0/96 | 0.000 [0.000,0.000] |
| T-M-reused | 53.8046 | .023487/.021895/.026076/.021647/.019216 | .146852 | 0/96 | 0.000 [0.000,0.000] |
| T-B-reused | 54.1853 | .019582/.022414/.024127/.020457/.017265 | .131945 | 0/96 | 0.000 [0.000,0.000] |

R3: `avyukth/exp-r3-missing-filter` diagnostic `51b1e93`, HEAD `192709d`. Sai v4 details: [`SAI_PRIOR_FINDINGS.md`](SAI_PRIOR_FINDINGS.md); different dev folds preclude direct matched-control comparisons. See [jackpot breakdown](r1_r3_jackpot.json).
