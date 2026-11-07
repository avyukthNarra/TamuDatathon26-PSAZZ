# Teammates' pushed work vs ours (all re-scored through the same harness)

Pulled all 26 remote branches (2026-11-07). Models that run locally were re-scored on our splits: dev = 72 validation episodes, audit = 24 reserved episodes
(the audit set got extra logged looks for this comparison, `who=TEAM_COMPARE`). Teammates' models were tuned on all 96 validation episodes (or on train),
ours on dev only, so the comparison slightly favors them.

## Ranking (harness-scored; points /60)
| Model | Branch | Dev | Audit | Notes |
|---|---|---|---|---|
| **TM_COMBINED** (20% rule model + 80% pair mixer) | `zhill-branch` | **54.14** [53.1,55.1] | **55.37** [54.5,56.1] | stdlib only, deterministic, entry already built (`make_entry.py`), ~30 s |
| TM_MIXER (Pranav pair mixer) | `COMBINED_SOLUTIONS` (same code on `pranav/adaptive-mixer`) | 53.87 | 55.20 | their own claim 54.21 on all 96 |
| TM_RULE (zhill rule model v4) | `zhill-branch` | 52.89 | 52.85 | best on stable (.057 dev) |
| Our W5 stack | `sai/master-test` | 52.32 (OOF) | 54.09 [53.08,54.90] | needs numpy/scipy |
| Our C2 | `sai/master-test` | 52.09 | 53.77 | |
| TM_E67 (Avyukth final: 50/50 E6 + E7 neural) | `avyukth/entry-final` | 50.02 [49.0,51.1] | not scored | E7 alone 49.42, E6 alone 47.69 (audit 50.37); torch 2.9.1 (installed in an isolated venv) |

TM_COMBINED minus our C2 on audit: +1.59 (CI +0.91, +2.24). Our team stack (members of both) is only +0.15 over TM_COMBINED on dev OOF (CI -0.05,+0.34).
**The teammates' combined model is the best model found so far, not ours.**

## Who found what (self-reported unless marked re-scored)
- **Pranav (`pranav/*`, `COMBINED_SOLUTIONS`)**: lag-feature model (superseded) then the *pair mixer*: rediscovers lag pairs on each round's train (found (1,3),(2,5),(4,7), same as ours), one "did the two lags match or differ" cell per pair (pools XOR evidence, half the data), two memories (half-lives 12 and 40), size experts keyed on exact earlier payouts. Self: 53.35 -> 53.75 -> 54.21. Re-scored 53.87 dev / 55.20 audit.
- **zhill (`zhill-branch`)**: rule model (searches all 37 lag rules per machine, size rules): 52.89 self, re-scored 52.89 dev (matches). Blended 20/80 with the mixer: 54.45 self, re-scored 54.14 dev / 55.37 audit. Weight chosen on validation; independent train-Brier check preferred the mixer alone.
- **Avyukth (`avyukth/*`, `avy` docs)**: E1-E11 experiments on a different 48/48 split. E6 signed-L1 interaction logistic 48.2 audit (+8 over 0/1 coding, same structure insight), E7 neural hypernetwork over prefix statistics 49.0 audit, blend 50.0. E1-E5, E8-E11 (0/1 pair logistic, trees, memory selection, shape head) 36-40, not re-run. Re-scored E6 (47.69 dev, 50.37 audit) and E7 (49.42 dev) and his final E6+E7 blend (50.02 dev) on our split after installing torch: all about 4 points below TM_COMBINED, and they add nothing to it as blend members (weights 0 to 0.09). His E1-E5, E8-E11 were not re-run (self-reported 36-40).
- **Codex (our W3/W4, `codex/w3-w4`)**: E1-E5, S1-S8 on the old B3 base, none adds to B4; a B4-based rerun (S5_B4 soft-target distillation) was pushed late and is not analysed here.

## Why theirs beats ours (hypotheses, partly tested)
- Same structure discovered independently (lag-pair parity rules). Our win model was lag 7 + pairs ridge at lambda 0.05; theirs pools pair evidence in parity cells and adapts memory with short half-lives (12, 40) chosen by recent squared error. Error share of p0: theirs .45-.46 vs ours .54, and drift .096 vs .184, shared_change .109 vs .142, short_history .159 vs .191.
- We concluded "half-lives 4-12 hurt"; that held for our near-OLS ridge, not for their mixer. Their short memories are what fix drift/shared_change.
- Our size model (A9 lags 1,2,4 + alpha prior) is better than theirs on p5 (share .06 vs .08), but size is a smaller part of the error than win, and composing it with their win model gains only +0.09..+0.23 (best: win prob of TM_COMBINED with sizes averaged 50/50 with C1: 54.37 dev, +0.23 [+0.10,+0.35] over TM_COMBINED).
- Remaining error in TM_COMBINED (dev): p0 .46, p1 .26, p2 .20, p5 .08, so ~54% is now payout size, which is where improvement should come from.

## Recommendation
1. Treat `zhill-branch` (combined 20/80) as the entry baseline: it is stdlib, deterministic, packaged, and our best audit number (55.37).
2. Only worthwhile additions we found: size-share averaging with our C1 (+0.23, needs porting A9/alpha to stdlib or shipping numpy) and the team stack (+0.15, not CI-clear). Both are small; weigh against packaging risk.
3. Next experiments with real upside: size modelling on top of their win probabilities (their size experts keyed on exact payouts vs our per-machine severity ridge), apply short-memory mixing to the size model, and re-check our M9b peer gate on their win model.
4. Caveats: their weights and rule model were tuned on validation; our audit is no longer blind for their models; Avyukth's E7 and the Codex B4 rerun were not re-scored.
