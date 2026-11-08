# Size heads on Zhill's win side

Win side is fixed: Zhill's 0.1 rule + 0.8 mixer6 + 0.1 hidden-rule HMM (the 56.59 blend). Only the payout-size mix given a win changes. Validation targets are read at runtime for the mode variants. Train EM, the pair prior, and the static mass are fit on this round's train. No main-round constants are hardcoded.

Halves are the kit tune/audit split (`episode_folds`: within each condition, episodes sorted by sha256, even positions tune, odd positions audit). That is the make_entry half split. Ratios are loss ratios on all 96 validation episodes (lower is better).

## Variants

| name | engine | atoms | prior |
|---|---|---|---|
| base | none | blend's own size | — |
| zhill | `size_hmm`, switch 0 | validation modes | 97% flat on the train core, 2% no-rule, rest on other lags/thresholds |
| hmm_pair | same HMM | validation modes | that 97% core mass redistributed by train `P(L,s \| win pair)` |
| hmm_em | same HMM | train EM atoms | flat 97/2 prior, core rediscovered on the EM atoms |
| hmm_pair_em | same HMM | train EM atoms | pair prior on the core mass |
| rule_val | `size_rule` (static + dynamic, floor 1e-4) | validation modes | pair prior and learned static mass |
| rule_em | `size_rule.apply` | train EM | pair prior and learned static mass |
| rule_em_raw | same, no 1e-4 floor | train EM | pair prior and learned static mass |
| rule_gate | rule_em only if the train gate accepts it | train EM | same; otherwise the untouched blend |

Nothing in the table is a main-round literal. Validation-mode rows call `modes_from_targets` on this round's targets, which is the procedure Zhill already ships. Train-EM rows do not need that two-mix structure; `hmm_pair` falls back to `hmm_pair_em` when it is absent.

## Scores

<!-- SCORES -->
| variant | points | tune | audit | stable | short | drift | shared |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 56.591 | 56.551 | 56.648 | 0.0450 | 0.0998 | 0.0438 | 0.0524 |
| zhill | 57.322 | 57.438 | 57.196 | 0.0347 | 0.0811 | 0.0329 | 0.0382 |
| hmm_pair | 57.523 | 57.687 | 57.346 | 0.0325 | 0.0711 | 0.0315 | 0.0371 |
| rule_val | 57.497 | 57.618 | 57.369 | 0.0325 | 0.0715 | 0.0316 | 0.0385 |
| hmm_em | 57.307 | 57.428 | 57.177 | 0.0350 | 0.0810 | 0.0334 | 0.0386 |
| hmm_pair_em | 57.511 | 57.675 | 57.333 | 0.0328 | 0.0710 | 0.0319 | 0.0373 |
| rule_em | 57.483 | 57.607 | 57.352 | 0.0328 | 0.0716 | 0.0321 | 0.0387 |
| rule_em_raw | 57.483 | 57.607 | 57.352 | 0.0328 | 0.0716 | 0.0321 | 0.0387 |
| rule_gate | 57.483 | 57.607 | 57.352 | 0.0328 | 0.0716 | 0.0321 | 0.0387 |
<!-- /SCORES -->

## Decision

<!-- DECISION -->
Selected `hmm_pair` at 57.523 (tune 57.687, audit 57.346).

Zhill's size HMM reproduces at 57.322 (tune 57.438, audit 57.196). The blend with its own size split, before any size head, is 56.591. Both kit halves and all four conditions move the same way: short history does most of the work (loss ratio 0.0811 to 0.0711).

The pair prior is the gain. Train-EM atoms with Zhill's flat 97/2 prior score 57.307, a hair under Zhill. The same atoms with the train prior `P(L, s | win pair)` score 57.511. Swapping Zhill's validation modes for those EM atoms, and keeping the pair prior, costs 0.012 (57.523 to 57.511). Atom L1 versus the validation mixes is 0.016 balanced, 0.041 steady, 0.077 jackpot.

`size_rule` does not beat that HMM. With the learned static mass of 0.193 it scores 57.483, floor or no floor. The train gate accepts it against the mixer's own size head (held-out Brier delta −0.0077, 1.96 SE upper bound −0.0054, 96 episodes, 58,639 wins, both folds), so `rule_gate` is also 57.483. The static hypothesis is the part that loses: Zhill's no-rule weight stays at 0.02, and that HMM plus the pair prior is ahead of the static/dynamic mixture.

`choice.json` is `hmm_pair`. If this round's targets do not contain two size mixes, the code falls back to `hmm_pair_em`.
<!-- /DECISION -->

## Switch

`choice.json` is `{"variant": "hmm_pair", "fallback": "hmm_pair_em"}`. `size_variants.active_variant()` reads `variant`. If the file is missing, the default is `zhill`. `EM_FALLBACK` sends `hmm_pair` to `hmm_pair_em` when `modes_from_targets` returns nothing. Other names in the score table are valid switch values.

## Notes

<!-- NOTES -->
- base keeps the blend's own size split (size HMM off).
- validation modes for ['balanced', 'jackpot', 'steady']. train core: [(2, 1, False), (2, 1, True), (2, 2, False), (2, 2, True), (2, 4, False), (2, 4, True)].
- zhill is size_hmm, switch 0, 97/2 prior, validation modes.
- val-atom pair prior lags [1, 2, 4], static mass 0.193, pairs [(1, 3), (2, 5), (4, 7)].
- EM lags [1, 2, 4], static mass 0.193, loglik -58798.9, core [(2, 1, False), (2, 1, True), (2, 2, False), (2, 2, True), (2, 4, False), (2, 4, True)]. atom L1 vs validation modes: balanced 0.0161, jackpot 0.0774, steady 0.0407.
- The pair prior keeps Zhill's 2% no-rule mass and the small mass on other lags and thresholds. It only redistributes the 97% core across the six `(payout >= 2, lag in {1,2,4}, inverted)` rules, using the machine's train win pair. Pairs kept: (1, 3), (2, 5), (4, 7).
- Size gate accepts `size_rule` over the mixer size head. Pooled held-out episode Brier delta −0.00768 (SE 0.00116, upper −0.00541), 96 episodes, 58639 wins. Fold 0: −0.00714, upper −0.00391, static mass 0.206. Fold 1: −0.00823, upper −0.00500, static mass 0.182. Both folds keep lags (1, 2, 4).
<!-- /NOTES -->
