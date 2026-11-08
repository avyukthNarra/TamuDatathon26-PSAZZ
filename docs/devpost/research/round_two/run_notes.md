# Round-two (adaptation) run, 2026-11-08

Kit: organizer `adaptation-participants` (metadata `round: adaptation`, `max_points: 20`, old 80/20 scoring lib).
Our `mainFiles/slot_forecasting/rounds.py` (60/40) rejects 80/20 kits, so ranking tools ran on a copy whose
metadata says `max_points: 40` (rescales scores only). The entry itself runs fine on the real kit.

- Structure moved: win pairs (1,3),(2,5),(4,7) -> (3,8),(1,4),(2,6); original-pair share 0.00. Rule families: parity only.
- `gate_members.py`: keeps all three members (v13_short, exact, sysid), size member v13_short.
  Cross-fitted validation **38.59 / 40** (= ~19.29 / 20 on the real kit scale).
- `adapted_entry_v17g_rules/gate.json` is the adaptation gate. `bundled/validation/` = the kit's `validation/` (not committed).
- Ranking (RANKING.md) and final CSV/ZIP SHA-256 will be added when done.

## Blend tweak check (40-point scale, cross-fitted members, fit on one half / score on the other)
Default equal-weight blend 38.588 (A 38.609 / B 38.593). Fitted weights do not transfer (fit B -> A 38.436),
temperature 1.0 is best, size split must stay v13_short (others ~30), dropping exact is +0.04/+0.06 (< 0.1 margin).
Verdict: no change. Script `tweak.py`, output `tweak_out.txt`.

## Code ZIP
`ADAPTATION_entry.zip` sha256 feb33968c7070000891f84547e8a896a3c1e5e0c4538498be840f995d76729d2 (clean git archive + gate.json + kit validation/ as bundled/).

## More checks (40-point scale), all no change
- Per-type blend weights (sysid heavier on jackpot/balanced): fit A -> B +0.11, fit B -> A -0.08. Inconsistent, rejected (`pertype.py`).
- Headroom (`headroom.py`): true win + our size 39.63; our win + true size 38.93; blend 38.59. Win prob is most of the gap.
- Train + validation half as extra training episodes, predicting the other half as test (`trainval_*.py`):
  held-out B 38.502 -> 38.495 (-0.007), held-out A 38.404 -> 38.401 (-0.003). Members are data-saturated.
- Size lags moved to {1,2,3} (smaller member of each new win pair); the size rule searches range(1,9) on train and picks them up.

## 11:15 checks (40-point scale), all no change
- Short-history-only weights/temperature (detected from history gaps, 192/192 correct; `shortcal.py`): fit A -> B +0.02, fit B -> A -0.30; T=1.0 best.
- 4th win member (Codex, `research_pf/r2-member4`): + v19 A +0.009 / B -0.005; + v7c A -0.013 / B -0.097. v19's file is contaminated by full-validation gate choices.
- v20 (pre-fix, `research_pf/r2-v20`): 38.582 (A 38.633 / B 38.561). Pranav's 7b89c64 fix reports 38.62 (his halves 38.3 / 38.8).
- Docker (python:3.13, pinned numpy/scipy, offline, 2 CPUs, real 20-point kit): matches ADAPTATION_submission.csv, max diff 9.7e-15.
- Running: size-channel pair evidence (Codex), episode pair sharing (Grok), peer residuals (Grok), web research (Grok).

## 11:30 peer signal (Grok r2-peers + check on orch/common.folds halves)
Correction P(win) = sigmoid(logit(p) + a + b * peer_win_lag3), peer_win_lag3 = share of the episode's other 7 machines that won at cutoff-3.
Condition-balanced halves (base 38.5879, A 38.5306 / B 38.6328): fit A -> B +0.062 (a -0.30, b 0.90); fit B -> A +0.107 (a -0.15, b 0.60).
Other lags 1-8 are noise (|delta| < 0.05, inconsistent). Grok's permutation test 0/200. Real but below the 0.1 rule (~+0.085 mean);
lag 3 was picked after looking at all 8 lags. Size-channel pair evidence (Codex): +0.0006. Episode pair sharing: null.
