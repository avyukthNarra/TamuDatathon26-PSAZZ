# Short-history P(win): per-machine pair identification (open task, 2026-11-08 11:05 CST)

Experiments stop 11:35 CST; the form closes 12:00. Safe upload is the default `adapted_entry_v17g_rules/` (38.588/40).

## The gap
- The round's pair support is found cleanly from train: (3,8) 185, (1,4) 180, (2,6) 164 confident votes on validation; the next best is 5.
  All rules are parity (family census parity 1.0, no xor3/AND/OR). The Walsh screen's top triples are artefacts.
- But only 529 of 768 validation machines get a confident pair (2-nat lead, at least 5 ticks per cell). The other 239 average over the three pairs.
  This is likely mostly short-history machines (not yet split by condition). Short history is the worst condition: loss ratio 0.049 vs about 0.031 elsewhere.
- Your headroom check: true P(win) gives 39.63 vs 38.59, so this is where the points are.

## Newer findings (11:07)
- Member-alone points by condition (40-point scale): every member is weakest on short history, by about the same amount, so
  re-weighting members cannot fix it; the short-history win model itself must improve.

  | member | stable | short_history | drift | shared_change |
  |---|---|---|---|---|
  | v17-guards (`members/v13_short.py`) | 38.59 | 37.75 | 38.52 | 38.46 |
  | exact filter | 38.62 | 37.88 | 38.08 | 38.66 |
  | sysid | 38.88 | 37.86 | 38.61 | 38.83 |
- Tried and failed (no gain): per-condition blend weights + temperature (38.50, half A -0.22); short-history-only weights chosen on one
  half (+0.02); mixing exact-filter sizes (worse on both halves). Perfect P(win) = +1.04, perfect sizes = +0.34.
- `members/v13_short.py` is v17-round2-guards (v17 + v7c at win weight 0.3) plus short/jackpot patches, not plain v13.

## Already being tried (Sai's Codex workers, do not duplicate)
- `sai/r2-shortwin`: self-supervised recalibration of short-history P(win) on train histories masked to look short.
- `sai/r2-blendw`: train-learned prior weights over the three pairs (per machine type) in the exact filter.
- `sai/r2-sizes`: empirical-Bayes pooling of short machines' cell win rates toward type x cell means.

## Suggested for Avyukth's bots
1. **Use the size channel to identify the pair.** Size lags moved to {1,2,3}, the smaller member of each win pair. If a machine's size
   rule lag tells you which win pair it is on, add the size outcomes' likelihood to the per-machine pair posterior. Check on train first:
   how often does the size lag predict the win pair?
2. **Use the unlabelled test/validation histories.** Each query's own history up to the cutoff is input, not a label. Make sure every
   observed tick (not just complete-lag ticks) enters the per-machine pair evidence for short machines (see AGENTS.md section 12, @short).
3. **Episode-level sharing.** If machines in one episode share their pair or the shared-change swap on train, pool pair evidence across
   an episode's machines. Check on train before building.

## How to score (same as Sai's workers)
Kit: a copy of `adaptation-participants/public` with `max_points` set to 40.
```
cd adapted_entry_v17g_rules && python3 -B predict.py --data <kit40> --split validation --protocol crossfit --out val.csv
```
Halves: use `orch/common.folds` (condition-balanced: episodes sorted by id within each condition, alternating A/B), the split the
pick rule uses. `tweak.py`'s global even/odd split gives different halves (38.609/38.593 vs 38.531/38.633 for the same 38.588 total).
Pick rule: beat 38.588 by more than 0.1 overall AND on both halves. Learn only from train (+ validation answers as priors); never test targets.
