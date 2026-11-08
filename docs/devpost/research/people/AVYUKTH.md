# Avyukth: research and building 

Drafted by Sai's agent from Avyukth's 38 commits (branches `avyukth/*`, `avy`). Scores are main-round validation /60 unless marked /40.
Reviewed by Avyukth; reasoning added under *Key ideas and reasoning*.

## Path
| When | Step | Score | Commit |
|---|---|---|---|
| Sat 13:40-16:39 | Research and brainstorming notes (incl. 20 papers); experiment summaries E1-E7 and the E6+E7 blend entry; handoffs for future agents | | `2213ab3`, `c954948`, `08898f3`, `5b71842` |
| Sat 18:55 | Master research branch with all findings (`research/MASTER.md`, data findings, eval-suite design) | | `dec062a` |
| Sun 00:34 | **v7c**: size-atom rule, episode flip swap, round-two guards | 57.37 (4-fold 57.36) | `9481d4c` |
| Sun 02:05-02:38 | Late-night results and round-two warnings; fix so v7c's guards stop losing to v6 on shifted rounds | | `3f842a9`, `87e4233` |
| Sun ~03:00 | Parallel eval suite (`--jobs`, full run 33 → 16 min) and the 10-kit adaptability benchmark, merged to main | | PR #20, PR #21 |
| Sun 03:19 | **Sai's blend adapted so every member learns its lag pairs per round** | 57.96 | `2730597` |
| Sun 03:43 | Blend's first member swapped to v17-guards | 57.97 | `3083c87` |
| Sun 04:13 | v7c gated in as a 4th member for mixed-rule rounds | | `c2eb7e3` |
| Sun 04:17 | **Exact filter learns 3-lag XOR and AND/OR rules per round** (K10 32.1 → 35.4 /40) | | `979d2d9` |
| Sun 03:55-05:21 | Portable `run_round2.py`; no crash without validation targets; saved-priors guard restored | | `23ac6de`, `148199b`, `a4b5448` |
| Sun 10:44-11:27 | Round two: adaptation gate (38.59/40), blend-tweak, per-type, headroom, train+val, short-history calibration, 4th member, Docker, peer lag-3 checks | 38.59 / 40 | `f5ec781` … `ef74e22` |
| Sun 11:32 | Round-two upload files (CSV, ZIP, hashes) | | `a80ddc6` |
| Sun 11:19-11:40 | Devpost write-up, checklist, submissions record, research reports | | `6f82662`, `3740c54` |

Also: the 10-kit simulated adaptability benchmark (`mainFiles/adapt_bench`), used to check every round-two change.

## Key ideas and reasoning
- **Simulator forensics first.** Before tuning, we reverse-engineered the generator from the validation targets: win = XOR/XNOR of a machine's own wins at one lag pair ((1,3),(2,5),(4,7) on main), per-type live/dead levels, drift as a linear live/dead swap, shared change as one episode-wide flip. Every later change was a hypothesis about that structure, tested against it.
- **Size-atom rule (v7c, +0.74).** Payout size given a win switches between two atoms by `s XOR 1{payout(t-L) >= 2}`, where L is the smaller member of the machine's win pair. Two atoms per type are learned by EM on train; the lag L and direction s are learned per machine, with a prior P(L, s | win pair) from train (+0.18) and the flip swap for shared change (+0.18). Train-only gates fall back to the old size head when the rule does not fit (which is what it did on static-size synthetic rounds).
- **Adapt, don't hard-code.** A model that scores well only on the main round is worth little in round two, so every constant became something learned per round: lag pairs (census on train), size lags, rule families (pairwise, 3-lag XOR, AND/OR), and member inclusion (gate). This is why round two needed no code changes: the pairs moved to (1,4),(2,6),(3,8) and the size lags to {1,2,3}, and the entry relearned both.
- **Measure adaptability.** The 10-kit benchmark simulates shifted rounds (moved pairs, mixed rules, static sizes, etc.) and is fast enough to run on every change (`--quick` ~4 kits). It caught the locked-pair collapse (~18/40 on shifted kits) before round two, and checked the 3-lag/AND-OR rules fix (K10 32.1 → 35.4 /40).
- **Honest selection.** Every tweak was fitted on one half of validation and scored on the other; a change shipped only if it beat the default by > 0.1 overall and on both halves. In round two this rejected a dozen plausible tweaks and kept the verified blend (38.59/40).
- **Verify what ships.** The uploaded files were rebuilt from a clean git archive, format-checked on the real kit, and rerun organizer-style in Docker (python 3.13, pinned numpy/scipy, offline, 2 CPUs): match to 9.7e-15.
- **Round-two finding left on the table.** The share of an episode's other machines that won 3 spins ago predicts a machine's win probability (+0.06/+0.11 on the two halves; no other lag), under the 0.1 bar at the deadline. Pranav's v21 builds the same peer signal into v20 (38.67/40).

## Notes and code
- `HOUSE_EDGE_EXPERIMENT_RESULTS.md`, `HOUSE_EDGE_ADDITIONAL_RESEARCH.md` on `avy`; `research/MASTER.md` on `avyukth/master-research`.
- v7c on `avyukth/v7c-size-flip`; round-two run and checks in `r2/adaptation_run/` on `avyukth/r2-adaptation`.

## For Avyukth to add
- [x] What v7c's size-atom rule does (see *Key ideas*). E1-E7 details are in `HOUSE_EDGE_EXPERIMENT_RESULTS.md` on `avy`.
- [x] Reasoning behind the adaptation design (see *Key ideas*).
