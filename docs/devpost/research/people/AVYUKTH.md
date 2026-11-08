# Avyukth: research and building (draft for Avyukth to correct)

Drafted by Sai's agent from Avyukth's 38 commits (branches `avyukth/*`, `avy`). Scores are main-round validation /60 unless marked /40.
**Avyukth: please fix anything wrong, add the reasoning behind each step, and approve.**

## Path
| When | Step | Score | Commit |
|---|---|---|---|
| Sat 13:40-16:39 | Research and brainstorming notes (incl. 20 papers); experiment summaries E1-E7 and the E6+E7 blend entry; handoffs for future agents | | `2213ab3`, `c954948`, `08898f3`, `5b71842` |
| Sat 18:55 | Master research branch with all findings (`research/MASTER.md`, data findings, eval-suite design) | | `dec062a` |
| Sun 00:34 | **v7c**: size-atom rule, episode flip swap, round-two guards | | `9481d4c` |
| Sun 02:05-02:38 | Late-night results and round-two warnings; fix so v7c's guards stop losing to v6 on shifted rounds | | `3f842a9`, `87e4233` |
| Sun 03:19 | **Sai's blend adapted so every member learns its lag pairs per round** | 57.96 | `2730597` |
| Sun 03:43 | Blend's first member swapped to v17-guards | 57.97 | `3083c87` |
| Sun 04:13 | v7c gated in as a 4th member for mixed-rule rounds | | `c2eb7e3` |
| Sun 04:17 | **Exact filter learns 3-lag XOR and AND/OR rules per round** (K10 32.1 → 35.4 /40) | | `979d2d9` |
| Sun 03:55-05:21 | Portable `run_round2.py`; no crash without validation targets; saved-priors guard restored | | `23ac6de`, `148199b`, `a4b5448` |
| Sun 10:44-11:27 | Round two: adaptation gate (38.59/40), blend-tweak, per-type, headroom, train+val, short-history calibration, 4th member, Docker, peer lag-3 checks | 38.59 / 40 | `f5ec781` … `ef74e22` |
| Sun 11:32 | Round-two upload files (CSV, ZIP, hashes) | | `a80ddc6` |
| Sun 11:19-11:40 | Devpost write-up, checklist, submissions record, research reports | | `6f82662`, `3740c54` |

Also: the 10-kit simulated adaptability benchmark (`mainFiles/adapt_bench`), used to check every round-two change.

## Notes and code
- `HOUSE_EDGE_EXPERIMENT_RESULTS.md`, `HOUSE_EDGE_ADDITIONAL_RESEARCH.md` on `avy`; `research/MASTER.md` on `avyukth/master-research`.
- v7c on `avyukth/v7c-size-flip`; round-two run and checks in `r2/adaptation_run/` on `avyukth/r2-adaptation`.

## For Avyukth to add
- [ ] The E1-E7 experiments in a sentence each, and what v7c's size-atom rule does.
- [ ] Anything missing, e.g. the reasoning behind the adaptation design.
