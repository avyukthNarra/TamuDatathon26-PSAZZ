# Devpost materials: House Edge (TAMU Datathon 2026, team PSAZZ)

| File | What it is |
|---|---|
| `WRITEUP.md` | Project text in Devpost's sections; paste into the project page |
| `CHECKLIST.md` | Submission steps; main-round files unchanged, round-two files labelled `ADAPTATION_*` |
| `SUBMISSIONS.md` | Exact files and SHA-256 for both rounds and where to get them |
| `SAI_AI_REPORT.md` | How Sai built with AI agents: usage stats, model split, what worked, where the AI was wrong |
| `SAI_ITERATIONS.md` | Sai's full iteration history and reasoning, with a score timeline (raw material for the story) |
| `research/main_round/` | Simulator forensics and the methods we tried before choosing the blend |
| `research/round_two/` | Adaptation run: what changed, every experiment, reproduction check |

## Results at a glance
| Round | Model | Validation | Notes |
|---|---|---|---|
| Main | adapted blend (`adapted_entry_v17g_rules/`) | 57.97 / 60 cross-fit, 57.86 honest 4-fold | Docker eval suite: 0 FAIL |
| Round two | same entry, relearned from round-two train | 38.59 / 40 cross-fit | pairs moved to (1,4),(2,6),(3,8); Docker rerun matches to 9.7e-15 |
