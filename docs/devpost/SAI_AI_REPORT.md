# How Sai built with AI: a usage report (TAMU Datathon 2026)

Sai worked the whole hackathon as an **orchestrator of AI agents** rather than as a solo coder: Claude Code ran the planning, analysis and
git/PR work, Codex ran the heavy experiments, and Pane held them together in parallel worktrees. This report measures that, with numbers
from the local agent logs and the repo, and pulls out what a senior engineer would want to know: what worked, what it cost, where the AI
was wrong, and what we would do differently.

**Window:** Sat 2026-11-07 10:28 → Sun 2026-11-08 12:10 CST (about 25.7 hours). **Sources:** Claude Code transcripts for this repo
(14 sessions + 22 subagent transcripts), 24 Codex session logs, Pane's session database, `git log` of every branch, and the GitHub API.
Only aggregate counts are reported here: no prompts, transcript text or secrets. Reproduce with `docs/devpost/tools/ai_stats.py`
(runs on Sai's machine, where the logs are).

## At a glance

| Metric | Value |
|---|---|
| AI-active hours (any agent working, 10-minute bins) | **16.2 h** of 25.7 h |
| Claude Code active hours / Codex active hours | 14.7 h / 6.2 h |
| Tokens of context processed (Claude + Codex) | **575 million** |
| Tokens generated (Claude + Codex, incl. Codex reasoning) | **2.4 million** |
| Context read per token written (Claude) | **260 : 1** |
| Prompt-cache hit rate (Claude / Codex) | 98.6% / 98.5% |
| Largest single context | 526k tokens |
| Claude tool calls / of which shell (`Bash`) | 1,711 / **86.9%** |
| Codex shell commands (`exec`) | 1,394 (16.4 per turn) |
| Subagents spawned (Claude) / Codex sub-agents | 22 / 9 |
| Pane panes for this repo / terminal tabs | 12 / 29 |
| Sai's commits (all branches) / with an AI co-author trailer | 135 / 92 (68%) |
| Team commits with an AI trailer or session link | **at least 187 of 243 (77%)** |
| Sai's hand-off-quality commit messages (median body) | 481 characters |
| Python written by Sai's agents | ~29.7k lines in 243 files (+9.2k lines of Markdown) |
| PRs opened by Sai / merged | 15 / 12 |
| Codex 5-hour rate limit, peak use | **96%** |

## 1. The shape of the work

Claude messages per hour (CST) and Sai's commits in the same hour:

```
Sat 11  ###########                                66 msgs    5 commits   playbook, harness
Sat 14  ########################################  246 msgs   20 commits   four parallel research tracks (B3 -> B4 jump)
Sat 16  #######################                   144 msgs   13 commits   stacking, team re-scoring
Sat 17  #######################                   140 msgs   14 commits   combination search, top-10 freeze
Sat 19  #############                              78 msgs    8 commits   mixer5; Walsh and size-switch kill tests
Sat 21  ############                               73 msgs    0 commits
Sun 01  #########################                 155 msgs   24 commits   fresh-start win models, 57.97 blend
Sun 02  #######################                   143 msgs   13 commits   round-two adaptation
Sun 03  #########################                 155 msgs    3 commits   run_round2.py, default choice
Sun 04  ###################                       114 msgs    1 commits   rules blend becomes the default
Sun 05  #########                                  54 msgs    2 commits
Sun 06-09                                          10 msgs    1 commit    sleep (about 3 hours)
Sun 10  ##########################                160 msgs    4 commits   round-two kit arrives 10:35
Sun 11  ##################################        207 msgs   15 commits   orchestrating 3 Codex workers + teammates' bots
```

Two bursts, one sleep. The busiest hours were not the coding hours: Sat 14:00 (four research tracks at once) and Sun 11:00 (round two)
were orchestration peaks, where the human's job was routing work between agents.

## 2. Division of labour between models

| Model | Share of Claude messages | Context processed | Output | Used for |
|---|---|---|---|---|
| Claude Opus 5.5 | 61.6% | 282M | 674k | Orchestration, analysis, git/PR work |
| Claude Sonnet 5.5 | 30.4% | 85M | 607k | Earlier sessions and workers |
| Claude Haiku 5.5 | 8.0% | 13M | 182k | Cheap subagents (search, small tasks) |
| Codex (GPT-6.1 Sol / GPT-6 Luna) | 50 / 34 turns | 196M | 963k (47% reasoning) | Heavy experiment runs in their own worktrees |

Codex worked almost exclusively **under Claude orchestration**: in 45 of the 46 ten-minute windows where Codex was active, a Claude
session was active too. The pattern was always the same: Claude wrote a brief with the baseline numbers, the scoring command and the
pick rule; Codex ran experiments in a Pane worktree; Claude read the report and decided.

## 3. Things that are cool (and some that are thought-provoking)

1. **Agents read 260 tokens for every token they write.** 380M tokens of context went in to produce 1.46M tokens out. With a 98.6%
   cache hit rate this is cheap, but it means the real product of a long session is the *context*: the same 500k-token conversation
   is re-read hundreds of times. Long-lived orchestrator sessions only work because of prompt caching.
2. **The shell is the universal tool.** 87% of Claude's tool calls were `Bash`; only 44 used the dedicated Edit/Write tools.
   Most file changes were scripted (Python heredocs, `sed`) and most analysis was one-off Python against the data. Top shell commands:
   `git` 263, `cat` 159, `python3` 151, `sed` 100, `grep` 65, `gh` 52, `runpane` 46.
3. **Git became the lab notebook.** 61 of Sai's 135 commits are `test(...)` commits that record an experiment and its verdict,
   including the negative ones ("no gain", "loses", "saturated"). Median commit body: 481 characters (Zhill's agent-written bodies:
   1,042). The repo's first three commits read like classic hackathon commits: "First Commit", "woooo", "stinky commit". After the playbook
   landed at 11:10, almost every commit became a structured Conventional Commit with its reasoning, because the agents follow `AGENTS.md`.
4. **Most of the code in the repo was AI-written.** At least 77% of all team commits carry an AI co-author trailer or a Claude session
   link (Pranav 49/51, Zhill 17/18). The humans chose the problems, set the rules, and judged the results.
5. **Process beat cleverness.** The biggest single gain of the weekend (+5.1 points, B3 → B4) came from noticing an over-regularized
   ridge penalty. The second biggest (+0.47, the 57.97 blend) came from combining three independently built models whose errors were
   only 0.8 correlated. Neither came from a fancier model. The agents were most valuable for running dozens of honest, cross-fitted
   checks fast enough to find those.
6. **Honest evaluation was enforced by rule, not willpower.** The pick rule (> 0.1 points on the mean *and* on both condition-balanced
   halves) rejected at least ten "improvements" that looked real on one half, including the final-hour best (+0.022 overall, −0.006 on one half).
7. **Diminishing returns are measurable, and the AI found the wall.** Swap-in-truth analysis on round two showed perfect short-history
   predictions were worth at most +0.46 points. Three Codex workers plus Avyukth's bots then spent the last hour on seven ideas; best
   result +0.022, nothing shipped, Codex at 96% of its rate limit. **Measure the ceiling before buying compute.**

## 4. Where the AI was wrong (and how it was caught)

A senior engineer should trust this report more because of this section. Every one of these happened during round two:

| Mistake | How it was caught | Cost |
|---|---|---|
| Reported "~30% of machines on 3-lag rules" from a Walsh screen | Avyukth's notes said "parity only"; the model's own rule census confirmed 0% 3-lag | Two workers briefly aimed at the wrong idea; docs corrected within minutes |
| Docs said payout sizes came from "v13" (trusting a file name) | Sai pointed out that sizes come from v17 + Avyukth's blend | Fixed in README, AGENTS, Devpost; the module docstring had it right |
| First error split used KL divergence; the official metric is squared error vs a reference | Re-checked the scoring code before acting | Conclusion held (short history's gap is sizes), magnitudes changed |
| Iteration history credited Sai for round-two adaptation built by Avyukth and Zhill | Cross-checked commit authors before publishing to `main` | Fixed before the PR merged |
| Assumed pronouns for teammates in drafts | Self-review before publishing | Fixed |
| Tried to merge `main` into a checkout where the live round-two run was going | The merge aborted on untracked files; Sai then set a "separate worktree only" rule | None, but close |
| Two organizer-kit commits (`ad704ad`, `9e86c27`) were pushed straight to `main` on round-two morning, against the playbook | Found while writing this report (`git log --first-parent main`) | None (data-only commits), but the rule exists for a reason |
| Redirects to busy Codex workers arrived late; two re-reported their old task | Read the report contents, not just the "done" event | About 10 minutes of worker time |

Pattern: the AI's errors were **confident readings of a proxy** (a screening statistic, a file name, the wrong metric, a branch name
standing in for authorship). The fix every time was to go back to the primary source: the census, the code, the scorer, `git log`.

## 5. What a senior engineer would take away

- **Write the playbook first.** `AGENTS.md` (branching, Conventional Commits, PR rules, "never push to main") was the highest-leverage
  hour of the weekend: four humans and many agents worked in parallel for 25 hours, and after the four kickoff commits every change
  reached `main` through a PR, with two exceptions (section 4).
- **One feature = one worktree = one agent.** Pane's worktree-per-agent model made parallel experiments safe; nothing collided.
- **Give agents a scoring contract.** Every worker brief carried the baseline, the exact scoring command, and the acceptance rule.
  That made results comparable across Claude, Codex and four teammates' bots.
- **Keep a skip list.** The most useful orchestration message was "already tried, do not duplicate": it saved more compute than any
  single idea produced.
- **Distrust fluent summaries; verify against primary sources.** See section 4.
- **Budget the human, not the model.** The bottlenecks were the human's attention at the two orchestration peaks and Codex's rate
  limit, not tokens.

## Method notes

- Active time = sum of gaps of 15 minutes or less between consecutive log events (per agent), or 10-minute bins for the combined figure.
- Claude usage is counted once per API message (transcripts repeat a message per content block). Context processed = uncached input +
  cache reads + cache writes. Codex totals are each session's final cumulative token count; output includes reasoning.
- "AI co-author" = a `Co-Authored-By: Claude` trailer or a `Claude-Session:` link in the commit body; commits without one may still be
  AI-assisted, so 77% is a lower bound. Line counts exclude CSV/JSON data (2.08M lines of predictions and caches).
- Teammates' own agent logs (their machines) are not included, except through their commits.
