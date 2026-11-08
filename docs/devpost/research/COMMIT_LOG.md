# Team commit log (all branches)

Every non-merge commit on every branch of the repo, oldest first, de-duplicated by author and subject (cherry-picks and rebases
appear once). Generated from `git log --remotes=origin --no-merges`; commit subjects follow Conventional Commits, and scores in
parentheses are main-round validation /60 unless marked /40. Look a commit up with `git show <sha>`.

Commits per person: Sai 142, Pranav 51, Avyukth 38, Zhill 18

| When (CST) | Commit | Who | What |
|---|---|---|---|
| Sat 11-07 10:28 | `0cc9cf7` | Avyukth | First Commit |
| Sat 11-07 10:32 | `a5f32ae` | Pranav | woooo |
| Sat 11-07 10:40 | `7dc55e8` | Zhill | stinky commit |
| Sat 11-07 11:10 | `89f1eed` | Sai | docs: add agent playbook (#1) |
| Sat 11-07 11:24 | `c63fde4` | Sai | docs: spell out pr flow, approval, and merge conditions (#3) |
| Sat 11-07 11:24 | `fde4147` | Sai | docs: treat section 0 TODOs as expected until kickoff (#2) |
| Sat 11-07 11:28 | `4b052b8` | Sai | docs: remove stale gitlab and branch-protection guidance (#5) |
| Sat 11-07 11:28 | `5bbda51` | Sai | chore: remove Hello.py test file (#4) |
| Sat 11-07 11:39 | `90d1e26` | Avyukth | chore(mainfiles): keep model run outputs out of git (#7) |
| Sat 11-07 12:04 | `b3854e2` | Pranav | Added main files |
| Sat 11-07 12:27 | `d8e3b5c` | Avyukth | chore(mainfiles): keep model run outputs out of git |
| Sat 11-07 12:31 | `532d06d` | Sai | docs: require deleting merged branches after each merge (#6) |
| Sat 11-07 12:35 | `ecce225` | Avyukth | docs(mainfiles): correct overwrite note and cover root output dirs |
| Sat 11-07 12:48 | `9f6e1c4` | Pranav | chore(model): write lag model runs to git-ignored runs/lag-model |
| Sat 11-07 12:48 | `e2f7884` | Pranav | feat(model): add lag-feature forecaster trained on every past spin |
| Sat 11-07 13:31 | `aed7329` | Pranav | feat(model): add lag-pair mixer with verified entry (validation 53.35) |
| Sat 11-07 13:37 | `98493d0` | Pranav | perf(model): tune pair mixer at cutoffs to 53.75 (two memories, size experts) |
| Sat 11-07 13:39 | `77c320b` | Pranav | feat(model): rediscover lag pairs and rule type from each round's train |
| Sat 11-07 13:40 | `2213ab3` | Avyukth | docs: add House Edge research and brainstorming notes |
| Sat 11-07 14:04 | `06d7204` | Sai | docs: merge duplicate Added heading in changelog |
| Sat 11-07 14:04 | `aefff96` | Sai | feat(harness): add shared experiment harness for the master test plan |
| Sat 11-07 14:04 | `51acd17` | Sai | docs(sai): add research notes, master test plan, and prior experiments |
| Sat 11-07 14:11 | `7795dc4` | Sai | feat(harness): add smoke split, blend-gain stack test, and worker brief |
| Sat 11-07 14:20 | `12dcc8e` | Sai | test(sai): record first W4 residual learner grid |
| Sat 11-07 14:20 | `3968ebe` | Pranav | chore(model): remove superseded lag-feature model |
| Sat 11-07 14:24 | `f696551` | Sai | test(sai): record episode and peer method experiments |
| Sat 11-07 14:30 | `98e9c50` | Zhill | feat(model): add per-machine lag-pair rule model (52.89 validation) |
| Sat 11-07 14:31 | `49baf63` | Pranav | docs(model): add paste-ready prompt that rebuilds the pair mixer |
| Sat 11-07 14:31 | `162ac33` | Sai | test(sai): record W2 memory methods M1-M10 and verdicts |
| Sat 11-07 14:36 | `bd7e94d` | Sai | test(sai): record W1 frequency/shrinkage methods A1-A8 and verdicts |
| Sat 11-07 14:41 | `55afa43` | Pranav | perf(model): key payout-size experts on exact earlier payouts (54.21) |
| Sat 11-07 14:42 | `5ba83b9` | Sai | test(sai): record M9b peer-gated episode changepoints and verdict |
| Sat 11-07 14:43 | `130bc9a` | Sai | test(sai): register L2 as base B4 and queue rebase ids |
| Sat 11-07 14:43 | `ed4c60e` | Sai | test(sai): record W1 lag-structure methods L1-L9; L2 becomes the new base |
| Sat 11-07 14:44 | `40af532` | Sai | test(sai): record Codex E1-E5/S1 verdicts and add rebase note for B4 |
| Sat 11-07 14:45 | `99cd3f8` | Pranav | docs(model): document the COMBINED_SOLUTIONS size change in MODEL_PROMPTS |
| Sat 11-07 14:45 | `180db2c` | Sai | test(sai): complete W3 and W4 experiment evidence |
| Sat 11-07 14:47 | `4e5ba56` | Sai | test(sai): record Codex S1-S8 verdicts; none adds to B4 |
| Sat 11-07 14:49 | `8ed0466` | Sai | test(sai): record A9 severity model and size rebase on B4 |
| Sat 11-07 14:50 | `851fbab` | Sai | test(sai): record memory rebase on B4; only M9b survives |
| Sat 11-07 14:51 | `fbe32d0` | Sai | test(sai): record C1 (severity + alpha) and C2 (M9b win memory + C1 sizes) |
| Sat 11-07 14:52 | `b1147f7` | Sai | test(sai): add W5 stack (win and size members) and dev OOF result |
| Sat 11-07 14:53 | `fd7b5d8` | Sai | test(sai): record L10 payout-level win features (neutral) |
| Sat 11-07 14:53 | `232f850` | Zhill | feat(model): blend rule model with the team pair mixer (54.48 validation) |
| Sat 11-07 14:54 | `d00d1f6` | Sai | test(sai): score W5 stack once on the audit set and write final summary |
| Sat 11-07 14:57 | `c954948` | Avyukth | docs: summarise House Edge main-round experiment results |
| Sat 11-07 14:59 | `62195cc` | Avyukth | docs: update experiment summary with verified E6 and E7 results |
| Sat 11-07 15:03 | `4fe7ba8` | Zhill | perf(model): lower rule-model blend weight to 0.2 after independent check |
| Sat 11-07 15:17 | `a95262d` | Avyukth | docs: record E2 matched C=10 result |
| Sat 11-07 15:17 | `2145c93` | Avyukth | docs: record final E6+E7 blend entry in experiment summary |
| Sat 11-07 15:24 | `08898f3` | Avyukth | docs: expand experiment results into a full handoff for future agents |
| Sat 11-07 16:09 | `bf6112b` | Sai | test(sai): rebase HistGB residual learner on B4 |
| Sat 11-07 16:14 | `8a86194` | Sai | test(sai): record boosted residual learner grids on B4 |
| Sat 11-07 16:16 | `5c5bdd4` | Sai | test(sai): record B4 soft-target distillation and summary |
| Sat 11-07 16:22 | `72f3a75` | Sai | test(sai): pull and re-score teammates' models; theirs is the best found |
| Sat 11-07 16:25 | `ce54376` | Sai | test(sai): run Avyukth E7 neural model and final blend through the harness |
| Sat 11-07 16:29 | `64edf9d` | Avyukth | docs: add deep-dive handouts for our experiment branches |
| Sat 11-07 16:31 | `6a8098c` | Pranav | docs(model): record round-2 combination work and prompts in MODEL_PROMPTS |
| Sat 11-07 16:31 | `b57b8fd` | Pranav | perf(model): predict payout size as a two-mode switch (54.90) |
| Sat 11-07 16:31 | `1faf993` | Pranav | test(model): score tuning candidates on validation's tuning/audit halves |
| Sat 11-07 16:36 | `260943d` | Sai | test(sai): record volatility modeling V1-V3 (neutral) |
| Sat 11-07 16:37 | `fd2b09a` | Sai | test(sai): record G1-G3 blends and full team stack on audit |
| Sat 11-07 16:39 | `5b71842` | Avyukth | docs: add House Edge additional research notes (20 papers) |
| Sat 11-07 16:39 | `ea3eefa` | Sai | test(sai): re-score Pranav's two-mode size mixer; new best 54.77 dev / 55.90 audit |
| Sat 11-07 16:39 | `08137b1` | Pranav | docs: move release and Databricks details out of always-loaded AGENTS.md |
| Sat 11-07 16:43 | `9673444` | Sai | test(sai): compare team ensembles and volatility memory |
| Sat 11-07 16:45 | `d07008f` | Sai | test(sai): record Q1/Q3 log-odds random-walk filter (no standalone skill) |
| Sat 11-07 16:45 | `24bfc3f` | Sai | test(sai): record K1 combination search; BEST is saturated |
| Sat 11-07 16:47 | `6c06e39` | Sai | test(sai): record K2 small-subset two-stage blends and gates (neutral) |
| Sat 11-07 16:49 | `cfc44ef` | Sai | test(sai): record Q2 and summarise saturation of blending in FINAL.md |
| Sat 11-07 16:52 | `81d4fff` | Pranav | docs(model): record round-3 win-prior work and prompts in MODEL_PROMPTS |
| Sat 11-07 16:52 | `39e5757` | Pranav | perf(model): pull win cells toward dead/live levels with a Beta-mixture prior (55.34) |
| Sat 11-07 16:52 | `819a334` | Pranav | test(model): compare tuning candidates against a chosen base model |
| Sat 11-07 17:02 | `c70f735` | Sai | test(sai): consolidate Codex branch into master and record its B4 results |
| Sat 11-07 17:07 | `4f8aca0` | Sai | feat(combo): copy pair mixer v3 and rule model sources from teammates |
| Sat 11-07 17:11 | `df1ffca` | Sai | feat(combo): add 20% rule + 80% pair mixer v3 entry |
| Sat 11-07 17:13 | `b678bca` | Sai | build(combo): add script that stages and zips the combo entry |
| Sat 11-07 17:14 | `a79aa0a` | Sai | test(combo): add reproducible verification and one-shot audit guard |
| Sat 11-07 17:16 | `3bdf336` | Zhill | feat(model): add auto_tune.py to re-pick the blend per round |
| Sat 11-07 17:37 | `91f6611` | Sai | test(combo): add bounded nested team combination search |
| Sat 11-07 17:38 | `3f813f2` | Sai | test(combo): score new teammate members on dev |
| Sat 11-07 17:40 | `fd8f7dd` | Zhill | feat(model): use pair mixer v3 in the combined entry (55.58 validation) |
| Sat 11-07 17:42 | `55a50d4` | Sai | test(combo): preserve distinct deployments and isolated dev replays |
| Sat 11-07 17:45 | `93f234a` | Sai | test(combo): document frozen top ten and one-shot audit |
| Sat 11-07 17:45 | `333c2bd` | Sai | test(combo): freeze top ten distinct team forecasts on dev |
| Sat 11-07 17:49 | `a373345` | Sai | docs(combo): add plain-English summary and improvement results to top ten |
| Sat 11-07 17:49 | `e73ecb5` | Sai | test(combo): cross-fit the validation-derived priors of mixer v3 and v4 |
| Sat 11-07 17:49 | `c28334e` | Sai | test(combo): record model-level fixes for short history and jackpot wins |
| Sat 11-07 17:49 | `8449266` | Sai | test(combo): record post-hoc gated blends and recalibration on the v3 combo |
| Sat 11-07 17:59 | `4d4dd9b` | Zhill | feat(model): optional live/dead win prior for the rule model |
| Sat 11-07 18:04 | `1602ae1` | Zhill | docs(model): log neighbour-rule and trend-extrapolation attempts |
| Sat 11-07 18:16 | `6ae9a8d` | Zhill | test(model): score validation on every quoted slice, incl. Sai's 24 |
| Sat 11-07 18:17 | `eef17aa` | Sai | test(boost): record signal prior and peer-triggered forgetting (no gain) |
| Sat 11-07 18:17 | `4d1fcc4` | Sai | test(boost): record gapped-history fixes for mixer v4 (no gain) |
| Sat 11-07 18:23 | `5e58ddd` | Sai | docs(sai): summarise mixer v4 as best model and where the error remains |
| Sat 11-07 18:28 | `77f6fd1` | Avyukth | feat(eval): add method-agnostic evaluation suite (#19) |
| Sat 11-07 18:33 | `27d776d` | Pranav | perf(model): per-type win prior and learned size lags, honest eval (56.12) |
| Sat 11-07 18:36 | `255e3b6` | Sai | test(win): add train-fitted Bayesian Walsh logistic win heads |
| Sat 11-07 18:38 | `952145d` | Sai | test(win): record negative logistic win-function experiment |
| Sat 11-07 18:39 | `f71ffca` | Sai | test(win): verify fixed payout shares in every saved forecast |
| Sat 11-07 18:47 | `a8bb60e` | Pranav | docs(model): record the rejected rule-model blend in MODEL_PROMPTS |
| Sat 11-07 18:48 | `a50b588` | Sai | docs: add self-contained research brief for win-probability methods |
| Sat 11-07 18:55 | `dec062a` | Avyukth | docs(research): add master research branch with all findings |
| Sat 11-07 19:04 | `948c4ee` | Avyukth | docs(research): archive history of branches being deleted |
| Sat 11-07 19:10 | `362d5b5` | Sai | docs(sai): record mixer5 as new best and Gemini research outcomes |
| Sat 11-07 19:10 | `293fcaf` | Sai | test(round): kill-test screened Walsh-logistic BMA win expert |
| Sat 11-07 19:10 | `3d991a8` | Sai | test(round): record exact 8-hypothesis size switch on mixer5 (loses) |
| Sat 11-07 19:10 | `20b5cd2` | Sai | test(round): score Pranav's mixer5 in the 20/80 rule blend |
| Sat 11-07 19:12 | `814cb60` | Pranav | perf(model): add a 100-spin win memory and slower size re-weighting (56.19) |
| Sat 11-07 19:16 | `ff262b9` | Pranav | docs(model): summarise every error-lowering step and the error map |
| Sat 11-07 19:33 | `8ab8586` | Pranav | perf(model): per-type middle level in the win prior (56.29) |
| Sat 11-07 19:37 | `c12b92b` | Sai | test(round): pin size-switch comparisons to the canonical mixer5 file |
| Sat 11-07 19:37 | `25b7367` | Sai | test(round): diagnose jackpot win probability (no supported fix) |
| Sat 11-07 19:37 | `81b8235` | Sai | test(round): search for the size-mode rule on train (no better family) |
| Sat 11-07 19:37 | `1dd88f9` | Sai | test(round): score Pranav's mixer6 alone and in rule blends |
| Sat 11-07 19:40 | `06c7883` | Pranav | docs(model): record round-7 jackpot and size attempts (no model change) |
| Sat 11-07 19:58 | `fb5636a` | Pranav | perf(model): win prior levels per machine type and signal (56.32) |
| Sat 11-07 20:02 | `e5ffc49` | Zhill | feat(model): add hidden-rule HMM and blend it 50/50 (56.28 validation) |
| Sat 11-07 20:18 | `59ad70b` | Pranav | perf(model): blend in a hidden-rule HMM win chance (56.51) |
| Sat 11-07 20:35 | `5d373f5` | Pranav | docs(entry): describe the HMM blend and data use in the entry README |
| Sat 11-07 22:22 | `8b57f9c` | Pranav | feat(model): experimental change-point regime model and 3-model ensemble |
| Sat 11-07 22:24 | `6f471e4` | Zhill | feat(model): build the entry on mixer6 plus our rule model and HMM (56.59) |
| Sat 11-07 22:33 | `bbad402` | Zhill | feat(entry): print one overall summary at the end of make_entry.py |
| Sat 11-07 23:12 | `d8ff0a2` | Pranav | feat(model): research harness and log-odds pooled ensemble (56.54) |
| Sat 11-07 23:43 | `5f7bdba` | Pranav | test(model): population priors for expert weights (no reliable gain) |
| Sat 11-07 23:55 | `52bc7d7` | Pranav | feat(model): peer correction found by residual analysis (56.59) |
| Sun 11-08 00:03 | `9dee55b` | Pranav | test(model): per-machine residual hunts, surprise tempering, ceiling (no gain) |
| Sun 11-08 00:12 | `00d0d9d` | Zhill | feat(model): add size-rule HMM for payout size (validation 57.32) |
| Sun 11-08 00:14 | `a5c53eb` | Pranav | feat(round2): one-command adaptation pipeline (build, compare, verify, stage) |
| Sun 11-08 00:20 | `5988955` | Pranav | test(model): per-lag and short-history peer refinements (not adopted) |
| Sun 11-08 00:24 | `77a9b25` | Pranav | test(model): wider residual families, joint pool/peer grid, middle-level HMM (no gain) |
| Sun 11-08 00:29 | `2f5bba3` | Avyukth | feat(eval): add --jobs to run independent entry executions in parallel (#20) |
| Sun 11-08 00:31 | `e6117ba` | Pranav | test(model): give the fake mixer in the plateau test the real mixer's attributes |
| Sun 11-08 00:31 | `b83b1a6` | Pranav | feat(model): v11 plateau average of three near-equivalent win models (56.61) |
| Sun 11-08 00:34 | `9481d4c` | Avyukth | feat(model): v7c size-atom rule, episode flip swap, round-two guards |
| Sun 11-08 00:35 | `8725bca` | Avyukth | fix(round2): make round2.sh and RUNBOOK run from this branch layout |
| Sun 11-08 00:47 | `29866e8` | Sai | feat(eda): quantify validation temporal structure |
| Sun 11-08 00:49 | `60ee751` | Sai | feat(model): add causal self-supervised global learning |
| Sun 11-08 00:57 | `44cd619` | Sai | feat(model): add empirical-bayes latent-state experiment |
| Sun 11-08 00:59 | `a70abc0` | Pranav | feat(model): v13 strongest build - our win model + size-rule HMM + v7c (57.52) |
| Sun 11-08 01:00 | `f26bec5` | Sai | test(orch): measure mixer6 error split and size-mode and signal ideas |
| Sun 11-08 01:01 | `35fd9fb` | Sai | feat(model): compare calibrated trees and neural forecasts |
| Sun 11-08 01:01 | `1185541` | Sai | test(orch): cross-fitted blend of mixer6 with the fresh HGB model |
| Sun 11-08 01:02 | `79052f0` | Sai | fix(eda): correct win and bootstrap definitions |
| Sun 11-08 01:02 | `8b4d215` | Sai | test(orch): per-machine size-mode rule recovers the true mode 66% of the time |
| Sun 11-08 01:06 | `2444e16` | Sai | docs(model): record independent refit and resource checks |
| Sun 11-08 01:09 | `b3e956f` | Sai | feat(model): add learned binary payout-size modes |
| Sun 11-08 01:19 | `113d3c2` | Sai | docs(sysid): record validation results and research handoff |
| Sun 11-08 01:19 | `d5df589` | Sai | feat(sysid): freeze and cross-fit the research forecaster |
| Sun 11-08 01:19 | `40bad08` | Pranav | feat(model): v14 - v13 with the full peer effect restored, checked on held-out train |
| Sun 11-08 01:19 | `437362f` | Sai | feat(sysid): infer finite rules and reversal states |
| Sun 11-08 01:19 | `4950b48` | Sai | feat(sysid): add independent generator diagnostics |
| Sun 11-08 01:21 | `77af0a8` | Sai | feat(model): stack causal mixer forecasts with global corrections |
| Sun 11-08 01:30 | `cf69862` | Sai | test(exact): fit Bayesian episode posterior |
| Sun 11-08 01:32 | `c90ba95` | Sai | feat(orch): truth-based level priors, all-history pair census, held-out-tick scorer |
| Sun 11-08 01:32 | `be6db30` | Pranav | feat(model): v15 - signal term on the blended win chance (57.55) |
| Sun 11-08 01:33 | `cac58d0` | Sai | feat(sysid): cross-fit continuous win-level forms |
| Sun 11-08 01:33 | `e602b5d` | Sai | test(exact): apply measured fold priors |
| Sun 11-08 01:35 | `9550748` | Sai | feat(model): marginalize missing short-history lag bits |
| Sun 11-08 01:36 | `9a0faf8` | Sai | test(orch): equal log-odds blend of v13, exact filter and sysid wins 57.86 |
| Sun 11-08 01:37 | `9569503` | Pranav | test(model): post-v15 attempts - short-history gates, regime member, size-flip detection (no gain) |
| Sun 11-08 01:37 | `ff95e3f` | Sai | feat(model): export short-history cross-fit patch |
| Sun 11-08 01:42 | `6f0b3f4` | Sai | feat(sysid): cross-fit type-specific win priors |
| Sun 11-08 01:43 | `cc2fe5c` | Sai | docs(sysid): prepare handoff for next win model test |
| Sun 11-08 01:43 | `b25bc44` | Sai | docs(sysid): state continuous win probability form |
| Sun 11-08 01:50 | `624506d` | Pranav | test(model): per-type blend weights and size-flip detectability (no gain) |
| Sun 11-08 01:52 | `e0cc475` | Sai | docs(orch): record the 57.91 win-model blend with reproducible inputs |
| Sun 11-08 01:54 | `b900879` | Sai | feat(jackpot): patch v13 win rate with four-cell jackpot rates |
| Sun 11-08 02:00 | `38b46c1` | Sai | feat(jackpot): restrict pairs and cross-fit truth level priors |
| Sun 11-08 02:02 | `8d878aa` | Sai | docs(orch): add the jackpot win patch to the blend (57.97) |
| Sun 11-08 02:05 | `3f842a9` | Avyukth | docs(research): add late-night experiment results and round-two warnings |
| Sun 11-08 02:07 | `7e1c1fc` | Pranav | feat(model): v16 - transfer v7c's episode flip swap to our win model (57.59) |
| Sun 11-08 02:08 | `c11895b` | Sai | feat(winform): measure the cutoff win-probability form |
| Sun 11-08 02:08 | `8e7588b` | Pranav | test(model): residual hunt on v16 is clean on both splits |
| Sun 11-08 02:12 | `16b2732` | Avyukth | docs(research): flip-switched v15 blend and the cause of v7c's preset losses |
| Sun 11-08 02:12 | `c826f20` | Sai | feat(winform): export train-prior forecast with v13 sizes |
| Sun 11-08 02:12 | `c2b8000` | Sai | test(orch): paired episode significance for every blend step |
| Sun 11-08 02:14 | `a5ea7f9` | Sai | feat(exact): learn the pair set from the kit's histories |
| Sun 11-08 02:15 | `2362337` | Sai | feat(orch): round-two structure screen over single lags, pairs and triples |
| Sun 11-08 02:16 | `83317ac` | Avyukth | feat(bench): add simulated round-two adaptability benchmark (#21) |
| Sun 11-08 02:17 | `435ab14` | Sai | test(merge): compare exact and sysid reversal posteriors |
| Sun 11-08 02:17 | `e82554f` | Sai | docs(orch): record round-two checks and the generator-proxy caveat |
| Sun 11-08 02:17 | `00b47d6` | Zhill | feat(model): learn the size-rule prior from each round's train (EM) |
| Sun 11-08 02:17 | `1096601` | Sai | test(round2): score v13 on a simulated shifted round |
| Sun 11-08 02:20 | `bfaf5a0` | Sai | docs(orch): v16 as a blend member adds only noise (+0.01) |
| Sun 11-08 02:22 | `fd65593` | Pranav | feat(model): super branch with v17 - jackpot sub-cell correction (57.72) |
| Sun 11-08 02:25 | `1851e9f` | Sai | fix(merge): use corrected independent level priors |
| Sun 11-08 02:30 | `ee1c180` | Sai | test(merge): measure robust dead-cell hypothesis weights |
| Sun 11-08 02:38 | `87e4233` | Avyukth | fix(model): stop v7c's round-two guards losing to v6 on shifted rounds |
| Sun 11-08 02:55 | `a2e5d86` | Zhill | fix(model): carry avyukth's v7c round-two guards into v17's vendored copy |
| Sun 11-08 03:06 | `3cc74b3` | Sai | docs(orch): choose v17-guards as the blend's first member for v14 |
| Sun 11-08 03:19 | `2730597` | Avyukth | feat(model): Sai's blend adapted to learn lag pairs per round (57.96) |
| Sun 11-08 03:34 | `278efff` | Zhill | feat(model): v18 backup entry - v17 guards blended with Sai's exact filter (57.95) |
| Sun 11-08 03:41 | `5b1d090` | Sai | feat(round2): one-command round-two build and ranking |
| Sun 11-08 03:43 | `3083c87` | Avyukth | feat(model): swap the adapted blend's first member to v17-guards |
| Sun 11-08 03:49 | `ee570b1` | Pranav | feat(model): v18 - round-adaptive add-on gates and v7c round-two guards (57.72) |
| Sun 11-08 03:55 | `23ac6de` | Avyukth | fix(round2): make run_round2.py portable across machines |
| Sun 11-08 03:56 | `dd23de5` | Sai | feat(round2): default to the adapted v17g blend with a 0.1 challenger margin |
| Sun 11-08 04:08 | `1edf5a8` | Zhill | feat(round2): round-adaptive member gate for the adapted blend |
| Sun 11-08 04:13 | `c2eb7e3` | Avyukth | feat(model): gate v7c in as a 4th blend member for mixed-rule rounds |
| Sun 11-08 04:17 | `979d2d9` | Avyukth | feat(model): exact filter learns 3-lag XOR and AND/OR rules per round |
| Sun 11-08 04:32 | `148199b` | Avyukth | fix(model): don't crash when validation targets are absent |
| Sun 11-08 04:34 | `15241ec` | Pranav | perf(model): v19 - 81 s entry, build-time validation gates, v7c in a 2nd process (57.74) |
| Sun 11-08 04:38 | `5467b72` | Sai | feat(round2): default to the rules-learning adapted blend |
| Sun 11-08 04:46 | `5b90a79` | Avyukth | docs(round2): README names the rules entry as the default blend |
| Sun 11-08 04:48 | `8931a6c` | Zhill | feat(round2): gate + bundled validation answers on the rules blend |
| Sun 11-08 05:09 | `5c873dc` | Sai | fix(round2): bundle the whole validation folder, not just its answers |
| Sun 11-08 05:18 | `5c1349b` | Sai | docs(round2): kit-day steps for the gated rules blend and its code ZIP |
| Sun 11-08 05:21 | `a4b5448` | Avyukth | fix(model): restore the saved-priors guard in adapted_entry's combo.py |
| Sun 11-08 05:39 | `0d45f25` | Pranav | feat(model): v20 super entry - Sai's blend plus v19 as a 4th member (58.00) |
| Sun 11-08 06:15 | `3e5f208` | Pranav | fix(model): v20 runs identically with no validation folder mounted |
| Sun 11-08 09:16 | `80f6b8c` | Sai | chore(round2): record the round-one upload and its gate.json |
| Sun 11-08 10:31 | `44c506a` | Sai | docs: README, changelog and playbook for the round-two model |
| Sun 11-08 10:35 | `ad704ad` | Sai | chore(round2): add the adaptation-round participant kit |
| Sun 11-08 10:40 | `9e86c27` | Sai | chore: add the refreshed main-round participant kit (60/40 rules) |
| Sun 11-08 10:44 | `f5ec781` | Avyukth | chore(round2): record adaptation gate (38.59/40) and structure census |
| Sun 11-08 10:53 | `720ad98` | Avyukth | chore(round2): record blend tweak check (no change) and adaptation ZIP hash |
| Sun 11-08 10:54 | `73b1ad1` | Sai | docs: describe the real size member and the two organizer kits |
| Sun 11-08 10:56 | `5378493` | Sai | chore(r2): record kit40 member gate |
| Sun 11-08 10:58 | `64946b2` | Sai | chore(r2): record adaptation gate and shortwin result |
| Sun 11-08 11:00 | `bd094d5` | Avyukth | chore(round2): record per-type, headroom and train+val checks (no change) |
| Sun 11-08 11:01 | `34fa9bf` | Sai | docs(r2): identify the active v7c size source |
| Sun 11-08 11:02 | `cd2f1fb` | Sai | docs: correct the round-two structure note (pairs only, no 3-lag rules) |
| Sun 11-08 11:06 | `fcc5ad5` | Sai | docs(round2): hand off short-history pair identification task |
| Sun 11-08 11:06 | `b5e80ba` | Sai | feat(blend): fit round-two condition weights |
| Sun 11-08 11:07 | `1dfca72` | Sai | docs(blend): correct first member description |
| Sun 11-08 11:07 | `41f5043` | Sai | docs(round2): add member-by-condition scores and failed tweaks to the task |
| Sun 11-08 11:07 | `7b89c64` | Pranav | fix(round2): rediscover sub-cell lag pairs per round; verify with the official command |
| Sun 11-08 11:10 | `a31e3e6` | Sai | feat(short-history): fit cell win priors on train |
| Sun 11-08 11:11 | `02070ec` | Avyukth | chore(round2): record short-history calibration, 4th-member and Docker checks |
| Sun 11-08 11:11 | `b1fff0f` | Sai | docs(r2): correct adapted member source description |
| Sun 11-08 11:19 | `6f82662` | Avyukth | docs(devpost): add project writeup and submission checklist |
| Sun 11-08 11:20 | `28de957` | Sai | docs(devpost): correct the size member and add final round-two diagnostics |
| Sun 11-08 11:23 | `3b75c1f` | Sai | fix(short-history): temper cell-rate prior spread |
| Sun 11-08 11:27 | `ef74e22` | Avyukth | chore(round2): record peer lag-3 win correction (+0.06/+0.11, below rule) |
| Sun 11-08 11:27 | `0b6fe1f` | Sai | feat(exact): add short-history rule identity prior |
| Sun 11-08 11:28 | `e7646f5` | Sai | revert(exact): remove unsupported triple prior |
| Sun 11-08 11:28 | `06158c9` | Sai | docs(devpost): record the final round-two experiments and the upload decision |
| Sun 11-08 11:29 | `154abac` | Sai | chore(round2): record the adaptation-kit gate.json |
| Sun 11-08 11:32 | `a80ddc6` | Avyukth | chore(round2): add the round-two upload files (CSV, code ZIP, hashes) |
| Sun 11-08 11:38 | `5419bb1` | Sai | docs(round2): use the full CSV hash in the round-two upload record |
| Sun 11-08 11:38 | `cccf8d5` | Sai | docs(round2): record the round-two upload and the cross-check of both builds |
| Sun 11-08 11:38 | `2500225` | Sai | docs(devpost): name Avyukth's pair as the round-two upload, with the cross-check |
| Sun 11-08 11:40 | `3740c54` | Avyukth | docs(devpost): add submissions record, research reports and round-two results |
| Sun 11-08 11:40 | `7968b04` | Pranav | feat(model): v21 round-two entry - peer term on the super blend (38.67/40) |
| Sun 11-08 11:41 | `8215f80` | Sai | docs(devpost): add Sai's full iteration history and reasoning |
| Sun 11-08 11:48 | `3f504ec` | Sai | docs(devpost): index the iteration history, add Pranav's v21 closest miss, tick the write-up step |
| Sun 11-08 11:49 | `e7ece1a` | Sai | docs(devpost): neutral wording in the adaptation credit line |
| Sun 11-08 11:49 | `cce7013` | Sai | docs(devpost): credit Avyukth and Zhill for the round-two adaptation work |
