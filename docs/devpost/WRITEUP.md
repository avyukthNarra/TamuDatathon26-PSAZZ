# House Edge: slot-forecasting for TAMU Datathon 2026 (team PSAZZ)

## Inspiration

House Edge asks for a forecast of the next spin of hidden-rule slot machines: for every test query, the probability that the machine pays nothing, and the distribution of the payout size given a win. The rules are hidden, but the history of each machine is in the data. We wanted a forecaster that finds the structure in that history rather than fitting a generic sequence model to it.

## What it does

For each test query, the entry outputs P(no win) and P(payout size | win). The final model (`adapted_entry_v17g_rules/`) is an equal log-odds blend of three win-probability models:

1. **v17-guards:** Pranav's v17 ensemble combined with Avyukth's v7c pair mixer (win weight 0.3, Zhill's round-two guards), plus Sai's short-history and jackpot patches. In the code it is still called `members/v13_short.py`.
2. **Exact Bayesian filter:** an exact Bayesian filter over lag-pair XOR rules learned from the training data. It also learns 3-lag XOR and 2-lag AND/OR rules.
3. **System-identification model:** a sparse rule library fitted to the machines' history.

Payout size uses the v17-guards member's size split, with a 1e-4 floor. A build-time member gate (`r2/gate_members.py`) scores the blend with and without each member on that round's cross-fitted validation and drops a member only if removing it gains more than 0.1 points; the result is saved in `gate.json` and shipped with the entry. On both rounds it kept all three members.

Every member learns its own lag pairs from the current round's training data, so nothing is hard-coded to the main round.

## How we built it

- The three members were compared on the same cross-fitted validation before they were blended.
- The blend was built with an equal log-odds average, so no member dominates.
- We wrote a simulated adaptability benchmark of 10 kits. It checks that the model still works when the rules change, and it is what we used to judge whether an adaptation was safe.
- Round-two work is a single command: `r2/run_round2.py` screens the new kit's lag structure, builds every candidate, scores each one on the kit's validation, and recommends what to upload. A challenger replaces the default only if it beats it by more than 0.1 on the mean and on both cross-fit halves.
- The result is reproducible: it runs offline on 2 CPUs in about 3–5 minutes, and the output is byte-identical with or without the validation answers mounted.
- Python 3.13, numpy and scipy for the submitted entry; scikit-learn, LightGBM and pandas for research.

## Challenges we ran into

- **Hidden structure is easy to overfit.** Lag pairs found on one dataset can be wrong on another, so each member relearns its lag structure from the round's own training data.
- **Validation is small.** We used cross-fitting so the reported score is not from the same data used to choose the blend. The honest 4-fold figure is lower than the cross-fitted figure, and we report both.
- **Payout size was harder than win probability.** Analysis of the earlier models showed that the size head carried a large share of the remaining error, and the size rule needed its own learned component.

## Accomplishments we're proud of

- Main-round validation of **57.97 / 60** (cross-fitted; halves 57.98 and 57.96).
- Honest 4-fold score of **57.86** on the Docker eval suite, with 0 FAIL.
- A round-two pipeline that needs one command and produces a ranking with a recommended upload.
- Reproducible output, with no dependence on whether validation answers are present.

## What we learned

- Simulator-style inversion can beat a generic model when the episode is stable, but it is weaker when history is short or the change is gradual. Combining the two needs a per-episode decision, which is why we kept the blend rather than replacing the model.
- Generic sequence learners did not beat a model built on the known lag structure.
- A learned, validated gate is a cheap guard against a member that hurts on a new round.

## What's next

- Improve the short-history case, where the blend is weakest: identify each short machine's lag pair with fewer spins (for example from the payout-size channel, whose lags are the smaller member of each win pair), and shrink short machines' size split toward type-level sizes.
- Tighten the payout-size rule for the largest-pot cases.
- Model the cross-machine signal directly: in round two, the share of an episode's other machines that won 3 spins ago predicts a machine's win probability (+0.06 / +0.11 on the two validation halves; no other lag helps).

## Round two (adaptation)

- **What changed:** the win lag pairs moved from (1,3), (2,5), (4,7) to **(1,4), (2,6), (3,8)** (pattern (k, 2k+2)); the payout-size lags moved from {1,2,4} to **{1,2,3}** (the smaller member of each pair). Rules stayed pairwise XOR/XNOR: no 3-lag or AND/OR rules (188/192 stable validation machines fit a 2-lag XOR). 0% of machines kept an old pair.
- **Adaptation:** no code changes. Every member relearned the pairs and size lags from the round's training data (pair census on train: (3,8) 185, (1,4) 180, (2,6) 164 confident machines), and the member gate kept all three members.
- **Score:** cross-fitted validation **38.59 / 40** (condition-balanced halves 38.53 / 38.63). Per-condition loss ratios: stable 0.031, shared change 0.031, drift 0.036, short history 0.049.
- **What we tried and rejected** (each fitted on one validation half and checked on the other; a change had to gain more than 0.1 on both halves): global and per-type blend weights, logit temperature, short-history-specific weights, other members' size splits, adding v19 or v7c as a fourth member, Pranav's v20 (38.58), and adding validation episodes to train (-0.007 / -0.003, the members are data-saturated). Headroom analysis: perfect win probabilities would give 39.63 and perfect sizes 38.93, so most of the remaining error is hidden win state.
- **Where the error is (swap-in-truth, official metric):** short history's unusual gap is payout size: true sizes cut its loss ratio from 0.049 to 0.0315 (-0.0175), about 3x more than on any other condition. Perfect short-history predictions alone would add at most +0.29 (win), +0.16 (sizes), +0.46 (both), so the remaining headroom is small. The round's pairs are found cleanly, but only 529 of 768 validation machines get a confident per-machine pair; the rest are averaged over the three pairs.
- **Members alone vs blend (adaptation validation /40):** sysid 38.54, v17-guards 38.33, exact filter 38.31; the blend 38.59 beats each. All three are weakest on short history (about 37.8 vs 38.5-38.9 elsewhere).
- **Also tried in the final hour (no gain):** 3-lag rule priors for short history (-0.001), empirical-Bayes pooling of short machines' win levels toward type means (-0.0003), per-condition blend weights with temperature (-0.09), and Pranav's subcell pair fix (identical, because the short-history patch already overrides those queries).
- **Final experiments (11:00-11:28, none cleared the 0.1-on-both-halves bar):** a train-only prior over pair identities for short-history machines in the exact filter (+0.022 overall, but -0.006 on one half); logit-normal pooling of short machines' win levels with the spread chosen on one half (+0.008, positive on both halves); self-supervised recalibration of short-history P(win) on masked train histories (-0.022; the fitted slope was about 1.1, so it was already well calibrated); type-level shrinkage of short-history payout sizes (worse at every weight); size-channel pair evidence (+0.0006); a peer lag-3 win correction (+0.06 / +0.11). **Uploaded: the default blend.**
- **Reproducibility:** the organizer-style Docker rerun (python 3.13, pinned numpy/scipy, offline, 2 CPUs) matches our CSV to 9.7e-15; 4.8 minutes locally.
- **Kit note:** the adaptation kit shipped the organizers' older 80/20 scoring library, which our newer 60/40 tooling rejects; we scored on a copy with `max_points` 40, which rescales points without changing any ranking.
- **Files uploaded:** `ADAPTATION_submission.csv` (SHA-256 `a7ec520d…190f`) and `ADAPTATION_entry.zip` (`feb33968…29d2`); two independent builds agree to 3.2e-14 with identical ZIP code (see `SUBMISSIONS.md`).

## Built with

Python 3.13, numpy 2.3.4, scipy 1.16.3, scikit-learn, LightGBM, pandas (research only), Git and GitHub, Docker (eval suite), and the organizers' `check_submission.py` and `build_entry.py` tools.
