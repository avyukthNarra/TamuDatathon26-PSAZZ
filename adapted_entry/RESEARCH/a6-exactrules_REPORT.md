# Exact filter: 3-lag XOR and 2-lag AND/OR

**Accept.** Main stays 57.971 (halves 57.98 / 57.96). K10 blend rises from 32.080 to 35.375. No other kit moves by more than 0.001.

Base is `a4-v17member/variant`: equal log-odds of v17-guards (short and jackpot patches), the exact Bayesian filter, and sysid, with v13's size split. This folder's `entry/` is that blend with the exact member extended. Patches: `model.py.patch`, `exact_filter.py.patch`.

## What changed

The exact filter's per-machine hypotheses were pair parity only (2-lag XOR/XNOR, both directions). They now also include 3-lag XOR (even parity of three lags) and 2-lag AND and OR, with the same direction flip and the same dead/live level grid.

Every constant comes from that round's train:

- Lag pairs are still `learn_pairs` (share 0.05, margin 2 nats, 5 counts per cell).
- A census scores parity, 3-lag XOR, AND, and OR on the first 40% of each train machine (before drift and the shared flip). A machine votes only when its best family beats the runner-up by 2 nats and its best rule beats the next rule in that family by 2 nats.
- A family with fewer than 5% of machines gets mass 0 and is left out of the mixture. The predict path is then the published pair filter, not a tiny extra prior.
- Kept-family mass is that family's share of decisive votes. Inside AND / XOR3 / OR, mass is split by vote counts. The parity family's internal weights stay the existing EM pair prior, rescaled by the parity mass.

On this benchmark only K10 clears the cut: parity 0.744, 3-lag XOR 0.126, AND 0.130, OR 0.000. The kept rules are 15 triples and the three AND pairs `(1,3)`, `(2,5)`, `(4,7)`, each with its train vote count in `runs/K10/predict.log`. Main and K1–K9 vote a handful of stray AND/XOR/OR machines (at most 3 of 384, or 3 of 768 on main), all under 5%, so those families are omitted.

Checked against K10's latents (research only; the model never reads them): every AND or XOR3 vote named the true rule. Recall is partial because short-history machines and close margins abstain. That is why the prior is conservative (about 74% parity against a 71% parity generator) and why some machines still fall back to pair parity.

## Main (out of 60)

Same cross-fit as `a4-v17member`: validation split, episode halves alternating within condition, `s3-final/score_pred.py`.

| Forecast | Points | Halves | Wall |
|---|---:|---|---:|
| Before, a4 blend | 57.971 | 57.98 / 57.96 | 145s there |
| After, this entry | **57.971** | **57.98 / 57.96** | 179s |

Train census: parity mass 1, extras 0. Exact alone is also unchanged: flat-size file 43.803, and 57.661 after the v13 size split, both identical to the a4 files.

## Kits (blend, out of 40)

Scored with `a3-saiadapt/score_one.py`. Before is the a4 variant column.

| Kit | Before | After | Diff |
|---|---:|---:|---:|
| K1 main-like | 38.801 | 38.801 | 0.000 |
| K2 disjoint pairs | 38.962 | 38.962 | 0.000 |
| K3 four pairs | 38.911 | 38.911 | 0.000 |
| K4 shared lag | 39.264 | 39.264 | 0.000 |
| K5 long lags | 39.195 | 39.195 | 0.000 |
| K6 static size | 39.365 | 39.365 | 0.000 |
| K7 size bit ≥ 5 | 39.098 | 39.098 | 0.000 |
| K8 late flip, fast drift | 39.131 | 39.131 | 0.000 |
| K9 shifted levels | 39.071 | 39.071 | 0.000 |
| K10 mixed 3-lag / AND | 32.080 | **35.375** | **+3.295** |

Largest K1–K9 gap versus the a4 printout is under 0.001 (rounding). K10 loss ratios, before → after: stable 0.293 → 0.147, drift 0.176 → 0.084, shared change 0.204 → 0.069, short history 0.314 → 0.223.

## Exact member alone

The raw exact file uses a flat size split, so it scores in the high 20s. The column that matches the a3 figure of 31.10 puts that member's P(win) through v13's size split and the same 1e-4 floor as the blend.

| | Flat size | With v13 size |
|---|---:|---:|
| K10 before | 25.557 | 31.095 |
| K10 after | **29.094** | **37.227** |
| Main before and after | 43.803 | 57.661 |

K10 exact with the v13 size split gains 6.13 points. Short history is still its weak condition (loss ratio 0.200 versus 0.012–0.063 on the other three): the census does not get a clean vote from short train machines, and missing lags stay marginalized.

## Runtime

One thread per member, three members at once. Main 179s, K10 173s, K1 114s. K10 is the only kit that evaluates the extra rules (18 of them, including the short-history state sum). On 2 CPUs the third member waits, so a round is a few minutes. That is well under 30 minutes.

## Decision

Accept. Main is 57.971 with both halves above 57.85, K10 improves, and K1–K9 do not drop.
