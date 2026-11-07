# House Edge — Our Model (rule model v4 + team pair mixer)

> **Submitted entry = combined model: 54.48 / 60 on validation.**
> It's a weighted average: **30% this rule model + 70% the team's pair mixer**
> (`slot_forecasting/pair_mixer.py`, from the `COMBINED_SOLUTIONS` branch). See [§13](#13-the-combined-model).
> Build it with `python3 make_entry.py --out runs/combined-v1`.

> **TL;DR (rule model)**
> - The slot machines follow hidden **rules**: whether a machine pays depends on whether it **won at two specific
>   earlier spins** (for example, "pays when exactly one of the spins 1 and 3 back was a win").
> - Each machine has its **own** rule. Our model tries every possible rule on each machine's own past and trusts
>   the rules that keep predicting it correctly.
> - How **much** it pays (1×/2×/5×/20×) is a second hidden switch with two modes per machine type. We handle it separately.
> - **Validation: 52.89 / 60** (starter `pooled_recent`: 30.83, official reference: 30.00).

| Version | Points / 60 | stable | short_history | drift | shared_change |
|---|---|---|---|---|---|
| pooled_recent (starter backup) | 30.83 | 0.958 | 1.001 | 0.946 | 0.880 |
| v2: win rules | 48.87 | 0.095 | 0.361 | 0.235 | 0.220 |
| v3: + separate payout size, blend | 52.31 | 0.085 | 0.209 | 0.150 | 0.143 |
| **v4: + per-machine size rules** | **52.89** | **0.068** | **0.208** | **0.130** | **0.132** |

Condition columns = our error ÷ the reference model's error. **Lower is better; 1.0 = same as the reference.**

---

## Contents

1. [What we're predicting](#1-what-were-predicting)
2. [Words you'll see](#2-words-youll-see)
3. [The data files](#3-the-data-files)
4. [How the simulator actually works (what we discovered)](#4-how-the-simulator-actually-works-what-we-discovered)
5. [How we're scored](#5-how-were-scored)
6. [The model, step by step](#6-the-model-step-by-step)
7. [What we tried that didn't work](#7-what-we-tried-that-didnt-work)
8. [How we tested it honestly](#8-how-we-tested-it-honestly)
9. [How to run it](#9-how-to-run-it)
10. [Round two](#10-round-two)
11. [Files](#11-files)
12. [Where the remaining error is](#12-where-the-remaining-error-is)

---

## 1. What we're predicting

For each slot machine we're asked: **"What are the chances of each payout on the next spin?"**
You always bet 1 chip, and there are five possible payouts:

| Payout | Meaning | You get back | Profit |
|---|---|---|---|
| **0×** | Lose | 0 chips | −1 |
| **1×** | Break even | 1 chip | 0 |
| **2×** | Small win | 2 chips | +1 |
| **5×** | Big win | 5 chips | +4 |
| **20×** | Jackpot | 20 chips | +19 |

We answer with **five chances that add up to 1**, always in the order 0, 1, 2, 5, 20:

```
[0.55, 0.20, 0.15, 0.08, 0.02]
 lose  even  small big  jackpot
```

We never pick one answer. The score compares our five chances with the machine's real hidden chances.

---

## 2. Words you'll see

| Word | Plain meaning |
|---|---|
| **Tick** | One moment in time. Each tick, a machine spins once. |
| **Cutoff** | The tick we predict. For cutoff 200 we may only use spins 0–199. |
| **Query** | One question: "machine X at cutoff Y, what are the chances?" |
| **Episode** | A group of 8 machines watched together, with the same cutoff. |
| **Lag** | How many spins ago. Lag 1 = the previous spin, lag 3 = three spins ago. |
| **Rule** | Which earlier spins a machine "looks at". Written as a pair of lags, e.g. **(1,3)**. |
| **Live / dead** | A machine that is about to pay out almost surely (live) or almost surely lose (dead). |
| **XOR** | "Exactly one of the two was a win." **XNOR** is the opposite: "both the same." |
| **True odds** | The machine's real hidden chances. Only visible in the validation data. |
| **Shrinkage / blending** | Mixing a machine's own numbers with a broader average so we don't overreact to little data. |

### The four conditions (each is 25% of the score)

| Condition | What's going on |
|---|---|
| **stable** | Machines behave the same the whole time |
| **short_history** | We only see 64 of the machine's past spins (40 recent in a row + 24 random older ones) |
| **drift** | Odds slowly change over time |
| **shared_change** | Several machines in the episode change at the same moment |

We can't see the condition for test data, so the model must work without it.

---

## 3. The data files

`public/` has three folders: **train** (build the model), **validation** (check it, includes the true odds),
and **test** (what we submit predictions for).

| File | What's in it | train | validation | test |
|---|---|:---:|:---:|:---:|
| `queries.csv` | The questions: machine, type, signal, cutoff | ✅ | ✅ | ✅ |
| `history.csv` | Every past spin: machine, tick, payout. **The model's input.** | ✅ | ✅ | ✅ |
| `labels.csv` | What the next spin actually paid (one noisy draw) | ✅ | ✅ | ❌ |
| `conditions.csv` | Which of the 4 conditions | ✅ | ✅ | ❌ |
| `targets.csv` | **The true odds**: what we're graded against | ❌ | ✅ | ❌ |
| `reference.csv` | The official reference model's predictions | ❌ | ✅ | ❌ |

Rules that are easy to get wrong:
- ⚠️ Only spins **before the cutoff** may be used, for every machine.
- ⚠️ A **missing tick is unknown**, not a 0 payout. "2 spins ago" means the row with `tick == cutoff − 2`.
- ⚠️ Conditions are never a model input (test doesn't have them).

---

## 4. How the simulator actually works (what we discovered)

### Discovery 1: machines are either "live" or "dead" on each spin

The true chance of losing is almost never in the middle. Most machines are either **live** (lose ~8%) or
**dead** (lose ~95%). Averaging a machine's past gives something in between, which is wrong both ways.
This is why all six starter models are stuck around 30 points.

### Discovery 2: live/dead is decided by win/lose at two earlier spins

For most machines, **the next spin is live when exactly one of two specific earlier spins was a win** (XOR),
or, for other machines, when both were the same (XNOR). Measured on training machines:

```
lags (1,3), XOR:   both lost → 9% win    one won → 87% win    both won → 8% win
lags (1,3), XNOR:  both lost → 90% win   one won → 11% win    both won → 85% win
```

- The same few lag pairs come up over and over: **(4,7), (2,5), (1,3)**, plus lag 7 or lag 8 alone.
  Together these cover about 80% of training machines.
- **Each machine has its own pair and its own direction.** Machines of the same type can do the opposite
  thing. This matches the organizers' hint: *"machines of the same type can respond differently, even in
  opposite directions."*
- **Why simple methods miss it:** with XOR, *each spin alone tells you nothing*. Only the combination does.
  Our first attempt (one clue per lag) scored 31 for exactly this reason.

### Discovery 3: payout SIZE is a second hidden switch

Given that a machine pays, the mix of 1×/2×/5×/20× takes **exactly one of two values per machine type**:

| Type | "Small" mode | "Big" mode |
|---|---|---|
| steady | [0.75, 0.20, 0.05, 0.00] | [0.46, 0.44, 0.11, 0.00] |
| balanced | [0.50, 0.35, 0.13, 0.02] | [0.21, 0.55, 0.21, 0.03] |
| jackpot | [0.34, 0.24, 0.31, 0.10] | [0.12, 0.32, 0.42, 0.14] |

Which mode a machine is in isn't explained by any simple shared feature (signal, episode, the win rule, time),
so we learn it per machine, the same way as the win rule (§6, Part C).

### Smaller findings

- **Neighbours matter only a little.** Machines in the same episode share a small hot/cold mood (~1–2 points)
  and don't share rules (18% same rule vs 13% by chance). We don't use neighbours.
- **`signal`** is a mild hint about generosity (strongest for jackpot machines). The starter's base forecast
  already uses it.

---

## 5. How we're scored

1. For each query: add up the squared differences between our five chances and the true five chances.
2. Average per condition, then divide by the reference model's error (1.0 = same as the reference).
3. Average the four numbers → **R**.
4. **Points = 60 ÷ (1 + R).** Main round is worth 60; round two is worth **40** (the kit's scorer says
   `ROUND_POINTS = {"main": 60, "adaptation": 40}`; the 80/20 in the guide is outdated).

| R | Points |
|---|---|
| 1.0 (same as reference) | 30 |
| 0.135 (**our v4**) | 52.89 |
| 0.09 | 55 |
| 0 (perfect) | 60 |

---

## 6. The model, step by step

**Name:** per-machine rule model. Code: `rule_model.py`. Entry point: `predict.py`.

```
 This machine's past spins (before the cutoff)
          │
          ▼
 A. Safe guess: pooled_recent (type/signal average + own spins, recent count more)
          │
          ▼
 B. WILL IT PAY?  Try every rule. Grade each on the machine's own past. Trust the good ones.
          │
          ▼
 C. HOW MUCH?    Machine's own payout-size mix → blended toward its type → refined by size rules
          │
          ▼
 D. Average two versions of B (see below), floor tiny numbers, rescale to add up to 1
          │
          ▼
    [p0, p1, p2, p5, p20]
```

### Part A: the safe guess

The starter's `pooled_recent` forecast. Everything else starts from it, so a machine with almost no
history still gets a sensible answer.

### Part B: will it pay? (the win rules)

**1. List every candidate rule.** All 28 pairs of lags from 1–8, all 8 single lags, and "no rule": 37 in total.

**2. For each rule, keep a tally.** The "situation" is win/lose at the rule's lags; the tally counts what came next.
For rule (1,3):

```
Situation (spin 1 back, spin 3 back)    What came next:  lose   win
lose, lose                                                 30     3
lose, win                                                   4    25
win,  lose                                                  3    28
win,  win                                                  27     2     ← this machine is XOR on (1,3)
```

Each tally cell starts with a tiny bit (strength 0.5) of the safe guess, so an empty cell isn't 0%.

**3. Grade each rule fairly.** Walk through the machine's history in time order. **Before** adding a spin to
the tally, ask the rule what it would have predicted for that spin and record how good the prediction was.
A rule is only ever graded on spins it hadn't seen yet, so a rule can't look good just by memorising.

**4. Short and long memory.** Every rule runs twice: once remembering everything, once letting old spins fade
(5% per tick). Machines whose behaviour changes (drift, shared_change) end up trusting the short-memory version.
Old grading evidence also fades slightly (1% per tick), so a rule that *used to* work loses trust.

**5. Head start for common rules.** Rules that fit many **training** machines, like (4,7), (2,5), (1,3), get a head
start. It's learned from `train/` each time the model runs. This matters most for short-history machines,
which don't have enough spins to pick their rule from 37 candidates on their own.

**6. Trust-weighted forecast.** Look up each rule's situation at the cutoff and read its tally. Average all
rules, weighted by how well each one has been predicting this machine.

### Part C: how much will it pay? (payout size)

The win rules decide **whether** the machine pays. Payout size is predicted separately:

1. **Machine's own mix.** Count the sizes of this machine's past wins.
2. **Blend toward its type.** Add 80 pretend wins with the type's average mix (learned from `train/`). Size
   is mostly a property of the type, so we lean on it heavily.
3. **Size rules.** Run the same rule search as Part B on the machine's *wins*: does the size of a win depend
   on win/lose or big/small payouts at earlier spins? Trust-weight the rules the same way. Only machines with
   at least 40 wins get this step; with fewer, it overreacts (it made short_history worse).

### Part D: average two versions

Part B runs in two versions that make different mistakes:
- **"full":** rules are graded on predicting the exact payout (better on stable machines)
- **"win/lose":** rules are graded on win vs lose only (cleaner signal, better on short histories)

The final forecast is the **50/50 average**. It beat both versions on the validation half we didn't tune on.

### The settings (in `predict.py`)

| Setting | Value | What it does |
|---|---|---|
| `strength` | 0.5 | How much of the safe guess goes in each tally cell |
| `forget` | (0.95, 1.0) | Short and long memory variants of every rule |
| `trust_forget` | 0.99 | How fast old evidence about a rule fades |
| `prior_weight` | 3.0 | Size of the head start for common rules |
| `size_strength` | 80 | Pretend wins of the type's size mix |
| `size_rule_strength` / `size_rule_min_wins` | 20 / 40 | Size rules: blending, and minimum wins needed |
| blend | 50/50 | full + win/lose versions |

### Why this design

- **It matches how the simulator works** (per-machine lag-pair rules, two size modes).
- **No randomness, no saved model file.** It re-learns from `train/` in about 80 seconds, deterministically.
- **Round two needs no code changes.** It learns rules and head starts from whatever data it's given.
- **Plain Python standard library**, so it runs on the organizers' Python 3.13 runner.

---

## 7. What we tried that didn't work

| Idea | Result | Why |
|---|---|---|
| One clue per lag (the original plan) | ~31 | XOR rules are invisible to single lags |
| Mixing all 8 lag clues at full weight | 17.6 | Wildly overconfident: 8 clues pushing the same way |
| Neighbour nudge / borrowing rules from neighbours | no gain | Machines in an episode don't share rules |
| Separate size blending per type | no gain | Type mix already captured by the big blend |
| Treating rule cells as strictly live or dead | skipped | Training cell win rates are spread out, not two-valued |
| Faster forgetting (0.90) or sharper rule choice | worse | Hurts stable and short_history |
| Neural nets, one big classifier for all machines | not tried | ~200 noisy spins per machine; opposite rules cancel out |

---

## 8. How we tested it honestly

- **The kit's own scorer** (`eval_lag.py` calls the same `contest_score` as `check_submission.py`).
  The baseline reproduces 30.83 exactly.
- **Two halves of validation.** Settings were chosen on the "tune" half and confirmed on the "audit" half the
  kit uses. v4: tune 53.08, audit 52.68. A change was kept only if it didn't lose on the audit half.
- **All four conditions every time**, not just the total.
- **No future leakage.** The kit's `index_history` rejects any tick at or after a cutoff.
- **Fresh-folder reproduction.** `make_entry.py` unzips the entry ZIP into an empty folder, reruns it, and
  checks the CSV is byte-identical.

⚠️ We tuned many times on validation, so 52.89 is probably slightly optimistic for the hidden test set.

---

## 9. How to run it

From the `mainFiles` folder (on a Mac use `python3`):

```sh
cd mainFiles

# Build everything for an entry: forecasts, code ZIP, reproduction check
python3 make_entry.py --out runs/rule-v4

# Score it on validation
python3 check_submission.py runs/rule-v4/validation_predictions.csv --split validation

# Check the test CSV's format
python3 check_submission.py runs/rule-v4/submission.csv
```

Upload `runs/rule-v4/submission.csv` and `runs/rule-v4/entry.zip` as two separate files.

To try new settings: edit `SETTINGS` in `predict.py`, or score a model directly:

```python
from eval_lag import Bench, show
from rule_model import RuleForecaster
bench = Bench()
show("my test", bench.run(RuleForecaster(strength=0.5, size_strength=80.0))[0])
```

`runs/` is git-ignored: results stay on your machine. Use a new `--out` name for every run.

---

## 10. Round two

Round two is worth **40 points**. When the new kit is released (Saturday 9 PM Central):

1. Extract it into a separate folder and check `public/metadata.json` says `"round": "adaptation"`.
2. Run, pointing at the new data:
   ```sh
   python3 make_entry.py --data <new-kit>/public --out runs/round2-v1
   python3 check_submission.py runs/round2-v1/validation_predictions.csv --split validation --data <new-kit>/public
   ```
3. Compare the per-condition scores with main. If a condition got much worse, the rules may have changed
   (e.g. new lag pairs), so check which rules win with `RuleForecaster().fit(...).rule_wins`.
4. Upload under **Adaptation**, well before 11 AM Sunday.

---

## 11. Files

| File | What it is |
|---|---|
| `rule_model.py` | The model (`RuleForecaster`) |
| `predict.py` | Organizer entry point: `python predict.py --data /data --out /output/submission.csv`. Holds `SETTINGS` and the 50/50 blend |
| `make_entry.py` | One command: forecasts + code ZIP + fresh-folder reproduction check |
| `eval_lag.py` | Validation bench using the kit's scorer (all / tune / audit + conditions) |
| `lag_model.py` | The failed single-lag attempt, kept for reference |

---

## 12. Where the remaining error is

- **short_history (0.208)** is the biggest gap: ~33 usable spins to pick a rule from 37 candidates.
- **Payout size** is still about half the error on stable machines: the size mode is hard to read from a
  machine's history.
- Getting to mid-50s needs R ≈ 0.09 (we're at 0.135). Each new idea has been worth less than the last
  (+18, +3.4, +0.6), and more tuning on validation risks fitting noise.

---

## 13. The combined model

The team's **pair mixer** (`slot_forecasting/pair_mixer.py`) found the same simulator structure independently and
builds it a bit differently. Both models make different mistakes, so we average them.

| | Rule model (this doc) | Pair mixer (team) |
|---|---|---|
| Rules searched | All 37 (28 pairs, 8 single lags, none), with a head start for common ones | Only the common pairs rediscovered on train: (1,3), (2,5), (4,7) |
| Cells | 4 cells (win/lose × win/lose) | 2 cells: "did the two lags match or differ?" (pools XOR evidence, so it needs half the data) |
| Memory | Infinite and fast-fading (0.95/tick) | Half-lives of 12 and 40 ticks |
| Rule weighting | Log-likelihood of past predictions | Recent squared error (exponential weights) |
| Payout size | Type mix + big/small and win size rules | Size experts keyed on the exact earlier payouts |
| Best at | **stable** (0.068) | **short_history, drift, shared_change** |

### Scores on validation

| Model | Points | tune / audit | stable | short_history | drift | shared_change |
|---|---|---|---|---|---|---|
| Rule model v4 | 52.89 | 53.08 / 52.68 | 0.068 | 0.208 | 0.130 | 0.132 |
| Pair mixer | 54.21 | 54.08 / 54.39 | 0.083 | 0.151 | 0.093 | 0.100 |
| **30% rule + 70% mixer** | **54.48** | **54.42 / 54.56** | **0.067** | **0.148** | **0.091** | **0.099** |

Blend weights tried: 0.2 → 54.45, 0.3 → 54.48, 0.4 → 54.44, 0.5 → 54.33 (weight = rule model's share).
A geometric average gave the same 54.48, so we kept the simpler straight average.
Adding a very long memory to the mixer gave only +0.08, so the mixer's own settings are unchanged.

### How it runs

`predict.py` fits both models on `/data/train` (the mixer rediscovers its lag pairs there, so round two
adapts automatically), predicts with each, and writes `0.3 × rule + 0.7 × mixer`. The weight is
`RULE_WEIGHT` in `predict.py`. Total runtime is about 105 s per split, deterministic.
