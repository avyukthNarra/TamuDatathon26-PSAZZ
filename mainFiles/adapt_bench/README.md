# adapt_bench: simulated round-two benchmark

Ten simulated "round two" kits for checking how well a candidate adapts to a
new generator. Each kit has the same public layout as the real round, so
`run_pair_mixer.py` and the other builds read them unchanged.

Purpose: the real main round can't show whether a model adapts when the game
changes (new lag pairs, a different size rule, timing shifts, mixed machine
rules). Each kit changes one or two of those things, so a build's score on
each kit shows where it holds up and where it breaks.

## Kits

Every kit has 8 machines per episode, 4 conditions (stable, short_history,
drift, shared_change), and 12 episodes per condition per split.

| kit | what changes |
| --- | --- |
| K1 | Main pairs (1,3), (4,7), (2,5); main size rule (L is the {1,2,4} member); main timing. |
| K2 | Disjoint pairs (1,6), (3,8), (2,4); size lag L is the smaller pair member. |
| K3 | Four pairs (1,6), (2,4), (3,8), (5,7) at weights 0.50/0.27/0.15/0.08; L = min. |
| K4 | Pairs share a lag: (1,6), (1,8), (3,6); L = min. |
| K5 | Long lags only: (5,8), (6,7), (3,8); L = min, so L is in {3,5,6}. |
| K6 | Main pairs with a static size atom (no lag rule). |
| K7 | Main pairs and size rule, but the size bit is payout >= 5 instead of >= 2. |
| K8 | Late shared flip (about 20 ticks before the cutoff) and a fast drift (18-tick ramp). |
| K9 | Shifted levels (jackpot live about 0.35, steady dead about 0.25) and signal slope 1.15. |
| K10 | 30% of machines use a 3-lag XOR or an AND rule; the rest stay 2-lag XOR/XNOR. |

## Quick start

```bash
./regenerate.sh                  # writes kits/K1..K10 (about 30 s, about 70 MB)
./bench.sh --quick --score-only v7c \
  'cd "$V7C_DIR" && python3 run_pair_mixer.py --data {kit} --out {out} --cv-val'
```

`bench.sh` runs each kit, reads the two cross-fit halves from the build output,
and writes `bench_<name>.csv` and `bench_<name>.md` here. The `{kit}` and
`{out}` placeholders are filled in per kit.

`templates.sh` has the ready-made builds (`v6`, `v6-fallback`, `v7c`, `v15`):
`./templates.sh v7c --quick --score-only`.

Paths to other candidate checkouts come from environment variables, with
defaults for sibling worktrees:

- `V7C_DIR` (default `../../../v7c/mainFiles` relative to this folder): a v7c
  tree with `run_pair_mixer.py`. It is not part of this repo, so set it to your
  own checkout if it lives elsewhere.
- `V15_DIR` (default `research_pf/h12-v15/wt/mainFiles` under the Datathon
  folder): a v15 tree with `run_strongest.py`. Also outside this repo.
- `SLOT_FORECASTING_ROOT` (optional): where `simulator.py` imports
  `slot_forecasting` from. Defaults to the `mainFiles/` folder this one sits in.

`regenerate.sh [OUT_DIR]` writes somewhere else if given a directory. Kits are
deterministic: two runs give identical bytes.

## Tiers

- `--quick`: K1, K2, K6, K8. About 2 to 3 minutes per model with `--score-only`.
- `--full` (default): all ten kits.

## Sanity anchor

K1 should rank v7c above v6 by about +0.8 out of 40. The real main round gives
v7c 57.37 against v6 56.29, which is +1.08 out of 60 (about +0.72 on a 40-point
scale). If K1 shows v6 ahead of v7c by a clear margin, the simulator or the
bench wiring is broken, not the model.

## Scoreboard (quick tier, cross-fit mean, out of 40)

Copied from `REPORT.md`, "After the fix", which used `--quick --score-only`.

| kit | v6 | v7c | v7c − v6 |
| --- | ---: | ---: | ---: |
| K1 | 37.18 | 38.02 | **+0.84** |
| K2 | 37.62 | 38.40 | **+0.78** |
| K6 | 36.51 | 38.94 | **+2.43** |
| K8 | 31.22 | 38.90 | **+7.68** |

K8's large gap comes from the late flip. v7c's flip head handles it. v6 runs
with the flip off, so it stays at the pre-fix level on this kit.

## Caveats

- 12 episodes per condition per split means 48 episodes per split, so the
  scores are noisy. Differences of 1 point or less on one kit are not evidence.
- The simulator is built from the reverse engineering of the main round. Round
  two may differ in ways none of these kits cover. Passing here is evidence of
  adaptability, not a guarantee for the real round two.
- Kits regenerated from this copy are byte-identical to the earlier
  `a2-benchfix/kits` for K1, K6 and K8, which were checked file by file.
