# ZHILL episode-pooled memory handoff

The implementation is on `avyukth/exp-zhill-improve`, score-producing source
SHA `95736662e8ecd5e1bb5d90635b94897740d2034e`. It adds
`slot_forecasting.pair_mixer_episode.EpisodeMemoryMixer` and an optional mixer
factory seam in `slot_forecasting.zhill_blend.ZhillBlendForecaster`.

Use `ZhillBlendForecaster(half_life=(12, 40, 1e9),
mixer_factory=EpisodeMemoryMixer, mixer_kwargs={"gamma": 0.25})`. The wrapper
retains the 20% two-rule / 80% PairMixer blend. On train data it rediscovers lag
pairs and chooses parity versus full cell mode by train cutoff Brier, just as
the original `predict.py` adapter does.

At each query cutoff, every machine's experts are replayed only through observed
ticks strictly before that cutoff. Their cumulative squared errors are summed
raw by half-life across all episode machines, including the target machine.
That total is applied once to the target machine's final expert logweights as
`-eta * gamma * pooled_loss[half_life]`; pair/mode relative weights within each
speed remain local. No pooled penalty is repeatedly applied during replay.

| Tuning-only control | Points / 60 | stable | short history | drift | shared change |
|---|---:|---:|---:|---:|---:|
| Original blend, half-lives `(12, 40)` | 54.367545 | 0.090015 | 0.140591 | 0.103843 | 0.079950 |
| Original blend, half-lives `(12, 40, 1e9)` | 54.427365 | 0.085728 | 0.140042 | 0.103843 | 0.079934 |
| Episode pooled, gamma `0.25`, half-lives `(12, 40, 1e9)`, full blend | 54.799193 | 0.061943 | 0.133868 | 0.103843 | 0.079972 |
| Same fitted mixer alone | 54.623895 | 0.068160 | 0.138361 | 0.107624 | 0.079537 |

Gamma `0.25` gained 0.431648 points over the unchanged blend and crossed the
predeclared +0.15 gate. Gamma `0.5` and `1.0` and queued size-key / lag-7 tests
were not run. Stable and short-history errors improved; shared-change error was
essentially unchanged/slightly worse. The run artifact did not retain the
selected lag-pair list or selected cell mode, so those are not asserted here.

All comparisons scored only the fixed tuning episodes (48 episodes, 384
queries); no audit score was computed. The saved CSV was independently rescored
with `episode_folds` and `subset_score`, reproducing exactly 54.799193334024395.
The focused suites passed 17 tests total: six episode-memory tests, three blend
wrapper tests, and eight existing PairMixer tests. `py_compile` and
`git diff --check` also passed.

Artifacts (ignored run outputs):

- Runner: `mainFiles/runs/zhill-task2-gamma/score_tuning.py`
- JSON: `mainFiles/runs/zhill-task2-gamma/outputs/summary.json`
- Blend tuning predictions: `mainFiles/runs/zhill-task2-gamma/outputs/gamma-0.25-blend-tuning.csv`
- Mixer tuning predictions: `mainFiles/runs/zhill-task2-gamma/outputs/gamma-0.25-mixer-tuning.csv`
- Append-only rows: `_coord/RESULTS.md`
