# House Edge blend: v13+short, exact filter, sysid

- Round: main
- Season ID: `5fffdc4d0c700bbf61c3f911`
- Environment: `python` image (Python 3.13). Pinned in `requirements.txt`: `numpy==2.3.4`, `scipy==1.16.3`.
- Randomness: none. `PYTHONHASHSEED=0` and one BLAS thread.
- Hardware: CPU. A full test run took 463 seconds on an 11-core laptop with the three members in parallel and one BLAS thread. v13 and the exact filter both ran the whole time; sysid finished first. On 2 CPUs that is still well under 30 minutes.

## Recreate the forecast

```sh
python predict.py --data /data --out /output/submission.csv
```

The command fits three win models on `/data/train` and predicts `/data/test`.

If `/data/validation/targets.csv` is present, v13 reads size modes and the win prior from it, and the exact filter reads its logit level prior from it. Those files are priors, not the rows being scored. Sysid never reads validation targets; it selects its rule and reversal width on train only.

If validation is absent, v13 uses train-only payout frequencies, the exact filter keeps the level prior fit on train, and the short-history filter uses train cell rates. That fallback is less accurate. It does not crash.

## What the model does

P(win) is the sigmoid of the equal-weight average of the three members' logit P(win). Weights are fixed at 1/3. The payout-size split given a win is v13's. Each probability is then floored at `1e-4` and the row is renormalized.

Members, each a replaceable script (see `members/hooks.py`):

- `members/v13_short.py`: v13 from `origin/pranav/strongest-build` at `a70abc0`. Short-history queries use `grid_marginal` with lag pairs and level centres learned from this round. Jackpot queries that are not short-history use the four-cell rate of the MAP pair inside that same support, shrunk toward its parity pool. The shrink strength is the train-label Brier choice. Without a condition file, a machine is short when fewer than 75% of its ticks were observed.
- `members/exact_filter.py`: exact posterior over the lag pairs voted in by train (both cells at least 5 ticks, a 2-nat lead, at least 5% of machines). There is no shipped list of pairs. Set `PAIRS_OVERRIDE` in `members/hooks.py` to force a list.
- `members/sysid_rule.py`: sparse rule library learned from train. It ignores `PAIRS_OVERRIDE`. Replace the script to drop in another library.

`members/hooks.py` lists the scripts. Another agent can replace one path without editing the blend.
