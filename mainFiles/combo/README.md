# House Edge entry: 20% rule model + 80% pair mixer v3

- Environment: `python` image, **standard library only**. Tested locally on Python 3.13.
- Randomness: none (fully deterministic, no seeds).
- Hardware: CPU only; about 35 s on a laptop, far under the 30-minute limit.

## Recreate the forecast

Run from this folder:

```sh
python predict.py --data /data --out /output/submission.csv
```

`--data` may be the public folder or its parent. `--split validation` writes the validation forecast.

## What it does

Both models are fitted on `/data/train` at run time (no saved checkpoint):

- **Rule model** (`rule_model.py`, zhill): per-machine search over lag rules, two variants
  (graded on exact payout, graded on win/lose) averaged.
- **Pair mixer v3** (`slot_forecasting/pair_mixer.py`, Pranav, pinned at 6a8098c): per-machine
  online mixture of lag-pair experts. Lag pairs are rediscovered on train, and the two payout-size
  modes per machine type are read from `/data/validation/targets.csv` when it exists (otherwise
  free frequencies).

Forecast = 0.2 * rule + 0.8 * mixer, renormalised per query (order 0, 1, 2, 5, 20).
