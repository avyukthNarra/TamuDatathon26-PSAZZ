"""One command to build a full entry: python3 make_entry.py --out runs/rule-v2

Writes, inside --out:
  submission.csv               test forecast to upload
  validation_predictions.csv   score it with check_submission.py --split validation
  entry/                       the code folder (README, reproduce.json, predict.py, model code)
  entry.zip                    code ZIP to upload
and checks that running the packaged code from a fresh copy recreates submission.csv exactly.

Use --data to point at another round's public/ folder (e.g. round two).
"""
import argparse
import filecmp
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT_MODULES = ("__init__.py", "baseline.py", "schema.py", "io.py")

README = """# House Edge entry: per-machine lag-pair rule model

- Round: {round}  ·  Season ID: {season}
- Environment: `python` image (Python 3.13), **standard library only**. Tested locally on Python {pyver}.
- Randomness: none (fully deterministic; no seeds needed).
- Hardware: CPU only; about {secs:.0f} s on a laptop, well under the 30-minute limit.

## Recreate the forecast

Run from this folder:

```sh
python predict.py --data /data --out /output/submission.csv
```

It fits the model on `/data/train` (training history + labels; there is no saved checkpoint,
training is deterministic and fast) and predicts `/data/test`. No other files are needed.

## What the model does

For each machine it tries every "rule": a pair of earlier spins (lags 1-8), a single earlier spin,
or no rule. Each rule keeps a tally of "win/lose at those lags -> next payout". Rules are scored
on how well they predicted the machine's own past spins (each spin is predicted before it is
added to the tally), with short and long memory variants so changing machines are tracked.
The forecast is the trust-weighted average of all rules, shrunk toward the pooled_recent
starter forecast. Rules common among training machines get a head start (learned from
`/data/train` at run time), which helps machines with short histories.

Rules decide WILL it pay; payout SIZE (1x/2x/5x/20x) comes from the machine's own wins blended
toward its machine type's average mix, refined by per-machine size rules (same trust-weighted
rule search, applied to the sizes of past wins). The final forecast averages two variants: rules graded on
the exact payout, and rules graded on win/lose only.

Files: `predict.py` (entry point), `rule_model.py` (model), `slot_forecasting/` (kit I/O helpers).
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "public")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out, data = args.out.resolve(), args.data.resolve()
    if out.exists():
        raise SystemExit(f"{out} already exists — pick a new --out folder so earlier runs are kept.")
    out.mkdir(parents=True)

    import json, time
    meta = json.loads((data / "metadata.json").read_text())

    # 1. stage the code folder
    entry = out / "entry"
    (entry / "slot_forecasting").mkdir(parents=True)
    for name in KIT_MODULES:
        shutil.copyfile(HERE / "slot_forecasting" / name, entry / "slot_forecasting" / name)
    shutil.copyfile(HERE / "predict.py", entry / "predict.py")
    shutil.copyfile(HERE / "rule_model.py", entry / "rule_model.py")
    (entry / "reproduce.json").write_text(json.dumps({
        "image": "python",
        "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"]}, indent=2) + "\n")

    # 2. forecasts, produced BY THE STAGED CODE (so the upload is exactly what the ZIP makes)
    start = time.time()
    for split, name in (("test", "submission.csv"), ("validation", "validation_predictions.csv")):
        subprocess.run([sys.executable, "predict.py", "--data", str(data), "--out", str(out / name),
                        "--split", split], cwd=entry, check=True)
    secs = (time.time() - start) / 2
    (entry / "README.md").write_text(README.format(round=meta["round"], season=meta["season_id"],
                                                   pyver=sys.version.split()[0], secs=secs))
    for cache in entry.rglob("__pycache__"):
        shutil.rmtree(cache)

    # 3. package + fresh-folder reproduction check
    subprocess.run([sys.executable, str(HERE / "build_entry.py"), "--source", str(entry), "--out",
                    str(out / "entry.zip")], cwd=HERE, check=True)
    with tempfile.TemporaryDirectory() as tmp:
        shutil.unpack_archive(out / "entry.zip", tmp)
        subprocess.run([sys.executable, "predict.py", "--data", str(data), "--out", str(Path(tmp) / "again.csv")],
                       cwd=tmp, check=True)
        same = filecmp.cmp(out / "submission.csv", Path(tmp) / "again.csv", shallow=False)
    print("Fresh-folder reproduction:", "IDENTICAL ✅" if same else "DIFFERENT ❌")
    if not same:
        raise SystemExit(1)
    print(f"\nUpload these two files:\n  {out / 'submission.csv'}\n  {out / 'entry.zip'}")


if __name__ == "__main__":
    main()
