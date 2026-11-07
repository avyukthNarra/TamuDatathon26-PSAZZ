"""Score the PairMixer on validation, write a test submission, and build a verified entry ZIP.

Usage (from mainFiles/):  python run_pair_mixer.py --out runs/pair-mixer
Writes to <out>/: validation_predictions.csv, submission.csv, validation_score.json,
entry/ (README.md, reproduce.json, predict.py, slot_forecasting/), entry.zip.
Then re-runs entry/predict.py from a fresh process and checks every probability
matches submission.csv within 1e-8 (the organizers' reproduction tolerance).
"""
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from slot_forecasting.contest import score_contest_files
from slot_forecasting.io import load_inputs, read_labels, read_submission, read_targets, write_json, write_submission
from slot_forecasting.pair_mixer import adapt, size_modes_from_targets
from slot_forecasting.rounds import VERSION, round_metadata
from slot_forecasting.starter import public_directory
from slot_forecasting.submission_contract import package_code

HERE = Path(__file__).resolve().parent
MODULES = ("__init__.py", "schema.py", "io.py", "pair_mixer.py")

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=str(HERE / "public"))
parser.add_argument("--out", default="runs/pair-mixer")
args = parser.parse_args()

started = perf_counter()
public = public_directory(args.data)
out = Path(args.out).expanduser().resolve()
out.mkdir(parents=True, exist_ok=True)
train_queries, train_history = load_inputs(public / "train")
size_modes = size_modes_from_targets(read_targets(public / "validation/targets.csv"), load_inputs(public / "validation")[0])
model = adapt(train_queries, train_history, read_labels(public / "train/labels.csv", train_queries), size_modes=size_modes)
files = {"validation": out / "validation_predictions.csv", "test": out / "submission.csv"}
for split, path in files.items():
    queries, history = load_inputs(public / split)
    write_submission(path, model.predict(queries, history))

# --- validation: official contest score + win-chance vs payout-size error split ---
event = {"max_points": round_metadata(public)["max_points"], "version": VERSION}
val = public / "validation"
report = score_contest_files(files["validation"], val / "targets.csv", val / "queries.csv",
                             val / "conditions.csv", val / "reference.csv", **event)
predictions = read_submission(files["validation"])
conditions = dict(line.strip().split(",") for line in (val / "conditions.csv").read_text().splitlines()[1:])
split_error = defaultdict(lambda: [0.0, 0.0, 0])
for target in read_targets(val / "targets.csv"):
    p, q = predictions[target.query_id], target.probabilities
    p_win, q_win = 1 - p[0], 1 - q[0]
    # error left if the win chance were exactly right (keeps our size split) = size part
    fixed_win = [q[0]] + [x / p_win * q_win for x in p[1:]]
    total = sum((a - b) ** 2 for a, b in zip(p, q))
    size_part = sum((a - b) ** 2 for a, b in zip(fixed_win, q))
    row = split_error[conditions[target.query_id]]
    row[0] += total - size_part
    row[1] += size_part
    row[2] += 1
write_json(out / "validation_score.json", report)
print(f"{'condition':<15}{'error':>8}{'baseline':>10}{'ratio':>8}{'win part':>10}{'size part':>11}")
for name, row in report["by_condition"].items():
    win, size, n = split_error[name]
    print(f"{name:<15}{row['error']:>8.4f}{row['reference_error']:>10.4f}{row['loss_ratio']:>8.3f}"
          f"{win / n:>10.4f}{size / n:>11.4f}")
print(f"R = {report['mean_loss_ratio']:.4f}  ->  {report['technical_points']:.2f} / {report['max_points']:g} points")

# --- entry folder + ZIP ---
entry = out / "entry"
if entry.exists():
    shutil.rmtree(entry)
(entry / "slot_forecasting").mkdir(parents=True)
for name in MODULES:
    shutil.copyfile(HERE / "slot_forecasting" / name, entry / "slot_forecasting" / name)
for name in ("predict.py", "reproduce.json"):
    shutil.copyfile(HERE / "entry_template" / name, entry / name)
(entry / "size_modes.json").write_text(json.dumps(size_modes), encoding="utf-8")  # fallback if /data lacks targets
metadata = json.loads((public / "metadata.json").read_text(encoding="utf-8"))
sha = hashlib.sha256(files["test"].read_bytes()).hexdigest()
commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip() or "unknown"
(entry / "README.md").write_text(f"""# House Edge entry: lag-pair mixer

- Round: {metadata.get('round', 'main')}
- Season ID: {metadata.get('season_id', 'see public/metadata.json')}
- Forecast SHA-256: {sha}
- Code version: git {commit} (built by mainFiles/run_pair_mixer.py)
- Environment: `python` image, Python standard library only (tested on Python {platform.python_version()})
- Validation: {report['technical_points']:.2f} / {report['max_points']:g} points (R = {report['mean_loss_ratio']:.4f})

## Recreate the forecast

Run from the archive root:

```sh
python predict.py --data /data --out /output/submission.csv
```

- What it does: on `/data/train` it rediscovers which lag pairs drive wins, picks
  match/differ (2-cell) vs 4-cell experts by Brier on `train/labels.csv` at the
  cutoff, fits payout-size priors per machine type, and reads the two exact
  payout-size mixes per type from `/data/validation/targets.csv`; then it predicts every
  `/data/test` query with a per-machine online mixture of lag-pair experts. Each expert tracks, with exponential forgetting, how often the machine
  won when its results at two earlier lags matched vs differed; experts are
  weighted by their recent squared error. Only ticks before each cutoff are read.
- Model files: none (the command refits in about a second).
- Seeds/settings: deterministic, no randomness; settings are the `PairMixer`
  defaults in `slot_forecasting/pair_mixer.py`.
- Runtime: about {perf_counter() - started:.0f} s for train + validation + test on one CPU core.
""", encoding="utf-8")
zip_path = out / "entry.zip"
if zip_path.exists():
    zip_path.unlink()
package_code(entry, zip_path)

# --- reproduction check from a fresh process, run from inside the entry folder ---
repro = out / "repro_check.csv"
subprocess.run([sys.executable, "predict.py", "--data", str(public), "--out", str(repro)],
               cwd=entry, check=True, capture_output=True)
original = read_submission(files["test"])
again = read_submission(repro)
worst = max(abs(a - b) for key in original for a, b in zip(original[key], again[key]))
if set(original) != set(again) or worst > 1e-8:
    raise SystemExit(f"Reproduction FAILED: max probability difference {worst:g}")
print(f"Entry: {zip_path}  (fresh rerun max diff {worst:g}; limit 1e-8)")
print(f"Submission: {files['test']}  sha256 {sha}  ({perf_counter() - started:.0f}s total)")
