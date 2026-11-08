"""Build v20, the super entry: Sai's 3-member blend (v13+short+jackpot, exact filter, sysid) plus
Pranav's v19 as a 4th equal log-odds member. Run from mainFiles/:

    runs/.venv-sai/bin/python run_super.py --out runs/super-v20

Needs the pinned NumPy / SciPy of entry_template_super/requirements.txt (the venv above) so the CSV
matches the organisers' rerun. Steps: copy entry_template_super/ (Sai's adapted_entry_v17g_rules,
read from sai/master-test, plus members/pranav_v19.py), vendor the built v19 entry
(runs/strongest-v19/entry) into vendor/v19/, then produce validation, test and cross-fitted
validation forecasts BY RUNNING THE ENTRY, score them, zip, and rerun the test forecast in a fresh
process (must match within 1e-8).
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from slot_forecasting.contest import score_contest_files  # noqa: E402
from slot_forecasting.experiments import episode_folds, subset_score  # noqa: E402
from slot_forecasting.io import read_rows, read_submission, read_targets, write_json  # noqa: E402
from slot_forecasting.rounds import VERSION, round_metadata  # noqa: E402
from slot_forecasting.submission_contract import package_code  # noqa: E402

EVAL_MODULES = ("experiments.py", "baseline.py", "contest.py", "rounds.py", "transition.py", "scoring.py")  # for crossfit only

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=str(HERE / "public"))
parser.add_argument("--out", default="runs/super-v20")
parser.add_argument("--cv-val", action="store_true", help="accepted for round_two.py; cross-fit always runs")
parser.add_argument("--v19", default=str(HERE / "runs/strongest-v19/entry"), help="built v19 entry folder")
args = parser.parse_args()
public = Path(args.data).resolve()
out = Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=True)
started = time.perf_counter()

entry = out / "entry"
if entry.exists():
    shutil.rmtree(entry)
shutil.copytree(HERE / "entry_template_super", entry, ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(Path(args.v19), entry / "vendor" / "v19", ignore=shutil.ignore_patterns("__pycache__"))
shutil.copyfile(public / "validation" / "targets.csv", entry / "bundled_validation_targets.csv")  # see predict.py
for name in EVAL_MODULES:
    shutil.copyfile(HERE / "slot_forecasting" / name, entry / "vendor" / "v19" / "slot_forecasting" / name)


def run(split, path, protocol="submission"):
    t0 = time.perf_counter()
    done = subprocess.run([sys.executable, "predict.py", "--data", str(public), "--out", str(path), "--split", split,
                           "--protocol", protocol], cwd=entry, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"entry failed ({split}, {protocol}):\n{done.stdout[-2000:]}\n{done.stderr[-3000:]}")
    for line in done.stdout.splitlines():
        if line.startswith("done "):
            print(f"  [{split}/{protocol}] {line}")
    return time.perf_counter() - t0


files = {"validation": out / "validation_predictions.csv", "test": out / "submission.csv"}
seconds = {split: run(split, path) for split, path in files.items()}
seconds["crossfit"] = run("validation", out / "crossfit_validation_predictions.csv", "crossfit")
print(f"entry runtime: validation {seconds['validation']:.0f}s, test {seconds['test']:.0f}s, crossfit {seconds['crossfit']:.0f}s")

val = public / "validation"
event = {"max_points": round_metadata(public)["max_points"], "version": VERSION}
report = score_contest_files(files["validation"], val / "targets.csv", val / "queries.csv", val / "conditions.csv",
                             val / "reference.csv", **event)
write_json(out / "validation_score.json", report)
targets = read_targets(val / "targets.csv")
groups = {r["query_id"]: r["condition"] for r in read_rows(val / "conditions.csv", ("query_id", "condition"))}
reference = read_submission(val / "reference.csv")
crossfit = read_submission(out / "crossfit_validation_predictions.csv")
halves = [subset_score(crossfit, targets, groups, reference, episodes, event["max_points"])
          for episodes in episode_folds(targets, groups)]
write_json(out / "crossfit_validation_score.json", {"tuning_using_audit_priors": halves[0], "audit_using_tuning_priors": halves[1]})
for name, row in report["by_condition"].items():
    print(f"{name:<15}{row['error']:>8.4f}{row['reference_error']:>10.4f}{row['loss_ratio']:>8.3f}")
print(f"R = {report['mean_loss_ratio']:.4f}  ->  {report['technical_points']:.2f} / {report['max_points']:g} points")
print(f"Cross-fitted validation: tuning {halves[0]['technical_points']:.2f}; audit {halves[1]['technical_points']:.2f}")

sha = hashlib.sha256(files["test"].read_bytes()).hexdigest()
readme = (entry / "README.md").read_text(encoding="utf-8")
(entry / "README.md").write_text(readme + f"""

## pranav/super v20 addition
A 4th equal win member, `members/pranav_v19.py`, runs Pranav's v19 entry (vendored unchanged in
`vendor/v19/`): pair mixer + hidden-rule HMM + peer / signal / plateau terms blended with Avyukth's
v7c (flip-swap transfer, jackpot sub-cell), standard library only. The blend is the mean logit of
four members; sizes stay with `v13_short`. Built by `mainFiles/run_super.py` on pranav/super.
- Validation {report['technical_points']:.2f} / {report['max_points']:g} (R = {report['mean_loss_ratio']:.4f});
  cross-fitted halves {halves[0]['technical_points']:.2f} / {halves[1]['technical_points']:.2f}
- Forecast SHA-256: {sha}
- Runtime: about {seconds['test']:.0f} s for the test forecast (members run as parallel processes).
""", encoding="utf-8")
zip_path = out / "entry.zip"
if zip_path.exists():
    zip_path.unlink()
for cache in entry.rglob("__pycache__"):
    shutil.rmtree(cache)
package_code(entry, zip_path)
repro = out / "repro_check.csv"
run("test", repro)
a, b = read_submission(files["test"]), read_submission(repro)
worst = max(abs(x - y) for k in a for x, y in zip(a[k], b[k]))
if set(a) != set(b) or worst > 1e-8:
    raise SystemExit(f"Reproduction FAILED: {worst:g}")
print(f"Entry: {zip_path}  (fresh rerun max diff {worst:g}; limit 1e-8)")
print(f"Submission: {files['test']}  sha256 {sha}  ({time.perf_counter() - started:.0f}s total)")
