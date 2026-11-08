"""One command for round two (adaptation): build, compare, verify and stage the entry.

Usage (from mainFiles/), after unzipping the new kit somewhere:

    python round_two.py --data /path/to/new-kit/public
    python round_two.py --data /path/to/new-kit/public --quick     # v19 + v20 only

Steps:
  1. Check public/metadata.json (warns unless round == "adaptation").
  2. Build up to three candidates on the NEW data; each refits everything (lag pairs, size lags,
     size modes, win priors, peer centre) from that data:
       v20  Sai's blend (v13+short+jackpot, exact filter, sysid; learns rules per round) + v19 as a 4th member
       v19  v12 + Avyukth's v7c (in a 2nd process) + add-ons gated on the new round's validation at build time
       v12  v11 with the size-rule HMM sizes (zhill branch)
       v11  log-odds average of v10, a 0.6 / 0.4 version and per-lag peers
       v10  log-odds pool of pair mixer + hidden-rule HMM, plus the peer correction
       v9   the same without the peer correction
       v8   the COMBINED_SOLUTIONS model (pair mixer + 30% HMM, linear)
  3. Pick the candidate with the best CROSS-FITTED validation score (each half scored with priors
     from the other half). Ties within 0.02 go to the simpler model (v8 < v9 < v10 in complexity).
  4. Verify the chosen entry: kit checker on test and validation, and a fresh rerun of the ZIP from
     a read-only copy WITHOUT validation/targets.csv must match the CSV byte for byte.
  5. Copy submission.csv + entry.zip to ~/Downloads/HouseEdge_ADAPTATION_submission/ and print
     the scores, so the two files can go straight into the Adaptation form.
Nothing is uploaded.
"""
import argparse
import json
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="the NEW kit's public/ folder")
parser.add_argument("--tag", default="round2")
parser.add_argument("--quick", action="store_true", help="build v19 and v20 only")
parser.add_argument("--downloads", default=str(Path.home() / "Downloads" / "HouseEdge_ADAPTATION_submission"))
args = parser.parse_args()
data = Path(args.data).expanduser().resolve()
if (data / "public").is_dir():
    data = data / "public"
meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
print(f"kit: round={meta.get('round')} season={meta.get('season_id')} max_points={meta.get('max_points')}")
if meta.get("round") != "adaptation":
    print("WARNING: metadata round is not 'adaptation' - is this the new kit?")

CANDIDATES = {
    "v19": [sys.executable, "run_strongest.py"],
    # v20 vendors the v19 just rebuilt on THIS round's data; needs numpy/scipy, so run round_two.py with
    # runs/.venv-sai/bin/python (the stdlib candidates run there too)
    "v20": [sys.executable, "run_super.py", "--v19", str(HERE / "runs" / f"{args.tag}-v19" / "entry")],
    "v12": [sys.executable, "run_ensemble.py", "--variant", "strongest"],
    "v11": [sys.executable, "run_ensemble.py", "--variant", "plateau"],
    "v10": [sys.executable, "run_ensemble.py", "--variant", "peers"],
    "v9": [sys.executable, "run_ensemble.py", "--variant", "logodds"],
    "v8": [sys.executable, "run_pair_mixer.py"],
}
if args.quick:
    CANDIDATES = {"v19": CANDIDATES["v19"], "v20": CANDIDATES["v20"]}
results = {}
for name, cmd in CANDIDATES.items():
    out = HERE / "runs" / f"{args.tag}-{name}"
    if out.exists():
        shutil.rmtree(out)
    print(f"\n=== building {name} -> {out}", flush=True)
    run = subprocess.run(cmd + ["--data", str(data), "--out", str(out), "--cv-val"], cwd=HERE,
                         capture_output=True, text=True)
    print("\n".join(run.stdout.strip().splitlines()[-9:]))
    if run.returncode != 0:
        print(f"{name} FAILED:\n{run.stderr[-2000:]}")
        continue
    cross = json.loads((out / "crossfit_validation_score.json").read_text())
    full = json.loads((out / "validation_score.json").read_text())
    halves = [cross["tuning_using_audit_priors"]["technical_points"], cross["audit_using_tuning_priors"]["technical_points"]]
    results[name] = {"out": out, "full": full["technical_points"], "halves": halves, "cross": sum(halves) / 2}

if not results:
    sys.exit("no candidate built - see errors above")
print("\n=== candidates (validation points; cross-fit = mean of the two halves)")
for name, r in results.items():
    print(f"  {name}: full {r['full']:.2f}  cross-fit {r['cross']:.2f}  halves {r['halves'][0]:.2f} / {r['halves'][1]:.2f}")
order = ["v8", "v9", "v10", "v11", "v12", "v19", "v20"]  # simplest first
best = max(results, key=lambda n: results[n]["cross"])
for name in order:  # prefer a simpler model within 0.02
    if name in results and results[name]["cross"] >= results[best]["cross"] - 0.02:
        best = name
        break
chosen = results[best]
print(f"\nCHOSEN: {best} (cross-fit {chosen['cross']:.2f}, full {chosen['full']:.2f})")

# ---- verify ----
out = chosen["out"]
for split, path in (("test", out / "submission.csv"), ("validation", out / "validation_predictions.csv")):
    chk = subprocess.run([sys.executable, "check_submission.py", str(path), "--data", str(data)]
                         + (["--split", "validation"] if split == "validation" else []), cwd=HERE, capture_output=True, text=True)
    if chk.returncode != 0:
        sys.exit(f"checker failed on {split}:\n{chk.stdout}\n{chk.stderr}")
    print(f"checker OK on {split}: " + " | ".join(l for l in chk.stdout.splitlines() if "points" in l.lower() or "SHA" in l)[:200])
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    shutil.unpack_archive(out / "entry.zip", tmp / "submission")
    shutil.copytree(data, tmp / "data")
    targets = tmp / "data" / "validation" / "targets.csv"
    if targets.exists():
        targets.unlink()
    for p in (tmp / "submission").rglob("*"):
        p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    rerun = subprocess.run([sys.executable, "-I", "predict.py", "--data", str(tmp / "data"), "--out", str(tmp / "out.csv")],
                           cwd=tmp / "submission", capture_output=True, text=True)
    for p in (tmp / "submission").rglob("*"):
        p.chmod(p.stat().st_mode | stat.S_IWUSR)
    if rerun.returncode != 0:
        sys.exit(f"fresh rerun of the ZIP failed:\n{rerun.stderr[-2000:]}")
    same = (tmp / "out.csv").read_bytes() == (out / "submission.csv").read_bytes()
    print("fresh read-only rerun of entry.zip without validation targets:", "byte-identical" if same else "DIFFERENT")
    if not same:
        sys.exit("reproduction mismatch - do not upload")

dest = Path(args.downloads).expanduser()
dest.mkdir(parents=True, exist_ok=True)
for name in ("submission.csv", "entry.zip"):
    shutil.copyfile(out / name, dest / name)
print(f"\nREADY TO UPLOAD (Adaptation): {dest}/submission.csv and {dest}/entry.zip")
