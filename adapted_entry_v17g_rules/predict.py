"""Equal log-odds blend of v13+short, the exact filter, and sysid.

    python predict.py --data /data --out /output/submission.csv

Fits each member on /data/train. Validation targets, when present, are priors
only (v13 size modes and win prior; exact level prior). Sysid never reads them.
P(win) is the sigmoid of the equal-weight mean logit. The size split is v13's.
Probabilities are floored at 1e-4 and renormalized.

Members are separate processes so their packages can be replaced independently.
See members/hooks.py.
"""
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from blend import blend, read_predictions, write_predictions
from members.hooks import REGISTRY, SIZE_MEMBER

ENTRY = Path(__file__).resolve().parent


def with_bundled_targets(data):
    """Same forecast with or without validation answers mounted: the members use
    validation/targets.csv as priors, so when /data lacks it, read the copy bundled at build time
    (bundled/validation_targets.csv, public kit data) through a /tmp view of /data (symlinks)."""
    bundled = ENTRY / "bundled" / "validation_targets.csv"
    if (data / "validation" / "targets.csv").exists() or not bundled.exists():
        return data
    view = Path(tempfile.mkdtemp(prefix="data-view-"))
    for item in data.iterdir():
        if item.name != "validation":
            (view / item.name).symlink_to(item)
    (view / "validation").mkdir()
    for item in (data / "validation").iterdir():
        (view / "validation" / item.name).symlink_to(item)
    (view / "validation" / "targets.csv").symlink_to(bundled)
    print("validation answers not mounted: using bundled/validation_targets.csv", flush=True)
    return view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--protocol", choices=("submission", "crossfit"), default="submission")
    parser.add_argument("--member-dir", default="", help="keep each member CSV in this directory")
    parser.add_argument("--all-members", action="store_true", help="ignore gate.json (used by r2/gate_members.py)")
    args = parser.parse_args()
    # gate.json (written at build time by r2/gate_members.py) lists the win members kept for this
    # round; a member that hurt the round's cross-fitted validation on both halves is left out.
    # Without gate.json every member runs, exactly as before.
    gate_path = ENTRY / "gate.json"
    win_names = [spec["name"] for spec in REGISTRY]
    if gate_path.exists() and not args.all_members:
        win_names = json.loads(gate_path.read_text())["win_members"]
        print(f"gate.json: win members {win_names}", flush=True)
    active = [spec for spec in REGISTRY if spec["name"] in win_names or spec["name"] == SIZE_MEMBER]
    started = time.perf_counter()
    data = str(with_bundled_targets(Path(args.data).resolve()))
    out_path = Path(args.out).resolve()
    env = os.environ.copy()
    hold = tempfile.TemporaryDirectory(prefix="blend-members-")
    member_dir = Path(args.member_dir).resolve() if args.member_dir else Path(hold.name)
    member_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for spec in active:
        out = member_dir / f"{spec['name']}.csv"
        command = [
            sys.executable, str(ENTRY / spec["script"]),
            "--data", data, "--split", args.split, "--out", str(out), "--protocol", args.protocol,
        ]
        print(f"start {spec['name']}", flush=True)
        processes.append((spec, out, time.perf_counter(), subprocess.Popen(command, cwd=ENTRY, env=env)))
    rows = {}
    for spec, out, t0, process in processes:
        code = process.wait()
        elapsed = time.perf_counter() - t0
        print(f"done {spec['name']} exit {code} in {elapsed:.1f}s", flush=True)
        if code != 0:
            raise SystemExit(f"{spec['name']} failed with exit {code}")
        rows[spec["name"]] = read_predictions(out)
    blended = blend([rows[name] for name in win_names], rows[SIZE_MEMBER], floor=True)
    write_predictions(out_path, blended)
    print(f"wrote {out_path} ({len(blended)} rows) in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
