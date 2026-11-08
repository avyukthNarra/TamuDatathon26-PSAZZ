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
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from blend import blend, read_predictions, write_predictions
from members.hooks import REGISTRY, SIZE_MEMBER
import peer_term

ENTRY = Path(__file__).resolve().parent


def with_validation_answers(data):
    """pranav/super: make the forecast identical whatever part of this round's public validation
    folder the runner mounts. Members build their priors from validation queries + answers and fall
    back to train-only priors without them, which changed the forecast by up to 0.047. The entry
    bundles this round's public validation folder at build time (bundled_validation/); when /data
    lacks validation/targets.csv, a scratch view of /data is made in the temp folder (links to every
    mounted file) and missing validation files are taken from the bundle."""
    root = data / "public" if (data / "public" / "train").is_dir() else data
    bundled = ENTRY / "bundled_validation"
    if (root / "validation" / "targets.csv").exists() or not bundled.is_dir():
        return root
    view = Path(tempfile.mkdtemp(prefix="data-view-"))
    for item in root.iterdir():
        if item.name != "validation":
            os.symlink(item, view / item.name)
    (view / "validation").mkdir()
    mounted = root / "validation"
    names = {item.name for item in mounted.iterdir()} if mounted.is_dir() else set()
    for name in names:
        os.symlink(mounted / name, view / "validation" / name)
    for item in bundled.iterdir():
        if item.name not in names:
            os.symlink(item, view / "validation" / item.name)
    print(f"validation folder incomplete: using the bundled copy via {view}", flush=True)
    return view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--protocol", choices=("submission", "crossfit"), default="submission")
    parser.add_argument("--member-dir", default="", help="keep each member CSV in this directory")
    args = parser.parse_args()
    started = time.perf_counter()
    data = str(with_validation_answers(Path(args.data).resolve()))
    out_path = Path(args.out).resolve()
    env = os.environ.copy()
    hold = tempfile.TemporaryDirectory(prefix="blend-members-")
    member_dir = Path(args.member_dir).resolve() if args.member_dir else Path(hold.name)
    member_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for spec in REGISTRY:
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
    blended = blend([rows[spec["name"]] for spec in REGISTRY], rows[SIZE_MEMBER], floor=True)
    blended = peer_term.apply(blended, Path(data) / args.split, peer_term.load_coef())  # v21; off without peer.json
    write_predictions(out_path, blended)
    print(f"wrote {out_path} ({len(blended)} rows) in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
