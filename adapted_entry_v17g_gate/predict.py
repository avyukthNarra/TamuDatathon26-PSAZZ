"""Equal log-odds blend of v13+short, the exact filter, and sysid.

    python predict.py --data /data --out /output/submission.csv

Fits each member on /data/train. Validation targets, when present, are priors
only (v13 size modes and win prior; exact level prior). Sysid never reads them.
P(win) is the sigmoid of the mean logit. The three members stay at weight 1.
v7c is added only when the train structure gate opens and member-dir/v7c.csv
is present. The size split is v13's.
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
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from blend import blend, read_predictions, write_predictions
from members.hooks import REGISTRY, SIZE_MEMBER
from structure_health import run_kit

ENTRY = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--protocol", choices=("submission", "crossfit"), default="submission")
    parser.add_argument("--member-dir", default="", help="keep each member CSV in this directory")
    args = parser.parse_args()
    started = time.perf_counter()
    data = str(Path(args.data).resolve())
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
    gate = run_kit(data, "round", np.random.default_rng(0))
    win_rows = [rows[spec["name"]] for spec in REGISTRY]
    weights = [1.0] * len(win_rows)
    v7c_path = member_dir / "v7c.csv"
    # v7c.csv is that member's cross-fit (or submission) forecast. The gate
    # weight is learned on this round's train. Weight 0 skips the file.
    if gate["weight"] > 0 and v7c_path.exists():
        win_rows.append(read_predictions(v7c_path))
        weights.append(gate["weight"])
    print(
        f"v7c gate ref={gate['ref']} tail={gate['tail']:.3f} "
        f"null={gate['tail0']:.3f} weight={gate['weight']:.4f}",
        flush=True,
    )
    blended = blend(win_rows, rows[SIZE_MEMBER], floor=True, weights=weights)
    write_predictions(out_path, blended)
    print(f"wrote {out_path} ({len(blended)} rows) in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
