"""v18: v17 (pranav/super + round-two guards) blended with Sai's exact Bayesian filter.

P(win) = sigmoid(0.5 * logit(v17 win) + 0.5 * logit(exact win)); payout size given a win = v17's.
The exact filter (research_exact/model.py, Sai, sai/master-test) fits on train only and learns
the round's lag-pair prior from the kit's own histories. Main validation: v17 57.69, exact 57.76,
blend 57.94 (cross-fit halves 58.05 / 57.84).
"""
import argparse
import csv
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

COLS = ("p_0", "p_1", "p_2", "p_5", "p_20")
EXACT_WEIGHT = 0.5
FLOOR = 1e-5


def logit(p):
    p = min(max(p, FLOOR), 1 - FLOOR)
    return math.log(p / (1 - p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    data = Path(args.data)
    if (data / "public").is_dir():
        data = data / "public"

    with tempfile.TemporaryDirectory() as tmp:
        v17_csv = Path(tmp) / "v17.csv"
        subprocess.run([sys.executable, str(HERE / "predict_v17.py"), "--data", str(data), "--out", str(v17_csv),
                        "--split", args.split], check=True)
        with open(v17_csv, newline="") as f:
            rows = list(csv.DictReader(f))

    from research_exact.model import ExactModel, fit, load_machines
    params = fit(data)
    machines, *_ = load_machines(data, args.split)
    exact, _ = ExactModel(params).predict_win(machines, verbose=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("query_id",) + COLS)
        for row in rows:
            p = [float(row[c]) for c in COLS]
            w17 = 1 - p[0]
            z = (1 - EXACT_WEIGHT) * logit(w17) + EXACT_WEIGHT * logit(float(exact[row["query_id"]]))
            win = 1 / (1 + math.exp(-z))
            sizes = [x / w17 for x in p[1:]] if w17 > 0 else [0.25] * 4
            vec = [1 - win] + [win * s for s in sizes]
            total = math.fsum(vec)
            writer.writerow([row["query_id"]] + [repr(x / total) for x in vec])
    print("wrote", out)


if __name__ == "__main__":
    main()
