"""Deliberately nondeterministic dummy: uniform plus random noise of up to 1e-6, renormalised.

Two runs differ by about 1e-6, far above the 1e-8 determinism tolerance. Standard library only.
"""
import argparse
import csv
import random
from pathlib import Path

NOISE = 1e-6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rng = random.Random()  # unseeded on purpose
    with (Path(args.data) / "test" / "queries.csv").open(newline="", encoding="utf-8") as stream:
        ids = [row["query_id"] for row in csv.DictReader(stream)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"])
        for qid in ids:
            raw = [0.2 + rng.uniform(-NOISE, NOISE) for _ in range(5)]
            total = sum(raw)
            writer.writerow([qid] + [format(p / total, ".17g") for p in raw])


if __name__ == "__main__":
    main()
