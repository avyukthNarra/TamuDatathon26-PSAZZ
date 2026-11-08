"""Uniform forecast: 0.2 for every payout. Standard library only."""
import argparse
import csv
from pathlib import Path

UNIFORM = (0.2, 0.2, 0.2, 0.2, 0.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with (Path(args.data) / "test" / "queries.csv").open(newline="", encoding="utf-8") as stream:
        ids = [row["query_id"] for row in csv.DictReader(stream)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"])
        for qid in ids:
            writer.writerow([qid] + [format(p, ".17g") for p in UNIFORM])


if __name__ == "__main__":
    main()
