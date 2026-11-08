"""Deliberately leaky dummy: uses history ticks at or after the query cutoff. Standard library only.

Those ticks are the future relative to the prediction, so a valid entry must never read them.
With no future ticks present the output is uniform; appending future ticks changes it. The
perturbation checks exist to catch exactly this.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

PAYOUTS = (0, 1, 2, 5, 20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    test = Path(args.data) / "test"
    with (test / "queries.csv").open(newline="", encoding="utf-8") as stream:
        queries = list(csv.DictReader(stream))
    cutoffs = {(q["episode_id"], q["machine_id"]): int(q["cutoff"]) for q in queries}
    future = defaultdict(lambda: [0] * len(PAYOUTS))
    history_path = test / "history.csv"
    if history_path.is_file():
        with history_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = (row["episode_id"], row["machine_id"])
                if key in cutoffs and int(row["tick"]) >= cutoffs[key]:
                    future[key][PAYOUTS.index(int(row["payout"]))] += 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"])
        for row in queries:
            counts = [1 + 10 * c for c in future[(row["episode_id"], row["machine_id"])]]
            total = sum(counts)
            writer.writerow([row["query_id"]] + [format(c / total, ".17g") for c in counts])


if __name__ == "__main__":
    main()
