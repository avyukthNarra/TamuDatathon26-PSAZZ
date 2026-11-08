"""Per-machine-type payout frequencies (no model, no history at prediction time). Standard library only.

COUNTS are the payout counts (0, 1, 2, 5, 20) observed in the released main train split's
history, grouped by machine_type. They are baked in because the organizer rerun mounts only
the current test round, not train data. Prediction is Laplace-smoothed frequency per type.
"""
import argparse
import csv
from pathlib import Path

COUNTS = {
    "steady": (21278, 17977, 8828, 2293, 47),
    "balanced": (29375, 7114, 9035, 3453, 590),
    "jackpot": (30989, 1889, 2661, 3261, 1154),
}
GLOBAL = tuple(sum(column) for column in zip(*COUNTS.values()))


def probabilities(machine_type):
    counts = COUNTS.get(machine_type, GLOBAL)
    total = sum(counts) + 5
    return [(c + 1) / total for c in counts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with (Path(args.data) / "test" / "queries.csv").open(newline="", encoding="utf-8") as stream:
        queries = list(csv.DictReader(stream))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"])
        for row in queries:
            writer.writerow([row["query_id"]] + [format(p, ".17g") for p in probabilities(row["machine_type"])])


if __name__ == "__main__":
    main()
