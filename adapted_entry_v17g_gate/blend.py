"""Equal log-odds blend of P(win), with one member's size split.

Called by entry/predict.py and entry/members/v13_short.py, exact_filter.py,
sysid_rule.py. wt/orch/blend_final.py does the same math on saved CSVs and is
not the entry. CSV columns are query_id, p_0, p_1, p_2, p_5, p_20 (no dates).
A synthetic row is query_id=q1, p_0=0.50, p_1=0.20, p_2=0.15, p_5=0.10, p_20=0.05.

Logit clip 1e-5. Weights default to equal. A fourth weight is v7c's, from the
train structure gate; 0 leaves the three-member average unchanged. Then
P = [1 - p_win] + p_win * size. The entry also floors at 1e-4 and renormalizes.
"""
import csv
import math
from pathlib import Path

COLUMNS = ("query_id", "p_0", "p_1", "p_2", "p_5", "p_20")


def logit(p):
    p = min(max(float(p), 1e-5), 1.0 - 1e-5)
    return math.log(p / (1.0 - p))


def blend(members, size_rows, *, floor=True, weights=None):
    """members: list of {query_id: [p_0..p_20]}. size_rows supplies P(size | win).

    weights are per member and do not need to sum to 1. None means equal.
    """
    if not members:
        raise ValueError("blend needs at least one win member")
    ids = sorted(size_rows)
    n = len(members)
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        raise ValueError("one weight per win member")
    wsum = math.fsum(weights)
    if wsum <= 0:
        raise ValueError("blend weights must sum to a positive number")
    out = {}
    for query_id in ids:
        for member in members:
            if query_id not in member:
                raise KeyError(f"member is missing {query_id}")
        mean_logit = sum(
            weight * logit(1.0 - member[query_id][0])
            for weight, member in zip(weights, members)
        ) / wsum
        p_win = 1.0 / (1.0 + math.exp(-mean_logit))
        size = list(size_rows[query_id][1:])
        total = math.fsum(size)
        if total <= 0:
            size = [0.25, 0.25, 0.25, 0.25]
        else:
            size = [value / total for value in size]
        raw = [1.0 - p_win] + [p_win * value for value in size]
        if floor:
            raw = [max(value, 1e-4) for value in raw]
        scale = math.fsum(raw)
        out[query_id] = [value / scale for value in raw]
    return out


def read_predictions(path):
    rows = {}
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            rows[row["query_id"]] = [float(row[column]) for column in COLUMNS[1:]]
    return rows


def write_predictions(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for query_id in sorted(rows):
            values = rows[query_id]
            writer.writerow({
                "query_id": query_id,
                **{column: format(float(value), ".17g") for column, value in zip(COLUMNS[1:], values)},
            })
