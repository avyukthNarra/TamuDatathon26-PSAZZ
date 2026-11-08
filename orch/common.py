"""Shared loaders and the official-equivalent scorer (numpy) for orchestrator experiments."""
import csv, collections
from pathlib import Path
import numpy as np

PUB = Path(__file__).resolve().parent.parent / "mainFiles" / "public"
SIZES = [1, 2, 5, 20]
PCOLS = ["p_0", "p_1", "p_2", "p_5", "p_20"]


def load(split, pub=PUB):
    d = pub / split
    Q = list(csv.DictReader(open(d / "queries.csv")))
    H = collections.defaultdict(dict)
    for r in csv.DictReader(open(d / "history.csv")):
        H[r["machine_id"]][int(r["tick"])] = int(r["payout"])
    out = {"Q": Q, "H": H}
    for name in ("targets", "reference"):
        f = d / f"{name}.csv"
        if f.exists():
            out[name] = {r["query_id"]: np.array([float(r[c]) for c in PCOLS]) for r in csv.DictReader(open(f))}
    f = d / "conditions.csv"
    if f.exists():
        out["cond"] = {r["query_id"]: r["condition"] for r in csv.DictReader(open(f))}
    f = d / "labels.csv"
    if f.exists():
        out["labels"] = {r["query_id"]: int(r["outcome"]) for r in csv.DictReader(open(f))}
    return out


def points(pred, data, qids=None, max_points=60):
    """Official formula: SE per query -> mean per episode -> mean per condition -> ratio vs reference."""
    T, R, C = data["targets"], data["reference"], data["cond"]
    ep = {q["query_id"]: q["episode_id"] for q in data["Q"]}
    qids = list(T) if qids is None else qids
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for q in qids:
        agg[C[q]][ep[q]].append((((pred[q] - T[q]) ** 2).sum(), ((R[q] - T[q]) ** 2).sum()))
    ratios = {}
    for c, eps in agg.items():
        e = np.mean([np.mean([a for a, _ in v]) for v in eps.values()])
        b = np.mean([np.mean([b for _, b in v]) for v in eps.values()])
        ratios[c] = e / max(b, 1e-9)
    Rm = np.mean(list(ratios.values()))
    return max_points / (1 + Rm), ratios


def folds(data):
    """Brief protocol: episodes sorted by id within condition, alternating A/B."""
    ep_c = {}
    for q in data["Q"]:
        ep_c[q["episode_id"]] = data["cond"][q["query_id"]]
    A, B = set(), set()
    for c in sorted(set(ep_c.values())):
        for i, e in enumerate(sorted(e for e in ep_c if ep_c[e] == c)):
            (A if i % 2 == 0 else B).add(e)
    return A, B
