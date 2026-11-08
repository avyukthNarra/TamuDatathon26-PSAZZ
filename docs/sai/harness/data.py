"""Dense tick grids. NaN = unobserved (never 0). Only ticks strictly below cutoff are kept."""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = Path(os.environ.get("HE_DATA", ROOT / "mainFiles" / "public"))   # round two: set HE_DATA
CACHE = HERE.parent / "cache"
PAY = (0, 1, 2, 5, 20)
PC = ["p_0", "p_1", "p_2", "p_5", "p_20"]
CONDITIONS = ("stable", "short_history", "drift", "shared_change")
PAYIDX = {v: i for i, v in enumerate(PAY)}


def _rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class Split:
    """One public split. Row i of every per-query array is queries.csv row i.

    grid[i, t]   payout of query i's machine at tick t, NaN if unobserved or t >= cutoff.
    ep_idx[i]    integer episode index; peers(i) = other machines in the same episode.
    Validation/train extras: outcome (payout value), condition, truth (validation, true q), ref
    (validation, frozen pooled reference).
    """

    def __init__(self, name, a, meta):
        self.name = name
        self.meta = meta
        self.max_points = float(meta.get("max_points", 60))
        self.round = meta.get("round", "main")
        for k in ("qid", "episode", "machine", "mtype", "signal", "cutoff", "grid"):
            setattr(self, k, a[k])
        self.n = len(self.qid)
        self.episodes = sorted(set(self.episode.tolist()))
        eidx = {e: i for i, e in enumerate(self.episodes)}
        self.ep_idx = np.array([eidx[e] for e in self.episode])
        self.outcome = a["outcome"] if "outcome" in a else None
        self.condition = a["condition"] if "condition" in a else None
        self.truth = a["truth"] if "truth" in a else None
        self.ref = a["ref"] if "ref" in a else None
        self.truth_outcome = a["truth_outcome"] if "truth_outcome" in a else None
        self._peers = None

    @property
    def win(self):
        """1.0 win (payout>0), 0.0 loss, NaN unobserved."""
        g = self.grid
        return np.where(np.isnan(g), np.nan, (g > 0).astype(float))

    def peers(self, i):
        if self._peers is None:
            self._peers = {}
            for j, e in enumerate(self.ep_idx):
                self._peers.setdefault(int(e), []).append(j)
        return [j for j in self._peers[int(self.ep_idx[i])] if j != i]

    def episode_rows(self, ep):
        return np.flatnonzero(self.ep_idx == self.episodes.index(ep))

    def seq(self, i):
        """{tick: payout} for query i (ticks < cutoff only)."""
        t = np.flatnonzero(~np.isnan(self.grid[i]))
        return {int(k): int(self.grid[i, k]) for k in t}

    def subset(self, rows):
        rows = np.asarray(rows)
        a = {k: getattr(self, k)[rows] for k in ("qid", "episode", "machine", "mtype", "signal", "cutoff", "grid")}
        for k in ("outcome", "condition", "truth", "ref", "truth_outcome"):
            if getattr(self, k) is not None:
                a[k] = getattr(self, k)[rows]
        return Split(self.name + "-subset", a, self.meta)

    def frame(self):
        import pandas as pd
        return pd.DataFrame({"query_id": self.qid, "episode_id": self.episode, "machine_id": self.machine,
                             "machine_type": self.mtype, "signal": self.signal, "cutoff": self.cutoff})


def _file_hash(*paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


def load_split(name, data=None, use_cache=True):
    root = Path(data) if data else DATA
    d = root / name
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8")) if (root / "metadata.json").is_file() else {}
    key = _file_hash(d / "queries.csv", d / "history.csv")
    cpath = CACHE / f"split_{name}_{key}.npz"
    if use_cache and cpath.is_file():
        z = np.load(cpath, allow_pickle=False)
        return Split(name, {k: z[k] for k in z.files}, meta)
    q = _rows(d / "queries.csv")
    n = len(q)
    index = {(r["episode_id"], r["machine_id"]): i for i, r in enumerate(q)}
    cutoff = np.array([int(r["cutoff"]) for r in q])
    maxt = int(cutoff.max())
    grid = np.full((n, maxt), np.nan)
    dropped = 0
    for r in _rows(d / "history.csv"):
        i = index[(r["episode_id"], r["machine_id"])]
        t = int(r["tick"])
        if t >= cutoff[i]:
            dropped += 1        # never expose same/future ticks
            continue
        grid[i, t] = float(r["payout"])
    a = {"qid": np.array([r["query_id"] for r in q]), "episode": np.array([r["episode_id"] for r in q]),
         "machine": np.array([r["machine_id"] for r in q]), "mtype": np.array([r["machine_type"] for r in q]),
         "signal": np.array([int(r["signal"]) for r in q]), "cutoff": cutoff, "grid": grid}
    pos = {r["query_id"]: i for i, r in enumerate(q)}
    if (d / "labels.csv").is_file():
        o = np.full(n, -1)
        for r in _rows(d / "labels.csv"):
            o[pos[r["query_id"]]] = int(r["outcome"])
        a["outcome"] = o
    if (d / "conditions.csv").is_file():
        c = np.full(n, "", dtype=object)
        for r in _rows(d / "conditions.csv"):
            c[pos[r["query_id"]]] = r["condition"]
        a["condition"] = c.astype(str)
    if (d / "targets.csv").is_file():
        t = np.zeros((n, 5))
        to = np.full(n, -1)
        for r in _rows(d / "targets.csv"):
            t[pos[r["query_id"]]] = [float(r[k]) for k in PC]
            to[pos[r["query_id"]]] = int(r["outcome"])
        a["truth"], a["truth_outcome"] = t, to
    if (d / "reference.csv").is_file():
        ref = np.zeros((n, 5))
        for r in _rows(d / "reference.csv"):
            ref[pos[r["query_id"]]] = [float(r[k]) for k in PC]
        a["ref"] = ref
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cpath, **a)
    s = Split(name, a, meta)
    s.dropped = dropped
    return s
