#!/usr/bin/env python3
"""House Edge prototype: per-machine lag-pair model + holdout-selected memory.

Idea (see docs/sai/solution-playbook.md): each machine's win probability is
a machine-specific function of PAIRS of earlier win/loss flags at fixed lags. Fit a
small ridge model on the machine's OWN history, forget old ticks with an exponential
half-life chosen per machine by predicting its most recent HOLDOUT ticks, then split
the win probability over payout sizes with recency-weighted size shares.

Usage (from mainFiles/):
  python3 ../docs/sai/scripts/pair_model_prototype.py --data public --split validation
  python3 ../docs/sai/scripts/pair_model_prototype.py --data public --split test --out proto_submission.csv

Round two: re-run lag_screen.py on the new TRAIN histories and edit LAGS / PAIRS below.
Only ticks strictly below each query's cutoff are used. No condition labels are used.
Needs numpy + pandas (a stdlib port is recommended before packaging; see playbook).
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

PC = ["p_0", "p_1", "p_2", "p_5", "p_20"]
PAY = (1, 2, 5, 20)

# ---- edit after screening a new round -------------------------------------------
LAGS = (1, 2, 3, 4, 5, 7)             # single lags used as +-1 win flags
PAIRS = ((1, 3), (2, 5), (4, 7))      # product features of two lagged win flags
LAM = 20.0                            # ridge strength (intercept unpenalised)
HL_GRID = (1e9, 128, 64, 32, 16)      # candidate half-lives in ticks; 1e9 = no forgetting
HOLDOUT = 40                          # ticks held out to choose the half-life
SIZE_HL, SIZE_PRIOR = 64.0, 8.0       # recency + shrinkage for payout-size shares
MIN_ROW_TICK = 8                      # earliest tick used as a training row
GAPPY_FRAC = 0.6                      # fewer observed ticks than this*cutoff -> no forgetting
# ---------------------------------------------------------------------------------


def lag_flag(d, t, k):
    v = d.get(t - k)
    return 0.0 if v is None else (1.0 if v > 0 else -1.0)  # missing tick -> 0 (unknown, not a loss)


def design(d, t):
    return [1.0] + [lag_flag(d, t, k) for k in LAGS] + [lag_flag(d, t, a) * lag_flag(d, t, b) for a, b in PAIRS]


def ridge(X, y, ticks, ref, hl):
    w = 0.5 ** ((ref - 1 - ticks) / hl)
    pen = np.eye(X.shape[1]) * LAM
    pen[0, 0] = 1e-6
    return np.linalg.solve(X.T @ (w[:, None] * X) + pen, X.T @ (w * y))


def type_shares(queries, seqs):
    out = {}
    for ty, g in queries.groupby("machine_type"):
        c = np.zeros(4)
        for r in g.itertuples():
            for t, p in seqs[r.machine_id].items():
                if t < r.cutoff and p > 0:
                    c[PAY.index(p)] += 1
        out[ty] = c / c.sum()
    return out


def predict_one(r, seqs, tshare):
    d = seqs[r.machine_id]
    ticks = np.array([t for t in sorted(d) if MIN_ROW_TICK <= t < r.cutoff])
    X = np.array([design(d, t) for t in ticks])
    y = np.array([float(d[t] > 0) for t in ticks])
    hl = 1e9
    if sum(1 for t in d if t < r.cutoff) >= GAPPY_FRAC * r.cutoff:
        cut = r.cutoff - HOLDOUT
        tr, te = ticks < cut, ticks >= cut
        errs = []
        for c in HL_GRID:
            if tr.sum() < 30 or te.sum() < 10:
                errs.append(0.0 if c == 1e9 else 1.0)
                continue
            b = ridge(X[tr], y[tr], ticks[tr], cut, c)
            errs.append(np.mean((np.clip(X[te] @ b, 0, 1) - y[te]) ** 2))
        hl = HL_GRID[int(np.argmin(errs))]
    b = ridge(X, y, ticks, r.cutoff, hl)
    pw = float(np.clip(np.array(design(d, r.cutoff)) @ b, 0.005, 0.995))
    cnt = SIZE_PRIOR * tshare[r.machine_type].copy()
    for t, p in d.items():
        if t < r.cutoff and p > 0:
            cnt[PAY.index(p)] += 0.5 ** ((r.cutoff - 1 - t) / SIZE_HL)
    return np.concatenate([[1 - pw], pw * cnt / cnt.sum()]), hl


def score(P, q, ref, max_points):
    T = q[PC].values
    err = ((P - T) ** 2).sum(1)
    base = ((ref.loc[q.query_id, PC].values - T) ** 2).sum(1)
    df = pd.DataFrame({"ep": q.episode_id.values, "cond": q.condition.values, "e": err, "b": base})
    ep = df.groupby(["cond", "ep"]).agg(e=("e", "mean"), b=("b", "mean")).reset_index()
    c = ep.groupby("cond").agg(e=("e", "mean"), b=("b", "mean"))
    ratio = c.e / np.maximum(c.b, 1e-9)
    R = float(ratio.mean())
    print(f"mean 5-class error {err.mean():.4f}  R={R:.3f}  points={max_points / (1 + R):.2f} / {max_points}")
    print("condition loss ratios:", {k: round(float(v), 3) for k, v in ratio.items()})
    print("per-class squared error (share of total):",
          dict(zip(PC, [f"{v:.4f} ({v / err.mean():.0%})" for v in ((P - T) ** 2).mean(0)])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="public")
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--out")
    a = ap.parse_args()
    S = Path(a.data) / a.split
    q = pd.read_csv(S / "queries.csv")
    h = pd.read_csv(S / "history.csv")
    seqs = {k: dict(zip(g.tick.values, g.payout.values)) for k, g in h.groupby("machine_id")}
    tshare = type_shares(q, seqs)
    out = [predict_one(r, seqs, tshare) for r in q.itertuples()]
    P = np.array([o[0] for o in out])
    P = P / P.sum(1, keepdims=True)
    q["hl"] = [o[1] for o in out]
    if a.split == "validation":
        qq = q.merge(pd.read_csv(S / "targets.csv").drop(columns=["episode_id"]), on="query_id") \
              .merge(pd.read_csv(S / "conditions.csv"), on="query_id")
        order = qq.set_index("query_id").index
        P = pd.DataFrame(P, index=q.query_id).loc[order].values
        meta = json.loads((Path(a.data) / "metadata.json").read_text())
        score(P, qq, pd.read_csv(S / "reference.csv").set_index("query_id"), meta.get("max_points", 60))
        print("chosen half-life by condition:\n", pd.crosstab(qq.condition, q.set_index("query_id").loc[order, "hl"].values).to_string())
    if a.out:
        pd.DataFrame(P if a.split != "validation" else P, columns=PC).assign(
            query_id=(q.query_id.values if a.split != "validation" else order)).loc[:, ["query_id"] + PC].to_csv(a.out, index=False, float_format="%.10f")
        print("wrote", a.out)


if __name__ == "__main__":
    main()
