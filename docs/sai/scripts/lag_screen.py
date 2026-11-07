#!/usr/bin/env python3
"""Screen a round's machine histories for recurring lag effects and lag-pair interactions.

Sign-agnostic: for every machine, test whether win/loss at tick t depends on the win/loss
flags at t-a (single lag) or on the PRODUCT of flags at t-a and t-b (pair interaction,
controlling for both main effects). Report the mean of (z^2 - 1) across machines of a type.
Under no effect it is ~0 (plus a baseline from slow regime drift); machines of one type may
react in opposite directions and still show a large positive value.
Lags 9..16 are a NULL CONTROL: the simulator only interacts up to lag 8, so anything the
in-range lags show beyond the 9..16 level is real. (Lag 14 = 2x7 echoes a real lag-7 effect.)

Usage (from mainFiles/ or the new round's kit folder):
  python3 ../docs/sai/scripts/lag_screen.py --data public --splits train validation
Run it on EACH ROUND's own histories, then edit LAGS / PAIRS in pair_model_prototype.py.
Uses observed payouts only (no targets, no labels).
"""
import argparse, itertools
from pathlib import Path
import numpy as np
import pandas as pd

TYPES = ("steady", "balanced", "jackpot")


def flags(d, thr):
    w = np.full(max(d) + 1, np.nan)
    for k, v in d.items():
        w[k] = 1.0 if v >= thr else -1.0
    return w


def z_single(w, a):
    x, y = w[:-a], w[a:]
    ok = ~np.isnan(x) & ~np.isnan(y)
    n = ok.sum()
    if n < 40 or x[ok].std() == 0 or y[ok].std() == 0:
        return np.nan
    return np.corrcoef(x[ok], y[ok])[0, 1] * np.sqrt(n - 2)


def z_pair(w, a, b, mincell=6):
    t = np.arange(max(a, b), len(w))
    y, xa, xb = w[t], w[t - a], w[t - b]
    ok = ~np.isnan(y) & ~np.isnan(xa) & ~np.isnan(xb)
    y, xa, xb = y[ok], xa[ok], xb[ok]
    n = len(y)
    if n < 60 or y.std() == 0:
        return np.nan
    if any(((xa == sa) & (xb == sb)).sum() < mincell for sa in (-1, 1) for sb in (-1, 1)):
        return np.nan  # near-constant flags make the product term degenerate
    X = np.column_stack([np.ones(n), xa, xb, xa * xb])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ beta
    var = (r @ r / (n - 4)) * np.linalg.inv(X.T @ X)[3, 3]
    return beta[3] / np.sqrt(var) if var > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="public")
    ap.add_argument("--splits", nargs="+", default=["train", "validation"])
    ap.add_argument("--thr", type=int, default=1, help="payout >= thr counts as a win (1 = any return; 2 = profit)")
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()
    machines = []
    for s in a.splits:
        h = pd.read_csv(Path(a.data) / s / "history.csv")
        q = pd.read_csv(Path(a.data) / s / "queries.csv").set_index("machine_id")
        for m, g in h.groupby("machine_id"):
            machines.append((q.loc[m, "machine_type"], flags(dict(zip(g.tick.values, g.payout.values)), a.thr)))
    print(f"{len(machines)} machines from splits {a.splits}; win = payout >= {a.thr}")
    print("\nSINGLE LAG excess z^2 (mean z^2 - 1). Lags 9-16 = null control.")
    print("lag  " + "  ".join(f"{t:>9s}" for t in TYPES))
    for lag in range(1, 17):
        row = []
        for ty in TYPES:
            z = np.array([z_single(w, lag) for t, w in machines if t == ty])
            row.append(np.nanmean(z ** 2) - 1)
        print(f"{lag:3d}  " + "  ".join(f"{v:9.2f}" for v in row) + ("   <- null" if lag > 8 else ""))
    print("\nPAIR interaction excess z^2 (product term). Null pairs: both lags in 9..16.")
    for ty in TYPES:
        ws = [w for t, w in machines if t == ty]
        res = {}
        for lo, hi in itertools.combinations(range(1, 17), 2):
            if not (hi <= 8 or lo >= 9):
                continue
            z = np.array([z_pair(w, lo, hi) for w in ws])
            z = z[~np.isnan(z)]
            if len(z) >= 30:
                res[(lo, hi)] = (np.mean(z ** 2) - 1, (z > 3).mean(), (z < -3).mean(), len(z))
        null = [v[0] for k, v in res.items() if k[0] >= 9]
        inn = sorted(((v, k) for k, v in res.items() if k[1] <= 8), reverse=True)
        print(f"[{ty}] null pairs: mean {np.mean(null):.2f}, max {np.max(null):.2f} | in-range mean {np.mean([v[0] for v, _ in inn]):.2f}")
        for v, k in inn[: a.top]:
            flag = "  <-- clearly above null" if v[0] > 3 * max(np.max(null), 1) else ""
            print(f"    pair {k}: excess z2 {v[0]:6.1f} | frac z>3 {v[1]:.2f}, z<-3 {v[2]:.2f} | n={v[3]}{flag}")


if __name__ == "__main__":
    main()
