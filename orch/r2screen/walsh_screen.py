"""Sign-invariant lag-structure screen for slot-machine kits (numpy only).

For each (episode, machine) series in train/validation/test histories, and for
every lag subset S of {1..8} with |S| in {1,2,3} (92 subsets), compute

    chi_S(t) = prod_{j in S} z_{t-j},   z = 2y - 1
    c_S      = mean over usable t of (y_t - pbar) * chi_S(t)
    E_m(S)   = T_m * c_S^2 / (pbar * (1 - pbar))

where a tick t is usable when y_t and every lag in S are observed. Under no
effect E_m(S) ~ chi2(1), so the mean of E_m(S) - 1 over machines is ~0 for
absent structure and about sqrt(2/n) in standard error.

Usage: python3 walsh_screen.py <kit_dir>
"""

import csv
import itertools
import sys
from pathlib import Path

import numpy as np

SPLITS = ("train", "validation", "test")
TYPES = ("steady", "balanced", "jackpot")
SUBSETS = [c for k in (1, 2, 3) for c in itertools.combinations(range(1, 9), k)]
TOP = 10


def load_kit(kit: Path):
    """Return {(split, episode, machine): (machine_type, y array with NaN gaps)}."""
    series = {}
    for split in SPLITS:
        base = kit / split
        qtype = {}
        with open(base / "queries.csv", newline="") as f:
            for row in csv.DictReader(f):
                qtype[(row["episode_id"], row["machine_id"])] = row["machine_type"]
        hist = {}
        with open(base / "history.csv", newline="") as f:
            for row in csv.DictReader(f):
                key = (row["episode_id"], row["machine_id"])
                hist.setdefault(key, {})[int(row["tick"])] = float(int(float(row["payout"])) > 0)
        for key, ticks in hist.items():
            if key not in qtype:
                continue
            length = max(ticks) + 1
            y = np.full(length, np.nan)
            for t, v in ticks.items():
                y[t] = v
            series[(split,) + key] = (qtype[key], y)
    return series


def series_energy(y: np.ndarray) -> np.ndarray:
    """E_m(S) for all 92 subsets of one series (zeros if pbar is degenerate)."""
    out = np.zeros(len(SUBSETS))
    obs = ~np.isnan(y)
    if not obs.any():
        return out
    yy = np.where(obs, y, 0.0)
    pbar = yy[obs].mean()
    var = pbar * (1.0 - pbar)
    if var <= 0:
        return out
    z = np.where(obs, 2.0 * yy - 1.0, np.nan)
    length = len(y)
    lagged = {}
    for j in range(1, 9):
        s = np.full(length, np.nan)
        if j < length:
            s[j:] = z[:-j]
        lagged[j] = s
    for idx, subset in enumerate(SUBSETS):
        usable = obs.copy()
        chi = np.ones(length)
        for j in subset:
            usable &= ~np.isnan(lagged[j])
            chi = chi * lagged[j]
        n_use = int(usable.sum())
        if n_use == 0:
            continue
        c = np.mean((yy[usable] - pbar) * chi[usable])
        out[idx] = n_use * c * c / var
    return out


def label(subset) -> str:
    return "(" + ",".join(str(j) for j in subset) + ")"


def top_table(E: np.ndarray, title: str) -> str:
    n = E.shape[0]
    excess = E.mean(axis=0) - 1.0
    z = excess / np.sqrt(2.0 / n)
    order = np.argsort(-excess)
    lines = [f"### {title} (n_machines = {n})", "",
             "| rank | subset | size | mean E-1 | z |", "|---|---|---|---|---|"]
    for r, idx in enumerate(order[:TOP], 1):
        lines.append(f"| {r} | {label(SUBSETS[idx])} | {len(SUBSETS[idx])} | "
                     f"{excess[idx]:.3f} | {z[idx]:.1f} |")
    lines.append("")
    best = {}
    for k in (1, 2, 3):
        cand = [i for i, s in enumerate(SUBSETS) if len(s) == k]
        i = max(cand, key=lambda i: excess[i])
        best[k] = (label(SUBSETS[i]), excess[i], z[i])
    lines.append("Best per size: " + "; ".join(
        f"size {k}: {v[0]} mean {v[1]:.3f} z {v[2]:.1f}" for k, v in best.items()))
    lines.append("")
    return "\n".join(lines), excess, z


def main(kit_arg: str) -> None:
    kit = Path(kit_arg)
    series = load_kit(kit)
    keys = list(series)
    types = np.array([series[k][0] for k in keys])
    E = np.vstack([series_energy(series[k][1]) for k in keys])
    print(f"kit: {kit}")
    print(f"series (episode x machine): {len(keys)}  per type: " +
          ", ".join(f"{t}={int((types == t).sum())}" for t in TYPES))
    print()
    overall_txt, excess, z = top_table(E, "Overall")
    print(overall_txt)
    for t in TYPES:
        txt, _, _ = top_table(E[types == t], f"Type {t}")
        print(txt)
    # Per-machine winner: subset with largest E_m(S), counted over machines
    # with any usable data.
    valid = E.sum(axis=1) > 0
    winners = np.argmax(E[valid], axis=1)
    counts = np.bincount(winners, minlength=len(SUBSETS))
    order = np.argsort(-counts)
    print(f"### Per-machine winner counts (argmax E_m(S); {int(valid.sum())} machines)")
    print()
    print("| subset | size | wins | share |")
    print("|---|---|---|---|")
    for idx in order[:15]:
        if counts[idx] == 0:
            break
        print(f"| {label(SUBSETS[idx])} | {len(SUBSETS[idx])} | {counts[idx]} | "
              f"{counts[idx] / valid.sum():.3f} |")
    print()
    print("singleton excess:", ", ".join(
        f"{label(SUBSETS[i])}={excess[i]:.3f}(z={z[i]:.1f})"
        for i in range(len(SUBSETS)) if len(SUBSETS[i]) == 1))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
