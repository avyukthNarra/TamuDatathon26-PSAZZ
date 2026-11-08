"""Train-only share of machines a lag-pair parity does not explain.

For each machine, even ticks and odd ticks take turns as the fit half.
The fit half picks one pair-parity, one 3-lag parity, and one 2-lag AND.
The other half scores them. A machine counts as poorly explained when the
better of the 3-lag parity and the AND beats the pair on the held-out ticks.

A second pass redraws each series from its own fitted pair rule, keeping the
real missingness, and repeats the test. That rate is the pair-world false
positive rate for this round. Nothing here reads validation or test files.

Usage: python3 structure_health.py KIT_OR_PUBLIC [LABEL]
"""
import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

PAIRS = list(combinations(range(8), 2))
TRIPLES = list(combinations(range(8), 3))


def public_dir(path):
    path = Path(path)
    if (path / "public").is_dir():
        return path / "public"
    return path


def load_train(kit):
    base = public_dir(kit) / "train"
    query_key = {}
    with (base / "queries.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            query_key[row["query_id"]] = (row["episode_id"], row["machine_id"])
    condition = {}
    cond_path = base / "conditions.csv"
    if cond_path.exists():
        with cond_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                key = query_key.get(row["query_id"])
                if key is not None:
                    condition[key] = row["condition"]
    hist = {}
    with (base / "history.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["episode_id"], row["machine_id"])
            hist.setdefault(key, {})[int(row["tick"])] = float(int(float(row["payout"])) > 0)
    series = []
    for key, ticks in hist.items():
        length = max(ticks) + 1
        y = np.full(length, np.nan)
        for tick, value in ticks.items():
            y[tick] = value
        series.append((condition.get(key, "?"), y))
    return series


def lagged_z(y):
    obs = ~np.isnan(y)
    z = np.where(obs, 2.0 * y - 1.0, np.nan)
    length = len(y)
    lagged = np.full((8, length), np.nan)
    for lag in range(1, 9):
        if lag < length:
            lagged[lag - 1, lag:] = z[:-lag]
    return obs, lagged


def candidate_ll(chi, usable, y, fit_mask):
    """Smoothed two-cell Bernoulli LL. Returns (fit, hold, p1, p0) or None."""
    fit = usable & fit_mask
    hold = usable & ~fit_mask
    pos = chi > 0
    n1 = int(np.count_nonzero(fit & pos))
    n0 = int(np.count_nonzero(fit & ~pos))
    if n1 < 1 or n0 < 1 or int(np.count_nonzero(hold)) < 1:
        return None
    k1 = float(y[fit & pos].sum())
    k0 = float(y[fit & ~pos].sum())
    p1 = (k1 + 1.0) / (n1 + 2.0)
    p0 = (k0 + 1.0) / (n0 + 2.0)
    log_p1, log_q1 = np.log(p1), np.log(1.0 - p1)
    log_p0, log_q0 = np.log(p0), np.log(1.0 - p0)

    def ll(mask):
        yy = y[mask]
        on = chi[mask] > 0
        return float(np.sum(np.where(
            on,
            yy * log_p1 + (1.0 - yy) * log_q1,
            yy * log_p0 + (1.0 - yy) * log_q0,
        )))

    return ll(fit), ll(hold), p1, p0


def chi_of(lagged, subset, mode):
    length = lagged.shape[1]
    usable = np.ones(length, dtype=bool)
    if mode == "and":
        bit = np.ones(length)
        for lag in subset:
            col = lagged[lag]
            known = ~np.isnan(col)
            usable &= known
            bit = bit * np.where(known, (col + 1.0) * 0.5, 1.0)
        chi = np.where(usable, 2.0 * bit - 1.0, 0.0)
    else:
        chi = np.ones(length)
        for lag in subset:
            col = lagged[lag]
            known = ~np.isnan(col)
            usable &= known
            chi = chi * np.where(known, col, 1.0)
        chi = np.where(usable, chi, 0.0)
    return chi, usable


def best_family(y, lagged, obs, fit_mask, subsets, mode):
    best_fit = -np.inf
    best = None
    for subset in subsets:
        chi, usable = chi_of(lagged, subset, mode)
        scored = candidate_ll(chi, usable & obs, y, fit_mask)
        if scored is None or scored[0] <= best_fit:
            continue
        best_fit = scored[0]
        best = (scored[0], scored[1], subset)
    return best


def crossfit_gap(y):
    """Holdout LL(best alt) - holdout LL(best pair). +inf if no pair fits."""
    obs, lagged = lagged_z(y)
    ticks = np.arange(len(y))
    pair_hold = 0.0
    alt_hold = 0.0
    used = 0
    and_votes = 0
    triple_votes = 0
    for fit_even in (True, False):
        fit_mask = (ticks % 2 == 0) if fit_even else (ticks % 2 == 1)
        pair = best_family(y, lagged, obs, fit_mask, PAIRS, "parity")
        triple = best_family(y, lagged, obs, fit_mask, TRIPLES, "parity")
        conj = best_family(y, lagged, obs, fit_mask, PAIRS, "and")
        cands = []
        if triple is not None:
            cands.append(("triple", triple[1]))
        if conj is not None:
            cands.append(("and", conj[1]))
        if pair is None or not cands:
            continue
        used += 1
        pair_hold += pair[1]
        name, hold = max(cands, key=lambda item: item[1])
        alt_hold += hold
        if name == "and":
            and_votes += 1
        else:
            triple_votes += 1
    if used == 0:
        return np.inf, "none"
    winner = "and" if and_votes > triple_votes else "triple"
    return alt_hold - pair_hold, winner


def best_pair_insample(y):
    obs, lagged = lagged_z(y)
    fit_mask = np.ones(len(y), dtype=bool)
    # Park the last observed tick as a dummy hold so the LL helper applies.
    observed = np.flatnonzero(obs)
    if len(observed) < 4:
        return None
    fit_mask[observed[-1]] = False
    best = best_family(y, lagged, obs, fit_mask, PAIRS, "parity")
    if best is None:
        return None
    return best[2]


def simulate_pair(y, subset, rng):
    """Redraw observed ticks from the fitted pair. Lags use the redraw."""
    obs, lagged = lagged_z(y)
    chi, usable = chi_of(lagged, subset, "parity")
    usable = usable & obs
    pos = usable & (chi > 0)
    neg = usable & (chi < 0)
    n1 = int(np.count_nonzero(pos))
    n0 = int(np.count_nonzero(neg))
    if n1 < 1 or n0 < 1:
        return None
    p1 = (float(y[pos].sum()) + 1.0) / (n1 + 2.0)
    p0 = (float(y[neg].sum()) + 1.0) / (n0 + 2.0)
    base_n = int(np.count_nonzero(obs))
    base = (float(y[obs].sum()) + 1.0) / (base_n + 2.0)
    out = np.full(len(y), np.nan)
    for tick in range(len(y)):
        if not obs[tick]:
            continue
        ready = True
        parity = 1.0
        for lag in subset:
            src = tick - (lag + 1)
            if src < 0 or np.isnan(out[src]):
                ready = False
                break
            parity *= 1.0 if out[src] > 0.5 else -1.0
        if not ready:
            p = base
        elif parity > 0:
            p = p1
        else:
            p = p0
        out[tick] = 1.0 if rng.random() < p else 0.0
    return out


def run_kit(kit, label, rng):
    series = load_train(kit)
    real_gaps = []
    null_by = {}
    by = {}
    and_wins = 0
    triple_wins = 0
    for cond, y in series:
        gap, winner = crossfit_gap(y)
        real_gaps.append(gap)
        by.setdefault(cond, []).append(gap)
        if gap > 0 and winner == "and":
            and_wins += 1
        elif gap > 0 and winner == "triple":
            triple_wins += 1
        subset = best_pair_insample(y)
        if subset is None:
            null_by.setdefault(cond, []).append(0.0)
            continue
        drawn = simulate_pair(y, subset, rng)
        if drawn is None:
            null_by.setdefault(cond, []).append(0.0)
            continue
        ngap, _ = crossfit_gap(drawn)
        null_by.setdefault(cond, []).append(ngap)
    real_d = np.array(real_gaps, dtype=float)
    null_d = np.array([g for gaps in null_by.values() for g in gaps], dtype=float)
    real_poor = (~np.isfinite(real_d)) | (real_d > 0)
    null_poor = (~np.isfinite(null_d)) | (null_d > 0)
    share = float(real_poor.mean())
    share0 = float(null_poor.mean())
    excess = max(0.0, share - share0)
    finite = real_d[np.isfinite(real_d)]
    mean_d = float(finite.mean()) if len(finite) else float("nan")
    cond_txt = []
    for cond in sorted(by):
        gaps = np.array(by[cond], dtype=float)
        base = np.array(null_by.get(cond, []), dtype=float)
        poor = float(((~np.isfinite(gaps)) | (gaps > 0)).mean())
        poor0 = float(((~np.isfinite(base)) | (base > 0)).mean()) if len(base) else float("nan")
        fin = gaps[np.isfinite(gaps)]
        md = float(fin.mean()) if len(fin) else float("nan")
        if len(fin):
            q = np.quantile(fin, [0.1, 0.5, 0.9])
            qtxt = f" q={q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f}"
        else:
            qtxt = ""
        cond_txt.append(f"{cond}={poor:.3f}/{poor0:.3f}/md={md:.2f}{qtxt}")
    ref = min(by, key=lambda cond: np.median(np.array(by[cond], dtype=float)[np.isfinite(by[cond])]))
    real_ref = np.array(by[ref], dtype=float)
    null_ref = np.array(null_by.get(ref, []), dtype=float)
    real_ref = np.where(np.isfinite(real_ref), real_ref, 0.0)
    null_ref = np.where(np.isfinite(null_ref), null_ref, 0.0)
    tail = float(np.maximum(real_ref, 0.0).mean())
    tail0 = float(np.maximum(null_ref, 0.0).mean()) if len(null_ref) else 0.0
    e = max(0.0, tail - tail0)
    winners = real_ref[real_ref < 0]
    scale = float(-winners.mean()) if len(winners) else 1.0
    frac = min(e / scale, 1.0 - 1.0 / max(len(real_ref), 2))
    weight = 3.0 * frac / (1.0 - frac)
    print(
        f"{label} n={len(series)} S={share:.4f} S0={share0:.4f} excess={excess:.4f} "
        f"mean_d={mean_d:.3f} and_wins={and_wins} triple_wins={triple_wins} "
        f"ref={ref} tail={tail:.3f} tail0={tail0:.3f} e={e:.3f} scale={scale:.2f} "
        f"frac={frac:.4f} w={weight:.4f} "
        + " ".join(cond_txt),
        flush=True,
    )
    return {
        "label": label,
        "ref": ref,
        "tail": tail,
        "tail0": tail0,
        "excess_tail": e,
        "scale": scale,
        "frac": frac,
        "weight": weight,
    }


def main():
    rng = np.random.default_rng(0)
    kit = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else Path(kit).name
    run_kit(kit, label, rng)


if __name__ == "__main__":
    main()
