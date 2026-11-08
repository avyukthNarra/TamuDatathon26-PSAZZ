#!/usr/bin/env python3
"""Forensics for the House Edge simulator's generative process.

Reads train histories and validation targets. Writes analysis_results.json
next to this file and prints a sectioned numeric summary.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/Users/avynarra/Documents/Datathon/worktrees/pranav-5629/mainFiles/public")
OUT = Path("/Users/avynarra/Documents/Datathon/research_pf/g1-data/analysis_results.json")

PAIRS = tuple((a, b) for a in range(1, 9) for b in range(a + 1, 9))
TRIPLES = tuple((a, b, c) for a in range(1, 9) for b in range(a + 1, 9) for c in range(b + 1, 9))
LAGS = tuple(range(1, 9))
PAYOUTS = (0, 1, 2, 5, 20)
SIZE_PAYOUTS = (1, 2, 5, 20)


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def bin_ll(wins: float, n: float) -> float:
    if n <= 0:
        return 0.0
    p = wins / n
    ll = 0.0
    if wins > 0:
        ll += wins * math.log(p)
    losses = n - wins
    if losses > 0:
        ll += losses * math.log(1.0 - p)
    return ll


def clamp(p: float, eps: float = 1e-6) -> float:
    return min(1.0 - eps, max(eps, p))


def logit(p: float) -> float:
    p = clamp(p)
    return math.log(p / (1.0 - p))


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    mid = len(xs) // 2
    if len(xs) % 2:
        return float(xs[mid])
    return 0.5 * (xs[mid - 1] + xs[mid])


def quantiles(xs, qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> dict:
    xs = sorted(xs)
    if not xs:
        return {str(q): None for q in qs}
    out = {}
    for q in qs:
        idx = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
        out[str(q)] = xs[idx]
    return out


def std(xs) -> float:
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def cluster_1d(values, tol: float):
    groups = []
    for value in sorted(values):
        if not groups or value - groups[-1][-1] > tol:
            groups.append([value])
        else:
            groups[-1].append(value)
    return groups


def load_split(name: str) -> dict:
    base = ROOT / name
    queries = load_csv(base / "queries.csv")
    conditions = {}
    cond_path = base / "conditions.csv"
    if cond_path.exists():
        for row in load_csv(cond_path):
            conditions[row["query_id"]] = row["condition"]
    targets = {}
    tgt_path = base / "targets.csv"
    if tgt_path.exists():
        for row in load_csv(tgt_path):
            probs = tuple(float(row[f"p_{p}"]) for p in PAYOUTS)
            targets[row["query_id"]] = probs
    episodes = {}
    for row in queries:
        eid = row["episode_id"]
        ep = episodes.setdefault(
            eid,
            {
                "episode_id": eid,
                "condition": conditions.get(row["query_id"]),
                "cutoff": int(row["cutoff"]),
                "machines": {},
            },
        )
        ep["machines"][row["machine_id"]] = {
            "machine_id": row["machine_id"],
            "query_id": row["query_id"],
            "machine_type": row["machine_type"],
            "signal": int(row["signal"]),
            "index": int(row["machine_id"].rsplit("m", 1)[-1]),
            "payouts": {},
        }
    for row in load_csv(base / "history.csv"):
        ep = episodes[row["episode_id"]]
        ep["machines"][row["machine_id"]]["payouts"][int(row["tick"])] = int(row["payout"])
    return {"episodes": episodes, "targets": targets}


def series_of(machine: dict, cutoff: int) -> list:
    payouts = machine["payouts"]
    return [payouts.get(t) for t in range(cutoff)]


def feat_gt0(payout: int) -> int:
    return int(payout > 0)


def feat_ge2(payout: int) -> int:
    return int(payout >= 2)


def feat_ge5(payout: int) -> int:
    return int(payout >= 5)


FEATS = {"gt0": feat_gt0, "ge2": feat_ge2, "ge5": feat_ge5}


def build_bits(series, feat) -> np.ndarray:
    """bit[t] is 0/1, or -1 if missing."""
    bits = np.full(len(series), -1, dtype=np.int8)
    for t, payout in enumerate(series):
        if payout is not None:
            bits[t] = feat(payout)
    return bits


def pair_prefix(bits: np.ndarray, a: int, b: int):
    """Prefix sums over outcome ticks. cell 1 = lags agree (XNOR), 0 = differ (XOR).

    Index t holds the sum of outcomes strictly before t. Outcome at t requires bits[t], bits[t-a], bits[t-b].
    """
    n = len(bits)
    count = np.zeros((2, n + 1), dtype=np.int32)
    wins = np.zeros((2, n + 1), dtype=np.int32)
    cells = np.full(n, -1, dtype=np.int8)
    for t in range(n):
        count[:, t + 1] = count[:, t]
        wins[:, t + 1] = wins[:, t]
        if t < b or bits[t] < 0 or bits[t - a] < 0 or bits[t - b] < 0:
            continue
        cell = 1 if bits[t - a] == bits[t - b] else 0
        cells[t] = cell
        count[cell, t + 1] += 1
        wins[cell, t + 1] += int(bits[t])
    return count, wins, cells


def range_stats(count, wins, lo: int, hi: int):
    n = count[:, hi] - count[:, lo]
    w = wins[:, hi] - wins[:, lo]
    return int(w[0]), int(n[0]), int(w[1]), int(n[1])


def cell_ll(w0, n0, w1, n1) -> float:
    return bin_ll(w0, n0) + bin_ll(w1, n1)


def pooled_ll(w0, n0, w1, n1) -> float:
    return bin_ll(w0 + w1, n0 + n1)


def rates_of(w0, n0, w1, n1):
    p0 = w0 / n0 if n0 else float("nan")
    p1 = w1 / n1 if n1 else float("nan")
    return p0, p1


def best_pair_in_range(prefixes, lo: int, hi: int, min_cell: int = 8):
    best = None
    for (a, b), (count, wins, _cells) in prefixes.items():
        w0, n0, w1, n1 = range_stats(count, wins, lo, hi)
        if n0 < min_cell or n1 < min_cell:
            continue
        ll = cell_ll(w0, n0, w1, n1)
        base = pooled_ll(w0, n0, w1, n1)
        gain = ll - base
        p0, p1 = rates_of(w0, n0, w1, n1)
        # direction: live cell is the higher rate. 1 = XNOR/agree is live, 0 = XOR/differ is live.
        if p1 >= p0:
            direction = "xnor"
            live, dead = p1, p0
            n_live, n_dead = n1, n0
        else:
            direction = "xor"
            live, dead = p0, p1
            n_live, n_dead = n0, n1
        rec = {
            "pair": (a, b),
            "gain": gain,
            "ll": ll,
            "base": base,
            "p_differ": p0,
            "p_agree": p1,
            "n_differ": n0,
            "n_agree": n1,
            "direction": direction,
            "p_live": live,
            "p_dead": dead,
            "n_live": n_live,
            "n_dead": n_dead,
            "sep": abs(p1 - p0),
        }
        if best is None or gain > best["gain"] + 1e-12 or (
            abs(gain - best["gain"]) <= 1e-12 and (a, b) < best["pair"]
        ):
            best = rec
    return best


def ranked_pairs(prefixes, lo: int, hi: int, min_cell: int = 8):
    rows = []
    for (a, b), (count, wins, _cells) in prefixes.items():
        w0, n0, w1, n1 = range_stats(count, wins, lo, hi)
        if n0 < min_cell or n1 < min_cell:
            continue
        ll = cell_ll(w0, n0, w1, n1)
        base = pooled_ll(w0, n0, w1, n1)
        p0, p1 = rates_of(w0, n0, w1, n1)
        rows.append(((a, b), ll - base, abs(p1 - p0), p0, p1, n0, n1))
    rows.sort(key=lambda r: (-r[1], r[0]))
    return rows


def four_cell_table(bits: np.ndarray, a: int, b: int, lo: int, hi: int):
    """Cells keyed by (bit_a, bit_b). Returns wins, n."""
    wins = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    ns = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    n = len(bits)
    hi = min(hi, n)
    for t in range(max(lo, b), hi):
        if bits[t] < 0 or bits[t - a] < 0 or bits[t - b] < 0:
            continue
        key = (int(bits[t - a]), int(bits[t - b]))
        ns[key] += 1
        wins[key] += int(bits[t])
    return wins, ns


def triple_prefix(bits: np.ndarray, lags):
    n = len(bits)
    need = max(lags)
    count = np.zeros((2, n + 1), dtype=np.int32)
    wins = np.zeros((2, n + 1), dtype=np.int32)
    for t in range(n):
        count[:, t + 1] = count[:, t]
        wins[:, t + 1] = wins[:, t]
        if t < need or bits[t] < 0 or any(bits[t - lag] < 0 for lag in lags):
            continue
        parity = 0
        for lag in lags:
            parity ^= int(bits[t - lag])
        # cell 1 = even parity (XNOR-like, even number of wins), cell 0 = odd
        cell = 1 if parity == 0 else 0
        count[cell, t + 1] += 1
        wins[cell, t + 1] += int(bits[t])
    return count, wins


def best_triple(bits, lo, hi, min_cell=8):
    best = None
    for lags in TRIPLES:
        count, wins = triple_prefix(bits, lags)
        w0, n0, w1, n1 = range_stats(count, wins, lo, hi)
        if n0 < min_cell or n1 < min_cell:
            continue
        gain = cell_ll(w0, n0, w1, n1) - pooled_ll(w0, n0, w1, n1)
        p0, p1 = rates_of(w0, n0, w1, n1)
        rec = {"lags": lags, "gain": gain, "sep": abs(p1 - p0), "p0": p0, "p1": p1, "n0": n0, "n1": n1}
        if best is None or gain > best["gain"]:
            best = rec
    return best


def third_lag_gain(bits, a, b, lo, hi, min_cell=5):
    """Does a third lag split the parity cells? Return best extra LL and the lag."""
    best = None
    n = len(bits)
    hi = min(hi, n)
    for c in LAGS:
        if c in (a, b):
            continue
        # 4 cells: parity x third bit. Compare to parity-only.
        wins = np.zeros(4)
        ns = np.zeros(4)
        for t in range(max(lo, max(b, c)), hi):
            if bits[t] < 0 or bits[t - a] < 0 or bits[t - b] < 0 or bits[t - c] < 0:
                continue
            parity = 1 if bits[t - a] == bits[t - b] else 0
            idx = parity * 2 + int(bits[t - c])
            ns[idx] += 1
            wins[idx] += bits[t]
        if np.any(ns < min_cell):
            continue
        ll4 = sum(bin_ll(wins[i], ns[i]) for i in range(4))
        # collapse third bit
        ll2 = bin_ll(wins[0] + wins[1], ns[0] + ns[1]) + bin_ll(wins[2] + wins[3], ns[2] + ns[3])
        gain = ll4 - ll2
        # rate lift: max abs difference inside a parity cell
        lifts = []
        for parity in (0, 1):
            i0, i1 = parity * 2, parity * 2 + 1
            if ns[i0] and ns[i1]:
                lifts.append(abs(wins[i0] / ns[i0] - wins[i1] / ns[i1]))
        rec = {"lag": c, "gain": float(gain), "lift": max(lifts) if lifts else 0.0}
        if best is None or gain > best["gain"]:
            best = rec
    return best


def sliding_cell_rate(count, wins, width: int, step: int, lo: int, hi: int):
    rows = []
    t = lo + width
    while t <= hi:
        w0, n0, w1, n1 = range_stats(count, wins, t - width, t)
        if n0 >= 8 and n1 >= 8:
            rows.append(
                {
                    "end": t,
                    "mid": t - width / 2.0,
                    "p_differ": w0 / n0,
                    "p_agree": w1 / n1,
                    "n_differ": n0,
                    "n_agree": n1,
                }
            )
        t += step
    return rows


def changepoint_scan(prefixes, lo: int, hi: int, min_cell: int = 8, step: int = 1):
    """Max LL of two independent pair-rules split at c, versus one rule on [lo, hi)."""
    full = best_pair_in_range(prefixes, lo, hi, min_cell=min_cell)
    if full is None:
        return None
    best = None
    # Precompute per-pair range stats via prefixes; scan c.
    c = lo + 30
    while c <= hi - 30:
        left = best_pair_in_range(prefixes, lo, c, min_cell=min_cell)
        right = best_pair_in_range(prefixes, c, hi, min_cell=min_cell)
        if left is not None and right is not None:
            gain = (left["ll"] + right["ll"]) - full["ll"]
            rec = {
                "c": c,
                "gain": gain,
                "left": {k: left[k] for k in ("pair", "direction", "p_live", "p_dead", "sep", "gain")},
                "right": {k: right[k] for k in ("pair", "direction", "p_live", "p_dead", "sep", "gain")},
            }
            if best is None or gain > best["gain"]:
                best = rec
        c += step
    if best is None:
        return {"full_gain": full["gain"], "full_pair": full["pair"], "split": None}
    best["full_gain"] = full["gain"]
    best["full_pair"] = full["pair"]
    best["full_direction"] = full["direction"]
    best["full_sep"] = full["sep"]
    return best


def fixed_pair_flip_scan(count, wins, lo, hi, min_cell=8):
    """For one pair, find split c maximizing LL of two separate rate-pairs. Sharpness of a flip/level change."""
    w0, n0, w1, n1 = range_stats(count, wins, lo, hi)
    if n0 < min_cell or n1 < min_cell:
        return None
    full = cell_ll(w0, n0, w1, n1)
    best_c, best_gain = None, -1e99
    profile = []
    for c in range(lo + 25, hi - 25):
        a = range_stats(count, wins, lo, c)
        b = range_stats(count, wins, c, hi)
        if min(a[1], a[3], b[1], b[3]) < min_cell:
            continue
        gain = cell_ll(*a) + cell_ll(*b) - full
        profile.append((c, gain))
        if gain > best_gain:
            best_gain, best_c = gain, c
    if best_c is None:
        return None
    gains = [g for _, g in profile]
    gains.sort()
    return {
        "c": best_c,
        "gain": best_gain,
        "median_gain": gains[len(gains) // 2] if gains else None,
        "p90_gain": gains[int(0.9 * (len(gains) - 1))] if gains else None,
        "width_near_max": sum(1 for g in gains if g > best_gain - 1.0),
    }


def residual_autocorr(bits, cells, p_agree, p_differ, max_lag=8):
    idx = [t for t in range(len(bits)) if cells[t] >= 0 and bits[t] >= 0]
    if len(idx) < 30:
        return None
    resid = []
    for t in idx:
        p = p_agree if cells[t] == 1 else p_differ
        resid.append((t, bits[t] - p))
    # map tick -> residual
    by_t = {t: r for t, r in resid}
    out = {}
    for lag in range(1, max_lag + 1):
        xs, ys = [], []
        for t, r in resid:
            if t - lag in by_t:
                xs.append(r)
                ys.append(by_t[t - lag])
        if len(xs) < 20:
            out[lag] = None
            continue
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        x = x - x.mean()
        y = y - y.mean()
        denom = math.sqrt(float((x * x).sum() * (y * y).sum()))
        out[lag] = float((x * y).sum() / denom) if denom > 0 else 0.0
    # extra-binomial within cell
    extra = {}
    for cell, p in ((0, p_differ), (1, p_agree)):
        obs = [int(bits[t]) for t in idx if cells[t] == cell]
        if len(obs) < 10 or not (0 < p < 1):
            continue
        m = sum(obs) / len(obs)
        var = sum((o - m) ** 2 for o in obs) / len(obs)
        extra[str(cell)] = {"n": len(obs), "var": var, "binom": m * (1 - m), "ratio": var / (m * (1 - m)) if m * (1 - m) else None}
    return {"acf": {str(k): v for k, v in out.items()}, "extra_binomial": extra}


def neighbour_scan(bits_by_machine, own_cells, lo, hi, min_cell=8):
    """Best other-machine lag, measured as extra LL inside the owner's parity cells."""
    machines = list(bits_by_machine)
    best = None
    # also aggregate lift by offset
    by_offset = []
    own_id = None
    return best


def neighbour_ll(own_bits, own_cells, other_bits, lag, lo, hi, min_cell=6):
    """Split each own parity cell by other machine's win at lag. Extra LL vs parity alone."""
    wins = np.zeros(4)
    ns = np.zeros(4)
    n = len(own_bits)
    hi = min(hi, n, len(other_bits))
    for t in range(max(lo, lag), hi):
        if own_cells[t] < 0 or own_bits[t] < 0 or other_bits[t - lag] < 0:
            continue
        idx = int(own_cells[t]) * 2 + int(other_bits[t - lag])
        ns[idx] += 1
        wins[idx] += own_bits[t]
    if np.any(ns < min_cell):
        # still compute if both parity margins exist after pooling? require all 4
        return None
    ll4 = sum(bin_ll(float(wins[i]), float(ns[i])) for i in range(4))
    ll2 = bin_ll(float(wins[0] + wins[1]), float(ns[0] + ns[1])) + bin_ll(
        float(wins[2] + wins[3]), float(ns[2] + ns[3])
    )
    lifts = []
    for parity in (0, 1):
        i0, i1 = parity * 2, parity * 2 + 1
        lifts.append(abs(wins[i0] / ns[i0] - wins[i1] / ns[i1]))
    return {"gain": float(ll4 - ll2), "lift": float(max(lifts)), "mean_lift": float(sum(lifts) / 2), "n": int(ns.sum())}


def size_counts(series, lo, hi, condition_fn):
    """Count payouts in SIZE_PAYOUTS among winning spins whose condition_fn(t) is True/False."""
    yes = Counter()
    no = Counter()
    for t in range(lo, min(hi, len(series))):
        payout = series[t]
        if payout is None or payout == 0:
            continue
        flag = condition_fn(t)
        if flag is None:
            continue
        (yes if flag else no)[payout] += 1
    return yes, no


def multinomial_ll(counter: Counter) -> float:
    n = sum(counter.values())
    if n <= 0:
        return 0.0
    ll = 0.0
    for payout in SIZE_PAYOUTS:
        c = counter[payout]
        if c:
            ll += c * math.log(c / n)
    return ll


def size_gain(yes: Counter, no: Counter) -> float:
    pooled = yes + no
    return multinomial_ll(yes) + multinomial_ll(no) - multinomial_ll(pooled)


def mix_of(counter: Counter):
    n = sum(counter.values())
    if n == 0:
        return None
    return {str(p): counter[p] / n for p in SIZE_PAYOUTS} | {"n": n}


def tv_distance(a: Counter, b: Counter) -> float:
    na, nb = sum(a.values()), sum(b.values())
    if na == 0 or nb == 0:
        return float("nan")
    return 0.5 * sum(abs(a[p] / na - b[p] / nb) for p in SIZE_PAYOUTS)


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def main():
    print("loading", flush=True)
    train = load_split("train")
    valid = load_split("validation")
    results = {}

    # ---------- inventory ----------
    cond_counts = Counter(ep["condition"] for ep in train["episodes"].values())
    val_cond = Counter(ep["condition"] for ep in valid["episodes"].values())
    cutoffs = [ep["cutoff"] for ep in train["episodes"].values()]
    gaps = []
    for ep in train["episodes"].values():
        if ep["condition"] == "short_history":
            continue
        for machine in ep["machines"].values():
            ticks = sorted(machine["payouts"])
            if ticks != list(range(ep["cutoff"])):
                gaps.append((ep["episode_id"], machine["machine_id"], len(ticks), ep["cutoff"]))
    short_lengths = []
    for ep in train["episodes"].values():
        if ep["condition"] != "short_history":
            continue
        for machine in ep["machines"].values():
            short_lengths.append(len(machine["payouts"]))
    payout_counts = Counter()
    for ep in train["episodes"].values():
        if ep["condition"] == "short_history":
            continue
        for machine in ep["machines"].values():
            payout_counts.update(machine["payouts"].values())
    results["inventory"] = {
        "train_episodes": len(train["episodes"]),
        "train_conditions": dict(cond_counts),
        "val_episodes": len(valid["episodes"]),
        "val_conditions": dict(val_cond),
        "cutoff_min": min(cutoffs),
        "cutoff_max": max(cutoffs),
        "cutoff_mean": mean(cutoffs),
        "long_history_gaps": len(gaps),
        "short_len_min": min(short_lengths) if short_lengths else None,
        "short_len_max": max(short_lengths) if short_lengths else None,
        "short_len_mean": mean(short_lengths) if short_lengths else None,
        "payout_counts_long": {str(k): payout_counts[k] for k in PAYOUTS},
        "payout_frac_long": {str(k): payout_counts[k] / sum(payout_counts.values()) for k in PAYOUTS},
    }
    print("inventory", results["inventory"], flush=True)

    # ---------- per stable machine, both feature defs ----------
    def analyze_machines(split, condition, feat_name, lo_pad=8):
        feat = FEATS[feat_name]
        rows = []
        for ep in split["episodes"].values():
            if ep["condition"] != condition:
                continue
            bits_map = {}
            prefix_map = {}
            for machine in ep["machines"].values():
                series = series_of(machine, ep["cutoff"])
                bits = build_bits(series, feat)
                prefixes = {pair: pair_prefix(bits, *pair) for pair in PAIRS}
                bits_map[machine["machine_id"]] = bits
                prefix_map[machine["machine_id"]] = prefixes
                best = best_pair_in_range(prefixes, lo_pad, ep["cutoff"], min_cell=10)
                ranked = ranked_pairs(prefixes, lo_pad, ep["cutoff"], min_cell=10)
                if best is None:
                    rows.append({"episode_id": ep["episode_id"], "machine_id": machine["machine_id"], "ok": False})
                    continue
                second = ranked[1][1] if len(ranked) > 1 else None
                wins4, ns4 = four_cell_table(bits, *best["pair"], lo_pad, ep["cutoff"])
                # parity collapse gaps
                def rate(key):
                    return wins4[key] / ns4[key] if ns4[key] else float("nan")

                same_gap = abs(rate((0, 0)) - rate((1, 1)))
                diff_gap = abs(rate((0, 1)) - rate((1, 0)))
                ll4 = sum(bin_ll(wins4[k], ns4[k]) for k in wins4 if ns4[k] > 0)
                ll2 = cell_ll(
                    wins4[(0, 1)] + wins4[(1, 0)],
                    ns4[(0, 1)] + ns4[(1, 0)],
                    wins4[(0, 0)] + wins4[(1, 1)],
                    ns4[(0, 0)] + ns4[(1, 1)],
                )
                triple = best_triple(bits, lo_pad, ep["cutoff"], min_cell=10)
                third = third_lag_gain(bits, *best["pair"], lo_pad, ep["cutoff"])
                rows.append(
                    {
                        "ok": True,
                        "episode_id": ep["episode_id"],
                        "machine_id": machine["machine_id"],
                        "machine_type": machine["machine_type"],
                        "signal": machine["signal"],
                        "cutoff": ep["cutoff"],
                        "pair": best["pair"],
                        "direction": best["direction"],
                        "sep": best["sep"],
                        "gain": best["gain"],
                        "second_gain": second,
                        "gap": None if second is None else best["gain"] - second,
                        "p_live": best["p_live"],
                        "p_dead": best["p_dead"],
                        "p_agree": best["p_agree"],
                        "p_differ": best["p_differ"],
                        "n_live": best["n_live"],
                        "n_dead": best["n_dead"],
                        "n_agree": best["n_agree"],
                        "n_differ": best["n_differ"],
                        "same_gap": same_gap,
                        "diff_gap": diff_gap,
                        "ll4_minus_ll2": ll4 - ll2,
                        "four_rates": {str(k): rate(k) for k in wins4},
                        "four_n": {str(k): ns4[k] for k in ns4},
                        "triple_gain": None if triple is None else triple["gain"],
                        "triple_lags": None if triple is None else triple["lags"],
                        "triple_sep": None if triple is None else triple["sep"],
                        "third": third,
                    }
                )
        return rows

    print("stable pair scan", flush=True)
    stable_by_feat = {}
    for feat_name in ("gt0", "ge2", "ge5"):
        rows = analyze_machines(train, "stable", feat_name)
        ok = [r for r in rows if r.get("ok")]
        pair_counts = Counter(tuple(r["pair"]) for r in ok)
        dir_counts = Counter(r["direction"] for r in ok)
        # per type level spread
        by_type = {}
        for mtype in ("steady", "balanced", "jackpot"):
            sub = [r for r in ok if r["machine_type"] == mtype]
            by_sig = {}
            for sig in (-1, 0, 1):
                ss = [r for r in sub if r["signal"] == sig]
                by_sig[str(sig)] = {
                    "n": len(ss),
                    "p_live_mean": mean(r["p_live"] for r in ss),
                    "p_live_std": std(r["p_live"] for r in ss),
                    "p_dead_mean": mean(r["p_dead"] for r in ss),
                    "p_dead_std": std(r["p_dead"] for r in ss),
                }
            by_type[mtype] = {
                "n": len(sub),
                "p_live_mean": mean(r["p_live"] for r in sub),
                "p_live_std": std(r["p_live"] for r in sub),
                "p_live_min": min((r["p_live"] for r in sub), default=None),
                "p_live_max": max((r["p_live"] for r in sub), default=None),
                "p_dead_mean": mean(r["p_dead"] for r in sub),
                "p_dead_std": std(r["p_dead"] for r in sub),
                "p_dead_min": min((r["p_dead"] for r in sub), default=None),
                "p_dead_max": max((r["p_dead"] for r in sub), default=None),
                "sep_mean": mean(r["sep"] for r in sub),
                "sep_min": min((r["sep"] for r in sub), default=None),
                "by_signal": by_sig,
            }
        weak = [r for r in ok if r["sep"] < 0.4]
        triple_better = [
            r for r in ok if r["triple_gain"] is not None and r["triple_gain"] > r["gain"] + 2.0
        ]
        third_gains = [r["third"]["gain"] for r in ok if r["third"]]
        third_lifts = [r["third"]["lift"] for r in ok if r["third"]]
        stable_by_feat[feat_name] = {
            "n": len(ok),
            "pair_counts": {str(k): v for k, v in pair_counts.most_common()},
            "direction_counts": dict(dir_counts),
            "sep_mean": mean(r["sep"] for r in ok),
            "sep_min": min((r["sep"] for r in ok), default=None),
            "sep_p10": quantiles([r["sep"] for r in ok]).get("0.1"),
            "gain_mean": mean(r["gain"] for r in ok),
            "gap_mean": mean(r["gap"] for r in ok if r["gap"] is not None),
            "gap_median": median([r["gap"] for r in ok if r["gap"] is not None]),
            "frac_gap_lt_2": mean(1.0 if r["gap"] is not None and r["gap"] < 2 else 0.0 for r in ok),
            "same_gap_mean": mean(r["same_gap"] for r in ok),
            "diff_gap_mean": mean(r["diff_gap"] for r in ok),
            "same_gap_p90": quantiles([r["same_gap"] for r in ok]).get("0.9"),
            "diff_gap_p90": quantiles([r["diff_gap"] for r in ok]).get("0.9"),
            "ll4_minus_ll2_mean": mean(r["ll4_minus_ll2"] for r in ok),
            "ll4_minus_ll2_p90": quantiles([r["ll4_minus_ll2"] for r in ok]).get("0.9"),
            "n_weak_sep_lt_0.4": len(weak),
            "n_triple_beats_pair_by_2": len(triple_better),
            "triple_minus_pair_mean": mean(r["triple_gain"] - r["gain"] for r in ok if r["triple_gain"] is not None),
            "third_gain_mean": mean(third_gains),
            "third_gain_p90": quantiles(third_gains).get("0.9"),
            "third_lift_mean": mean(third_lifts),
            "third_lift_p90": quantiles(third_lifts).get("0.9"),
            "by_type": by_type,
            "machines": ok,
        }
        print(
            feat_name,
            "sep",
            round(stable_by_feat[feat_name]["sep_mean"], 3),
            "pairs",
            pair_counts.most_common(6),
            "dirs",
            dict(dir_counts),
            "4cell extra",
            round(stable_by_feat[feat_name]["ll4_minus_ll2_mean"], 3),
            flush=True,
        )

    # Keep machine rows in JSON but they are large; store them.
    results["stable_features"] = {
        name: {k: v for k, v in block.items() if k != "machines"} | {
            "pair_top": block["pair_counts"],
        }
        for name, block in stable_by_feat.items()
    }
    # store compact machine rows for gt0 only
    results["stable_machines_gt0"] = [
        {
            "episode_id": r["episode_id"],
            "machine_id": r["machine_id"],
            "machine_type": r["machine_type"],
            "signal": r["signal"],
            "pair": r["pair"],
            "direction": r["direction"],
            "sep": r["sep"],
            "gain": r["gain"],
            "gap": r["gap"],
            "p_live": r["p_live"],
            "p_dead": r["p_dead"],
            "same_gap": r["same_gap"],
            "diff_gap": r["diff_gap"],
            "ll4_minus_ll2": r["ll4_minus_ll2"],
            "triple_gain": r["triple_gain"],
            "triple_lags": r["triple_lags"],
            "third_gain": None if not r["third"] else r["third"]["gain"],
            "third_lift": None if not r["third"] else r["third"]["lift"],
            "third_lag": None if not r["third"] else r["third"]["lag"],
        }
        for r in stable_by_feat["gt0"]["machines"]
    ]

    # ---------- validation targets vs cells ----------
    print("validation target alignment", flush=True)

    def attach_targets(split, feat_name, conditions):
        feat = FEATS[feat_name]
        rows = []
        for ep in split["episodes"].values():
            if ep["condition"] not in conditions:
                continue
            for machine in ep["machines"].values():
                probs = split["targets"].get(machine["query_id"])
                if probs is None:
                    continue
                series = series_of(machine, ep["cutoff"])
                bits = build_bits(series, feat)
                prefixes = {pair: pair_prefix(bits, *pair) for pair in PAIRS}
                # For drift, discover the pair on the early half so the blend does not wash it out.
                if ep["condition"] == "drift":
                    hi_fit = max(40, ep["cutoff"] // 2)
                elif ep["condition"] == "shared_change":
                    hi_fit = ep["cutoff"]  # refined later
                else:
                    hi_fit = ep["cutoff"]
                best = best_pair_in_range(prefixes, 8, hi_fit, min_cell=8)
                if best is None:
                    continue
                cells = prefixes[best["pair"]][2]
                # cell at cutoff: need bits at cutoff-a, cutoff-b. Those ticks are < cutoff so in series.
                cutoff = ep["cutoff"]
                a, b = best["pair"]
                if cutoff - 1 < b or bits[cutoff - a] < 0 or bits[cutoff - b] < 0:
                    # predicting the spin AT cutoff, features are lags before cutoff, i.e. bits[cutoff-a]
                    cell = -1
                else:
                    cell = 1 if bits[cutoff - a] == bits[cutoff - b] else 0
                # Wait: series is indexed by tick 0..cutoff-1. The outcome being predicted is tick=cutoff,
                # which is NOT in the history. Features are payouts at cutoff-a and cutoff-b, which ARE in history.
                # bits has length cutoff, indices 0..cutoff-1. bits[cutoff - a] is correct.
                # I mistakenly used bits built on series of length cutoff, so index cutoff-a is valid if a>=1.
                if cell < 0:
                    # recompute explicitly
                    pass
                pa = series[cutoff - a] if cutoff - a >= 0 else None
                pb = series[cutoff - b] if cutoff - b >= 0 else None
                if pa is None or pb is None:
                    continue
                agree = int(feat(pa) == feat(pb))
                live = int((best["direction"] == "xnor" and agree == 1) or (best["direction"] == "xor" and agree == 0))
                p_win = 1.0 - probs[0]
                rows.append(
                    {
                        "episode_id": ep["episode_id"],
                        "machine_id": machine["machine_id"],
                        "machine_type": machine["machine_type"],
                        "signal": machine["signal"],
                        "condition": ep["condition"],
                        "cutoff": cutoff,
                        "pair": best["pair"],
                        "direction": best["direction"],
                        "agree": agree,
                        "live": live,
                        "p_win": p_win,
                        "probs": probs,
                        "p_live_hat": best["p_live"],
                        "p_dead_hat": best["p_dead"],
                        "sep": best["sep"],
                    }
                )
        return rows

    target_rows = {}
    for feat_name in ("gt0", "ge2"):
        target_rows[feat_name] = attach_targets(
            valid, feat_name, {"stable", "drift", "shared_change", "short_history"}
        )

    def summarize_target_rows(rows, feat_name):
        out = {"by_condition": {}}
        for cond in ("stable", "drift", "shared_change", "short_history"):
            sub = [r for r in rows if r["condition"] == cond]
            by_type = {}
            for mtype in ("steady", "balanced", "jackpot"):
                ss = [r for r in sub if r["machine_type"] == mtype]
                cell_stats = {}
                for live in (0, 1):
                    cell = [r for r in ss if r["live"] == live]
                    vals = [r["p_win"] for r in cell]
                    groups = cluster_1d(vals, 1e-4) if vals else []
                    # tighter and looser
                    groups_tight = cluster_1d(vals, 1e-8) if vals else []
                    groups_loose = cluster_1d(vals, 5e-3) if vals else []
                    by_sig = {}
                    for sig in (-1, 0, 1):
                        sv = [r["p_win"] for r in cell if r["signal"] == sig]
                        by_sig[str(sig)] = {
                            "n": len(sv),
                            "mean": mean(sv) if sv else None,
                            "std": std(sv) if sv else None,
                            "min": min(sv) if sv else None,
                            "max": max(sv) if sv else None,
                            "n_unique_1e-6": len(cluster_1d(sv, 1e-6)) if sv else 0,
                        }
                    cell_stats[str(live)] = {
                        "n": len(vals),
                        "mean": mean(vals) if vals else None,
                        "std": std(vals) if vals else None,
                        "min": min(vals) if vals else None,
                        "max": max(vals) if vals else None,
                        "n_clusters_1e-8": len(groups_tight),
                        "n_clusters_1e-4": len(groups),
                        "n_clusters_5e-3": len(groups_loose),
                        "cluster_sizes_1e-4": [len(g) for g in groups],
                        "cluster_means_1e-4": [mean(g) for g in groups],
                        "by_signal": by_sig,
                    }
                by_type[mtype] = cell_stats
            # global uniqueness of full prob vector
            rounded = [tuple(round(x, 8) for x in r["probs"]) for r in sub]
            out["by_condition"][cond] = {
                "n": len(sub),
                "by_type_live": by_type,
                "n_unique_prob_vectors_1e-8": len(set(rounded)),
            }
        return out

    results["targets"] = {name: summarize_target_rows(rows, name) for name, rows in target_rows.items()}

    # Exact group-variance test: do NOT use discovered direction. Group p_win by (type, agree bit)
    # for every pair, on stable validation, using gt0 and ge2. The true pair collapses variance.
    print("pair collapse test", flush=True)

    def collapse_test(feat_name, condition="stable"):
        feat = FEATS[feat_name]
        # collect per machine the agree bit at cutoff for every pair, plus p_win, type, signal
        records = []
        for ep in valid["episodes"].values():
            if ep["condition"] != condition:
                continue
            for machine in ep["machines"].values():
                probs = valid["targets"][machine["query_id"]]
                series = series_of(machine, ep["cutoff"])
                cutoff = ep["cutoff"]
                agrees = {}
                four = {}
                for a, b in PAIRS:
                    pa, pb = series[cutoff - a], series[cutoff - b]
                    if pa is None or pb is None:
                        agrees[(a, b)] = None
                        continue
                    fa, fb = feat(pa), feat(pb)
                    agrees[(a, b)] = int(fa == fb)
                    four[(a, b)] = (fa, fb)
                records.append(
                    {
                        "type": machine["machine_type"],
                        "signal": machine["signal"],
                        "p_win": 1.0 - probs[0],
                        "probs": probs,
                        "agrees": agrees,
                        "four": four,
                        "series": series,
                        "cutoff": cutoff,
                        "index": machine["index"],
                    }
                )
        # For each pair, within (type, agree) variance of p_win, and within (type, signal, agree)
        pair_scores = []
        for pair in PAIRS:
            groups = defaultdict(list)
            groups_sig = defaultdict(list)
            groups4 = defaultdict(list)
            for rec in records:
                bit = rec["agrees"][pair]
                if bit is None:
                    continue
                groups[(rec["type"], bit)].append(rec["p_win"])
                groups_sig[(rec["type"], rec["signal"], bit)].append(rec["p_win"])
                groups4[(rec["type"], rec["four"][pair])].append(rec["p_win"])
            def mean_var(groups_):
                num = den = 0.0
                for vals in groups_.values():
                    if len(vals) < 2:
                        continue
                    num += (len(vals) - 1) * (std(vals) ** 2)
                    den += len(vals) - 1
                return math.sqrt(num / den) if den else float("nan"), den

            rmse, n = mean_var(groups)
            rmse_s, n_s = mean_var(groups_sig)
            rmse4, n4 = mean_var(groups4)
            # separation of group means
            seps = []
            for mtype in ("steady", "balanced", "jackpot"):
                a = groups.get((mtype, 1), [])
                b = groups.get((mtype, 0), [])
                if a and b:
                    seps.append(abs(mean(a) - mean(b)))
            pair_scores.append(
                {
                    "pair": pair,
                    "rmse": rmse,
                    "rmse_with_signal": rmse_s,
                    "rmse_4cell": rmse4,
                    "mean_sep": mean(seps) if seps else None,
                    "n": n,
                }
            )
        pair_scores.sort(key=lambda r: (r["rmse"], -r["mean_sep"]))
        # single lags
        single = []
        for lag in LAGS:
            groups = defaultdict(list)
            for rec in records:
                payout = rec["series"][rec["cutoff"] - lag]
                if payout is None:
                    continue
                groups[(rec["type"], feat(payout))].append(rec["p_win"])
            num = den = 0.0
            for vals in groups.values():
                if len(vals) < 2:
                    continue
                num += (len(vals) - 1) * std(vals) ** 2
                den += len(vals) - 1
            single.append({"lag": lag, "rmse": math.sqrt(num / den) if den else None})
        single.sort(key=lambda r: r["rmse"] if r["rmse"] is not None else 99)
        # triples
        triple_scores = []
        for lags in TRIPLES:
            groups = defaultdict(list)
            for rec in records:
                bits = []
                ok = True
                for lag in lags:
                    payout = rec["series"][rec["cutoff"] - lag]
                    if payout is None:
                        ok = False
                        break
                    bits.append(feat(payout))
                if not ok:
                    continue
                parity = bits[0] ^ bits[1] ^ bits[2]
                groups[(rec["type"], parity)].append(rec["p_win"])
            num = den = 0
            for vals in groups.values():
                if len(vals) < 2:
                    continue
                num += (len(vals) - 1) * std(vals) ** 2
                den += len(vals) - 1
            triple_scores.append({"lags": lags, "rmse": math.sqrt(num / den) if den else None})
        triple_scores.sort(key=lambda r: r["rmse"] if r["rmse"] is not None else 99)
        return {
            "best_pairs": pair_scores[:8],
            "worst_pair_rmse": pair_scores[-1]["rmse"] if pair_scores else None,
            "best_singles": single[:4],
            "best_triples": triple_scores[:5],
            "n_machines": len(records),
        }

    results["collapse"] = {feat: collapse_test(feat) for feat in ("gt0", "ge2", "ge5")}
    for feat, block in results["collapse"].items():
        print(feat, "best collapse", block["best_pairs"][:3], "triple", block["best_triples"][:2], flush=True)

    # ---------- levels: exact values on stable, gt0, discovered per machine ----------
    # Use train-discovered pairs? Validation history can discover the pair; targets give exact p.
    # Additionally cluster ALL stable p_win regardless of cell.
    print("level clusters", flush=True)
    level_info = {}
    for cond in ("stable", "drift", "shared_change", "short_history"):
        vals_by_type = defaultdict(list)
        vectors = []
        for ep in valid["episodes"].values():
            if ep["condition"] != cond:
                continue
            for machine in ep["machines"].values():
                probs = valid["targets"][machine["query_id"]]
                vals_by_type[machine["machine_type"]].append(
                    {
                        "p_win": 1.0 - probs[0],
                        "signal": machine["signal"],
                        "probs": probs,
                    }
                )
                vectors.append(probs)
        level_info[cond] = {}
        for mtype, rows in vals_by_type.items():
            vals = [r["p_win"] for r in rows]
            groups = cluster_1d(vals, 1e-3)
            groups_tight = cluster_1d(vals, 1e-6)
            level_info[cond][mtype] = {
                "n": len(vals),
                "std": std(vals),
                "n_clusters_1e-3": len(groups),
                "n_clusters_1e-6": len(groups_tight),
                "cluster_means_1e-3": [
                    {"mean": mean(g), "n": len(g), "std": std(g), "min": min(g), "max": max(g)}
                    for g in sorted(groups, key=mean)
                ],
            }
        level_info[cond]["n_unique_vectors_1e-8"] = len({tuple(round(x, 8) for x in v) for v in vectors})
        level_info[cond]["n_unique_vectors_1e-4"] = len({tuple(round(x, 4) for x in v) for v in vectors})
    results["raw_level_clusters"] = level_info

    # ---------- drift ----------
    print("drift", flush=True)
    drift_rows = []
    for ep in train["episodes"].values():
        if ep["condition"] != "drift":
            continue
        for machine in ep["machines"].values():
            series = series_of(machine, ep["cutoff"])
            bits = build_bits(series, feat_gt0)
            prefixes = {pair: pair_prefix(bits, *pair) for pair in PAIRS}
            early_hi = min(ep["cutoff"], 80)
            early = best_pair_in_range(prefixes, 8, early_hi, min_cell=8)
            late_lo = max(8, ep["cutoff"] - 70)
            late = best_pair_in_range(prefixes, late_lo, ep["cutoff"], min_cell=8)
            if early is None:
                continue
            count, wins, cells = prefixes[early["pair"]]
            windows = sliding_cell_rate(count, wins, width=36, step=8, lo=8, hi=ep["cutoff"])
            # orient so early agree-rate is the "positive" cell tracked as p_pos = p_agree
            # record both
            drift_rows.append(
                {
                    "episode_id": ep["episode_id"],
                    "machine_id": machine["machine_id"],
                    "machine_type": machine["machine_type"],
                    "signal": machine["signal"],
                    "cutoff": ep["cutoff"],
                    "early_pair": early["pair"],
                    "early_direction": early["direction"],
                    "early_sep": early["sep"],
                    "early_p_agree": early["p_agree"],
                    "early_p_differ": early["p_differ"],
                    "late_pair": None if late is None else late["pair"],
                    "late_direction": None if late is None else late["direction"],
                    "late_sep": None if late is None else late["sep"],
                    "pair_same": late is not None and late["pair"] == early["pair"],
                    "direction_same": late is not None and late["direction"] == early["direction"],
                    "windows": windows,
                }
            )

    # Fit a shared linear blend on the early-live cell.
    # p_live_cell(t) = (1-u)*L + u*D, u = clip((t - t0)/T, 0, 1)
    # Use window midpoints. L and D estimated per type from stable gt0 means, also free.
    stable_levels = {}
    for mtype, block in stable_by_feat["gt0"]["by_type"].items():
        stable_levels[mtype] = (block["p_dead_mean"], block["p_live_mean"])

    def fit_drift(rows, t0_grid, T_grid, level_mode="per_type"):
        # Precompute oriented series: y = rate of the cell that was live early, x = mid tick
        series = []
        for row in rows:
            if row["early_direction"] == "xnor":
                # agree was live early; track p_agree. As it flips, p_agree falls toward dead.
                key = "p_agree"
            else:
                key = "p_differ"
            L = stable_levels[row["machine_type"]][1]
            D = stable_levels[row["machine_type"]][0]
            pts = [(w["mid"], w[key], w["n_agree"] if key == "p_agree" else w["n_differ"], L, D, row["cutoff"]) for w in row["windows"]]
            series.append(pts)
        best = None
        for t0_rule in t0_grid:
            for T in T_grid:
                sse = 0.0
                n = 0
                # t0_rule: ("before_cutoff", k) means t0 = cutoff - k, or ("absolute", t0)
                for pts in series:
                    for mid, y, nn, L, D, cutoff in pts:
                        if t0_rule[0] == "before_cutoff":
                            t0 = cutoff - t0_rule[1]
                        else:
                            t0 = t0_rule[1]
                        u = min(1.0, max(0.0, (mid - t0) / T))
                        pred = (1.0 - u) * L + u * D
                        sse += nn * (y - pred) ** 2
                        n += nn
                rmse = math.sqrt(sse / n) if n else 99
                rec = {"t0_rule": t0_rule, "T": T, "rmse": rmse, "n": n}
                if best is None or rmse < best["rmse"]:
                    best = rec
        return best

    t0_grid = [("before_cutoff", k) for k in range(40, 161, 4)] + [("absolute", t) for t in range(80, 201, 10)]
    T_grid = list(range(16, 161, 4))
    # This grid is 30*40* ~200 machines * ~20 windows = 5e6, fine.
    drift_fit = fit_drift(drift_rows, t0_grid, T_grid)
    # Also a no-drift baseline: u=0 always, and a fully flipped baseline at the end only.
    def rmse_constant(which):
        sse = n = 0
        for row in drift_rows:
            key = "p_agree" if row["early_direction"] == "xnor" else "p_differ"
            L = stable_levels[row["machine_type"]][1]
            D = stable_levels[row["machine_type"]][0]
            for w in row["windows"]:
                y = w[key]
                nn = w["n_agree"] if key == "p_agree" else w["n_differ"]
                pred = L if which == "stay" else D
                sse += nn * (y - pred) ** 2
                n += nn
        return math.sqrt(sse / n) if n else None

    # Per-episode t0 with shared T from the global fit, and also per-episode free t0.
    T_hat = drift_fit["T"]
    per_ep = []
    episodes_drift = defaultdict(list)
    for row in drift_rows:
        episodes_drift[row["episode_id"]].append(row)
    for eid, rows in episodes_drift.items():
        cutoff = rows[0]["cutoff"]
        best_local = None
        for k in range(30, 180, 2):
            t0 = cutoff - k
            sse = n = 0
            for row in rows:
                key = "p_agree" if row["early_direction"] == "xnor" else "p_differ"
                L = stable_levels[row["machine_type"]][1]
                D = stable_levels[row["machine_type"]][0]
                for w in row["windows"]:
                    u = min(1.0, max(0.0, (w["mid"] - t0) / T_hat))
                    y = w[key]
                    nn = w["n_agree"] if key == "p_agree" else w["n_differ"]
                    sse += nn * (y - pred_clip(u, L, D)) ** 2
                    n += nn
            rmse = math.sqrt(sse / n) if n else 99
            if best_local is None or rmse < best_local[0]:
                best_local = (rmse, k, t0)
        per_ep.append({"episode_id": eid, "cutoff": cutoff, "k": best_local[1], "t0": best_local[2], "rmse": best_local[0]})

    # Logit-space blend vs probability blend, at the chosen (k, T)
    def blend_rmse(space, k, T):
        sse = n = 0
        for row in drift_rows:
            key = "p_agree" if row["early_direction"] == "xnor" else "p_differ"
            L = clamp(stable_levels[row["machine_type"]][1])
            D = clamp(stable_levels[row["machine_type"]][0])
            cutoff = row["cutoff"]
            t0 = cutoff - k
            for w in row["windows"]:
                u = min(1.0, max(0.0, (w["mid"] - t0) / T))
                if space == "prob":
                    pred = (1 - u) * L + u * D
                else:
                    pred = 1.0 / (1.0 + math.exp(-((1 - u) * logit(L) + u * logit(D))))
                y = w[key]
                nn = w["n_agree"] if key == "p_agree" else w["n_differ"]
                sse += nn * (y - pred) ** 2
                n += nn
        return math.sqrt(sse / n) if n else None

    # Refine k, T on a fine grid in both spaces using stable levels.
    fine = []
    for k in range(50, 150, 2):
        for T in range(20, 140, 2):
            fine.append((blend_rmse("prob", k, T), blend_rmse("logit", k, T), k, T))
    fine.sort()
    best_prob = min(fine, key=lambda r: r[0])
    best_logit = min(fine, key=lambda r: r[1])

    # Does the early pair remain the best late pair?
    pair_same = mean(1.0 if r["pair_same"] else 0.0 for r in drift_rows)
    dir_same = mean(1.0 if r["direction_same"] else 0.0 for r in drift_rows)
    # Among machines whose late pair matches, how often direction flipped.
    matched = [r for r in drift_rows if r["pair_same"]]
    dir_flip_given_same_pair = mean(0.0 if r["direction_same"] else 1.0 for r in matched) if matched else None

    # Window curve: average oriented rate in bins of (cutoff - mid)
    bins = defaultdict(list)
    for row in drift_rows:
        key = "p_agree" if row["early_direction"] == "xnor" else "p_differ"
        L = stable_levels[row["machine_type"]][1]
        D = stable_levels[row["machine_type"]][0]
        span = max(1e-6, L - D)
        for w in row["windows"]:
            # progress into the dead level: 0 = still live, 1 = fully dead
            y = w[key]
            norm = (L - y) / span
            lag_to_end = row["cutoff"] - w["mid"]
            bucket = int(round(lag_to_end / 10.0) * 10)
            bins[bucket].append(norm)
    curve = {str(k): {"mean_flip_frac": mean(v), "n": len(v)} for k, v in sorted(bins.items(), reverse=True)}

    # Validation drift targets: p_win vs oriented cell, and where it sits between levels.
    val_drift = [r for r in target_rows["gt0"] if r["condition"] == "drift"]
    drift_target_pos = []
    for r in val_drift:
        D, L = stable_levels[r["machine_type"]]
        # live flag from EARLY-half discovery (attach_targets uses first half for drift)
        p = r["p_win"]
        # If live==1, p should be near L if no drift completed, near D if fully flipped.
        # Map to flip fraction from the early-live assignment.
        if r["live"] == 1:
            # still labeled live by early direction; p near L => flip 0, p near D => flip 1
            frac = (L - p) / (L - D) if L != D else None
        else:
            frac = (p - D) / (L - D) if L != D else None
            # if early said this cutoff cell is dead, a full flip would make it live, frac~1 means p~L
        drift_target_pos.append({"frac": frac, "p_win": p, "type": r["machine_type"], "live": r["live"], "cutoff": r["cutoff"]})

    results["drift"] = {
        "n_machines": len(drift_rows),
        "frac_late_pair_equals_early": pair_same,
        "frac_late_direction_equals_early": dir_same,
        "frac_direction_flip_given_same_pair": dir_flip_given_same_pair,
        "early_pair_counts": dict(Counter(str(r["early_pair"]) for r in drift_rows)),
        "global_fit_prob_coarse": {"t0_rule": drift_fit["t0_rule"], "T": drift_fit["T"], "rmse": drift_fit["rmse"]},
        "rmse_stay_live": rmse_constant("stay"),
        "rmse_always_dead": rmse_constant("flip"),
        "best_prob_blend": {"rmse": best_prob[0], "k_before_cutoff": best_prob[2], "T": best_prob[3]},
        "best_logit_blend": {"rmse": best_logit[0], "k_before_cutoff": best_logit[2], "T": best_logit[3]},
        "per_episode_k_with_global_T": {
            "T": T_hat,
            "k_mean": mean(r["k"] for r in per_ep),
            "k_std": std(r["k"] for r in per_ep),
            "k_min": min(r["k"] for r in per_ep),
            "k_max": max(r["k"] for r in per_ep),
            "rmse_mean": mean(r["rmse"] for r in per_ep),
            "rows": per_ep,
        },
        "flip_curve_by_ticks_before_cutoff": curve,
        "val_target_flip_frac": {
            "mean": mean(r["frac"] for r in drift_target_pos if r["frac"] is not None),
            "std": std(r["frac"] for r in drift_target_pos if r["frac"] is not None),
            "quantiles": quantiles([r["frac"] for r in drift_target_pos if r["frac"] is not None]),
            "by_type": {
                mtype: {
                    "mean": mean(r["frac"] for r in drift_target_pos if r["type"] == mtype and r["frac"] is not None),
                    "std": std(r["frac"] for r in drift_target_pos if r["type"] == mtype and r["frac"] is not None),
                    "n": sum(1 for r in drift_target_pos if r["type"] == mtype),
                }
                for mtype in ("steady", "balanced", "jackpot")
            },
        },
        "stable_levels_used": {k: {"dead": v[0], "live": v[1]} for k, v in stable_levels.items()},
    }
    print("drift fit", results["drift"]["best_prob_blend"], results["drift"]["best_logit_blend"], flush=True)

    # ---------- shared change + stable null ----------
    print("changepoints", flush=True)

    def scan_condition(condition, step=2):
        rows = []
        for ep in train["episodes"].values():
            if ep["condition"] != condition:
                continue
            machine_rows = []
            for machine in ep["machines"].values():
                series = series_of(machine, ep["cutoff"])
                bits = build_bits(series, feat_gt0)
                prefixes = {pair: pair_prefix(bits, *pair) for pair in PAIRS}
                split = changepoint_scan(prefixes, 8, ep["cutoff"], min_cell=8, step=step)
                if split is None or split.get("split") is None and "c" not in split:
                    # changepoint_scan returns the best rec directly, or a dict with split None
                    pass
                if split is None or "c" not in split:
                    continue
                # fixed-pair profile on the full-history best pair, for sharpness
                full_best = best_pair_in_range(prefixes, 8, ep["cutoff"], min_cell=8)
                sharp = None
                if full_best is not None:
                    count, wins, _ = prefixes[full_best["pair"]]
                    sharp = fixed_pair_flip_scan(count, wins, 8, ep["cutoff"], min_cell=8)
                machine_rows.append(
                    {
                        "machine_id": machine["machine_id"],
                        "machine_type": machine["machine_type"],
                        "c": split["c"],
                        "gain": split["gain"],
                        "left_pair": split["left"]["pair"],
                        "right_pair": split["right"]["pair"],
                        "left_direction": split["left"]["direction"],
                        "right_direction": split["right"]["direction"],
                        "left_sep": split["left"]["sep"],
                        "right_sep": split["right"]["sep"],
                        "left_p_live": split["left"]["p_live"],
                        "left_p_dead": split["left"]["p_dead"],
                        "right_p_live": split["right"]["p_live"],
                        "right_p_dead": split["right"]["p_dead"],
                        "pair_changed": split["left"]["pair"] != split["right"]["pair"],
                        "direction_changed": split["left"]["direction"] != split["right"]["direction"],
                        "full_pair": split["full_pair"],
                        "sharp": sharp,
                    }
                )
            if not machine_rows:
                continue
            cs = [r["c"] for r in machine_rows]
            # episode change = median machine c
            med = median(cs)
            rows.append(
                {
                    "episode_id": ep["episode_id"],
                    "cutoff": ep["cutoff"],
                    "c_median": med,
                    "c_std": std(cs),
                    "c_min": min(cs),
                    "c_max": max(cs),
                    "c_over_cutoff": med / ep["cutoff"],
                    "ticks_before_cutoff": ep["cutoff"] - med,
                    "mean_gain": mean(r["gain"] for r in machine_rows),
                    "frac_pair_changed": mean(1.0 if r["pair_changed"] else 0.0 for r in machine_rows),
                    "frac_direction_changed": mean(1.0 if r["direction_changed"] else 0.0 for r in machine_rows),
                    "mean_sharp_width": mean(r["sharp"]["width_near_max"] for r in machine_rows if r["sharp"]),
                    "mean_sharp_gain": mean(r["sharp"]["gain"] for r in machine_rows if r["sharp"]),
                    "machines": machine_rows,
                }
            )
        return rows

    shared_cps = scan_condition("shared_change", step=1)
    stable_cps = scan_condition("stable", step=2)
    drift_cps = scan_condition("drift", step=2)

    def cp_summary(rows):
        return {
            "n_episodes": len(rows),
            "c_over_cutoff_mean": mean(r["c_over_cutoff"] for r in rows),
            "c_over_cutoff_std": std(r["c_over_cutoff"] for r in rows),
            "c_over_cutoff_quantiles": quantiles([r["c_over_cutoff"] for r in rows]),
            "ticks_before_cutoff_mean": mean(r["ticks_before_cutoff"] for r in rows),
            "ticks_before_cutoff_std": std(r["ticks_before_cutoff"] for r in rows),
            "ticks_before_cutoff_quantiles": quantiles([r["ticks_before_cutoff"] for r in rows]),
            "within_episode_c_std_mean": mean(r["c_std"] for r in rows),
            "within_episode_c_range_mean": mean(r["c_max"] - r["c_min"] for r in rows),
            "mean_gain": mean(r["mean_gain"] for r in rows),
            "frac_pair_changed_mean": mean(r["frac_pair_changed"] for r in rows),
            "frac_direction_changed_mean": mean(r["frac_direction_changed"] for r in rows),
            "sharp_width_mean": mean(r["mean_sharp_width"] for r in rows),
            "sharp_gain_mean": mean(r["mean_sharp_gain"] for r in rows),
            "episodes": [
                {k: v for k, v in r.items() if k != "machines"}
                for r in rows
            ],
        }

    results["changepoints"] = {
        "shared_change": cp_summary(shared_cps),
        "stable": cp_summary(stable_cps),
        "drift": cp_summary(drift_cps),
    }
    print(
        "cp gains",
        results["changepoints"]["shared_change"]["mean_gain"],
        results["changepoints"]["stable"]["mean_gain"],
        results["changepoints"]["drift"]["mean_gain"],
        "shared c/cutoff",
        results["changepoints"]["shared_change"]["c_over_cutoff_mean"],
        flush=True,
    )

    # What changes at the shared tick: levels vs pair vs size. Use episode median c.
    print("what changes", flush=True)
    change_detail = []
    for ep_row in shared_cps:
        ep = train["episodes"][ep_row["episode_id"]]
        c = int(round(ep_row["c_median"]))
        for machine in ep["machines"].values():
            series = series_of(machine, ep["cutoff"])
            bits = build_bits(series, feat_gt0)
            prefixes = {pair: pair_prefix(bits, *pair) for pair in PAIRS}
            left = best_pair_in_range(prefixes, 8, c, min_cell=8)
            right = best_pair_in_range(prefixes, c, ep["cutoff"], min_cell=8)
            if left is None or right is None:
                continue
            # size mixes before/after, unconditional on win
            def mix(lo, hi):
                cnt = Counter(series[t] for t in range(lo, hi) if series[t] is not None and series[t] > 0)
                return mix_of(cnt)

            change_detail.append(
                {
                    "episode_id": ep["episode_id"],
                    "machine_type": machine["machine_type"],
                    "left_pair": left["pair"],
                    "right_pair": right["pair"],
                    "left_dir": left["direction"],
                    "right_dir": right["direction"],
                    "left_live": left["p_live"],
                    "left_dead": left["p_dead"],
                    "right_live": right["p_live"],
                    "right_dead": right["p_dead"],
                    "pair_changed": left["pair"] != right["pair"],
                    "dir_changed": left["direction"] != right["direction"],
                    "live_delta": abs(left["p_live"] - right["p_live"]),
                    "dead_delta": abs(left["p_dead"] - right["p_dead"]),
                    "size_left": mix(8, c),
                    "size_right": mix(c, ep["cutoff"]),
                }
            )
    # size TV before vs after, pooled by type
    size_tvs = []
    for row in change_detail:
        if row["size_left"] and row["size_right"]:
            a = Counter({int(k): row["size_left"][k] * row["size_left"]["n"] for k in row["size_left"] if k != "n"})
            # mix stored as fractions; recover TV from fractions directly
            tv = 0.5 * sum(abs(row["size_left"][str(p)] - row["size_right"][str(p)]) for p in SIZE_PAYOUTS)
            size_tvs.append(tv)
    # Compare to stable split at same relative location — already have gains.
    # Fraction of machines whose right-hand levels still match the type's stable levels (rule continues, maybe new pair).
    def near_type_levels(row, live_key, dead_key):
        D, L = stable_levels[row["machine_type"]]
        return abs(row[live_key] - L) < 0.08 and abs(row[dead_key] - D) < 0.08

    results["shared_what_changes"] = {
        "n": len(change_detail),
        "frac_pair_changed": mean(1 if r["pair_changed"] else 0 for r in change_detail),
        "frac_dir_changed": mean(1 if r["dir_changed"] else 0 for r in change_detail),
        "frac_pair_or_dir_changed": mean(1 if r["pair_changed"] or r["dir_changed"] else 0 for r in change_detail),
        "live_delta_mean": mean(r["live_delta"] for r in change_detail),
        "dead_delta_mean": mean(r["dead_delta"] for r in change_detail),
        "frac_left_matches_stable_levels": mean(1 if near_type_levels(r, "left_live", "left_dead") else 0 for r in change_detail),
        "frac_right_matches_stable_levels": mean(1 if near_type_levels(r, "right_live", "right_dead") else 0 for r in change_detail),
        "size_tv_mean": mean(size_tvs),
        "size_tv_median": median(size_tvs),
        "right_pair_counts": dict(Counter(str(r["right_pair"]) for r in change_detail)),
        "left_pair_counts": dict(Counter(str(r["left_pair"]) for r in change_detail)),
    }
    print("what changes", {k: results["shared_what_changes"][k] for k in list(results["shared_what_changes"])[:8]}, flush=True)

    # ---------- size modes ----------
    print("size", flush=True)
    # On stable train, among wins, which lag of which feature best splits the size multinom.
    size_lag_scores = {feat_name: Counter() for feat_name in ("gt0", "ge2", "ge5")}
    size_lag_tvs = {feat_name: defaultdict(list) for feat_name in ("gt0", "ge2", "ge5")}
    # Also joint pattern of ge5 at lags 1,2,4
    joint_cells = defaultdict(lambda: Counter())  # (type, pattern) -> size counter
    type_cells = defaultdict(lambda: Counter())
    for ep in train["episodes"].values():
        if ep["condition"] != "stable":
            continue
        for machine in ep["machines"].values():
            series = series_of(machine, ep["cutoff"])
            for feat_name, feat in FEATS.items():
                for lag in LAGS:
                    def cond(t, lag=lag, feat=feat, series=series):
                        if t - lag < 0 or series[t - lag] is None:
                            return None
                        return bool(feat(series[t - lag]))

                    yes, no = size_counts(series, 8, ep["cutoff"], cond)
                    if sum(yes.values()) < 8 or sum(no.values()) < 8:
                        continue
                    gain = size_gain(yes, no)
                    tv = tv_distance(yes, no)
                    size_lag_scores[feat_name][lag] += gain
                    size_lag_tvs[feat_name][lag].append(tv)
            # joint
            for t in range(8, ep["cutoff"]):
                if series[t] is None or series[t] == 0:
                    continue
                if any(series[t - lag] is None for lag in (1, 2, 4)):
                    continue
                pattern = tuple(int(series[t - lag] >= 5) for lag in (1, 2, 4))
                joint_cells[(machine["machine_type"], pattern)][series[t]] += 1
                type_cells[machine["machine_type"]][series[t]] += 1

    size_summary = {}
    for feat_name in ("gt0", "ge2", "ge5"):
        scores = size_lag_scores[feat_name]
        size_summary[feat_name] = {
            str(lag): {
                "total_ll_gain": scores[lag],
                "mean_tv": mean(size_lag_tvs[feat_name][lag]),
                "n_machines": len(size_lag_tvs[feat_name][lag]),
            }
            for lag in LAGS
        }
    joint_summary = {}
    for (mtype, pattern), cnt in sorted(joint_cells.items()):
        joint_summary[f"{mtype}:{pattern}"] = mix_of(cnt)
    # Collapse joint to "any big" vs "no big" and to xor of the three bits, per type.
    collapse_modes = {}
    for mtype in ("steady", "balanced", "jackpot"):
        any_big, no_big = Counter(), Counter()
        xor1, xor0 = Counter(), Counter()
        for pattern, cnt in ((p, c) for (mt, p), c in joint_cells.items() if mt == mtype):
            if any(pattern):
                any_big += cnt
            else:
                no_big += cnt
            if sum(pattern) % 2:
                xor1 += cnt
            else:
                xor0 += cnt
        # two modes from the 8 patterns: cluster later via validation
        collapse_modes[mtype] = {
            "any_big": mix_of(any_big),
            "no_big": mix_of(no_big),
            "tv_any": tv_distance(any_big, no_big),
            "xor1": mix_of(xor1),
            "xor0": mix_of(xor0),
            "tv_xor": tv_distance(xor1, xor0),
            "patterns": {str(p): mix_of(c) for (mt, p), c in joint_cells.items() if mt == mtype},
        }

    # Validation: conditional size mix = probs[1:] / p_win, cluster per type on stable.
    val_size = defaultdict(list)
    for ep in valid["episodes"].values():
        if ep["condition"] != "stable":
            continue
        for machine in ep["machines"].values():
            probs = valid["targets"][machine["query_id"]]
            p_win = 1.0 - probs[0]
            if p_win < 0.2:
                continue  # dead cell: size mix is weakly identified / same mix actually still in the vector
            mix = tuple(probs[i] / p_win for i in range(1, 5))
            series = series_of(machine, ep["cutoff"])
            cutoff = ep["cutoff"]
            bigs = tuple(int(series[cutoff - lag] >= 5) if series[cutoff - lag] is not None else -1 for lag in (1, 2, 4))
            wins = tuple(int(series[cutoff - lag] > 0) if series[cutoff - lag] is not None else -1 for lag in (1, 2, 4))
            val_size[machine["machine_type"]].append({"mix": mix, "bigs": bigs, "wins": wins, "signal": machine["signal"], "p_win": p_win})

    def cluster_mixes(rows, tol=0.01):
        groups = []
        for row in rows:
            mix = row["mix"]
            placed = False
            for g in groups:
                if max(abs(a - b) for a, b in zip(g["center"], mix)) < tol:
                    n = g["n"]
                    g["center"] = tuple((n * a + b) / (n + 1) for a, b in zip(g["center"], mix))
                    g["n"] += 1
                    g["members"].append(row)
                    placed = True
                    break
            if not placed:
                groups.append({"center": mix, "n": 1, "members": [row]})
        return groups

    size_clusters = {}
    for mtype, rows in val_size.items():
        groups = cluster_mixes(rows, tol=0.015)
        # also try to explain cluster id by bigs pattern
        # assign each row its nearest of the big groups
        big_groups = [g for g in groups if g["n"] >= 3]
        pred = []
        for row in rows:
            if not big_groups:
                continue
            nearest = min(big_groups, key=lambda g: max(abs(a - b) for a, b in zip(g["center"], row["mix"])))
            pred.append((row["bigs"], tuple(round(x, 4) for x in nearest["center"]), row["wins"]))
        # purity: for each bigs pattern, fraction sharing the modal cluster
        by_pat = defaultdict(list)
        for bigs, center, wins in pred:
            by_pat[bigs].append(center)
        purity = {}
        for pat, centers in by_pat.items():
            mode = Counter(centers).most_common(1)[0]
            purity[str(pat)] = {"n": len(centers), "modal_frac": mode[1] / len(centers), "center": mode[0]}
        size_clusters[mtype] = {
            "n_rows_liveish": len(rows),
            "n_clusters_tol_0.015": len(groups),
            "clusters": [
                {"n": g["n"], "center": g["center"]} for g in sorted(groups, key=lambda g: -g["n"])
            ],
            "purity_by_bigs_124": purity,
        }

    # Which single lag's big-bit predicts validation size cluster most cleanly?
    # Use all stable targets including dead, because the mix is in the probability vector even if p_win is small.
    # Dead-cell probabilities still equal p_win * mix, so mix is identified whenever p_win is not tiny.
    val_size_all = defaultdict(list)
    for ep in valid["episodes"].values():
        if ep["condition"] != "stable":
            continue
        for machine in ep["machines"].values():
            probs = valid["targets"][machine["query_id"]]
            p_win = 1.0 - probs[0]
            if p_win < 0.05:
                continue
            mix = tuple(probs[i] / p_win for i in range(1, 5))
            series = series_of(machine, ep["cutoff"])
            cutoff = ep["cutoff"]
            feats = {}
            for lag in LAGS:
                payout = series[cutoff - lag]
                feats[lag] = {
                    "gt0": int(payout > 0),
                    "ge2": int(payout >= 2),
                    "ge5": int(payout >= 5),
                    "payout": payout,
                }
            val_size_all[machine["machine_type"]].append({"mix": mix, "feats": feats, "signal": machine["signal"]})

    lag_purity = {feat_name: {} for feat_name in ("gt0", "ge2", "ge5")}
    for mtype, rows in val_size_all.items():
        groups = cluster_mixes(rows, tol=0.02)
        big = [g for g in groups if g["n"] >= 3]
        if len(big) < 1:
            continue
        for feat_name in ("gt0", "ge2", "ge5"):
            lag_purity[feat_name][mtype] = {}
            for lag in LAGS:
                by = defaultdict(list)
                for row in rows:
                    nearest = min(big, key=lambda g: sum(abs(a - b) for a, b in zip(g["center"], row["mix"])))
                    label = tuple(round(x, 3) for x in nearest["center"])
                    by[row["feats"][lag][feat_name]].append(label)
                # uncertainty: majority purity averaged
                purities = []
                for bit, labels in by.items():
                    if not labels:
                        continue
                    purities.append(Counter(labels).most_common(1)[0][1] / len(labels))
                # also between-bit center distance
                lag_purity[feat_name][mtype][str(lag)] = {
                    "mean_purity": mean(purities) if purities else None,
                    "n0": len(by.get(0, [])),
                    "n1": len(by.get(1, [])),
                }

    results["size"] = {
        "train_stable_lag_scores": size_summary,
        "joint_124_ge5": collapse_modes,
        "val_live_clusters": {
            mtype: {
                "n_rows_liveish": block["n_rows_liveish"],
                "n_clusters": block["n_clusters_tol_0.015"],
                "clusters": block["clusters"],
                "purity_by_bigs_124": block["purity_by_bigs_124"],
            }
            for mtype, block in size_clusters.items()
        },
        "val_lag_purity": lag_purity,
    }
    print("size lag ge5", {k: round(v["total_ll_gain"], 1) for k, v in size_summary["ge5"].items()}, flush=True)

    # ---------- neighbours ----------
    print("neighbours", flush=True)
    neigh_gains = []  # per machine best gain
    offset_gains = defaultdict(list)  # (offset, lag) -> gains
    # null: use own lag that is NOT in the pair, already measured as third. Compare.
    for ep in train["episodes"].values():
        if ep["condition"] != "stable":
            continue
        machines = list(ep["machines"].values())
        bits_list = []
        cells_list = []
        pairs_used = []
        for machine in machines:
            series = series_of(machine, ep["cutoff"])
            bits = build_bits(series, feat_gt0)
            prefixes = {pair: pair_prefix(bits, *pair) for pair in PAIRS}
            best = best_pair_in_range(prefixes, 8, ep["cutoff"], min_cell=10)
            bits_list.append(bits)
            if best is None:
                cells_list.append(None)
                pairs_used.append(None)
            else:
                cells_list.append(prefixes[best["pair"]][2])
                pairs_used.append(best["pair"])
        for i, machine in enumerate(machines):
            if cells_list[i] is None:
                continue
            best_rec = None
            for j, other in enumerate(machines):
                if i == j:
                    continue
                offset = (other["index"] - machine["index"]) % 8
                for lag in LAGS:
                    rec = neighbour_ll(bits_list[i], cells_list[i], bits_list[j], lag, 8, ep["cutoff"], min_cell=6)
                    if rec is None:
                        continue
                    offset_gains[(offset, lag)].append(rec["gain"])
                    if best_rec is None or rec["gain"] > best_rec["gain"]:
                        best_rec = {**rec, "offset": offset, "lag": lag, "j": other["index"]}
            if best_rec:
                neigh_gains.append(best_rec)

    # Validation residual: if collapse rmse is already ~0, neighbour cannot move p_win.
    # Still measure within (type, own agree of BEST global pairs) — use per-machine discovered pair.
    # Group stable val p_win by (type, live) already in results. Additionally residual std.

    # A direct target test: for each offset, lag, feature, regress residual of p_win after (type, own cell).
    # Own cell from the machine's best pair on its history (gt0).
    own_rows = [r for r in target_rows["gt0"] if r["condition"] == "stable"]
    # Build lookup of bits at cutoff for every machine
    val_bits = {}
    for ep in valid["episodes"].values():
        if ep["condition"] != "stable":
            continue
        for machine in ep["machines"].values():
            series = series_of(machine, ep["cutoff"])
            val_bits[(ep["episode_id"], machine["index"])] = {
                "series": series,
                "cutoff": ep["cutoff"],
                "type": machine["machine_type"],
                "signal": machine["signal"],
                "p_win": 1.0 - valid["targets"][machine["query_id"]][0],
                "probs": valid["targets"][machine["query_id"]],
            }
    # Map own live from target_rows
    own_live = {(r["episode_id"], int(r["machine_id"].rsplit("m", 1)[-1])): r for r in own_rows}

    neigh_target = []
    for offset in range(1, 8):
        for lag in LAGS:
            # residual groups: (type, own_live, neigh_bit)
            groups = defaultdict(list)
            base = defaultdict(list)
            for (eid, idx), info in val_bits.items():
                own = own_live.get((eid, idx))
                if own is None:
                    continue
                j = (idx + offset) % 8
                other = val_bits.get((eid, j))
                if other is None:
                    continue
                payout = other["series"][info["cutoff"] - lag]
                if payout is None:
                    continue
                bit = int(payout > 0)
                key = (info["type"], own["live"], bit)
                groups[key].append(info["p_win"])
                base[(info["type"], own["live"])].append(info["p_win"])
            # variance reduction
            def sse(g):
                total = 0.0
                for vals in g.values():
                    if len(vals) < 2:
                        continue
                    m = mean(vals)
                    total += sum((v - m) ** 2 for v in vals)
                return total
            sse0 = sse(base)
            sse1 = sse(groups)
            # mean abs difference of the two neigh bits inside each (type, live)
            lifts = []
            for mtype in ("steady", "balanced", "jackpot"):
                for live in (0, 1):
                    a = groups.get((mtype, live, 0), [])
                    b = groups.get((mtype, live, 1), [])
                    if len(a) >= 3 and len(b) >= 3:
                        lifts.append(abs(mean(a) - mean(b)))
            neigh_target.append(
                {
                    "offset": offset,
                    "lag": lag,
                    "sse0": sse0,
                    "sse1": sse1,
                    "var_drop": (sse0 - sse1) / sse0 if sse0 else None,
                    "mean_abs_lift": mean(lifts) if lifts else None,
                    "max_abs_lift": max(lifts) if lifts else None,
                }
            )
    neigh_target.sort(key=lambda r: -(r["var_drop"] or -1))

    results["neighbours"] = {
        "n_machines": len(neigh_gains),
        "best_extra_ll_mean": mean(r["gain"] for r in neigh_gains),
        "best_extra_ll_median": median([r["gain"] for r in neigh_gains]),
        "best_lift_mean": mean(r["lift"] for r in neigh_gains),
        "best_lift_p90": quantiles([r["lift"] for r in neigh_gains]).get("0.9"),
        "best_offset_counts": dict(Counter(r["offset"] for r in neigh_gains)),
        "best_lag_counts": dict(Counter(str(r["lag"]) for r in neigh_gains)),
        "mean_gain_by_offset_lag_top": [
            {
                "offset": off,
                "lag": lag,
                "mean_gain": mean(v),
                "n": len(v),
            }
            for (off, lag), v in sorted(offset_gains.items(), key=lambda kv: -mean(kv[1]))[:12]
        ],
        "target_residual_top": neigh_target[:12],
        "target_residual_median_var_drop": median([r["var_drop"] for r in neigh_target if r["var_drop"] is not None]),
    }
    print("neigh best lift", results["neighbours"]["best_lift_mean"], "target", neigh_target[0], flush=True)

    # ---------- residual autocorrelation ----------
    print("autocorr", flush=True)
    acfs = defaultdict(list)
    ratios = []
    for ep in train["episodes"].values():
        if ep["condition"] != "stable":
            continue
        for machine in ep["machines"].values():
            series = series_of(machine, ep["cutoff"])
            bits = build_bits(series, feat_gt0)
            prefixes = {pair: pair_prefix(bits, *pair) for pair in PAIRS}
            best = best_pair_in_range(prefixes, 8, ep["cutoff"], min_cell=10)
            if best is None:
                continue
            count, wins, cells = prefixes[best["pair"]]
            rec = residual_autocorr(bits, cells, best["p_agree"], best["p_differ"])
            if rec is None:
                continue
            for lag, val in rec["acf"].items():
                if val is not None:
                    acfs[lag].append(val)
            for cell, info in rec["extra_binomial"].items():
                if info["ratio"] is not None:
                    ratios.append(info["ratio"])
    # Also autocorrelation of RAW wins for comparison
    raw_acf = defaultdict(list)
    for ep in train["episodes"].values():
        if ep["condition"] != "stable":
            continue
        for machine in ep["machines"].values():
            series = series_of(machine, ep["cutoff"])
            bits = build_bits(series, feat_gt0)
            xs = [(t, int(bits[t])) for t in range(len(bits)) if bits[t] >= 0]
            by = {t: v for t, v in xs}
            for lag in range(1, 9):
                a, b = [], []
                for t, v in xs:
                    if t - lag in by:
                        a.append(v)
                        b.append(by[t - lag])
                if len(a) < 30:
                    continue
                x = np.asarray(a, float)
                y = np.asarray(b, float)
                x = x - x.mean()
                y = y - y.mean()
                denom = math.sqrt(float((x * x).sum() * (y * y).sum()))
                raw_acf[lag].append(float((x * y).sum() / denom) if denom else 0.0)

    results["autocorr"] = {
        "residual_mean": {lag: mean(v) for lag, v in acfs.items()},
        "residual_std": {lag: std(v) for lag, v in acfs.items()},
        "raw_mean": {str(lag): mean(v) for lag, v in raw_acf.items()},
        "extra_binomial_ratio_mean": mean(ratios),
        "extra_binomial_ratio_median": median(ratios),
        "extra_binomial_ratio_p90": quantiles(ratios).get("0.9"),
        "n_cells": len(ratios),
    }
    print("acf", results["autocorr"]["residual_mean"], "extra", results["autocorr"]["extra_binomial_ratio_mean"], flush=True)

    # ---------- signal effect on exact targets, using best collapsing pair set ----------
    # For each stable val machine, take agree bit of its history-best pair and see p_win by signal.
    sig_rows = [r for r in target_rows["gt0"] if r["condition"] == "stable"]
    signal_effect = {}
    for mtype in ("steady", "balanced", "jackpot"):
        signal_effect[mtype] = {}
        for live in (0, 1):
            signal_effect[mtype][str(live)] = {}
            for sig in (-1, 0, 1):
                vals = [r["p_win"] for r in sig_rows if r["machine_type"] == mtype and r["live"] == live and r["signal"] == sig]
                signal_effect[mtype][str(live)][str(sig)] = {
                    "n": len(vals),
                    "mean": mean(vals) if vals else None,
                    "std": std(vals) if vals else None,
                    "min": min(vals) if vals else None,
                    "max": max(vals) if vals else None,
                }
    results["signal_effect_stable_val"] = signal_effect

    # Machine-level vs type-level: distribution of per-machine p_live from train stable already stored.
    # Exact: within a (type, live, signal) cell, std of validation p_win. If ~0, levels are global not per machine.

    # ---------- how often the three known pairs win ----------
    known = {(1, 3), (2, 5), (4, 7)}
    gt0_machines = stable_by_feat["gt0"]["machines"]
    results["known_pair_coverage"] = {
        "frac_in_known": mean(1.0 if tuple(r["pair"]) in known else 0.0 for r in gt0_machines),
        "n": len(gt0_machines),
        "outside": dict(Counter(str(tuple(r["pair"])) for r in gt0_machines if tuple(r["pair"]) not in known)),
    }

    # Direction balance within each pair
    dir_by_pair = defaultdict(Counter)
    for r in gt0_machines:
        dir_by_pair[str(tuple(r["pair"]))][r["direction"]] += 1
    results["direction_by_pair"] = {k: dict(v) for k, v in dir_by_pair.items()}

    # ---------- shared change: do ALL 8 machines change? ----------
    all8 = []
    for ep_row in shared_cps:
        machines = ep_row["machines"]
        # a machine "changes" if pair or direction differs across the split
        flags = [m["pair_changed"] or m["direction_changed"] for m in machines]
        all8.append(
            {
                "episode_id": ep_row["episode_id"],
                "n_changed": sum(flags),
                "n": len(flags),
                "c_std": ep_row["c_std"],
            }
        )
    results["shared_all_machines"] = {
        "episodes": all8,
        "frac_episodes_all8_changed": mean(1.0 if r["n_changed"] == r["n"] == 8 else 0.0 for r in all8),
        "mean_n_changed": mean(r["n_changed"] for r in all8),
    }

    # ---------- stable: is the best split gain compatible with noise? ----------
    # Already compared mean gains. Add per-machine gain quantiles.
    def flat_gains(rows):
        return [m["gain"] for ep in rows for m in ep["machines"]]

    results["split_gain_quantiles"] = {
        "shared": quantiles(flat_gains(shared_cps)),
        "stable": quantiles(flat_gains(stable_cps)),
        "drift": quantiles(flat_gains(drift_cps)),
        "shared_mean": mean(flat_gains(shared_cps)),
        "stable_mean": mean(flat_gains(stable_cps)),
        "drift_mean": mean(flat_gains(drift_cps)),
    }

    OUT.write_text(json.dumps(jsonable(results), indent=2))
    print("wrote", OUT, flush=True)


def pred_clip(u, L, D):
    return (1.0 - u) * L + u * D


if __name__ == "__main__":
    main()
