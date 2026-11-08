"""Train-only descriptive calibration for the documented synthetic data family.

This module deliberately knows nothing about participant models or validation
forecasts. It summarizes public/train observations and suggests only the
type/signal win-rate and type-level payout-mix fields of ``main_like``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from slot_forecasting.io import read_history, read_labels, read_queries
from slot_forecasting.schema import index_history


SIGNALS = (-1, 0, 1)
POSITIVE_PAYOUTS = (1, 2, 5, 20)
TOLERANCES = {
    "type_signal_win_rate_max_abs": 0.08,
    "positive_payout_mix_tv_max": 0.10,
    "lag_correlation_max_abs": 0.10,
    "machine_win_rate_sd_abs": 0.06,
    "cutoff_mean_abs_ticks": 8,
    "short_sampling_fraction_abs": 0.08,
    "sampling_pattern_exact": True,
}


def _train_dir(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    return path / "train" if (path / "train").is_dir() else path


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _phi(pairs: list[tuple[bool, bool]]) -> float | None:
    """Pearson/phi correlation for binary outcomes; undefined constants -> None."""
    if not pairs:
        return None
    x = [int(a) for a, _ in pairs]
    y = [int(b) for _, b in pairs]
    mx, my = _mean(x), _mean(y)
    vx = math.fsum((v - mx) ** 2 for v in x)
    vy = math.fsum((v - my) ** 2 for v in y)
    if vx == 0 or vy == 0:
        return None
    cov = math.fsum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / math.sqrt(vx * vy)


def summarize_dataset(data_root_or_train_dir: str | Path) -> dict:
    """Describe one public train split using queries, observed history and labels."""
    train = _train_dir(data_root_or_train_dir)
    queries = read_queries(train / "queries.csv")
    history = read_history(train / "history.csv")
    history_by_machine = index_history(queries, history)
    labels = read_labels(train / "labels.csv", queries)
    # Use the full observable history for the simulator's empirical base rates
    # and payout marginals. The single next-outcome label is kept as a separate
    # diagnostic and is never substituted for a machine's observed history.
    rates = defaultdict(Counter)
    payout_counts = defaultdict(Counter)
    history_by_query = {(q.episode_id, q.machine_id): q for q in queries}
    history_outcomes_by_machine = defaultdict(list)
    for play in history:
        q = history_by_query[(play.episode_id, play.machine_id)]
        rates[(q.machine_type, q.signal)]["n"] += 1
        rates[(q.machine_type, q.signal)]["wins"] += play.payout > 0
        history_outcomes_by_machine[(q.episode_id, q.machine_id)].append(play.payout > 0)
        if play.payout > 0:
            payout_counts[q.machine_type][play.payout] += 1
    label_rates = defaultdict(Counter)
    for q in queries:
        outcome = labels[q.query_id]
        label_rates[(q.machine_type, q.signal)]["n"] += 1
        label_rates[(q.machine_type, q.signal)]["wins"] += outcome > 0

    rate_stats = {}
    for machine_type in sorted({q.machine_type for q in queries}):
        rate_stats[machine_type] = {}
        for signal in SIGNALS:
            cell = rates[(machine_type, signal)]
            rate_stats[machine_type][str(signal)] = {
                "n": cell["n"], "wins": cell["wins"],
                "rate": cell["wins"] / cell["n"] if cell["n"] else None,
            }

    payout_mix = {}
    for machine_type in sorted({q.machine_type for q in queries}):
        counts = payout_counts[machine_type]
        total = sum(counts.values())
        payout_mix[machine_type] = {
            "n_positive": total,
            "counts": {str(p): counts[p] for p in POSITIVE_PAYOUTS},
            "proportions": [counts[p] / total if total else 0.0 for p in POSITIVE_PAYOUTS],
        }

    # Label each historical tick as win/loss, then pair only exact observed
    # contexts (query cutoff-lag). Missing observations are never imputed.
    wins_at = {(p.episode_id, p.machine_id, p.tick): p.payout > 0 for p in history}
    pairs_by_lag_type = defaultdict(list)
    pairs_by_lag = defaultdict(list)
    for q in queries:
        ticks = sorted(p.tick for p in history_by_machine.get((q.episode_id, q.machine_id), ()))
        for tick in ticks:
            current = wins_at[(q.episode_id, q.machine_id, tick)]
            for lag in range(1, 9):
                previous = wins_at.get((q.episode_id, q.machine_id, tick - lag))
                if previous is not None:
                    pairs_by_lag_type[(q.machine_type, lag)].append((previous, current))
                    pairs_by_lag[lag].append((previous, current))
    lag_corr = {
        "overall": {str(lag): {"n": len(pairs_by_lag[lag]), "phi": _phi(pairs_by_lag[lag])}
                    for lag in range(1, 9)},
        "by_type": {},
    }
    for machine_type in sorted({q.machine_type for q in queries}):
        lag_corr["by_type"][machine_type] = {
            str(lag): {"n": len(pairs_by_lag_type[(machine_type, lag)]),
                       "phi": _phi(pairs_by_lag_type[(machine_type, lag)])}
            for lag in range(1, 9)
        }

    cutoffs = [q.cutoff for q in queries]
    observed_lengths = []
    recent_counts = []
    older_counts = []
    short_exact = []
    long_exact = []
    for q in queries:
        ticks = {p.tick for p in history_by_machine.get((q.episode_id, q.machine_id), ())}
        recent = sum(q.cutoff - 40 <= tick < q.cutoff for tick in ticks)
        older = sum(tick < q.cutoff - 40 for tick in ticks)
        observed_lengths.append(len(ticks))
        recent_counts.append(recent)
        older_counts.append(older)
        short_exact.append(recent == 40 and older == 24 and len(ticks) == 64)
        long_exact.append(len(ticks) == q.cutoff and recent == min(40, q.cutoff) and older == max(0, q.cutoff - 40))

    # Distinct machine-cutoff pairs are the independent observational units
    # available here; report overall and per-type sample SD of empirical rates.
    machine_outcomes = history_outcomes_by_machine
    machine_types = {}
    for q in queries:
        key = (q.episode_id, q.machine_id)
        machine_types[key] = q.machine_type
    machine_rates = defaultdict(list)
    for key, outcomes in machine_outcomes.items():
        machine_rates[machine_types[key]].append(sum(outcomes) / len(outcomes))
    all_machine_rates = [sum(v) / len(v) for v in machine_outcomes.values()]

    conditions_path = train / "conditions.csv"
    condition_counts = Counter()
    condition_by_query = {}
    if conditions_path.exists():
        import csv
        with conditions_path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                condition_counts[row.get("condition", "unknown")] += 1
                condition_by_query[row["query_id"]] = row.get("condition", "unknown")

    sampling_conforms = True
    for q in queries:
        condition = condition_by_query.get(q.query_id)
        key = (q.episode_id, q.machine_id)
        ticks = {p.tick for p in history_by_machine.get(key, ())}
        dense = len(ticks) == q.cutoff and ticks == set(range(q.cutoff))
        recent = sum(q.cutoff - 40 <= tick < q.cutoff for tick in ticks)
        older = sum(tick < q.cutoff - 40 for tick in ticks)
        sparse = recent == 40 and older == 24 and len(ticks) == 64
        sampling_conforms &= (sparse if condition == "short_history" else dense)

    return {
        "source": str(train),
        "counts": {"queries": len(queries), "history_rows": len(history), "episodes": len({q.episode_id for q in queries}),
                   "machine_cutoffs": len(machine_outcomes), "conditions": dict(condition_counts)},
        "type_signal_win_rates": rate_stats,
        "label_type_signal_win_rates": {
            t: {str(s): {"n": label_rates[(t, s)]["n"],
                         "rate": (label_rates[(t, s)]["wins"] / label_rates[(t, s)]["n"]
                                  if label_rates[(t, s)]["n"] else None)}
                for s in SIGNALS}
            for t in sorted({q.machine_type for q in queries})
        },
        "positive_payout_mix_by_type": payout_mix,
        "win_lag_correlations": lag_corr,
        "cutoffs": {"n": len(cutoffs), "mean": _mean(cutoffs), "min": min(cutoffs), "max": max(cutoffs),
                    "distribution": {str(k): v for k, v in sorted(Counter(cutoffs).items())}},
        "history_sampling": {
            "short_fraction": sum(short_exact) / len(short_exact),
            "short_40_recent_24_older_exact_fraction": sum(short_exact) / len(short_exact),
            "long_complete_fraction": sum(long_exact) / len(long_exact),
            "all_histories_match_long_or_short_pattern": all(s or l for s, l in zip(short_exact, long_exact)),
            "condition_specific_pattern_exact": sampling_conforms,
            "recent_count_distribution": {str(k): v for k, v in sorted(Counter(recent_counts).items())},
            "older_count_distribution": {str(k): v for k, v in sorted(Counter(older_counts).items())},
            "observed_length_distribution": {str(k): v for k, v in sorted(Counter(observed_lengths).items())},
        },
        "machine_win_rate_spread": {
            "n": len(all_machine_rates),
            "sd": statistics.stdev(all_machine_rates) if len(all_machine_rates) > 1 else 0.0,
            "by_type": {t: {"n": len(v), "sd": statistics.stdev(v) if len(v) > 1 else 0.0}
                        for t, v in sorted(machine_rates.items())},
        },
    }


def _tv(left: list[float], right: list[float]) -> float:
    return 0.5 * math.fsum(abs(a - b) for a, b in zip(left, right))


def compare_statistics(real: dict, synthetic: dict) -> dict:
    """Compare summary statistics against fixed descriptive tolerances."""
    cells = []
    for t, signals in real["type_signal_win_rates"].items():
        for signal, stats in signals.items():
            other = synthetic["type_signal_win_rates"].get(t, {}).get(signal, {}).get("rate")
            value = stats["rate"]
            cells.append({"type": t, "signal": int(signal), "real": value, "generated": other,
                          "abs_diff": abs(value - other) if value is not None and other is not None else None})
    mixes = []
    for t, stats in real["positive_payout_mix_by_type"].items():
        other = synthetic["positive_payout_mix_by_type"].get(t, {}).get("proportions")
        mixes.append({"type": t, "real": stats["proportions"], "generated": other,
                      "tv": _tv(stats["proportions"], other) if other is not None else None})
    lags = []
    for t, lag_data in real["win_lag_correlations"]["by_type"].items():
        for lag, stats in lag_data.items():
            other = synthetic["win_lag_correlations"]["by_type"].get(t, {}).get(lag, {}).get("phi")
            value = stats["phi"]
            lags.append({"type": t, "lag": int(lag), "n_real": stats["n"], "real": value,
                         "generated": other,
                         "abs_diff": abs(value - other) if value is not None and other is not None else None})
    sd_diff = abs(real["machine_win_rate_spread"]["sd"] - synthetic["machine_win_rate_spread"]["sd"])
    cutoff_diff = abs(real["cutoffs"]["mean"] - synthetic["cutoffs"]["mean"])
    short_diff = abs(real["history_sampling"]["short_fraction"] - synthetic["history_sampling"]["short_fraction"])
    pattern_equal = (real["history_sampling"]["condition_specific_pattern_exact"]
                     and synthetic["history_sampling"]["condition_specific_pattern_exact"])
    def max_or_unavailable(values):
        return None if not values or any(v is None for v in values) else max(values)
    metrics = {
        "type_signal_win_rate_max_abs": max_or_unavailable([r["abs_diff"] for r in cells]),
        "positive_payout_mix_tv_max": max_or_unavailable([r["tv"] for r in mixes]),
        "lag_correlation_max_abs": max_or_unavailable([r["abs_diff"] for r in lags]),
        "machine_win_rate_sd_abs": sd_diff,
        "cutoff_mean_abs_ticks": cutoff_diff,
        "short_sampling_fraction_abs": short_diff,
        "sampling_pattern_exact": pattern_equal,
    }
    passed = {}
    for k, tolerance in TOLERANCES.items():
        value = metrics[k]
        if isinstance(tolerance, bool):
            passed[k] = value is tolerance
        else:
            passed[k] = value is not None and math.isfinite(value) and value <= tolerance
    return {"tolerances": TOLERANCES, "metrics": metrics, "within_descriptive_bands": passed,
            "all_bands_met": all(passed.values()), "type_signal_cells": cells,
            "positive_payout_mix": mixes, "lag_correlations": lags}


def calibrate_main_like(real_train_dir: str | Path) -> dict:
    """Return data-only main_like overrides (signal order -1,0,1; payout 1,2,5,20)."""
    stats = summarize_dataset(real_train_dir)
    # GeneratorConfig requires strict interior probabilities; reserve one
    # pseudo-count at each boundary for sparse/future-compatible train splits.
    eps = 1e-6
    rates = {}
    for machine_type, cells in stats["type_signal_win_rates"].items():
        available = [v["rate"] for v in cells.values() if v["rate"] is not None]
        fallback = _mean(available) if available else 0.5
        rates[machine_type] = [min(1.0 - eps, max(eps, cells[str(signal)]["rate"]
                                                  if cells[str(signal)]["rate"] is not None else fallback))
                               for signal in SIGNALS]
    mixes = {}
    for t, v in stats["positive_payout_mix_by_type"].items():
        proportions = v["proportions"]
        if v["n_positive"] == 0:
            proportions = [0.25] * 4
        mixes[t] = proportions
    return {"base_win_rates": rates, "payout_mixes": mixes}


def _markdown_comparison(real: dict, generated: dict, comparison: dict) -> str:
    lines = ["| Statistic | Real train | Generated train | Difference | Band |", "|---|---:|---:|---:|---:|"]
    metric = comparison["metrics"]
    for label, key, fmt in (
        ("Type×signal win rate (max abs)", "type_signal_win_rate_max_abs", ".4f"),
        ("Positive payout mix TV (max)", "positive_payout_mix_tv_max", ".4f"),
        ("Observed lag phi (max abs)", "lag_correlation_max_abs", ".4f"),
        ("Machine win-rate SD", "machine_win_rate_sd_abs", ".4f"),
        ("Mean cutoff (ticks)", "cutoff_mean_abs_ticks", ".2f"),
        ("Short-history fraction", "short_sampling_fraction_abs", ".4f"),
    ):
        k = key
        if key == "machine_win_rate_sd_abs":
            a, b, d = real["machine_win_rate_spread"]["sd"], generated["machine_win_rate_spread"]["sd"], metric[k]
            tol = TOLERANCES[k]
        elif key == "cutoff_mean_abs_ticks":
            a, b, d, tol = real["cutoffs"]["mean"], generated["cutoffs"]["mean"], metric[k], TOLERANCES[k]
        elif key == "short_sampling_fraction_abs":
            a, b, d, tol = real["history_sampling"]["short_fraction"], generated["history_sampling"]["short_fraction"], metric[k], TOLERANCES[k]
        else:
            a, b, d, tol = "—", "—", metric[k], TOLERANCES[k]
        def f(value):
            return value if isinstance(value, str) else ("n/a" if value is None else format(value, fmt))
        lines.append(f"| {label} | {f(a)} | {f(b)} | {f(d)} | ≤ {tol} {'✓' if comparison['within_descriptive_bands'][key] else '✗'} |")
    lines.append(f"| Sampling distributions exact | — | — | {'yes' if comparison['metrics']['sampling_pattern_exact'] else 'no'} | {'✓' if comparison['within_descriptive_bands']['sampling_pattern_exact'] else '✗'} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", default="public", help="public dataset root or its train directory")
    parser.add_argument("--generated", help="generated dataset root or train directory to compare")
    parser.add_argument("--out", help="optional JSON output path")
    parser.add_argument("--calibrate", action="store_true", help="include train-derived main_like overrides")
    args = parser.parse_args(argv)
    real = summarize_dataset(args.real)
    result = {"real_train": real}
    if args.calibrate:
        result["main_like_overrides"] = calibrate_main_like(args.real)
    if args.generated:
        generated = summarize_dataset(args.generated)
        result["generated_train"] = generated
        result["comparison"] = compare_statistics(real, generated)
        print(_markdown_comparison(real, generated, result["comparison"]))
    else:
        print(f"Real train: {real['counts']['queries']} queries, {real['counts']['episodes']} episodes; "
              f"short-history fraction {real['history_sampling']['short_fraction']:.3f}; "
              f"mean cutoff {real['cutoffs']['mean']:.2f}.")
    if args.calibrate:
        print("Train-derived main_like base_win_rates and payout_mixes included in JSON.")
    if args.out:
        path = Path(args.out).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
