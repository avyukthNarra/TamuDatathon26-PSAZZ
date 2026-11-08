"""Reproducible, data-only EDA for the House Edge validation set."""

from collections import Counter, defaultdict
import csv
import itertools
import math
from pathlib import Path

import fresh_eval


ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "mainFiles" / "public"
PAYOUTS = (0, 1, 2, 5, 20)
WIN = {0: 0, 1: 0, 2: 0, 5: 1, 20: 1}


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def mi(xs, ys):
    n = len(xs)
    joint = Counter(zip(xs, ys))
    px, py = Counter(xs), Counter(ys)
    return sum(
        count / n * math.log((count * n) / (px[x] * py[y]))
        for (x, y), count in joint.items()
    )


def corr(xs, ys):
    if len(xs) < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else 0.0


def load():
    q = rows(PUBLIC / "validation/queries.csv")
    h = rows(PUBLIC / "validation/history.csv")
    y = rows(PUBLIC / "validation/labels.csv")
    t = rows(PUBLIC / "validation/targets.csv")
    c = rows(PUBLIC / "validation/conditions.csv")
    tq = rows(PUBLIC / "train/queries.csv")
    th = rows(PUBLIC / "train/history.csv")
    ty = rows(PUBLIC / "train/labels.csv")
    return q, h, y, t, c, tq, th, ty


def base_rates(q, t):
    qmap = {r["query_id"]: r for r in q}
    by_group = defaultdict(list)
    for target in t:
        row = qmap[target["query_id"]]
        p = [float(target[f"p_{x}"]) for x in PAYOUTS]
        by_group[(row["machine_type"], row["signal"])].append(p)
    result = {}
    for key, values in sorted(by_group.items()):
        result["%s/%s" % key] = {
            "n": len(values),
            "p_win": mean(1 - p[0] for p in values),
            "p0": mean(p[0] for p in values),
            "positive_size": [mean(p[i] / (1 - p[0]) for p in values) for i in range(1, 5)],
        }
    return result


def lag_effects(q, h, y):
    qmap = {r["query_id"]: r for r in q}
    histories = defaultdict(dict)
    for row in h:
        histories[(row["episode_id"], row["machine_id"])][int(row["tick"])] = int(row["payout"])
    result = {}
    for lag_key in list(range(1, 9)) + list(itertools.combinations(range(1, 9), 2)):
        effects = []
        for query_id, query in qmap.items():
            key = (query["episode_id"], query["machine_id"])
            lags = (lag_key,) if isinstance(lag_key, int) else lag_key
            for tick, payout in histories[key].items():
                if tick >= int(query["cutoff"]) or any(tick - lag not in histories[key] for lag in lags):
                    continue
                feature = tuple(histories[key][tick - lag] for lag in lags)
                if len(feature) == 1:
                    feature = feature[0]
                effects.append((query["machine_type"], key, feature, WIN[payout]))
        by_machine = defaultdict(list)
        for machine_type, key, feature, outcome in effects:
            by_machine[(machine_type, key)].append((feature, outcome))
        machine_stats = []
        signs = defaultdict(list)
        for (machine_type, key), values in by_machine.items():
            if len({x for x, _ in values}) < 2:
                continue
            features, outcomes = zip(*values)
            rates = {value: mean(y for x, y in values if x == value) for value in set(features)}
            difference = max(rates.values()) - min(rates.values())
            # The sign is the direction of win-rate change from the lowest to
            # highest payout category, a deterministic summary for mixed levels.
            ordered = sorted(rates)
            slope = corr([float(i) for i in range(len(ordered))],
                         [rates[x] for x in ordered])
            sign = 1 if slope > 1e-12 else -1 if slope < -1e-12 else 0
            machine_stats.append((machine_type, mi(features, outcomes), difference))
            signs[machine_type].append(sign)
        conflicts = {}
        for machine_type, values in signs.items():
            nonzero = [v for v in values if v]
            conflicts[machine_type] = (
                len(nonzero) > 1 and len(set(nonzero)) > 1
            )
        result[str(lag_key)] = {
            "machines": len(machine_stats),
            "mean_mi": mean([x[1] for x in machine_stats]),
            "mean_win_rate_difference": mean([x[2] for x in machine_stats]),
            "sign_conflict_by_type": conflicts,
            "sign_conflict_any_type": any(conflicts.values()),
        }
    return result


def temporal_and_missing(q, h, y, c):
    qmap = {r["query_id"]: r for r in q}
    labels = {r["query_id"]: int(r["outcome"]) > 0 for r in y}
    condition = {r["query_id"]: r["condition"] for r in c}
    histories = defaultdict(dict)
    for row in h:
        histories[(row["episode_id"], row["machine_id"])][int(row["tick"])] = int(row["payout"])
    by_condition = defaultdict(lambda: {"first": [], "second": [], "changes": [], "missing": []})
    episode_series = defaultdict(lambda: defaultdict(dict))
    for query_id, query in qmap.items():
        key = (query["episode_id"], query["machine_id"])
        observed = histories[key]
        ticks = sorted(observed)
        midpoint = len(ticks) // 2
        first, second = ticks[:midpoint], ticks[midpoint:]
        bucket = by_condition[condition[query_id]]
        bucket["first"].append(mean([WIN[observed[t]] for t in first]))
        bucket["second"].append(mean([WIN[observed[t]] for t in second]))
        bucket["missing"].append(
            (int(query["cutoff"]) - len(observed), len(observed))
        )
        changes = 0
        for left, right in zip(ticks, ticks[1:]):
            if right == left + 1 and WIN[observed[left]] != WIN[observed[right]]:
                changes += 1
        bucket["changes"].append(changes)
        for tick in ticks:
            episode_series[query["episode_id"]][key][tick] = WIN[observed[tick]]
    correlations = {}
    for episode, machines in episode_series.items():
        values = []
        for a, b in itertools.combinations(machines, 2):
            common = sorted(set(machines[a]) & set(machines[b]))
            if len(common) >= 2:
                values.append(corr([machines[a][t] for t in common],
                                   [machines[b][t] for t in common]))
        cond = condition[next(k for k, v in qmap.items() if v["episode_id"] == episode)]
        correlations.setdefault(cond, []).extend(values)
    return {
        condition: {
            "first_win_rate": mean(value["first"]),
            "second_win_rate": mean(value["second"]),
            "change": mean(
                b - a for a, b in zip(value["first"], value["second"])
            ),
            "change_point_count_mean": mean(value["changes"]),
            "missing_rows_mean": mean(x[0] for x in value["missing"]),
            "observed_rows_mean": mean(x[1] for x in value["missing"]),
            "cross_machine_corr_mean": mean(correlations[condition]),
        }
        for condition, value in sorted(by_condition.items())
    }


def ceilings(q, t, tq, ty):
    qmap = {r["query_id"]: r for r in q}
    train_qmap = {r["query_id"]: r for r in tq}
    train_labels = {r["query_id"]: int(r["outcome"]) for r in ty}
    groups = defaultdict(list)
    for query_id, outcome in train_labels.items():
        row = train_qmap[query_id]
        groups[(row["machine_type"], row["signal"])].append(outcome)
    global_outcomes = list(train_labels.values())
    predictions_win = {}
    predictions_size = {}
    for target in t:
        row = qmap[target["query_id"]]
        true = [float(target[f"p_{x}"]) for x in PAYOUTS]
        outcomes = groups[(row["machine_type"], row["signal"])] or global_outcomes
        positive = [x for x in outcomes if x > 0]
        size = [positive.count(x) / len(positive) if positive else 0 for x in (1, 2, 5, 20)]
        pwin = 1 - true[0]
        predictions_win[target["query_id"]] = tuple([1 - pwin] + [pwin * x for x in size])
        empirical_win = mean(x > 0 for x in outcomes)
        true_size = [x / pwin for x in true[1:]] if pwin else [0] * 4
        predictions_size[target["query_id"]] = tuple(
            [1 - empirical_win] + [empirical_win * x for x in true_size]
        )
    return {
        "true_win_empirical_size": fresh_eval.score(predictions_win),
        "empirical_win_true_size": fresh_eval.score(predictions_size),
    }


def main():
    q, h, y, t, c, tq, th, ty = load()
    result = {
        "base_rates_validation_targets": base_rates(q, t),
        "lag_effects_validation_labels": lag_effects(q, h, y),
        "temporal_missingness_validation": temporal_and_missing(q, h, y, c),
        "ceilings": ceilings(q, t, tq, ty),
    }
    # Distinct target positive-size vectors, rounded only for cluster reporting.
    vectors = Counter()
    for target in t:
        p = [float(target[f"p_{x}"]) for x in PAYOUTS]
        vectors[tuple(round(p[i] / (1 - p[0]), 3) for i in range(1, 5))] += 1
    result["positive_size_clusters"] = {
        "distinct_rounded_3dp": len(vectors),
        "cluster_sizes": {",".join(map(str, key)): value for key, value in vectors.items()},
    }
    out = ROOT / "eda_results.json"
    out.write_text(__import__("json").dumps(result, indent=2, sort_keys=True) + "\n")
    report = ["# Fresh-start EDA", "", "All measurements below were produced by `eda_analysis.py`; the complete machine-level and pair-level data is in `eda_results.json`.", ""]
    report += ["## Base rates and target size distributions", "",
               "| machine type / signal | n | p0 | p(win) | positive payout shares (1,2,5,20) |",
               "|---|---:|---:|---:|---|"]
    for key, value in result["base_rates_validation_targets"].items():
        shares = ", ".join(f"{x:.4f}" for x in value["positive_size"])
        report.append(f"| {key} | {value['n']} | {value['p0']:.4f} | {value['p_win']:.4f} | {shares} |")
    report += ["", f"Positive-size vectors have {result['positive_size_clusters']['distinct_rounded_3dp']} distinct clusters after rounding each conditional share to 3 decimals; cluster sizes are in `eda_results.json`.", ""]
    report += ["## Lag effects", "",
               "Each row uses every available within-machine history transition before the validation cutoff. MI is between the categorical lag (or lag-pair) payout and binary win; win-rate difference is max conditional win rate minus min conditional win rate. A sign conflict means both positive and negative ordinal trends occur among machines of that type.",
               "", "| lag(s) | machines | MI | win-rate difference | sign conflict by type |", "|---|---:|---:|---:|---|"]
    for key, value in result["lag_effects_validation_labels"].items():
        conflicts = ", ".join(sorted(k for k, v in value["sign_conflict_by_type"].items() if v)) or "none"
        report.append(f"| {key} | {value['machines']} | {value['mean_mi']:.6f} | {value['mean_win_rate_difference']:.4f} | {conflicts} |")
    report += ["", "## Non-stationarity, correlation, and missingness", "",
               "Change points count adjacent observed ticks where binary win/loss changes; it is descriptive rather than a fitted detector. Missing rows are `cutoff - observed history rows`.", "",
               "| condition | first win rate | second win rate | change | mean change points | mean missing | mean observed | cross-machine corr |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, value in result["temporal_missingness_validation"].items():
        report.append(f"| {key} | {value['first_win_rate']:.4f} | {value['second_win_rate']:.4f} | {value['change']:.4f} | {value['change_point_count_mean']:.2f} | {value['missing_rows_mean']:.2f} | {value['observed_rows_mean']:.2f} | {value['cross_machine_corr_mean']:.4f} |")
    report += ["", "## Validation score ceilings", "",
               "The first ceiling gives each target's true total win probability but uses train empirical positive-payout shares. The second gives the train empirical win rate but uses each target's true positive-payout shares.", "",
               "| construction | points | R | condition ratios (stable, short_history, drift, shared_change) |", "|---|---:|---:|---|"]
    for key, value in result["ceilings"].items():
        ratios = ", ".join(f"{value['per_condition_ratio'][c]:.4f}" for c in ("stable", "short_history", "drift", "shared_change"))
        report.append(f"| {key} | {value['points']:.4f} | {value['R']:.4f} | {ratios} |")
    (ROOT / "eda_report.md").write_text("\n".join(report) + "\n")
    print(out)


if __name__ == "__main__":
    main()
