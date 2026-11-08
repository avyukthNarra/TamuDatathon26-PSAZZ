"""Adaptability kits from the g7 / g1 kernel, plus a main-round size rule.

Each machine draws one lag pair and a fair direction. Win probability is
logit-normal by type and signal. Drift is a linear swap of the two levels,
anchored at the cutoff. Shared change is one episode-wide direction flip.
Short history hides ticks; the target still uses the true past.

Size rule (g5): given a win at tick t, the atom is
``s XOR 1{payout(t-L) >= threshold}``. On main, L is the {1, 2, 4} member of
the win pair. ``size_lag="min"`` uses the smaller pair member instead.
``size_rule=False`` keeps a static atom (a fair coin on the two atoms).
The payout draw conditions on the win first, then samples the atom. An earlier
version placed the atom bins on the complementary interval and collapsed
almost every win to 1x.

Writes the public layout run_pair_mixer.py accepts, plus validation/oracle.csv
(the true probabilities; contest score 40 by construction) and
oracle_score.json (oracle and the pooled starter).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# slot_forecasting comes from the mainFiles/ folder this script sits in. Set
# SLOT_FORECASTING_ROOT to point at another checkout.
MAIN = Path(os.environ.get("SLOT_FORECASTING_ROOT", HERE.parent))
sys.path.insert(0, str(MAIN))

from slot_forecasting.baseline import FrequencyForecaster  # noqa: E402
from slot_forecasting.contest import score_contest_files  # noqa: E402
from slot_forecasting.io import (  # noqa: E402
    load_inputs, read_labels, read_targets, write_rows, write_submission,
)
from slot_forecasting.rounds import round_metadata  # noqa: E402
from slot_forecasting.schema import (  # noqa: E402
    CONDITIONS, HISTORY_COLUMNS, LABEL_COLUMNS, PROBABILITY_COLUMNS,
    QUERY_COLUMNS, TARGET_COLUMNS,
)

POSITIVE = (1, 2, 5, 20)
# 12 per condition, not 24. A full candidate build (fit, zip, fresh-process
# rerun) on 10 kits x 3 candidates does not finish at main's 24 inside the box.
COUNTS = {"train": 12, "validation": 12, "test": 12}
TYPES = ("steady", "balanced", "jackpot")

ATOMS = {
    "steady": (
        (0.753989, 0.196808, 0.048218, 0.000984),
        (0.455122, 0.435902, 0.106796, 0.002180),
    ),
    "balanced": (
        (0.499853, 0.347928, 0.130473, 0.021746),
        (0.214066, 0.546737, 0.205026, 0.034171),
    ),
    "jackpot": (
        (0.335146, 0.244946, 0.314931, 0.104977),
        (0.120787, 0.323921, 0.416469, 0.138823),
    ),
}

LEVELS = {
    "g1": {
        "steady": dict(a=-1.627, b=0.318, c=4.638, sigma=0.52, rho=0.38),
        "balanced": dict(a=-2.964, b=0.598, c=4.850, sigma=0.56, rho=0.31),
        "jackpot": dict(a=-4.196, b=0.428, c=4.190, sigma=0.63, rho=0.07),
    },
    # Signal 0: steady dead ~0.25, jackpot live ~0.35. Slope 1.15 is about
    # three times g1's steady slope, so signal ±1 moves the logit by 1.15.
    "strong": {
        "steady": dict(a=-1.099, b=1.15, c=4.20, sigma=0.35, rho=0.30),
        "balanced": dict(a=-2.20, b=1.15, c=4.30, sigma=0.35, rho=0.30),
        "jackpot": dict(a=-3.60, b=1.15, c=2.981, sigma=0.28, rho=0.10),
    },
}

MAIN_PAIRS = ((1, 3), (4, 7), (2, 5))
MAIN_TIMING = dict(drift_anchor=96, drift_duration=62, flip_frac=0.663, flip_sd=11.0)


def _kit(**overrides):
    config = dict(
        pairs=MAIN_PAIRS,
        weights=None,
        levels="g1",
        size_rule=True,
        size_lag="member",
        size_threshold=2,
        mix_frac=0.0,
        **MAIN_TIMING,
    )
    config.update(overrides)
    return config


KITS = {
    "K1": _kit(
        seed=8101,
        note="main pairs and the main size rule (L is the {1,2,4} member); main timing",
    ),
    "K2": _kit(
        pairs=((1, 6), (3, 8), (2, 4)),
        size_lag="min",
        seed=8102,
        note="disjoint pairs (1,6),(3,8),(2,4); size rule L = smallest member",
    ),
    "K3": _kit(
        pairs=((1, 6), (2, 4), (3, 8), (5, 7)),
        weights=(0.50, 0.27, 0.15, 0.08),
        size_lag="min",
        seed=8103,
        note="four pairs at weights 0.50/0.27/0.15/0.08; size rule L = min",
    ),
    "K4": _kit(
        pairs=((1, 6), (1, 8), (3, 6)),
        size_lag="min",
        seed=8104,
        note="pairs share a lag: (1,6),(1,8),(3,6); size rule L = min",
    ),
    "K5": _kit(
        pairs=((5, 8), (6, 7), (3, 8)),
        size_lag="min",
        seed=8105,
        note="long lags only (5,8),(6,7),(3,8); size rule L = min, so L in {5,6,3}",
    ),
    "K6": _kit(
        size_rule=False,
        seed=8106,
        note="main pairs, static size atom (no lag rule)",
    ),
    "K7": _kit(
        size_threshold=5,
        seed=8107,
        note="main pairs and size rule, but the bit is payout >= 5",
    ),
    "K8": _kit(
        drift_anchor=28,
        drift_duration=18,
        flip_frac=0.91,
        flip_sd=4.0,
        seed=8108,
        note="late shared flip (~0.09 of the episode, about 20 ticks out) and a fast drift (18-tick ramp, finished ~10 ticks out)",
    ),
    "K9": _kit(
        levels="strong",
        seed=8109,
        note="shifted levels (jackpot live ~0.35, steady dead ~0.25) and signal slope 1.15",
    ),
    "K10": _kit(
        mix_frac=0.30,
        seed=8110,
        note="30% of machines use a 3-lag XOR or an AND rule; the rest stay 2-lag XOR/XNOR",
    ),
}


def _sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _normalize(values):
    total = math.fsum(values)
    return tuple(v / total for v in values)


def _atoms():
    return {k: tuple(_normalize(a) for a in v) for k, v in ATOMS.items()}


def _size_lag(pair, how: str) -> int:
    if how == "min":
        return min(pair)
    if how == "member":
        hits = [lag for lag in pair if lag in (1, 2, 4)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            return min(hits)
        return min(pair)
    raise ValueError(how)


def _sample_levels(rng: random.Random, spec: dict, signal: int):
    z1, z2 = rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)
    sigma, rho = spec["sigma"], spec["rho"]
    e1 = sigma * z1
    e2 = sigma * (rho * z1 + math.sqrt(max(0.0, 1.0 - rho * rho)) * z2)
    center = spec["a"] + spec["b"] * signal
    p_dead = min(1.0 - 1e-4, max(1e-4, _sigmoid(center + e1)))
    p_live = min(1.0 - 1e-4, max(1e-4, _sigmoid(center + spec["c"] + e2)))
    return p_dead, p_live


def _early_live(t, series, machine):
    """True when the pre-regime rule says live. None before the lags exist."""
    rule = machine["rule"]
    a, b = machine["pair"]
    if rule == "xor3":
        c = machine["third"]
        if t < max(a, b, c):
            return None
        even = ((series[t - a] > 0) + (series[t - b] > 0) + (series[t - c] > 0)) % 2 == 0
        bit = even
    else:
        if t < b:
            return None
        left, right = series[t - a] > 0, series[t - b] > 0
        bit = (left and right) if rule == "and" else (left == right)
    if machine["direction"] == 1:
        return bit
    return not bit


def _p_win(t, series, machine, condition, cutoff, shared, anchor, duration):
    early = _early_live(t, series, machine)
    p_dead, p_live = machine["p_dead"], machine["p_live"]
    if early is None:
        return 0.5 * (p_dead + p_live)
    if condition == "drift":
        u = min(1.0, max(0.0, (t - (cutoff - anchor)) / float(duration)))
        if early:
            return (1.0 - u) * p_live + u * p_dead
        return (1.0 - u) * p_dead + u * p_live
    if condition == "shared_change" and t >= shared:
        return p_dead if early else p_live
    return p_live if early else p_dead


def _atom_index(series, t, machine, use_rule, threshold):
    polarity = machine["size_mode"]
    if not use_rule:
        return polarity
    lag = machine["size_lag"]
    bit = 0 if t < lag else int(series[t - lag] >= threshold)
    return polarity ^ bit


def _sample_payout(rng, p_win, atom) -> int:
    """Draw one payout. Losses are ``point >= p_win``; a win's class is the atom.

    The atom bins have to live on ``[0, p_win)``. Starting them at ``1 - p_win``
    puts 2x/5x/20x past the early return, so a moderate ``p_win`` that won was
    recorded as 1x. Targets still used the analytic atom, so the self-check
    passed while the history the size model trains on did not.
    """
    point = rng.random()
    if p_win <= 0.0 or point >= p_win:
        return 0
    u = point / p_win
    cumulative = 0.0
    for payout, prob in zip(POSITIVE, atom):
        cumulative += prob
        if u <= cumulative:
            return payout
    return 20


def _vector(p_win, atom):
    raw = (1.0 - p_win,) + tuple(p_win * a for a in atom)
    return _normalize(raw)


def _cell_gap(series, machine, condition, cutoff, shared, anchor, duration):
    cells = {True: [0, 0], False: [0, 0]}
    for t in range(len(series)):
        early = _early_live(t, series, machine)
        if early is None:
            continue
        if condition == "drift":
            u = min(1.0, max(0.0, (t - (cutoff - anchor)) / float(duration)))
            if u > 0.5:
                early = not early
        elif condition == "shared_change" and t >= shared:
            early = not early
        cells[early][0] += int(series[t] > 0)
        cells[early][1] += 1
    if cells[True][1] < 15 or cells[False][1] < 15:
        return None
    return cells[True][0] / cells[True][1] - cells[False][0] / cells[False][1]


def simulate_episode(rng, episode_id, condition, config, atoms):
    cutoff = rng.randint(192, 256)
    levels = LEVELS[config["levels"]]
    anchor, duration = config["drift_anchor"], config["drift_duration"]
    threshold = int(config["size_threshold"])
    use_rule = bool(config["size_rule"])
    if condition == "shared_change":
        shared = int(round(rng.gauss(config["flip_frac"] * cutoff, config["flip_sd"])))
        shared = max(8, min(cutoff - 1, shared))
    else:
        shared = None
    machines = []
    series_by_machine = []
    gaps = []
    rules = {"xor": 0, "and": 0, "xor3": 0}
    for i in range(8):
        machine_type = rng.choice(TYPES)
        signal = rng.choice((-1, 0, 1))
        weights = config.get("weights")
        pair = tuple(sorted(
            rng.choices(config["pairs"], weights=weights, k=1)[0] if weights
            else rng.choice(config["pairs"])
        ))
        direction = rng.choice((0, 1))
        p_dead, p_live = _sample_levels(rng, levels[machine_type], signal)
        rule, third = "xor", None
        if config["mix_frac"] and rng.random() < config["mix_frac"]:
            rule = rng.choice(("and", "xor3"))
            if rule == "xor3":
                third = rng.choice([lag for lag in range(1, 9) if lag not in pair])
        rules[rule] += 1
        machine = {
            "machine_id": f"{episode_id}-m{i:02d}",
            "query_id": f"{episode_id}-m{i:02d}-next",
            "machine_type": machine_type,
            "signal": signal,
            "pair": pair,
            "direction": direction,
            "p_dead": p_dead,
            "p_live": p_live,
            "size_mode": rng.choice((0, 1)),
            "size_lag": _size_lag(pair, config["size_lag"]),
            "rule": rule,
            "third": third,
        }
        spec = machine
        series = []
        for t in range(cutoff):
            p_win = _p_win(t, series, spec, condition, cutoff, shared, anchor, duration)
            atom = atoms[machine_type][_atom_index(series, t, spec, use_rule, threshold)]
            series.append(_sample_payout(rng, p_win, atom))
        gap = _cell_gap(series, spec, condition, cutoff, shared, anchor, duration)
        if gap is not None:
            gaps.append(gap)
        p_star = _p_win(cutoff, series, spec, condition, cutoff, shared, anchor, duration)
        atom_at = _atom_index(series, cutoff, spec, use_rule, threshold)
        if use_rule:
            expected = spec["size_mode"] ^ int(series[cutoff - spec["size_lag"]] >= threshold)
            if atom_at != expected:
                raise RuntimeError(f"{episode_id}: size bit disagrees with the lag rule")
        atom = atoms[machine_type][atom_at]
        machine["atom_at_cutoff"] = atom_at
        machine["p_at_cutoff"] = p_star
        machine["outcome"] = _sample_payout(rng, p_star, atom)
        machine["probabilities"] = _vector(p_star, atom)
        machines.append(machine)
        series_by_machine.append(series)

    history_rows = []
    for i, machine in enumerate(machines):
        ticks = list(range(cutoff))
        if condition == "short_history":
            recent = list(range(cutoff - 40, cutoff))
            older = list(range(0, cutoff - 40))
            ticks = sorted(set(recent + rng.sample(older, 24)))
            if len(ticks) != 64:
                raise RuntimeError(f"short history length {len(ticks)} != 64")
        machine["n_observed"] = len(ticks)
        for t in ticks:
            history_rows.append({
                "episode_id": episode_id,
                "machine_id": machine["machine_id"],
                "tick": str(t),
                "payout": str(series_by_machine[i][t]),
            })
    queries = [{
        "query_id": m["query_id"],
        "episode_id": episode_id,
        "machine_id": m["machine_id"],
        "machine_type": m["machine_type"],
        "signal": str(m["signal"]),
        "cutoff": str(cutoff),
    } for m in machines]
    kept = (
        "machine_id", "query_id", "machine_type", "signal", "pair", "direction",
        "p_dead", "p_live", "size_mode", "size_lag", "rule", "third",
        "atom_at_cutoff", "p_at_cutoff", "outcome", "n_observed",
    )
    return {
        "episode_id": episode_id,
        "condition": condition,
        "cutoff": cutoff,
        "shared_tick": shared,
        "ticks_before_flip": (None if shared is None else cutoff - shared),
        "u_at_cutoff": (
            min(1.0, max(0.0, anchor / float(duration))) if condition == "drift" else None
        ),
        "rules": rules,
        "machines": [{**{k: m[k] for k in kept}, "pair": list(m["pair"])} for m in machines],
        "queries": queries,
        "history": history_rows,
        "labels": [{"query_id": m["query_id"], "outcome": str(m["outcome"])} for m in machines],
        "targets": [{
            "query_id": m["query_id"],
            "episode_id": episode_id,
            **{c: format(p, ".17g") for c, p in zip(PROBABILITY_COLUMNS, m["probabilities"])},
            "outcome": str(m["outcome"]),
        } for m in machines],
        "oracle": {m["query_id"]: m["probabilities"] for m in machines},
        "gaps": gaps,
    }


def _write_split(public, split, episodes, with_targets):
    folder = public / split
    write_rows(folder / "queries.csv", QUERY_COLUMNS, [q for e in episodes for q in e["queries"]])
    write_rows(folder / "history.csv", HISTORY_COLUMNS, [h for e in episodes for h in e["history"]])
    if split != "test":
        write_rows(folder / "labels.csv", LABEL_COLUMNS, [r for e in episodes for r in e["labels"]])
        write_rows(folder / "conditions.csv", ("query_id", "condition"), [
            {"query_id": q["query_id"], "condition": e["condition"]}
            for e in episodes for q in e["queries"]
        ])
    if with_targets:
        write_rows(folder / "targets.csv", TARGET_COLUMNS, [t for e in episodes for t in e["targets"]])


def _reference(public):
    train_q, train_h = load_inputs(public / "train")
    labels = read_labels(public / "train" / "labels.csv", train_q)
    model = FrequencyForecaster(method="pooled", prior_strength=20.0).fit(train_q, train_h, labels)
    val_q, val_h = load_inputs(public / "validation")
    write_submission(public / "validation" / "reference.csv", model.predict(val_q, val_h))


def _score_pair(public, oracle_rows):
    write_submission(public / "validation" / "oracle.csv", oracle_rows)
    meta = round_metadata(public)
    val = public / "validation"
    kwargs = dict(max_points=meta["max_points"], version="evidence-contest-v1")
    common = (
        val / "targets.csv", val / "queries.csv", val / "conditions.csv", val / "reference.csv",
    )
    oracle = score_contest_files(val / "oracle.csv", *common, **kwargs)
    starter = score_contest_files(val / "reference.csv", *common, **kwargs)
    return {
        "oracle_points": oracle["technical_points"],
        "oracle_error": oracle["mean_loss_ratio"],
        "starter_points": starter["technical_points"],
        "starter_loss_ratio": starter["mean_loss_ratio"],
        "max_points": meta["max_points"],
    }


def _metadata(public, name, config, counts):
    payload = {
        "conditions": list(CONDITIONS),
        "counts_per_condition": counts,
        "history": "Synthetic g1 kernel. Size is either a static atom or s XOR 1{payout(t-L) >= threshold}.",
        "machines_per_episode": 8,
        "max_points": 40,
        "mode": "event",
        "reference": "Pooled, strength=20, fitted separately on this round's training observations.",
        "round": "adaptation",
        "season_id": hashlib.sha256(f"a1-bench-{name}".encode()).hexdigest()[:24],
        "suite_version": "experimentation-suite-v3",
        "target": "Next-spin probabilities for 0x, 1x, 2x, 5x, 20x.",
        "test_policy": "Test targets stay in latents.json only; the public test split has no targets.",
        "generator": {
            "name": name,
            "pairs": [list(p) for p in config["pairs"]],
            "weights": list(config["weights"]) if config.get("weights") else None,
            "levels": config["levels"],
            "size_rule": config["size_rule"],
            "size_lag": config["size_lag"],
            "size_threshold": config["size_threshold"],
            "mix_frac": config["mix_frac"],
            "drift_anchor": config["drift_anchor"],
            "drift_duration": config["drift_duration"],
            "flip_frac": config["flip_frac"],
            "flip_sd": config["flip_sd"],
            "note": config["note"],
            "seed": config["seed"],
            "episodes_per_condition": dict(counts),
        },
    }
    (public / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def generate_config(name, config, out_root: Path, counts=COUNTS):
    atoms = _atoms()
    root = out_root / name
    public = root / "public"
    splits = {}
    gap_sum = [0.0, 0]
    rule_sum = {"xor": 0, "and": 0, "xor3": 0}
    flip_gaps = []
    oracle = {}
    for index, split in enumerate(("train", "validation", "test")):
        rng = random.Random(config["seed"] + index * 10007)
        episodes = []
        n = 0
        for condition in CONDITIONS:
            for _ in range(counts[split]):
                episode_id = f"{name}-{split[:3]}-e{n:02d}"
                episode = simulate_episode(rng, episode_id, condition, config, atoms)
                gap_sum[0] += math.fsum(episode["gaps"])
                gap_sum[1] += len(episode["gaps"])
                for key, count in episode["rules"].items():
                    rule_sum[key] += count
                if episode["ticks_before_flip"] is not None:
                    flip_gaps.append(episode["ticks_before_flip"])
                if split == "validation":
                    oracle.update(episode["oracle"])
                episodes.append(episode)
                n += 1
        _write_split(public, split, episodes, with_targets=(split == "validation"))
        splits[split] = [{
            k: episode[k] for k in (
                "episode_id", "condition", "cutoff", "shared_tick", "ticks_before_flip",
                "u_at_cutoff", "machines",
            )
        } for episode in episodes]
    _metadata(public, name, config, dict(counts))
    _reference(public)
    scores = _score_pair(public, oracle)
    if scores["oracle_points"] < 39.99:
        raise RuntimeError(f"{name}: oracle scored {scores['oracle_points']}, expected 40")
    dumped = {**config, "pairs": [list(p) for p in config["pairs"]]}
    if dumped.get("weights") is not None:
        dumped["weights"] = list(dumped["weights"])
    latents = {"config": dumped, "counts": dict(counts), "splits": splits, "scores": scores}
    (root / "latents.json").write_text(json.dumps(latents) + "\n", encoding="utf-8")
    (root / "oracle_score.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    load_inputs(public / "train")
    load_inputs(public / "validation")
    load_inputs(public / "test")
    targets = read_targets(public / "validation" / "targets.csv")
    if gap_sum[1] == 0:
        raise RuntimeError(f"{name}: no machine had both parity cells")
    mean_gap = gap_sum[0] / gap_sum[1]
    flip_mean = math.fsum(flip_gaps) / len(flip_gaps) if flip_gaps else float("nan")
    print(
        f"{name}: episodes {len(splits['train'])}/{len(splits['validation'])}/{len(splits['test'])} "
        f"val targets {len(targets)} oracle {scores['oracle_points']:.4f} "
        f"starter {scores['starter_points']:.4f} cell gap {mean_gap:.3f} "
        f"on {gap_sum[1]} machines flip-before-cutoff {flip_mean:.1f} "
        f"rules xor/and/xor3 {rule_sum['xor']}/{rule_sum['and']}/{rule_sum['xor3']}"
    )
    if mean_gap < 0.20:
        raise RuntimeError(f"{name}: cell gap {mean_gap:.3f} is too small; the rule is not in the series")
    return scores


def _check_sampler():
    """p_win=0.3 used to emit 1x for every win. Steady atom 0 must not."""
    rng = random.Random(0)
    atom = _atoms()["steady"][0]
    counts = {payout: 0 for payout in (0,) + POSITIVE}
    n = 20000
    p_win = 0.3
    for _ in range(n):
        counts[_sample_payout(rng, p_win, atom)] += 1
    wins = n - counts[0]
    frac1 = counts[1] / wins
    if abs(counts[0] / n - (1.0 - p_win)) > 0.02 or abs(frac1 - atom[0]) > 0.04:
        raise RuntimeError(
            f"sampler check failed: P(0)={counts[0] / n:.3f} P(1|win)={frac1:.3f} atom={atom[0]:.3f}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write the simulated round-two kits.")
    parser.add_argument("--out", type=Path, default=HERE / "kits",
                        help="output directory (default: kits/ next to this script)")
    parser.add_argument("--only", default="",
                        help="comma-separated kit names, e.g. K1,K6 (default: all)")
    parser.add_argument("--episodes", type=int, default=COUNTS["train"],
                        help="episodes per condition per split (default: %(default)s)")
    args = parser.parse_args(argv)
    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(KITS)
    unknown = [n for n in names if n not in KITS]
    if unknown:
        parser.error(f"unknown kit(s): {', '.join(unknown)}")
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    counts = {split: args.episodes for split in COUNTS}
    _check_sampler()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in names:
        summary[name] = generate_config(name, KITS[name], out, counts)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(names)} kits under {out} ({args.episodes} episodes per condition per split)")


if __name__ == "__main__":
    main()
