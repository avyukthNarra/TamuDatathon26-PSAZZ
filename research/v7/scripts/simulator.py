"""Small simulator of the g1 generative model with movable lag pairs.

Follows research_pf/g1-data/REPORT.md §9: each machine draws one lag pair, a
fair direction (agree = early-live, or not), logit-normal dead/live levels,
and a fair coin on the two main-round payout-size atoms. Drift is a linear
swap of those two levels, anchored to the cutoff. Shared change is one
episode-wide direction flip. Short history hides ticks; the target still uses
the true past.

Writes the public-data layout run_pair_mixer.py accepts:

    <out>/<config>/public/{train,validation,test}/{queries,history}.csv
    train and validation also get labels.csv and conditions.csv
    validation also gets targets.csv and reference.csv
    public/metadata.json and <config>/latents.json

Reference forecasts are the pooled starter (strength 20) fit on that config's
train split, matching the real release note.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path

MAIN = Path("/Users/avynarra/Documents/Datathon/worktrees/pranav-5629/mainFiles")
sys.path.insert(0, str(MAIN))

from slot_forecasting.baseline import FrequencyForecaster  # noqa: E402
from slot_forecasting.io import (  # noqa: E402
    load_inputs, read_labels, read_targets, write_rows, write_submission,
)
from slot_forecasting.rounds import round_metadata  # noqa: E402
from slot_forecasting.schema import (  # noqa: E402
    CONDITIONS, HISTORY_COLUMNS, LABEL_COLUMNS, PROBABILITY_COLUMNS,
    QUERY_COLUMNS, TARGET_COLUMNS,
)

HERE = Path(__file__).resolve().parent
POSITIVE = (1, 2, 5, 20)

# g1 §6, P(payout = 1, 2, 5, 20 | win). Two atoms per type, a fair coin per machine.
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

# g1 §2 / §9. logit p_dead ~ N(a + b*signal, σ), logit p_live ~ N(a + b*signal + c, σ), corr ρ.
LEVELS = {
    "g1": {
        "steady": dict(a=-1.627, b=0.318, c=4.638, sigma=0.52, rho=0.38),
        "balanced": dict(a=-2.964, b=0.598, c=4.850, sigma=0.56, rho=0.31),
        "jackpot": dict(a=-4.196, b=0.428, c=4.190, sigma=0.63, rho=0.07),
    },
    # Jackpot live logit stays negative, so every jackpot target has p_win < 0.5.
    # That breaks win_prior_from_targets, which splits the two levels at 0.5.
    "shifted": {
        "steady": dict(a=-1.20, b=0.25, c=3.80, sigma=0.20, rho=0.30),
        "balanced": dict(a=-2.20, b=0.25, c=4.20, sigma=0.20, rho=0.30),
        "jackpot": dict(a=-3.50, b=0.25, c=2.80, sigma=0.12, rho=0.10),
    },
}

CONFIGS = {
    # Classic round-two move: three new disjoint pairs, main timing.
    "disjoint": dict(
        pairs=((1, 6), (3, 8), (2, 4)),
        levels="g1",
        drift_anchor=96,
        drift_duration=62,
        flip_frac=0.663,
        flip_sd=11.0,
        seed=701,
        note="new disjoint pairs, main drift (done ~34 ticks before cutoff) and flip (~0.66)",
    ),
    # Four pairs, one of them rare enough that choose_pairs' 0.3 share test drops it.
    # Drift ramp is centered so the early half and the late half have opposite
    # directions and a pooled history cancels; the ramp is finished by the cutoff.
    "four": dict(
        pairs=((1, 6), (2, 4), (3, 8), (5, 7)),
        weights=(0.50, 0.27, 0.15, 0.08),
        levels="shifted",
        drift_anchor=120,
        drift_duration=30,
        flip_frac=0.45,
        flip_sd=8.0,
        seed=702,
        note="four pairs with weights 0.50/0.27/0.15/0.08, shifted levels, drift cancels in a pooled history",
    ),
    # Every pair shares a lag with another. Flip and the end of drift sit close to the cutoff.
    "sharedlag": dict(
        pairs=((1, 6), (1, 8), (3, 6)),
        levels="g1",
        drift_anchor=50,
        drift_duration=35,
        flip_frac=0.90,
        flip_sd=5.0,
        seed=703,
        note="pairs share a lag; drift finishes ~15 ticks out; flip ~0.90 of the episode",
    ),
}

COUNTS = {"train": 6, "validation": 6, "test": 2}  # episodes per condition
TYPES = ("steady", "balanced", "jackpot")


def _sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _normalize(values):
    total = math.fsum(values)
    return tuple(v / total for v in values)


def _atoms():
    return {k: tuple(_normalize(a) for a in v) for k, v in ATOMS.items()}


def _sample_levels(rng: random.Random, spec: dict, signal: int):
    z1, z2 = rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)
    sigma, rho = spec["sigma"], spec["rho"]
    e1 = sigma * z1
    e2 = sigma * (rho * z1 + math.sqrt(max(0.0, 1.0 - rho * rho)) * z2)
    center = spec["a"] + spec["b"] * signal
    p_dead = min(1.0 - 1e-4, max(1e-4, _sigmoid(center + e1)))
    p_live = min(1.0 - 1e-4, max(1e-4, _sigmoid(center + spec["c"] + e2)))
    return p_dead, p_live


def _sample_payout(rng: random.Random, p_win: float, atom) -> int:
    point = rng.random()
    if point >= p_win:
        return 0
    cumulative = 1.0 - p_win
    for payout, prob in zip(POSITIVE, atom):
        cumulative += p_win * prob
        if point <= cumulative:
            return payout
    return 20


def _vector(p_win: float, atom):
    raw = (1.0 - p_win,) + tuple(p_win * a for a in atom)
    return _normalize(raw)


def p_win_at(t, series, machine, condition, cutoff, shared_tick, drift_anchor, drift_duration):
    """True P(win) at tick t given payouts series[0:t]. Matches g1 §9."""
    a, b = machine["pair"]
    p_dead, p_live = machine["p_dead"], machine["p_live"]
    if t < b:
        return 0.5 * (p_dead + p_live)
    agree = (series[t - a] > 0) == (series[t - b] > 0)
    early_live = agree if machine["direction"] == 1 else (not agree)
    if condition == "drift":
        u = min(1.0, max(0.0, (t - (cutoff - drift_anchor)) / float(drift_duration)))
        if early_live:
            return (1.0 - u) * p_live + u * p_dead
        return (1.0 - u) * p_dead + u * p_live
    if condition == "shared_change" and t >= shared_tick:
        return p_dead if early_live else p_live
    return p_live if early_live else p_dead


def _cell_gap(series, machine, condition, cutoff, shared_tick, drift_anchor, drift_duration):
    """Oriented live-minus-dead win rate on the full series (regime undone)."""
    a, b = machine["pair"]
    cells = {True: [0, 0], False: [0, 0]}
    for t in range(b, len(series)):
        agree = (series[t - a] > 0) == (series[t - b] > 0)
        early_live = agree if machine["direction"] == 1 else (not agree)
        if condition == "drift":
            u = min(1.0, max(0.0, (t - (cutoff - drift_anchor)) / float(drift_duration)))
            if u > 0.5:
                early_live = not early_live
        elif condition == "shared_change" and t >= shared_tick:
            early_live = not early_live
        cells[early_live][0] += int(series[t] > 0)
        cells[early_live][1] += 1
    if cells[True][1] < 15 or cells[False][1] < 15:
        return None
    return cells[True][0] / cells[True][1] - cells[False][0] / cells[False][1]


def simulate_episode(rng, episode_id, condition, config, atoms):
    cutoff = rng.randint(192, 256)
    levels = LEVELS[config["levels"]]
    anchor, duration = config["drift_anchor"], config["drift_duration"]
    if condition == "shared_change":
        shared = int(round(rng.gauss(config["flip_frac"] * cutoff, config["flip_sd"])))
        shared = max(8, min(cutoff - 1, shared))
    else:
        shared = None
    machines = []
    series_by_machine = []
    gaps = []
    for i in range(8):
        machine_type = rng.choice(TYPES)
        signal = rng.choice((-1, 0, 1))
        weights = config.get("weights")
        pair = tuple(rng.choices(config["pairs"], weights=weights, k=1)[0] if weights
                     else rng.choice(config["pairs"]))
        direction = rng.choice((0, 1))
        p_dead, p_live = _sample_levels(rng, levels[machine_type], signal)
        size_mode = rng.choice((0, 1))
        machine = {
            "machine_id": f"{episode_id}-m{i:02d}",
            "query_id": f"{episode_id}-m{i:02d}-next",
            "machine_type": machine_type,
            "signal": signal,
            "pair": list(pair),
            "direction": direction,
            "p_dead": p_dead,
            "p_live": p_live,
            "size_mode": size_mode,
        }
        atom = atoms[machine_type][size_mode]
        series = []
        spec = {**machine, "pair": pair}
        for t in range(cutoff):
            series.append(_sample_payout(rng, p_win_at(
                t, series, spec, condition, cutoff, shared, anchor, duration), atom))
        gap = _cell_gap(series, spec, condition, cutoff, shared, anchor, duration)
        if gap is not None:
            gaps.append(gap)
        p_star = p_win_at(cutoff, series, spec, condition, cutoff, shared, anchor, duration)
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
        "p_dead", "p_live", "size_mode", "p_at_cutoff", "outcome", "n_observed",
    )
    return {
        "episode_id": episode_id,
        "condition": condition,
        "cutoff": cutoff,
        "shared_tick": shared,
        "u_at_cutoff": (
            min(1.0, max(0.0, (anchor / float(duration)))) if condition == "drift" else None
        ),
        "machines": [{k: m[k] for k in kept} for m in machines],
        "queries": queries,
        "history": history_rows,
        "labels": [{"query_id": m["query_id"], "outcome": str(m["outcome"])} for m in machines],
        "targets": [{
            "query_id": m["query_id"],
            "episode_id": episode_id,
            **{c: format(p, ".17g") for c, p in zip(PROBABILITY_COLUMNS, m["probabilities"])},
            "outcome": str(m["outcome"]),
        } for m in machines],
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


def _metadata(public, name, config, counts):
    payload = {
        "conditions": list(CONDITIONS),
        "counts_per_condition": counts,
        "history": "Synthetic g1 kernel with configurable lag pairs, levels, drift and flip timing.",
        "machines_per_episode": 8,
        "max_points": 40,
        "mode": "event",
        "reference": "Pooled, strength=20, fitted separately on this round's training observations.",
        "round": "adaptation",
        "season_id": hashlib.sha256(f"g7-round2-{name}".encode()).hexdigest()[:24],
        "suite_version": "experimentation-suite-v3",
        "target": "Next-spin probabilities for 0x, 1x, 2x, 5x, 20x.",
        "test_policy": "Test targets stay in latents.json only; the public test split has no targets.",
        "generator": {
            "name": name,
            "pairs": [list(p) for p in config["pairs"]],
            "levels": config["levels"],
            "drift_anchor": config["drift_anchor"],
            "drift_duration": config["drift_duration"],
            "weights": list(config["weights"]) if config.get("weights") else None,
            "flip_frac": config["flip_frac"],
            "flip_sd": config["flip_sd"],
            "note": config["note"],
            "seed": config["seed"],
        },
    }
    (public / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def generate_config(name, config, out_root: Path):
    atoms = _atoms()
    root = out_root / name
    public = root / "public"
    splits = {}
    gap_sum = [0.0, 0]
    for index, split in enumerate(("train", "validation", "test")):
        rng = random.Random(config["seed"] + index * 10007)
        episodes = []
        n = 0
        for condition in CONDITIONS:
            for _ in range(COUNTS[split]):
                episode_id = f"{name[:2]}-{split[:3]}-e{n:02d}"
                episode = simulate_episode(rng, episode_id, condition, config, atoms)
                gap_sum[0] += math.fsum(episode["gaps"])
                gap_sum[1] += len(episode["gaps"])
                episodes.append(episode)
                n += 1
        _write_split(public, split, episodes, with_targets=(split == "validation"))
        splits[split] = [{
            k: episode[k] for k in (
                "episode_id", "condition", "cutoff", "shared_tick", "u_at_cutoff", "machines",
            )
        } for episode in episodes]
    _metadata(public, name, config, dict(COUNTS))
    _reference(public)
    dumped = {**config, "pairs": [list(p) for p in config["pairs"]]}
    if dumped.get("weights") is not None:
        dumped["weights"] = list(dumped["weights"])
    latents = {"config": dumped, "splits": splits}
    (root / "latents.json").write_text(json.dumps(latents) + "\n", encoding="utf-8")
    load_inputs(public / "train")
    load_inputs(public / "validation")
    load_inputs(public / "test")
    targets = read_targets(public / "validation" / "targets.csv")
    meta = round_metadata(public)
    mean_gap = gap_sum[0] / gap_sum[1]
    print(
        f"{name}: episodes train/val/test "
        f"{len(splits['train'])}/{len(splits['validation'])}/{len(splits['test'])} "
        f"val targets {len(targets)} round {meta['round']} points {meta['max_points']} "
        f"mean oriented cell gap {mean_gap:.3f} on {gap_sum[1]} machines"
    )
    if mean_gap < 0.25:
        raise RuntimeError(f"{name}: cell gap {mean_gap:.3f} is too small; the parity rule is not in the series")
    return root


def main():
    out = HERE / "data"
    out.mkdir(parents=True, exist_ok=True)
    for name, config in CONFIGS.items():
        generate_config(name, config, out)


if __name__ == "__main__":
    main()
