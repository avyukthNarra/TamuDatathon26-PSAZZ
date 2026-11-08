"""Deterministic, spec-driven synthetic datasets for the evaluation suite.

The simulator is deliberately a broad family rather than a reproduction of
any submitted model.  A target vector is conditional on the latent state that
exists immediately before the synchronous tick is sampled; all other inputs
are earlier outcomes only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Mapping, Sequence

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import write_json, write_rows
from slot_forecasting.schema import (
    CONDITIONS, HISTORY_COLUMNS, LABEL_COLUMNS, PAYOUTS, QUERY_COLUMNS,
    TARGET_COLUMNS, Play, Query, Target,
)

MACHINE_TYPES = ("steady", "balanced", "jackpot")
SIGNALS = (-1, 0, 1)
POSITIVE_PAYOUTS = (1, 2, 5, 20)
DEFAULT_COUNTS = {"train": 24, "validation": 24, "test": 48}
PRESETS = (
    "main_like", "round2_shift", "all_lags", "neighbour_heavy",
    "hidden_state_heavy", "fast_drift", "edge_cases",
)


@dataclass(frozen=True)
class GeneratorConfig:
    """Observable simulator controls; probability mappings are type keyed."""

    base_win_rates: Mapping[str, Sequence[float]] = field(default_factory=lambda: {
        "steady": (0.5268497145, 0.5639115667, 0.6408032596),
        "balanced": (0.3661945813, 0.4004863978, 0.4578806085),
        "jackpot": (0.1699834815, 0.2091516780, 0.2960722702),
    })
    payout_mixes: Mapping[str, Sequence[float]] = field(default_factory=lambda: {
        "steady": (0.6168124893, 0.3028992966, 0.0786755876, 0.0016126265),
        "balanced": (0.3523177496, 0.4474544374, 0.1710083201, 0.0292194929),
        "jackpot": (0.2107083101, 0.2968209704, 0.3637479085, 0.1287228109),
    })
    recurring_lags: tuple[int, ...] = tuple(range(1, 9))
    interaction_count_min: int = 1
    interaction_count_max: int = 3
    interaction_strength: float = 1.3
    neighbour_strength: float = 0.18
    neighbour_probability: float = 0.30
    hidden_state_strength: float = 0.45
    hidden_state_persistence: float = 0.88
    drift_duration: int = 64
    shared_change_fraction: float = 0.65
    cutoff_min: int = 192
    cutoff_max: int = 256
    size_shift_strength: float = 0.7
    edge_sparse: bool = False
    edge_specials: bool = False


@dataclass(frozen=True)
class LagInteraction:
    lags: tuple[int, ...]
    operator: str
    sign: int


@dataclass(frozen=True)
class MachineParameters:
    interactions: tuple[LagInteraction, ...]
    neighbour_offset: int | None
    neighbour_lag: int | None
    neighbour_sign: int
    size_lag: int
    size_shift_sign: int
    drift_sign: int
    drift_start: int
    drift_duration: int
    shared_sign: int
    all_loss: bool = False
    rare_jackpot: bool = False


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: str
    condition: str
    cutoff: int
    queries: tuple[Query, ...]
    history: tuple[Play, ...]
    labels: Mapping[str, int]
    targets: tuple[Target, ...]
    parameters: tuple[MachineParameters, ...]
    shared_change_tick: int | None
    machine_types: tuple[str, ...]
    signals: tuple[int, ...]
    metadata: Mapping[str, object]


def _sigmoid(value: float) -> float:
    value = max(-20.0, min(20.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def _logit(probability: float) -> float:
    probability = min(1.0 - 1e-9, max(1e-9, probability))
    return math.log(probability / (1.0 - probability))


def _normalise(values: Sequence[float]) -> tuple[float, ...]:
    vals = tuple(max(0.0, float(value)) for value in values)
    total = math.fsum(vals)
    if total <= 0 or not math.isfinite(total):
        raise ValueError("Probability weights must have positive finite mass.")
    return tuple(value / total for value in vals)


def _merge_config(config: GeneratorConfig, overrides: Mapping[str, object] | None) -> GeneratorConfig:
    if not overrides:
        return config
    names = set(GeneratorConfig.__dataclass_fields__)
    unknown = set(overrides) - names
    if unknown:
        raise ValueError(f"Unknown generator config field(s): {', '.join(sorted(unknown))}.")
    return replace(config, **dict(overrides))


def preset_config(name: str, overrides: Mapping[str, object] | None = None) -> GeneratorConfig:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}; choose one of {', '.join(PRESETS)}.")
    config = GeneratorConfig()
    if name == "round2_shift":
        config = replace(config, recurring_lags=(2, 3, 5, 7), interaction_strength=0.85,
                         shared_change_fraction=0.55)
    elif name == "all_lags":
        config = replace(config, recurring_lags=tuple(range(1, 9)), interaction_count_min=2,
                         interaction_count_max=3, interaction_strength=0.85)
    elif name == "neighbour_heavy":
        config = replace(config, neighbour_probability=1.0, neighbour_strength=0.9)
    elif name == "hidden_state_heavy":
        config = replace(config, hidden_state_strength=1.15, hidden_state_persistence=0.96)
    elif name == "fast_drift":
        config = replace(config, drift_duration=20, interaction_strength=0.9)
    elif name == "edge_cases":
        config = replace(config, cutoff_min=192, cutoff_max=256, edge_sparse=True,
                         edge_specials=True)
    return _merge_config(config, overrides)


def _validate_config(config: GeneratorConfig) -> None:
    if not 1 <= config.interaction_count_min <= config.interaction_count_max <= 3:
        raise ValueError("Interaction count bounds must be ordered within 1..3.")
    if not config.recurring_lags or any(lag < 1 or lag > 8 for lag in config.recurring_lags):
        raise ValueError("recurring_lags must be a nonempty subset of 1..8.")
    if config.cutoff_min < 192 or config.cutoff_max > 256 or config.cutoff_min > config.cutoff_max:
        raise ValueError("Cutoffs must lie within 192..256.")
    for name in ("interaction_strength", "neighbour_strength", "hidden_state_strength", "size_shift_strength"):
        value = getattr(config, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if not 0.0 <= config.hidden_state_persistence < 1.0:
        raise ValueError("hidden_state_persistence must be in [0,1).")
    if not 0.0 <= config.neighbour_probability <= 1.0:
        raise ValueError("Sampling probabilities must be in [0,1].")
    if not 0.0 < config.shared_change_fraction < 1.0 or config.drift_duration < 1:
        raise ValueError("Change fraction and drift duration are invalid.")
    for machine_type in MACHINE_TYPES:
        rates = tuple(config.base_win_rates[machine_type])
        mixes = tuple(config.payout_mixes[machine_type])
        if len(rates) != 3 or any(not 0 <= rate <= 1 for rate in rates):
            raise ValueError(f"Invalid win rates for {machine_type}.")
        if len(mixes) != 4 or any(value < 0 for value in mixes) or math.fsum(mixes) <= 0:
            raise ValueError(f"Invalid payout mix for {machine_type}.")


def _history_payout(prefix, machine_index: int, tick: int, *, machine_ids: Sequence[str] | None = None) -> int | None:
    """Read a prior payout from a machine/tick mapping or tick-major rows."""
    if tick < 0:
        return None
    machine_key = machine_ids[machine_index] if machine_ids is not None else machine_index
    if isinstance(prefix, Mapping):
        machine_history = prefix.get(machine_key)
        if machine_history is None:
            machine_history = prefix.get(str(machine_key))
        if isinstance(machine_history, Mapping):
            value = machine_history.get(tick)
        elif isinstance(machine_history, Sequence) and not isinstance(machine_history, (str, bytes)):
            value = machine_history[tick] if tick < len(machine_history) else None
        else:
            value = None
        return None if value is None else int(value)
    if isinstance(prefix, Sequence):
        if tick >= len(prefix):
            return None
        row = prefix[tick]
        if isinstance(row, Mapping):
            value = row.get(machine_key, row.get(str(machine_key)))
        else:
            value = row[machine_index] if machine_index < len(row) else None
        return None if value is None else int(value)
    raise TypeError("prefix must be a tick-major or machine-indexed history mapping/sequence")


def _history_value(prefix, machine_index: int, tick: int, *, machine_ids: Sequence[str] | None = None) -> int | None:
    payout = _history_payout(prefix, machine_index, tick, machine_ids=machine_ids)
    return None if payout is None else int(payout > 0)


def _interaction_value(prefix, machine_index: int, interaction: LagInteraction,
                       tick: int, machine_ids: Sequence[str]) -> int | None:
    values = [_history_value(prefix, machine_index, tick - lag, machine_ids=machine_ids)
              for lag in interaction.lags]
    if any(value is None for value in values):
        return None
    if interaction.operator == "single":
        return 1 if values[0] else -1
    if interaction.operator == "xor":
        return 1 if sum(values) % 2 else -1
    if interaction.operator == "and":
        return 1 if all(values) else -1
    if interaction.operator == "or":
        return 1 if any(values) else -1
    raise ValueError(f"Unknown interaction operator: {interaction.operator}")


def conditional_probabilities(
    prefix,
    machine_types: Sequence[str],
    signals: Sequence[int],
    params: Sequence[MachineParameters],
    tick: int,
    latent_state: Sequence[float],
    *,
    shared_changed: bool = False,
    config: GeneratorConfig | None = None,
    machine_ids: Sequence[str] | None = None,
    drift_progress: float | None = None,
) -> tuple[tuple[float, ...], ...]:
    """Return synchronous five-outcome probabilities using only ticks `< tick`.

    `prefix` can be tick-major rows indexed by machine position, or a mapping
    from machine id/index to a tick-indexed history. Latent states are supplied
    by the simulator and are fixed before outcomes for this tick are drawn.
    """
    config = config or GeneratorConfig()
    count = len(machine_types)
    if count != 8 or len(signals) != count or len(params) != count or len(latent_state) != count:
        raise ValueError("An episode must supply exactly eight aligned machines.")
    ids = tuple(machine_ids or tuple(range(count)))
    if len(ids) != count:
        raise ValueError("machine_ids must align with machines.")
    output = []
    for i, (machine_type, signal, parameter) in enumerate(zip(machine_types, signals, params)):
        rates = config.base_win_rates[machine_type]
        log_odds = _logit(float(rates[SIGNALS.index(signal)]))
        progress = drift_progress
        if progress is None:
            progress = min(1.0, max(0.0, (tick - parameter.drift_start) / max(1, parameter.drift_duration)))
        for interaction in parameter.interactions:
            value = _interaction_value(prefix, i, interaction, tick, ids)
            if value is not None:
                log_odds += (config.interaction_strength * interaction.sign * value
                             * parameter.drift_sign * (1.0 - 2.0 * progress))
        if parameter.neighbour_offset is not None and parameter.neighbour_lag is not None:
            neighbor = (i + parameter.neighbour_offset) % count
            value = _history_value(prefix, neighbor, tick - parameter.neighbour_lag, machine_ids=ids)
            if value is not None:
                log_odds += config.neighbour_strength * parameter.neighbour_sign * (1 if value else -1)
        log_odds += config.hidden_state_strength * float(latent_state[i])
        if shared_changed:
            log_odds += 0.42 * parameter.shared_sign
        if parameter.all_loss:
            win_probability = 0.0
        elif parameter.rare_jackpot:
            win_probability = min(0.08, max(0.004, _sigmoid(log_odds) * 0.20))
        else:
            win_probability = _sigmoid(log_odds)

        mix = _normalise(config.payout_mixes[machine_type])
        prior_payout = _history_payout(prefix, i, tick - parameter.size_lag, machine_ids=ids)
        if prior_payout is not None:
            direction = parameter.size_shift_sign * (1 if prior_payout >= 5 else -1)
            tilt = (0.5, 0.75, 1.25, 2.3) if direction > 0 else (1.5, 1.2, 0.85, 0.35)
            shifted = _normalise(tuple(weight * factor for weight, factor in zip(mix, tilt)))
            amount = min(1.0, config.size_shift_strength)
            mix = _normalise(tuple((1.0 - amount) * base + amount * new
                                   for base, new in zip(mix, shifted)))
        if parameter.rare_jackpot:
            rare_mix = (0.01, 0.02, 0.07, 0.90)
            mix = _normalise(tuple((1.0 - 0.90) * base + 0.90 * rare
                                   for base, rare in zip(mix, rare_mix)))
        output.append((1.0 - win_probability, *(win_probability * value for value in mix)))
    return tuple(output)


def _make_parameters(rng: random.Random, config: GeneratorConfig, cutoff: int,
                     machine_index: int, *, force_all_lags: bool = False,
                     all_loss: bool = False, rare_jackpot: bool = False) -> MachineParameters:
    n = rng.randint(config.interaction_count_min, config.interaction_count_max)
    lags = list(config.recurring_lags)
    if force_all_lags and len(lags) == 8:
        chosen_lags = [(machine_index + 1)]
        n = max(n, 2)
        rest = [lag for lag in lags if lag not in chosen_lags]
        rng.shuffle(rest)
        chosen_lags.extend(rest[:n - 1])
    else:
        chosen_lags = rng.choices(lags, k=n)
    interactions = []
    for lag in chosen_lags:
        operator = "single" if len(config.recurring_lags) == 1 else rng.choice(("single", "xor", "and", "or"))
        pair = (lag, rng.choice(tuple(x for x in config.recurring_lags if x != lag))) if operator != "single" else (lag,)
        interactions.append(LagInteraction(pair, operator, rng.choice((-1, 1))))
    if rng.random() < config.neighbour_probability:
        offsets = tuple(value for value in range(-7, 8) if value)
        neighbour_offset = rng.choice(offsets)
        neighbour_lag = rng.choice(config.recurring_lags)
        neighbour_sign = rng.choice((-1, 1))
    else:
        neighbour_offset = neighbour_lag = None
        neighbour_sign = 1
    return MachineParameters(
        tuple(interactions), neighbour_offset, neighbour_lag, neighbour_sign,
        rng.choice(config.recurring_lags), rng.choice((-1, 1)), rng.choice((-1, 1)),
        max(0, cutoff - config.drift_duration), config.drift_duration,
        rng.choice((-1, 1)), all_loss, rare_jackpot,
    )


def _sample_payout(rng: random.Random, probabilities: Sequence[float]) -> int:
    point = rng.random()
    cumulative = 0.0
    for payout, probability in zip(PAYOUTS, probabilities):
        cumulative += probability
        if point <= cumulative:
            return payout
    return PAYOUTS[-1]


def _episode(seed: int, episode_id: str, condition: str, config: GeneratorConfig,
             *, edge_index: int | None = None, force_all_lags: bool = False) -> EpisodeResult:
    rng = random.Random(seed)
    if edge_index is not None and edge_index % 2:
        cutoff = config.cutoff_max
    elif edge_index is not None:
        cutoff = config.cutoff_min
    else:
        cutoff = rng.randint(config.cutoff_min, config.cutoff_max)
    machine_ids = tuple(f"m{i:02d}" for i in range(8))
    machine_types = tuple(MACHINE_TYPES[i % len(MACHINE_TYPES)] for i in range(8))
    signals = tuple(SIGNALS[(i + (seed % 3)) % 3] for i in range(8))
    parameters = tuple(_make_parameters(
        rng, config, cutoff, i, force_all_lags=force_all_lags,
        all_loss=bool(config.edge_specials and edge_index == 0 and i == 0),
        rare_jackpot=bool(config.edge_specials and edge_index == 0 and i == 2),
    ) for i in range(8))
    shared_change_tick = (max(1, min(cutoff - 1, round(cutoff * config.shared_change_fraction)))
                          if condition == "shared_change" else None)
    latent = []
    state = [rng.choice((-1.0, 1.0)) for _ in range(8)]
    for _tick in range(cutoff + 1):
        latent.append(tuple(state))
        for i in range(8):
            if rng.random() > config.hidden_state_persistence:
                state[i] *= -1.0

    outcomes: list[list[int]] = [[] for _ in range(8)]
    vectors = None
    for tick in range(cutoff + 1):
        # At the query tick this produces the true next-outcome vector. Every
        # machine's vector is fixed before any machine's outcome is sampled.
        prefix = {machine_ids[i]: outcomes[i] for i in range(8)}
        drift_progress = (min(1.0, max(0.0, (tick - parameters[0].drift_start)
                                      / max(1, config.drift_duration)))
                          if condition == "drift" else 0.0)
        vectors = conditional_probabilities(
            prefix, machine_types, signals, parameters, tick, latent[tick],
            shared_changed=shared_change_tick is not None and tick >= shared_change_tick,
            config=config, machine_ids=machine_ids, drift_progress=drift_progress,
        )
        if tick < cutoff:
            sampled = [_sample_payout(rng, vector) for vector in vectors]
            for machine_index, outcome in enumerate(sampled):
                outcomes[machine_index].append(outcome)

    queries = tuple(Query(f"{episode_id}:{machine_id}", episode_id, machine_id,
                          machine_types[i], signals[i], cutoff)
                    for i, machine_id in enumerate(machine_ids))
    targets = tuple(Target(query.query_id, episode_id, vectors[i], _sample_payout(rng, vectors[i]))
                    for i, query in enumerate(queries))
    labels = {target.query_id: target.outcome for target in targets}
    short = condition == "short_history" or (config.edge_sparse and edge_index is not None and edge_index % 4 == 0)
    history = []
    for i, machine_id in enumerate(machine_ids):
        ticks = list(range(cutoff))
        if short:
            recent = list(range(max(0, cutoff - 40), cutoff))
            older = list(range(0, max(0, cutoff - 40)))
            older = rng.sample(older, min(24, len(older)))
            ticks = sorted(set(recent + older))
            if len(ticks) < 64:
                raise AssertionError("A short history must contain at least 64 observations.")
        history.extend(Play(episode_id, machine_id, tick, outcomes[i][tick]) for tick in ticks)
    meta = {
        "cutoff": cutoff,
        "shared_change_tick": shared_change_tick,
        "short_history": short,
        "history_counts": {machine_ids[i]: sum(play.machine_id == machine_ids[i] for play in history) for i in range(8)},
        "all_loss_machines": [machine_ids[i] for i, p in enumerate(parameters) if p.all_loss],
        "rare_jackpot_machines": [machine_ids[i] for i, p in enumerate(parameters) if p.rare_jackpot],
    }
    return EpisodeResult(episode_id, condition, cutoff, queries, tuple(history), labels, targets,
                         parameters, shared_change_tick, machine_types, signals, meta)


def simulate_episode(episode_id: str, condition: str, rng: random.Random,
                     config: GeneratorConfig, *, edge_index: int | None = None,
                     force_all_lags: bool = False) -> EpisodeResult:
    """Generate one episode; `rng` supplies a deterministic per-episode seed."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    if not isinstance(rng, random.Random):
        raise TypeError("rng must be an instance of random.Random")
    return _episode(rng.getrandbits(64), episode_id, condition, config,
                    edge_index=edge_index, force_all_lags=force_all_lags)


def _rows(values):
    return [asdict(value) for value in values]


def _target_rows(targets):
    from slot_forecasting.schema import PROBABILITY_COLUMNS
    return [{"query_id": target.query_id, "episode_id": target.episode_id,
             **{column: format(probability, ".17g") for column, probability in zip(PROBABILITY_COLUMNS, target.probabilities)},
             "outcome": target.outcome} for target in targets]


def _condition_rows(episodes):
    return [{"query_id": query.query_id, "condition": episode.condition}
            for episode in episodes for query in episode.queries]


def _label_rows(episodes):
    return [{"query_id": query.query_id, "outcome": episode.labels[query.query_id]}
            for episode in episodes for query in episode.queries]


def _write_split(root: Path, name: str, episodes: Sequence[EpisodeResult]) -> None:
    split = root / name
    queries = [query for episode in episodes for query in episode.queries]
    history = [play for episode in episodes for play in episode.history]
    write_rows(split / "queries.csv", QUERY_COLUMNS, _rows(queries))
    write_rows(split / "history.csv", HISTORY_COLUMNS, _rows(history))
    if name in ("train", "validation"):
        write_rows(split / "labels.csv", LABEL_COLUMNS, _label_rows(episodes))
        write_rows(split / "conditions.csv", ("query_id", "condition"), _condition_rows(episodes))
    if name == "validation":
        write_rows(split / "targets.csv", TARGET_COLUMNS,
                   _target_rows([target for episode in episodes for target in episode.targets]))


def _episode_private_rows(episodes):
    rows = []
    for episode in episodes:
        rows.append({
            "episode_id": episode.episode_id,
            "condition": episode.condition,
            **episode.metadata,
            "parameters": [asdict(parameter) for parameter in episode.parameters],
        })
    return rows


def _write_jsonl(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def generate_dataset(out_dir, *, preset="main_like", seed=0,
                     episodes_per_condition=None, config=None) -> Path:
    """Write a complete kit-layout dataset and return its root path.

    The public test split contains only query/history inputs. Its targets,
    conditions, and starter reference are written under `_truth/test/`.
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}.")
    config = config if config is not None else preset_config(preset)
    if not isinstance(config, GeneratorConfig):
        raise TypeError("config must be GeneratorConfig")
    _validate_config(config)
    if episodes_per_condition is not None and (type(episodes_per_condition) is not int or episodes_per_condition < 1):
        raise ValueError("episodes_per_condition must be a positive integer")
    root = Path(out_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    count = int(episodes_per_condition) if episodes_per_condition is not None else None
    counts = {split: count if count is not None else default for split, default in DEFAULT_COUNTS.items()}
    base_seed = int(seed)
    episode_sets: dict[str, list[EpisodeResult]] = {split: [] for split in counts}
    root_rng = random.Random(base_seed)
    for split in ("train", "validation", "test"):
        for condition in CONDITIONS:
            for index in range(counts[split]):
                episode_seed = root_rng.getrandbits(64)
                episode_id = f"syn-{base_seed}-{split}-{condition}-{index:04d}"
                edge_index = index if config.edge_specials else None
                result = _episode(episode_seed, episode_id, condition, config,
                                  edge_index=edge_index,
                                  force_all_lags=preset == "all_lags")
                episode_sets[split].append(result)
    for split, episodes in episode_sets.items():
        _write_split(root, split, episodes)

    train_queries = tuple(query for episode in episode_sets["train"] for query in episode.queries)
    train_history = tuple(play for episode in episode_sets["train"] for play in episode.history)
    train_labels = {query.query_id: episode.labels[query.query_id]
                    for episode in episode_sets["train"] for query in episode.queries}
    starter = FrequencyForecaster("pooled", prior_strength=20).fit(train_queries, train_history, train_labels)
    validation_queries = tuple(query for episode in episode_sets["validation"] for query in episode.queries)
    validation_history = tuple(play for episode in episode_sets["validation"] for play in episode.history)
    test_queries = tuple(query for episode in episode_sets["test"] for query in episode.queries)
    test_history = tuple(play for episode in episode_sets["test"] for play in episode.history)
    validation_reference = starter.predict(validation_queries, validation_history)
    test_reference = starter.predict(test_queries, test_history)
    from slot_forecasting.io import write_submission
    write_submission(root / "validation/reference.csv", validation_reference)

    for split in ("train", "validation", "test"):
        private = root / "_truth" / split
        write_rows(private / "targets.csv", TARGET_COLUMNS,
                   _target_rows([target for episode in episode_sets[split] for target in episode.targets]))
        write_rows(private / "conditions.csv", ("query_id", "condition"), _condition_rows(episode_sets[split]))
        _write_jsonl(private / "episode_metadata.jsonl", _episode_private_rows(episode_sets[split]))
        if split == "test":
            write_submission(private / "reference.csv", test_reference)
    metadata = {
        "dataset_version": "synthetic-eval-suite-v1",
        "suite_version": "experimentation-suite-v3",
        "preset": preset,
        "seed": base_seed,
        "round": "adaptation" if preset == "round2_shift" else "main",
        "max_points": 40 if preset == "round2_shift" else 60,
        "season_id": hashlib.sha256(f"eval-suite:{preset}:{base_seed}:{counts}".encode("utf-8")).hexdigest()[:24],
        "conditions": list(CONDITIONS),
        "machine_types": list(MACHINE_TYPES),
        "signals": list(SIGNALS),
        "payouts": list(PAYOUTS),
        "counts": counts,
        "counts_per_condition": dict(counts),
        "machines_per_episode": 8,
        "config": asdict(config),
        "private_test_truth": "_truth/test/",
    }
    # Main public metadata follows the kit's round metadata parser.
    write_json(root / "metadata.json", metadata)
    return root


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="main_like")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes-per-condition", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = generate_dataset(args.out, preset=args.preset, seed=args.seed,
                              episodes_per_condition=args.episodes_per_condition)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
