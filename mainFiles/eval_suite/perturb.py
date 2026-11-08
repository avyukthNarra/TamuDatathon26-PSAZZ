"""Black-box causality and invariance variants of a kit-layout dataset.

Each variant is a full copy of the dataset directory (all splits) in which
only the scored split changes. An entry is run on the base dataset and on
each variant; compare() then checks the two prediction CSVs against the
relation the variant declares:

- "identical": predictions must match the base within tol, keyed by query_id.
- "identical_after_map": as above, after mapping variant query_ids back to
  base query_ids through Variant.id_map.
- "identical_subset": as "identical", but only the queries in Variant.subset
  (the chosen episode of CAU-CROSS-EPISODE) are compared.
- "info": not a pass/fail requirement. compare() reports how far predictions
  move; compare_missing_vs_zero() compares the two INFO-MISSING-VS-ZERO
  variants with each other.

The causal variants test DATA.md / RULES.md "At the table": a prediction for
cutoff c may use only observations strictly before tick c.

Caveats:
- CAU-SAME-TICK-OTHERS adds a tick == cutoff row for EVERY machine, including
  the query's own machine. Each query's label is that tick, so no query may use
  any machine's outcome at its cutoff. The variant only means something
  alongside per-machine predictions (one query per machine), so it is compared
  per query_id like the others.
- The kit loader (slot_forecasting.schema.index_history) rejects history at or
  after a query's cutoff, so the two history-appending variants will not load
  through the kit's loader. An entry that loads data through the kit loader
  fails on them with a loader error, which the caller should treat as the leak
  guard firing, not as a prediction difference.
"""

from __future__ import annotations

import csv
import functools
import math
import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from eval_suite.contract import CheckResult

PAYOUTS = (0, 1, 2, 5, 20)
PROBABILITY_COLUMNS = tuple(f"p_{payout}" for payout in PAYOUTS)
SPLITS = ("train", "validation", "test")
ANSWER_FILES = ("labels.csv", "targets.csv", "reference.csv")
CONDITIONS_FILE = "conditions.csv"
ID_COLUMNS = ("query_id", "episode_id", "machine_id")
FUTURE_TICKS = 8
PROBE_LOOKBACK = 8
PROBE_MACHINES = 16
RELATIONS = ("identical", "identical_after_map", "identical_subset", "info")
EPISODE_NUMBER_OFFSET = 50000
MACHINE_NUMBER_OFFSET = 100
MISSING_VS_ZERO_ID = "INFO-MISSING-VS-ZERO"
CROSS_EPISODE_ID = "CAU-CROSS-EPISODE"


@dataclass(frozen=True)
class Variant:
    name: str
    data_root: Path
    relation: str
    id_map: dict[str, str] | None = None
    requirement_ids: list[str] = field(default_factory=list)
    description: str = ""
    subset: tuple[str, ...] | None = None  # query_ids compared under "identical_subset"

    def __post_init__(self):
        if self.relation not in RELATIONS:
            raise ValueError(f"relation must be one of {RELATIONS}, got {self.relation!r}")
        if self.relation == "identical_after_map" and self.id_map is None:
            raise ValueError(f"{self.name}: identical_after_map needs an id_map")
        if self.relation == "identical_subset" and not self.subset:
            raise ValueError(f"{self.name}: identical_subset needs a non-empty subset")
        if self.relation != "identical_subset" and self.subset is not None:
            raise ValueError(f"{self.name}: subset is only used by identical_subset")
        object.__setattr__(self, "data_root", Path(self.data_root))


# ---------------------------------------------------------------------------
# CSV helpers (kit layout: UTF-8, LF, header row, no empty fields)
# ---------------------------------------------------------------------------

def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path}: empty file")
        rows = [dict(zip(header, row)) for row in reader if row]
    return header, rows


def _write_csv(path, header, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _history_row(episode_id, machine_id, tick, payout):
    return {"episode_id": episode_id, "machine_id": machine_id,
            "tick": str(tick), "payout": str(payout)}


# ---------------------------------------------------------------------------
# Variant builders. Each takes the copied dataset root, the split, and a
# per-variant RNG, rewrites only the scored split, and returns
# (description, id_map).
# ---------------------------------------------------------------------------

def _check_history_before_cutoffs(queries, history):
    cutoffs = {(q["episode_id"], q["machine_id"]): int(q["cutoff"]) for q in queries}
    for row in history:
        key = (row["episode_id"], row["machine_id"])
        if key in cutoffs and int(row["tick"]) >= cutoffs[key]:
            raise ValueError(f"base history already at/after cutoff for {key}")


def _append_future_ticks(dst, split, rng):
    scored = dst / split
    header, history = _read_csv(scored / "history.csv")
    _, queries = _read_csv(scored / "queries.csv")
    _check_history_before_cutoffs(queries, history)
    added = [
        _history_row(q["episode_id"], q["machine_id"], tick, rng.choice(PAYOUTS))
        for q in queries
        for tick in range(int(q["cutoff"]), int(q["cutoff"]) + FUTURE_TICKS)
    ]
    _write_csv(scored / "history.csv", header, history + added)
    return (f"appends {len(added)} rows at ticks cutoff..cutoff+{FUTURE_TICKS - 1} "
            f"for every query machine, random payouts"), None


def _append_same_tick(dst, split, rng):
    scored = dst / split
    header, history = _read_csv(scored / "history.csv")
    _, queries = _read_csv(scored / "queries.csv")
    _check_history_before_cutoffs(queries, history)
    added = [
        _history_row(q["episode_id"], q["machine_id"], int(q["cutoff"]), rng.choice(PAYOUTS))
        for q in queries
    ]
    _write_csv(scored / "history.csv", header, history + added)
    return (f"appends {len(added)} rows at tick == cutoff, one per machine of every "
            f"episode, random payouts"), None


def _remove_conditions(dst, split, rng):
    path = dst / split / CONDITIONS_FILE
    if path.exists():
        path.unlink()
        return f"removes {CONDITIONS_FILE} from the scored split", None
    return f"scored split has no {CONDITIONS_FILE}; variant equals base", None


def _remove_answers(dst, split, rng):
    removed = []
    for name in ANSWER_FILES:
        path = dst / split / name
        if path.exists():
            path.unlink()
            removed.append(name)
    return f"removes from the scored split: {', '.join(removed) or 'nothing (none present)'}", None


def _shuffle_rows(dst, split, rng):
    for name in ("queries.csv", "history.csv"):
        path = dst / split / name
        header, rows = _read_csv(path)
        shuffled = list(rows)
        for _ in range(100):
            rng.shuffle(shuffled)
            if shuffled != rows:
                break
        _write_csv(path, header, shuffled)
    return "shuffles row order of queries.csv and history.csv", None


def _lookup(mapping, key, kind):
    try:
        return mapping[key]
    except KeyError:
        raise ValueError(f"{kind} {key!r} appears in a scored-split file but not in queries.csv") from None


def _rename_maps(queries, rng):
    """New ids keep the '<prefix>-e#####' / '-m###' / '-next' shape."""
    episodes = sorted({q["episode_id"] for q in queries})
    numbers = list(range(EPISODE_NUMBER_OFFSET, EPISODE_NUMBER_OFFSET + len(episodes)))
    rng.shuffle(numbers)
    episode_map = {}
    for old, number in zip(episodes, numbers):
        match = re.fullmatch(r"(.*)-e(\d+)", old)
        if not match:
            raise ValueError(f"episode id {old!r} does not match '<prefix>-e<digits>'")
        episode_map[old] = f"{match.group(1)}-e{number:0{len(match.group(2))}d}"

    machine_map = {}
    for old_episode in episodes:
        machines = sorted({q["machine_id"] for q in queries if q["episode_id"] == old_episode})
        slots = list(range(MACHINE_NUMBER_OFFSET, MACHINE_NUMBER_OFFSET + len(machines)))
        rng.shuffle(slots)
        for old_machine, slot in zip(machines, slots):
            if not re.fullmatch(r"-m\d+", old_machine[len(old_episode):]):
                raise ValueError(f"machine id {old_machine!r} does not match '<episode>-m<digits>'")
            machine_map[old_machine] = f"{episode_map[old_episode]}-m{slot:03d}"

    query_map = {}
    for q in queries:
        old_query, old_machine = q["query_id"], q["machine_id"]
        if not old_query.startswith(old_machine):
            raise ValueError(f"query id {old_query!r} does not start with its machine id")
        query_map[old_query] = machine_map[old_machine] + old_query[len(old_machine):]
    if len(set(query_map.values())) != len(query_map):
        raise ValueError("renamed query ids are not unique")
    return episode_map, machine_map, query_map


def _randomize_other_episodes(dst, split, rng):
    """Replace history payouts (and labels, when the split has them) of every episode but one.

    Ticks and cutoffs are kept. Returns (description, query_ids of the kept episode).
    """
    scored = dst / split
    _, queries = _read_csv(scored / "queries.csv")
    episodes = sorted({q["episode_id"] for q in queries})
    kept = rng.choice(episodes)
    header, history = _read_csv(scored / "history.csv")
    for row in history:
        if row["episode_id"] != kept:
            row["payout"] = str(rng.choice(PAYOUTS))
    _write_csv(scored / "history.csv", header, history)
    labels_path = scored / "labels.csv"
    if labels_path.exists():
        episode_of = {q["query_id"]: q["episode_id"] for q in queries}
        header, labels = _read_csv(labels_path)
        for row in labels:
            if episode_of[row["query_id"]] != kept:
                row["outcome"] = str(rng.choice(PAYOUTS))
        _write_csv(labels_path, header, labels)
    subset = tuple(sorted(q["query_id"] for q in queries if q["episode_id"] == kept))
    description = (f"keeps episode {kept} and replaces the history payouts of the other "
                   f"{len(episodes) - 1} episodes with random payouts (ticks kept)")
    return description, subset


def _rename_ids(dst, split, rng):
    scored = dst / split
    _, queries = _read_csv(scored / "queries.csv")
    episode_map, machine_map, query_map = _rename_maps(queries, rng)
    for path in sorted(scored.glob("*.csv")):
        header, rows = _read_csv(path)
        if not set(header) & set(ID_COLUMNS):
            continue
        for row in rows:
            if "episode_id" in row:
                row["episode_id"] = _lookup(episode_map, row["episode_id"], "episode_id")
            if "machine_id" in row:
                row["machine_id"] = _lookup(machine_map, row["machine_id"], "machine_id")
            if "query_id" in row:
                row["query_id"] = _lookup(query_map, row["query_id"], "query_id")
        _write_csv(path, header, rows)
    id_map = {new: old for old, new in query_map.items()}
    return "renames episode_id, machine_id and query_id consistently in the scored split", id_map


def _probe_indices(queries, history, rng):
    """Indices into history of one observed row per sampled machine, ticks in
    [cutoff - PROBE_LOOKBACK, cutoff)."""
    cutoffs = {(q["episode_id"], q["machine_id"]): int(q["cutoff"]) for q in queries}
    candidates = {}
    for index, row in enumerate(history):
        key = (row["episode_id"], row["machine_id"])
        if key in cutoffs and cutoffs[key] - PROBE_LOOKBACK <= int(row["tick"]) < cutoffs[key]:
            candidates.setdefault(key, []).append(index)
    keys = sorted(candidates)
    chosen = rng.sample(keys, min(PROBE_MACHINES, len(keys)))
    return sorted(rng.choice(candidates[key]) for key in chosen)


def _build_probe(dst, split, rng, *, probe, drop):
    """Shared by the missing and zero variants: same rows, different encoding.
    The probe is chosen once in make_variants, so rng is unused here."""
    scored = dst / split
    header, history = _read_csv(scored / "history.csv")
    if drop:
        probed = set(probe)
        history = [row for index, row in enumerate(history) if index not in probed]
        text = f"deletes {len(probe)} observed rows (ticks cutoff-8..cutoff-1) of sampled machines"
    else:
        for index in probe:
            history[index]["payout"] = "0"
        text = f"sets the payout of {len(probe)} observed rows (ticks cutoff-8..cutoff-1) to 0"
    _write_csv(scored / "history.csv", header, history)
    return text, None


def _fresh_copy(data_root, dst):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(data_root, dst)


def make_variants(data_root, out_root, split="test", seed=0) -> list[Variant]:
    """Write one dataset copy per variant under out_root and describe each.

    The INFO-MISSING-VS-ZERO probe is chosen once from the base split, so the
    missing and zero variants touch the same history rows.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    data_root = Path(data_root)
    out_root = Path(out_root)
    base = data_root / split
    _, queries = _read_csv(base / "queries.csv")
    _, history = _read_csv(base / "history.csv")
    probe = _probe_indices(queries, history, random.Random(f"{seed}:{MISSING_VS_ZERO_ID}"))

    builders = [
        ("CAU-FUTURE-TICKS", "identical", _append_future_ticks),
        ("CAU-SAME-TICK-OTHERS", "identical", _append_same_tick),
        ("CAU-CONDITIONS-HIDDEN", "identical", _remove_conditions),
        ("CAU-ANSWERS-HIDDEN", "identical", _remove_answers),
        ("ROB-ROW-ORDER", "identical", _shuffle_rows),
        ("ROB-ID-RENAME", "identical_after_map", _rename_ids),
        (f"{MISSING_VS_ZERO_ID}-missing", "info",
         functools.partial(_build_probe, probe=probe, drop=True)),
        (f"{MISSING_VS_ZERO_ID}-zero", "info",
         functools.partial(_build_probe, probe=probe, drop=False)),
    ]

    variants = []
    for name, relation, build in builders:
        dst = out_root / name
        if dst.resolve().is_relative_to(data_root.resolve()):
            raise ValueError(f"out_root must not be inside data_root: {out_root}")
        _fresh_copy(data_root, dst)
        description, id_map = build(dst, split, random.Random(f"{seed}:{name}"))
        requirement_ids = [MISSING_VS_ZERO_ID] if relation == "info" else [name]
        variants.append(Variant(name=name, data_root=dst, relation=relation, id_map=id_map,
                                requirement_ids=requirement_ids, description=description))

    cross_dst = out_root / CROSS_EPISODE_ID
    if cross_dst.resolve().is_relative_to(data_root.resolve()):
        raise ValueError(f"out_root must not be inside data_root: {out_root}")
    _fresh_copy(data_root, cross_dst)
    description, subset = _randomize_other_episodes(
        cross_dst, split, random.Random(f"{seed}:{CROSS_EPISODE_ID}"))
    variants.append(Variant(name=CROSS_EPISODE_ID, data_root=cross_dst, relation="identical_subset",
                            subset=subset, requirement_ids=[CROSS_EPISODE_ID], description=description))
    return variants


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _load_predictions(path):
    header, rows = _read_csv(path)
    if set(header) != {"query_id", *PROBABILITY_COLUMNS}:
        raise ValueError(f"{path}: expected columns query_id,{','.join(PROBABILITY_COLUMNS)}")
    predictions = {}
    for row in rows:
        values = tuple(float(row[c]) for c in PROBABILITY_COLUMNS)
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"{path}: non-finite probability for {row['query_id']}")
        if row["query_id"] in predictions:
            raise ValueError(f"{path}: duplicate query_id {row['query_id']}")
        predictions[row["query_id"]] = values
    return predictions


def _differences(base, actual, tol):
    max_abs = 0.0
    differing = []
    for key in sorted(base):
        diff = max(abs(a - b) for a, b in zip(base[key], actual[key]))
        max_abs = max(max_abs, diff)
        if diff > tol:
            differing.append(key)
    return max_abs, differing


def _details(max_abs, differing, total, tol):
    return {"max_abs_diff": max_abs, "n_differing": len(differing), "n_queries": total,
            "tol": tol, "differing_query_ids": differing[:10]}


def _align(base, actual, id_map=None):
    """Return (actual keyed by base query_id, error message or None)."""
    if id_map is not None:
        unmapped = sorted(q for q in actual if q not in id_map)
        if unmapped:
            return None, f"variant query_ids missing from id_map: {unmapped[:5]}"
        mapped = {id_map[q]: probs for q, probs in actual.items()}
        if len(mapped) != len(actual):
            return None, "id_map is not one-to-one on the variant query_ids"
        actual = mapped
    missing = sorted(set(base) - set(actual))
    extra = sorted(set(actual) - set(base))
    if missing or extra:
        return None, f"query_id mismatch: missing={missing[:5]}, unexpected={extra[:5]}"
    return actual, None


def compare(base_csv, variant_csv, variant, tol=1e-9) -> CheckResult:
    """Compare a variant's predictions with the base predictions.

    passed is False only for a broken file, an id mismatch, or (for identical
    relations) any query whose probabilities differ by more than tol. For the
    info relation, passed is True when the files align, and severity is "info".
    For the other relations severity is "fail", the severity of a failure.
    """
    try:
        base = _load_predictions(base_csv)
        actual = _load_predictions(variant_csv)
    except (OSError, ValueError, KeyError) as exc:
        return CheckResult(variant.name, False, "fail", f"unreadable predictions: {exc}", {})

    actual, error = _align(base, actual, variant.id_map if variant.relation == "identical_after_map" else None)
    if error:
        return CheckResult(variant.name, False, "fail", error, {})

    scope = base
    if variant.relation == "identical_subset":
        unknown = sorted(set(variant.subset) - set(base))
        if unknown:
            return CheckResult(variant.name, False, "fail",
                               f"subset query_ids not in the predictions: {unknown[:5]}", {})
        scope = {query: base[query] for query in variant.subset}
    total = len(scope)
    max_abs, differing = _differences(scope, actual, tol)
    details = _details(max_abs, differing, total, tol)
    if variant.relation == "info":
        return CheckResult(
            variant.name, True, "info",
            f"info: {len(differing)} of {total} queries move (max abs diff {max_abs:.3g}) "
            "when the variant changes the data",
            details)
    if differing:
        return CheckResult(
            variant.name, False, "fail",
            f"{len(differing)} of {total} queries changed (max abs diff {max_abs:.3g} > tol {tol:g})",
            details)
    return CheckResult(variant.name, True, "fail",
                       f"predictions identical within tol (max abs diff {max_abs:.3g}) over {total} queries",
                       details)


def compare_missing_vs_zero(missing_csv, zero_csv, tol=1e-9) -> CheckResult:
    """Compare predictions from the -missing and -zero variants.

    Docs say a missing tick is unknown, not a zero payout. An entry that gives
    identical predictions for both gets a warning (passed=True, severity "warn").
    An entry that separates them gets an info result. A misaligned pair fails.
    """
    try:
        missing = _load_predictions(missing_csv)
        zero = _load_predictions(zero_csv)
    except (OSError, ValueError, KeyError) as exc:
        return CheckResult(MISSING_VS_ZERO_ID, False, "fail", f"unreadable predictions: {exc}", {})

    zero, error = _align(missing, zero)
    if error:
        return CheckResult(MISSING_VS_ZERO_ID, False, "fail", error, {})

    max_abs, differing = _differences(missing, zero, tol)
    details = _details(max_abs, differing, len(missing), tol)
    if not differing:
        return CheckResult(
            MISSING_VS_ZERO_ID, True, "warn",
            "entry treats a missing tick the same as a zero payout; docs say missing is unknown, not a loss",
            details)
    return CheckResult(
        MISSING_VS_ZERO_ID, True, "info",
        f"entry distinguishes missing from zero: {len(differing)} of {len(missing)} queries differ "
        f"(max abs diff {max_abs:.3g})",
        details)
