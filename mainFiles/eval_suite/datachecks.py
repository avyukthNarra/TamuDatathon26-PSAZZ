"""Data, package and mount-input checks (requirements.py CHECKS, DAT/CON/PKG/ROB/GEN IDs).

Entry points:
- check_dataset(data_root, round_name=None): one kit-layout dataset (real or generated).
- check_zip(zip_path, metadata_path=None): a packaged code ZIP.
- package_entry(entry_dir, out_zip): package a code folder with the kit's rules.
- check_mount_inputs(data_root, entry_dir): what the organizer would mount.

Result convention (same as contract.py): passed=True means the check holds.
A failing check has passed=False and severity "fail". The two warning checks
(PKG-NO-DATA-COPY, PKG-README-SEASON) return passed=False with severity "warn"
when they fire, and passed=True with severity "info" otherwise. Only severity
"fail" should block a submission.
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from eval_suite.contract import CheckResult
from slot_forecasting.schema import ValidationError
from slot_forecasting.submission_contract import inspect_archive, package_code

SPLITS = ("train", "validation", "test")
LABELED_SPLITS = ("train", "validation")
CONDITIONS = ("stable", "short_history", "drift", "shared_change")
MACHINE_TYPES = frozenset({"steady", "balanced", "jackpot"})
SIGNALS = frozenset({-1, 0, 1})
PAYOUTS = frozenset({0, 1, 2, 5, 20})
CUTOFF_MIN, CUTOFF_MAX = 192, 256
MACHINES_PER_EPISODE = 8
SHORT_RECENT, SHORT_OLDER = 40, 24
PROB_COLUMNS = ("p_0", "p_1", "p_2", "p_5", "p_20")
COLUMNS = {
    "queries.csv": ("query_id", "episode_id", "machine_id", "machine_type", "signal", "cutoff"),
    "history.csv": ("episode_id", "machine_id", "tick", "payout"),
    "labels.csv": ("query_id", "outcome"),
    "conditions.csv": ("query_id", "condition"),
    "targets.csv": ("query_id", "episode_id", *PROB_COLUMNS, "outcome"),
    "reference.csv": ("query_id", *PROB_COLUMNS),
}
REQUIRED_FILES = {
    "train": ("queries.csv", "history.csv", "labels.csv", "conditions.csv"),
    "validation": ("queries.csv", "history.csv", "labels.csv", "conditions.csv", "targets.csv", "reference.csv"),
    "test": ("queries.csv", "history.csv"),
}
ANSWER_BASENAMES = ("targets.csv", "labels.csv", "conditions.csv", "reference.csv")
SUITE_VERSIONS = ("temporal-suite-v2", "experimentation-suite-v3")
ROUND_POINTS = {"main": 60, "adaptation": 40}
SEASON_RE = re.compile(r"[0-9a-f]{24}")
INT_RE = re.compile(r"-?\d+")
REQUIREMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9._,-]+\])?==[A-Za-z0-9.!+_-]+")
ZIP_MAX_BYTES = 100_000_000
ZIP_MAX_ENTRIES = 5000
ZIP_MAX_UNCOMPRESSED = 536_870_912
JSON_MAX_BYTES = 65536
COMMAND_MAX_ARGS, COMMAND_MAX_CHARS = 64, 4096
DATASET_CHECK_IDS = (
    "DAT-SCHEMA-COLUMNS", "DAT-TEST-SPLIT-FILES", "DAT-QUERY-ENUMS", "DAT-CUTOFF-RANGE",
    "DAT-PAYOUT-SET", "DAT-CONDITION-SET", "DAT-COUNTS-PER-CONDITION", "DAT-MACHINES-PER-EPISODE",
    "DAT-SPLIT-DISJOINT", "DAT-HISTORY-CAUSAL", "CON-ROUND-METADATA", "GEN-HISTORY-SHAPE",
)
ZIP_CHECK_IDS = (
    "PKG-ZIP-SIZE", "PKG-ZIP-LIMITS", "PKG-ZIP-SAFE", "PKG-ZIP-ROOT", "PKG-REQUIREMENTS-PINNED",
    "PKG-NO-ANSWER-FILES", "PKG-NO-DATA-COPY", "PKG-README-SEASON", "CON-REPRODUCE-SCHEMA",
    "CON-IMAGE-PYTHON",
)
MOUNT_CHECK_IDS = ("ROB-MOUNT-NO-ANSWERS", "ROB-NO-FORECAST-INPUT")
# Messages from submission_contract.inspect_archive, first match wins; anything else is a safety error.
_INSPECT_TRANSLATION = (
    ("exceeds 5,000 entries", "PKG-ZIP-LIMITS"),
    ("at most 128 MiB", "PKG-ZIP-SIZE"),
    ("at its root", "PKG-ZIP-ROOT"),
    ("must explain how to reproduce", "PKG-ZIP-ROOT"),
    ("needs exactly image and command", "CON-REPRODUCE-SCHEMA"),
    ("(char ", "CON-REPRODUCE-SCHEMA"),  # json decode error inside reproduce.json
    ("command must be", "CON-REPRODUCE-SCHEMA"),
    ("image must name", "CON-IMAGE-PYTHON"),
)


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _pass(check_id, message, **details):
    return CheckResult(check_id, True, "info", message, details)


def _fail(check_id, message, severity="fail", **details):
    return CheckResult(check_id, False, severity, message, details)


def _verdict(check_id, problems, ok_message, severity="fail", **details):
    if not problems:
        return _pass(check_id, ok_message, **details)
    shown = problems[:5]
    more = f" (+{len(problems) - len(shown)} more)" if len(problems) > len(shown) else ""
    return _fail(check_id, "; ".join(shown) + more, severity, count=len(problems), examples=shown, **details)


def _unevaluated(check_ids, reason, severity="fail"):
    return [_fail(cid, f"cannot evaluate: {reason}", severity) for cid in check_ids]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _read_rows(path):
    text = Path(path).read_bytes().decode("utf-8")
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows:
        raise ValueError("empty file")
    return rows[0], [row for row in rows[1:] if row]


def _load_split(root, split, schema_problems):
    """Return {file name: list of well-formed row dicts} for the files that load and match their header."""
    split_dir = root / split
    if not split_dir.is_dir():
        schema_problems.append(f"missing split directory {split}/")
        return {}
    for name in REQUIRED_FILES[split]:
        if not (split_dir / name).is_file():
            schema_problems.append(f"{split}/{name} is missing")
    tables = {}
    for name, columns in COLUMNS.items():
        path = split_dir / name
        if not path.is_file():
            continue
        try:
            header, rows = _read_rows(path)
        except (OSError, UnicodeDecodeError, csv.Error, ValueError) as exc:
            schema_problems.append(f"{split}/{name} unreadable: {exc}")
            continue
        if tuple(header) != columns:
            schema_problems.append(f"{split}/{name} columns {header} != {list(columns)}")
            continue
        good = []
        for number, row in enumerate(rows, 2):
            if len(row) != len(header):
                schema_problems.append(f"{split}/{name} line {number} has {len(row)} fields")
            else:
                good.append(dict(zip(header, row)))
        tables[name] = good
    return tables


def _as_int(text):
    return int(text) if isinstance(text, str) and INT_RE.fullmatch(text) else None


def _load_metadata(root, problems, round_name):
    path = root / "metadata.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        problems.append(f"metadata.json unreadable: {exc}")
        return {}
    if not isinstance(meta, dict):
        problems.append("metadata.json is not a JSON object")
        return {}
    if meta.get("suite_version") not in SUITE_VERSIONS:
        problems.append(f"suite_version {meta.get('suite_version')!r} not in {list(SUITE_VERSIONS)}")
    round_value = meta.get("round")
    if round_value not in ROUND_POINTS:
        problems.append(f"round {round_value!r} must be 'main' or 'adaptation'")
    elif meta.get("max_points") != ROUND_POINTS[round_value]:
        problems.append(f"round {round_value!r} needs max_points {ROUND_POINTS[round_value]}, "
                        f"got {meta.get('max_points')!r}")
    if round_name is not None:
        if round_name not in ROUND_POINTS:
            problems.append(f"round_name {round_name!r} must be 'main' or 'adaptation'")
        elif round_value != round_name:
            problems.append(f"metadata round {round_value!r} != expected {round_name!r}")
    season = meta.get("season_id")
    if not isinstance(season, str) or not SEASON_RE.fullmatch(season):
        problems.append("season_id must be 24 lowercase hex characters")
    conditions = meta.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != 4 or set(conditions) != set(CONDITIONS):
        problems.append(f"conditions must be the four names {list(CONDITIONS)}")
    counts = meta.get("counts_per_condition")
    if not isinstance(counts, dict) or any(not isinstance(counts.get(s), int) for s in SPLITS):
        problems.append("counts_per_condition needs integer entries for train, validation and test")
    if meta.get("machines_per_episode") != MACHINES_PER_EPISODE:
        problems.append(f"machines_per_episode must be {MACHINES_PER_EPISODE}")
    return meta


# ---------------------------------------------------------------------------
# Dataset checks
# ---------------------------------------------------------------------------

def _check_queries(split, queries, problems):
    per_episode = defaultdict(set)
    pairs = Counter()
    for row in queries:
        label = f"{split} {row['query_id']}"
        if row["machine_type"] not in MACHINE_TYPES:
            problems["DAT-QUERY-ENUMS"].append(f"{label}: machine_type {row['machine_type']!r}")
        if _as_int(row["signal"]) not in SIGNALS:
            problems["DAT-QUERY-ENUMS"].append(f"{label}: signal {row['signal']!r}")
        cutoff = _as_int(row["cutoff"])
        if cutoff is None or not CUTOFF_MIN <= cutoff <= CUTOFF_MAX:
            problems["DAT-CUTOFF-RANGE"].append(f"{label}: cutoff {row['cutoff']!r}")
        per_episode[row["episode_id"]].add(row["machine_id"])
        pairs[(row["episode_id"], row["machine_id"])] += 1
    for episode, machines in per_episode.items():
        if len(machines) != MACHINES_PER_EPISODE:
            problems["DAT-MACHINES-PER-EPISODE"].append(
                f"{split} {episode}: {len(machines)} machines, expected {MACHINES_PER_EPISODE}")
    for (episode, machine), count in pairs.items():
        if count > 1:
            problems["DAT-MACHINES-PER-EPISODE"].append(f"{split} {episode}/{machine} appears {count} times")


def _is_short(ticks, cutoff):
    recent = set(range(cutoff - SHORT_RECENT, cutoff))
    older = ticks - recent
    return (recent <= ticks and len(older) == SHORT_OLDER and len(ticks) == SHORT_RECENT + SHORT_OLDER
            and all(0 <= t < cutoff - SHORT_RECENT for t in older))


def _check_history(split, queries, history, problems, shape_counts):
    cutoffs = {}
    for row in queries:
        cutoff = _as_int(row["cutoff"])
        if cutoff is not None:
            cutoffs[(row["episode_id"], row["machine_id"])] = cutoff
    seen = set()
    observed = defaultdict(set)
    for row in history:
        key = (row["episode_id"], row["machine_id"])
        label = f"{split} {key[1]} tick {row['tick']}"
        if _as_int(row["payout"]) not in PAYOUTS:
            problems["DAT-PAYOUT-SET"].append(f"{label}: payout {row['payout']!r}")
        tick = _as_int(row["tick"])
        if tick is None or tick < 0:
            problems["DAT-HISTORY-CAUSAL"].append(f"{label}: tick must be an integer >= 0")
            continue
        if key not in cutoffs:
            problems["DAT-HISTORY-CAUSAL"].append(f"{label}: no query for this machine")
            continue
        if tick >= cutoffs[key]:
            problems["DAT-HISTORY-CAUSAL"].append(f"{label}: at or after cutoff {cutoffs[key]}")
            continue
        if (key, tick) in seen:
            problems["DAT-HISTORY-CAUSAL"].append(f"{label}: duplicate observation")
        seen.add((key, tick))
        observed[key].add(tick)

    longs = shorts = 0
    for key, cutoff in cutoffs.items():
        ticks = observed.get(key, set())
        if ticks == set(range(cutoff)):
            longs += 1
        elif _is_short(ticks, cutoff):
            shorts += 1
        else:
            problems["GEN-HISTORY-SHAPE"].append(
                f"{split} {key[1]} cutoff {cutoff}: {len(ticks)} observed ticks is neither "
                "every tick < cutoff nor 40 recent + 24 older")
    shape_counts[split] = {"long": longs, "short": shorts}


def _check_conditions(split, queries, conditions, meta, problems):
    counts = meta.get("counts_per_condition") or {}
    query_episode = {row["query_id"]: row["episode_id"] for row in queries}
    seen_queries = set()
    episode_conditions = defaultdict(set)
    for row in conditions:
        if row["condition"] not in CONDITIONS:
            problems["DAT-CONDITION-SET"].append(f"{split} {row['query_id']}: condition {row['condition']!r}")
        if row["query_id"] not in query_episode:
            problems["DAT-CONDITION-SET"].append(f"{split} {row['query_id']}: not a query")
            continue
        seen_queries.add(row["query_id"])
        episode_conditions[query_episode[row["query_id"]]].add(row["condition"])
    unlabelled = len(set(query_episode) - seen_queries)
    if unlabelled:
        problems["DAT-CONDITION-SET"].append(f"{split}: {unlabelled} queries have no condition")
    occurring = {c for conds in episode_conditions.values() for c in conds}
    for name in CONDITIONS:
        if name not in occurring:
            problems["DAT-CONDITION-SET"].append(f"{split}: condition {name!r} never occurs")
    per_condition = Counter()
    for episode, conds in episode_conditions.items():
        if len(conds) != 1:
            problems["DAT-CONDITION-SET"].append(f"{split} {episode}: maps to {sorted(conds)}")
        else:
            per_condition[next(iter(conds))] += 1
    expected = counts.get(split)
    for name in CONDITIONS:
        if per_condition[name] != expected:
            problems["DAT-COUNTS-PER-CONDITION"].append(
                f"{split} {name}: {per_condition[name]} episodes, metadata says {expected}")


def _check_splits_disjoint(split_queries, problems):
    names = list(split_queries)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            qa, qb = split_queries[first], split_queries[second]
            for field in ("episode_id", "query_id"):
                shared = {row[field] for row in qa} & {row[field] for row in qb}
                if shared:
                    problems["DAT-SPLIT-DISJOINT"].append(
                        f"{first} and {second} share {len(shared)} {field} values, e.g. {sorted(shared)[0]}")


def check_dataset(data_root, round_name=None) -> list[CheckResult]:
    """Run the DAT-*, CON-ROUND-METADATA and GEN-HISTORY-SHAPE checks on one kit-layout dataset.

    round_name ('main' or 'adaptation') is compared with metadata.json's round when given.
    Returns one CheckResult per ID in DATASET_CHECK_IDS, in that order.
    """
    root = Path(data_root)
    if not root.is_dir():
        return _unevaluated(DATASET_CHECK_IDS, f"dataset root is not a directory: {root}")
    problems = defaultdict(list)
    meta = _load_metadata(root, problems["CON-ROUND-METADATA"], round_name)
    tables = {split: _load_split(root, split, problems["DAT-SCHEMA-COLUMNS"]) for split in SPLITS}
    shape_counts = {}
    split_queries = {}

    for split in SPLITS:
        table = tables[split]
        queries = table.get("queries.csv")
        history = table.get("history.csv")
        if queries is None:
            for cid in ("DAT-QUERY-ENUMS", "DAT-CUTOFF-RANGE", "DAT-MACHINES-PER-EPISODE"):
                problems[cid].append(f"{split}/queries.csv unavailable")
            problems["DAT-HISTORY-CAUSAL"].append(f"{split}/queries.csv unavailable")
            problems["GEN-HISTORY-SHAPE"].append(f"{split}/queries.csv unavailable")
        else:
            _check_queries(split, queries, problems)
            split_queries[split] = queries
            if history is None:
                problems["GEN-HISTORY-SHAPE"].append(f"{split}/history.csv unavailable")
            else:
                _check_history(split, queries, history, problems, shape_counts)
        if history is None:
            problems["DAT-HISTORY-CAUSAL"].append(f"{split}/history.csv unavailable")
        for name, column in (("labels.csv", "outcome"), ("targets.csv", "outcome")):
            for row in table.get(name, []):
                if _as_int(row[column]) not in PAYOUTS:
                    problems["DAT-PAYOUT-SET"].append(f"{split}/{name} {row['query_id']}: outcome {row[column]!r}")
        conditions = table.get("conditions.csv")
        if conditions is not None and queries is not None:
            _check_conditions(split, queries, conditions, meta, problems)
        if split in LABELED_SPLITS and conditions is None:
            problems["DAT-COUNTS-PER-CONDITION"].append(f"{split}/conditions.csv unavailable")

    # Episode counts for test (no conditions file): 4 x counts_per_condition.test.
    counts = meta.get("counts_per_condition") or {}
    if "test" in split_queries and isinstance(counts.get("test"), int):
        episodes = {row["episode_id"] for row in split_queries["test"]}
        if len(episodes) != len(CONDITIONS) * counts["test"]:
            problems["DAT-COUNTS-PER-CONDITION"].append(
                f"test has {len(episodes)} episodes, metadata implies {len(CONDITIONS) * counts['test']}")

    _check_splits_disjoint(split_queries, problems)

    test_dir = root / "test"
    if test_dir.is_dir():
        for name in ("labels.csv", "conditions.csv", "targets.csv", "reference.csv"):
            if (test_dir / name).exists():
                problems["DAT-TEST-SPLIT-FILES"].append(f"test/{name} must not be present")
        for name in ("queries.csv", "history.csv"):
            if not (test_dir / name).is_file():
                problems["DAT-TEST-SPLIT-FILES"].append(f"test/{name} is missing")

    total_shape = {k: sum(v[k] for v in shape_counts.values()) for k in ("long", "short")}
    shape_details = {"per_split": shape_counts, **total_shape}
    results = []
    for cid in DATASET_CHECK_IDS:
        if cid == "GEN-HISTORY-SHAPE":
            results.append(_verdict(
                cid, problems[cid],
                f"every history matches a long or short shape: {total_shape['long']} long, "
                f"{total_shape['short']} short", **shape_details))
        elif cid == "CON-ROUND-METADATA":
            results.append(_verdict(cid, problems[cid],
                                    f"metadata.json consistent (round {meta.get('round')!r})"))
        else:
            results.append(_verdict(cid, problems[cid], f"{cid} holds"))
    return results


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------

def package_entry(entry_dir, out_zip) -> Path:
    """Package a code folder into a ZIP with the kit's rules (refuses an existing out_zip)."""
    return package_code(Path(entry_dir), Path(out_zip))


def _translate(message):
    for needle, check_id in _INSPECT_TRANSLATION:
        if needle in message:
            return check_id
    return "PKG-ZIP-SAFE"


def _member_bytes(archive, name):
    try:
        return archive.read(name)
    except (KeyError, RuntimeError, NotImplementedError, zipfile.BadZipFile, OSError):
        return None


def _parse_config(data):
    """Return (config dict or None, problems list) for reproduce.json bytes."""
    if data is None:
        return None, ["reproduce.json is not at the ZIP root"]
    problems = []
    if len(data) > JSON_MAX_BYTES:
        problems.append(f"reproduce.json is {len(data)} bytes, limit {JSON_MAX_BYTES}")
    try:
        config = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        return None, problems + [f"reproduce.json is not valid JSON: {exc}"]
    if not isinstance(config, dict):
        return None, problems + ["reproduce.json must be a JSON object"]
    if set(config) != {"image", "command"}:
        problems.append(f"reproduce.json keys must be exactly image and command; got {sorted(config)}")
    command = config.get("command")
    if not isinstance(command, list) or not 1 <= len(command) <= COMMAND_MAX_ARGS or not all(
            isinstance(x, str) and x and "\x00" not in x and len(x) <= COMMAND_MAX_CHARS for x in command):
        problems.append(f"command must be a list of 1-{COMMAND_MAX_ARGS} non-empty strings "
                        f"of at most {COMMAND_MAX_CHARS} characters")
    return config, problems


def _zip_unreadable(reason):
    results = []
    for cid in ZIP_CHECK_IDS:
        results.append(_fail(cid, f"cannot evaluate: {reason}"))
    return results


def check_zip(zip_path, metadata_path=None) -> list[CheckResult]:
    """Run the PKG-* and CON-REPRODUCE-SCHEMA / CON-IMAGE-PYTHON checks on a packaged code ZIP.

    metadata_path (the round's metadata.json) enables PKG-README-SEASON; without it that
    check passes with an info message saying it was not checked.
    """
    path = Path(zip_path)
    try:
        content = path.read_bytes()
    except OSError as exc:
        return _zip_unreadable(f"cannot read ZIP: {exc}")

    meta, meta_error = None, None
    if metadata_path is not None:
        try:
            meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                raise ValueError("not a JSON object")
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            meta_error = f"metadata unreadable: {exc}"

    size_ok = len(content) <= ZIP_MAX_BYTES
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        results = []
        for cid in ZIP_CHECK_IDS:
            if cid == "PKG-ZIP-SIZE":
                results.append(_verdict(cid, [] if size_ok else [f"{len(content)} bytes"],
                                        f"{len(content)} bytes <= {ZIP_MAX_BYTES}"))
            elif cid == "PKG-ZIP-SAFE":
                results.append(_fail(cid, f"not a ZIP archive: {exc}"))
            else:
                results.append(_fail(cid, "cannot evaluate: not a ZIP archive"))
        return results

    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        files = [info for info in infos if not info.is_dir()]
        file_names = {info.filename for info in files}
        declared = sum(info.file_size for info in infos)

        inspect_error = None
        try:
            inspect_archive(content)
        except ValidationError as exc:
            inspect_error = str(exc)
        translated = _translate(inspect_error) if inspect_error else None

        # PKG-ZIP-SIZE / PKG-ZIP-LIMITS
        size_problems = [] if size_ok else [f"{len(content)} bytes > {ZIP_MAX_BYTES}"]
        if translated == "PKG-ZIP-SIZE":
            size_problems.append(inspect_error)
        results = [_verdict("PKG-ZIP-SIZE", size_problems, f"{len(content)} bytes <= {ZIP_MAX_BYTES}")]
        limit_problems = []
        if len(infos) > ZIP_MAX_ENTRIES:
            limit_problems.append(f"{len(infos)} entries > {ZIP_MAX_ENTRIES}")
        if declared > ZIP_MAX_UNCOMPRESSED:
            limit_problems.append(f"{declared} declared bytes > {ZIP_MAX_UNCOMPRESSED}")
        if translated == "PKG-ZIP-LIMITS":
            limit_problems.append(inspect_error)
        results.append(_verdict("PKG-ZIP-LIMITS", limit_problems,
                                f"{len(infos)} entries, {declared} declared bytes within limits"))

        # PKG-ZIP-SAFE: inspect_archive does the safety rules and CRC/size streaming; testzip re-verifies CRCs
        # because inspect stops at its first error and may not have reached the streaming pass.
        safe_problems = []
        if translated == "PKG-ZIP-SAFE":
            safe_problems.append(inspect_error)
        try:
            bad_member = archive.testzip()
        except (RuntimeError, zipfile.BadZipFile, NotImplementedError, OSError, ValueError) as exc:
            bad_member = f"unreadable member: {exc}"
        if bad_member:
            safe_problems.append(f"member CRC or size check failed: {bad_member}")
        results.append(_verdict("PKG-ZIP-SAFE", safe_problems, "archive is safe (paths, links, CRC, sizes)"))

        # PKG-ZIP-ROOT
        root_problems = []
        if translated == "PKG-ZIP-ROOT":
            root_problems.append(inspect_error)
        for required in ("README.md", "reproduce.json"):
            if required not in file_names:
                root_problems.append(f"{required} is not a top-level member")
            else:
                info = next(i for i in files if i.filename == required)
                if info.file_size > JSON_MAX_BYTES:
                    root_problems.append(f"{required} is {info.file_size} bytes > {JSON_MAX_BYTES}")
        readme = _member_bytes(archive, "README.md") if "README.md" in file_names else None
        if readme is not None:
            try:
                if not readme.decode("utf-8").strip():
                    root_problems.append("README.md is empty")
            except UnicodeDecodeError:
                root_problems.append("README.md is not UTF-8")
        results.append(_verdict("PKG-ZIP-ROOT", root_problems,
                                "README.md and reproduce.json are top-level and within size"))

        # PKG-REQUIREMENTS-PINNED
        pin_problems = []
        if "requirements.txt" in file_names:
            data = _member_bytes(archive, "requirements.txt") or b""
            try:
                lines = data.decode("utf-8").splitlines()
            except UnicodeDecodeError:
                lines = None
                pin_problems.append("requirements.txt is not UTF-8")
            for number, raw in enumerate(lines or [], 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if not REQUIREMENT_RE.fullmatch(line):
                    pin_problems.append(f"requirements.txt line {number}: {line!r} is not an exact name==version pin")
            ok_message = "requirements.txt lines are exact pins"
        else:
            ok_message = "no root requirements.txt (standard library only)"
        results.append(_verdict("PKG-REQUIREMENTS-PINNED", pin_problems, ok_message))

        # PKG-NO-ANSWER-FILES
        answer_problems = []
        for name in names:
            parts = PurePosixPath(name).parts
            if (parts and parts[-1] in ANSWER_BASENAMES) or "_truth" in parts:
                answer_problems.append(f"{name} is an answer file or inside _truth")
        results.append(_verdict("PKG-NO-ANSWER-FILES", answer_problems, "no answer files in the ZIP"))

        # PKG-NO-DATA-COPY (warn)
        copy_hits = []
        for name in names:
            parts = PurePosixPath(name).parts
            if (parts and parts[-1] in ("queries.csv", "history.csv", "sample_submission.csv")) or "public" in parts[:-1]:
                copy_hits.append(name)
        if copy_hits:
            results.append(_fail("PKG-NO-DATA-COPY",
                                 f"ZIP appears to contain organizer data: {copy_hits[:5]}", severity="warn",
                                 count=len(copy_hits)))
        else:
            results.append(_pass("PKG-NO-DATA-COPY", "no copied organizer data files"))

        # PKG-README-SEASON (warn)
        if metadata_path is None:
            results.append(_pass("PKG-README-SEASON", "no metadata given; season and round not checked"))
        elif meta_error or meta is None:
            results.append(_fail("PKG-README-SEASON", meta_error or "metadata unreadable", severity="warn"))
        else:
            season = str(meta.get("season_id", ""))
            round_value = str(meta.get("round", ""))
            text = readme.decode("utf-8", "replace") if readme is not None else ""
            missing = []
            if not text:
                missing.append("README.md missing")
            if season and season not in text:
                missing.append(f"season_id {season} not in README.md")
            if round_value and not re.search(rf"\b{re.escape(round_value)}\b", text, re.IGNORECASE):
                missing.append(f"round name {round_value!r} not in README.md")
            if missing:
                results.append(_fail("PKG-README-SEASON", "; ".join(missing), severity="warn"))
            else:
                results.append(_pass("PKG-README-SEASON", "README.md names the season and round"))

        # CON-REPRODUCE-SCHEMA and CON-IMAGE-PYTHON
        config, config_problems = _parse_config(_member_bytes(archive, "reproduce.json")
                                                if "reproduce.json" in file_names else None)
        schema_problems = list(config_problems)
        if translated == "CON-REPRODUCE-SCHEMA":
            schema_problems.append(inspect_error)
        results.append(_verdict("CON-REPRODUCE-SCHEMA", schema_problems,
                                "reproduce.json has exactly image and command, valid argv"))
        image_problems = []
        if config is None:
            image_problems.append("reproduce.json unreadable, image not verified")
        elif config.get("image") != "python":
            image_problems.append(f"image must be 'python', got {config.get('image')!r}")
        if translated == "CON-IMAGE-PYTHON":
            image_problems.append(inspect_error)
        results.append(_verdict("CON-IMAGE-PYTHON", image_problems, "image is 'python'"))
    return results


# ---------------------------------------------------------------------------
# Mounted inputs
# ---------------------------------------------------------------------------

def check_mount_inputs(data_root, entry_dir) -> list[CheckResult]:
    """Check what the organizer would mount: the dataset as /data and the entry folder as /submission.

    ROB-MOUNT-NO-ANSWERS: no answer file under test/ and no _truth directory anywhere in the
    mounted dataset. Train and validation answers are expected to be present (they are training
    data), so only the scored split and the private truth tree are checked.
    ROB-NO-FORECAST-INPUT: no precomputed submission.csv or validation_predictions.csv in the entry
    folder or the dataset. /output starts empty by construction (a fresh tmpfs in docker mode).
    """
    root, entry = Path(data_root), Path(entry_dir)
    if not root.is_dir() or not entry.is_dir():
        reason = f"not a directory: {root if not root.is_dir() else entry}"
        return _unevaluated(MOUNT_CHECK_IDS, reason)

    answer_problems = []
    test_dir = root / "test"
    if test_dir.is_dir():
        for path in sorted(test_dir.rglob("*")):
            if path.is_file() and path.name in ANSWER_BASENAMES:
                answer_problems.append(f"{path.relative_to(root)} would be visible to the entry")
    else:
        answer_problems.append("test/ is missing from the mounted dataset")
    for path in sorted(root.rglob("*")):
        if "_truth" in path.relative_to(root).parts:
            answer_problems.append(f"{path.relative_to(root)} is private truth")

    forecast_problems = []
    for base in (entry, root):
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.name in ("submission.csv", "validation_predictions.csv"):
                forecast_problems.append(f"{path} is a precomputed forecast")

    return [
        _verdict("ROB-MOUNT-NO-ANSWERS", answer_problems, "no answer files under test/ and no _truth"),
        _verdict("ROB-NO-FORECAST-INPUT", forecast_problems,
                 "no precomputed submission.csv or validation_predictions.csv"),
    ]
