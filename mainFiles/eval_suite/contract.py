"""Submission contract checks: forecast CSV, entry directory (code ZIP root), runtime, determinism.

Every check returns a CheckResult with a stable id. check_csv and check_entry_dir return
lists; the others return one CheckResult. Packaging reuses the kit's
slot_forecasting.submission_contract so the ZIP rules are the organizers' rules.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

_KIT_ROOT = Path(__file__).resolve().parent.parent
if str(_KIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_KIT_ROOT))

from slot_forecasting.schema import ValidationError  # noqa: E402
from slot_forecasting.submission_contract import inspect_archive, package_code  # noqa: E402

from .entry import TIMEOUT_EXIT_CODE, RunResult  # noqa: E402

CSV_COLUMNS = ("query_id", "p_0", "p_1", "p_2", "p_5", "p_20")
PROB_COLUMNS = CSV_COLUMNS[1:]
MAX_CSV_BYTES = 10_000_000
MAX_ZIP_BYTES = 100_000_000
RUNTIME_LIMIT_S = 1800
SUM_TOL = 1e-6
REQUIRED_OUTPUT = "/output/submission.csv"
_PIN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9._,-]+\])?==[A-Za-z0-9][A-Za-z0-9._+!-]*")


@dataclass
class CheckResult:
    id: str
    passed: bool
    severity: str  # "fail" | "warn" | "info"
    message: str
    details: dict = field(default_factory=dict)


def _ok(check_id, message, **details) -> CheckResult:
    return CheckResult(check_id, True, "info", message, details)


def _bad(check_id, message, severity="fail", **details) -> CheckResult:
    return CheckResult(check_id, False, severity, message, details)


def _read_table(path):
    """Header and non-blank rows of a UTF-8 CSV. Raises on undecodable or empty input."""
    text = Path(path).read_bytes().decode("utf-8-sig")
    rows = [row for row in csv.reader(io.StringIO(text, newline="")) if row]
    if not rows:
        raise ValueError("file has no header")
    return rows[0], rows[1:]


def _query_ids(queries_csv) -> list[str]:
    with Path(queries_csv).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "query_id" not in reader.fieldnames:
            raise ValueError(f"{queries_csv} has no query_id column")
        return [row["query_id"] for row in reader]


def _parse_values(row) -> tuple[float, ...] | None:
    """Five floats (may be non-finite), or None when any cell is not a number."""
    try:
        return tuple(float(v) for v in row[1:])
    except ValueError:
        return None


def check_csv(csv_path, queries_csv) -> list[CheckResult]:
    """Format and content checks on a forecast CSV against the released query list."""
    csv_path, queries_csv = Path(csv_path), Path(queries_csv)
    if not csv_path.is_file():
        return [_bad("CON-CSV-COLUMNS", f"forecast CSV not found: {csv_path}")]
    results = []
    size = csv_path.stat().st_size
    if size <= MAX_CSV_BYTES:
        results.append(_ok("CON-CSV-SIZE", f"{size} bytes <= {MAX_CSV_BYTES}", bytes=size))
    else:
        results.append(_bad("CON-CSV-SIZE", f"{size} bytes exceeds {MAX_CSV_BYTES}", bytes=size))

    try:
        header, rows = _read_table(csv_path)
        expected_ids = _query_ids(queries_csv)
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        results.append(_bad("CON-CSV-COLUMNS", f"unreadable CSV: {exc}"))
        return results

    short = [i + 2 for i, row in enumerate(rows) if len(row) != len(CSV_COLUMNS)]
    if tuple(header) == CSV_COLUMNS and not short:
        results.append(_ok("CON-CSV-COLUMNS", f"columns are {','.join(CSV_COLUMNS)}"))
    else:
        results.append(_bad("CON-CSV-COLUMNS",
                            f"columns must be exactly {','.join(CSV_COLUMNS)}; got {header}"
                            + (f"; rows with wrong field count at lines {short[:5]}" if short else ""),
                            header=list(header)))
        skipped = "skipped: CSV columns invalid"
        for check_id in ("CON-CSV-COVERAGE", "CON-CSV-FINITE", "CON-CSV-RANGE", "CON-CSV-SUM"):
            results.append(_bad(check_id, skipped))
        return results

    counts: dict[str, int] = {}
    for row in rows:
        counts[row[0]] = counts.get(row[0], 0) + 1
    expected = set(expected_ids)
    missing = sorted(expected - set(counts))
    extra = sorted(set(counts) - expected)
    duplicated = sorted(q for q, c in counts.items() if c > 1)
    if not (missing or extra or duplicated):
        results.append(_ok("CON-CSV-COVERAGE", f"all {len(expected)} queries exactly once"))
    else:
        results.append(_bad("CON-CSV-COVERAGE",
                            f"missing {len(missing)} {missing[:3]}, extra {len(extra)} {extra[:3]}, "
                            f"duplicated {len(duplicated)} {duplicated[:3]}",
                            missing=len(missing), extra=len(extra), duplicated=len(duplicated)))

    parsed = [(row[0], _parse_values(row)) for row in rows]
    not_numeric = [q for q, v in parsed if v is None]
    nonfinite = [q for q, v in parsed if v is not None and not all(math.isfinite(x) for x in v)]
    if not (not_numeric or nonfinite):
        results.append(_ok("CON-CSV-FINITE", "all probabilities are finite numbers"))
    else:
        results.append(_bad("CON-CSV-FINITE",
                            f"{len(not_numeric)} non-numeric rows, {len(nonfinite)} non-finite rows",
                            examples=(not_numeric + nonfinite)[:5]))

    finite_rows = [(q, v) for q, v in parsed if v is not None and all(math.isfinite(x) for x in v)]
    out_of_range = [q for q, v in finite_rows if any(x < 0 or x > 1 for x in v)]
    if not out_of_range:
        results.append(_ok("CON-CSV-RANGE", "all probabilities in [0, 1]"))
    else:
        results.append(_bad("CON-CSV-RANGE", f"{len(out_of_range)} rows outside [0, 1]",
                            examples=out_of_range[:5]))

    sums = [(q, math.fsum(v)) for q, v in finite_rows]
    bad_sums = [(q, s) for q, s in sums if abs(s - 1.0) > SUM_TOL]
    if not bad_sums:
        worst = max((abs(s - 1.0) for _, s in sums), default=0.0)
        results.append(_ok("CON-CSV-SUM", f"row sums within {SUM_TOL} of 1 (worst {worst:.3g})"))
    else:
        results.append(_bad("CON-CSV-SUM", f"{len(bad_sums)} rows sum outside 1 +/- {SUM_TOL}",
                            examples=[(q, s) for q, s in bad_sums[:5]]))
    return results


def _requirement_problems(path: Path) -> list[str]:
    problems = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split(" #")[0].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-") or not _PIN_RE.fullmatch(line):
            problems.append(f"line {number}: {line!r} is not an exact pin (name==version)")
    return problems


def check_entry_dir(entry_dir) -> list[CheckResult]:
    """Entry layout and reproduce.json rules, plus the packaged ZIP size and archive rules."""
    entry = Path(entry_dir)
    results = []

    missing = [name for name in ("README.md", "reproduce.json") if not (entry / name).is_file()]
    if missing:
        results.append(_bad("CON-ZIP-ROOT", f"missing at entry root: {', '.join(missing)}"))
    else:
        results.append(_ok("CON-ZIP-ROOT", "README.md and reproduce.json at entry root"))

    config = None
    if "reproduce.json" not in missing:
        try:
            config = json.loads((entry / "reproduce.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            results.append(_bad("CON-IMAGE", f"reproduce.json is not valid JSON: {exc}"))
            results.append(_bad("CON-OUTPUT-PATH", "skipped: reproduce.json unreadable"))
    if config is not None and isinstance(config, dict):
        if set(config) != {"image", "command"}:
            results.append(_bad("CON-IMAGE", f"reproduce.json keys must be exactly image, command; got {sorted(config)}"))
        elif config["image"] != "python":
            results.append(_bad("CON-IMAGE", f"image must be 'python', got {config['image']!r}"))
        else:
            results.append(_ok("CON-IMAGE", "image is 'python'"))
        command = config.get("command")
        if not isinstance(command, list) or not 1 <= len(command) <= 64 or not all(
                isinstance(x, str) and x and "\x00" not in x and len(x) <= 4096 for x in command):
            results.append(_bad("CON-OUTPUT-PATH", "command must be a list of 1-64 nonempty strings"))
        elif not any(arg == REQUIRED_OUTPUT or arg == f"--out={REQUIRED_OUTPUT}" for arg in command):
            results.append(_bad("CON-OUTPUT-PATH",
                                f"command must write {REQUIRED_OUTPUT} (pass it as an argument)",
                                command=command))
        else:
            results.append(_ok("CON-OUTPUT-PATH", f"command writes {REQUIRED_OUTPUT}"))
    elif config is not None:
        results.append(_bad("CON-IMAGE", "reproduce.json must be a JSON object"))
        results.append(_bad("CON-OUTPUT-PATH", "reproduce.json must be a JSON object"))

    requirements = entry / "requirements.txt"
    if not requirements.exists():
        results.append(_ok("CON-REQ-PINNED", "no requirements.txt: standard library only"))
    else:
        problems = _requirement_problems(requirements)
        if problems:
            results.append(_bad("CON-REQ-PINNED", f"{len(problems)} unpinned lines", examples=problems[:5]))
        else:
            results.append(_ok("CON-REQ-PINNED", "every requirements.txt line is an exact pin"))

    if missing:
        results.append(_bad("CON-ZIP-SIZE", "skipped: entry root incomplete"))
        return results
    with tempfile.TemporaryDirectory(prefix="eval-zip-") as scratch:
        zip_path = Path(scratch) / "entry.zip"
        try:
            package_code(entry, zip_path)
            content = zip_path.read_bytes()
            inspect_archive(content)
        except (ValidationError, OSError, ValueError) as exc:
            results.append(_bad("CON-ZIP-ROOT", f"code ZIP would be rejected: {exc}"))
            results.append(_bad("CON-ZIP-SIZE", "skipped: packaging failed"))
            return results
    results.append(_ok("CON-ZIP-ROOT", "packaged ZIP opens with README.md and reproduce.json at root"))
    if len(content) <= MAX_ZIP_BYTES:
        results.append(_ok("CON-ZIP-SIZE", f"ZIP {len(content)} bytes <= {MAX_ZIP_BYTES}", bytes=len(content)))
    else:
        results.append(_bad("CON-ZIP-SIZE", f"ZIP {len(content)} bytes exceeds {MAX_ZIP_BYTES}",
                            bytes=len(content)))
    return results


def check_output_written(run_result: RunResult) -> CheckResult:
    """The run produced /output/submission.csv (the file, not just a clean exit)."""
    if run_result.csv_path is not None:
        return _ok("CON-OUTPUT-PATH", f"output written: {run_result.csv_path.name}")
    return _bad("CON-OUTPUT-PATH", "command finished without writing /output/submission.csv")


def check_runtime(run_result: RunResult, limit_s: float = RUNTIME_LIMIT_S) -> CheckResult:
    """Clean exit within limit_s seconds of wall time."""
    if run_result.exit_code == TIMEOUT_EXIT_CODE:
        return _bad("CON-RUNTIME", f"killed at the {limit_s}s limit or exited 124",
                    wall_s=run_result.wall_s, limit_s=limit_s)
    if run_result.exit_code != 0:
        return _bad("CON-RUNTIME", f"exit code {run_result.exit_code}",
                    stderr_tail=run_result.stderr_tail[-500:])
    if run_result.wall_s > limit_s:
        return _bad("CON-RUNTIME", f"wall {run_result.wall_s:.1f}s exceeds {limit_s}s",
                    wall_s=run_result.wall_s, limit_s=limit_s)
    return _ok("CON-RUNTIME", f"exit 0 in {run_result.wall_s:.1f}s (limit {limit_s}s)",
               wall_s=run_result.wall_s, peak_mem_mb=run_result.peak_mem_mb)


def _renormalised(path) -> dict[str, tuple[float, ...]]:
    header, rows = _read_table(path)
    if tuple(header) != CSV_COLUMNS:
        raise ValueError(f"{path} columns are not {','.join(CSV_COLUMNS)}")
    result = {}
    for row in rows:
        values = _parse_values(row)
        if values is None or not all(math.isfinite(x) for x in values):
            raise ValueError(f"{path}: non-finite or non-numeric row {row[0]!r}")
        total = math.fsum(values)
        if total <= 0:
            raise ValueError(f"{path}: row {row[0]!r} sums to {total}")
        result[row[0]] = tuple(x / total for x in values)
    return result


def check_determinism(csv_a, csv_b, tol: float = 1e-8) -> CheckResult:
    """Two runs agree: same query ids, max abs difference <= tol after renormalising each row."""
    try:
        first, second = _renormalised(csv_a), _renormalised(csv_b)
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        return _bad("CON-DETERMINISM", f"cannot compare runs: {exc}")
    if first.keys() != second.keys():
        return _bad("CON-DETERMINISM", "the two runs cover different query ids",
                    only_first=len(first.keys() - second.keys()), only_second=len(second.keys() - first.keys()))
    worst = max((abs(a - b) for q in first for a, b in zip(first[q], second[q])), default=0.0)
    if worst <= tol:
        return _ok("CON-DETERMINISM", f"max difference {worst:.3g} <= {tol:g}", max_abs_diff=worst)
    return _bad("CON-DETERMINISM", f"max difference {worst:.3g} exceeds {tol:g}", max_abs_diff=worst, tol=tol)
