"""Real-data protocol and entry-independent probes used by the suite runner.

k-fold validation: fold i mounts a dataset whose ``validation/`` keeps only the
NON-scored validation episodes (their answers stay, as training data would), and
whose ``test/`` holds only the scored episodes' queries and history. The scored
answers live in a sibling ``<fold>-truth/_truth`` directory that is never inside
the mounted root. Putting ``_truth`` under the mount would let the entry read the
very answers it is scored on (ROB-MOUNT-NO-ANSWERS), so that layout is refused.

Everything here is independent of any team model: it only copies files, reads
CSVs and calls the kit's scorer through eval_suite.scoring.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from . import scoring as S
from .contract import CheckResult

KIT_ROOT = Path(__file__).resolve().parent.parent
PREDICTION_HEADER = list(S.PREDICTION_COLUMNS)
PAYOUTS = (0, 1, 2, 5, 20)
ANSWER_NAMES = ("targets.csv", "labels.csv", "conditions.csv", "reference.csv")
FORECAST_NAMES = ("submission.csv", "validation_predictions.csv")
PRESETS = ("main_like", "round2_shift", "all_lags", "neighbour_heavy",
           "hidden_state_heavy", "fast_drift", "edge_cases")
VALIDATION_ANSWER_FILES = ("targets.csv", "conditions.csv", "reference.csv")


class GeneratorError(RuntimeError):
    """The data generator is missing or failed."""


@dataclass(frozen=True)
class Fold:
    index: int
    mount: Path          # dataset root mounted as /data; never contains _truth
    truth_dir: Path      # the _truth directory (holds test/targets.csv etc.)
    scored_episodes: tuple
    scored_query_ids: tuple


# ------------------------------------------------------------------ file helpers

def _read_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path}: empty file")
        rows = [row for row in reader if row]
    return header, rows


def _write_rows(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _filter_file(src, dst, keep):
    """Copy a CSV keeping rows for which keep(row_dict) is true; header unchanged."""
    header, rows = _read_rows(src)
    _write_rows(dst, header, [row for row in rows if keep(dict(zip(header, row)))])


def _keep_in(column, values):
    values = frozenset(values)
    return lambda row: row[column] in values


def _is_fresh(path: Path):
    if path.exists():
        shutil.rmtree(path)
    return path


# ------------------------------------------------------------------ mounts

def build_kit_test_mount(kit, dst) -> Path:
    """Organizer-style mount of the kit: train and test with metadata, no validation answers."""
    kit, dst = Path(kit), _is_fresh(Path(dst))
    dst.mkdir(parents=True)
    shutil.copy(kit / "metadata.json", dst / "metadata.json")
    shutil.copytree(kit / "train", dst / "train")
    shutil.copytree(kit / "test", dst / "test")
    return dst


def strip_truth_copy(src, dst) -> Path:
    """Copy a dataset without its `_truth` answers, for mounting into an entry."""
    src, dst = Path(src), _is_fresh(Path(dst))
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("_truth"))
    return dst


def build_fold_datasets(kit_root, out_dir, *, k=4, seed=S.DEFAULT_SEED) -> list[Fold]:
    """Write one mounted dataset and one truth directory per validation fold."""
    kit, out = Path(kit_root), Path(out_dir)
    validation = kit / "validation"
    folds = []
    for meta in S.kfold_validation_protocol(validation, k=k, seed=seed):
        index = meta["fold"]
        scored_eps, visible_eps = set(meta["scored_episodes"]), set(meta["visible_episodes"])
        scored_q, visible_q = set(meta["scored_query_ids"]), set(meta["visible_query_ids"])
        mount = _is_fresh(out / f"fold{index}")
        truth = _is_fresh(out / f"fold{index}-truth") / "_truth"

        mount.mkdir(parents=True)
        shutil.copy(kit / "metadata.json", mount / "metadata.json")
        shutil.copytree(kit / "train", mount / "train")

        # Visible validation: history by episode, every query-keyed file by query.
        for src in sorted(validation.glob("*.csv")):
            header, _ = _read_rows(src)
            if "query_id" in header:
                keep = _keep_in("query_id", visible_q)
            elif "episode_id" in header:
                keep = _keep_in("episode_id", visible_eps)
            else:
                raise ValueError(f"{src.name} has neither query_id nor episode_id")
            _filter_file(src, mount / "validation" / src.name, keep)

        # Scored episodes: queries and history only, no answers.
        _filter_file(validation / "queries.csv", mount / "test" / "queries.csv",
                     _keep_in("query_id", scored_q))
        _filter_file(validation / "history.csv", mount / "test" / "history.csv",
                     _keep_in("episode_id", scored_eps))

        for name in VALIDATION_ANSWER_FILES:
            _filter_file(validation / name, truth / "test" / name, _keep_in("query_id", scored_q))

        folds.append(Fold(index, mount, truth, tuple(sorted(scored_eps)), tuple(sorted(scored_q))))
    return folds


# ------------------------------------------------------------------ stitching and scoring

def stitch(parts, out_path) -> Path:
    """Concatenate per-fold prediction CSVs into one validation-wide CSV."""
    rows = {}
    for part in parts:
        header, body = _read_rows(part)
        if header != PREDICTION_HEADER:
            raise ValueError(f"{part}: header is not {PREDICTION_HEADER}")
        for row in body:
            if row[0] in rows:
                raise ValueError(f"query_id {row[0]} appears in more than one fold")
            rows[row[0]] = row
    _write_rows(out_path, PREDICTION_HEADER, [rows[q] for q in sorted(rows)])
    return Path(out_path)


def score_bundle(pred_csv, split, *, history_path, max_points, seed, repeats, n_splits=20) -> dict:
    """Official points, condition ratios, bootstrap CI, half-split spread and slices."""
    args = (pred_csv, split.targets, split.queries, split.conditions, split.reference, max_points)
    official = S.official_score(*args)
    ci = S.bootstrap_ci(*args, repeats=repeats, seed=seed)
    halves = S.repeated_half_splits(*args, n_splits=n_splits, seed=seed)
    slices = S.slices(pred_csv, split.targets, split.queries, split.conditions, split.reference,
                      history=history_path)
    return {
        "points": official["points"], "R": official["R"], "E": official["E"], "B": official["B"],
        "ratio": official["ratio"], "n_episodes": official["n_episodes"], "max_points": max_points,
        "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "ci_level": ci["level"],
        "bootstrap_repeats": repeats, "bootstrap_sd": ci["bootstrap"]["sd"],
        "half_splits": {"n_splits": n_splits, "full_points": halves["full_points"],
                        "half_a": halves["half_a"], "half_b": halves["half_b"]},
        "slices": slices,
    }


def paired(pred_a, pred_b, split, *, max_points, repeats, seed) -> dict:
    result = S.paired_compare(pred_a, pred_b, split.targets, split.queries, split.conditions,
                              split.reference, max_points, repeats=repeats, seed=seed)
    return {k: result[k] for k in ("points_a", "points_b", "mean_diff", "ci_low", "ci_high",
                                   "share_a_better", "repeats", "seed")}


def independent_points(pred, targets, queries, conditions, reference, max_points) -> float:
    """SCO-FORMULA: the RULES.md formula written out directly, without scoring.py's helpers."""
    members = defaultdict(list)
    condition_of = {}
    for qid in targets:
        episode = queries[qid]["episode_id"]
        members[episode].append(qid)
        condition_of[episode] = conditions[qid]

    def episode_error(source, episode):
        return math.fsum(
            math.fsum((p - t) ** 2 for p, t in zip(source[q], targets[q])) for q in members[episode]
        ) / len(members[episode])

    e_by, b_by = defaultdict(list), defaultdict(list)
    for episode in members:
        e_by[condition_of[episode]].append(episode_error(pred, episode))
        b_by[condition_of[episode]].append(episode_error(reference, episode))
    ratios = []
    for condition in S.CONDITIONS:
        e_mean = math.fsum(e_by[condition]) / len(e_by[condition])
        b_mean = math.fsum(b_by[condition]) / len(b_by[condition])
        ratios.append(e_mean / max(b_mean, S.REFERENCE_FLOOR))
    return max_points / (1 + math.fsum(ratios) / len(ratios))


def scoring_rule_checks(kit, split, max_points, metadata) -> list[CheckResult]:
    """Scorer rules checked on the real validation split (no entry involved)."""
    results = []
    expected = {"main": 60, "adaptation": 40}.get(metadata.get("round"))
    if expected == max_points:
        results.append(CheckResult("SCO-MAX-POINTS", True, "info",
                                   f"round {metadata.get('round')} max_points {max_points}", {}))
    else:
        results.append(CheckResult("SCO-MAX-POINTS", False, "fail",
                                   f"metadata max_points {max_points} != {expected} for round "
                                   f"{metadata.get('round')!r}", {}))

    truth = S.official_score(split.targets, split.targets, split.queries, split.conditions,
                             split.reference, max_points)
    results.append(_numeric("SCO-TRUTH-MAX", truth["points"], max_points, "truth as prediction"))
    half = S.official_score(split.reference, split.targets, split.queries, split.conditions,
                            split.reference, max_points)
    results.append(_numeric("SCO-REF-HALF", half["points"], max_points / 2, "frozen reference as prediction"))

    ok_formula = independent_points(split.reference, split.targets, split.queries, split.conditions,
                                    split.reference, max_points)
    results.append(_numeric("SCO-FORMULA", half["points"], ok_formula, "independent formula vs scorer",
                            tol=1e-9))

    moved = []
    for condition in S.CONDITIONS:
        pred = dict(split.reference)
        for qid, meta in split.queries.items():
            if split.conditions[qid] == condition:
                pred[qid] = (0.2, 0.2, 0.2, 0.2, 0.2)
        ratios = S.official_score(pred, split.targets, split.queries, split.conditions,
                                  split.reference, max_points)["ratio"]
        others = [abs(ratios[c] - 1.0) for c in S.CONDITIONS if c != condition]
        moved.append(abs(ratios[condition] - 1.0) > 1e-9 and max(others) <= 1e-12)
    results.append(CheckResult("SCO-CONDITION-EQUAL", all(moved), "fail" if not all(moved) else "info",
                               "changing one condition's predictions moves only that ratio" if all(moved)
                               else "a condition's predictions move other conditions' ratios", {}))
    return results


def _numeric(check_id, actual, expected, label, tol=1e-9):
    diff = abs(actual - expected)
    ok = diff <= tol
    return CheckResult(check_id, ok, "info" if ok else "fail",
                       f"{label}: {actual:.12g} vs expected {expected:.12g} (diff {diff:.3g})",
                       {"actual": actual, "expected": expected})


def reference_reproduction(kit, split) -> list[CheckResult]:
    """SCO-REFERENCE-REPRO / SCO-REFERENCE-SOURCE: refit the pooled starter on train only."""
    try:
        fitted = S.reference_from_starter(kit)
    except Exception as exc:  # the starter is kit code; report rather than crash the suite
        message = f"could not refit the kit starter: {type(exc).__name__}: {exc}"
        return [CheckResult("SCO-REFERENCE-REPRO", False, "warn", message, {}),
                CheckResult("SCO-REFERENCE-SOURCE", False, "warn", message, {})]
    diff = max(max(abs(a - b) for a, b in zip(split.reference[q], fitted[q])) for q in split.reference)
    ok = diff <= 1e-9
    repro = CheckResult("SCO-REFERENCE-REPRO", ok, "info" if ok else "warn",
                        f"starter refit on train reproduces reference.csv within {diff:.3g}",
                        {"max_abs_diff": diff})
    source = CheckResult("SCO-REFERENCE-SOURCE", ok, "info" if ok else "warn",
                         "validation/reference.csv is the frozen pooled starter output"
                         if ok else "reference.csv is not the starter refit on train", {})
    return [repro, source]


def kit_points_check(pred_csv, kit, ours) -> CheckResult:
    """SCO-MATCHES-KIT: the suite's points equal check_submission's (rounds.score_round)."""
    if str(KIT_ROOT) not in sys.path:
        sys.path.insert(0, str(KIT_ROOT))
    from slot_forecasting.rounds import score_round

    try:
        kit_points = score_round(Path(pred_csv), Path(kit))["technical_points"]
    except Exception as exc:
        return CheckResult("SCO-MATCHES-KIT", False, "fail", f"kit scorer rejected the file: {exc}", {})
    diff = abs(kit_points - ours)
    rounded_equal = round(kit_points, 6) == round(ours, 6)
    ok = diff <= 1e-9 and rounded_equal
    return CheckResult("SCO-MATCHES-KIT", ok, "info" if ok else "fail",
                       f"suite {ours:.9f} vs kit {kit_points:.9f} (diff {diff:.3g})",
                       {"suite": ours, "kit": kit_points})


# ------------------------------------------------------------------ format and mount checks

def check_strict_csv(path) -> list[CheckResult]:
    """CON-CSV-UTF8 (BOM is a warning) and CON-CSV-STRICT (exactly six non-empty fields per row)."""
    raw = Path(path).read_bytes()
    results = []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        results.append(CheckResult("CON-CSV-UTF8", False, "fail", f"not UTF-8: {exc}", {}))
        text = None
    if text is not None:
        if text.startswith("﻿"):
            results.append(CheckResult("CON-CSV-UTF8", True, "warn",
                                       "UTF-8 with a byte-order mark (accepted, but save without BOM)", {}))
        else:
            results.append(CheckResult("CON-CSV-UTF8", True, "info", "UTF-8", {}))
    if text is None:
        results.append(CheckResult("CON-CSV-STRICT", False, "fail", "skipped: not UTF-8", {}))
        return results
    rows = [row for row in csv.reader(text.splitlines()) if row]
    bad = [i + 1 for i, row in enumerate(rows) if len(row) != 6 or any(not cell.strip() for cell in row)]
    if bad:
        results.append(CheckResult("CON-CSV-STRICT", False, "fail",
                                   f"{len(bad)} rows are ragged or have empty cells", {"lines": bad[:5]}))
    else:
        results.append(CheckResult("CON-CSV-STRICT", True, "info", f"{len(rows)} rows, six fields each", {}))
    return results


def check_reproduce_schema(entry_dir) -> CheckResult:
    """CON-REPRODUCE-SCHEMA: exact keys, argv list of 1-64 strings, file size limit."""
    path = Path(entry_dir) / "reproduce.json"
    try:
        if path.stat().st_size > 65536:
            return CheckResult("CON-REPRODUCE-SCHEMA", False, "fail", "reproduce.json exceeds 65,536 bytes", {})
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return CheckResult("CON-REPRODUCE-SCHEMA", False, "fail", f"cannot read reproduce.json: {exc}", {})
    if not isinstance(config, dict) or set(config) != {"image", "command"}:
        return CheckResult("CON-REPRODUCE-SCHEMA", False, "fail", "keys must be exactly image and command", {})
    command = config["command"]
    if not isinstance(command, list) or not 1 <= len(command) <= 64:
        return CheckResult("CON-REPRODUCE-SCHEMA", False, "fail", "command must be an argv list of 1-64 strings", {})
    bad = [i for i, arg in enumerate(command)
           if not isinstance(arg, str) or not arg or "\x00" in arg or len(arg) > 4096]
    if bad:
        return CheckResult("CON-REPRODUCE-SCHEMA", False, "fail",
                           f"command arguments {bad[:5]} are empty, not strings, or too long", {})
    return CheckResult("CON-REPRODUCE-SCHEMA", True, "info", "command is an argv list of strings", {})


def mount_answer_check(mount) -> CheckResult:
    """ROB-MOUNT-NO-ANSWERS: no answer file or _truth at the mount root or in test/."""
    mount = Path(mount)
    found = []
    for folder in (mount, mount / "test"):
        if not folder.is_dir():
            continue
        for child in folder.iterdir():
            if child.name in ANSWER_NAMES or child.name == "_truth":
                found.append(str(child.relative_to(mount)))
    if found:
        return CheckResult("ROB-MOUNT-NO-ANSWERS", False, "fail", f"answer files visible to the entry: {found}",
                           {"found": found})
    return CheckResult("ROB-MOUNT-NO-ANSWERS", True, "info", "no answer files or _truth at the mount root or test/", {})


def forecast_input_check(entry_dir, mount) -> CheckResult:
    """ROB-NO-FORECAST-INPUT: no uploaded forecast sitting in the entry or the mount."""
    found = []
    for root in (Path(entry_dir), Path(mount)):
        for name in FORECAST_NAMES:
            found.extend(str(p) for p in root.rglob(name))
    if found:
        return CheckResult("ROB-NO-FORECAST-INPUT", False, "fail", f"forecast files present: {found[:5]}",
                           {"found": found[:10]})
    return CheckResult("ROB-NO-FORECAST-INPUT", True, "info", "no forecast file in the entry or the mount", {})


# ------------------------------------------------------------------ causal probes

# ------------------------------------------------------------------ optional modules

def load_datachecks():
    """eval_suite.datachecks, or None when it is not there yet."""
    try:
        from . import datachecks
    except ImportError:
        return None
    return datachecks


def generator_available() -> bool:
    return importlib.util.find_spec("eval_suite.generator") is not None


def generate_preset(preset, *, seed, episodes_per_condition, out_dir) -> Path:
    """Run `python -m eval_suite.generator` for one preset into out_dir."""
    if not generator_available():
        raise GeneratorError("eval_suite/generator.py is not present yet")
    out_dir = _is_fresh(Path(out_dir))
    argv = [sys.executable, "-m", "eval_suite.generator", "--preset", preset, "--seed", str(seed),
            "--episodes-per-condition", str(episodes_per_condition), "--out", str(out_dir)]
    proc = subprocess.run(argv, cwd=KIT_ROOT, capture_output=True, text=True, timeout=1800,
                          stdin=subprocess.DEVNULL)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        raise GeneratorError(f"generator exited {proc.returncode}: {' | '.join(tail)}")
    return out_dir
