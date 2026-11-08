"""`python -m eval_suite run`: run an entry through the suite levels and write the report.

Levels
  quick  contract on the kit test run; k-fold real-validation score; determinism
         (fold 0 rerun); three causality variants on fold 0.
  full   quick + all perturbation variants + CAU-CROSS-EPISODE + fresh-folder rerun
         + packaging/ZIP and dataset checks (when datachecks.py exists) + every
         generator preset scored against _truth (when generator.py exists).
  --compare DIR2 runs the same datasets for a second entry and reports paired differences.

Every outcome becomes a record. Anything that could not run is recorded as NOT RUN
with a reason, so report.md always lists every requirement.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import contract as C
from . import protocol as X
from . import report as R
from . import scoring as S
from . import perturb as P
from .contract import CheckResult
from .entry import EntryError, docker_available, run_entry
from .requirements import REQUIREMENTS

KIT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_LIMIT_S = 1800
CPUS, MEMORY_GB = 2, 16
DEFAULT_REPEATS = 1000
QUICK_CAUSAL = ("CAU-FUTURE-TICKS", "ROB-ROW-ORDER", "CAU-ANSWERS-HIDDEN")
MISSING_NAMES = ("INFO-MISSING-VS-ZERO-missing", "INFO-MISSING-VS-ZERO-zero")

# Variant or check names used by perturb.py and contract.py, mapped to REQUIREMENTS IDs.
ALIASES = {
    "CON-CSV-COVERAGE": ["CON-QUERY-SET"],
    "CON-CSV-FINITE": ["CON-PROB-VALUES"],
    "CON-CSV-RANGE": ["CON-PROB-VALUES"],
    "CON-CSV-SUM": ["CON-PROB-SUM"],
    "CON-IMAGE": ["CON-IMAGE-PYTHON"],
    "CON-ZIP-ROOT": ["PKG-ZIP-ROOT", "PKG-ZIP-SAFE"],
    "CON-ZIP-SIZE": ["PKG-ZIP-SIZE"],
    "CON-REQ-PINNED": ["PKG-REQUIREMENTS-PINNED"],
    "CON-RUNTIME": ["CON-EXIT-ZERO", "ROB-RUNTIME-LIMIT"],
    "CON-DETERMINISM": ["ROB-DETERMINISM"],
    "CAU-SAME-TICK-OTHERS": ["CAU-CUTOFF-OTHERS"],
    "CAU-CONDITIONS-HIDDEN": ["CAU-CONDITIONS-REMOVED"],
    "CAU-ANSWERS-HIDDEN": ["CAU-ANSWERS-REMOVED"],
    "INFO-MISSING-VS-ZERO": ["CAU-MISSING-PROBE"],
}
# Records whose PASS is informational only (never a requirement pass).
INFO_ONLY = {"INFO-MISSING-VS-ZERO", "ENT-BASE-PY313", "ENT-MAIN-DATA-MOUNT", "SCO-TIE-RANK"}

LAYOUT_IDS = ["CON-ZIP-ROOT", "CON-IMAGE-PYTHON", "CON-OUTPUT-PATH", "CON-REQ-PINNED", "CON-ZIP-SIZE",
              "CON-REPRODUCE-SCHEMA"]
RUN_IDS = ["CON-OUTPUT-PATH", "CON-EXIT-ZERO", "ROB-RUNTIME-LIMIT"]
KFOLD_IDS = RUN_IDS + ["CON-QUERY-SET", "CON-PROB-VALUES", "CON-PROB-SUM", "SCO-MATCHES-KIT", "SCO-FORMULA",
                       "SCO-MAX-POINTS", "SCO-TRUTH-MAX", "SCO-REF-HALF", "SCO-CONDITION-EQUAL",
                       "SCO-REFERENCE-REPRO", "SCO-REFERENCE-SOURCE"]
PKG_DATACHECK_IDS = ["PKG-ZIP-LIMITS", "PKG-NO-ANSWER-FILES", "PKG-NO-DATA-COPY", "PKG-README-SEASON"]
DAT_IDS = sorted({c for r in REQUIREMENTS for c in r["check_ids"] if c.startswith("DAT-")} | {"CON-ROUND-METADATA"})
GEN_IDS = sorted({c for r in REQUIREMENTS for c in r["check_ids"] if c.startswith("GEN-")})
# Requirement check IDs that no module in this suite evaluates; the reason says why.
RULE_LEVEL_REASONS = {
    "SCO-NO-IMPROVEMENT-BONUS": "rule-level: round combination is organizer code (rounds.combine_rounds), not a file check",
    "SCO-FINAL-SUM": "rule-level: round combination is organizer code (rounds.combine_rounds), not a file check",
    "SCO-MISSING-ZERO": "rule-level: a missing round entry scores zero in the organizer's ranking, not in this suite",
    "SCO-MAIN-REQUIRED": "ranking-level: overall ranking flag is organizer code, not a file check",
    "SCO-TIE-RANK": "ranking-level: tie scheme is unannounced (RUL-14); no ranking code in this suite",
    "SCO-OUTCOME-INDEPENDENT": "by construction: scored probabilities never read the outcome column; no perturbation run",
    "SCO-SCORE-NOT-RANKING": "diagnostic only: the suite never prints the 'score' field, so there is nothing to check",
    "SCO-HALF-SPLIT": "not checked: scoring.repeated_half_splits does not expose its partition",
    "SCO-FORECAST-ONLY": "by construction: points are a pure function of the CSV and targets (see ROB-DETERMINISM)",
    "SCO-REF-FLOOR": "not checked: needs a synthetic zero-reference case (covered by scoring unit tests)",
}


def _slug(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label)


def _status(passed: bool, severity: str, check_id: str) -> str:
    if not passed:
        return "WARN" if severity == "warn" else "FAIL"
    if severity == "warn":
        return "WARN"
    return "INFO" if check_id in INFO_ONLY else "PASS"


def _normalize_identical(result: CheckResult) -> CheckResult:
    """perturb.compare marks passed identical checks with severity 'fail'; a pass is not a failure."""
    if result.passed:
        return CheckResult(result.id, True, "info", result.message, result.details)
    return result


def _renamed(result: CheckResult, new_id: str) -> CheckResult:
    return CheckResult(new_id, result.passed, result.severity, result.message, result.details)


def _last_line(text: str) -> str:
    lines = [line for line in (text or "").splitlines() if line.strip()]
    return lines[-1][:200] if lines else ""


class Suite:
    def __init__(self, *, entry, compare, level, mode, kit, out, work, seed, k, repeats, log):
        self.entry = Path(entry).resolve()
        self.compare = Path(compare).resolve() if compare else None
        self.level, self.mode = level, mode
        self.kit, self.out, self.work = Path(kit), Path(out), Path(work)
        self.seed, self.k, self.repeats = seed, k, repeats
        self.log = log
        self.kit_meta = json.loads((self.kit / "metadata.json").read_text(encoding="utf-8"))
        self.max_points = int(self.kit_meta.get("max_points", 60))
        self.started = datetime.now(timezone.utc)
        self.t0 = time.monotonic()
        self.kept = self.out / "predictions"
        self.kept.mkdir(parents=True, exist_ok=True)
        self.records, self.runs, self.scores, self.paired = [], [], {}, {}
        self.not_run_reasons, self.errors, self.notes, self.datasets = {}, [], [], []
        self.folds, self.fold_csv, self.stitched = [], {}, {}
        self.kit_test_mount = None
        self.kit_test_csv = None
        self.dataset_roots = []
        self._configure_mode()

    # ------------------------------------------------------------ records

    def add(self, result: CheckResult, source: str):
        covers = ALIASES.get(result.id, [result.id])
        self.records.append({"id": result.id, "covers": list(covers),
                             "status": _status(result.passed, result.severity, result.id),
                             "passed": result.passed, "severity": result.severity,
                             "message": result.message, "source": source,
                             "details": result.details})

    def add_all(self, results, source: str):
        for result in results:
            self.add(result, source)

    def fail(self, ids, message, source):
        for check_id in ids:
            self.add(CheckResult(check_id, False, "fail", message, {}), source)

    def not_run(self, ids, reason, source):
        for check_id in ids:
            self.records.append({"id": check_id, "covers": ALIASES.get(check_id, [check_id]),
                                 "status": "NOT RUN", "passed": None, "severity": "info",
                                 "message": reason, "source": source, "details": {}})

    def _configure_mode(self):
        local_reasons = {
            "ROB-OFFLINE": "local mode does not disable the network (docker mode only)",
            "ROB-READONLY-MOUNTS": "local mode does not mount /data or /submission read-only (docker mode only)",
            "ROB-TMP-QUOTA": "local mode does not impose the 128 MiB /tmp quota",
            "ROB-OUTPUT-QUOTA": "local mode does not impose the 32 MiB /output quota",
            "ROB-CPU-LIMIT": "local mode has no CPU quota (docker mode only)",
            "ROB-MEM-LIMIT": "local mode has no memory limit (docker mode only)",
            "ENT-DEPS-PREINSTALL": "local mode does not install requirements.txt (docker mode builds the image)",
            "ENT-BASE-PY313": f"local mode runs the host Python {sys.version_info[0]}.{sys.version_info[1]}; "
                              "the base image is not probed",
        }
        docker_reasons = {
            "ENT-BASE-PY313": "not probed separately: docker mode runs python:3.13-slim by construction (README.md:349)",
        }
        self.not_run_reasons.update(RULE_LEVEL_REASONS)
        self.not_run_reasons.update(local_reasons if self.mode == "local" else docker_reasons)

    def run_entry(self, label, data_root, queries_csv, *, entry=None, tag="A", record=True):
        """Run the entry on data_root; record runtime, output and CSV checks unless record=False."""
        entry = Path(entry) if entry is not None else self.entry
        result = run_entry(entry, data_root, mode=self.mode, timeout_s=RUNTIME_LIMIT_S,
                           cpus=CPUS, memory_gb=MEMORY_GB)
        self.runs.append({"label": label, "tag": tag, "entry": str(entry), "mode": self.mode,
                          "wall_s": round(result.wall_s, 3), "exit_code": result.exit_code,
                          "peak_mem_mb": result.peak_mem_mb, "csv": result.csv_path is not None,
                          "limit_s": RUNTIME_LIMIT_S})
        self.log(f"  ran {label}: exit {result.exit_code} in {result.wall_s:.2f}s")
        kept = None
        if record:
            self.add(C.check_runtime(result), label)
            self.add(C.check_output_written(result), label)
        if result.csv_path is not None:
            if record:
                self.add_all(C.check_csv(result.csv_path, queries_csv), label)
                self.add_all(X.check_strict_csv(result.csv_path), label)
            kept = self.kept / f"{_slug(label)}.csv"
            shutil.copy(result.csv_path, kept)
            shutil.rmtree(result.csv_path.parent, ignore_errors=True)
        return kept, result

    def refused(self, name, result, source):
        tail = _last_line(result.stderr_tail)
        message = (f"refused: entry exited {result.exit_code} on the variant"
                   + (f" ({tail})" if tail else "")
                   + "; treated as a leak guard or loader error, not as a prediction difference")
        self.add(CheckResult(name, False, "warn", message, {}), source)

    def step(self, title, ids, fn):
        self.log(f"[{title}]")
        try:
            fn()
        except EntryError as exc:
            self.fail(ids, f"entry cannot run ({title}): {exc}", title)
        except Exception as exc:  # the suite itself failed: report, never pass silently
            message = f"{type(exc).__name__}: {exc}"
            self.errors.append(f"{title}: {message}")
            self.not_run(ids, f"suite error in '{title}': {message}", title)
            traceback.print_exc(file=sys.stderr)

    # ------------------------------------------------------------ steps

    def _entry_layout(self):
        self.add_all(C.check_entry_dir(self.entry), "entry layout")
        self.add(X.check_reproduce_schema(self.entry), "entry layout")

    def _kit_test(self):
        mount = X.build_kit_test_mount(self.kit, self.work / "kit-test-mount")
        self.kit_test_mount = mount
        csv, result = self.run_entry("kit-test", mount, mount / "test" / "queries.csv")
        self.kit_test_csv = csv
        self.add(X.mount_answer_check(mount), "kit-test")
        self.add(X.forecast_input_check(self.entry, mount), "kit-test")
        if self.kit_meta.get("round") == "main":
            self.add(CheckResult("ENT-MAIN-DATA-MOUNT", True, "info",
                                 "round main: no /main_data mount is expected", {}), "kit-test")
        else:
            self.not_run(["ENT-MAIN-DATA-MOUNT"], "adaptation round mounts /main_data; not exercised here", "kit-test")
        self._docker_records(result, csv is not None)

    def _docker_records(self, result, produced):
        """Limits that only the docker run enforces; local mode records them as NOT RUN instead."""
        if self.mode != "docker":
            return
        ok = result.exit_code == 0 and produced
        if ok:
            limits = "completed under --network none, --cpus 2 and --memory 16g"
            for check_id in ("ROB-OFFLINE", "ROB-CPU-LIMIT", "ROB-MEM-LIMIT"):
                self.add(CheckResult(check_id, True, "info", limits, {"peak_mem_mb": result.peak_mem_mb}), "kit-test")
        else:
            message = (f"exit {result.exit_code}"
                       + (" (killed, likely out of memory)" if result.exit_code == 137 else "")
                       + "; a failed network attempt or memory kill cannot be told apart from stderr")
            for check_id in ("ROB-OFFLINE", "ROB-CPU-LIMIT", "ROB-MEM-LIMIT"):
                self.fail([check_id], message, "kit-test")
        self.add(CheckResult("ROB-READONLY-MOUNTS", True, "info",
                             "entry.py mounts /submission and /data with :ro (flags, not probed from inside)", {}),
                 "kit-test")
        has_requirements = (self.entry / "requirements.txt").is_file()
        if has_requirements:
            self.add(CheckResult("ENT-DEPS-PREINSTALL", ok, "info" if ok else "fail",
                                 "requirements.txt installed into the derived image before the offline run"
                                 if ok else f"offline run failed after preinstall (exit {result.exit_code})", {}),
                     "kit-test")
        else:
            self.add(CheckResult("ENT-DEPS-PREINSTALL", True, "info",
                                 "no requirements.txt: nothing to preinstall", {}), "kit-test")

    def _kfold(self, tag, entry, folds):
        parts, csvs = [], {}
        for fold in folds:
            label = f"{tag}:fold{fold.index}"
            csv, _ = self.run_entry(label, fold.mount, fold.mount / "test" / "queries.csv",
                                    entry=entry, tag=tag)
            if csv is not None:
                csvs[fold.index] = csv
                parts.append(csv)
        self.fold_csv[tag] = csvs
        if len(parts) != len(folds):
            self.notes.append(f"{tag}: k-fold incomplete, {len(parts)} of {len(folds)} folds produced a CSV")
            return None
        stitched = X.stitch(parts, self.work / f"{tag}-validation-stitched.csv")
        validation_queries = self.kit / "validation" / "queries.csv"
        self.add_all(C.check_csv(stitched, validation_queries), f"{tag} validation-kfold")
        self.add_all(X.check_strict_csv(stitched), f"{tag} validation-kfold")
        return stitched

    def _validation(self):
        self.folds = X.build_fold_datasets(self.kit, self.work / "folds", k=self.k, seed=self.seed)
        stitched = self._kfold("A", self.entry, self.folds)
        split = S.load_split(self.kit / "validation")
        if stitched is None:
            self.not_run(KFOLD_IDS, "k-fold predictions incomplete; no score", "validation-kfold")
            return
        self.stitched["A"] = stitched
        try:
            bundle = X.score_bundle(stitched, split, history_path=self.kit / "validation" / "history.csv",
                                    max_points=self.max_points, seed=self.seed, repeats=self.repeats)
        except S.ScoringError as exc:
            self.add(CheckResult("SCO-INPUT", False, "fail", f"predictions cannot be scored: {exc}", {}),
                     "validation-kfold")
            self.not_run(KFOLD_IDS, "predictions could not be scored", "validation-kfold")
            return
        self.scores["validation-kfold"] = bundle
        self.add(X.kit_points_check(stitched, self.kit, bundle["points"]), "validation-kfold")
        self.add_all(X.scoring_rule_checks(self.kit, split, self.max_points, self.kit_meta), "scoring rules")
        self.add_all(X.reference_reproduction(self.kit, split), "scoring rules")
        self.shelve(stitched, "validation-kfold")
        if self.compare:
            other = self._kfold("B", self.compare, self.folds)
            if other is not None:
                self.stitched["B"] = other
                self.paired["validation-kfold"] = X.paired(stitched, other, split, max_points=self.max_points,
                                                           repeats=self.repeats, seed=self.seed)

    def shelve(self, path, name):
        shutil.copy(path, self.kept / f"{_slug(name)}.csv")

    def _determinism(self):
        first = self.fold_csv.get("A", {}).get(0)
        if first is None:
            self.not_run(["ROB-DETERMINISM"], "fold 0 produced no CSV to compare", "determinism")
            return
        fold = self.folds[0]
        second, _ = self.run_entry("A:fold0-rerun", fold.mount, fold.mount / "test" / "queries.csv")
        if second is None:
            self.fail(["ROB-DETERMINISM"], "rerun produced no CSV", "determinism")
            return
        self.add(C.check_determinism(first, second), "determinism")

    def _causality(self):
        fold = self.folds[0]
        base = self.fold_csv["A"][0]
        variants = P.make_variants(fold.mount, self.work / "variants", split="test", seed=self.seed)
        wanted = variants if self.level == "full" else [v for v in variants if v.name in QUICK_CAUSAL]
        # identical_subset (CAU-CROSS-EPISODE) and identical variants go through perturb.compare.
        missing = zero = None
        for variant in wanted:
            label = f"A:{variant.name}"
            csv, result = self.run_entry(label, variant.data_root, variant.data_root / "test" / "queries.csv",
                                         record=False)
            if csv is None:
                check = "INFO-MISSING-VS-ZERO" if variant.relation == "info" else variant.name
                self.refused(check, result, "causality")
                continue
            if variant.relation == "info":
                if variant.name.endswith("-missing"):
                    missing = csv
                else:
                    zero = csv
                continue
            self.add(_normalize_identical(P.compare(base, csv, variant, tol=1e-8)), "causality")
        if missing is not None and zero is not None:
            self.add(P.compare_missing_vs_zero(missing, zero, tol=1e-8), "causality")

    def _fresh_folder(self):
        fresh = self.work / "fresh-entry"
        if fresh.exists():
            shutil.rmtree(fresh)
        shutil.copytree(self.entry, fresh)
        csv, result = self.run_entry("A:fresh-folder", self.kit_test_mount, self.kit_test_mount / "test" / "queries.csv",
                                     entry=fresh, record=False)
        if csv is None or result.exit_code != 0:
            self.add(CheckResult("ENT-FRESH-FOLDER", False, "fail",
                                 f"run from an empty folder exited {result.exit_code}: {_last_line(result.stderr_tail)}",
                                 {}), "fresh-folder")
            return
        self.add(CheckResult("ENT-FRESH-FOLDER", True, "info",
                             f"reproduce command ran from a fresh copy in {result.wall_s:.2f}s", {}), "fresh-folder")
        if self.kit_test_csv is None:
            self.not_run(["ENT-MATCHES-SUBMITTED"], "kit test run produced no forecast to compare with", "fresh-folder")
            return
        self.add(_renamed(C.check_determinism(self.kit_test_csv, csv), "ENT-MATCHES-SUBMITTED"), "fresh-folder")

    def _packaging(self):
        dc = X.load_datachecks()
        if dc is None:
            self.not_run(PKG_DATACHECK_IDS, "eval_suite/datachecks.py is not present yet", "packaging")
            return
        archive = Path(dc.package_entry(self.entry, self.work / "entry-package.zip"))
        self.add_all(dc.check_zip(archive, metadata_path=self.kit / "metadata.json"), "packaging")

    def _presets(self):
        if not X.generator_available():
            for preset in X.PRESETS:
                self.scores[f"preset:{preset}"] = {"status": "NOT RUN",
                                                   "reason": "eval_suite/generator.py is not present yet"}
            self.not_run(GEN_IDS, "eval_suite/generator.py is not present yet", "generator")
            return
        episodes = int(self.kit_meta["counts_per_condition"]["validation"])
        for preset in X.PRESETS:
            key = f"preset:{preset}"
            try:
                dest = X.generate_preset(preset, seed=self.seed, episodes_per_condition=episodes,
                                         out_dir=self.work / "presets" / preset)
            except X.GeneratorError as exc:
                self.scores[key] = {"status": "NOT RUN", "reason": str(exc)}
                continue
            mount = X.strip_truth_copy(dest, self.work / "preset-mounts" / preset)
            self.dataset_roots.append(mount)
            try:
                self._score_preset(preset, key, dest, mount)
            except Exception as exc:
                self.scores[key] = {"status": "NOT RUN", "reason": f"could not score: {exc}"}
                self.errors.append(f"{key}: {type(exc).__name__}: {exc}")
        self.not_run(GEN_IDS, "generator-level checks (GEN-*) are not emitted by the suite; presets are scored only",
                     "generator")

    def _score_preset(self, preset, key, dest, mount):
        csv, result = self.run_entry(f"A:preset:{preset}", mount, mount / "test" / "queries.csv")
        if csv is None:
            self.scores[key] = {"status": "NOT RUN", "reason": f"entry produced no CSV (exit {result.exit_code})"}
            return
        split = S.load_split(dest / "test", truth_dir=dest / "_truth")
        meta_path = dest / "metadata.json"
        max_points = int(json.loads(meta_path.read_text(encoding="utf-8")).get("max_points", 60)) \
            if meta_path.is_file() else 60
        self.scores[key] = X.score_bundle(csv, split, history_path=mount / "test" / "history.csv",
                                          max_points=max_points, seed=self.seed, repeats=self.repeats)
        if self.compare:
            other, _ = self.run_entry(f"B:preset:{preset}", mount, mount / "test" / "queries.csv",
                                      entry=self.compare, tag="B")
            if other is not None:
                self.paired[key] = X.paired(csv, other, split, max_points=max_points,
                                            repeats=self.repeats, seed=self.seed)

    def _datasets(self):
        dc = X.load_datachecks()
        if dc is None:
            self.not_run(DAT_IDS, "eval_suite/datachecks.py is not present yet", "datasets")
            return
        self.datasets.append(str(self.kit))
        self.add_all(dc.check_dataset(self.kit), "dataset: kit public")
        for root in self.dataset_roots:
            self.datasets.append(str(root))
            self.add_all(dc.check_dataset(root), f"dataset: {root.name}")

    def _mount_inputs(self):
        dc = X.load_datachecks()
        if dc is None:
            self.notes.append("datachecks.check_mount_inputs not present; mount checks used the suite's own")
            return
        self.add_all(dc.check_mount_inputs(self.kit_test_mount, self.entry), "mount inputs")

    # ------------------------------------------------------------ driver

    def execute(self):
        full = self.level == "full"
        self.step("entry layout", LAYOUT_IDS, self._entry_layout)
        self.step("kit test run", RUN_IDS, self._kit_test)
        self.step("real validation k-fold", KFOLD_IDS, self._validation)
        self.step("determinism", ["ROB-DETERMINISM"], self._determinism)
        self.step("causality probes", ["CAU-FUTURE-TICKS", "ROB-ROW-ORDER", "CAU-ANSWERS-REMOVED", "CAU-CROSS-EPISODE"],
                  self._causality)
        if full:
            self.step("fresh folder", ["ENT-FRESH-FOLDER", "ENT-MATCHES-SUBMITTED"], self._fresh_folder)
            self.step("packaging", PKG_DATACHECK_IDS, self._packaging)
            self.step("generator presets", GEN_IDS, self._presets)
            self.step("dataset checks", DAT_IDS, self._datasets)
            self.step("mount inputs", ["ROB-MOUNT-NO-ANSWERS"], self._mount_inputs)

    def caveats(self):
        items = [
            "Reproducibility: the suite runs each entry with `python -m eval_suite run`; every number is "
            "recomputed from the CSVs it writes, and predictions are kept under predictions/.",
            "Real-validation score is a k-fold stitch: each fold's entry sees the other folds' validation answers "
            "as training data and never its own scored episodes. This is comparable across entries but is not "
            "identical to training on all of validation.",
            "The kit test run mounts train and test only, with no validation answers. It checks format and "
            "runtime; it never scores anything (EXP-03: test answers are private).",
            "Fold datasets are partial by design (validation holds only visible episodes, test holds the scored "
            "ones), so check_dataset is run on the kit and on generator presets, not on fold datasets.",
            "The scored answers live in <fold>-truth/_truth, outside the mounted dataset, so ROB-MOUNT-NO-ANSWERS "
            "can hold for the mounts the entry sees.",
            "CAU-ANSWERS-REMOVED and CAU-CONDITIONS-REMOVED change the scored split, which has no answer or "
            "condition files in the kit, so those variants are equal to the base by construction.",
            "CAU-CUTOFF-OTHERS (perturb CAU-SAME-TICK-OTHERS) also adds a tick-at-cutoff row for the query's own "
            "machine. That is stricter than the rule's wording (other machines only).",
            "CAU-CROSS-EPISODE randomises payouts of the other scored episodes' history in the scored split only. "
            "Train and validation labels are not touched, since changing them would legitimately change every "
            "prediction.",
            "CAU-MISSING-PROBE is information only. A WARN means the entry treats a missing tick as a zero payout.",
            "A variant the entry refuses (non-zero exit) is recorded as WARN ('refused'), not PASS and not FAIL.",
            "Requirements marked MANUAL (deadlines, form process, provenance) have no file check; see REQUIREMENTS.md.",
            "Generator presets are spec-derived approximations of the organizer's data, not the organizer's data.",
            "Docker mode maps the image 'python' to python:3.13-slim (README.md:349). The Docker Desktop VM on "
            "this machine may have less than 16 GiB, so the memory limit is applied but not fully reproducible.",
        ]
        if self.mode == "local":
            items.insert(0, "NON-AUTHORITATIVE: local mode runs the host Python with no network block, no read-only "
                            "mounts, no CPU or memory limits and no requirements install. Use --mode docker for "
                            "organizer-equivalent runs.")
        else:
            items.insert(0, "Docker mode is the organizer emulation. Peak memory is sampled with `docker stats` "
                            "every 0.5 s, so it is an estimate, and Docker VM memory may differ from the host. "
                            "The /tmp and /output quotas are not enforced by entry.py.")
        if self.compare is None:
            items.append("No --compare entry was given, so no paired comparison was made.")
        return items

    def context(self):
        return {
            "entry_name": self.entry.name, "entry": str(self.entry),
            "compare": str(self.compare) if self.compare else None,
            "level": self.level, "mode": self.mode, "kit": str(self.kit), "seed": self.seed, "k": self.k,
            "repeats": self.repeats, "started": self.started.isoformat(timespec="seconds"),
            "elapsed_s": time.monotonic() - self.t0,
            "records": self.records, "runs": self.runs, "scores": self.scores, "paired": self.paired,
            "not_run_reasons": self.not_run_reasons, "errors": self.errors, "notes": self.notes,
            "datasets": self.datasets, "caveats": self.caveats(),
        }


# ------------------------------------------------------------------ CLI

def _kit_root(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_dir():
        path = KIT_ROOT / value
    if not all((path / name).exists() for name in ("validation", "train", "test", "metadata.json")):
        raise SystemExit(f"--kit {value!r} is not a kit public directory (needs train/, validation/, test/, metadata.json)")
    return path.resolve()


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m eval_suite")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the suite on an entry")
    run.add_argument("--entry", required=True, help="entry directory (code ZIP root)")
    run.add_argument("--compare", help="second entry directory for a paired comparison")
    run.add_argument("--suite", choices=("quick", "full"), default="quick")
    run.add_argument("--mode", choices=("local", "docker"), default="local",
                     help="local is fast and non-authoritative; docker is the organizer emulation")
    run.add_argument("--kit", default="public", help="kit public directory (default: public/)")
    run.add_argument("--out", help="output directory (default: runs/eval/<entry name>)")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--k", type=int, default=4, help="validation folds (default 4)")
    run.add_argument("--bootstrap-repeats", type=int, default=DEFAULT_REPEATS,
                     help="bootstrap resamples per CI (default 1000)")
    run.add_argument("--keep-work", action="store_true", help="keep copies of mounted datasets for debugging")
    return parser


def _log(message):
    print(message, file=sys.stderr, flush=True)


def run_suite(args) -> int:
    kit = _kit_root(args.kit)
    entry = Path(args.entry).expanduser().resolve()
    if not entry.is_dir():
        raise SystemExit(f"--entry is not a directory: {entry}")
    if args.compare and not Path(args.compare).expanduser().is_dir():
        raise SystemExit(f"--compare is not a directory: {args.compare}")
    if args.mode == "docker":
        ok, message = docker_available()
        if not ok:
            raise SystemExit(f"--mode docker needs a running Docker daemon: {message}")
    out = Path(args.out) if args.out else Path("runs") / "eval" / entry.name
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="eval-suite-work-"))
    suite = Suite(entry=entry, compare=args.compare and Path(args.compare).expanduser().resolve(),
                  level=args.suite, mode=args.mode, kit=kit, out=out, work=work, seed=args.seed,
                  k=args.k, repeats=args.bootstrap_repeats, log=_log)
    try:
        _log(f"eval_suite {args.suite} on {entry} ({args.mode} mode), output {out}")
        suite.execute()
    finally:
        if args.keep_work:
            _log(f"work directory kept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
    result = R.write_report(out, suite.context())
    _log(f"wrote {out / 'report.md'} and {out / 'report.json'}")
    print(R.verdict_line(result["tally"], result["record_fail"], suite.context()))
    if result["tally"]["FAIL"] or result["record_fail"]:
        return 1
    return 2 if suite.errors else 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return run_suite(args)
    return 2
