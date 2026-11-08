"""Tests of the suite runner, its real-data protocol and its report.

They run the quick and full levels in local mode on a small kit subset (two
validation episodes per condition, k=2) so the whole module stays fast. Run from
the kit root: python3 -m unittest discover -s eval_suite/tests -t . -v
"""

import contextlib
import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from eval_suite import cli, protocol, scoring as S
from eval_suite.requirements import REQUIREMENTS

REPO = Path(__file__).resolve().parents[2]
PUBLIC = REPO / "public"
FIXTURES = REPO / "eval_suite" / "fixtures"
EPISODES_PER_CONDITION = 2
K = 2
REPEATS = 100
MAX_POINTS = 60
COLUMNS = ["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"]


def _rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def _write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_subset_kit(root: Path) -> Path:
    """A kit public directory with EPISODES_PER_CONDITION validation episodes per condition."""
    kit = root / "public"
    validation = kit / "validation"
    validation.mkdir(parents=True)
    shutil.copy(PUBLIC / "metadata.json", kit / "metadata.json")
    shutil.copytree(PUBLIC / "train", kit / "train")
    shutil.copytree(PUBLIC / "test", kit / "test")

    _, conditions = _rows(PUBLIC / "validation" / "conditions.csv")
    _, queries = _rows(PUBLIC / "validation" / "queries.csv")
    query_episode = {row["query_id"]: row["episode_id"] for row in queries}
    by_condition = defaultdict(list)
    for row in conditions:
        episode = query_episode[row["query_id"]]
        if episode not in by_condition[row["condition"]]:
            by_condition[row["condition"]].append(episode)
    keep = {e for episodes in by_condition.values() for e in sorted(episodes)[:EPISODES_PER_CONDITION]}

    for src in sorted((PUBLIC / "validation").glob("*.csv")):
        header, rows = _rows(src)
        if "query_id" in header:
            kept = [r for r in rows if query_episode[r["query_id"]] in keep]
        else:
            kept = [r for r in rows if r["episode_id"] in keep]
        _write(validation / src.name, header, kept)
    return kit


def make_frozen_replay_entry(dst: Path, reference_csv: Path) -> Path:
    """An entry that replays the frozen reference (validation/reference.csv) by query_id."""
    dst.mkdir(parents=True)
    shutil.copy(reference_csv, dst / "frozen.csv")
    (dst / "reproduce.json").write_text(json.dumps({
        "image": "python",
        "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"],
    }), encoding="utf-8")
    (dst / "predict.py").write_text(
        "import argparse, csv\n"
        "from pathlib import Path\n"
        f"COLS = {COLUMNS!r}\n"
        "def main():\n"
        "    ap = argparse.ArgumentParser(); ap.add_argument('--data'); ap.add_argument('--out')\n"
        "    args = ap.parse_args()\n"
        "    frozen = {}\n"
        "    with (Path(__file__).resolve().parent / 'frozen.csv').open(newline='', encoding='utf-8') as s:\n"
        "        for row in csv.DictReader(s):\n"
        "            frozen[row['query_id']] = [row[c] for c in COLS[1:]]\n"
        "    with (Path(args.data) / 'test' / 'queries.csv').open(newline='', encoding='utf-8') as s:\n"
        "        ids = [row['query_id'] for row in csv.DictReader(s)]\n"
        "    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)\n"
        "    with out.open('w', newline='', encoding='utf-8') as s:\n"
        "        writer = csv.writer(s); writer.writerow(COLS)\n"
        "        for q in ids:\n"
        "            writer.writerow([q] + (frozen.get(q) or ['0.2'] * 5))\n"
        "main()\n", encoding="utf-8")
    return dst


_SHARED = {}


def shared_runs():
    """Run every level once per test process; the tests read the cached reports."""
    if not _SHARED:
        tmp = Path(tempfile.mkdtemp(prefix="suite-cli-test-"))
        kit = make_subset_kit(tmp / "kit")
        runs = {"tmp": tmp, "kit": kit, "reports": {}, "codes": {}}

        def run_level(name, entry, level):
            out = tmp / "runs" / name
            with contextlib.redirect_stdout(io.StringIO()):
                code = cli.main(["run", "--entry", str(entry), "--suite", level, "--mode", "local",
                                 "--kit", str(kit), "--out", str(out), "--k", str(K),
                                 "--bootstrap-repeats", str(REPEATS)])
            runs["codes"][name] = code
            runs["reports"][name] = json.loads((out / "report.json").read_text(encoding="utf-8"))

        for name in ("reference", "uniform", "future_leaky", "broken_sum", "nondeterministic"):
            run_level(name, FIXTURES / name, "quick")
        replay = make_frozen_replay_entry(tmp / "replay", kit / "validation" / "reference.csv")
        run_level("replay", replay, "quick")
        run_level("reference-full", FIXTURES / "reference", "full")
        _SHARED.update(runs)
        atexit_cleanup(tmp)
    return _SHARED


def atexit_cleanup(path):
    import atexit
    atexit.register(shutil.rmtree, path, True)


class SuiteRuns(unittest.TestCase):
    """Base for tests that read the shared level runs."""

    @classmethod
    def setUpClass(cls):
        cls.shared = shared_runs()
        cls.tmp = cls.shared["tmp"]
        cls.kit = cls.shared["kit"]
        cls.results = cls.shared["codes"]

    def report(self, name):
        return self.shared["reports"][name]

    def requirement(self, name, requirement_id):
        rows = {row["id"]: row for row in self.report(name)["requirements"]}
        return rows[requirement_id]

    def checks(self, name, check_id):
        return [r for r in self.report(name)["checks"] if check_id in r["covers"]]


class QuickLevelTests(SuiteRuns):
    def test_reference_scores_near_frozen_half_and_no_failures(self):
        report = self.report("reference")
        self.assertEqual(self.results["reference"], 0, report["verdict"])
        uniform = self.report("uniform")["scores"]["validation-kfold"]["points"]
        points = report["scores"]["validation-kfold"]["points"]
        self.assertGreater(points, uniform)
        # The fixture is a type-only frequency model, not the frozen starter, so it is close to but not exactly max/2.
        self.assertLess(abs(points - MAX_POINTS / 2), 2.0)

    def test_frozen_reference_replay_scores_exactly_half(self):
        points = self.report("replay")["scores"]["validation-kfold"]["points"]
        self.assertAlmostEqual(points, MAX_POINTS / 2, delta=1e-9)
        for ratio in self.report("replay")["scores"]["validation-kfold"]["ratio"].values():
            self.assertAlmostEqual(ratio, 1.0, delta=1e-9)

    def test_uniform_scores_below_reference(self):
        self.assertLess(self.report("uniform")["scores"]["validation-kfold"]["points"], MAX_POINTS / 2)

    def test_future_leaky_fails_causality(self):
        self.assertEqual(self.results["future_leaky"], 1)
        results = self.checks("future_leaky", "CAU-FUTURE-TICKS")
        self.assertTrue(results)
        self.assertTrue(any(r["status"] == "FAIL" for r in results))
        self.assertEqual(self.requirement("future_leaky", "RUL-18")["status"], "FAIL")

    def test_broken_sum_fails_con_prob_sum(self):
        self.assertEqual(self.results["broken_sum"], 1)
        self.assertTrue(any(r["status"] == "FAIL" for r in self.checks("broken_sum", "CON-PROB-SUM")))
        self.assertEqual(self.requirement("broken_sum", "RUL-27")["status"], "FAIL")

    def test_nondeterministic_fails_determinism(self):
        self.assertEqual(self.requirement("nondeterministic", "RUL-37")["status"], "FAIL")

    def test_coverage_table_lists_every_requirement(self):
        report = self.report("reference")
        self.assertEqual(len(report["requirements"]), len(REQUIREMENTS))
        ids = [row["id"] for row in report["requirements"]]
        self.assertEqual(sorted(ids), sorted(r["id"] for r in REQUIREMENTS))
        markdown = (self.tmp / "runs" / "reference" / "report.md").read_text(encoding="utf-8")
        for requirement in REQUIREMENTS:
            self.assertIn(f"| {requirement['id']} |", markdown, requirement["id"])

    def test_every_requirement_has_a_status_and_every_not_run_has_a_reason(self):
        allowed = {"PASS", "FAIL", "WARN", "INFO", "NOT RUN", "MANUAL"}
        for row in self.report("reference")["requirements"]:
            self.assertIn(row["status"], allowed, row["id"])
            if row["status"] == "NOT RUN":
                self.assertTrue(row["reason"], row["id"])

    def test_quick_level_does_not_run_full_only_checks(self):
        self.assertEqual(self.requirement("reference", "RUL-19")["status"], "NOT RUN")

    def test_scores_carry_confidence_intervals_and_slices(self):
        score = self.report("reference")["scores"]["validation-kfold"]
        self.assertLessEqual(score["ci_low"], score["points"])
        self.assertLessEqual(score["points"], score["ci_high"])
        self.assertIn("condition", score["slices"])
        self.assertEqual(len(score["slices"]["condition"]), 4)


class FullLevelTests(SuiteRuns):
    def test_full_level_runs_causal_probes_and_fresh_folder(self):
        report = self.report("reference-full")
        # The subset has two episodes per condition on purpose, so only the count check against
        # metadata (24 per condition) may fail here.
        failing = {row["id"] for row in report["requirements"] if row["status"] == "FAIL"}
        self.assertEqual(failing, {"MET-02"}, report["verdict"])
        for check_id in ("CAU-CROSS-EPISODE", "CAU-CUTOFF-OTHERS", "ROB-ID-RENAME", "ENT-FRESH-FOLDER",
                         "ENT-MATCHES-SUBMITTED"):
            statuses = {r["status"] for r in self.checks("reference-full", check_id)}
            self.assertTrue(statuses, check_id)
            self.assertNotIn("FAIL", statuses, check_id)

    def test_full_level_records_runs_and_missing_generator(self):
        report = self.report("reference-full")
        self.assertTrue(report["runs"])
        self.assertTrue(all(r["wall_s"] < r["limit_s"] for r in report["runs"]))
        self.assertEqual(self.requirement("reference-full", "RUL-20")["status"], "NOT RUN")


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="suite-protocol-test-"))
        cls.kit = make_subset_kit(cls.tmp / "kit")
        cls.folds = protocol.build_fold_datasets(cls.kit, cls.tmp / "folds", k=K)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_no_fold_mount_contains_answers(self):
        for fold in self.folds:
            self.assertFalse((fold.mount / "_truth").exists())
            self.assertEqual(protocol.mount_answer_check(fold.mount).passed, True)
            for name in ("targets.csv", "reference.csv", "conditions.csv", "labels.csv"):
                self.assertFalse((fold.mount / "test" / name).exists(), name)

    def test_scored_queries_partition_validation(self):
        validation_ids = [r["query_id"] for r in _rows(self.kit / "validation" / "queries.csv")[1]]
        scored = [q for fold in self.folds for q in fold.scored_query_ids]
        self.assertEqual(sorted(scored), sorted(validation_ids))
        self.assertEqual(len(scored), len(set(scored)))

    def test_visible_validation_excludes_scored_episodes(self):
        for fold in self.folds:
            visible = {r["episode_id"] for r in _rows(fold.mount / "validation" / "queries.csv")[1]}
            self.assertFalse(visible & set(fold.scored_episodes))

    def test_truth_directory_holds_scored_answers(self):
        for fold in self.folds:
            _, targets = _rows(fold.truth_dir / "test" / "targets.csv")
            self.assertEqual({r["query_id"] for r in targets}, set(fold.scored_query_ids))

    def test_stitched_predictions_score_like_the_kit(self):
        validation = S.load_split(self.kit / "validation")
        parts = []
        for fold in self.folds:
            out = self.tmp / f"fold{fold.index}.csv"
            ids = fold.scored_query_ids
            _write(out, COLUMNS, [{"query_id": q, **{c: "0.2" for c in COLUMNS[1:]}} for q in ids])
            parts.append(out)
        stitched = protocol.stitch(parts, self.tmp / "stitched.csv")
        self.assertEqual(len(_rows(stitched)[1]), len(validation.queries))
        mine = S.official_score(stitched, validation.targets, validation.queries, validation.conditions,
                                validation.reference, MAX_POINTS)["points"]
        kit = protocol.kit_points_check(stitched, self.kit, mine)
        self.assertTrue(kit.passed, kit.message)

    def test_independent_formula_matches_scorer(self):
        split = S.load_split(self.kit / "validation")
        uniform = {q: (0.2,) * 5 for q in split.reference}
        scored = S.official_score(uniform, split.targets, split.queries, split.conditions,
                                  split.reference, MAX_POINTS)["points"]
        independent = protocol.independent_points(uniform, split.targets, split.queries, split.conditions,
                                                  split.reference, MAX_POINTS)
        self.assertAlmostEqual(scored, independent, delta=1e-9)


class CommandTests(unittest.TestCase):
    def test_python_m_eval_suite_run_help_works(self):
        completed = subprocess.run([sys.executable, "-m", "eval_suite", "run", "--help"], cwd=REPO,
                                   capture_output=True, text=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--compare", completed.stdout)


if __name__ == "__main__":
    unittest.main()
