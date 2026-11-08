"""Tests of the suite's entry runner and contract checks, using the dummy fixtures.

Run from the kit root:  python -m unittest discover -s eval_suite/tests -t . -v
"""

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from eval_suite import contract
from eval_suite.contract import (
    check_csv, check_determinism, check_entry_dir, check_output_written, check_runtime,
)
from eval_suite.entry import EntryError, RunResult, _rewrite, docker_available, run_entry

KIT = Path(__file__).resolve().parents[2]
PUBLIC = KIT / "public"
TEST_QUERIES = PUBLIC / "test" / "queries.csv"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ALL_FIXTURES = ("uniform", "reference", "future_leaky", "broken_sum", "missing_rows",
                "nondeterministic", "slow")
COLUMNS = ["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"]


def failing_ids(results):
    return sorted(r.id for r in results if not r.passed)


def run_result(exit_code=0, wall_s=1.0, csv_path=None):
    return RunResult(csv_path=csv_path, wall_s=wall_s, exit_code=exit_code, stdout_tail="",
                     stderr_tail="", peak_mem_mb=None, mode="local")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="entry-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def copy_fixture(self, name):
        target = self.tmp / name
        shutil.copytree(FIXTURES / name, target)
        return target

    def run_fixture(self, name, data=PUBLIC, timeout_s=120):
        return run_entry(FIXTURES / name, data, mode="local", timeout_s=timeout_s)

    def run_ok(self, name, data=PUBLIC):
        result = self.run_fixture(name, data)
        self.assertEqual(result.exit_code, 0, result.stderr_tail)
        self.assertIsNotNone(result.csv_path, "no /output/submission.csv was written")
        return result

    def write_csv(self, name, rows, header=COLUMNS):
        path = self.tmp / name
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def uniform_rows(self):
        with TEST_QUERIES.open(newline="", encoding="utf-8") as stream:
            ids = [row["query_id"] for row in csv.DictReader(stream)]
        return [[q, "0.2", "0.2", "0.2", "0.2", "0.2"] for q in ids]


class FixtureContractTests(Base):
    def test_every_fixture_has_a_valid_entry_layout(self):
        for name in ALL_FIXTURES:
            with self.subTest(fixture=name):
                results = check_entry_dir(FIXTURES / name)
                self.assertEqual(failing_ids(results), [], [r.message for r in results if not r.passed])

    def test_uniform_passes_every_csv_check_on_public_test(self):
        result = self.run_ok("uniform")
        results = check_csv(result.csv_path, TEST_QUERIES)
        self.assertEqual(failing_ids(results), [], [r.message for r in results if not r.passed])
        self.assertEqual(len(results), 6)

    def test_reference_passes_every_csv_check_on_public_test(self):
        result = self.run_ok("reference")
        results = check_csv(result.csv_path, TEST_QUERIES)
        self.assertEqual(failing_ids(results), [], [r.message for r in results if not r.passed])

    def test_uniform_and_reference_pass_runtime_and_determinism(self):
        for name in ("uniform", "reference"):
            with self.subTest(fixture=name):
                first = self.run_ok(name)
                second = self.run_ok(name)
                self.assertTrue(check_runtime(first).passed)
                self.assertTrue(check_output_written(first).passed)
                determinism = check_determinism(first.csv_path, second.csv_path)
                self.assertTrue(determinism.passed, determinism.message)

    def test_reference_is_not_uniform(self):
        reference = self.run_ok("reference").csv_path.read_text().splitlines()[1:]
        uniform = self.run_ok("uniform").csv_path.read_text().splitlines()[1:]
        self.assertNotEqual(reference, uniform)


class BrokenFixtureTests(Base):
    def test_broken_sum_fails_only_the_sum_check(self):
        results = check_csv(self.run_ok("broken_sum").csv_path, TEST_QUERIES)
        self.assertEqual(failing_ids(results), ["CON-CSV-SUM"])

    def test_missing_rows_fails_only_the_coverage_check(self):
        results = check_csv(self.run_ok("missing_rows").csv_path, TEST_QUERIES)
        self.assertEqual(failing_ids(results), ["CON-CSV-COVERAGE"])

    def test_nondeterministic_fails_determinism_but_not_format(self):
        first = self.run_ok("nondeterministic")
        second = self.run_ok("nondeterministic")
        self.assertEqual(failing_ids(check_csv(first.csv_path, TEST_QUERIES)), [])
        determinism = check_determinism(first.csv_path, second.csv_path)
        self.assertEqual(determinism.id, "CON-DETERMINISM")
        self.assertFalse(determinism.passed, "two unseeded runs should differ by more than 1e-8")

    def test_slow_entry_is_killed_and_fails_runtime(self):
        result = self.run_fixture("slow", timeout_s=2)
        self.assertEqual(result.exit_code, contract.TIMEOUT_EXIT_CODE)
        self.assertIsNone(result.csv_path)
        self.assertLess(result.wall_s, 20, "the timed-out process was not killed promptly")
        self.assertFalse(check_runtime(result).passed)
        self.assertFalse(check_output_written(result).passed)

    def test_future_leaky_passes_format_but_changes_under_future_ticks(self):
        base = self.run_ok("future_leaky")
        self.assertEqual(failing_ids(check_csv(base.csv_path, TEST_QUERIES)), [])
        perturbed_root = self.tmp / "perturbed"
        shutil.copytree(PUBLIC / "test", perturbed_root / "test")
        with TEST_QUERIES.open(newline="", encoding="utf-8") as stream:
            first = next(csv.DictReader(stream))
        with (perturbed_root / "test" / "history.csv").open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow([first["episode_id"], first["machine_id"], first["cutoff"], 20])
        variant = self.run_ok("future_leaky", data=perturbed_root)
        leak = check_determinism(base.csv_path, variant.csv_path)
        self.assertFalse(leak.passed, "future-tick perturbation should change a leaky forecast")

    def test_uniform_is_invariant_to_the_same_future_ticks(self):
        perturbed_root = self.tmp / "perturbed"
        shutil.copytree(PUBLIC / "test", perturbed_root / "test")
        with TEST_QUERIES.open(newline="", encoding="utf-8") as stream:
            first = next(csv.DictReader(stream))
        with (perturbed_root / "test" / "history.csv").open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow([first["episode_id"], first["machine_id"], first["cutoff"], 20])
        base = self.run_ok("uniform")
        variant = self.run_ok("uniform", data=perturbed_root)
        self.assertTrue(check_determinism(base.csv_path, variant.csv_path).passed)


class CsvContractTests(Base):
    def test_extra_column_fails_columns_and_skips_the_rest(self):
        path = self.write_csv("extra.csv", self.uniform_rows(), header=COLUMNS + ["p_extra"])
        results = check_csv(path, TEST_QUERIES)
        self.assertIn("CON-CSV-COLUMNS", failing_ids(results))

    def test_nan_fails_finite(self):
        rows = self.uniform_rows()
        rows[3][2] = "nan"
        results = check_csv(self.write_csv("nan.csv", rows), TEST_QUERIES)
        self.assertEqual(failing_ids(results), ["CON-CSV-FINITE"])

    def test_out_of_range_value_fails_range(self):
        rows = self.uniform_rows()
        rows[0] = [rows[0][0], "1.5", "-0.5", "0", "0", "0"]
        results = check_csv(self.write_csv("range.csv", rows), TEST_QUERIES)
        self.assertIn("CON-CSV-RANGE", failing_ids(results))

    def test_duplicate_and_extra_queries_fail_coverage(self):
        rows = self.uniform_rows()
        rows.append(rows[0])
        rows.append(["not-a-query", "0.2", "0.2", "0.2", "0.2", "0.2"])
        results = check_csv(self.write_csv("dup.csv", rows), TEST_QUERIES)
        self.assertEqual(failing_ids(results), ["CON-CSV-COVERAGE"])

    def test_sum_within_tolerance_passes_and_outside_fails(self):
        rows = self.uniform_rows()
        rows[0] = [rows[0][0], "0.2", "0.2", "0.2", "0.2", "0.2000005"]
        self.assertNotIn("CON-CSV-SUM", failing_ids(check_csv(self.write_csv("tiny.csv", rows), TEST_QUERIES)))
        rows[0] = [rows[0][0], "0.2", "0.2", "0.2", "0.2", "0.200002"]
        self.assertIn("CON-CSV-SUM", failing_ids(check_csv(self.write_csv("big.csv", rows), TEST_QUERIES)))

    def test_size_limit_fails_size_only(self):
        path = self.write_csv("big-size.csv", self.uniform_rows())
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n" * (contract.MAX_CSV_BYTES + 1 - path.stat().st_size))
        self.assertEqual(path.stat().st_size, contract.MAX_CSV_BYTES + 1)
        self.assertEqual(failing_ids(check_csv(path, TEST_QUERIES)), ["CON-CSV-SIZE"])

    def test_missing_file_fails(self):
        results = check_csv(self.tmp / "absent.csv", TEST_QUERIES)
        self.assertEqual(failing_ids(results), ["CON-CSV-COLUMNS"])


class EntryContractTests(Base):
    def test_image_must_be_python(self):
        entry = self.copy_fixture("uniform")
        (entry / "reproduce.json").write_text(json.dumps({"image": "ubuntu", "command": ["true"]}))
        self.assertIn("CON-IMAGE", failing_ids(check_entry_dir(entry)))

    def test_missing_root_file_fails_zip_root(self):
        entry = self.copy_fixture("uniform")
        (entry / "README.md").unlink()
        self.assertIn("CON-ZIP-ROOT", failing_ids(check_entry_dir(entry)))

    def test_requirements_must_be_exact_pins(self):
        entry = self.copy_fixture("uniform")
        (entry / "requirements.txt").write_text("numpy>=1.0\n")
        self.assertEqual(failing_ids(check_entry_dir(entry)), ["CON-REQ-PINNED"])
        (entry / "requirements.txt").write_text("# deps\nnumpy==1.26.4\n")
        self.assertEqual(failing_ids(check_entry_dir(entry)), [])

    def test_command_must_write_the_output_csv(self):
        entry = self.copy_fixture("uniform")
        (entry / "reproduce.json").write_text(json.dumps(
            {"image": "python", "command": ["python", "predict.py", "--data", "/data", "--out", "/tmp/x.csv"]}))
        self.assertEqual(failing_ids(check_entry_dir(entry)), ["CON-OUTPUT-PATH"])

    def test_zip_size_limit_fails_zip_size(self):
        with mock.patch.object(contract, "MAX_ZIP_BYTES", 10):
            self.assertEqual(failing_ids(check_entry_dir(FIXTURES / "uniform")), ["CON-ZIP-SIZE"])

    def test_run_entry_rejects_unmapped_image_before_running(self):
        entry = self.copy_fixture("uniform")
        (entry / "reproduce.json").write_text(json.dumps({"image": "ubuntu", "command": ["true"]}))
        with self.assertRaises(EntryError):
            run_entry(entry, PUBLIC, mode="local")

    def test_run_entry_rejects_unknown_mode(self):
        with self.assertRaises(EntryError):
            run_entry(FIXTURES / "uniform", PUBLIC, mode="cloud")


class RuntimeAndDeterminismTests(Base):
    def test_runtime_limit(self):
        self.assertTrue(check_runtime(run_result(wall_s=10)).passed)
        self.assertFalse(check_runtime(run_result(wall_s=1801)).passed)
        self.assertTrue(check_runtime(run_result(wall_s=1801), limit_s=3600).passed)

    def test_nonzero_exit_fails_runtime(self):
        self.assertFalse(check_runtime(run_result(exit_code=1)).passed)
        self.assertFalse(check_runtime(run_result(exit_code=contract.TIMEOUT_EXIT_CODE)).passed)

    def test_determinism_compares_after_renormalisation(self):
        base = self.write_csv("a.csv", self.uniform_rows())
        doubled_rows = [[r[0]] + [format(float(v) * 2, ".17g") for v in r[1:]] for r in self.uniform_rows()]
        doubled = self.write_csv("b.csv", doubled_rows)
        self.assertTrue(check_determinism(base, doubled).passed)

    def test_determinism_tolerance(self):
        base = self.write_csv("c.csv", self.uniform_rows())
        rows = self.uniform_rows()
        rows[0] = [rows[0][0], "0.2000001", "0.2", "0.2", "0.2", "0.1999999"]
        other = self.write_csv("d.csv", rows)
        self.assertFalse(check_determinism(base, other, tol=1e-8).passed)
        self.assertTrue(check_determinism(base, other, tol=1e-3).passed)

    def test_determinism_rejects_different_query_sets(self):
        base = self.write_csv("e.csv", self.uniform_rows())
        other = self.write_csv("f.csv", self.uniform_rows()[:-1])
        self.assertFalse(check_determinism(base, other).passed)


class LocalModeTests(Base):
    def test_local_run_writes_valid_csv_for_public_test(self):
        result = self.run_ok("uniform")
        self.assertEqual(result.mode, "local")
        self.assertGreater(result.wall_s, 0)
        self.assertIsNone(result.peak_mem_mb)
        self.assertEqual(failing_ids(check_csv(result.csv_path, TEST_QUERIES)), [])
        with result.csv_path.open(newline="", encoding="utf-8") as stream:
            self.assertEqual(sum(1 for _ in csv.DictReader(stream)), 1536)

    def test_path_rewrite_maps_container_prefixes_only(self):
        mapping = [("/data", "/host/data"), ("/output", "/host/out")]
        self.assertEqual(_rewrite("/output/submission.csv", mapping), "/host/out/submission.csv")
        self.assertEqual(_rewrite("--out=/output/a.csv", mapping), "--out=/host/out/a.csv")
        self.assertEqual(_rewrite("/data", mapping), "/host/data")
        self.assertEqual(_rewrite("/data2/x", mapping), "/data2/x")
        self.assertEqual(_rewrite("predict.py", mapping), "predict.py")


DOCKER_OK, DOCKER_MESSAGE = docker_available()


@unittest.skipUnless(DOCKER_OK, f"docker unavailable: {DOCKER_MESSAGE}")
class DockerSmokeTests(Base):
    def test_uniform_runs_in_docker_and_passes_checks(self):
        result = run_entry(FIXTURES / "uniform", PUBLIC, mode="docker", timeout_s=600)
        self.assertEqual(result.mode, "docker")
        self.assertEqual(result.exit_code, 0, result.stderr_tail)
        self.assertIsNotNone(result.csv_path)
        self.assertEqual(failing_ids(check_csv(result.csv_path, TEST_QUERIES)), [])
        self.assertTrue(check_runtime(result).passed)


if __name__ == "__main__":
    unittest.main()
