"""--jobs N must change only the schedule, never the results or the report content."""

import contextlib
import io
import json
import re
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from eval_suite import cli
from eval_suite.entry import EntryError, RunResult
from eval_suite.tests.test_cli import FIXTURES, K, PUBLIC, REPEATS, make_subset_kit

TIMING_KEYS = {"wall_s", "elapsed_s", "started", "peak_mem_mb"}


def _normalize(value):
    """Drop timing fields and blank the seconds printed inside messages."""
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items() if k not in TIMING_KEYS}
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, str):
        return re.sub(r"\d+\.\d+s\b", "<secs>s", value)
    return value


def _quiet_run(argv):
    stderr = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
        code = cli.main(argv)
    return code, stderr.getvalue()


class JobsReportTests(unittest.TestCase):
    def test_quick_suite_report_is_identical_for_jobs_1_and_3(self):
        tmp = Path(tempfile.mkdtemp(prefix="suite-jobs-test-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        kit = make_subset_kit(tmp / "kit")
        reports, stderrs = {}, {}
        for jobs in (1, 3):
            out = tmp / f"jobs-{jobs}"
            code, stderrs[jobs] = _quiet_run(
                ["run", "--entry", str(FIXTURES / "reference"), "--suite", "quick", "--mode", "local",
                 "--kit", str(kit), "--out", str(out), "--k", str(K), "--bootstrap-repeats", str(REPEATS),
                 "--jobs", str(jobs)])
            self.assertIn(code, (0, 1))
            reports[jobs] = json.loads((out / "report.json").read_text(encoding="utf-8"))
        self.assertNotIn("prefetch", stderrs[1])
        self.assertIn("prefetch", stderrs[3])
        self.assertEqual(_normalize(reports[1]), _normalize(reports[3]))


class PrefetchTests(unittest.TestCase):
    def _suite(self, jobs):
        tmp = Path(tempfile.mkdtemp(prefix="suite-prefetch-test-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        suite = cli.Suite(entry=FIXTURES / "reference", compare=None, level="quick", mode="local",
                          kit=PUBLIC, out=tmp / "out", work=tmp / "work", seed=0, k=K, repeats=REPEATS,
                          log=lambda message: None, jobs=jobs)
        return suite, tmp

    def test_prefetch_is_a_no_op_for_one_job(self):
        suite, tmp = self._suite(jobs=1)
        with patch.object(cli, "run_entry") as stub:
            suite.prefetch([("A:fold0", tmp, None), ("A:fold1", tmp, None)])
        stub.assert_not_called()
        self.assertEqual(suite._prefetched, {})

    def test_prefetch_stores_results_by_label_and_run_entry_consumes_them(self):
        suite, tmp = self._suite(jobs=3)
        calls = []

        def fake(entry_dir, data_root, **kwargs):
            calls.append(data_root)
            time.sleep(0.05 if data_root.name == "a" else 0)  # finish out of order on purpose
            return RunResult(csv_path=None, wall_s=0.0, exit_code=0, stdout_tail="", stderr_tail="",
                             peak_mem_mb=None, mode="local")

        a, b = tmp / "a", tmp / "b"
        with patch.object(cli, "run_entry", side_effect=fake):
            suite.prefetch([("A:fold0", a, None), ("A:fold1", b, None)])
        self.assertEqual(set(suite._prefetched), {"A:fold0", "A:fold1"})
        self.assertEqual(sorted(p.name for p in calls), ["a", "b"])
        with patch.object(cli, "run_entry") as direct:
            result = suite._prefetched.pop("A:fold0")
        direct.assert_not_called()
        self.assertEqual(result.exit_code, 0)

    def test_prefetched_entry_error_is_raised_where_the_sequential_loop_would_raise_it(self):
        suite, tmp = self._suite(jobs=2)

        def fake(entry_dir, data_root, **kwargs):
            if data_root.name == "bad":
                raise EntryError("boom")
            return RunResult(csv_path=None, wall_s=0.0, exit_code=0, stdout_tail="", stderr_tail="",
                             peak_mem_mb=None, mode="local")

        with patch.object(cli, "run_entry", side_effect=fake):
            suite.prefetch([("A:fold0", tmp / "good", None), ("A:fold1", tmp / "bad", None)])
        with self.assertRaises(EntryError):
            suite.run_entry("A:fold1", tmp / "bad", tmp / "bad" / "queries.csv")


if __name__ == "__main__":
    unittest.main()
