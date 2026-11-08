"""Regression checks for comparisons and audit guards; never reads/scorers audit data."""
import csv
import io
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from contextlib import redirect_stdout

from common import COLUMNS, compare, read_csv, reserve_audit, run


class VerificationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name, rows):
        path = self.root / name
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['query_id', *COLUMNS])
            writer.writerows(rows)
        return path

    def compare(self, *args):
        with redirect_stdout(io.StringIO()):
            return compare(*args)

    def test_reordered_query_ids_match(self):
        rows = [('a', 1, 0, 0, 0, 0), ('b', 0, 1, 0, 0, 0)]
        a, b = self.write('a', rows), self.write('b', rows[::-1])
        self.assertEqual(self.compare(a, b, 1e-6)['maximum'], 0)

    def test_mismatched_ids_rejected(self):
        a = self.write('a', [('a', 1, 0, 0, 0, 0)])
        b = self.write('b', [('b', 1, 0, 0, 0, 0)])
        with self.assertRaisesRegex(ValueError, 'query IDs differ'):
            self.compare(a, b, 1e-6)

    def test_bad_probability_duplicate_and_bad_sum_rejected(self):
        for rows in ([('a', 'nan', 0, 0, 0, 0)],
                     [('a', 1.1, -.1, 0, 0, 0)],
                     [('a', 0, 0, 0, 0, 0)],
                     [('a', 1, 0, 0, 0, 0)] * 2):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                read_csv(self.write('bad', rows))

    def test_difference_at_tolerance_fails(self):
        a = self.write('a', [('a', 1, 0, 0, 0, 0)])
        b = self.write('b', [('a', .75, .25, 0, 0, 0)])
        with self.assertRaisesRegex(ValueError, 'strictly below'):
            self.compare(a, b, .25)

    def test_audit_marker_prevents_second_attempt(self):
        prediction = self.write('prediction', [('a', 1, 0, 0, 0, 0)])
        marker, log = self.root / 'marker', self.root / 'log'
        reserve_audit(marker, log, prediction)
        with self.assertRaises(FileExistsError):
            reserve_audit(marker, log, prediction)

    def test_audit_log_prevents_repeat_without_marker(self):
        prediction = self.write('prediction', [('a', 1, 0, 0, 0, 0)])
        marker, log = self.root / 'marker', self.root / 'log'
        log.write_text('2026-11-07 00:00:00\taudit\tcombo-sai\n')
        with self.assertRaisesRegex(RuntimeError, 'already appears'):
            reserve_audit(marker, log, prediction)
        self.assertFalse(marker.exists())

    def test_runner_stops_over_memory_budget(self):
        with redirect_stdout(io.StringIO()), self.assertRaises(MemoryError):
            run([sys.executable, '-c',
                 'import time; x = bytearray(64*1024**2); time.sleep(10)'],
                timeout=5, memory_bytes=32 * 1024**2)

    def test_runner_stops_timeout(self):
        with redirect_stdout(io.StringIO()), self.assertRaises(subprocess.TimeoutExpired):
            run([sys.executable, '-c', 'import time; time.sleep(10)'], timeout=.1)


if __name__ == '__main__':
    unittest.main()
