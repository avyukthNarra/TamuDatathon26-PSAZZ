"""Smoke tests for simulator.py: file layout, target sums, seed determinism.

Run from this folder:  python3 -m unittest test_simulator -v
Each test generates one kit with 2 episodes per condition per split, so the
run takes a few seconds.
"""

import filecmp
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import simulator  # noqa: E402

SPLIT_FILES = {
    "train": {"queries.csv", "history.csv", "labels.csv", "conditions.csv"},
    "validation": {"queries.csv", "history.csv", "labels.csv", "conditions.csv",
                   "targets.csv", "reference.csv", "oracle.csv"},
    "test": {"queries.csv", "history.csv"},
}


def generate(out_dir: Path, name: str = "K1") -> Path:
    subprocess.run(
        [sys.executable, str(HERE / "simulator.py"), "--out", str(out_dir),
         "--only", name, "--episodes", "2"],
        check=True, stdout=subprocess.DEVNULL, cwd=str(HERE),
    )
    return out_dir / name


def digest_tree(root: Path) -> dict:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


class SimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.kit = generate(cls.root / "a")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_expected_files_per_split(self):
        for split, expected in SPLIT_FILES.items():
            folder = self.kit / "public" / split
            present = {p.name for p in folder.iterdir() if p.is_file()}
            self.assertTrue(expected <= present, f"{split}: missing {expected - present}")
            if split != "validation":
                self.assertFalse(present & {"targets.csv"}, f"{split} must not carry targets")
        self.assertTrue((self.kit / "public" / "metadata.json").is_file())
        self.assertTrue((self.kit / "latents.json").is_file())
        self.assertTrue((self.kit / "oracle_score.json").is_file())

    def test_episode_counts_follow_flag(self):
        latents = json.loads((self.kit / "latents.json").read_text(encoding="utf-8"))
        self.assertEqual(latents["counts"], {"train": 2, "validation": 2, "test": 2})
        # 4 conditions x 2 episodes, 8 machines each.
        for split in ("train", "validation", "test"):
            self.assertEqual(len(latents["splits"][split]), 8)

    def test_targets_rows_sum_to_one(self):
        columns = simulator.PROBABILITY_COLUMNS
        lines = (self.kit / "public" / "validation" / "targets.csv").read_text(
            encoding="utf-8").splitlines()
        header = lines[0].split(",")
        idx = [header.index(c) for c in columns]
        rows = [line.split(",") for line in lines[1:] if line]
        self.assertEqual(len(rows), 8 * 4 * 2)
        for row in rows:
            total = sum(float(row[i]) for i in idx)
            self.assertLess(abs(total - 1.0), 1e-9, f"row {row[0]} sums to {total}")

    def test_same_seed_gives_identical_bytes(self):
        again = generate(self.root / "b")
        left = digest_tree(self.kit)
        right = digest_tree(again)
        self.assertEqual(left.keys(), right.keys())
        differing = [k for k in left if left[k] != right[k]]
        self.assertEqual(differing, [], f"files differ between runs: {differing[:5]}")
        # filecmp as a second check on one large file.
        self.assertTrue(filecmp.cmp(self.kit / "public" / "train" / "history.csv",
                                    again / "public" / "train" / "history.csv", shallow=False))


if __name__ == "__main__":
    unittest.main()
