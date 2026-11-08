"""Tests of the data, package and mount checks (eval_suite/datachecks.py) and the cross-episode variant.

Built from the real kit at public/ (data checks, cross-episode variant) and the dummy entry in
eval_suite/fixtures/uniform (packaging, mounts, docker). Every corruption is applied to a temp copy.

Run from the kit root:  python -m unittest discover -s eval_suite/tests -t . -v
"""

import csv
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from eval_suite import datachecks
from eval_suite.datachecks import (
    DATASET_CHECK_IDS, ZIP_CHECK_IDS, check_dataset, check_mount_inputs, check_zip, package_entry,
)
from eval_suite.entry import docker_available, run_entry
from eval_suite.contract import check_csv
from eval_suite.perturb import CROSS_EPISODE_ID, compare, make_variants
from slot_forecasting.schema import ValidationError

KIT = Path(__file__).resolve().parents[2]
PUBLIC = KIT / "public"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
TEST_QUERIES = PUBLIC / "test" / "queries.csv"
PROBS = ("p_0", "p_1", "p_2", "p_5", "p_20")


def failing(results):
    return {r.id for r in results if not r.passed}


def by_id(results):
    return {r.id: r for r in results}


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_zip(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in members.items():
            archive.writestr(name, text)
    return path


def rewrite(path, transform):
    """Apply transform(rows) -> rows to a CSV file, keeping its header."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = [dict(zip(header, row)) for row in reader]
    rows = transform(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="datachecks-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def dataset(self):
        """A writable copy of the kit dataset at <tmp>/data."""
        target = self.tmp / "data"
        shutil.copytree(PUBLIC, target)
        return target

    def zip_entry(self, source_name="uniform", extra=None, reproduce=None, requirements=None, readme=None):
        """Copy a fixture into a temp folder, optionally change it, and package it. Returns the ZIP path."""
        folder = self.tmp / f"entry-{len(list(self.tmp.iterdir()))}"
        shutil.copytree(FIXTURES / source_name, folder)
        if reproduce is not None:
            (folder / "reproduce.json").write_text(json.dumps(reproduce))
        if requirements is not None:
            (folder / "requirements.txt").write_text(requirements)
        if readme is not None:
            (folder / "README.md").write_text(readme)
        for name, text in (extra or {}).items():
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return package_entry(folder, self.tmp / f"out-{folder.name}.zip")


class DatasetCheckTests(TempCase):
    def assert_only_fails(self, data, expected_ids):
        results = check_dataset(data)
        self.assertEqual([r.id for r in results], list(DATASET_CHECK_IDS))
        self.assertEqual(failing(results), set(expected_ids), [r.message for r in results if not r.passed])

    def test_public_kit_passes_every_dataset_check(self):
        results = check_dataset(PUBLIC)
        self.assertEqual([r.id for r in results], list(DATASET_CHECK_IDS))
        self.assertEqual(failing(results), set(), [r.message for r in results if not r.passed])

    def test_round_name_must_match_metadata(self):
        self.assertTrue(by_id(check_dataset(PUBLIC, "main"))["CON-ROUND-METADATA"].passed)
        self.assertEqual(failing(check_dataset(PUBLIC, "adaptation")), {"CON-ROUND-METADATA"})

    def test_shape_counts_are_reported_for_the_real_kit(self):
        shape = by_id(check_dataset(PUBLIC))["GEN-HISTORY-SHAPE"]
        self.assertEqual((shape.details["long"], shape.details["short"]), (2304, 768))
        self.assertEqual(shape.details["per_split"]["test"], {"long": 1152, "short": 384})

    def test_schema_header_change_fails_schema(self):
        data = self.dataset()
        lines = (data / "train" / "queries.csv").read_text().splitlines()
        lines[0] = lines[0].replace("cutoff", "cut_off")
        (data / "train" / "queries.csv").write_text("\n".join(lines) + "\n")
        self.assertIn("DAT-SCHEMA-COLUMNS", failing(check_dataset(data)))

    def test_answer_file_in_test_split_fails_test_split_files(self):
        data = self.dataset()
        shutil.copy(data / "validation" / "labels.csv", data / "test" / "labels.csv")
        self.assert_only_fails(data, {"DAT-TEST-SPLIT-FILES"})

    def test_bad_enums_fail_query_enums(self):
        data = self.dataset()

        def corrupt(rows):
            rows[0]["machine_type"] = "unknown"
            rows[1]["signal"] = "2"
            return rows
        rewrite(data / "validation" / "queries.csv", corrupt)
        self.assert_only_fails(data, {"DAT-QUERY-ENUMS"})

    def test_cutoff_out_of_range_fails(self):
        data = self.dataset()
        rewrite(data / "test" / "queries.csv", lambda rows: [dict(r, cutoff="300") if i == 0 else r
                                                             for i, r in enumerate(rows)])
        failed = failing(check_dataset(data))
        self.assertIn("DAT-CUTOFF-RANGE", failed)

    def test_payout_outside_set_fails(self):
        data = self.dataset()

        def corrupt(rows):
            rows[0]["payout"] = "3"
            return rows
        rewrite(data / "train" / "history.csv", corrupt)
        self.assert_only_fails(data, {"DAT-PAYOUT-SET"})

    def test_unknown_condition_label_fails(self):
        data = self.dataset()

        def corrupt(rows):
            rows[0]["condition"] = "bogus"
            return rows
        rewrite(data / "train" / "conditions.csv", corrupt)
        self.assertIn("DAT-CONDITION-SET", failing(check_dataset(data)))

    def test_moving_one_episode_breaks_per_condition_counts(self):
        data = self.dataset()
        episode = read_rows(data / "train" / "queries.csv")[0]["episode_id"]
        ids = {r["query_id"] for r in read_rows(data / "train" / "queries.csv") if r["episode_id"] == episode}

        def corrupt(rows):
            return [dict(r, condition="drift") if r["query_id"] in ids else r for r in rows]
        rewrite(data / "train" / "conditions.csv", corrupt)
        self.assert_only_fails(data, {"DAT-COUNTS-PER-CONDITION"})

    def test_machine_missing_from_an_episode_fails(self):
        data = self.dataset()
        first = read_rows(TEST_QUERIES)[0]
        rewrite(data / "test" / "queries.csv",
                lambda rows: [r for r in rows if not (r["episode_id"] == first["episode_id"]
                                                      and r["machine_id"] == first["machine_id"])])
        self.assertIn("DAT-MACHINES-PER-EPISODE", failing(check_dataset(data)))

    def test_query_repeated_across_splits_fails_disjoint(self):
        data = self.dataset()
        copied = read_rows(data / "train" / "queries.csv")[0]
        rewrite(data / "test" / "queries.csv", lambda rows: rows + [copied])
        self.assertIn("DAT-SPLIT-DISJOINT", failing(check_dataset(data)))

    def test_history_at_cutoff_fails_causal(self):
        data = self.dataset()
        query = read_rows(data / "test" / "queries.csv")[0]
        with (data / "test" / "history.csv").open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow([query["episode_id"], query["machine_id"], query["cutoff"], 20])
        self.assertEqual(failing(check_dataset(data)), {"DAT-HISTORY-CAUSAL"})

    def test_duplicate_history_observation_fails_causal(self):
        data = self.dataset()
        rewrite(data / "test" / "history.csv", lambda rows: rows + [rows[0]])
        self.assert_only_fails(data, {"DAT-HISTORY-CAUSAL"})

    def test_metadata_points_mismatch_fails_round_metadata(self):
        data = self.dataset()
        meta = json.loads((data / "metadata.json").read_text())
        meta["max_points"] = 50
        (data / "metadata.json").write_text(json.dumps(meta))
        self.assert_only_fails(data, {"CON-ROUND-METADATA"})

    def test_missing_tick_breaks_history_shape(self):
        data = self.dataset()
        queries = read_rows(data / "train" / "queries.csv")
        history = read_rows(data / "train" / "history.csv")
        counts = {}
        for row in history:
            counts[(row["episode_id"], row["machine_id"])] = counts.get((row["episode_id"], row["machine_id"]), 0) + 1
        long_key = next((q["episode_id"], q["machine_id"]) for q in queries
                        if counts[(q["episode_id"], q["machine_id"])] == int(q["cutoff"]))

        def corrupt(rows):
            return [r for r in rows if not ((r["episode_id"], r["machine_id"], r["tick"]) == (*long_key, "5"))]
        rewrite(data / "train" / "history.csv", corrupt)
        self.assert_only_fails(data, {"GEN-HISTORY-SHAPE"})

    def test_missing_directory_fails_every_schema_style_check(self):
        data = self.dataset()
        shutil.rmtree(data / "validation")
        failed = failing(check_dataset(data))
        self.assertIn("DAT-SCHEMA-COLUMNS", failed)
        self.assertIn("DAT-COUNTS-PER-CONDITION", failed)

    def test_nonexistent_root_fails_everything(self):
        results = check_dataset(self.tmp / "absent")
        self.assertEqual(failing(results), set(DATASET_CHECK_IDS))


class ZipCheckTests(TempCase):
    def test_packaged_uniform_passes_every_zip_check(self):
        zip_path = self.zip_entry()
        results = check_zip(zip_path)
        self.assertEqual([r.id for r in results], list(ZIP_CHECK_IDS))
        self.assertEqual(failing(results), set(), [r.message for r in results if not r.passed])

    def test_answer_file_in_zip_fails_no_answer_files(self):
        zip_path = self.zip_entry(extra={"labels.csv": "query_id,outcome\n"})
        self.assertEqual(failing(check_zip(zip_path)), {"PKG-NO-ANSWER-FILES"})

    def test_truth_directory_in_zip_fails_no_answer_files(self):
        zip_path = self.zip_entry(extra={"_truth/test/targets.csv": "x\n"})
        self.assertIn("PKG-NO-ANSWER-FILES", failing(check_zip(zip_path)))

    def test_unpinned_requirement_fails_pinned(self):
        zip_path = self.zip_entry(requirements="numpy>=1.0\n")
        self.assertEqual(failing(check_zip(zip_path)), {"PKG-REQUIREMENTS-PINNED"})

    def test_exact_pin_passes(self):
        zip_path = self.zip_entry(requirements="# deps\nnumpy==1.26.4\n")
        self.assertEqual(failing(check_zip(zip_path)), set())

    def test_image_python_colon_tag_fails_image(self):
        # package_entry refuses this image itself, so build the ZIP by hand.
        source = self.tmp / "image.zip"
        reproduce = {"image": "python:3.12", "command": ["python", "predict.py", "--data", "/data",
                                                         "--out", "/output/submission.csv"]}
        write_zip(source, {"README.md": "run it\n", "reproduce.json": json.dumps(reproduce),
                           "predict.py": "print('x')\n"})
        self.assertEqual(failing(check_zip(source)), {"CON-IMAGE-PYTHON"})

    def test_nested_root_fails_zip_root(self):
        source = self.tmp / "nested.zip"
        with zipfile.ZipFile(source, "w") as archive:
            for name in ("README.md", "reproduce.json", "predict.py"):
                archive.writestr(f"pkg/{name}", (FIXTURES / "uniform" / name).read_bytes())
        failed = failing(check_zip(source))
        self.assertIn("PKG-ZIP-ROOT", failed)
        self.assertIn("CON-REPRODUCE-SCHEMA", failed)

    def test_path_escape_fails_safe(self):
        source = self.tmp / "escape.zip"
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("README.md", "run it\n")
            archive.writestr("reproduce.json", (FIXTURES / "uniform" / "reproduce.json").read_text())
            archive.writestr("../escape.py", "print('x')\n")
        self.assertIn("PKG-ZIP-SAFE", failing(check_zip(source)))

    def test_size_limit_fails_size(self):
        zip_path = self.zip_entry()
        with mock.patch.object(datachecks, "ZIP_MAX_BYTES", 10):
            self.assertEqual(failing(check_zip(zip_path)), {"PKG-ZIP-SIZE"})

    def test_not_a_zip_fails_every_check(self):
        source = self.tmp / "garbage.zip"
        source.write_bytes(b"not a zip at all")
        results = check_zip(source)
        self.assertEqual([r.id for r in results], list(ZIP_CHECK_IDS))
        # The size check still passes for a small file; every other check cannot read a ZIP.
        self.assertEqual(failing(results), set(ZIP_CHECK_IDS) - {"PKG-ZIP-SIZE"})

    def test_copied_organizer_data_warns(self):
        zip_path = self.zip_entry(extra={"queries.csv": "query_id\n"})
        result = by_id(check_zip(zip_path))["PKG-NO-DATA-COPY"]
        self.assertFalse(result.passed)
        self.assertEqual(result.severity, "warn")

    def test_readme_without_season_warns_and_with_season_passes(self):
        meta = str(PUBLIC / "metadata.json")
        zip_path = self.zip_entry()
        warned = by_id(check_zip(zip_path, meta))["PKG-README-SEASON"]
        self.assertFalse(warned.passed)
        self.assertEqual(warned.severity, "warn")
        season = json.loads((PUBLIC / "metadata.json").read_text())["season_id"]
        good = self.zip_entry(readme=f"Run predict.py. Season {season}, round main.\n")
        self.assertTrue(by_id(check_zip(good, meta))["PKG-README-SEASON"].passed)

    def test_package_entry_refuses_existing_output(self):
        out = self.tmp / "exists.zip"
        out.write_bytes(b"x")
        with self.assertRaises(ValidationError):
            package_entry(FIXTURES / "uniform", out)


class MountInputTests(TempCase):
    def entry(self):
        return FIXTURES / "uniform"

    def test_public_kit_and_uniform_pass(self):
        self.assertEqual(failing(check_mount_inputs(PUBLIC, self.entry())), set())

    def test_truth_directory_fails_mount(self):
        data = self.dataset()
        (data / "_truth" / "test").mkdir(parents=True)
        (data / "_truth" / "test" / "targets.csv").write_text("x\n")
        self.assertEqual(failing(check_mount_inputs(data, self.entry())), {"ROB-MOUNT-NO-ANSWERS"})

    def test_answer_file_in_test_fails_mount(self):
        data = self.dataset()
        shutil.copy(data / "validation" / "targets.csv", data / "test" / "targets.csv")
        self.assertEqual(failing(check_mount_inputs(data, self.entry())), {"ROB-MOUNT-NO-ANSWERS"})

    def test_sample_submission_is_allowed(self):
        self.assertTrue((PUBLIC / "test" / "sample_submission.csv").exists())
        self.assertNotIn("ROB-MOUNT-NO-ANSWERS", failing(check_mount_inputs(PUBLIC, self.entry())))

    def test_precomputed_forecast_in_entry_fails(self):
        entry = self.tmp / "entry"
        shutil.copytree(self.entry(), entry)
        (entry / "submission.csv").write_text("query_id\n")
        self.assertEqual(failing(check_mount_inputs(PUBLIC, entry)), {"ROB-NO-FORECAST-INPUT"})

    def dataset(self):
        target = self.tmp / "data"
        shutil.copytree(PUBLIC, target)
        return target


class CrossEpisodeVariantTests(TempCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls._tmp.name)
        cls.variants = {v.name: v for v in make_variants(PUBLIC, cls.out / "variants", "test", seed=0)}
        cls.variant = cls.variants[CROSS_EPISODE_ID]
        cls.base_queries = read_rows(TEST_QUERIES)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_variant_is_identical_subset_on_one_episode(self):
        self.assertEqual(self.variant.relation, "identical_subset")
        self.assertEqual(self.variant.requirement_ids, [CROSS_EPISODE_ID])
        kept = {q["episode_id"] for q in self.base_queries if q["query_id"] in set(self.variant.subset)}
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(self.variant.subset), 8)

    def test_only_other_episode_payouts_change_and_ticks_stay(self):
        base = read_rows(PUBLIC / "test" / "history.csv")
        changed = read_rows(self.variant.data_root / "test" / "history.csv")
        kept = {q["episode_id"] for q in self.base_queries if q["query_id"] in set(self.variant.subset)}.pop()
        self.assertEqual([(r["episode_id"], r["machine_id"], r["tick"]) for r in base],
                         [(r["episode_id"], r["machine_id"], r["tick"]) for r in changed])
        for before, after in zip(base, changed):
            if before["episode_id"] == kept:
                self.assertEqual(before["payout"], after["payout"])
        self.assertTrue(any(b["payout"] != a["payout"] for b, a in zip(base, changed)
                            if b["episode_id"] != kept))
        for name in ("train", "validation"):
            self.assertEqual((PUBLIC / name / "history.csv").read_bytes(),
                             (self.variant.data_root / name / "history.csv").read_bytes())

    def write_predictions(self, path, overrides=None):
        with Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["query_id", *PROBS])
            for row in self.base_queries:
                probs = (overrides or {}).get(row["query_id"], (0.2, 0.2, 0.2, 0.2, 0.2))
                writer.writerow([row["query_id"], *(repr(p) for p in probs)])
        return Path(path)

    def test_compare_ignores_other_episodes_and_catches_the_subset(self):
        base = self.write_predictions(self.tmp / "base.csv")
        other = [r["query_id"] for r in self.base_queries if r["query_id"] not in set(self.variant.subset)]
        same_for_kept = self.write_predictions(self.tmp / "other.csv",
                                               {q: (0.5, 0.5, 0.0, 0.0, 0.0) for q in other})
        result = compare(base, same_for_kept, self.variant)
        self.assertTrue(result.passed, result.message)
        self.assertEqual(result.severity, "info")
        self.assertEqual(result.details["n_queries"], 8)

        leaked = self.write_predictions(self.tmp / "leak.csv", {self.variant.subset[0]: (0.5, 0.5, 0.0, 0.0, 0.0)})
        result = compare(base, leaked, self.variant)
        self.assertFalse(result.passed)
        self.assertEqual(result.severity, "fail")
        self.assertEqual(result.details["n_differing"], 1)

    def test_identical_subset_needs_a_subset(self):
        from eval_suite.perturb import Variant
        with self.assertRaises(ValueError):
            Variant(name="x", data_root=self.tmp, relation="identical_subset")
        with self.assertRaises(ValueError):
            Variant(name="x", data_root=self.tmp, relation="identical", subset=("q",))


DOCKER_OK, DOCKER_MESSAGE = docker_available()


@unittest.skipUnless(DOCKER_OK, f"docker unavailable: {DOCKER_MESSAGE}")
class DockerRunTests(TempCase):
    def test_uniform_on_public_runs_offline_with_limits_recorded(self):
        result = run_entry(FIXTURES / "uniform", PUBLIC, mode="docker", timeout_s=600)
        self.assertEqual(result.exit_code, 0, result.stderr_tail)
        self.assertIsNotNone(result.csv_path, "no /output/submission.csv was copied out")
        self.assertEqual([r.id for r in check_csv(result.csv_path, TEST_QUERIES) if not r.passed], [])
        limits = result.limits
        self.assertEqual(limits["source"], "docker inspect")
        self.assertEqual(limits["cpus"], 2.0)
        self.assertEqual(limits["memory_bytes"], 16 * 2**30)
        self.assertEqual(limits["memory_swap_bytes"], 16 * 2**30)
        self.assertEqual(limits["network"], "none")
        self.assertEqual(limits["working_dir"], "/submission")
        self.assertEqual(limits["tmpfs"]["/tmp"], "rw,size=134217728")
        self.assertEqual(limits["tmpfs_size_bytes"]["/output"], 33554432)
        mounts = {m["destination"]: m for m in limits["mounts"]}
        self.assertFalse(mounts["/submission"]["rw"])
        self.assertFalse(mounts["/data"]["rw"])

    def entry_with_command(self, command):
        folder = self.tmp / "entry"
        folder.mkdir(exist_ok=True)
        (folder / "README.md").write_text("probe\n")
        (folder / "reproduce.json").write_text(json.dumps({"image": "python", "command": command}))
        return folder

    def test_output_quota_is_enforced(self):
        probe = "open('/output/submission.csv','wb').write(b'x' * (40 * 2**20))"
        entry = self.entry_with_command(["python", "-c", probe])
        result = run_entry(entry, PUBLIC, mode="docker", timeout_s=300)
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("No space left on device", result.stderr_tail)

    def test_data_mount_is_read_only_and_network_is_off(self):
        entry = self.entry_with_command(["python", "-c", "open('/data/probe.txt','w').write('x')"])
        self.assertNotEqual(run_entry(entry, PUBLIC, mode="docker", timeout_s=300).exit_code, 0)
        network = ("import socket\n"
                   "socket.create_connection(('1.1.1.1', 80), timeout=5)\n")
        entry = self.entry_with_command(["python", "-c", network])
        self.assertNotEqual(run_entry(entry, PUBLIC, mode="docker", timeout_s=300).exit_code, 0)


if __name__ == "__main__":
    unittest.main()
