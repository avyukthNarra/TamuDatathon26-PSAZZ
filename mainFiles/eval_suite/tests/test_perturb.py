"""Tests of the perturbation suite itself, built from the real kit at public/."""

import csv
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from eval_suite.perturb import (
    ANSWER_FILES, CONDITIONS_FILE, FUTURE_TICKS, PROBABILITY_COLUMNS, PROBE_LOOKBACK,
    Variant, compare, compare_missing_vs_zero, make_variants,
)

REPO = Path(__file__).resolve().parents[2]
PUBLIC = REPO / "public"
EXPECTED_NAMES = [
    "CAU-FUTURE-TICKS", "CAU-SAME-TICK-OTHERS", "CAU-CONDITIONS-HIDDEN", "CAU-ANSWERS-HIDDEN",
    "ROB-ROW-ORDER", "ROB-ID-RENAME", "INFO-MISSING-VS-ZERO-missing", "INFO-MISSING-VS-ZERO-zero",
]


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def history_key(row):
    return (row["episode_id"], row["machine_id"], int(row["tick"]), int(row["payout"]))


def write_predictions(path, query_ids, rng, overrides=None):
    """Write a prediction CSV; overrides maps query_id to a replacement tuple."""
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["query_id", *PROBABILITY_COLUMNS])
        for qid in query_ids:
            # Always draw, so overriding one row does not shift the others.
            raw = [rng.random() + 0.01 for _ in PROBABILITY_COLUMNS]
            probs = [v / sum(raw) for v in raw]
            if overrides and qid in overrides:
                probs = overrides[qid]
            writer.writerow([qid, *(repr(float(p)) for p in probs)])


class VariantsMixin:
    """Shared checks; split is set by the concrete TestCase subclass."""

    split = None

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls._tmp.name)
        cls.variants = {v.name: v for v in make_variants(PUBLIC, cls.out / cls.split, cls.split, seed=0)}
        cls.base_queries = read_rows(PUBLIC / cls.split / "queries.csv")
        cls.base_history = read_rows(PUBLIC / cls.split / "history.csv")
        cls.base_cutoff = {(q["episode_id"], q["machine_id"]): int(q["cutoff"]) for q in cls.base_queries}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def variant_dir(self, name):
        return self.variants[name].data_root

    def assert_other_splits_unchanged(self, name):
        """Only the scored split may differ from the base dataset."""
        base_root, var_root = PUBLIC, self.variant_dir(name)
        for base_file in base_root.rglob("*"):
            if not base_file.is_file():
                continue
            rel = base_file.relative_to(base_root)
            if rel.parts[0] == self.split:
                continue
            self.assertEqual(base_file.read_bytes(), (var_root / rel).read_bytes(), str(rel))

    def assert_scored_split_unchanged_except(self, name, changed):
        for base_file in (PUBLIC / self.split).iterdir():
            if base_file.name in changed:
                continue
            var_file = self.variant_dir(name) / self.split / base_file.name
            self.assertTrue(var_file.exists(), f"{base_file.name} should be kept")
            self.assertEqual(base_file.read_bytes(), var_file.read_bytes(), base_file.name)

    def test_variant_names_and_relations(self):
        self.assertEqual(sorted(self.variants), sorted(EXPECTED_NAMES))
        self.assertEqual(self.variants["ROB-ID-RENAME"].relation, "identical_after_map")
        self.assertIsNotNone(self.variants["ROB-ID-RENAME"].id_map)
        for name in EXPECTED_NAMES:
            if name != "ROB-ID-RENAME":
                self.assertIsNone(self.variants[name].id_map, name)
            if name.startswith("INFO-"):
                self.assertEqual(self.variants[name].relation, "info")
            else:
                self.assertIn(self.variants[name].relation, ("identical", "identical_after_map"))

    def test_other_splits_are_byte_identical(self):
        for name in EXPECTED_NAMES:
            with self.subTest(variant=name):
                self.assert_other_splits_unchanged(name)

    def test_future_ticks_only_append_at_or_after_cutoff(self):
        name = "CAU-FUTURE-TICKS"
        rows = read_rows(self.variant_dir(name) / self.split / "history.csv")
        self.assertEqual(rows[:len(self.base_history)], self.base_history)
        added = rows[len(self.base_history):]
        self.assertEqual(len(added), FUTURE_TICKS * len(self.base_queries))
        per_machine = Counter((r["episode_id"], r["machine_id"]) for r in added)
        self.assertEqual(set(per_machine.values()), {FUTURE_TICKS})
        for row in added:
            cutoff = self.base_cutoff[(row["episode_id"], row["machine_id"])]
            tick = int(row["tick"])
            self.assertTrue(cutoff <= tick < cutoff + FUTURE_TICKS, row)
            self.assertIn(int(row["payout"]), (0, 1, 2, 5, 20))
        self.assert_scored_split_unchanged_except(name, changed={"history.csv"})

    def test_same_tick_rows_sit_at_each_machines_cutoff(self):
        name = "CAU-SAME-TICK-OTHERS"
        rows = read_rows(self.variant_dir(name) / self.split / "history.csv")
        self.assertEqual(rows[:len(self.base_history)], self.base_history)
        added = rows[len(self.base_history):]
        # One added row per machine (every machine has a query), all at tick == cutoff.
        self.assertEqual(len(added), len(self.base_queries))
        self.assertEqual({(r["episode_id"], r["machine_id"]) for r in added}, set(self.base_cutoff))
        for row in added:
            self.assertEqual(int(row["tick"]), self.base_cutoff[(row["episode_id"], row["machine_id"])])
        self.assert_scored_split_unchanged_except(name, changed={"history.csv"})

    def test_conditions_removed_from_scored_split(self):
        name = "CAU-CONDITIONS-HIDDEN"
        self.assertFalse((self.variant_dir(name) / self.split / CONDITIONS_FILE).exists())
        self.assert_scored_split_unchanged_except(name, changed={CONDITIONS_FILE})

    def test_answer_files_removed_from_scored_split(self):
        name = "CAU-ANSWERS-HIDDEN"
        for answer in ANSWER_FILES:
            self.assertFalse((self.variant_dir(name) / self.split / answer).exists(), answer)
        self.assert_scored_split_unchanged_except(name, changed=set(ANSWER_FILES))

    def test_row_order_shuffled_but_same_rows(self):
        name = "ROB-ROW-ORDER"
        for filename, base_rows in (("queries.csv", self.base_queries), ("history.csv", self.base_history)):
            var_rows = read_rows(self.variant_dir(name) / self.split / filename)
            self.assertEqual(sorted(map(str, var_rows)), sorted(map(str, base_rows)), filename)
            self.assertNotEqual(var_rows, base_rows, f"{filename} order should change")
        self.assert_scored_split_unchanged_except(name, changed={"queries.csv", "history.csv"})

    def test_id_rename_is_consistent_and_keeps_shape(self):
        name = "ROB-ID-RENAME"
        variant = self.variants[name]
        var_queries = read_rows(variant.data_root / self.split / "queries.csv")
        var_history = read_rows(variant.data_root / self.split / "history.csv")
        self.assertEqual(len(var_queries), len(self.base_queries))

        # Row i keeps its position, so the id_map must send it back to base row i.
        for base_row, var_row in zip(self.base_queries, var_queries):
            self.assertNotEqual(var_row["query_id"], base_row["query_id"])
            self.assertEqual(variant.id_map[var_row["query_id"]], base_row["query_id"])
            self.assertEqual(var_row["cutoff"], base_row["cutoff"])
            self.assertEqual(var_row["machine_type"], base_row["machine_type"])
            self.assertEqual(var_row["signal"], base_row["signal"])
        self.assertEqual(set(variant.id_map.values()), {r["query_id"] for r in self.base_queries})

        # Episode and machine maps must be one-to-one and agree across all files.
        episode_map, machine_map = {}, {}
        for base_row, var_row in zip(self.base_queries, var_queries):
            self.assertEqual(episode_map.setdefault(base_row["episode_id"], var_row["episode_id"]),
                             var_row["episode_id"])
            self.assertEqual(machine_map.setdefault(base_row["machine_id"], var_row["machine_id"]),
                             var_row["machine_id"])
        self.assertEqual(len(set(episode_map.values())), len(episode_map))
        self.assertEqual(len(set(machine_map.values())), len(machine_map))

        for var_row in var_queries:
            self.assertRegex(var_row["episode_id"], rf"^main-{self.split}-e\d{{5}}$")
            self.assertRegex(var_row["machine_id"], rf"^main-{self.split}-e\d{{5}}-m\d{{3}}$")
            self.assertTrue(var_row["query_id"].endswith("-next"))

        # History is the base history with ids mapped.
        expected_history = sorted(
            (episode_map[r["episode_id"]], machine_map[r["machine_id"]], int(r["tick"]), int(r["payout"]))
            for r in self.base_history)
        actual_history = sorted(history_key(r) for r in var_history)
        self.assertEqual(actual_history, expected_history)

        # Every scored-split file with query ids uses only renamed ids.
        new_ids = {r["query_id"] for r in var_queries}
        for path in (variant.data_root / self.split).glob("*.csv"):
            for row in read_rows(path):
                if "query_id" in row:
                    self.assertIn(row["query_id"], new_ids, path.name)

    def test_info_probe_deletes_or_zeroes_same_recent_rows(self):
        missing_root = self.variant_dir("INFO-MISSING-VS-ZERO-missing")
        zero_root = self.variant_dir("INFO-MISSING-VS-ZERO-zero")
        missing = read_rows(missing_root / self.split / "history.csv")
        zero = read_rows(zero_root / self.split / "history.csv")
        self.assertEqual(len(zero), len(self.base_history))
        self.assertLess(len(missing), len(self.base_history))

        removed = Counter(map(history_key, self.base_history)) - Counter(map(history_key, missing))
        removed_keys = list(removed.elements())
        self.assertGreater(len(removed_keys), 0)
        self.assertLessEqual(len(removed_keys), 16)
        for ep, machine, tick, _ in removed_keys:
            cutoff = self.base_cutoff[(ep, machine)]
            self.assertTrue(cutoff - PROBE_LOOKBACK <= tick < cutoff, (ep, machine, tick))
        self.assertEqual(len(removed_keys), len({(e, m) for e, m, _, _ in removed_keys}))

        # Zero variant: same keys as base, payouts changed only at the probe rows, and set to 0.
        self.assertEqual([(r["episode_id"], r["machine_id"], r["tick"]) for r in zero],
                         [(r["episode_id"], r["machine_id"], r["tick"]) for r in self.base_history])
        changed = [(z, b) for z, b in zip(zero, self.base_history) if z["payout"] != b["payout"]]
        self.assertTrue(changed)
        removed_set = {(e, m, t) for e, m, t, _ in removed_keys}
        for z, _ in changed:
            self.assertEqual(z["payout"], "0")
            self.assertIn((z["episode_id"], z["machine_id"], int(z["tick"])), removed_set)

        self.assert_scored_split_unchanged_except(
            "INFO-MISSING-VS-ZERO-missing", changed={"history.csv"})
        self.assert_scored_split_unchanged_except(
            "INFO-MISSING-VS-ZERO-zero", changed={"history.csv"})


class TestValidationVariants(VariantsMixin, unittest.TestCase):
    split = "validation"

    def test_validation_conditions_and_answers_are_removed(self):
        # The real validation split has all answer files; the variants must remove them.
        self.assertTrue((PUBLIC / self.split / CONDITIONS_FILE).exists())
        self.assertFalse((self.variant_dir("CAU-CONDITIONS-HIDDEN") / self.split / CONDITIONS_FILE).exists())
        answers = self.variant_dir("CAU-ANSWERS-HIDDEN") / self.split
        for answer in ANSWER_FILES:
            self.assertFalse((answers / answer).exists(), answer)
        # Conditions are not an answer file, so the answers variant keeps them.
        self.assertTrue((answers / CONDITIONS_FILE).exists())


class TestTestVariants(VariantsMixin, unittest.TestCase):
    split = "test"

    def test_test_split_has_no_answer_or_condition_files(self):
        # The real test split ships without answers or conditions, so those variants are no-ops.
        for name in ("CAU-CONDITIONS-HIDDEN", "CAU-ANSWERS-HIDDEN"):
            with self.subTest(variant=name):
                self.assertFalse((PUBLIC / self.split / CONDITIONS_FILE).exists())
                self.assertEqual(
                    (self.variant_dir(name) / self.split / "queries.csv").read_bytes(),
                    (PUBLIC / self.split / "queries.csv").read_bytes())

    def test_kit_loader_rejects_future_history_variants(self):
        # The kit's leak guard refuses history at/after cutoff, so the appending variants do not load.
        from slot_forecasting.io import load_inputs
        from slot_forecasting.schema import ValidationError

        load_inputs(PUBLIC / self.split)  # base loads
        for name in ("CAU-FUTURE-TICKS", "CAU-SAME-TICK-OTHERS"):
            with self.subTest(variant=name), self.assertRaises(ValidationError):
                load_inputs(self.variant_dir(name) / self.split)


class CompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls._tmp.name)
        cls.variants = {v.name: v for v in make_variants(PUBLIC, cls.out / "data", "test", seed=0)}
        cls.queries = read_rows(PUBLIC / "test" / "queries.csv")
        cls.query_ids = [q["query_id"] for q in cls.queries]
        rng = random.Random(1)
        cls.base_csv = cls.out / "base.csv"
        write_predictions(cls.base_csv, cls.query_ids, rng)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def variant(self, name="CAU-FUTURE-TICKS", relation="identical", id_map=None):
        return Variant(name=name, data_root=self.out, relation=relation, id_map=id_map)

    def test_identical_csv_passes(self):
        copy = self.out / "same.csv"
        copy.write_bytes(self.base_csv.read_bytes())
        result = compare(self.base_csv, copy, self.variant())
        self.assertTrue(result.passed)
        self.assertEqual(result.details["n_differing"], 0)
        self.assertEqual(result.details["max_abs_diff"], 0.0)

    def test_one_probability_changed_by_1e6_fails(self):
        read = read_rows(self.base_csv)
        value = float(read[0]["p_0"]) + 1e-6
        changed = self.out / "changed.csv"
        write_predictions(changed, self.query_ids, random.Random(1),
                          overrides={read[0]["query_id"]: [value] + [float(read[0][c]) for c in PROBABILITY_COLUMNS[1:]]})
        result = compare(self.base_csv, changed, self.variant(), tol=1e-9)
        self.assertFalse(result.passed)
        self.assertEqual(result.details["n_differing"], 1)
        self.assertAlmostEqual(result.details["max_abs_diff"], 1e-6, places=9)
        self.assertEqual(result.severity, "fail")

    def test_identical_after_map_uses_id_map(self):
        variant = self.variants["ROB-ID-RENAME"]
        # Write the base predictions under the renamed ids, the way a rename-invariant entry would.
        inverse = {old: new for new, old in variant.id_map.items()}
        renamed = self.out / "renamed.csv"
        base_rows = read_rows(self.base_csv)
        with renamed.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["query_id", *PROBABILITY_COLUMNS])
            for row in base_rows:
                writer.writerow([inverse[row["query_id"]], *(row[c] for c in PROBABILITY_COLUMNS)])
        self.assertTrue(compare(self.base_csv, renamed, variant).passed)

        # Keep the rename but change one probability: must fail.
        read = read_rows(renamed)
        read[3]["p_2"] = repr(float(read[3]["p_2"]) + 1e-6)
        with renamed.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["query_id", *PROBABILITY_COLUMNS], lineterminator="\n")
            writer.writeheader()
            writer.writerows(read)
        self.assertFalse(compare(self.base_csv, renamed, variant).passed)

    def test_identical_after_map_fails_on_unmapped_ids(self):
        result = compare(self.base_csv, self.base_csv, self.variants["ROB-ID-RENAME"])
        self.assertFalse(result.passed)

    def test_id_mismatch_fails(self):
        short = self.out / "short.csv"
        write_predictions(short, self.query_ids[:-1], random.Random(2))
        result = compare(self.base_csv, short, self.variant())
        self.assertFalse(result.passed)
        self.assertIn("mismatch", result.message)

    def test_info_relation_never_fails(self):
        read = read_rows(self.base_csv)
        changed = self.out / "info_changed.csv"
        write_predictions(changed, self.query_ids, random.Random(1),
                          overrides={read[0]["query_id"]: [0.5, 0.2, 0.1, 0.1, 0.1]})
        result = compare(self.base_csv, changed, self.variant(relation="info"))
        self.assertTrue(result.passed)
        self.assertEqual(result.severity, "info")
        self.assertEqual(result.details["n_differing"], 1)

    def test_missing_vs_zero_identical_is_warning(self):
        result = compare_missing_vs_zero(self.base_csv, self.base_csv)
        self.assertTrue(result.passed)
        self.assertEqual(result.severity, "warn")

    def test_missing_vs_zero_different_is_info(self):
        read = read_rows(self.base_csv)
        zero = self.out / "zero.csv"
        write_predictions(zero, self.query_ids, random.Random(1),
                          overrides={read[5]["query_id"]: [0.4, 0.2, 0.2, 0.1, 0.1]})
        result = compare_missing_vs_zero(self.base_csv, zero)
        self.assertTrue(result.passed)
        self.assertEqual(result.severity, "info")
        self.assertEqual(result.details["n_differing"], 1)

    def test_missing_vs_zero_noise_below_tol_is_still_warning(self):
        read = read_rows(self.base_csv)
        noisy = self.out / "noisy.csv"
        write_predictions(noisy, self.query_ids, random.Random(1),
                          overrides={read[0]["query_id"]: [float(read[0][c]) + 1e-12 for c in PROBABILITY_COLUMNS]})
        self.assertEqual(compare_missing_vs_zero(self.base_csv, noisy, tol=1e-9).severity, "warn")


if __name__ == "__main__":
    unittest.main()
