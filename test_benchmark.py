import unittest
from collections import defaultdict

from benchmark import KINDS, fixture, generate


class SyntheticBenchmarkTest(unittest.TestCase):
    def test_repeatability(self):
        self.assertEqual(list(generate(10)), list(generate(10)))

    def test_grouped_split(self):
        groups = defaultdict(set)
        for sample in generate(100):
            groups[sample["seed"]].add(sample["split"])
        self.assertTrue(all(len(group) == 1 for group in groups.values()))
        self.assertEqual(set.union(*groups.values()), {"train", "calibration", "test"})

    def test_equal_count_defects(self):
        for kind in ("rounded_decimal", "null_to_empty", "missing_grant",
                     "extra_grant", "type_drift", "timezone_shift"):
            sample = fixture(42, kind)
            self.assertTrue(sample["defect"], kind)
            self.assertEqual(len(sample["source"]["rows"]), len(sample["target"]["rows"]), kind)
            self.assertNotEqual(sample["source"], sample["target"], kind)

    def test_reorder_is_benign(self):
        sample = fixture(42, "reorder")
        self.assertFalse(sample["defect"])
        self.assertNotEqual(sample["source"]["rows"], sample["target"]["rows"])
        self.assertEqual(sorted(sample["source"]["rows"], key=lambda r: r["id"]),
                         sorted(sample["target"]["rows"], key=lambda r: r["id"]))

    def test_approved_padding_is_benign(self):
        sample = fixture(42, "approved_padding")
        self.assertFalse(sample["defect"])
        self.assertTrue(any(row["account"].endswith("  ") for row in sample["target"]["rows"]))
        self.assertTrue(all(not row["account"].endswith(" ") for row in sample["source"]["rows"]))

    def test_all_cases(self):
        self.assertEqual(len(list(generate(1))), len(KINDS))


if __name__ == "__main__":
    unittest.main()
