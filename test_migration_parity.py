import csv
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from migration_parity import compare, main


class ParityTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.source, self.target = (Path(directory.name) / name for name in ("s.csv", "t.csv"))

    def compare(self, source, target, keys=("id",)):
        self.source.write_text(source, encoding="utf-8")
        self.target.write_text(target, encoding="utf-8")
        return compare(self.source, self.target, list(keys))

    def test_match_ignores_row_and_column_order(self):
        result = self.compare("id,value\n1,cat\n2,dog\n", "value,id\ndog,2\ncat,1\n")
        self.assertEqual(result["status"], "match")
        self.assertTrue(result["schema"]["reordered"])

    def test_composite_key_drift_counts_and_omits_values(self):
        examples = Path(__file__).resolve().parent
        report = compare(examples / "source.csv", examples / "target.csv", ["tenant_id", "widget_id"])
        self.assertEqual(report["status"], "drift")
        self.assertEqual(report["source"]["nulls_by_column"]["quota"], 1)
        self.assertEqual(report["target"]["duplicate_keys"], 1)
        self.assertEqual(report["target"]["extra_duplicate_rows"], 1)
        self.assertEqual(report["schema"]["source_only_columns"], ["legacy_tag"])
        self.assertEqual(report["comparison"], {
            "source_only_keys": 1, "target_only_keys": 1, "compared_keys": 2,
            "skipped_ambiguous_keys": 1, "changed_keys": 1, "mismatches_by_column": {"quota": 1},
        })
        self.assertNotIn("lab-west", json.dumps(report))

    def test_blank_and_lexical_keys(self):
        report = self.compare("id,value\n01,a\n,b\n", "id,value\n1,a\n")
        self.assertEqual(report["source"]["blank_key_rows"], 1)
        self.assertEqual(report["comparison"]["source_only_keys"], 1)
        self.assertEqual(report["comparison"]["target_only_keys"], 1)

    def test_rejects_invalid_records_headers_and_keys(self):
        for bad_csv in ("", "id,id\n1,2\n", "id,value\n1\n", 'id,value\n1,"unterminated', "name\n1\n"):
            with self.subTest(csv=bad_csv), self.assertRaises((ValueError, csv.Error)):
                self.compare(bad_csv, "id,value\n1,x\n")
        with self.assertRaises(ValueError):
            self.compare("id\n", "id\n", ("id", "id"))

    def test_cli_status_and_json(self):
        self.compare("id,value\n1,cat\n", "id,value\n1,dog\n")
        output = io.StringIO()
        argv = [str(self.source), str(self.target), "--key", "id", "--format", "json"]
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(argv), 1)
        self.assertEqual(json.loads(output.getvalue())["comparison"]["changed_keys"], 1)
        self.target.unlink()
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            self.assertEqual(main(argv), 2)
        self.assertIn("migration-parity:", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
