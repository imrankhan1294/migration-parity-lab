from pathlib import Path
import unittest

from cutover_agent import investigate


EXAMPLES = Path(__file__).resolve().parent
SOURCE = EXAMPLES / "source.csv"
TARGET = EXAMPLES / "target.csv"
KEYS = ["tenant_id", "widget_id"]


class CutoverAgentTest(unittest.TestCase):
    def test_dispatches_relevant_checks_and_explains_no_go(self):
        result = investigate(SOURCE, TARGET, KEYS)
        self.assertEqual(result["recommendation"], "no-go")
        self.assertEqual(result["severity"], "blocking")
        self.assertEqual([event["tool"] for event in result["trace"]],
                         ["inspect_inputs", "compare_keyed_snapshots",
                          "inspect_key_integrity", "inspect_value_coverage"])
        self.assertEqual(result["trace"][2]["evidence"]["target_duplicate_keys"], 1)
        self.assertEqual(result["trace"][3]["evidence"]["comparison"]["changed_keys"], 1)

    def test_match_does_not_dispatch_diagnostic_checks(self):
        result = investigate(SOURCE, SOURCE, KEYS)
        self.assertEqual(result["recommendation"], "go-for-csv-parity")
        self.assertEqual(len(result["trace"]), 2)

    def test_stops_at_budget_and_fails_closed_without_comparison(self):
        result = investigate(SOURCE, TARGET, KEYS, tool_budget=1)
        self.assertEqual(result["severity"], "unknown")
        self.assertEqual(len(result["trace"]), 1)
        result = investigate(SOURCE, TARGET, KEYS, tool_budget=2)
        self.assertEqual(result["severity"], "blocking")
        self.assertEqual(len(result["trace"]), 2)

    def test_refuses_oversized_or_missing_inputs(self):
        for result in (investigate(SOURCE, TARGET, KEYS, max_file_bytes=1),
                       investigate(SOURCE, EXAMPLES / "missing.csv", KEYS)):
            self.assertEqual(result["severity"], "unknown")
            self.assertEqual(len(result["trace"]), 1)
