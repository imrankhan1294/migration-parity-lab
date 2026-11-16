import random
import unittest

from budget_policy import (
    Learned, Session, calibration_threshold, disposition, make_cases, run,
)


class BudgetPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = list(make_cases(100))
        cls.model = Learned(r for r in cls.rows if r[0] == "train")
        cls.outputs = next(r[3] for r in cls.rows if r[0] == "test")

    def test_group_split_and_repeatability(self):
        self.assertEqual(self.rows, list(make_cases(100)))
        self.assertEqual({r[0] for r in self.rows}, {"calibration", "test", "train"})
        for i in range(0, len(self.rows), 5):
            self.assertEqual(len({r[0] for r in self.rows[i:i + 5]}), 1)

    def test_every_policy_buys_exact_budget(self):
        for method in ("learned", "random", "fixed", "heuristic"):
            for budget in range(1, 5):
                score, trace = run(method, self.model, self.outputs, budget, random.Random(7))
                self.assertEqual(len({t["tool"] for t in trace}), budget, (method, budget))
                self.assertTrue(0 < score < 1)
                for item in trace:
                    self.assertEqual(item["observed"], self.outputs[item["tool"]])

    def test_unseen_probe_outputs_cannot_affect_policy(self):
        # Every policy is blind to this budget-1 unbought observation.
        for method in ("learned", "random", "fixed", "heuristic"):
            before = run(method, self.model, self.outputs, 1, random.Random(8))
            selected = before[1][0]["tool"]
            unseen = next(x for x in ("values", "schema", "grants", "counts") if x != selected)
            altered = {**self.outputs, unseen: "hit" if self.outputs[unseen] == "clear" else "clear"}
            self.assertEqual(before, run(method, self.model, altered, 1, random.Random(8)))

    def test_session_rejects_over_budget_and_repeat(self):
        session = Session(self.outputs, 2)
        session.inspect("counts", "test")
        with self.assertRaises(ValueError):
            session.inspect("counts", "duplicate")
        session.inspect("grants", "test")
        with self.assertRaises(ValueError):
            session.inspect("schema", "over")

    def test_observed_alert_blocks_and_clear_is_not_go(self):
        self.assertEqual(disposition([{"tool": "grants", "observed": "hit"}])["decision"],
                         "HOLD_FOR_REVIEW")
        self.assertEqual(disposition([{"tool": "counts", "observed": "clear"}])["decision"],
                         "INCOMPLETE_EVIDENCE")

    def test_calibration_constraint(self):
        scores = [(0.7, False), (0.1, False), (0.8, True), (0.55, True)]
        self.assertEqual(calibration_threshold(scores, target_fpr=0), 0.8)
        self.assertEqual(calibration_threshold(scores, target_fpr=0.5), 0.55)


if __name__ == "__main__":
    unittest.main()
