import math
import unittest
from unittest.mock import patch

import review_queue_frontier as queue


HOLD = "HOLD_FOR_REVIEW"
ABS = "INCOMPLETE_EVIDENCE"
ROWS = [(1, .1, 1, HOLD), (2, .9, 1, ABS),
        (3, .9, 0, HOLD), (4, .4, 0, ABS)]


class QueueTests(unittest.TestCase):
    def test_warn_and_hold_are_deduplicated(self):
        row = queue.Profile(ROWS).at(.9)
        self.assertEqual((row["hold"], row["warn"], row["warn_and_hold"]), (2, 2, 1))
        self.assertEqual((row["queued"], row["abstain"], row["unqueued_abstain"]), (3, 2, 1))
        self.assertEqual((row["tp"], row["fp"], row["fn"], row["tn"]), (2, 1, 0, 1))
        self.assertEqual(row["go"], 0)
        self.assertEqual(row["warn_and_abstain"], 1)

    def test_no_warn_cannot_remove_hold(self):
        row = queue.Profile(ROWS).at(math.inf)
        self.assertEqual((row["queued"], row["hold"], row["warn"]), (2, 2, 0))
        self.assertFalse(queue.within(row, 49))
        self.assertTrue(queue.within(row, 50))

    def test_profiles_match_direct_counting_and_atomic_ties(self):
        profile = queue.Profile(ROWS)
        for cut in (-math.inf, .1, .4, .9, 1, math.inf):
            selected = [r for r in ROWS if r[1] >= cut or r[3] == HOLD]
            result = profile.at(cut)
            self.assertEqual(result["queued"], len(selected))
            self.assertEqual(result["tp"], sum(r[2] for r in selected))
            self.assertEqual(result["fp"], sum(not r[2] for r in selected))

    def test_infeasible_families_are_not_silently_admitted(self):
        table = queue.sweep({f: {r: ROWS for r in queue.REGIMES} for f in ("A", "B")})
        result = queue.select(table, [49, 50, 75], {"fixed": ("A",), "hybrid": ("B",)})
        self.assertEqual(result["49"], {"fixed": None, "hybrid": None})
        self.assertEqual(result["50"]["fixed"][1], math.inf)
        self.assertEqual(result["75"]["fixed"][1], .9)

    def test_third_calibration_regime_can_veto_choice(self):
        low = [(i, score, y, ABS) for i, score, y, _ in ROWS]
        by_regime = dict(zip(queue.REGIMES, [low, low, ROWS]))
        table = queue.sweep({"A": by_regime})
        self.assertIsNone(queue.select(table, [49], {"fixed": ("A",)})["49"]["fixed"])

    def test_refit_uses_all_fixed_candidates_and_freezes_before_held_out(self):
        real_select = queue.select
        real_rows = queue.regime_rows
        calls = []
        def rows(*args):
            calls.append("rows")
            return real_rows(*args)
        def selection(*args, **kwargs):
            calls.append("select")
            return real_select(*args, **kwargs)
        with patch.object(queue, "select", selection), patch.object(queue, "regime_rows", rows):
            result = queue.one_run(0, 10, queue.SEED, [100])
        self.assertEqual(calls, ["rows"] * 4 + ["select"] + ["rows"] * 3)
        self.assertEqual(len(result["evidence_queue_floors"]), 11)
        self.assertEqual(len(set(result["group_starts"])), 7)

    def test_zero_budget_has_no_paired_bootstrap(self):
        result = queue.evaluate(repeats=2, groups=10, draws=100, budgets=[0])
        point = result["frontier"]["0"]
        self.assertEqual(point["paired_feasible_runs"], 0)
        for regime in queue.REGIMES:
            self.assertIsNone(point["regimes"][regime]["paired_hybrid_minus_fixed"])

    def test_invalid_inputs_fail_closed(self):
        for rows in ([(1, .1, 1, ABS)],
                     [(1, math.nan, 1, ABS), (2, .2, 0, ABS)],
                     [(1, .1, 1, "GO"), (2, .2, 0, ABS)]):
            with self.assertRaises(ValueError):
                queue.Profile(rows)
        for budgets in ([20, 10], [10, 10], [101], [True], []):
            with self.assertRaises(ValueError):
                queue.evaluate(budgets=budgets)


if __name__ == "__main__":
    unittest.main()
