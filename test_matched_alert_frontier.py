"""Invariants for the calibration-only matched warning-budget experiment."""
import itertools
import json
import math
import unittest
from unittest.mock import patch

import matched_alert_frontier as f


HOLD, ABSTAIN = "HOLD_FOR_REVIEW", "INCOMPLETE_EVIDENCE"


class MatchedAlertFrontierTests(unittest.TestCase):
    def setUp(self):
        # Good, mediocre, and poor cuts fitted on identical labeled cases.
        self.good = [(0, .9, True, ABSTAIN), (0, .7, True, HOLD),
                     (0, .6, False, ABSTAIN), (0, .5, False, HOLD)]
        self.mid = [(0, .9, True, ABSTAIN), (0, .2, True, HOLD),
                    (0, .6, False, ABSTAIN), (0, .5, False, HOLD)]
        self.poor = [(0, .5, True, ABSTAIN), (0, .4, True, HOLD),
                     (0, .9, False, ABSTAIN), (0, .8, False, HOLD)]

    def test_shared_constraint_is_total_warn_not_false_positive_rate(self):
        row = f.metrics(self.good, .7)
        self.assertEqual(row["fpr"], 0)
        self.assertEqual(row["warn_rate"], .5)
        self.assertTrue(f.within_budget(row, 50))
        self.assertFalse(f.within_budget(row, 49))
        choices = f.select_frontier(f.sweep({"one": dict.fromkeys(f.REGIMES, self.good)}), (49, 50))
        for objective in f.OBJECTIVES:
            self.assertEqual(choices[49][objective][1], .9)
            self.assertEqual(choices[50][objective][1], .7)

    def test_cap_does_not_split_tied_scores(self):
        tied = [(g, .9, y, d) for g, _, y, d in self.good]
        choices = f.select_frontier(f.sweep({"one": dict.fromkeys(f.REGIMES, tied)}), (75, 100))
        for objective in f.OBJECTIVES:
            self.assertEqual(choices[75][objective][1], math.inf)
            self.assertEqual(choices[100][objective][1], .9)

    def test_shared_feasible_space_but_objective_can_change_selection(self):
        calibration = {
            "fragile": dict(zip(f.REGIMES, (self.good, self.poor, self.poor))),
            "stable": dict.fromkeys(f.REGIMES, self.mid),
        }
        choices = f.select_frontier(f.sweep(calibration), (50,))[50]
        self.assertEqual(choices["source_loss"][0], "fragile")
        self.assertEqual(choices["worst_regime_loss"][0], "stable")
        self.assertTrue(all(f.within_budget(m, 50) for _, _, by_regime in choices.values()
                            for m in by_regime.values()))

    def test_warning_and_trace_are_independent_axes(self):
        half, never = f.metrics(self.good, .7), f.metrics(self.good, math.inf)
        self.assertEqual((half["hold"], half["abstain"], half["go"]), (2, 2, 0))
        self.assertEqual(half["hold"], never["hold"])
        self.assertEqual(half["abstain"], never["abstain"])
        self.assertEqual([half[k] for k in ("warn_and_hold", "warn_and_abstain", "no_warn_and_hold",
                                           "no_warn_and_abstain")], [1, 1, 1, 1])
        self.assertEqual(never["warn"], 0)
        self.assertEqual(never["weighted_loss"], (2 * f.MISS_COST) / 4)

    def test_frozen_choices_precede_tests_and_reuse_one_sweep(self):
        events = []
        starts = f.split_starts(f.SEED, 0)
        genuine_rows, genuine_select, genuine_sweep, genuine_observe = (
            f.regime_rows, f.select_frontier, f.sweep, f.observe)

        def rows(*args):
            events.append("test" if args[1] in starts[4:] else "development")
            return genuine_rows(*args)

        def sweep(calibration):
            events.append("sweep")
            self.assertEqual(set(calibration), set(f.POLICIES))
            self.assertEqual(len(calibration), 14)
            self.assertEqual(sum(isinstance(p, tuple) for p in f.POLICIES.values()), 10)
            self.assertTrue(all(set(m) == set(f.REGIMES) for m in calibration.values()))
            return genuine_sweep(calibration)

        def select(table, budgets):
            events.append("select")
            return genuine_select(table, budgets)

        def observe(values, policy, model):
            events.append(("observe_test" if values[0][0] in starts[4:] else "observe_cal",
                           values[0][0], policy))
            return genuine_observe(values, policy, model)

        with patch.object(f, "regime_rows", rows), patch.object(f, "sweep", sweep), \
                patch.object(f, "select_frontier", select), patch.object(f, "observe", observe):
            f.one_run(0, 10, f.SEED, (0, 10, 30))
        self.assertEqual(events.count("sweep"), 1)
        self.assertLess(events.index("sweep"), events.index("select"))
        self.assertLess(events.index("select"), events.index("test"))
        self.assertEqual(sum(isinstance(e, tuple) and e[0] == "observe_cal" for e in events), 42)
        self.assertEqual(len([e for e in events if isinstance(e, tuple)]),
                         len({e for e in events if isinstance(e, tuple)}))

    def test_reproducibility_pairing_and_disjoint_groups(self):
        args = (3, 10, 100, f.SEED, (0, 15, 30))
        report = f.evaluate(*args)
        encode = lambda r: json.dumps(r, sort_keys=True, allow_nan=False)
        self.assertEqual(encode(report), encode(f.evaluate(*args)))
        boundaries = sorted(r for run in report["runs"] for r in run["group_ranges"].values())
        self.assertTrue(all(a[1] < b[0] for a, b in zip(boundaries, boundaries[1:])))
        for run in report["runs"]:
            for pct, objective in itertools.product((0, 15, 30), f.OBJECTIVES):
                point = run["points"][str(pct)][objective]
                for m in point["calibration"].values():
                    self.assertTrue(f.within_budget(m, pct))
                for m in point["held_out"].values():
                    self.assertEqual(m["hold"] + m["abstain"], m["cases"])
                    self.assertEqual(m["go"], 0)
                    self.assertEqual(m["cases"], 50)
        for regime in f.REGIMES:
            a = report["frontier"]["15"]["regimes"][regime]
            for key in f.METRICS:
                paired = a["paired_worst_minus_source"][key]["mean"]
                self.assertAlmostEqual(paired,
                    a["worst_regime_loss"][key]["mean"] - a["source_loss"][key]["mean"])

    def test_invalid_budget_and_incomplete_regime(self):
        for budgets in ((), (-1,), (101,), (.1,), (True,), (25, 10), (10, 10)):
            with self.assertRaises(ValueError):
                f.evaluate(2, 10, 100, budgets=budgets)
        with self.assertRaises(ValueError):
            f.sweep({"one": {"source": self.good}})


if __name__ == "__main__":
    unittest.main()
