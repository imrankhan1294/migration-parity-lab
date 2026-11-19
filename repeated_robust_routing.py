"""Independent synthetic refit/calibrate/test repetitions of robust_routing.

python3 repeated_robust_routing.py > repeated-robust-report.json
python3 repeated_robust_routing.py --self-test

Stdlib only. No real data, network, credentials, deployment, or GO decisions.
Every repetition refits the observer and reselects policy/threshold using fresh,
group-disjoint source-fit, three calibration, and three test sets. Selection is
frozen before test generation. Both selectors share the 14-policy candidate set,
two purchased tools, miss/review costs, and nominal calibration FPR cap. Source
selection constrains source calibration only; minimax constrains all three regimes.
This is NOT a matched-realized-FPR comparison: report FPR, recall, and cost together.

Run percentiles describe Monte Carlo repeat spread, NOT confidence intervals.
Separately labeled bootstrap intervals estimate uncertainty in the across-run mean
by resampling whole independent refits. Paired differences resample paired runs.
Neither interval is a deployment guarantee or simultaneous inference. Repeat
variation includes fit/selection/test generation but only under this fixed toy
generator. Statistical threshold escalations and operational probe-hit holds are
reported separately: a threshold never cancels a trace hold or authorizes cutover.
"""
import argparse
from collections import Counter
import itertools
import json
import math
import random
import statistics
import unittest
from unittest.mock import patch

from budget_policy import PROBES, disposition
from robust_routing import REGIMES, regime_rows
from routing_shift import CachedObserver, SUBSETS, sample

SEED = 261119
MISS_COST = 5
REVIEW_COST = 1
CAP = 0.20
STRIDE = 10000
METRICS = ("weighted_loss", "fpr", "recall")
POLICIES = {"+".join(p): p for p in SUBSETS}
POLICIES.update({p: p for p in ("learned", "heuristic", "random", "hybrid")})


def split_starts(seed, repeat):
    """Seven disjoint group ranges per repeat; no sibling crosses a split."""
    return [seed * 10000000 + (repeat * 7 + slot) * STRIDE for slot in range(7)]


def observe(rows, policy, model):
    observations = []
    for index, (group, _, label, outputs) in enumerate(rows):
        # Independent repeat-specific random policy streams, shared between
        # comparisons. No label or unpurchased value enters the observation score.
        score, trace = sample(policy, model, outputs, group * 5 + index % 5, PROBES[:2])
        if len(trace) != 2 or len({t["tool"] for t in trace}) != 2:
            raise AssertionError("Each candidate must purchase exactly two distinct tools.")
        decision = disposition(trace)["decision"]
        if decision not in {"HOLD_FOR_REVIEW", "INCOMPLETE_EVIDENCE"}:
            raise AssertionError("A synthetic score must never authorize GO.")
        observations.append((group, score, label, decision))
    return observations


def escalation_metrics(rows, threshold):
    """Statistical score threshold, not the trace's operational disposition."""
    positives = sum(bool(y) for _, _, y, _ in rows)
    negatives = len(rows) - positives
    if not positives or not negatives:
        raise ValueError("Every measured split must contain both classes; no runs are silently dropped.")
    tp = sum(y and score >= threshold for _, score, y, _ in rows)
    fp = sum(not y and score >= threshold for _, score, y, _ in rows)
    fn, tn = positives - tp, negatives - fp
    return dict(cases=len(rows), positives=positives, negatives=negatives,
                tp=tp, fp=fp, fn=fn, tn=tn, statistical_escalations=tp + fp,
                recall=tp / positives, fpr=fp / negatives,
                weighted_loss=(MISS_COST * fn + REVIEW_COST * fp) / len(rows))


def measure(rows, threshold):
    holds = sum(d == "HOLD_FOR_REVIEW" for _, _, _, d in rows)
    return {
        "calibrated_escalation": escalation_metrics(rows, threshold),
        "trace_disposition": {"hold_for_review": holds,
                              "incomplete_evidence": len(rows) - holds, "go": 0},
        "inspection_budget": {"distinct_tools_per_case": 2, "total_tool_calls": 2 * len(rows)},
    }


def select(calibration, cap=CAP):
    """Select from calibration only with deterministic original minimax tie-break."""
    if not 0 <= cap <= 1 or not calibration:
        raise ValueError("A nonempty calibration set and FPR cap in [0, 1] are required.")
    choices = []
    for name, regimes in calibration.items():
        if not regimes:
            raise ValueError("A candidate has no calibration regimes.")
        cuts = sorted({score for rows in regimes.values() for _, score, _, _ in rows}) + [math.inf]
        for cut in cuts:
            metrics = [escalation_metrics(rows, cut) for rows in regimes.values()]
            if any(m["fpr"] > cap for m in metrics):
                continue
            losses = [m["weighted_loss"] for m in metrics]
            choices.append((max(losses), statistics.mean(losses), -cut, name, cut))
    if not choices:
        raise ValueError("No valid calibration candidate.")
    worst, mean, _, policy, threshold = min(choices)
    return dict(policy=policy, threshold=threshold, calibration_worst_loss=worst,
                calibration_mean_loss=mean)


def one_run(repeat, groups, seed):
    starts = split_starts(seed, repeat)
    train = regime_rows(groups, starts[0], "source")
    model = CachedObserver([(s, y, None, out) for _, s, y, out in train])
    calibration = {r: regime_rows(groups, starts[i + 1], r) for i, r in enumerate(REGIMES)}
    purchased = {name: {r: observe(rows, policy, model) for r, rows in calibration.items()}
                 for name, policy in POLICIES.items()}
    selected = {
        "source_only": select({name: {"source": regimes["source"]} for name, regimes in purchased.items()}),
        "minimax": select(purchased),
    }
    # Selection is complete. No test labels or test outcomes were available above.
    reports = {}
    for i, regime in enumerate(REGIMES):
        test = regime_rows(groups, starts[i + 4], regime)
        reports[regime] = {}
        cache = {}
        for selector, choice in selected.items():
            name, cut = choice["policy"], choice["threshold"]
            if name not in cache:
                cache[name] = observe(test, POLICIES[name], model)
            reports[regime][selector] = measure(cache[name], cut)
    for label, choice in selected.items():
        choice["calibration_by_regime"] = {
            r: escalation_metrics(rows, choice["threshold"])
            for r, rows in purchased[choice["policy"]].items()
        }
        # Strict JSON: null means positive infinity (never a statistical escalation).
        choice["threshold"] = choice["threshold"] if math.isfinite(choice["threshold"]) else None
    model.choose.cache_clear()  # Release inherited lru_cache references between refits.
    return dict(repeat=repeat, group_ranges={
        name: [start, start + groups - 1]
        for name, start in zip(("fit_source", "cal_source", "cal_route_shift", "cal_noisy_shift",
                                "test_source", "test_route_shift", "test_noisy_shift"), starts)},
                selections=selected, tests=reports)


def quantile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize(values, bootstrap, seed):
    if len(values) < 2 or bootstrap < 1:
        raise ValueError("At least two independent repeats and one bootstrap draw are required.")
    rng = random.Random(seed)
    means = [statistics.mean(values[rng.randrange(len(values))] for _ in values)
             for _ in range(bootstrap)]
    return dict(mean=statistics.mean(values), across_run_sample_sd=statistics.stdev(values),
                monte_carlo_run_spread_p025_p975=[quantile(values, q) for q in (.025, .975)],
                bootstrap_95_ci_for_mean=[quantile(means, q) for q in (.025, .975)])


def evaluate(repeats=20, groups=80, bootstrap=2000, seed=SEED):
    if not 2 <= repeats <= 200 or not 10 <= groups <= STRIDE or not 100 <= bootstrap <= 100000 or not 0 <= seed <= 2**31 - 1:
        raise ValueError("repeats 2..200; groups 10..10000; bootstrap 100..100000; seed 0..2147483647")
    runs = [one_run(repeat, groups, seed) for repeat in range(repeats)]
    summary = {}
    for regime in REGIMES:
        summary[regime] = {}
        for selector in ("source_only", "minimax"):
            values = [r["tests"][regime][selector]["calibrated_escalation"] for r in runs]
            summary[regime][selector] = {metric: summarize([v[metric] for v in values], bootstrap, seed + 9000001)
                                         for metric in METRICS}
            summary[regime][selector]["test_fpr_cap_exceedance_runs"] = sum(v["fpr"] > CAP for v in values)
        summary[regime]["paired_delta_minimax_minus_source"] = {
            metric: summarize([r["tests"][regime]["minimax"]["calibrated_escalation"][metric]
                               - r["tests"][regime]["source_only"]["calibrated_escalation"][metric]
                               for r in runs], bootstrap, seed + 9000001)
            for metric in METRICS}
    return dict(experiment="independent_refit_robust_routing_v1", base_commit="61e1826abed9cd73ae20b20383bc3dafed5173cf",
                seed=seed, independent_repeats=repeats, groups_per_split_regime=groups,
                cases_per_group=5, bootstrap_draws=bootstrap, calibration_fpr_cap=CAP,
                policy_candidates=len(POLICIES), candidate_names=sorted(POLICIES),
                costs={"miss": MISS_COST, "benign_statistical_escalation": REVIEW_COST},
                tool_budget=2, hybrid_anchor=list(PROBES[:2]),
                uncertainty_notes={
                    "run_spread": "2.5th/97.5th empirical run percentiles, not a confidence interval.",
                    "mean_ci": "Percentile bootstrap of complete independent runs; paired deltas resample paired runs.",
                    "limits": "Fixed synthetic generator; no simultaneous coverage, production guarantee, or matched-realized-FPR claim."},
                safety="Only HOLD_FOR_REVIEW or INCOMPLETE_EVIDENCE. Statistical thresholds do not override trace holds. Never GO.",
                selection_counts={selector: dict(sorted(Counter(r["selections"][selector]["policy"] for r in runs).items()))
                                  for selector in ("source_only", "minimax")},
                summary=summary, runs=runs)


class Tests(unittest.TestCase):
    def test_split_and_repeat_ranges_disjoint(self):
        ranges = [set(range(start, start + STRIDE)) for repeat in range(3) for start in split_starts(SEED, repeat)]
        self.assertTrue(all(not a & b for a, b in itertools.combinations(ranges, 2)))

    def test_seed_changes_data(self):
        a = regime_rows(10, split_starts(SEED, 0)[0], "source")
        b = regime_rows(10, split_starts(SEED + 1, 0)[0], "source")
        self.assertNotEqual([row[2:] for row in a], [row[2:] for row in b])

    def test_escalations_are_not_trace_holds(self):
        rows = [(0, .8, True, "INCOMPLETE_EVIDENCE"), (1, .1, False, "HOLD_FOR_REVIEW")]
        low, high = measure(rows, .5), measure(rows, math.inf)
        self.assertEqual(low["calibrated_escalation"]["statistical_escalations"], 1)
        self.assertEqual(high["calibrated_escalation"]["statistical_escalations"], 0)
        self.assertEqual(low["trace_disposition"], high["trace_disposition"])
        self.assertEqual(high["calibrated_escalation"]["weighted_loss"], 2.5)

    def test_select_uses_worst_regime_and_cap(self):
        good = [(0, .9, True, "INCOMPLETE_EVIDENCE"), (1, .1, False, "INCOMPLETE_EVIDENCE")]
        bad = [(g, 1-p, y, d) for g, p, y, d in good]
        choice = select({"fragile": {"a": good, "b": bad}, "stable": {"a": good, "b": good}})
        self.assertEqual(choice["policy"], "stable")
        self.assertEqual(choice["calibration_worst_loss"], 0)

    def test_no_classes_are_silently_skipped(self):
        with self.assertRaises(ValueError):
            select({"p": {"r": [(0, .5, True, "INCOMPLETE_EVIDENCE")]}})

    def test_selection_precedes_test_generation(self):
        events = []
        original_rows, original_select = regime_rows, select
        starts = split_starts(SEED, 0)
        def record_rows(groups, start, regime):
            events.append("test" if start in starts[4:] else "development")
            return original_rows(groups, start, regime)
        def record_select(data, cap=CAP):
            events.append("select")
            return original_select(data, cap)
        with patch(__name__ + ".regime_rows", record_rows), patch(__name__ + ".select", record_select):
            one_run(0, 10, SEED)
        self.assertEqual(events.count("select"), 2)
        self.assertLess(max(i for i, e in enumerate(events) if e == "select"), events.index("test"))

    def test_all_candidates_buy_two_tools_and_never_go(self):
        rows = regime_rows(10, 0, "source")
        model = CachedObserver([(s, y, None, out) for _, s, y, out in rows])
        for policy in POLICIES.values():
            observed = observe(rows, policy, model)
            self.assertEqual(len(observed), 50)
            self.assertTrue(all(d in {"HOLD_FOR_REVIEW", "INCOMPLETE_EVIDENCE"} for _, _, _, d in observed))
        model.choose.cache_clear()

    def test_uncertainty_labels_and_constant(self):
        summary = summarize([2., 2., 2.], 100, SEED)
        self.assertEqual(summary["monte_carlo_run_spread_p025_p975"], [2., 2.])
        self.assertEqual(summary["bootstrap_95_ci_for_mean"], [2., 2.])
        self.assertEqual(summary["across_run_sample_sd"], 0)

    def test_repeat_spread_and_mean_interval_are_distinct(self):
        summary = summarize([0., 1., 2., 3., 4.], 1000, SEED)
        self.assertEqual(summary["monte_carlo_run_spread_p025_p975"], [.1, 3.9])
        self.assertGreater(summary["bootstrap_95_ci_for_mean"][0], .1)

    def test_end_to_end_and_byte_reproduction(self):
        a, b = evaluate(2, 10, 100), evaluate(2, 10, 100)
        self.assertEqual(json.dumps(a, sort_keys=True, allow_nan=False), json.dumps(b, sort_keys=True, allow_nan=False))
        for run in a["runs"]:
            for selector, choice in run["selections"].items():
                applicable = choice["calibration_by_regime"].values() if selector == "minimax" else [choice["calibration_by_regime"]["source"]]
                self.assertTrue(all(m["fpr"] <= CAP for m in applicable))
            for regimes in run["tests"].values():
                for result in regimes.values():
                    self.assertEqual(result["inspection_budget"]["total_tool_calls"], 100)
                    self.assertEqual(sum(result["trace_disposition"].values()), 50)
                    m = result["calibrated_escalation"]
                    self.assertEqual(m["weighted_loss"], (5*m["fn"] + m["fp"])/50)

    def test_invalid_arguments(self):
        for args in ((1, 10, 100, SEED), (2, 9, 100, SEED), (2, 10, 99, SEED), (2, 10, 100, -1)):
            with self.assertRaises(ValueError):
                evaluate(*args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--groups", type=int, default=80)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=["repeated_robust_routing"], verbosity=2)
    else:
        try:
            output = evaluate(args.repeats, args.groups, args.bootstrap, args.seed)
        except ValueError as error:
            parser.error(str(error))
        print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
