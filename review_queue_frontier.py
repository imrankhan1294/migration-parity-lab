"""Synthetic review-queue frontier: count evidence HOLD OR statistical WARN.

Compares every fixed two-probe pair against the conservative hybrid at matched
nominal calibration caps. An unqueued case remains INCOMPLETE_EVIDENCE, not GO.
"""
import argparse
import bisect
import itertools
import json
import math
import random
import statistics
from collections import Counter

from matched_alert_frontier import uncertainty
from repeated_robust_routing import (
    CachedObserver, MISS_COST, POLICIES, REGIMES, REVIEW_COST, STRIDE,
    observe, regime_rows, split_starts,
)

BASE = "55a1d7949e44c7e44611f72d6de6cdba6dcbb9a0"
SEED = 261121
BUDGETS = (20, 30, 40, 45, 50, 55, 60, 70)
FAMILIES = {"fixed": tuple(sorted(n for n, policy in POLICIES.items()
                                  if isinstance(policy, tuple))),
            "hybrid": ("hybrid",)}
METRICS = ("queue_rate", "recall", "fpr", "weighted_loss", "hold_rate",
           "warn_rate", "unqueued_abstain_rate")


class Profile:
    """Sort WARN candidates once; add only non-HOLD rows to the queue."""
    def __init__(self, rows):
        self.cases = len(rows)
        self.positive = sum(bool(label) for _, _, label, _ in rows)
        if not 0 < self.positive < self.cases:
            raise ValueError("Both classes are required in every split.")
        self.keys = []
        self.hold_positive = self.hold_negative = 0
        self.extra_positive = [0]
        self.extra_negative = [0]
        self.overlap = [0]
        for _, score, label, dec in sorted(rows, key=lambda row: row[1]):
            if not math.isfinite(score):
                raise ValueError("Scores must be finite.")
            if dec not in ("HOLD_FOR_REVIEW", "INCOMPLETE_EVIDENCE"):
                raise ValueError("Unexpected evidence disposition.")
            hold = dec == "HOLD_FOR_REVIEW"
            self.keys.append(score)
            self.hold_positive += bool(hold and label)
            self.hold_negative += bool(hold and not label)
            self.extra_positive.append(self.extra_positive[-1] + bool(not hold and label))
            self.extra_negative.append(self.extra_negative[-1] + bool(not hold and not label))
            self.overlap.append(self.overlap[-1] + hold)

    def at(self, cut):
        k = bisect.bisect_left(self.keys, cut)
        n, positives = self.cases, self.positive
        hold = self.hold_positive + self.hold_negative
        warn_and_hold = self.overlap[-1] - self.overlap[k]
        tp = self.hold_positive + self.extra_positive[-1] - self.extra_positive[k]
        fp = self.hold_negative + self.extra_negative[-1] - self.extra_negative[k]
        queue = tp + fp
        fn, warn = positives - tp, n - k
        return dict(
            cases=n, positives=positives, queued=queue, queue_rate=queue / n,
            hold=hold, hold_rate=hold / n, warn=warn, warn_rate=warn / n,
            warn_and_hold=warn_and_hold, warn_and_abstain=warn - warn_and_hold,
            no_warn_and_hold=hold - warn_and_hold,
            abstain=n - hold, unqueued_abstain=n - queue,
            unqueued_abstain_rate=(n - queue) / n, go=0,
            tp=tp, fp=fp, fn=fn, tn=n - positives - fp,
            recall=tp / positives, fpr=fp / (n - positives),
            weighted_loss=(MISS_COST * fn + REVIEW_COST * fp) / n,
        )


def within(measured, cap):
    return 100 * measured["queued"] <= cap * measured["cases"]


def sweep(calibration):
    if not calibration or any(set(p) != set(REGIMES) for p in calibration.values()):
        raise ValueError("Each policy must supply all calibration regimes.")
    table = {}
    for name, observed in sorted(calibration.items()):
        profiles = {r: Profile(observed[r]) for r in REGIMES}
        cuts = sorted({s for prof in profiles.values() for s in prof.keys})
        table[name] = [(name, cut, {r: profiles[r].at(cut) for r in REGIMES})
                       for cut in itertools.chain(cuts, (math.inf,))]
    return table


def select(table, budgets, families=FAMILIES):
    def order(choice):
        name, cut, m = choice
        loss = [m[r]["weighted_loss"] for r in REGIMES]
        return max(loss), statistics.mean(loss), -cut, name

    return {str(cap): {
        family: min((choice for name in names for choice in table[name]
                     if all(within(m, cap) for m in choice[2].values())),
                    key=order, default=None)
        for family, names in families.items()} for cap in budgets}


def one_run(repeat, groups, seed, budgets):
    starts = split_starts(seed, repeat)
    fit = regime_rows(groups, starts[0], "source")
    model = CachedObserver([(s, y, None, out) for _, s, y, out in fit])
    try:
        names = sorted({n for members in FAMILIES.values() for n in members})
        train = {r: regime_rows(groups, starts[i+1], r) for i, r in enumerate(REGIMES)}
        calibration = {n: {r: observe(train[r], POLICIES[n], model) for r in REGIMES}
                       for n in names}
        table = sweep(calibration)
        choices = select(table, budgets)
        selected = {c[0] for slot in choices.values() for c in slot.values() if c}
        # No held-out cases are generated before all choices are fixed.
        test = {}
        for i, r in enumerate(REGIMES):
            rows = regime_rows(groups, starts[i+4], r)
            test[r] = {n: Profile(observe(rows, POLICIES[n], model)) for n in selected}
        floors = {n: {r: table[n][-1][2][r]["queue_rate"] for r in REGIMES} for n in names}
        points = {cap: {family: None if choice is None else dict(
            policy=choice[0], threshold=choice[1] if math.isfinite(choice[1]) else None,
            calibration=choice[2],
            held_out={r: test[r][choice[0]].at(choice[1]) for r in REGIMES})
            for family, choice in slot.items()} for cap, slot in choices.items()}
        return dict(repeat=repeat, group_starts=starts, evidence_queue_floors=floors, points=points)
    finally:
        model.choose.cache_clear()


def evaluate(repeats=20, groups=80, draws=2000, seed=SEED, budgets=BUDGETS):
    if not (2 <= repeats <= 200 and 10 <= groups <= STRIDE and 100 <= draws <= 100000
            and 0 <= seed <= 2**31-1):
        raise ValueError("repeats 2..200; groups 10..10000; bootstrap 100..100000; seed 0..2147483647")
    if (not budgets or any(type(p) is not int or not 0 <= p <= 100 for p in budgets)
            or tuple(sorted(set(budgets))) != tuple(budgets)):
        raise ValueError("Budgets must be unique increasing integer percents in 0..100.")
    runs = [one_run(i, groups, seed, budgets) for i in range(repeats)]
    frontier = {}
    for cap in budgets:
        slots = [run["points"][str(cap)] for run in runs]
        # A pair with either infeasible cannot be treated as an observed comparison.
        paired = [slot for slot in slots if all(slot.values())]
        rng = random.Random(seed + 870003 + cap)
        indexes = [tuple(rng.randrange(len(paired)) for _ in paired)
                   for _ in range(draws)] if len(paired) >= 2 else []
        comparison = {}
        for r in REGIMES:
            by_family = {family: [slot[family]["held_out"][r] for slot in paired]
                         for family in FAMILIES}
            comparison[r] = {family: None if not rows else dict(
                means={m: statistics.mean(row[m] for row in rows) for m in METRICS},
                paired_test_cap_breaches=sum(not within(row, cap) for row in rows))
                for family, rows in by_family.items()}
            comparison[r]["paired_hybrid_minus_fixed"] = None if not indexes else {
                m: uncertainty([b[m]-a[m] for a, b in zip(by_family["fixed"], by_family["hybrid"])],
                               indexes) for m in METRICS}
        frontier[str(cap)] = dict(
            feasible_runs={f: sum(s[f] is not None for s in slots) for f in FAMILIES},
            paired_feasible_runs=len(paired),
            policy_counts={f: dict(sorted(Counter(s[f]["policy"] for s in slots if s[f]).items()))
                           for f in FAMILIES},
            regimes=comparison,
        )
    return dict(
        experiment="matched_calibration_review_queue_frontier_v1", base_commit=BASE,
        seed=seed, independent_refits=repeats, groups_per_split_regime=groups, cases_per_group=5,
        distinct_tools_per_case=2, bootstrap_draws=draws, calibration_queue_cap_pct=list(budgets),
        families=FAMILIES, regimes=REGIMES,
        costs={"defect_not_queued": MISS_COST, "benign_case_queued": REVIEW_COST},
        meanings=dict(queue="Deduplicated union: evidence HOLD OR statistical WARN.",
                      abstain="Evidence remains incomplete without a probe hit, even if statistically warned.",
                      missing_selection="No threshold fits the calibration cap; HOLD may never be suppressed.",
                      interpretation="Nominal calibration caps only; estimates conditional on jointly feasible refits; never GO."),
        frontier=frontier, runs=runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--groups", type=int, default=80)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--budgets", type=int, nargs="+", default=BUDGETS)
    args = parser.parse_args()
    try:
        report = evaluate(args.repeats, args.groups, args.bootstrap, args.seed, args.budgets)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
