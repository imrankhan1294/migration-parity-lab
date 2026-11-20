"""Matched calibration warning-budget frontier for synthetic two-probe routing.

The selectors share one candidate sweep, all three calibration subsets, all
three held-out subsets, and the same maximum calibration warn rate. Only their
loss objective differs. A statistical WARN is not an operational HOLD or
permission to proceed. The other operational outcome is ABSTAIN; never GO.
"""
import argparse
import itertools
import json
import math
import random
import statistics
from collections import Counter

from repeated_robust_routing import (
    MISS_COST, POLICIES, REGIMES, REVIEW_COST, SEED, STRIDE, CachedObserver,
    escalation_metrics, observe, quantile, regime_rows, split_starts,
)

BASE = "f253e0e3d878b958d04d7890ea98ec4dba5dde1e"
BUDGETS = (0, 5, 10, 15, 20, 25, 30, 40)
OBJECTIVES = ("source_loss", "worst_regime_loss")
METRICS = ("warn_rate", "fpr", "recall", "weighted_loss", "hold_rate", "abstain_rate")


def metrics(rows, cut):
    raw = escalation_metrics(rows, cut)
    cases = raw["cases"]
    warn = raw.pop("statistical_escalations")
    holds = sum(decision == "HOLD_FOR_REVIEW" for _, _, _, decision in rows)
    overlap = sum(decision == "HOLD_FOR_REVIEW" and score >= cut
                  for _, score, _, decision in rows)
    return dict(raw, warn=warn, hold=holds, abstain=cases - holds, go=0,
                warn_rate=warn / cases, hold_rate=holds / cases,
                abstain_rate=(cases - holds) / cases,
                warn_and_hold=overlap, warn_and_abstain=warn - overlap,
                no_warn_and_hold=holds - overlap,
                no_warn_and_abstain=cases - warn - holds + overlap)


def within_budget(row, pct):
    # Integer comparison treats score ties atomically at the boundary.
    return 100 * row["warn"] <= pct * row["cases"]


def sweep(calibration):
    """One shared table of policy/cut/regime measurements, never test data."""
    if not calibration or any(set(by_regime) != set(REGIMES)
                              for by_regime in calibration.values()):
        raise ValueError("Each candidate requires all three calibration subsets.")
    table = []
    for name, by_regime in sorted(calibration.items()):
        cuts = sorted({score for rows in by_regime.values() for _, score, _, _ in rows})
        for cut in itertools.chain(cuts, (math.inf,)):
            measurements = {r: metrics(by_regime[r], cut) for r in REGIMES}
            table.append((name, cut, measurements))
    return table


def select_frontier(table, budgets):
    """Both objectives see exactly the same feasible choices for each cap."""
    chosen = {}
    for pct in budgets:
        feasible = [(name, cut, measured) for name, cut, measured in table
                    if all(within_budget(row, pct) for row in measured.values())]
        if not feasible:
            raise ValueError("No calibration candidate; no-warning cut should be feasible.")

        def key(choice, objective):
            name, cut, measured = choice
            losses = [measured[r]["weighted_loss"] for r in REGIMES]
            # For each objective, only its own loss breaks candidate ties.
            # Shared deterministic tie-break favors the more conservative cut.
            return ((losses[0],) if objective == "source_loss"
                    else (max(losses), statistics.mean(losses))) + (-cut, name)

        chosen[pct] = {
            objective: min(feasible, key=lambda choice: key(choice, objective))
            for objective in OBJECTIVES
        }
    return chosen


def one_run(repeat, groups, seed, budgets):
    starts = split_starts(seed, repeat)
    fit_rows = regime_rows(groups, starts[0], "source")
    model = CachedObserver([(s, y, None, out) for _, s, y, out in fit_rows])
    try:
        cal_rows = {r: regime_rows(groups, starts[i + 1], r) for i, r in enumerate(REGIMES)}
        purchased = {name: {r: observe(rows, policy, model) for r, rows in cal_rows.items()}
                     for name, policy in POLICIES.items()}
        selections = select_frontier(sweep(purchased), budgets)
        names = {choice[0] for pair in selections.values() for choice in pair.values()}
        # All thresholds for all budgets are frozen before generating held-out cases.
        tests = {}
        for i, regime in enumerate(REGIMES):
            rows = regime_rows(groups, starts[i + 4], regime)
            tests[regime] = {n: observe(rows, POLICIES[n], model) for n in sorted(names)}

        points = {}
        for pct, choices in selections.items():
            points[str(pct)] = {}
            for objective, (name, cut, calibration) in choices.items():
                points[str(pct)][objective] = dict(
                    policy=name, threshold=cut if math.isfinite(cut) else None,
                    calibration=calibration,
                    held_out={r: metrics(tests[r][name], cut) for r in REGIMES},
                )
        labels = ("fit_source", *(f"cal_{r}" for r in REGIMES), *(f"test_{r}" for r in REGIMES))
        return dict(repeat=repeat, group_ranges={
            name: [start, start + groups - 1] for name, start in zip(labels, starts)},
            points=points)
    finally:
        model.choose.cache_clear()


def uncertainty(values, bootstrap_samples):
    estimates = [sum(values[i] for i in indices) / len(indices) for indices in bootstrap_samples]
    return dict(
        mean=statistics.mean(values),
        bootstrap_95_ci_for_mean=[quantile(estimates, q) for q in (.025, .975)],
        monte_carlo_run_spread_p025_p975=[quantile(values, q) for q in (.025, .975)],
    )


def evaluate(repeats=20, groups=80, draws=2000, seed=SEED, budgets=BUDGETS):
    if not (2 <= repeats <= 200 and 10 <= groups <= STRIDE and 100 <= draws <= 100000
            and 0 <= seed <= 2**31 - 1):
        raise ValueError("repeats 2..200; groups 10..10000; draws 100..100000; seed 0..2147483647")
    if not budgets or any(type(p) is not int or not 0 <= p <= 100 for p in budgets):
        raise ValueError("At least one integer warning-budget percent in 0..100 is required.")
    if tuple(sorted(set(budgets))) != tuple(budgets):
        raise ValueError("Warning-budget percents must be unique and increasing.")

    runs = [one_run(i, groups, seed, budgets) for i in range(repeats)]
    rng = random.Random(seed + 9000002)
    # The same resampled repeat indices apply to both selectors and every budget.
    # Each indexed repeat preserves the three regimes and all five sibling cases.
    draws_index = [tuple(rng.randrange(repeats) for _ in range(repeats)) for _ in range(draws)]
    frontier = {}
    for pct in budgets:
        p = str(pct)
        frontier[p] = {
            "policy_selection_counts": {
                o: dict(sorted(Counter(run["points"][p][o]["policy"] for run in runs).items()))
                for o in OBJECTIVES
            },
            "regimes": {},
        }
        for regime in REGIMES:
            by_objective = {o: [run["points"][p][o]["held_out"][regime] for run in runs]
                            for o in OBJECTIVES}
            result = {o: {
                **{m: uncertainty([row[m] for row in by_objective[o]], draws_index)
                   for m in METRICS},
                "test_warning_budget_exceedance_runs": sum(
                    not within_budget(row, pct) for row in by_objective[o]),
            } for o in OBJECTIVES}
            source, minimax = (by_objective[o] for o in OBJECTIVES)
            result["paired_worst_minus_source"] = {
                m: uncertainty([b[m] - a[m] for a, b in zip(source, minimax)], draws_index)
                for m in METRICS
            }
            result["paired_absolute_realized_warn_rate_gap"] = uncertainty(
                [abs(b["warn_rate"] - a["warn_rate"]) for a, b in zip(source, minimax)], draws_index)
            frontier[p]["regimes"][regime] = result
    return dict(
        experiment="matched_calibration_warning_budget_frontier_v1", base_commit=BASE,
        seed=seed, independent_refits=repeats, groups_per_split_regime=groups, cases_per_group=5,
        distinct_tools_per_case=2, bootstrap_draws=draws, calibration_warning_budget_pct=list(budgets),
        policy_candidates=sorted(POLICIES), calibration_regimes=list(REGIMES),
        costs={"miss": MISS_COST, "benign_statistical_warning": REVIEW_COST},
        definitions={
            "budget": "Maximum observed statistical WARN fraction of all cases in each calibration regime for both selectors, not a fixed FPR.",
            "warn": "Model score at or above calibration-selected cut; used for classification only.",
            "hold": "Purchased direct probe hit requires operational HOLD_FOR_REVIEW, irrespective of WARN.",
            "abstain": "INCOMPLETE_EVIDENCE, irrespective of WARN; does not clear a cutover.",
            "threshold_null": "Positive infinity: never WARN.",
            "uncertainty": "Paired percentile bootstrap resamples whole independent refits; marginal, fixed-generator intervals, not production guarantees.",
            "limits": "Matched nominal calibration cap only; realized held-out warning rates may diverge. No held-out threshold selection. Never GO.",
        }, frontier=frontier, runs=runs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--groups", type=int, default=80)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--budgets", nargs="+", type=int, default=BUDGETS)
    args = parser.parse_args()
    try:
        result = evaluate(args.repeats, args.groups, args.bootstrap, args.seed, args.budgets)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
