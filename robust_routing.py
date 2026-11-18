"""Calibration-only robust selection for synthetic inspection routing.

Reproduce: python3 robust_routing.py > robust-report.json
Tests: python3 robust_routing.py --self-test
Requires budget_policy.py and routing_shift.py from this repository; stdlib only.

Unlike routing_shift.py, selects a POLICY AND THRESHOLD against the worst of
three predeclared calibration regimes, with asymmetric cost: 5 per missed
synthetic defect, 1 per benign review. All policies purchase two outputs.
A 20% observed calibration FPR cap applies in every included regime.
Selection never receives test rows. Source-only selection uses the same loss,
cap and candidate set, but enforces the cap on source calibration alone.
The comparison changes both calibration regimes and constraint coverage;
it does not establish dominance at matched realized FPR.

Regimes: original generator, weakened route-family link, and weakened link
plus extra benign probe alerts. The last is a stress assumption, not field data.
Training, source calibration, stress calibration and evaluation groups are
disjoint. Thresholds do not authorize actions: alerts mean HOLD_FOR_REVIEW;
otherwise evidence stays INCOMPLETE_EVIDENCE, never GO. 'Miss cost' here measures
a statistical classification, NOT a claim that an actual cutover was approved.

Intervals resample paired groups conditional on the fitted/selected models;
they do not include training or selection uncertainty and are not simultaneous
or distribution-free guarantees. Minimax protection covers only the declared
regimes. Held-out FPR is measured, not guaranteed by the calibration cap. No real credentials,
customer datasets, database connections, external calls, or dependencies.

Default measured result (250 groups/split/regime, 1,250 cases each):
  Source-only: values+grants, cut 0.6206896551724138.
  Minimax: schema+grants, cut 0.6204379562043796.
  Held-out regime   source loss/FPR    minimax loss/FPR
  source            0.9400 / 14.99%    1.2936 /  8.03%
  route_shift       0.9776 / 12.74%    1.3184 /  6.31%
  noisy_shift       0.8600 / 31.93%    1.2632 / 17.35%
  Paired 95% loss-difference intervals (minimax minus source):
  source [0.2784, 0.4368], shift [0.2704, 0.4144], noisy [0.3144, 0.4928].

This is a negative cost result: extra regime constraints reduce benign flags
but miss more defects, increasing measured asymmetric loss in EVERY regime.
Noisy-shift recall falls from 61.43% to 31.67%. The conservative selector is
not a better detector; the FPR-versus-miss tradeoff must be made explicit.
The source-only selector exceeds 20% FPR on noisy test data despite satisfying
its own calibration cap. Do not interpret synthetic calibration as assurance.
Numbers were reproduced byte-for-byte; changing seeds or costs can change them.
"""
import argparse
from collections import defaultdict
import json
import math
import random
import unittest

from budget_policy import PROBES
from routing_shift import CachedObserver, SUBSETS, cases, collect, stats

REGIMES = ("source", "route_shift", "noisy_shift")
MISS_COST = 5
REVIEW_COST = 1
CALIBRATION_FPR_CAP = .20
SEED = 261118


def regime_rows(groups, start, regime):
    if regime not in REGIMES:
        raise ValueError("unknown regime")
    rows = list(cases(groups, start, regime != "source"))
    if regime == "noisy_shift":
        rng = random.Random(SEED + start)
        # Independent extra alerts, including in true-negative cases. Never
        # alter ground-truth labels to make the inspector appear correct.
        rows = [(g, s, y, {k: ("hit" if k in PROBES and rng.random() < .12 else v)
                            for k, v in out.items()}) for g, s, y, out in rows]
    return rows


def report(obs, threshold):
    m = stats(obs, threshold)
    m["weighted_loss"] = (MISS_COST * (m["positives"] - m["tp"]) + REVIEW_COST * m["fp"]) / len(obs)
    m["mean_tool_calls"] = 2
    m["incomplete_evidence"] = len(obs) - m["holds"]
    return m


def select(calibration):
    """Pure selector: {policy: {regime: purchased observations}}; no tests."""
    choices = []
    for name, regimes in calibration.items():
        cuts = sorted({p for rows in regimes.values() for _, p, _, _ in rows}) + [math.inf]
        for cut in cuts:
            reports = [report(rows, cut) for rows in regimes.values()]
            if any(m["fpr"] is None or m["fpr"] > CALIBRATION_FPR_CAP for m in reports):
                continue
            losses = [m["weighted_loss"] for m in reports]
            # Deterministic conservative tie break: higher cut, then name.
            choices.append((max(losses), sum(losses) / len(losses), -cut, name, cut))
    if not choices:
        raise ValueError("calibration requires negative examples in every regime")
    best = min(choices)
    return best[3], best[4], best[0]


def loss_delta_interval(candidate, reference, cut, ref_cut, repetitions):
    if repetitions < 1 or len(candidate) != len(reference):
        raise ValueError("positive repetitions and paired rows required")
    groups = defaultdict(lambda: [0, 0])
    def loss(p, y, threshold):
        return MISS_COST * int(y and p < threshold) + REVIEW_COST * int(not y and p >= threshold)
    for (g, p, y, _), (h, q, z, _) in zip(candidate, reference):
        if (g, y) != (h, z):
            raise ValueError("unpaired observations")
        groups[g][0] += loss(p, y, cut) - loss(q, z, ref_cut)
        groups[g][1] += 1
    items = list(groups.values())
    rng = random.Random(SEED)
    draws = []
    for _ in range(repetitions):
        sampled = [items[rng.randrange(len(items))] for _ in items]
        draws.append(sum(x for x, _ in sampled) / sum(n for _, n in sampled))
    draws.sort()
    return [draws[int((repetitions - 1) * q)] for q in (.025, .975)]


def evaluate(groups=250, repetitions=1000):
    # Explicit nonoverlapping ranges; each group has five sibling cases.
    if not 10 <= groups <= 10000:
        raise ValueError("groups must be between 10 and 10000")
    train = regime_rows(groups, 0, "source")
    model = CachedObserver([(s, y, None, out) for _, s, y, out in train])
    cal = {r: regime_rows(groups, (i + 1) * 100000, r) for i, r in enumerate(REGIMES)}
    # Hybrid anchor is fixed a priori here, not selected using test labels.
    direct = PROBES[:2]
    policies = {"+".join(p): p for p in SUBSETS}
    policies.update({p: p for p in ("learned", "heuristic", "random", "hybrid")})
    purchased_cal = {name: {r: collect(rows, p, model, direct) for r, rows in cal.items()}
                     for name, p in policies.items()}
    robust, robust_cut, robust_loss = select(purchased_cal)
    source, source_cut, source_loss = select({n: {"source": r["source"]} for n, r in purchased_cal.items()})
    # Freeze both selectors before any evaluation rows are generated.
    selected = {"source_only": (source, source_cut), "minimax": (robust, robust_cut)}
    thresholds = {name: select({name: regimes})[1] for name, regimes in purchased_cal.items()}
    results = {}
    for i, r in enumerate(REGIMES):
        rows = regime_rows(groups, (i + 4) * 100000, r)
        observed = {name: collect(rows, p, model, direct) for name, p in policies.items()}
        baseline = observed[source]
        results[r] = {
            "selected": {label: report(observed[n], cut) for label, (n, cut) in selected.items()},
            "paired_loss_delta_minimax_minus_source_95": loss_delta_interval(observed[robust], baseline, robust_cut, source_cut, repetitions),
            "all_candidates_robust_calibrated": {n: report(obs, thresholds[n]) for n, obs in observed.items()},
        }
    return dict(seed=SEED, groups_per_split_regime=groups, cases_per_group=5,
                fixed_pairs=len(SUBSETS), policy_candidates=len(policies), bootstrap_repetitions=repetitions,
                calibration_fpr_cap=CALIBRATION_FPR_CAP,
                costs={"miss": MISS_COST, "benign_review": REVIEW_COST}, hybrid_anchor=list(direct),
                selections={label: {"policy": n, "threshold": cut if math.isfinite(cut) else None,
                                   "calibration_worst_loss": robust_loss if label == "minimax" else source_loss}
                            for label, (n, cut) in selected.items()},
                calibration={n: {r: report(obs, thresholds[n]) for r, obs in regimes.items()} for n, regimes in purchased_cal.items()},
                tests=results)


class Tests(unittest.TestCase):
    def test_ranges_and_reproducibility(self):
        sets = [{g for g, _, _, _ in regime_rows(10, i * 100000, REGIMES[i % 3])} for i in range(7)]
        for i, a in enumerate(sets):
            for b in sets[i + 1:]:
                self.assertFalse(a & b)
        self.assertEqual(regime_rows(10, 0, "noisy_shift"), regime_rows(10, 0, "noisy_shift"))

    def test_selector_worst_regime(self):
        good = [(0, .9, True, "HOLD_FOR_REVIEW"), (0, .1, False, "INCOMPLETE_EVIDENCE")]
        bad = [(g, 1-p, y, d) for g, p, y, d in good]
        name, _, loss = select({"fragile": {"a": good, "b": bad}, "stable": {"a": good, "b": good}})
        self.assertEqual((name, loss), ("stable", 0))

    def test_pairing(self):
        a = [(1, .9, True, "HOLD_FOR_REVIEW")]
        b = [(1, .1, True, "INCOMPLETE_EVIDENCE")]
        self.assertEqual(loss_delta_interval(a, b, .5, .5, 20), [-5., -5.])
        with self.assertRaises(ValueError):
            loss_delta_interval(a, [(2, .1, True, "INCOMPLETE_EVIDENCE")], .5, .5, 20)

    def test_end_to_end(self):
        r = evaluate(10, 10)
        self.assertEqual(r["fixed_pairs"], 10)
        self.assertEqual(r["policy_candidates"], 14)
        for regimes in r["calibration"].values():
            for m in regimes.values():
                self.assertLessEqual(m["fpr"], CALIBRATION_FPR_CAP)
        for regime in r["tests"].values():
            for m in regime["all_candidates_robust_calibrated"].values():
                self.assertEqual(m["mean_tool_calls"], 2)
                self.assertEqual(m["holds"] + m["incomplete_evidence"], 50)
                self.assertAlmostEqual(m["weighted_loss"], (5 * (m["positives"] - m["tp"]) + m["fp"]) / 50)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--groups", type=int, default=250)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        unittest.main(argv=["robust_routing"], verbosity=2)
    elif not 10 <= args.groups <= 10000 or args.bootstrap < 1:
        p.error("groups: 10..10000; bootstrap: at least 1")
    else:
        print(json.dumps(evaluate(args.groups, args.bootstrap), indent=2, allow_nan=False))
