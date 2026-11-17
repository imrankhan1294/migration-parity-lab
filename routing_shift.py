"""Two-probe routing under shift in the synthetic Migration Parity Lab.

Run python3 routing_shift.py > routing-report.json and python3 routing_shift.py --self-test.
Each regime holds out groups; thresholds and the selected fixed reference use calibration only.
Point estimates and percentile intervals are conditional on the fitted model and fixture generator.
Neither a risk score nor a lack of alerts is permission to cut over.
"""
import argparse
import itertools
import json
import math
import random
import unittest

from budget_policy import (
    Learned, PROBES, Session, TOOLS, calibration_threshold, disposition, make_cases, run,
)

SUBSETS = tuple(itertools.combinations(TOOLS, 2))
BEST = "calibration-selected-fixed"
SEED = 240917


class CachedObserver(Learned):
    """Reuse posteriors for identical purchased observations, never hidden outputs."""
    def __init__(self, rows):
        super().__init__(rows)
        self._scores = {}

    def score(self, seen):
        key = tuple(sorted(seen.items()))
        if key not in self._scores:
            self._scores[key] = super().score(seen)
        return self._scores[key]


def cases(groups, start=0, shifted=False):
    for i, (split, defect, _family, outputs) in enumerate(make_cases(groups, shifted, start)):
        yield start + i // 5, split, defect, outputs


def sample(policy, model, outputs, key, direct):
    if policy in ("learned", "heuristic", "random"):
        return run(policy, model, outputs, 2, random.Random(SEED + key))
    session = Session(outputs, 2)
    if policy == "hybrid":
        first, backup = direct
        value = session.inspect(first, "selected direct-probe anchor")
        second = "telemetry" if value == "hit" else backup
        session.inspect(second, "telemetry on alert; direct fallback otherwise")
    else:
        for tool in policy:
            session.inspect(tool, "fixed subset")
    seen = {x["tool"]: x["observed"] for x in session.trace}
    return model.score(seen), session.trace


def collect(rows, policy, model, direct):
    observations = []
    for i, (group, _, label, outputs) in enumerate(rows):
        score, trace = sample(policy, model, outputs, i, direct)
        # Clinical/operational authority is deliberately not modeled by probabilities.
        observations.append((group, score, label, disposition(trace)["decision"]))
    return observations


def stats(rows, threshold):
    tp = fp = pos = neg = hold = held_defects = held_benign = 0
    for _, score, label, decision in rows:
        if label:
            pos += 1
            tp += score >= threshold
        else:
            neg += 1
            fp += score >= threshold
        flagged = decision == "HOLD_FOR_REVIEW"
        hold += flagged
        held_defects += flagged and label
        held_benign += flagged and not label
    return dict(tp=tp, fp=fp, positives=pos, negatives=neg, holds=hold,
                recall=tp / pos if pos else None, fpr=fp / neg if neg else None,
                operational_hold_recall=held_defects / pos if pos else None,
                benign_operational_hold_rate=held_benign / neg if neg else None)


def fit(obs):
    return calibration_threshold([(score, label) for _, score, label, _ in obs])


def interval(tied, repetitions, seed):
    """Paired group bootstrap for recall differences, preserving sibling cases."""
    if not tied or repetitions < 1:
        raise ValueError("bootstrap requires groups and repetitions")
    rng = random.Random(seed)
    values = []
    for _ in range(repetitions):
        delta = positives = 0
        for _ in tied:
            d, n = tied[rng.randrange(len(tied))]
            delta += d
            positives += n
        if positives:
            values.append(delta / positives)
    if not values:
        return None
    values.sort()
    return [values[int((len(values) - 1) * p)] for p in (.025, .975)]


def paired(candidate, reference, cut, ref_cut):
    if len(candidate) != len(reference):
        raise ValueError("unpaired observations")
    per_group = {}
    for (ga, a, ya, _), (gb, b, yb, _) in zip(candidate, reference):
        if (ga, ya) != (gb, yb):
            raise ValueError("unpaired observations")
        if ga not in per_group:
            per_group[ga] = [0, 0]
        if ya:
            per_group[ga][0] += int(a >= cut) - int(b >= ref_cut)
            per_group[ga][1] += 1
    return list(per_group.values())


def evaluate(groups=500, repetitions=1200):
    data = list(cases(groups))
    train = [(split, label, None, outputs) for _, split, label, outputs in data if split == "train"]
    model = CachedObserver(train)
    calibration = [row for row in data if row[1] == "calibration"]
    regimes = {
        "held_out": [row for row in data if row[1] == "test"],
        "routing_shift": list(cases(max(groups // 3, 30), 100000, True)),
        "routing_shift_replication": list(cases(max(groups // 3, 30), 200000, True)),
    }
    all_cal = {p: collect(calibration, p, model, PROBES[:2]) for p in SUBSETS}
    thresholds = {p: fit(obs) for p, obs in all_cal.items()}

    def rank(p):
        report = stats(all_cal[p], thresholds[p])
        return report["tp"], -report["fp"]

    best = max(SUBSETS, key=rank)
    direct = max((p for p in SUBSETS if "telemetry" not in p), key=rank)
    policies = [(BEST, best), ("hybrid", "hybrid"), ("learned", "learned"),
                ("heuristic", "heuristic"), ("random", "random")]
    policies.extend(("+".join(p), p) for p in SUBSETS)
    result = {}
    for name, policy in policies:
        threshold = thresholds[policy] if policy in thresholds else fit(collect(calibration, policy, model, direct))
        result[name] = dict(threshold=threshold if math.isfinite(threshold) else None,
                            calibration=stats(collect(calibration, policy, model, direct), threshold))
        for regime, rows in regimes.items():
            actual = collect(rows, policy, model, direct)
            reference = collect(rows, best, model, direct)
            m = stats(actual, threshold)
            ref = stats(reference, thresholds[best])
            result[name][regime] = dict(
                **m, recall_delta_vs_selected_fixed=m["recall"] - ref["recall"],
                paired_group_bootstrap_95=interval(paired(actual, reference, threshold, thresholds[best]),
                                                    repetitions, SEED),
            )
    return dict(fixed_subsets=len(SUBSETS), selected_fixed=list(best), hybrid_direct_order=list(direct),
                bootstrap_repetitions=repetitions, policies=result)


class Tests(unittest.TestCase):
    def test_subsets_and_trace(self):
        self.assertEqual(len(set(SUBSETS)), 10)
        rows = list(cases(100))
        m = Learned([(s, y, None, o) for _, s, y, o in rows if s == "train"])
        for p in (*SUBSETS, "hybrid", "learned", "heuristic", "random"):
            for _, _, _, obs in rows[:20]:
                _, trace = sample(p, m, obs, 4, PROBES[:2])
                self.assertEqual(len({x["tool"] for x in trace}), 2)
                self.assertNotEqual(disposition(trace)["decision"], "GO")

    def test_pairing_and_difference(self):
        r = [(0, .8, True, "HOLD_FOR_REVIEW"), (1, .7, True, "INCOMPLETE_EVIDENCE")]
        z = [(g, .1, y, d) for g, _, y, d in r]
        self.assertEqual(interval(paired(r, z, .5, .5), 50, 19), [1., 1.])
        with self.assertRaises(ValueError):
            paired(r, list(reversed(z)), .5, .5)

    def test_group_separation(self):
        primary = list(cases(120))
        train, cal, test = [{g for g, s, _, _ in primary if s == split}
                            for split in ("train", "calibration", "test")]
        shifted = {g for g, _, _, _ in cases(40, 100000, True)}
        self.assertTrue(all(not a & b for a, b in itertools.combinations((train, cal, test, shifted), 2)))

    def test_end_to_end(self):
        r = evaluate(100, 20)
        self.assertEqual(len(r["policies"]), 15)
        for m in r["policies"].values():
            self.assertLessEqual(m["calibration"]["fpr"], .15)
            self.assertEqual(m["routing_shift"]["recall"],
                             m["routing_shift"]["tp"] / m["routing_shift"]["positives"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=1200)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=["routing_shift"], verbosity=2)
    elif args.groups < 75 or args.bootstrap < 1:
        parser.error("use at least 75 groups and one bootstrap repetition")
    else:
        print(json.dumps(evaluate(args.groups, args.bootstrap), indent=2, allow_nan=False))
