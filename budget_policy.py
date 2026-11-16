"""Synthetic equal-cost tool-choice experiment, not a real database cutover gate."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math
import random

TOOLS = ("telemetry", "values", "schema", "grants", "counts")
ROUTES = {"ledger": "values", "deploy": "schema", "identity": "grants", "copy": "counts"}
PROBES = TOOLS[1:]


def make_cases(groups=500, stress=False, start=0):
    """Five independent sibling cases per group share an operational route."""
    for group in range(start, start + groups):
        rng = random.Random(2129 + group)
        route = rng.choice(tuple(ROUTES))
        digest = hashlib.sha256(str(group).encode()).digest()[0]
        split = "train" if digest < 153 else "calibration" if digest < 205 else "test"
        for _ in range(5):
            defect = rng.random() < 0.33
            matching = ROUTES[route]
            route_matches = rng.random() < (0.72 if not stress else 0.22)
            family = (matching if route_matches else rng.choice([p for p in PROBES if p != matching])) if defect else "none"
            routed = route if rng.random() < 0.8 else rng.choice(tuple(ROUTES))
            probes = {
                p: "hit" if rng.random() < (0.85 if p == family else 0.07) else "clear"
                for p in PROBES
            }
            # Outputs are noisy and can miss a true defect or flag benign cases.
            # Telemetry is observable only by paying for this tool.
            yield split, defect, family, {"telemetry": routed, **probes}


class Session:
    def __init__(self, outputs, budget):
        self.__outputs = outputs
        self.budget = budget
        self.trace = []

    def inspect(self, tool, reason):
        if tool not in TOOLS or len(self.trace) >= self.budget or any(t["tool"] == tool for t in self.trace):
            raise ValueError("invalid, repeated or over-budget action")
        result = self.__outputs[tool]
        self.trace.append({"tool": tool, "observed": result, "reason": reason})
        return result


def probability(counts):
    return (counts[1] + 1) / (sum(counts.values()) + 2)


def entropy(p):
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


class Learned:
    """Trains a smoothed joint observer model; chooses tools with up to two-step information gain."""
    def __init__(self, rows):
        self.rows = [(defect, outputs) for _, defect, _, outputs in rows]

    def compatible(self, seen):
        return [(y, obs) for y, obs in self.rows if all(obs[tool] == value for tool, value in seen.items())]

    def score(self, seen):
        return probability(Counter(y for y, _ in self.compatible(seen)))

    @lru_cache(maxsize=None)
    def choose(self, key, depth):
        seen = dict(key)
        rows = self.compatible(seen)
        current = entropy(probability(Counter(y for y, _ in rows)))
        if depth == 0 or not rows:
            return current, {}
        scores = {}
        for tool in TOOLS:
            if tool in seen:
                continue
            outcomes = Counter(obs[tool] for _, obs in rows)
            scores[tool] = sum(
                count * self.choose(tuple(sorted((*key, (tool, outcome)))), depth - 1)[0]
                for outcome, count in outcomes.items()
            ) / len(rows)
        return min(scores.values()), {tool: current - value for tool, value in scores.items()}

    def select(self, seen, remaining):
        _, gains = self.choose(tuple(sorted(seen.items())), min(2, remaining))
        if not gains:
            return next(t for t in TOOLS if t not in seen), "unseen observation pattern"
        tool = max(gains, key=gains.get)
        return tool, {"lookahead_bits": {k: round(v, 5) for k, v in gains.items()},
                      "matching_training_cases": len(self.compatible(seen))}


def run(method, model, outputs, budget, rng):
    session = Session(outputs, budget)
    seen = {}
    for step in range(budget):
        if method == "learned":
            tool, why = model.select(seen, budget - step)
        elif method == "heuristic":
            order = ("telemetry", ROUTES.get(seen.get("telemetry"), "values"), *PROBES)
            tool, why = next(t for t in order if t not in seen), "telemetry routing"
        elif method == "fixed":
            tool, why = PROBES[step], "fixed direct-probe order"
        elif method == "random":
            tool, why = rng.choice([t for t in TOOLS if t not in seen]), "seeded uniform unused-tool selection"
        else:
            raise ValueError(f"unknown policy {method!r}")
        seen[tool] = session.inspect(tool, why)
    # Every policy shares the exact same fitted observer and information boundary.
    return model.score(seen), session.trace


def disposition(trace):
    """Observed probe alerts block pending review; this experiment never gives GO."""
    alerts = [entry["tool"] for entry in trace
              if entry["tool"] in PROBES and entry["observed"] == "hit"]
    return {"decision": "HOLD_FOR_REVIEW" if alerts else "INCOMPLETE_EVIDENCE",
            "observed_alerts": alerts,
            "uninspected_probes": [probe for probe in PROBES
                                  if probe not in {entry["tool"] for entry in trace}]}


def calibration_threshold(scores, target_fpr=0.15):
    """Choose max observed recall under calibration FPR constraint; no test input."""
    negatives = sum(not y for _, y in scores)
    positives = sum(y for _, y in scores)
    if not negatives or not positives:
        raise ValueError("calibration needs both classes")
    candidates = [*sorted({p for p, _ in scores}), math.inf]
    allowed = []
    for threshold in candidates:
        fp = sum(p >= threshold and not y for p, y in scores)
        tp = sum(p >= threshold and y for p, y in scores)
        if fp / negatives <= target_fpr:
            allowed.append((tp, -fp, threshold))
    return max(allowed)[2]


def metrics(scores, threshold):
    labels = Counter()
    missed = Counter()
    confusion = Counter({"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    brier = 0.0
    for p, y, family in scores:
        pred = p >= threshold
        confusion[("t" if pred == y else "f") + ("p" if pred else "n")] += 1
        brier += (p - y) ** 2
        if y:
            labels[family] += 1
            if not pred:
                missed[family] += 1
    return {"confusion": dict(confusion),
            "recall": round(confusion["tp"] / (confusion["tp"] + confusion["fn"]), 4),
            "false_positive_rate": round(confusion["fp"] / (confusion["fp"] + confusion["tn"]), 4),
            "precision": round(confusion["tp"] / max(1, confusion["tp"] + confusion["fp"]), 4),
            "raw_probability_brier": round(brier / len(scores), 4),
            "missed_by_family": {k: [missed[k], n] for k, n in sorted(labels.items())}}


def evaluate(groups, budget):
    rows = list(make_cases(groups))
    model = Learned([row for row in rows if row[0] == "train"])
    stress = list(make_cases(max(groups // 3, 30), stress=True, start=100000))
    result = {}
    for method in ("learned", "fixed", "random", "heuristic"):
        out = defaultdict(list)
        rng = random.Random(8721)
        example = None
        for split, y, family, outputs in rows + [("stress", *s[1:]) for s in stress]:
            if split == "train":
                continue
            score, trace = run(method, model, outputs, budget, rng)
            out[split].append((score, y, family))
            if split == "test" and example is None:
                example = {"trace": trace, "raw_risk": round(score, 5), **disposition(trace)}
        threshold = calibration_threshold([(p, y) for p, y, _ in out["calibration"]])
        result[method] = {"threshold": threshold if math.isfinite(threshold) else None,
                          "test": metrics(out["test"], threshold),
                          "stress": metrics(out["stress"], threshold),
                          "example": example}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=500)
    parser.add_argument("--budget", type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    if args.groups < 75:
        parser.error("use at least 75 groups to keep all splits represented")
    print(json.dumps(evaluate(args.groups, args.budget), indent=2))
