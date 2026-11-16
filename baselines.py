"""Evaluate deterministic, label-blind baselines on the fixed held-out synthetic split."""
import json
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal

from benchmark import generate


def row_count(sample):
    return len(sample["source"]["rows"]) != len(sample["target"]["rows"])


def raw_diff(sample):
    return sample["source"] != sample["target"]


def canonical_gold(sample):
    """Full-information reference, not a scalable sampling or ML detector."""
    def canonical(snap):
        rows = []
        for row in snap["rows"]:
            row = row.copy()
            for col in sample["contract"]["strip_trailing_ascii_spaces"]:
                if row[col] is not None:
                    row[col] = row[col].rstrip(" ")
            row["balance"] = Decimal(row["balance"])
            row["opened_at"] = datetime.fromisoformat(row["opened_at"].replace("Z", "+00:00")).astimezone(timezone.utc)
            rows.append(tuple(sorted(row.items())))
        return Counter(rows), snap["schema"], Counter(tuple(g) for g in snap["grants"])
    return canonical(sample["source"]) != canonical(sample["target"])


def evaluate(seed_count=100):
    results = {}
    for method in (row_count, raw_diff, canonical_gold):
        counts = dict(tp=0, fp=0, tn=0, fn=0)
        missed = Counter()
        for sample in generate(seed_count):
            if sample["split"] != "test":
                continue
            observed = {key: sample[key] for key in ("source", "target", "contract")}
            prediction = method(observed)
            positive = sample["defect"]
            counts[("t" if prediction == positive else "f") + ("p" if prediction else "n")] += 1
            if positive and not prediction:
                missed[sample["case"]] += 1
        results[method.__name__] = {"counts": counts, "missed_cases": dict(missed)}
    return results


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
