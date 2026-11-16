"""Read-only, bounded policy agent for CSV cutover investigations."""
import argparse
import csv
import json
from pathlib import Path
import sys

from migration_parity import compare


def investigate(source, target, keys, tool_budget=4, max_file_bytes=8 * 1024 * 1024):
    """Acquire evidence, dispatch only applicable checks, and explain the decision."""
    trace = []

    def run(name, tool):
        if len(trace) >= tool_budget:
            return None
        event = {"tool": name, "outcome": "ok"}
        trace.append(event)
        try:
            event["evidence"] = tool()
            return event["evidence"]
        except (OSError, ValueError, csv.Error):
            event["outcome"] = "error"
            event["evidence"] = {"reason": "snapshot could not be read or parsed"}
            return None

    def preflight():
        sizes = [item.stat().st_size for item in (source, target)]
        return {"source_bytes": sizes[0], "target_bytes": sizes[1],
                "within_limit": all(item.is_file() for item in (source, target))
                and max(sizes) <= max_file_bytes}

    admission = run("inspect_inputs", preflight)
    report = run("compare_keyed_snapshots", lambda: compare(source, target, keys)) if admission and admission["within_limit"] else None
    if report is None:
        return {"recommendation": "no-go", "severity": "unknown",
                "reasons": ["Could not establish parity within input and tool limits."], "trace": trace}

    def key_integrity():
        source_state, target_state = report["source"], report["target"]
        return {"source_blank_keys": source_state["blank_key_rows"],
                "target_blank_keys": target_state["blank_key_rows"],
                "source_duplicate_keys": source_state["duplicate_keys"],
                "target_duplicate_keys": target_state["duplicate_keys"],
                "skipped_ambiguous_keys": report["comparison"]["skipped_ambiguous_keys"]}

    def value_coverage():
        source_nulls, target_nulls = [report[side]["nulls_by_column"] for side in ("source", "target")]
        return {"columns_with_more_target_empty_cells": sorted(
            col for col in source_nulls.keys() & target_nulls.keys() if target_nulls[col] > source_nulls[col]
        ), "comparison": report["comparison"]}

    if report["status"] == "match":
        return {"recommendation": "go-for-csv-parity", "severity": "none",
                "reasons": ["Column membership and all uniquely keyed CSV rows match."], "trace": trace}

    reasons = []
    if any(report["schema"][name] for name in ("source_only_columns", "target_only_columns")):
        reasons.append("Column membership differs.")
    if any(report[side][field] for side in ("source", "target") for field in ("blank_key_rows", "duplicate_keys")):
        run("inspect_key_integrity", key_integrity)
        reasons.append("Blank or duplicate keys prevent full keyed reconciliation.")
    if any(report["comparison"][key] for key in ("source_only_keys", "target_only_keys", "changed_keys")):
        run("inspect_value_coverage", value_coverage)
        reasons.append("Rows are missing, added, or changed by key.")
    return {"recommendation": "no-go", "severity": "blocking", "reasons": reasons, "trace": trace}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--key", action="append", required=True)
    parser.add_argument("--tool-budget", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args(argv)
    result = investigate(args.source, args.target, args.key, args.tool_budget)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return {"none": 0, "blocking": 1, "unknown": 2}[result["severity"]]


if __name__ == "__main__":
    sys.exit(main())
