"""Compare two CSV snapshots using stable keys, without exposing row values."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import sys
from pathlib import Path


def snapshot(path, keys):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        columns = next(reader, [])
        if not columns or any(not col.strip() for col in columns):
            raise ValueError(f"{path}: missing or blank header")
        if len(set(columns)) != len(columns):
            raise ValueError(f"{path}: duplicate column names")
        missing = sorted(set(keys) - set(columns))
        if missing:
            raise ValueError(f"{path}: missing key columns: {', '.join(missing)}")
        groups = defaultdict(list)
        nulls = Counter(dict.fromkeys(columns, 0))
        rows = blank_key_rows = 0
        for cells in reader:
            if len(cells) != len(columns):
                raise ValueError(f"{path}: record ending at line {reader.line_num} has {len(cells)} fields; expected {len(columns)}")
            row = dict(zip(columns, cells))
            rows += 1
            nulls.update(col for col, value in row.items() if value == "")
            key = tuple(row[col] for col in keys)
            if "" in key:
                blank_key_rows += 1
            else:
                groups[key].append(row)
    return columns, groups, dict(nulls), rows, blank_key_rows


def compare(source, target, keys):
    if not keys or any(not k.strip() for k in keys) or len(keys) != len(set(keys)):
        raise ValueError("provide distinct, nonblank --key names")
    sc, sg, sn, sr, sb = snapshot(source, keys)
    tc, tg, tn, tr, tb = snapshot(target, keys)
    shared = sorted((set(sc) & set(tc)) - set(keys))
    common = sg.keys() & tg.keys()
    comparable = [k for k in common if len(sg[k]) == len(tg[k]) == 1]
    mismatches = Counter()
    changed = 0
    for key in comparable:
        fields = [col for col in shared if sg[key][0][col] != tg[key][0][col]]
        mismatches.update(fields)
        changed += bool(fields)

    def shape(groups, rows, blanks, nulls):
        return {"rows": rows, "blank_key_rows": blanks,
                "duplicate_keys": sum(len(group) > 1 for group in groups.values()),
                "extra_duplicate_rows": sum(len(group) - 1 for group in groups.values()),
                "nulls_by_column": dict(sorted(nulls.items()))}

    source_shape = shape(sg, sr, sb, sn)
    target_shape = shape(tg, tr, tb, tn)
    source_only = len(sg.keys() - tg.keys())
    target_only = len(tg.keys() - sg.keys())
    removed, added = sorted(set(sc) - set(tc)), sorted(set(tc) - set(sc))
    reordered = not (removed or added) and sc != tc
    return {
        "status": "drift" if any((removed, added, source_only, target_only, sb, tb,
                                  changed, source_shape["duplicate_keys"], target_shape["duplicate_keys"])) else "match",
        "key_columns": keys,
        "source": source_shape, "target": target_shape,
        "schema": {"source_only_columns": removed, "target_only_columns": added, "reordered": reordered},
        "comparison": {"source_only_keys": source_only, "target_only_keys": target_only,
                       "compared_keys": len(comparable), "skipped_ambiguous_keys": len(common) - len(comparable),
                       "changed_keys": changed, "mismatches_by_column": dict(sorted(mismatches.items()))},
    }


def markdown(report):
    # JSON scalar/collection syntax is compact and makes blank names/counts explicit.
    def render(value):
        return json.dumps(value, ensure_ascii=False)
    lines = [f"# Migration parity: {report['status']}", "",
             f"Key columns: {render(report['key_columns'])}"]
    for section in ("source", "target", "schema", "comparison"):
        lines.extend(("", f"## {section.title()}", ""))
        for name, value in report[section].items():
            lines.append(f"- {name.replace('_', ' ').capitalize()}: {render(value)}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--key", action="append", required=True, help="key column; repeat for a composite key")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args(argv)
    try:
        report = compare(args.source, args.target, args.key)
    except (OSError, ValueError, csv.Error) as error:
        print(f"migration-parity: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False) if args.format == "json" else markdown(report))
    return int(report["status"] == "drift")


if __name__ == "__main__":
    sys.exit(main())
