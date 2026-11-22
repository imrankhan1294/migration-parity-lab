"""Demonstrate aggregate migration parity with a pinned Pagila projection."""
import argparse
import csv
from datetime import datetime
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

PIN = "eb65be5bc01833c4a4b716e611b82d0f680e359a"
FIXTURE = {
    "pagila-schema.sql": "69972968c7c78f78b478a7b578400eeb411d31b2d3afd881497b075e1db5edc7",
    "pagila-data.sql": "dc8b9998c2fd2ad2a52fde80f72805e2142db232c5b3853016d84d9efb03ba0f",
}
CUTOFF = "2007-04-01 00:00:00"
SCHEMA = (
    "CREATE TABLE inventory (inventory_id integer PRIMARY KEY, store_id smallint NOT NULL)",
    "CREATE TABLE rental (rental_id integer PRIMARY KEY, inventory_id integer NOT NULL REFERENCES inventory)",
    "CREATE TABLE payment (payment_id integer PRIMARY KEY, rental_id integer NOT NULL REFERENCES rental, "
    "amount numeric(5,2) NOT NULL, payment_date timestamp without time zone NOT NULL)",
)
TABLES = ("inventory", "rental", "payment")
HEADERS = {
    "inventory": "inventory_id, film_id, store_id, last_update",
    "rental": "rental_id, inventory_id, customer_id, staff_id, last_update, rental_period",
    "payment": "payment_id, customer_id, staff_id, rental_id, amount, payment_date",
}
COLUMNS = {"inventory": (0, 2), "rental": (0, 1), "payment": (0, 3, 4, 5)}


def project_dump(data):
    """Extract exact COPY blocks only. No upstream SQL is ever executed."""
    result = {table: [] for table in TABLES}
    copy = re.compile(r"COPY public\.(inventory|rental|payment_p\w+) \(([^)]+)\) FROM stdin;")
    current = None
    for line in data.decode("utf-8").splitlines():
        if current is not None:
            if line == "\\.":
                current = None
            else:
                cols = line.split("\t")
                if len(cols) != len(HEADERS[current].split(", ")):
                    raise ValueError(f"unexpected fixture row in {current}")
                result[current].append([cols[i] for i in COLUMNS[current]])
            continue
        match = copy.fullmatch(line)
        if match:
            name, header = match.groups()
            current = "payment" if name.startswith("payment_p") else name
            if header != HEADERS[current]:
                raise ValueError(f"unexpected header for {name}")
    if current is not None or any(not values for values in result.values()):
        raise ValueError("incomplete fixture COPY data")
    return result


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.reader(stream))


def fold_source(tables):
    """Independently join raw source exports in Python, with Decimal sums."""
    def lookup(table, rows):
        keys = {key: value for key, value in rows}
        if len(keys) != len(rows):
            raise ValueError(f"duplicate {table} key")
        return keys

    stores = lookup("inventory", tables["inventory"])
    inventory = lookup("rental", tables["rental"])
    buckets = {}
    seen = set()
    cutoff = datetime.fromisoformat(CUTOFF)
    for payment_id, rental_id, raw_amount, timestamp in tables["payment"]:
        if payment_id in seen:
            raise ValueError("duplicate payment key")
        seen.add(payment_id)
        if datetime.fromisoformat(timestamp) >= cutoff:
            continue
        key = (stores[inventory[rental_id]], timestamp[:10])
        n, amount = buckets.get(key, (0, Decimal("0.00")))
        buckets[key] = (n + 1, amount + Decimal(raw_amount))
    return buckets


def bucket_diff(src, dst, boundary):
    if boundary is None:
        return {"status": "UNRESOLVED", "reason": "common logical boundary not established"}
    differences = []
    for key in sorted(src.keys() | dst.keys()):
        a, b = src.get(key), dst.get(key)
        if a != b:
            differences.append({
                "store": int(key[0]), "date": key[1],
                "source": [a[0], str(a[1])] if a is not None else None,
                "target": [b[0], str(b[1])] if b is not None else None,
            })
    return {"status": "MISMATCH" if differences else "MATCH_WITHIN_SCOPE",
            "differences": differences}


def reproduce(bin_dir, fixture_dir):
    bin_dir = bin_dir.resolve(strict=True)
    inputs = {name: (fixture_dir / name).read_bytes() for name in FIXTURE}
    for name, contents in inputs.items():
        if hashlib.sha256(contents).hexdigest() != FIXTURE[name]:
            raise ValueError(f"fixture checksum mismatch: {name}")
    data = project_dump(inputs["pagila-data.sql"])
    env = {"PATH": os.defpath, "LC_ALL": "C", "TZ": "UTC"}
    if "LD_LIBRARY_PATH" in os.environ:
        env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]

    def run(exe, *argv, data=None):
        result = subprocess.run(
            [str(bin_dir / exe), *argv], input=data, capture_output=True, env=env, timeout=60,
        )
        stderr = result.stderr.decode(errors="replace")
        # The standalone backend can report SQL errors yet exit zero.
        if result.returncode or re.search(r"\b(?:ERROR|FATAL|PANIC):", stderr):
            raise RuntimeError(f"{exe} failed: {stderr[-3000:]}")
        return result.stdout.decode()

    def literal(value):
        return "'" + str(value).replace("'", "''") + "'"

    with tempfile.TemporaryDirectory(prefix="pagila-parity-") as tmp:
        root = Path(tmp)
        if any(char in tmp for char in "\r\n"):
            raise ValueError("temporary directory path must not contain CR or LF")
        env["HOME"] = tmp
        src, dst = root / "src", root / "dst"
        run("initdb", "-D", str(src), "-U", "parity_demo", "--auth-local=reject",
            "--auth-host=reject", "--no-locale", "-E", "UTF8")

        def sql(cluster, commands):
            return run("postgres", "--single", "-D", str(cluster),
                       "-c", "timezone=UTC", "-c", "DateStyle=ISO,YMD",
                       "-c", "standard_conforming_strings=on", "postgres",
                       data=("\n".join(commands) + "\n").encode())

        def export(cluster, query, label):
            path = root / f"{label}.csv"
            sql(cluster, [f"COPY ({query}) TO {literal(path)} WITH CSV;"])
            return read_csv(path)

        version = export(
            src, "SELECT current_setting('server_version_num')", "version")
        if int(version[0][0]) < 170000:
            raise RuntimeError("Use PostgreSQL 17 or newer")
        for table in TABLES:
            with (root / f"{table}-import.csv").open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerows(data[table])
        sql(src, [
            "BEGIN;", *(stmt + ";" for stmt in SCHEMA),
            *(f"COPY {table} FROM {literal(root / (table + '-import.csv'))} WITH CSV;" for table in TABLES),
            "COMMIT;", "CHECKPOINT;",
        ])
        expected = fold_source({
            table: export(src, f"SELECT * FROM {table}", table + "-source") for table in TABLES
        })
        control = run("pg_controldata", str(src))
        if not re.search(r"^Database cluster state:\s+shut down\s*$", control, re.MULTILINE):
            raise RuntimeError("source must be shut down cleanly before copying")
        # Closed standalone source, no sockets, no external DSN or other writers.
        # No symlinks/tablespaces are created by this fixture.
        shutil.copytree(src, dst)
        boundary = {
            "method": "offline_cluster_copy_after_clean_shutdown",
            "cutoff_exclusive": CUTOFF,
            "fixture_commit": PIN,
            "timestamp_semantics": "fixture-local naive timestamp",
        }
        baseline = (
            sum(n for n, _ in expected.values()),
            sum((amount for _, amount in expected.values()), Decimal("0.00")),
        )
        if baseline != (9626, Decimal("40185.74")):
            raise ValueError("unexpected pinned fixture baseline")
        where = f"WHERE p.payment_date < TIMESTAMP '{CUTOFF}'"
        join = "FROM payment p JOIN rental r USING (rental_id) JOIN inventory i USING (inventory_id)"

        def snapshot(case, provenance):
            total = export(dst, f"SELECT COUNT(*), SUM(amount) FROM payment p {where}", case + "-total")
            buckets = export(dst, f"SELECT store_id,payment_date::date,COUNT(*),SUM(amount) "
                             f"{join} {where} GROUP BY 1,2 ORDER BY 1,2", case + "-buckets")
            actual = {(s, d): (int(n), Decimal(a)) for s, d, n, a in buckets}
            return {
                "case": case,
                "global_matches": (int(total[0][0]), Decimal(total[0][1])) == baseline,
                **bucket_diff(expected, actual, provenance),
            }

        def inspect_rows(cluster, ids, label):
            return export(cluster, f"SELECT payment_id,store_id,payment_date::date,amount {join} "
                          f"WHERE payment_id IN ({','.join(map(str, ids))}) ORDER BY payment_id", label)

        clean = snapshot("clean_clone", boundary)
        touched = inspect_rows(dst, (2, 34), "pinned-rows")
        if touched != [["2", "2", "2007-03-15", "0.99"], ["34", "1", "2007-03-15", "2.99"]]:
            raise ValueError("expected two unique fixture rows on separate stores")
        sql(dst, ["UPDATE payment SET amount = CASE payment_id WHEN 2 THEN 2.99 WHEN 34 THEN 0.99 END "
                  "WHERE payment_id IN (2,34) AND payment_date::date = DATE '2007-03-15';"])
        swapped = snapshot("balanced_store_swap", boundary)
        unresolved = snapshot("withheld_boundary_evidence_control", None)
        # Reset from the unchanged source before an independent mutation.
        shutil.rmtree(dst)
        shutil.copytree(src, dst)
        same_bucket_before = inspect_rows(src, (34, 241), "same-bucket-before")
        if same_bucket_before != [["34", "1", "2007-03-15", "2.99"],
                                  ["241", "1", "2007-03-15", "5.99"]]:
            raise ValueError("expected two unique fixture rows in the same store/day")
        sql(dst, ["UPDATE payment SET amount = CASE payment_id WHEN 34 THEN 5.99 WHEN 241 THEN 2.99 END "
                  "WHERE payment_id IN (34,241) AND payment_date::date = DATE '2007-03-15';"])
        same_bucket_after = inspect_rows(dst, (34, 241), "same-bucket-after")
        if same_bucket_after != [["34", "1", "2007-03-15", "5.99"],
                                 ["241", "1", "2007-03-15", "2.99"]]:
            raise ValueError("same-bucket mutation did not apply to both fixture rows")
        same_bucket = snapshot("balanced_within_bucket_swap", boundary)
        same_bucket["mutated_keyed_rows_match"] = same_bucket_before == same_bucket_after
        return {"source": {"count": baseline[0], "total": str(baseline[1])},
                "boundary": boundary, "cases": [clean, swapped, unresolved, same_bucket]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pg-bin", required=True, type=Path)
    parser.add_argument("--fixture-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(reproduce(args.pg_bin, args.fixture_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
