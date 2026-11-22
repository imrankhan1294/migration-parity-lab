# When counts and grand totals agree but store reports do not

A worked PostgreSQL example for the migration review queue. Compare three
signals: a whole-window count and sum, the same metrics by business slice,
and whether the two sides have a documented common comparison boundary.

## Run the experiment

Use a trusted PostgreSQL 17+ installation and Python 3.9+. The demo creates a
disposable cluster as an unprivileged local user and uses PostgreSQL
[single-user mode](https://www.postgresql.org/docs/17/app-postgres.html).
It never connects to an existing database or opens a listener. Do not point
it at production data. Fetch the dataset separately; it is not checked in:

```sh
mkdir -p pagila-fixture
commit=eb65be5bc01833c4a4b716e611b82d0f680e359a
for file in pagila-schema.sql pagila-data.sql; do
  curl -fLSs "https://raw.githubusercontent.com/xzilla/pagila/$commit/$file" \
    -o "pagila-fixture/$file"
done
python3 pagila_parity_demo.py --pg-bin "$(pg_config --bindir)" --fixture-dir pagila-fixture
```

Both input files are pinned by SHA-256. Attribution and license:
[Pagila v17.a at the pinned revision](https://github.com/xzilla/pagila/tree/eb65be5bc01833c4a4b716e611b82d0f680e359a).
The experiment loads the fields required from the three fixture tables into
typed local relational tables, not the full Pagila schema. It attributes
payments to stores via payment → rental → inventory. The comparison uses
fixture-local dates and an exclusive reporting cutoff of April 1, 2007.

## Measured on PostgreSQL 17.12

| Scenario | Global count and sum | Store/date check |
| --- | --- | --- |
| Clean target | 9,626 payments, $40,185.74 | Matches within scope |
| Swap two payment amounts across stores | Still 9,626 and $40,185.74 | Two slices disagree |
| Withhold shared-boundary evidence | Same global count and sum | Unresolved |
| Swap two payment amounts within one store/date | Same global count and sum | Matches within scope; keyed rows disagree |

On March 15, Store 1 still had 58 payments, but its sum fell from $242.42
to $240.42. Store 2 still had 62, but rose from $237.38 to $239.38. A
whole-window alert based on counts and totals alone missed both changes.

## What this does not establish

A match means only these checks found no discrepancy. In the last case,
amounts were swapped between two payments within a single store and date.
All aggregates stayed unchanged, but both affected keyed records differed.
Use keyed comparisons when row identity matters. The demo fixes the
comparison window and starts from a local immutable
fixture, copying the cluster only after clean shutdown. It does not
establish a common boundary
between independently changing production databases, benchmark performance,
or verify schemas, permissions, indexes, replication lag or release
readiness. If a shared logical boundary cannot be established, agreement
is not evidence of parity. The two swaps are deliberate fixture changes,
not observed production defects or estimates of detection rates.
