# Migration Parity Lab

A dependency-free Python CLI for checking whether two CSV snapshots agree after a migration. It reconciles by business key rather than row position, so sorting one export differently won't hide or invent drift.

## Quickstart

Run with Python 3.9+ from this directory:

```sh
python3 migration_parity.py source.csv target.csv \
  --key tenant_id --key widget_id
```

The synthetic example reports a removed column, one source-only key, one target-only key, one ambiguous key excluded from comparison, and one unique key with a changed quota. Add `--format json` for machine-readable output. Exit codes: `0` match, `1` drift, `2` invalid input or arguments.

## What the report checks

- Row counts, blank-key rows, duplicates and empty-cell counts per column on each side.
- Added and removed columns; reordering alone is informational.
- Keys present on only one side and exact value changes by column for keys occurring once on both sides. Duplicate keys are flagged, not arbitrarily paired.
- No key values or row contents are printed. Reports do include column names, so review them before sharing.

CSV fields are compared literally: `01` differs from `1`, and only an empty field counts as null. UTF-8 with or without a BOM is supported. Both snapshots are held in memory. This is a snapshot comparison, not proof of database parity: it does not inspect types, permissions, indexes or live data.

## Tests

```sh
python3 -m unittest -v
```

## Cutover investigation agent

```sh
python3 cutover_agent.py source.csv target.csv \
  --key tenant_id --key widget_id
```

Starts with local-file admission (up to 8 MiB per snapshot) and keyed reconciliation. Based on that evidence it selectively dispatches blank/duplicate-key and value-coverage diagnostics, up to four tool calls total. The JSON trace shows which tools ran and why the result is a go/no-go. If the comparison cannot run, the agent fails closed. Exit codes: `0` CSV parity checks passed, `1` blocking drift, `2` incomplete evidence. A CSV parity pass does not authorize a database cutover.

This version is a deterministic policy agent: no model, network, writes to inputs, or autonomous deployment. Candidate next experiment: benchmark calibrated anomaly detection against deterministic checks on labeled synthetic migration failures.

## Reproduce the drift benchmark

```sh
python3 -m unittest -v
python3 baselines.py
```

On the fixed synthetic test split (180 cases), row counts alone miss 108 of 126 defects. A raw exact diff flags 36 of 54 benign cases. A full-information, contract-aware comparison makes no errors on these constructed fixtures; this is a sanity check, not evidence of production accuracy. The benchmark is separate from the CSV agent and includes schema and grant fixtures the CSV agent does not inspect. See [evaluation notes](EVALUATION.md) for the next experiment: calibrated ML triage versus a threshold baseline at the same sampling budget.

## Budgeted inspection experiment

`python3 budget_policy.py --groups 500 --budget 2` compares learned tool choice with fixed, random and heuristic baselines at the same synthetic inspection budget. The learned policy ties the simple heuristic on the held-out set and both lose recall badly under a routing shift. See [methods, results and limitations](BUDGETED_INSPECTION.md). The experiment never reports that a cutover is safe.

## Which routing policy survives a distribution shift?

`python3 routing_shift.py` compares all ten fixed tool pairs, adaptive routes and a direct-first hybrid under the same two-tool budget. In two synthetic routing-shift sets, a calibration-selected direct pair caught more defects than the learned router, at the cost of more false alarms. The hybrid did not improve detection. See [routing-shift results and reproduction instructions](ROUTING_SHIFT.md).
