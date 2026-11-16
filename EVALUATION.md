# Synthetic migration drift benchmark

Run `python -m unittest -v` for generator tests, `python benchmark.py --seeds 100 > synthetic.jsonl` to generate 1,000 labeled cases, or `python baselines.py` for held-out baseline results. Python 3.9+; no third-party packages. All account IDs, amounts, permissions and dates are synthetic. This is a snapshot benchmark, not a live Db2 or PostgreSQL connector.

Variants from the same base `seed` remain in the same split. Remove `seed`, `case`, `split`, and `defect` before detector inference, and never train or choose thresholds on the held-out split. This dataset has a deliberately artificial defect prevalence. Scores are not production error estimates.

## Rapid release roadmap

1. **Cutover agent:** read-only snapshot inspection → compare data, schema and grants to a declared contract → report evidence and a human review decision. No remediation or GRANT execution. Demo fails on wrong values/grants despite equal counts, then passes on a repaired snapshot. Explicitly call any fake database source a fixture.
2. **Evaluation and baseline release:** report raw row counts, raw exact comparison, and the full-information canonical reference separately. A raw diff is an intentionally simplistic baseline; the canonical reference is the appropriate exact baseline when scanning whole snapshots is inexpensive.
3. **Budgeted ML detector:** when a full scan is costly, compare a calibrated classifier to a simple threshold using the **same** sample size and feature observations. Train on seed groups, calibrate with held-out groups using sigmoid calibration, set the alert threshold before test, and report per-defect recall, precision, benign false positives, PR-AUC, calibrated Brier score and sampled bytes/rows. Schema and role invariants remain deterministic gates. ML is triage, never authority to permit a cutover. Publish results even if the threshold baseline wins.
4. **Harder independent scenarios:** test unseen defects and synthetic distribution shifts with new generators, grouped bootstrap intervals, severity breakdowns, and a clear statement of what remains untested. An exact reference may trivially score perfectly on its own constructed fixtures; that validates fixture semantics, not superiority in production.

Publish changes as they become useful: first green/red demo, then benchmark reproducibility, then one honest result with failure examples, then a feature or fix prompted by those results. Avoid filler and sweeping originality claims.
