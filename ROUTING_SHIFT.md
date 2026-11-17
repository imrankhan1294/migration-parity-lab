# When the route stops predicting the defect

An extension to the [budgeted inspection experiment](BUDGETED_INSPECTION.md), using only the existing synthetic fixture generator. All policies buy exactly two tool outputs. The experiment includes all ten fixed pairs (six direct-only), the original two-step learned router, the hand-written telemetry router, seeded random, and a hybrid. The hybrid buys a direct check first, uses telemetry on a hit to supply context, and buys a second direct check on a clear. Its direct pair and the fixed reference are selected separately using calibration labels only; both selected values + counts. All policies share a fitted observer and their own calibration-only threshold under a 15% observed false-positive-rate ceiling.

## Reproduce

```sh
python3 -m unittest -v
python3 routing_shift.py --self-test
python3 routing_shift.py > routing-report.json
```

There are 500 grouped development/holdout fixtures by default, five cases per group, and two separate sets of 166 previously unseen groups for the same synthetic shift. The stress generator weakens the link between route and defect family, not every aspect of the data. The report includes each pair, raw counts, score-threshold recall and false-positive rate, and distinct operational hold rates. Within each evaluation set, 1,200 paired group bootstrap draws resample groups, not individual cases, to estimate recall differences against the frozen fixed reference. These percentile intervals are conditional on the fitted policies and generator, not simultaneous bounds or external validation.

| Policy | Held-out recall / FPR | Shift 1 recall / FPR | Shift 2 recall / FPR |
| --- | ---: | ---: | ---: |
| Learned two-step | 63.64% / 8.39% | 23.84% / 8.38% | 22.81% / 7.34% |
| Telemetry heuristic | 63.64% / 8.39% | 23.84% / 8.38% | 22.81% / 7.34% |
| Fixed values + counts | 59.85% / 15.44% | 51.25% / 15.48% | 47.37% / 13.39% |
| Direct-first hybrid | 59.85% / 15.44% | 51.25% / 15.48% | 47.37% / 13.39% |
| Seeded random | 46.21% / 15.77% | 42.35% / 15.66% | 36.84% / 14.31% |

The learned router's held-out recall advantage over fixed is about 3.8 percentage points (paired interval −9.1 to +16.3). Under shift it loses about 27.4 points (−35.0 to −19.4) and 24.6 points (−32.0 to −16.8), with fewer false positives than fixed in both. This is a tradeoff, not dominance at the same realized FPR. The hybrid changes what gets observed on a hit but makes the same binary threshold calls as the fixed pair in these runs. It gives route context, not an observed detection gain. A 15% calibration ceiling is not a test-set guarantee.

The operational output is independent of the score threshold: an observed probe alert means hold for review; otherwise evidence is incomplete. Neither outcome approves a cutover. This experiment does not inspect real systems, and the shift replication reuses the generator assumptions. Before applying the design to a real migration, measure real probe cost, disagreement and class coverage on an approved benchmark and evaluate policy decisions against actual operational risk.
