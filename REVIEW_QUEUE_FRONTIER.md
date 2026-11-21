# Budget the review queue, not just the model warnings

A post-migration inspector has two independent reasons to send a case to a
human: a direct evidence **HOLD**, or a statistical **WARN**. Counting only
warnings misses mandatory holds; adding both counts double-counts overlap.
This experiment charges one review slot for the union, regardless of reason.
Cases without a direct hold still **ABSTAIN** on the evidence, including those
with a statistical warning. No case is cleared for cutover.

## Reproduce

```sh
python3 -m unittest -v
python3 review_queue_frontier.py > review-queue-report.json
```

The default is Python standard library only, with 20 independent refits,
80 groups (400 cases) per split, and 2,000 paired bootstrap draws. It uses
the existing synthetic generator and observer with a new default seed,
261121: disjoint synthetic group IDs from the earlier default runs, not
an independent generator or real data. There are seven separate group ranges
per refit: source training, then three calibration and three test regimes.

Every candidate purchases two distinct probes. The fixed family considers all
ten pairs. The hybrid anchors on the values probe, choosing telemetry after a
direct hit and the schema probe otherwise. For both families, selection
minimizes the worst regime's calibration loss, then mean loss; a higher
warning threshold and alphabetic policy name break ties. Loss is
`(5 × defects not queued + 1 × benign cases queued) / all cases`.
The experiment sweeps unique calibration scores, keeping ties intact; there
is also a no-warning option. A choice qualifies only if its realized
calibration queue size stays within the given cap in **each** regime.
Selection is frozen before generating held-out cases.

Direct holds cannot be turned off. A family with no qualifying candidate is
reported as infeasible, not compared with a made-up fallback. Held-out means,
breach counts and paired intervals below use jointly feasible repeats only;
the report retains each run's raw counts. At least two jointly feasible runs
are necessary for an interval. Resampling whole paired repeats gives marginal
intervals for the mean under the fixed generator, not simultaneous inference
or a production guarantee.

## Default result

| Nominal calibration queue cap | Fixed feasible | Hybrid feasible | Paired |
| ---: | ---: | ---: | ---: |
| 20% | 5/20 | 0/20 | 0/20 |
| 30% | 20/20 | 0/20 | 0/20 |
| 40% | 20/20 | 6/20 | 6/20 |
| 45% | 20/20 | 18/20 | 18/20 |
| 50% | 20/20 | 20/20 | 20/20 |

Hybrid's mandatory evidence holds alone occupied **37.25%–47.00%** of the
noisy calibration sets. At a 30% cap, turning off all statistical warnings
could not make the hybrid fit in any refit; every refit had a feasible fixed
pair. At 40%, only six refits permit a fair paired comparison. Four of those
six exceed the cap on noisy held-out cases under each family. Shared nominal
calibration caps are not shared realized workload.

At the 50% cap, comparisons use all 20 refits:

| Held-out regime | Queue rate (fixed → hybrid) | Defect recall (fixed → hybrid) | Paired hybrid-minus-fixed mean loss (95% interval) |
| --- | --- | --- | --- |
| Source | 25.50% → 25.89% | 50.08% → 50.78% | −0.007 [−0.064, +0.046] |
| Route shift | 25.40% → 24.74% | 50.66% → 48.91% | +0.030 [−0.015, +0.073] |
| Noisy shift | 42.39% → 42.44% | 61.15% → 59.74% | +0.030 [−0.013, +0.072] |

These cuts add no held-out statistical warnings at the 50% point; the
comparison there is driven by evidence holds. At 70%, the fixed family adds
statistical warnings in some refits and its mean noisy queue rises to 45.71%.
Do not read results across caps as a monotone held-out curve, because the
selected pair and threshold can change. No interval at 50% excludes zero.
At a tight cap, feasibility is the stronger finding; under noise, budget for
mandatory holds and route overflow to additional capacity rather than dropping
required review. Unqueued cases remain incomplete evidence.

This models a case count, not staffing minutes, team capacity, service times
or downstream review outcomes. It is a synthetic dispatch experiment, not
production performance or an operational cutover recommendation.
