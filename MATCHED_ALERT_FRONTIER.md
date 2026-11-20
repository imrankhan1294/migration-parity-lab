# What changes when both selectors get the same warning budget?

The [prior repeated experiment](README.md#does-robust-routing-hold-up-when-retrained)
changed which synthetic regimes had to satisfy a calibration false-positive cap.
This follow-up holds the calibration constraint constant: **both selectors get
the same cap on the fraction of all cases they may warn on, separately in each
of three calibration regimes**. It varies that cap rather than selecting a single
point. A false-positive rate is still measured separately: its denominator is
benign cases, not all cases.

## Reproduce

```sh
python3 -m unittest -v
python3 repeated_robust_routing.py --self-test
python3 matched_alert_frontier.py > matched-alert-report.json
```

Python standard library only. No real data, connections, credentials, or actions.
The default uses 20 independent refits, 80 groups (400 cases) in each of seven
disjoint splits per refit, and 2,000 paired bootstrap draws. It deliberately
reuses the prior experiment's fixed default seed, synthetic group ranges,
generator and observer: held-out groups are disjoint from this run's
calibration but **not new confirmation populations relative to the prior
experiment**. Use `--seed` to probe a different synthetic draw, not to tune on
the example results. `--budgets` takes ascending integer percentages.

Each refit purchases two distinct tool outputs per case. It sweeps all ten
fixed tool pairs and four adaptive policies once over the three shared
calibration subsets (source, route shift, noisy shift). For each cap, each policy
is eligible at any observed calibration cut satisfying the cap in **all three**
subsets. WARN means score greater than or equal to the cut; score ties stay
together. An infinite cut, serialized as null, never warns. The source-loss
selector minimizes source calibration loss; the minimax selector minimizes the
worst calibration loss, then mean calibration loss. Both then prefer a higher
cut and order remaining ties by policy name. Loss is
`(5 × missed defects + 1 × warnings on benign cases) / all cases`.
All selections are frozen before generating held-out outcomes. For each test
regime, every selected policy is observed once, then its same observations are
reused across the frontier.

The report includes raw counts and calibration metrics for every selection,
held-out cap-breach counts, realized warning rates, FPR, recall, loss, hold and
abstain rates. Paired differences reuse equal group ranges and resample whole
independent refits, not individual fixtures. The report distinguishes
repeat-to-repeat variability from percentile intervals on the across-refit
mean. Intervals are marginal and limited to this fixed toy generator.

## Default result

At a **20% calibration warning cap** (source-loss → minimax):

| Held-out regime | Warn rate | FPR | Recall | Mean loss | Cap-breach runs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source | 14.86% → 10.86% | 4.66% → 5.00% | 34.81% → 22.37% | 1.132 → 1.345 | 0 → 0 of 20 |
| Route shift | 9.08% → 10.69% | 4.86% → 5.16% | 17.46% → 21.78% | 1.403 → 1.334 | 0 → 0 of 20 |
| Noisy shift | 17.39% → 19.60% | 13.84% → 14.29% | 24.22% → 30.06% | 1.371 → 1.278 | 4 → 9 of 20 |

The paired minimax-minus-source mean loss difference is +0.214 on source
(95% percentile interval +0.157 to +0.268), −0.069 on route shift (−0.107 to
−0.034) and −0.093 on noisy shift (−0.147 to −0.038). On noisy shift the mean
absolute within-refit gap in realized warning rates is 3.14 percentage points.
At a 30% calibration cap, the noisy-shift mean reaches 30.20% with minimax
versus 23.46% with the source objective, and minimax exceeds the cap in 14 of 20
runs, versus none for source-loss. On the 5% cap, the paired noisy-shift loss
interval spans zero. These are tradeoffs at matched **nominal calibration
caps**, not matched realized review loads or FPR, and not a uniform improvement.

A statistical WARN and an evidence HOLD can coincide but are not interchangeable.
The JSON shows their full cross-tabulation. A purchased direct-probe alert
means HOLD_FOR_REVIEW even if the score does not warn. Without a direct-probe
alert the trace ABSTAINS as INCOMPLETE_EVIDENCE, with or without a score
warning. Even at the zero-warning point, 41.45% of noisy-shift cases receive
an evidence hold under the tied default policy. This is not a cap on operational
review workload. No outcome authorizes a cutover.
