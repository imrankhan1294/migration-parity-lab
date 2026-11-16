# What should the cutover investigator inspect next?

A four-policy experiment in buying evidence, not making go decisions. All policies get the exact same number of equal-cost observations and use the same fitted scorer. Each sees only its purchased tool output. An observed noisy probe alert independently forces a hold for review. Clean observations are incomplete evidence, not cutover approval.

## Reproduce

```sh
python3 -m unittest -v
python3 budget_policy.py --groups 500 --budget 2
```

Four noisy diagnostic probes represent values, schema, grants and counts. Optional telemetry gives a coarse and fallible operational clue about where to inspect; it is not free. A fitted joint observer estimates class probabilities from training examples. The learned policy enumerates tool outcomes up to two observations ahead and picks the tool expected to reduce uncertainty most. Its trace exposes observed results, estimated information gain and training support. Fixed buys direct probes in the predeclared order shown in the code; seeded random picks uniformly among unused tools; heuristic buys telemetry first and probes its mapped route next. Each scores only purchased results.

Training, threshold selection and testing use disjoint groups. Each policy's threshold is tuned only on the calibration groups to maximize recall under a 15% observed false-positive-rate ceiling there. This does not guarantee that ceiling in testing. Brier scores describe raw smoothed probabilities; the experiment does not claim probability calibration. The separate stress set uses an unseen group range and deliberately weakens the link between telemetry routes and defect families. No label, true family or group membership is available to a policy at inference.

## Default run (two observations each)

| Policy | Held-out recall | Held-out false-positive rate | Stress recall | Stress false-positive rate |
| --- | ---: | ---: | ---: | ---: |
| Learned two-step | 63.64% | 8.39% | 23.84% | 8.38% |
| Fixed direct order | 43.18% | 15.44% | 46.98% | 12.93% |
| Seeded random | 40.91% | 10.07% | 40.57% | 11.11% |
| Telemetry heuristic | 63.64% | 8.39% | 23.84% | 8.38% |

The learned policy recovered the obvious hand-written heuristic, **not an improvement over it**. Both missed most shifted defects; fixed direct inspection had better recall under that shift, though at a higher false-positive rate. Calibration and test constraints are not interchangeable. Raw counts, family-specific misses and example traces are in the command output.

## Scope

This synthetic observation simulator is separate from the CSV agent and snapshot benchmark. Actions do not query real databases. Costs are equal by construction; real scans have uneven cost, correlated errors and richer output. The fixed policy is one predeclared order, not the best possible fixed policy; the random baseline uses one reproducible seed. No confidence intervals or external validation are claimed. A stronger experiment should sweep all fixed tool subsets and multiple random seeds, report group-bootstrap intervals, test probability calibration, and collect trustworthy measured costs before using any result operationally.
