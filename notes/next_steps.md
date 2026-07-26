# Next Steps & Current Status

Living planning document (updated as the plan evolves). The optimization log
(`notes/optimization_log.md`) is append-only historical entries; this file
holds the current tier matrix, the next levers, and forward-looking design
notes.

## Current tier status (after step 26: 1110 cyc)

| tier                     | threshold | status |
|--------------------------|-----------|--------|
| baseline                 | 147 734   | PASS   |
| updated-starting         | 18 532    | PASS   |
| opus4-many-hours         | 2 164     | PASS   |
| opus45-casual            | 1 790     | PASS   |
| opus45-2hr               | 1 579     | PASS   |
| sonnet45                 | 1 548     | PASS   |
| opus45-11hr              | 1 487     | PASS   |
| opus45-improved-harness  | 1 363     | **PASS (1110, 253 clear)** |

All nine tiers pass. Shipped config: rollout scheduler **K=1**,
`LEVEL0_DIRECT_TREE0=True`, progress-interpolated priority
`ROLLOUT_SORT_FUNCS = [make_interp_greedy(INTERP_W_LATE, INTERP_W_EARLY)]`
(early: serialize early groups group=-4.8 + sink push; late: no group
dial; freeing ~7-8 throughout). Prologue merged into the body DAG; pauses
ride existing bundles' flow slots.

## Current bottleneck (step-26 analysis)

Compute-work floor (alu + 8xvalu lane-slots, cap 60/cyc): **1077**
(64 578 lane-slots; L0-direct deleted the 64 copy ops). Schedule 1110 =
floor + 33 slack. `diag_underfill.py` classifies the slack: ramp-up
dependency latency (~4 cyc, irreducible), the round-15 gather-feed wall
(gather feeds 4 cyc/group at 2 load ports, hash drains 2.3 cyc/group -
synchronized arrivals force valu starvation), tail drain. The round-15
wall is a feed-rate problem, not register pressure (pressure slack is
small); one-step scheduling search cannot recover it.

## Next levers

1. **Op-count / structural** (the big pot now): the roofline note stands -
   sub-floor requires fewer than 4096 hashes (a structural dedup lever:
   identical (idx, val) lanes hash identically) or a sub-12-slot hash.
2. **K=N diverse-func trial sets + trained ScoreWeights** (deferred):
   bounded upside (~some of the 33 slack cyc); the scorer has never been
   trained. Evidence from step 26: do NOT expect it to fix the round-15
   feed wall; at best it shaves pressure/starvation elsewhere.
3. **Round-15 feed wall**: only structural fixes apply - spread arrivals
   (needs a delay/hold DOF the scheduler lacks; valu is too full for
   idling to be cheap) or reduce gather cost (no ISA scatter/gather).
4. **Fine polish continues**: `weights/_weights_refined_l0i3.json` winner is one
   +0.25 fine step deep; more finetune budget may find 1-3 cyc.

## Tools

- `tools/train_weights.py` v2: parallel trainer (multiprocessing pool, default
  min(28, cpu-4) workers, ~15 evals/s). Search space: raw dropped (inert),
  sign-biased per-prop bounds, interp as base+tilt, two-stage finetune
  deltas. Phases: `random <budget> --mode single|interp`,
  `finetune <budget> --mode ... --in <found.json> --out <refined.json>`.
- `diag_underfill.py`: per-cycle valu-underfill classification
  (register-pressure vs work-starvation) for the K=1 rollout schedule.
- `analyze_slots.py` (commit `cf59b74`): builds the kernel, extracts the
  scheduled body bundles, and plots per-cycle slot usage by engine
  (valu/load/alu/flow) as separate subplots, each scaled to its own capacity.
  Two views: per-cycle bars (shows the alternating gather/compute pattern)
  and 10-cycle rolling average (shows macro phase trends). Usage:
  `python -m tools.analyze_slots [--show] [--picker fma_first|idx|random]`.
- `pmu.py`: `InstrumentedMachine` subclasses the frozen simulator to count
  slot fires / op breakdowns / per-cycle histograms without touching it.
  Run: `python -m tools.pmu`.

## Roofline reminders

(See `notes/architecture.md`.) Compute-work floor now 1077 cyc (64 578
lane-slots over 60/cyc). The Opus-4.5 1487 score sits far above. Sub-1k
requires <=11 slots/lane/round (below the verified 12-slot hash minimum)
or fewer than 4096 hashes (a structural dedup lever).
