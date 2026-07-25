# Next Steps & Current Status

Living planning document (updated as the plan evolves). The optimization log
(`notes/optimization_log.md`) is append-only historical entries; this file
holds the current tier matrix, the next levers, and forward-looking design
notes.

## Current tier status (after step 18: 1413 cyc)

| tier                     | threshold | status |
|--------------------------|-----------|--------|
| baseline                 | 147 734   | PASS   |
| updated-starting         | 18 532    | PASS   |
| opus4-many-hours         | 2 164     | PASS   |
| opus45-casual            | 1 790     | PASS   |
| opus45-2hr               | 1 579     | PASS   |
| sonnet45                 | 1 548     | PASS   |
| opus45-11hr              | 1 487     | PASS   |
| opus45-improved-harness  | 1 363     | FAIL (50 cyc short) |

Shipped config: rollout scheduler (per-cycle trial-and-score, K=6, trial 0
= weighted greedy + shuffles), `ScoreWeights(reads=2, reg_delta=-1)` =
**1413 cyc** at 1536 scratch, level-3 preload-select kernel active.

## Next levers

1. **Close the 50-cyc gap to 1363**: the big-scratch validation of the same
   dataflow ran 1389 (recompute) / 1363 (retention). Retention
   (`RECOMPUTE_PATH_BITS=False`, ~26 cyc cheaper) needed +64 granules the
   greedy scheduler couldn't fit - retry it under the rollout scheduler,
   which meters pressure by construction.
2. **Score/feature training**: the reads=2/rd=-1 point is a manual sweep
   minimum; coordinate descent over ScoreWeights (and K) may find more.
3. **Horizon depth** (parked, non-trivial): trials = 1 decided cycle + H
   greedy continuation cycles, scoring the horizon state. Unlocks both
   deadlock-avoidance foresight and performance tuning. Needs careful
   planning before attempting.
4. **Slot-fill performance terms**: with pressure solved, alu_work /
   load-fill terms become performance levers rather than feasibility
   noise - retrain with feasibility secured.

## Tools

- `analyze_slots.py` (commit `cf59b74`): builds the kernel, extracts the
  scheduled body bundles, and plots per-cycle slot usage by engine
  (valu/load/alu/flow) as separate subplots, each scaled to its own capacity.
  Two views: per-cycle bars (shows the alternating gather/compute pattern)
  and 10-cycle rolling average (shows macro phase trends). Usage:
  `python analyze_slots.py [--show] [--picker fma_first|idx|random]`.
- `pmu.py`: `InstrumentedMachine` subclasses the frozen simulator to count
  slot fires / op breakdowns / per-cycle histograms without touching it.
  Run: `python pmu.py`.

## Roofline reminders

(See `notes/architecture.md`.) Compute floor ~1280-1600 cyc (12-slot hash ×
4096 lane-rounds over 6 valu + 12 alu/cyc); the Opus-4.5 1487 score sits in
that band. Sub-1k requires ≤11 slots/lane/round (below the verified 12-slot
hash minimum) or fewer than 4096 hashes (a structural dedup lever).
