# Next Steps & Current Status

Living planning document (updated as the plan evolves). The optimization log
(`notes/optimization_log.md`) is append-only historical entries; this file
holds the current tier matrix, the next levers, and forward-looking design
notes.

## Current tier status (after step 19: 1289 cyc)

| tier                     | threshold | status |
|--------------------------|-----------|--------|
| baseline                 | 147 734   | PASS   |
| updated-starting         | 18 532    | PASS   |
| opus4-many-hours         | 2 164     | PASS   |
| opus45-casual            | 1 790     | PASS   |
| opus45-2hr               | 1 579     | PASS   |
| sonnet45                 | 1 548     | PASS   |
| opus45-11hr              | 1 487     | PASS   |
| opus45-improved-harness  | 1 363     | **PASS (1289, 74 clear)** |

All nine tiers pass. Shipped config: rollout scheduler (K=6, sort funcs
[greedy] + [random]*5), `ScoreWeights(reads=2, reg_delta=-1)`,
`REGALLOC_WEIGHTS = Weights(sink=-1, load=5, raw=1, war=1, rigid=1,
idx=-4, group=-4)` (group priority: finish earlier DAG sink groups first).

## Next levers

1. **Score/feature training**: coordinate descent over ScoreWeights +
   REGALLOC_WEIGHTS (now 7 dims incl. group) around the manual minimum.
2. **Sort-func mix**: with group=-4 the greedy trial dominates (K=3 ties
   K=6); try K=2-3 for iteration speed, or interp variants once the base
   weights are retrained.
3. **Retention retry** (`RECOMPUTE_PATH_BITS=False`, ~26 cyc cheaper at big
   scratch): pressure is now metered by group priority - retest at 1536.
4. **Slot-fill performance terms** (alu_work / load-fill) now that
   feasibility is secured.
5. **Horizon depth** (parked, non-trivial): trials = 1 decided cycle + H
   greedy continuation cycles, scoring the horizon state.

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
