# Next Steps & Current Status

Living planning document (updated as the plan evolves). The optimization log
(`notes/optimization_log.md`) is append-only historical entries; this file
holds the current tier matrix, the next levers, and forward-looking design
notes.

## Current tier status (after step 20: 1203 cyc)

| tier                     | threshold | status |
|--------------------------|-----------|--------|
| baseline                 | 147 734   | PASS   |
| updated-starting         | 18 532    | PASS   |
| opus4-many-hours         | 2 164     | PASS   |
| opus45-casual            | 1 790     | PASS   |
| opus45-2hr               | 1 579     | PASS   |
| sonnet45                 | 1 548     | PASS   |
| opus45-11hr              | 1 487     | PASS   |
| opus45-improved-harness  | 1 363     | **PASS (1203, 160 clear)** |

All nine tiers pass. Shipped config: rollout scheduler **K=1** (greedy
priority only; K=6 adds just 2 cyc - raise K again after other parts
settle), `ScoreWeights(reads=2, reg_delta=-1)`, `REGALLOC_WEIGHTS =
Weights(sink=-1, load=5, raw=1, war=1, rigid=1, idx=-4, group=-4)`.
Prologue is merged into the body DAG (no serial prologue); pauses ride
existing bundles' flow slots.

## Current bottleneck (step-20 utilization)

**valu-bound**: 6 870 valu slots -> 1 145-cyc floor vs 1 203 actual (58
cyc slack). load 88.7%, alu 88.0%, flow 58.7%. Former-prologue ramp-up
(first ~60 cyc) runs ~60% utilized.

## Next levers

1. **K=N diverse-func trial sets** (deferred investigation): compose
   [func_w1, func_w2, func_w3, rand*2] from `_weights_refined.json`
   (singles) + `_weights_refined_interp.json`. NOTE: the K=N state-scoring
   function (ScoreWeights: reads/reg_delta) has never been properly
   trained - train it before judging K=N.
2. **Interp re-training** under the dynamic allocator (the 1126 interp
   result predates it).
3. **`LEVEL0_DIRECT_TREE0` retry under interp/K=N**: the op cut is real
   (~10 cyc floor) but the K=1 scheduler needs the copies as a frontier
   pacemaker; a better search may hold the flooded ramp-up. Candidates in
   `_weights_*_l0.json` (best 1127).
4. **valu op-count reduction** is otherwise exhausted (hash/xor/addr all
   at proven minimums); floor ~1085 combined is the algorithm's floor.
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
