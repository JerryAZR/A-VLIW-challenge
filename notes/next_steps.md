# Next Steps & Current Status

Living planning document (updated as the plan evolves). The optimization log
(`notes/optimization_log.md`) is append-only historical entries; this file
holds the current tier matrix, the next levers, and forward-looking design
notes.

## Current tier status (after step 10: 1546 cyc)

| tier                     | threshold | status |
|--------------------------|-----------|--------|
| baseline                 | 147 734   | PASS   |
| updated-starting         | 18 532    | PASS   |
| opus4-many-hours         | 2 164     | PASS   |
| opus45-casual            | 1 790     | PASS   |
| opus45-2hr               | 1 579     | PASS   |
| sonnet45                 | 1 548     | PASS   |
| opus45-11hr              | 1 487     | FAIL (59 cyc short) |
| opus45-improved-harness  | 1 363     | FAIL   |

Shipped config: rounds-outer loop, weighted picker
`Weights(sink=-2, load=4, raw=-6, war=7, rigid=2)`, epilogue vstores overlapped
into the body = **1546 cyc**.

## Where we are

Cross-group WAR gone (step 8); picker property-weighted (step 9); const region
cleaned + 23 words freed (step 9b, scratch_ptr 1504->1481); epilogue vstores
overlapped with the body tail (step 10, -52 cyc). Body is gather-bound at
~1280 cyc (2560 loads / 2 ports); 1546 = 185 fixed prologue/epilogue + ~1361
body (vstores now hidden). Store engine saturated during the tail.

## Next levers (order = do the clear wins first, then train)

1. **Reduce op counts / dependency edges** (clear wins, pre-train): fewer
   slots and fewer edges both shrink the scheduling problem and the resource
   floor. Candidates:
   - hash-stage algebraic reductions beyond the verified 12-slot form,
   - collapsing the idx-update ops,
   - trimming debug edges (debug vcompares are 0-cycle but still graph nodes).
2. **Fill the remaining ~134 body cycles over the load floor** - overlap
   gather-round loads with select-round compute across groups. The weighted
   picker (load=+4, war=+7) already pushes this; a better load-feeding
   strategy (prefetch / both load ports) could close more.
3. **Real picker training** (planned): the weighted picker's 5 weights were
   found by ~180 random samples + refinement. Real training (e.g. gradient /
   Bayesian opt over weights, or a learned scoring net) needs a stable
   architecture first - land the op/edge reductions, then train.

## Deferred: Direction 2 (IR + register allocator)

Introduce an IR with pinned globals + renamed temporaries and a group-aware
register allocator that never coalesces across groups (so cross-group WAR
can't return). Future-proofs scratch management for further kernel
restructuring. Not needed while scratch fits the 5-plane coalesced layout
(step 8 reduced scratch usage). Trigger: when we need more scratch space for
other work, or when performance is done and we want to optimize scratch
usage instead.

Register renaming is familiar territory (hardware/CPU background); the
algorithm can be sketched jointly when we start.

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
