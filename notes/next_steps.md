# Next Steps & Current Status

Living planning document (updated as the plan evolves). The optimization log
(`notes/optimization_log.md`) is append-only historical entries; this file
holds the current tier matrix, the next levers, and forward-looking design
notes.

## Current tier status (after step 8: 1773 cyc)

| tier                     | threshold | status |
|--------------------------|-----------|--------|
| baseline                 | 147 734   | PASS   |
| updated-starting         | 18 532    | PASS   |
| opus4-many-hours         | 2 164     | PASS   |
| opus45-casual            | 1 790     | **PASS** (cleared at step 8) |
| opus45-2hr               | 1 579     | FAIL (194 cyc short) |
| sonnet45                 | 1 548     | FAIL   |
| opus45-11hr              | 1 487     | FAIL   |
| opus45-improved-harness  | 1 363     | FAIL   |

Shipped config: rounds-outer loop, `random` picker (seed=42) = **1773 cyc**.

## Where we are

The cross-group WAR is gone (step 8): the body is loop-order-independent and
gather-bound at ~1280 cyc (2560 loads from rounds 3-10, 14-15 / 2 ports). The
readiness lower bound is the within-group hash critical path (~223 RAW edges)
in both loop orders, so the body is resource-bound, not dependency-bound.

All deterministic priority functions (`idx`, `fma_first`) land at ~1822-1827;
`random`+seed=42 hits 1773 via a lucky shuffle (variance 1729-1799 across
seeds). The picker is confirmed sub-optimal.

## Next levers (order = do the clear wins first, then train)

1. **Reduce op counts / dependency edges** (clear wins, pre-train): fewer
   slots and fewer edges both shrink the scheduling problem and the resource
   floor. Candidates:
   - hash-stage algebraic reductions beyond the verified 12-slot form,
   - collapsing the idx-update ops,
   - trimming debug edges (debug vcompares are 0-cycle but still graph nodes).
2. **Fill the 334 idle load cycles** - during preload-select rounds (0-2,
   11-13) both load ports sit idle. The clean DAG (no cross-group WAR) now
   lets the scheduler overlap gather-round loads with select-round compute
   across groups; a better picker realizes this.
3. **Trained picker** (planned): replace the static priority functions with a
   learned scoring function (e.g. sum of weight * node-property). Requires an
   otherwise stable architecture, so land the clear-win op/edge reductions
   first.

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

## Roofline reminders

(See `notes/architecture.md`.) Compute floor ~1280-1600 cyc (12-slot hash ×
4096 lane-rounds over 6 valu + 12 alu/cyc); the Opus-4.5 1487 score sits in
that band. Sub-1k requires ≤11 slots/lane/round (below the verified 12-slot
hash minimum) or fewer than 4096 hashes (a structural dedup lever).
