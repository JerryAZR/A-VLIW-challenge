# Next Steps & Current Status

Living planning document (updated as the plan evolves). The optimization log
(`notes/optimization_log.md`) is append-only historical entries; this file
holds the current tier matrix, the next levers, and forward-looking design
notes.

## Current tier status (after step 14: 1459 cyc)

| tier                     | threshold | status |
|--------------------------|-----------|--------|
| baseline                 | 147 734   | PASS   |
| updated-starting         | 18 532    | PASS   |
| opus4-many-hours         | 2 164     | PASS   |
| opus45-casual            | 1 790     | PASS   |
| opus45-2hr               | 1 579     | PASS   |
| sonnet45                 | 1 548     | PASS   |
| opus45-11hr              | 1 487     | PASS   |
| opus45-improved-harness  | 1 363     | FAIL (96 cyc short) |

Shipped config: rounds-outer loop, trained weighted picker (6 properties)
`Weights(sink=-3, load=-1.5, raw=-0.25, war=6, rigid=0.25, idx=-4)` = **1459 cyc**.

## DAG quality (unchanged since step 12; steps 13-14 are picker-only)

| metric              | idx (s10) | addr (s11) | addr+parity (s12-14) |
|---------------------|----------:|-----------:|---------------------:|
| nodes               | 16 864    | 16 160     | 15 776               |
| height (crit path)  | 223       | 216        | 204                  |
| RAW edges           | 23 840    | 23 104     | 22 720               |
| WAR edges           | 14 208    | 20 416     | 20 352               |
| valu nodes          | 8 832     | 8 640      | 8 256                |
| cycles              | 1 546     | 1 559      | 1 535 -> 1522 -> **1 459** |

The addr direction (steps 11-12) cut structure; steps 13-14 trained the picker
(5 props -> 6 props with idx). The 5-prop picker plateaued at 1514; adding idx
(reverse program order, idx=-4) broke through to 1459. Picker training
converged (coordinate descent + DE + random search all agree on the optimum).

## Next levers

The picker has plateaued at 1459 with 6 linear properties. To go lower:
1. **Richer features**: property interactions (e.g. sink×load, war×rigid) or
   new properties (engine-specific urgency, remaining-capacity-aware).
2. **Structural reductions**: reduce WAR edges (20k, the addr-plane churn) or
   reduce the prologue (123 cyc of setup).
3. **Direction 2 (IR + register allocator)**: automate scratch management for
   further kernel restructuring.

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
