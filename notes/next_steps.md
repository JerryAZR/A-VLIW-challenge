# Next Steps & Current Status

Living planning document (updated as the plan evolves). The optimization log
(`notes/optimization_log.md`) is append-only historical entries; this file
holds the current tier matrix, the next levers, and forward-looking design
notes.

## Current tier status (after step 28: 1076 cyc)

| tier                     | threshold | status |
|--------------------------|-----------|--------|
| baseline                 | 147 734   | PASS   |
| updated-starting         | 18 532    | PASS   |
| opus4-many-hours         | 2 164     | PASS   |
| opus45-casual            | 1 790     | PASS   |
| opus45-2hr               | 1 579     | PASS   |
| sonnet45                 | 1 548     | PASS   |
| opus45-11hr              | 1 487     | PASS   |
| opus45-improved-harness  | 1 363     | **PASS (1076, 287 clear)** |

All nine tiers pass. Shipped config: rollout scheduler **K=1**,
`LEVEL0_DIRECT_TREE0=True`, `FUSE_HASH_STAGES_23=True` (11-slot hash),
`WRAP_ROOT_C5_DEFER=True`, `PRELOAD_L4_GROUPS=16` (round-15-only scope,
JIT consts), progress-interpolated priority
`ROLLOUT_SORT_FUNCS = [make_interp_greedy(INTERP_W_LATE, INTERP_W_EARLY)]`
(freeing-heavy 13-16; group -4.8 early / -1.6 late). Prologue merged
into the body DAG; pauses ride existing bundles' flow slots.

## Current bottleneck (step-28 analysis)

**Load floor 1063 binding; schedule 1076 = floor + 13.** valu 1055,
flow ~950 (16 preloaded groups x 15 selects), alu ~823. The load floor
is the frontier: level 5-9 gathers remain (5 rounds x 32 groups x 8
loads = 1 280 of the 2 069 remaining loads). Options against it:
level-5+ preloading is 31+ selects/node on the 1-wide flow port
(prohibitive); fewer loads means fewer gathers means either more
preload levels (flow-bound) or the dedup lever.

## Next levers

0. **Borrowed from the 1063-cycle solution** (see "Prior art" section
   below for details + attribution): (1) hash stage 2+3 fusion -> 11-slot
   hash - DONE (step 27); (2) wrap-root stage-5 ^C5 deferral - DONE
   (step 27); (3) level-4 partial preloading - DONE (step 28):
   round-15-only scope at G=16 won (both-rounds loses on flow-port
   cost); 1100 -> 1076.
1. **Op-count / structural** (the big pot now): the roofline note stands -
   sub-floor requires fewer than 4096 hashes (a structural dedup lever:
   identical (idx, val) lanes hash identically). The "<=11-slot hash"
   route is now covered by lever 0(1).
2. **K=N diverse-func trial sets + trained ScoreWeights** (deferred):
   bounded upside (~some of the 33 slack cyc); the scorer has never been
   trained. Evidence from step 26: do NOT expect it to fix the round-15
   feed wall; at best it shaves pressure/starvation elsewhere.
3. **Round-15 feed wall**: SOLVED IN PRINCIPLE by lever 0(3) - the wall is
   the level-4 gather; convert it to preloaded select trees (flow/valu)
   instead of re-spreading arrivals. (Prior dismissal - "no structural
   fix, no ISA gather" - missed that 16 nodes are preloadable.)
4. **Fine polish continues**: `weights/_l4_g16_refined.json`
   converged; G=14/18 probed worse. Knob space is one-dimensional and
   mapped; remaining variance is weight-basin luck.

## Prior art: the 1063-cycle solution (external)

PROVENANCE: everything up to and including 1110 cyc (step 26, commit
`e728293`) was independent work, developed without knowledge of other
solutions (record: `JOURNEY.md` + `notes/optimization_log.md`, steps
1-26). Every optimization below 1110 borrows from external solutions -
principally the repo below - and is credited item-by-item here and in
the optimization log when shipped.

Repo: github.com/rubinownz111/1063-cycles-original-performance-takehome
(author: rubinownz111; local copy at ~/Projects/1063-cycles-original-
performance-takehome). Its `problem.py` is byte-identical to ours, so every
trick transfers. Their `perf_takehome.py` reaches 1063 cyc; their `AGENTS.md`
is a detailed optimization journal. Ideas below are due to that repo's author
- credit rubinownz111 if any of this ships.

Their true per-engine bounds: valu 5999/6 = 1000, alu 11964/12 = 997,
load 1991/2 = 996 -> floor ~1000, they land 63 above it.

Ideas to borrow (in planned order):

1. **Hash stage 2+3 algebraic fusion (12 -> 11 slots)** [from rubinownz111].
   IMPLEMENTED (FUSE_HASH_STAGES_23=True, attribution in the toggle
   comment): fused form derived from frozen HASH_STAGES with shape
   asserts; const_vec_16896 computed as 33<<9 in the prologue (no const
   load); K2/K3 die, K23/K2S9 const-loaded; stage-2 DebugVCompare
   dropped. Verified -512 vec ops in the scheduled program (valu 6566 ->
   6327, alu 12050 -> 9874); floor 1077 -> 1009 combined, load 1063 now
   the binding per-engine bound. CORRECT via tools._check at 1536;
   1154 cyc on the OLD weights (+44, expected - priority fn needs
   retraining on the fused DAG).
   Mechanism: `b = 33a + C2`, and `<<` distributes over `+` mod 2^32, so
   `b << 9 = 16896a + (C2<<9)`; stages 2+3 become
   `(33a + (C2+C3)) ^ (16896a + (C2<<9))` = fma + fma + xor = 3 ops
   instead of 4 (numerically verified, 200k random inputs exact).
   **This falsifies the "verified 12-slot hash minimum" in
   notes/hash_dag.md - update it when shipped.**

2. **Wrap-root stage-5 ^C5 deferral (round 10)** [from rubinownz111].
   IMPLEMENTED (WRAP_ROOT_C5_DEFER=True, attribution in the toggle
   comment): round-10 stage 5 emits a ^ (a>>16); round-11 entry XOR reads
   tree0_xor5_vec (precomputed just-in-time at round-11 start, short
   liveness). Oracle checks skipped where the trace includes C5:
   round-10 stage-5 + hashed_val, round-11 pre-entry val. CORRECT via
   tools._check_big (4096 scratch, 1152 cyc); DEADLOCKS at 1536 (fusion
   alone had 8854 exhaustion stalls - zero-slack pressure; the extra live
   vector tips it over). Fit is expected from the priority retrain (the
   trainer scores deadlocks as non-completes and searches around them).
   Mechanism: on the wrap round the & branch-bit and addr update are
   skipped, so nothing reads the deferred val's parity; the only other
   consumer is round 11's entry XOR, repaired free via tree0^C5. NOT dead
   code - prune_to_stores cannot find this; it's a reassociation across
   the round boundary.

3. **Level-4 preloading / partial preloading** [from rubinownz111; the
   follow-on once 1+2 land]. Loads are already nearly co-binding (2125
   slots -> floor 1063 vs compute floor 1077); after fusion drops compute
   to ~1009, loads become THE constraint. The round-15 feed wall IS the
   level-4 gather (rounds 4 and 15) - don't re-spread it, delete it.
   Depth 4 = 16 contiguous nodes (tree[15..30]): 2 prologue vloads + 16
   broadcasts, then a 4-bit select tree per group - vselect nodes on the
   idle flow engine, diff-pair leaves `bit*(odd-even)+even` = one fma on
   valu (8 diff vectors precomputed once, shared by all groups and both
   rounds). Full conversion: -512 scalar loads (load floor 1063 -> ~807)
   for ~960 select ops split flow/valu. The knob is PARTIAL conversion
   (convert g of 32 groups, tune g; their final 1064->1063 step added one
   group to the select set). Risks: 16 broadcast + 8 diff registers live
   kernel-wide (why we stopped at level 3; they peaked 1465/1536 scratch);
   interacts with RECOMPUTE_PATH_BITS - a round-4 select needs path[0..3]
   live into round 4 while the recompute trick frees path[0..2] at the
   round-3 select (retention window shifts one round).

Deliberately NOT borrowed (revisit only if a wall remains):

- **1-indexed addressing** [rubinownz111]: bias the stored value by
  (1-forest_p) so the child update is `2a + bit` = one fma with the path
  bit as addend. Wash on gather rounds (the add comes back to form the
  gather address); saves 1 op only on the 7 select rounds = ~224 vec ops,
  but requires reworking everything touching the addr plane (round-0 init,
  wrap reset, level-3 path-bit recompute, gather formation). With true
  addresses the update constant is irreducible - no fold without the
  gather-side add.
- **Retroactive ALU packing**: split pending vector ops into 8 scalar ALU
  lanes inserted into PAST cycles' partially-filled bundles (safe under
  end-of-cycle write semantics). We only spill within the current cycle.
- **Round-gate tokens**: cap groups in flight during load-heavy rounds -
  the hold/delay DOF we noted lacking. Moot if level-4 selects land.

Already-equivalent items (verified, no action): last-round dead index
update (our prune_to_stores removes all 32 final addr[g] writers +
path[g][4] & + feeding t2 fmas, 111 nodes; their emission-time skip has
identical net effect), wrap-round idx-update skip, level-0 direct tree0,
valu->alu spill, const-0 via zero-init scratch.

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
lane-slots over 60/cyc) - but per-engine floors are nearly co-binding:
load 2125/2 = 1063, valu occupancy 6566/6 = 1094. The Opus-4.5 1487 score
sits far above. Sub-1k requires <=11 slots/lane/round or fewer than 4096
hashes (a structural dedup lever). NOTE: the "verified 12-slot hash
minimum" claim was WRONG - the stage 2+3 fusion above (rubinownz111's
trick) yields an 11-slot hash; see "Prior art" section.
