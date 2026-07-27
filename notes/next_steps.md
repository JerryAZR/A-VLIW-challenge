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

0. **Borrowed from the 1063-cycle solution** (see "Prior art" section
   below for details + attribution): (1) hash stage 2+3 fusion -> 11-slot
   hash, floor 1077 -> ~1009; (2) wrap-root stage-5 ^C5 deferral (-4 cyc);
   (3) level-4 partial preloading - deletes the round-15 gather wall AND
   relieves the load floor (1063), which becomes binding after (1).
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
4. **Fine polish continues**: `weights/_weights_refined_l0i3.json` winner is one
   +0.25 fine step deep; more finetune budget may find 1-3 cyc.

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
   `b = 33a + C2`, and `<<` distributes over `+` mod 2^32, so
   `b << 9 = 16896a + (C2<<9)`; stages 2+3 become
   `(33a + (C2+C3)) ^ (16896a + (C2<<9))` = fma + fma + xor = 3 ops instead
   of 4. Verified numerically against the frozen constants (200k random
   inputs, exact match). Gain: -512 vec ops (~68 floor cyc); floor 1077 ->
   ~1009. Side effects: rigid fma per hash 3 -> 4 (still far from binding);
   new consts (mult 16896, addends C2+C3 and C2<<9), K2/K3 vecs die,
   const_vec_9 loses its stage-3-shift sharing. Touchpoints:
   `build_vec_hash`, fma/irr const dicts. **This falsifies the "verified
   12-slot hash minimum" in notes/hash_dag.md - update it when shipped.**

2. **Wrap-root stage-5 ^C5 deferral (round 10)** [from rubinownz111]. On the
   wrap round emit stage 5 as `val' = a ^ (a>>16)` (drop the ^C5); repair
   free via a precomputed `tree0_xor5_vec = tree0_vec ^ C5` broadcast that
   round 11's entry XOR reads instead of tree0_vec (LEVEL0_DIRECT_TREE0
   already reads it directly). Safe because round 10 skips the branch-bit &
   and addr update; the only other consumer is the dev `hashed_val`
   DebugVCompare (drop that one check on round 10). NOT dead code - the
   carried val is eventually stored, so prune_to_stores cannot find this;
   it's a reassociation across the round boundary. Gain: -32 vec ops (~4
   floor cyc). Cost: ~2 prologue ops + 1 live broadcast register.

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
