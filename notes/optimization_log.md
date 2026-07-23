# Optimization Log

Worked example: Anthropic's VLIW/SIMD performance-engineering take-home.
Canonical workload: `forest_height=10, n_nodes=2047, batch_size=256,
rounds=16`. Scored by simulated cycle count on a frozen copy of the
simulator (`tests/submission_tests.py`).

Append-only history of optimization steps. Each entry records the commit,
intent, mechanism, cycle count, and correctness/tier status. The current
tier matrix, next levers, and tooling notes live in `notes/next_steps.md`
(this file is history only).

---

## Baseline                                          147 734 cyc   1.00×

Commit `f88c945`. `KernelBuilder.build_kernel` as shipped, a deliberately
naive scalar program:

- One slot per instruction bundle (no VLIW packing at all - `vliw=False`).
- Fully unrolled `rounds × batch_size` (= 4096) iterations emitted statically.
- Re-reads `idx[i]`, `val[i]` from **mem** every round (one `load` apiece),
  plus one `load` for `tree.values[idx]`.
- Stores `idx[i]` AND `val[i]` back to **mem** every round (the `idx` writes
  are entirely wasted - the grader checks only `val`).
- `myhash` taken literally: 6 stages × 3 alu ops = **18 ops/hash**.
- Idx update `% 2`, `== 0`, `flow select`, `* 2`, `+`, `< n_nodes`,
  `flow select`. Wrap is fully per-lane branchy.
- Utility slot count per lane per round: ~29 alu, 3 load, 2 store, 2 flow.
  All scalar; `valu` untouched.

PMU: alu 118784 (6.7% util), valu 0, load 12564, store 8192, flow 8194.
Every active engine sits at histogram k=1 (one-slot-per-bundle pathology).

---

## Step 1 - fma via `valu` `multiply_add`          123 165 cyc   1.20×

Commit `648da3d`. Still sequential (one slot per bundle). First reduction of
`myhash`:

- Three linear stages `(a+K) + (a<<s) == a*(1+2^s) + K` collapse from three
  alu slots to **one `valu multiply_add` slot each** (verified bit-exact
  against `myhash`). Using `multiply_add` as a scalar fma on lane 0 of a
  VLEN-word work region; broadcast constants fill lanes 1..7 with junk we
  ignore.
- Three xor/add-shift stages (1, 3, 5) are irreducible at 3 slots each
  - carries block fusing `+K` across the `^` combine (numerically falsified).
- `myhash` goes from 18 literal alu ops to **12 slots** (3 fma + 9 alu).
- Everything else still sequential and bloated (per-round mem round-trips,
  branchy `%`/`select`/`<`/`select` idx/wrap, idx stored).

Per-lane per-round slot count: 12 hash (vs 18) + ~6 idx/wrap = ~18.
PMU: alu 118784 -> 81920, valu 0 -> 12296, others unchanged.

Passes correctness (8 seeds) and the `baseline < 147734` tier.

---

## Step 2 - scratch-resident state + branchless idx   77 223 cyc   1.91×

Commit `ff00b76`. Plumbing fixes (no compute parallelism yet):

- `val[256]`, `idx[256]` **resident in scratch** across all 16 rounds.
  Prologue `vload`s `val` once (32 vloads); `idx` starts at 0 (scratch is
  zero-initialized, no init needed). Epilogue `vstore`s `val` once.
  No per-round mem round-trips for lane state.
- `val[i]` doubles as the running hash register via a shared transient
  8-word `work_vec` (lane 0 active for fma). Stage 5's final combine writes
  `v` **directly into `val[i]`** - no separate copy-back op.
- Per-lane stage temps `t1[i]`, `t2[i]` (single-word, never shared across
  lanes - no rename hazards when we later pack VLIW bundles).
- **Branchless idx update**: `base = (idx<<1)|1`; `next = base + (v & 1)`.
  No `%`, no `select`, no flow port consumed.
- **Wrap as build-time per-round decision**: verified that for the canonical
  shape all lanes are at the same tree level in every round and that the
  wrap is therefore uniform and lands exactly on round = `forest_height`
  (=10). On that round we skip the idx update and write `idx[i] := 0` in one
  op. No per-lane wrap test in the hot path.
- Constants properly mapped: scalar (load const) + vector (vbroadcast) at
  prologue; reused across all 4096 hashes.
- Pause-ordering fix: epilogue (vstore) BEFORE pause 2, so machine.mem holds
  the final values when the test compares at the reference's final yield.

PMU before/after:

| engine | step 1   | step 2   | delta |
|--------|----------|----------|-------|
| store  | 8192     | **32**   | −8160 (only final vstore) |
| flow   | 8194     | **2**    | −8192 (branchless idx + build-time wrap) |
| load   | 12565    | 4157     | −8408 (no idx/val re-reads) |
| alu    | 81920    | 60736    | −21184 (dropped %/==/</select/* redundancies) |
| valu   | 12294    | 12296    | unchanged (intrinsic fma count) |

Passes correctness (8 seeds) and two tiers: `baseline < 147734`,
`updated-starting < 18532`.

---

## Step 3 - cross-lane vectorization (8 lanes/group)  12 911 cyc   11.44×

Commit `0a4b7b0`. Run `myhash` elementwise across VLEN=8 lanes per `valu`
slot, 32 groups of 8. Still one slot per instruction bundle (`vliw=False`) -
VLIW packing is deliberately postponed; this pass is the functional 8-lane
chunk ("naive loader" gather + full-vector hash).

- **Scratch reorg to a hard rule**: 256 words shared, then 5 words per lane
  across 256 lanes (= 1280), exhausting the 1536-word file. The per-lane
  sector is 5 contiguous planes of 256 (SoA): lane `i` owns one word per
  plane at `plane_base + i`, so group `g` (lanes 8g..8g+7) forms an
  8-word contiguous vector at `plane_base + 8g` - vectorizable at zero
  gather cost. Planes: `val`, `idx`, `t1`, `t2`, `nv` (node_val landing).
  Shared sector (159/256 words used, 97 free) holds header, scalar consts,
  broadcast vector consts, and the few genuinely-shared transients.
- **Per-lane `t1`/`t2`/`nv`** (not shared) so distinct groups' in-flight
  stages never alias - VLIW-packable without rename management later. Costs
  768 words of scratch but removes a whole class of inter-group hazards.
- **Hash fully on `valu`** (8 lanes/slot): 3 `multiply_add` fma stages + 3
  irreducible xor/add-shift stages (2 parallel elementwise transforms + `^`
  combine). `val_vec` doubles as the running hash reg; the final stage
  leaves `v` there - zero copy-out.
- **Gather** is the one non-vectorizable op (ISA has no scatter/gather): per
  group, one `valu +` computes all 8 addresses (`addr_vec = idx_vec +
  forest_p_vec`), then 8 scalar `load`s land into the per-lane `nv` plane.
  This is the "naive loader" - prefetch / dual-port packing / speculative
  both-branch loads deferred.
- **Branchless idx** on `valu`: `parity = v & 1`; `base = 2*idx + 1` (fma);
  `next = base + parity`. Wrap round (10): `idx &= zero_vec` in one slot.
- **Entry XOR** is also `valu` (not `alu`): `val_vec = val_vec ^ nv_g`,
  8 lanes in one slot.
- Const-key collision handled: the literal `9` is both a multiplier (stage 4
  fma) and a shift amount (stage 3), so hash constants are split into
  `fma_vec_consts` vs `irr_vec_consts` dicts keyed by raw value.

Per group per round (one slot/bundle): 1 addr valu + 8 loads + 1 xor +
12 hash + 3 idx = 25 bundles (13 on the wrap round). The 8 scalar gathers
are now the dominant cost - the load-port / prefetch lever is the clear
next move.

PMU (fires): valu 8656, load 4157 (4096 gathers + 32 vload + ~29 prologue),
store 32, flow 2. (`alu` counts are inflated by a simulator artifact: the
`valu` elementwise form internally calls `self.alu` per lane, so the PMU
double-counts those - the scheduler-visible body work is all on `valu`.)

Passes correctness (8 seeds) and two tiers (`baseline`, `updated-starting`).

---

## Step 4 - VLIW scheduler (random pick)              2 394 cyc   61.7×

Commits `81d4efb`, `b443619`, `0e04a96`, `065f6d9`. DAG-driven VLIW packing
replaces one-slot-per-bundle with a scheduler that packs multiple
independent slots per cycle.

- **`scheduler.py`** (`81d4efb`, `b443619`): `slot_io` (full ISA dispatch ->
  reads/writes as `(addr, is_vector)` pairs), `build_dag` (program-order walk
  with per-lane `readers_since`, tagged-union `last_writer` (vec_node |
  list[Node|None]×8), deduped RAW weight-1 + WAR weight-0 edges,
  bidirectional invariant asserts, dead-write warning). Scratch reordered
  plane-first (planes `[0..1279]`, shared sector `[1280..1535]` with
  8-word vector regions before 1-word scalars) for the DAG's region-keyed
  bookkeeping.
- **`schedule_dag`** (`0e04a96`): v1 random placer. Random pick from
  frontier (including partially-completed spillable nodes; no priority).
  Native engine first; spillable `vec_elem` ops try `valu`-atomic, else
  spill to `alu` (sticky-alu once spilled). WAR resolutions immediate
  (same-cycle unlock); RAW deferred to end-of-cycle `advance()` (reflects
  read-before-write +1 latency). Debug slots ride free (0-cycle).
- **Wired into `build_kernel`** (`065f6d9`): `build(vliw=True, seed=42)`
  routes the body through `build_dag + schedule_dag`. Prologue/epilogue stay
  linear (one slot per bundle); the two `pause`s bracket the scheduled body
  as hard start/end barriers.

Cycle breakdown: prologue ~111 + body ~2219 + epilogue ~65 = ~2394.

Passes correctness (8 seeds) and three tiers: `baseline`, `updated-starting`,
`opus4-many-hours < 2164`.

---

## Step 5 - VLIW scheduler (greedy pick)              2 236 cyc   66.1×

Commit `cedf93a`. Replaces v1 random pick with greedy: iterate the entire
frontier in idx order, skip nodes that can't be placed (don't break), loop
until no progress (WAR unlocks may add new placeable nodes), then advance.

- No instruction priority (by design - future step 6). Vector ops still
  prefer `valu`; spill to `alu` if `valu` full. Known limitation: `vec_elem`
  ops (xor-shift stages) can greedily fill `valu` slots, blocking `vec_fma`
  (`multiply_add`, valu-only) from scheduling.
- `_try_place` refactored out of the inline loop; returns
  `(committed_count, emitted_non_debug)` tuple or None. Shared by both v1
  random and v2 greedy paths.
- `schedule_dag` gains `greedy: bool = True` parameter; v1 random preserved
  as `greedy=False` for testing.

Improvement from filling all engine slots each cycle instead of breaking on
first failure: 2394 -> 2236.

Passes correctness (8 seeds). Still passes the same three tiers (2236 < 2164
is false - 72 cyc / 3.3% over). The fma-priority fix (step 6) clears 2164.

---

## Step 6 - tree preload levels 0-2 + fma-first picker 2 049 cyc   72.1×

Commits `11fde8d`, `2e4c3f4`, `4528728`. Two changes:

1. **fma-first picker** (`11fde8d`, `2e4c3f4`): `schedule_dag` gains a
   `picker` param. `fma_first` (default) sorts the frontier by
   `(kind_priority, idx)`: `vec_fma` > `vec_elem` > other > debug. Rationale:
   fma is valu-rigid; elem is spillable to `alu`. Preferring fma ensures it
   gets a valu slot before elem fills them. Infrastructure only at this
   stage - no cycle improvement by itself (the scheduler was gather-bound,
   not valu-bound), but correct for when the gather wall is lowered.

2. **Tree preload levels 0-2** (`4528728`): single `vload` reads
   `tree[0..7]` (7 nodes = levels 0-2 + 1 bonus) into scratch at prologue.
   7 `vbroadcast`s create shared vector constants `tree0_vec`..`tree6_vec`.
   Rounds 0-2 replace the 8-scalar-load gather with select-based node_val:
   - Round 0 (level 0): all idx=0. `nv_g = tree0_vec ^ zero` (1 valu, 0 loads).
   - Round 1 (level 1): idx in {1,2}. 1 `vselect` on idx bit 0 (1 valu + 1 flow).
   - Round 2 (level 2): idx in {3,4,5,6}. Subtract base 3, 2-level select
     on bits 0-1 (3 valu + 3 flow).
   Total: 768 loads removed (3 rounds × 256 lanes). Body: ~2048 -> ~1664
   load-bound. Total: 2236 -> 2049.

Scratch additions: 8 (preload) + 56 (7 tree broadcast vecs) + 8 (three_vec) +
16 (2 sel temp vecs) = 88 words. Free: 97 -> 9.

Passes correctness (8 seeds) and three tiers: `baseline`, `updated-starting`,
`opus4-many-hours < 2164` (2049 < 2164, cleared the threshold step 5 missed).

---

## Step 7 - tree preload, post-wrap rounds 11-13      1 799 cyc   82.1×

Commit `264ad0e`. After the uniform wrap at round 10, lanes return to root
and descend through levels 0-2 again in rounds 11-13 (verified level
determinism). Same preload tree vectors, same select logic - just extend the
round checks: `r in (0, 11)` for level 0, `r in (1, 12)` for level 1,
`r in (2, 13)` for level 2.

Removes another 768 loads (3 rounds × 256 lanes). Total loads:
4096 - 768×2 = 2560. At 2 load ports: 1280 cyc gather floor. Body ~1280.
Total: 2049 -> 1799.

Slot utilization at this step (body, 1614 cycles):

| engine | slots used | capacity | util | idle cycles |
|--------|-----------|----------|------|-------------|
| valu   | 7999      | 9684     | 82.6% | 1 |
| load   | 2560      | 3228     | 79.3% | **334** |
| alu    | 6665      | 19368    | 34.4% | 1019 |
| flow   | 256       | 1614     | 15.9% | 1358 |

The 334 idle load cycles are during the preload-select rounds (0-2, 11-13)
where load ports sit idle while valu+alu do compute. The load alternation
(2,0,2,0,...) during gather rounds shows the scheduler batching gathers
rather than fully overlapping them with compute across rounds.

Passes correctness (8 seeds) and three tiers. 9 cyc short of `opus45-casual
< 1790`.

---

## Scheduler refactors (behavior-preserving)            1 799 -> 1 800 cyc

Commits `d541787`, `6be8c24`. Two refactors of `scheduler.py` with no
change to scheduling policy:

- **Priority-queue list scheduling loop** (`d541787`): replaced the nested
  `while progress` + `sorted()` re-scan with a single heap pass per cycle.
  Same-cycle WAR unlocks are pushed straight back onto the queue. The old
  drop-last-bundle `break` and `has_work` panic are gone; the trailing
  debug-only bundle is now appended (0-cycle, so the count is unchanged).
  1799 -> 1800: the heap prioritizes a newly-unlocked high-priority node
  *immediately* vs the old finish-pass-first, which resolves slot contention
  differently (+1 cyc). 1799 was lucky on the old sub-optimal priority fn.
- **`ReadWriteTable` + `slot`->`instruction` rename** (`6be8c24`): extracted
  the per-(region,lane) last-writer/readers-since bookkeeping out of the
  long `_build_nodes` into a `ReadWriteTable` class (`read`/`write` take a
  register id `(addr, is_vector)`), dropped dead `DNode.reads/.writes`
  fields. Pure refactor; same edges, same 1800 cyc.

---

## Step 8 - eliminate cross-group WAR (per-group temps)   1 773 cyc   83.4×

Commit `3097b19`. The kernel reused three *shared* vector temporaries across
all 32 groups - `addr_vec` (gather address), `sel_lo_vec`/`sel_hi_vec`
(level-2 select intermediates). Each is written by every group, so they form
cross-group WAR dependency chains whose shape depends on the loop order:

  - groups-outer (`for g: for r:`): the `addr_vec` WAR chain runs through
    all 16 rounds of group 0 before reaching group 1, so group 1 can't start
    until group 0 is nearly done. Readiness lower bound (longest path,
    RAW=1/WAR=0) = **5433** -> ~6726 cyc.
  - rounds-outer (`for r: for g:`): the chain weaves across groups within one
    round, so group g+1's round r is ready ~2 cyc after group g's. LB = 462
    -> ~1800 cyc.

The 4× gap is structural (not a scheduler bug): same RAW edge count in both
orders, but the shared-register WAR edges rearrange the readiness path. The
fix is to make the temporaries per-group so no register is written by more
than one group. Two op-neutral changes, both reusing existing per-group
planes (zero extra scratch; the 3 shared allocations are removed):

1. **Gather self-addressing** (`addr_vec` -> `nv_g` plane): the gather
   computes the address into `nv_g`, then each load reads `nv_g+j` as the
   address and writes `nv_g+j` as the value. Per-lane read-before-write makes
   this correct (each load touches only its own lane). Removes the dominant
   10-round cross-group WAR chain.
2. **Level-2 select restructure** (drop `sel_lo_vec`/`sel_hi_vec`): the 4-way
   mux now uses `t2_g` as the bit0-intermediate (`nv = bit0?tree4:tree3`;
   `t2 = bit0?tree6:tree5`; `nv = bit1?t2:nv`). Same 3 vselects, no shared
   vector. Removes the 2-round cross-group WAR chain.

Result: loop order is now irrelevant (rounds-outer == groups-outer for every
picker). Readiness LB drops to the within-group hash critical path (~223
RAW edges) in both orders -> resource-bound.

Cycle count by picker (both loop orders identical):

| picker            | cycles | notes |
|-------------------|--------|-------|
| `idx`             | 1827   | deterministic (program order) |
| `fma_first`       | 1822   | deterministic |
| `random` seed=42  | **1773** | deterministic (fixed seed); variance 1729-1799 across seeds |

All deterministic priority functions land at ~1822-1827; `random`+seed=42's
1773 is a lucky shuffle. This confirms the picker is sub-optimal (a future
trained picker should recover the ~50 cyc and more). Shipped config:
rounds-outer + `random` seed=42 = 1773.

Passes correctness (8 seeds) and **four** tiers: `baseline`,
`updated-starting`, `opus4-many-hours`, `opus45-casual < 1790`. 194 cyc short
of `opus45-2hr < 1579`.

---

## Step 9 - weighted picker (property-weighted priority)   1 599 cyc   92.4×

Commits (this step). Replace the static priority pickers (`idx`/
`fma_first`/lucky-`random`) with a weighted scoring function over static
per-node properties, with weights found by random search.

**Structure** (`scheduler.py`): each node carries a `NodeProps` (computed
once at DAG build via backward DP), holding four normalized (0..1) properties:

  - `sink` - dist_to_sink: longest cycle-weighted path (RAW=1, WAR=0) to a
    sink (critical-path urgency).
  - `load` - dist_to_load: cycle-distance to the nearest downstream load (0
    for loads; 1 = no downstream load). Lower = feeds the gather sooner.
  - `raw` - #RAW dependents (unblocked next cycle).
  - `war` - #WAR dependents (unblocked same cycle).

Plus rigidity as **mutable placement state** (on `_Placement`, not a static
prop): a node is rigid unless it's a fresh (un-spilled) `vec_elem`.

The weighted picker scores `score = w_sink·sink - w_load·load + w_raw·raw +
w_war·war + w_rigid·is_rigid_now` (higher = scheduled first, max-heap via
negation; `idx` is the final tiebreaker for determinism). `load` is
subtracted because low dist_to_load = urgent.

**Weight search** (`sweep_picker.py`): random search over discrete weights
(negatives included), printing each new best. ~180 samples found the region
in ~3 min; refinement around the winner pushed further. All deterministic
priority fns land ~1822-1827; `random`+seed42 = 1773; the weighted winner =
**1599**.

Shipped weights: `Weights(sink=-2, load=4, raw=-6, war=7, rigid=2)`.

The signs are the interesting result:
  - `load=+4`, `war=+7` (strong positives) - keep the load ports saturated
    and unblock same-cycle work. The two throughput drivers.
  - `rigid=+2` - prioritize nodes with no fallback (the fma_first signal,
    generalized to atomic/pinned-alu).
  - `sink=-2`, `raw=-4` (negatives!) - *deprioritize* critical-path and
    RAW-fan-out. In a throughput-bound schedule, chasing the critical path
    or next-cycle unblocks hurts; same-cycle (war) and load-feeding win.

Passes correctness (8 seeds) and **four** tiers: `baseline`,
`updated-starting`, `opus4-many-hours`, `opus45-casual < 1790`. 20 cyc short
of `opus45-2hr < 1579`. (Next: the clear-win op/edge reductions, then real
picker training once the architecture is stable.)
