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

---

## Step 10 - overlap epilogue vstores with the body tail   1 546 cyc   95.7×

Commit (this step). The epilogue vstored `val[256]` back to mem as a linear
32-vstore + 31-increment chain *after* the body - 64 cyc fully serial, while
the store engine sat idle the entire body (~1400 cyc with 0 store slots used).

Move the vstores into the scheduled body so they overlap the tail. Each group's
vstore fires once its round-15 `val` is ready, filling idle store slots
(2/cyc). The addressing avoids both the old single-`addr_a` chain (which
serialized vstores 1/cyc in group order) and a 32-step running-add chain:

each output address is `inp_values_p + 8g` (runtime base `inp_values_p` from
the header + compile-time offset `8g`). The 32 offsets are `load const`-ed
independently (no chain) and `inp_values_p` added to each (32 independent alus),
all early in the body. 32 per-group `out_addr` scalars (32 words) hold them.

Result: the 64-cyc serial epilogue is gone; the vstores hide behind the body
tail. 1598 -> 1546 cyc (-52). Store engine now saturated during the tail.

Passes correctness (8 seeds) and **six** tiers: `baseline`, `updated-starting`,
`opus4-many-hours`, `opus45-casual`, `opus45-2hr < 1579`, `sonnet45 < 1548`.
59 cyc short of `opus45-11hr < 1487`.

---

## Step 11 - store tree address, not index (structural)   1 559 cyc   94.8×

Commit (this step). Store `addr = idx + forest_p` per lane (replaces the idx
plane) so the gather reads the address directly - eliminating the per-round
`idx + forest_p` address-add (and its 1-cycle RAW before the loads).

The next-idx logic becomes next-address: `next_addr = 2·addr + (1−forest_p) +
parity`, using a broadcast `neg_fp1 = 1 − forest_p` vec (3 ops, same as
next-idx). Round 0 (idx=0 initial) computes `next_addr = (2−neg_fp1)+parity`
without reading addr (no addr-plane init). Wrap sets `addr = 1−neg_fp1` (=
forest_p). Select rounds (1/2/12/13) recover `idx = addr − forest_p_vec` (the
part to optimize away next, e.g. via the round-0 parity-as-select idea).

Structural effect (the point of the change):

| metric            | idx-scheme | addr-scheme | Δ |
|-------------------|-----------|-------------|------|
| nodes             | 16 864    | 16 160      | −704 |
| valu nodes        | 8 832     | 8 640       | −192 (gather adds) |
| critical path (LB)| 223       | 216         | −7 |
| WAR edges         | 14 208    | 20 416      | +6 208 |

Temporary cycle regression: 1546 -> 1559 (+13). The gather-add was load-bound
(hid behind the 8 loads, so removing it saves a slot not a cycle); the
select-round idx recovery is compute-bound (costs cycles); and WAR edges
jumped (addr read+written every round). Net worse *today*, but this is the
structural base for follow-ups: the 704 fewer nodes and 7-shorter critical
path are where the next optimizations (parity-as-select, better scheduling,
trained picker) land.

Best weights re-tuned for the addr-scheme DAG: `Weights(sink=-1, load=1,
raw=-2, war=-2, rigid=-1)` (the idx-scheme weights were a poor fit - WAR
edges dominate differently now).

Passes correctness (8 seeds) and **five** tiers: `baseline`, `updated-starting`,
`opus4-many-hours`, `opus45-casual`, `opus45-2hr < 1579`. 11 cyc short of
`sonnet45 < 1548` (a tier the idx-scheme cleared - temporary).

---

## Step 12 - parity-carry + addr-compare selects (on addr-scheme)   1 535 cyc   96.2×

Commit (this step). Eliminate the select-round `idx = addr − forest_p` recovery
(the compute-bound part that made step 11 regress) with two ideas:

1. **Parry carried in `t1_g`**: the next-addr writes `parity = val & 1` into
   `t1_g` as its last step. At the next round's start `t1_g` still holds it -
   and it *is* the level-1 select bit (`idx = 1 + parity`). So round 1/12
   collapses to one `vselect` (1 flow op, no idx recovery / `& 1`). It also
   feeds round 2/13's bit-0: `bit0(idx−3) = bit0(2·(idx₁−1) + parity₁) =
   parity₁` = the carried `t1_g`.
2. **`addr < pos_fp5`** for the level-2 high bit: `{3,4}` vs `{5,6}` is just
   `addr < forest_p + 5` (broadcast `pos_fp5 = 5 + forest_p`). One `<` valu
   replaces idx-recovery + shift + and.

Also: `neg_fp1` now a `valu` sub (1 op, no scalar temp/broadcast); new
`pos_fp5_vec` (5 + forest_p, 2 prologue valu).

### DAG quality (structure - honest; cycles depend on the untrained picker)

| metric              | idx (step 10) | addr (step 11) | addr+parity (step 12) |
|---------------------|--------------:|---------------:|----------------------:|
| nodes               | 16 864        | 16 160         | **15 776**            |
| height (crit path)  | 223           | 216            | **204**               |
| RAW edges           | 23 840        | 23 104         | **22 720**            |
| WAR edges           | 14 208        | 20 416         | 20 352                |
| valu nodes          | 8 832         | 8 640          | **8 256**             |
| cycles (current wts)| 1 546         | 1 559          | **1 535**             |

Step 12 alone: −384 nodes, −12 height, −384 valu, −64 WAR. The addr direction
now beats the idx-scheme on every structural metric *and* on cycles. WAR edges
remain high (the addr plane is read+written every round) - the lever for the
trained picker.

Passes correctness (8 seeds) and **six** tiers: `baseline`, `updated-starting`,
`opus4-many-hours`, `opus45-casual`, `opus45-2hr < 1579`, `sonnet45 < 1548`.
48 cyc short of `opus45-11hr < 1487`.

*(Going forward, log entries include DAG-quality metrics - structure is honest
where the untrained picker's cycle count is not.)*

---

## Step 13 - trained picker weights (coordinate descent)   1 522 cyc   97.1×

Commit (this step). The hand-tuned weights (step 12) were found by ~180 random
samples. Train them properly via coordinate descent with random restarts
(`train_picker.py`): line-search each weight over a discrete grid, cycle to
convergence, restart from random points, then refine around the global best.

The loss is `body_cycles(schedule(dag, w))` - deterministic and data-independent
(the DAG is structural; all submission seeds give the same count for fixed
weights), so no train/val split. Landscape is piecewise-constant (gradients
useless) and scale-invariant (optimum is a direction) - sampling-based
coordinate descent is the right tool. ~588 evals in ~500s.

Winner: `Weights(sink=-3.5, load=-0.25, raw=0.5, war=2, rigid=4)` = **1522 cyc**
(-13 vs the hand-tuned 1535). The signs differ from the hand-tuned guess:
`rigid=4` and `war=2` (strong positives) now dominate; `load≈0` (the gather is
already saturated, so load-urgency barely matters); `sink=-3.5` (deprioritize
critical-path - throughput-bound).

### DAG quality (unchanged from step 12 - this step is picker-only)

| metric              | step 12 (= step 13) |
|---------------------|--------------------:|
| nodes               | 15 776              |
| height (crit path)  | 204                 |
| RAW edges           | 22 720              |
| WAR edges           | 20 352              |
| valu nodes          | 8 256               |
| cycles              | 1 535 -> **1 522**  |

(DAG unchanged; only the picker weights changed, so structure is identical -
the -13 is pure scheduling improvement.)

Passes correctness (8 seeds) and **six** tiers: `baseline`, `updated-starting`,
`opus4-many-hours`, `opus45-casual`, `opus45-2hr < 1579`, `sonnet45 < 1548`.
35 cyc short of `opus45-11hr < 1487`.

---

## Step 14 - add idx property + longer training   1 459 cyc   101.4×

The 5-property picker (sink/load/raw/war/rigid) plateaued at 1514 cyc.
Adding program-order `idx` (normalized 0..1) as a 6th weighted property broke
through: the trained picker found `idx=-4` (strongly negative = reverse program
order) unlocks a much better schedule. The idx weight lets the picker decide
how much locality/ordering matters (was only a tiebreaker before).

Training: coordinate descent with random restarts + fine refinement
(`train_picker.py`, checkpointed to `_best_weights.txt` for chained runs).
Also tried scipy differential_evolution and broad random search - all converge
on the same optimum. ~2500+ evals total across runs.

Winner: `Weights(sink=-3, load=-1.5, raw=-0.25, war=6, rigid=0.25, idx=-4)` =
**1459 cyc** (-55 vs step 13's 1514, -76 vs step 12's 1535). The `idx=-4`
(reverse program order) is the dominant new signal; `war=6` (same-cycle
unblock) and `sink=-3` (deprioritize critical path) remain strong.

### DAG quality (unchanged from step 12 - picker-only steps 13-14)

| metric              | step 12 (= 13 = 14) |
|---------------------|--------------------:|
| nodes               | 15 776              |
| height (crit path)  | 204                 |
| RAW edges           | 22 720              |
| WAR edges           | 20 352              |
| valu nodes          | 8 256               |
| cycles              | 1 535 -> 1 522 -> **1 459** |

(DAG unchanged across steps 13-14; the -76 is pure scheduling improvement from
better picker weights, first via coordinate descent on 5 properties, then
adding idx as a 6th.)

Passes correctness (8 seeds) and **seven** tiers: `baseline`, `updated-starting`,
`opus4-many-hours`, `opus45-casual`, `opus45-2hr < 1579`, `sonnet45 < 1548`,
`opus45-11hr < 1487`. 96 cyc short of `opus45-improved-harness < 1363`.

---

## Step 15 - prune-to-stores dead-code pass   1 458 cyc   101.3×

Commit (this step). Prune DAG nodes that do not contribute to the final
stores, in `scheduler.py` (`prune_to_stores`), wired into
`KernelBuilder.build` ahead of `schedule()`.

- **Pass 1**: backward walk from the 32 store sinks following RAW
  (weight-1, true data dependency) edges only, marking nodes useful. WAR
  edges are anti-dependencies (register-reuse ordering), not data flow -
  neither walked nor marking.
- **Pass 2**: drop every unmarked node and its attached edges. Debug nodes
  inherit the usefulness of their producers: kept iff all their RAW
  producers are kept (vacuously true with none - the 32 round-0 "val before
  xor" vcompares read the prologue-vloaded val, outside the body DAG).
- **No dependency re-analysis**: kept-kept edges are exactly the induced
  subgraph; only edges incident to removed nodes disappear. Compaction
  (filter + re-index) suffices, no ReadWriteTable re-walk. WAW safety is
  preserved: a kept writer is useful -> has a kept reader on a RAW path to
  a store -> that reader precedes the next kept writer of the lane, so the
  W1 ->RAW R' ->WAR W2 bridge survives. (A writer with no kept reader
  before the next writer is not RAW-reachable from a store and is pruned.)
- Counters/frontier/props re-derived from the filtered edge lists via the
  extracted `DAG._finish_init()` (shared with `__init__`). `dist_to_sink`
  is now anchored on the real sinks only - before, the zero-out-degree set
  included 96 debug nodes and 32 dead round-15 `+` writes.

Pruned exactly the expected **96 nodes**: the round-15 addr-update chain
(`&` parity -> `multiply_add` -> `+`, x 32 groups; round 16 never reads
addr). Only -1 cycle (1459 -> 1458): the dead chain sat in the store-bound
tail where its valu ops hid in otherwise-idle slots. The kernel is
otherwise RAW-tight - everything else reaches a store (11 072 useful + 4
608 inherited debug of 15 776 nodes). The pass's value is structural:
clean sink anchoring for the picker props, and a safety net for future
restructuring.

Picker weights unchanged (no retraining; compute is limited).

### DAG quality

| metric              | step 12-14 | step 15 |
|---------------------|-----------:|--------:|
| nodes               | 15 776     | 15 680  |
| cycles              | 1 459      | **1 458** |

Passes correctness (8 seeds) and **seven** tiers (unchanged): `baseline`,
`updated-starting`, `opus4-many-hours`, `opus45-casual`, `opus45-2hr < 1579`,
`sonnet45 < 1548`, `opus45-11hr < 1487`. 95 cyc short of
`opus45-improved-harness < 1363`.

---

## Step 16 - two-phase register allocation (parallel path)   1 628 cyc

The one-pass FIFO rename engine (step a9f00d8+) had regressed the body to
1 765 via cross-group WAR chains: the auto-free FIFO recycled t1/t2/nv homes
across groups, and the DAG's no-WAW-edge design meant a dead write silently
broke the reader-bridged ordering chain. Replaced with a clean two-phase
architecture in `regalloc.py` (new module, FIFO path kept for comparison):

1. **`tag_raw_chains`** - forward SSA pass. Each write becomes a unique
   version tag (`sym#n`); reads map to the current tag; `read_count[tag]`
   records how many readers. Tags are unique by construction, so the DAG has
   **RAW edges only** - no false WAR/WAW by design.
2. **`build_dag`** - weight-1 RAW edges from each tag's unique writer.
   Prologue-written tags are pre-committed inputs (no edge).
3. **`RegisterAllocator` + `schedule`** - registers allocated at writer
   placement (sticky across partial nodes), freed when all reads commit. A
   write node is placeable only when a unit slot AND a free register are both
   available. Gather = one partial-completion node (8 scalar loads, sticky nv
   register). Dead writes are dropped (they'd have no reader to free).
4. **Pressure-aware priority** - a node that is the last reader of a tag frees
   its register on commit; under low free-pool, prefer such nodes so chains
   finish instead of piling up un-freed temps.

Key correctness fixes: resolve operands at allocation time (not emit time, when
read tags may already be freed); `unwrite` rollback only when `lanes_done==0`
(a partial node's dest register must not be freed mid-spill).

Result: **1 628 cyc** (all-temps, 0/256) vs FIFO 1 827. `b1d4bb7`, `78a7f0b`.

## Step 17 - always-on freeing bias + tuned weights   1 466 cyc   100.7×

- The register-freeing priority is now **always-on** (not threshold-gated):
  always prefer a node whose commit frees a register. A sweep
  (`sweep_regalloc.py`) showed always-free beats threshold-gating (1 455 vs
  1 482 body) and the bias weight barely matters. The bias is also *required*
  for the schedule to complete at all under register pressure (the pure
  weighted base deadlocks - 47 ready nodes, none placeable).
- Base picker weights tuned for the RAW-only DAG via random search + local
  refinement + DE: `REGALLOC_WEIGHTS = (sink=-1, load=5, raw=1, war=1,
  rigid=1, idx=-4)`, distinct from the FIFO path's `BODY_WEIGHTS`.

Result: **1 466 cyc** (body 1 273), all rounds correct. `26c708f`.

### The body is now LOAD-bound, not valu-bound (structural ceiling)

`analyze_slots.py --regalloc` on the body (1 273 cyc):

| engine | slots used | capacity | util | idle cycles |
|--------|-----------:|---------:|-----:|------------:|
| valu   | 6 168      | 7 638    | 80.8%| 192 |
| **load** | **2 484** | **2 546**| **97.6%** | **31** |
| alu    | 11 948     | 15 276   | 78.2%| 274 |
| flow   | 249        | 1 273    | 19.6%| 1 024 |

**Load is saturated at 97.6%.** 2 484 load slots / 2-per-cycle = a **1 242-cycle
load floor**; the body is 1 273, only 31 cycles of slack. The hogs are the
**320 Gathers** (tree-node value lookups), each decomposed to 8 scalar loads =
2 560 load slots. valu has headroom (80.8%) but cannot help - the load unit is
the wall. Further body gains require *reducing load slots* (fewer/cheaper
gathers), not better scheduling. The earlier "valu resource bound ~1 088" was
computed before counting the 8x gather decomposition and is not the binding
constraint; the real floor is the load floor ~1 242.

Passes correctness; regalloc path beats the one-pass FIFO rename (1 466 vs
1 827) and is within 8 cycles of the pre-rename scratch-register scheme
(step 15, 1 458).

---

## Step 18 - rollout scheduler (per-cycle trial-and-score)   1 413 cyc   104.6×

Commits `9631e92` (infra), (this step). The level-3 preload-select kernel
(`36a1a23`) deadlocked the greedy regalloc scheduler at SCRATCH_SIZE=1536:
too many in-flight hash chains -> free_vec=0 with ~170 live tags at one
read remaining, their last-reader nodes mechanically unplaceable
(allocation precedes freeing). Replaced the single global priority order
with a per-cycle search (rollout.py):

- **Trial-and-score loop**: each cycle, K candidate orderings of the
  (fixed, RAW-only) ready set are simulated as placement *decisions* (no
  operand resolution/emission - the pool is per-cycle scratch), scored on
  the post-`advance()` state, rolled back, and the winner replayed with
  emission on. Trial 0 = the incumbent weighted-greedy order (load-bearing:
  pure shuffles deadlock at C=65); trials 1..K-1 = uniform shuffles.
- **Uniform structure-owned rollback**: DAG / RegisterAllocator /
  _Placement each implement `checkpoint() -> token` / `rollback(token)`
  with internal undo logs; rollout.py orchestrates. Greedy path untouched
  (logs None = zero overhead).
- **The metric that matters**: `reads` - read obligations *consumed* this
  cycle (weight +2), plus reg_delta (-1) as pressure tiebreaker. Rewarding
  obligation consumption meters chain openings against closings. Failed
  alternatives (all deadlock): reg_delta alone (any weight -1..-16),
  frontier, alu_work, obligations=-w (starves chain starts, deadlocks
  C=99). Key equivalence: post-cycle pool level == start - reg_delta, so
  level and delta give identical trial rankings - one of them only.
- K=6 (K=4 deadlocks; K=10 no better). ~5s per scheduling pass.

Result: **1 413 cyc at 1536 scratch, correct on all rounds** - the
level-3 kernel (deadlocked under greedy) now schedules, beating the
previous 1536-feasible best (step 17, 1 466) and landing 50 cyc short of
the big-scratch validation of the same dataflow (1 389).

Passes correctness (8 seeds) and **eight** tiers: everything except
`opus45-improved-harness < 1363` (50 cyc short).

---

## Step 19 - DAG sink groups + group priority   1 289 cyc   114.6×   ALL TIERS

Commits (this step). Two additions, proposed and designed in discussion:

1. **DAG-derived sink groups** (`NodeProps.group`): each non-debug sink (the
   32 stores) anchors a group; sinks sorted by cycle-depth get ids 1..N;
   every node inherits the id of its DEEPEST descendant sink (ties -> lower
   id); debug nodes stay ungrouped (0). On this DAG the partition is exact:
   32 groups x 296 nodes - the DAG groups align perfectly with the
   lane-processing groups (by construction: per-group SSA tags, shared
   consts live in the prologue).
2. **Progress in ctx** (`SortCtx.progress = committed/total`) +
   `make_interp_greedy(w1, w2)`: progress-interpolated weighted priority
   `progress*(w1.props) + (1-progress)*(w2.props)` - prop importance may
   change over the schedule's lifetime.

The win came from the group dial, not interpolation: prioritising earlier
groups (negative group weight) serialises chain openings - lower pressure
AND better throughput. Sweep (K=6, score reads=2/rd=-1):

| config | cycles |
|--------|-------:|
| group=-1 | deadlock C=600 |
| group=-2 | 1 347 |
| **group=-4** | **1 289** |
| group=-6/-8 | 1 304 / 1 298 |
| group=-12/-20 | 1 347 / 1 348 |
| interp grp-4 early -> 0 late | 1 342 |

Shipped: `REGALLOC_WEIGHTS = Weights(sink=-1, load=5, raw=1, war=1,
rigid=1, idx=-4, group=-4)` = **1 289 cyc** (K=3 ties at half the
wall-clock). Interpolation infrastructure kept (make_interp_greedy) but no
interp config shipped - flat group=-4 beat every interp variant tried.

Also this step (interface alignment): sort funcs are `f(ready_nodes, ctx)
-> permutation`, K = len(sort_funcs); ctx carries the live allocator +
progress; weights/rng bind via factories. Node metadata travels on the
nodes (DNode.props attached by DAG._finish_init, DNode.placement by
_classify) - the dag.props / placements side lists are gone.

Passes correctness (8 seeds) and **all nine** tiers, including
`opus45-improved-harness < 1363` (1 289, 74 cyc clear).

---

## Step 20 - prologue merged into the body DAG   1 203 cyc   122.8×

Commit `3c3ea98`. The prologue (const loads, vbroadcasts, header loads,
val/tree vloads) was still emitted linearly - one slot per bundle, 133
cycles at ~5% utilization - because `_emit_regalloc` pre-committed its
registers before building the body DAG. Merged both into ONE DAG:

- `tag_raw_chains(prologue + body)` -> single `build_dag` -> single
  schedule. Setup work now interleaves with early body compute through
  the scheduler (the pattern was proven in step 10 when out_addr consts
  moved into the body). Prologue nodes reach all 32 stores, so the group
  prop assigns them group ~1 - the group=-4 priority naturally schedules
  them first.
- **Pauses ride existing bundles** (`_insert_pauses`): the two dev-oracle
  barriers (pause 1 = initial-mem barrier before any store, pause 2 =
  final-mem barrier) are injected into existing bundles' flow slots
  instead of standalone bundles. At grading (`enable_pause=False`) they
  now cost 0 cycles (were 2 standalone bundles).
- K=1 shipped (greedy priority only, no search trials): with group=-4
  the priority fn alone lands within 2 cyc of K=6 (1291 vs 1289) at 5x
  the scheduling speed. K will be raised again after other parts settle.

Result: 1 289 -> **1 203 cyc** (-86). Merged-schedule utilization:

| engine | slots | cap | util |
|--------|------:|----:|-----:|
| valu   | 6 870 | 7 218 | **95.2%** (the wall: floor 1 145) |
| load   | 2 133 | 2 406 | 88.7% |
| alu    | 12 706 | 14 436 | 88.0% |
| flow   | 706   | 1 203 | 58.7% |

First 60 cycles (former-prologue ramp-up): valu 61.9%, load 65.8%,
alu 56.9%, flow 11.7%. Remaining levers: valu op-count reduction (the
floor), prologue op tricks (const loads ride the 88.7% load engine;
alu-computed consts trade load->alu, slightly freer), the 58 cyc of
valu slack.

Passes correctness (8 seeds) and **all nine** tiers (1 203 < 1 363,
160 cyc clear).

---

## Step 21 - computed small const vectors   1 190 cyc   124.1×

Commit `0aab7e9`. The 8 small const vectors no longer cost a `const` load
+ `vbroadcast` each - they are computed from each other on valu:

  1 = (0==0)   2 = 1+1   3 = 1+2   9 = 3*3   16 = 2<<3   19 = 16+3
  33 = 16*2+1 (fma)   4097 = 64*64+1 (fma; 64 = 16<<2)

Trades 8 load slots for ~1 net valu op and cuts 16 prologue nodes to 10;
the load engine was the busier one in the ramp-up window. Only the 6 big
hash addends (K0..K5) remain const+vbroadcast. 1 203 -> **1 190** (-13).

Passes correctness (8 seeds) and **all nine** tiers (173 cyc clear).

---

## Step 22 - path-bit retention (recompute off)   1 179 cyc   125.2×

Commit `7a9ae1c`. `RECOMPUTE_PATH_BITS=False`: the level-3 selects (rounds
3/14) consume the path bits RETAINED from rounds 0-2 instead of
recomputing them from addr (the 5-op offset/shift/and chain per group per
level-3 round). This version deadlocked the greedy scheduler at 1536 (the
+64 granules of round 0-2 retention pressure) - the rollout scheduler's
group priority meters it without effort. 1 190 -> **1 179** (-11).

Passes correctness (8 seeds) and **all nine** tiers (184 cyc clear).

---

## Step 23 - valu->alu swap for rigid fma   1 174 cyc   125.8×

Commit `292e6bb`. Placement-engine swap (proposed in discussion): when a
vec_fma finds the valu unit full, the pool evicts one valu-placed vec_elem
(LIFO) down to the alu (1 valu slot <-> 8 alu lane slots), if the alu can
take ALL VLEN lanes. Key insight (user's): this is a pure slot-MAPPING
change - the evicted node stays complete in the same cycle (same pre-cycle
reads, same end-of-cycle writes whichever engine realises it), so no
commit/register state is touched and the trials/rollback machinery is
unaffected. Implemented inside `FuncUnitPool.place` (records valu-placed
elems per cycle for LIFO eviction); both scheduler paths benefit, zero
caller changes. 478 fma stalls were measured at step 22's schedule.

1 179 -> **1 174** (-5). 19 unit tests (3 new pool-swap tests).

Passes correctness (8 seeds) and **all nine** tiers (189 cyc clear).

---

## Step 24 - weight training: trimmed space + sign-free search   1 119 cyc   132.1×

Commit `4ae89db`. Three changes:

1. **Trim** (`war`, `idx` removed): war was provably inert (RAW-only DAG);
   idx (program order) is not DAG structure - "the DAG is the program".
   Evidence backed the user's call: my sign-biased single-coordinate
   sweep could not compensate for idx (everything deadlocked C=75-169),
   but a sign-free JOINT random search found idx-free configs beating the
   old best within 200 samples - idx was proxying for weight regions the
   biased sweep could not reach, not for real information. `freeing`
   became a regular weight (was hardcoded 1000; winners use 0.7-9.3).
2. **`train_weights.py`**: budget-driven, file-checkpointed phases -
   `random` (sign-free joint search) and `finetune` (breadth-first
   coordinate descent), each restartable from the other's dump. Modes:
   `single` (6-dim) and `interp` (12-dim, w_early ++ w_late for
   make_interp_greedy, random-seeded per the user's reasoning that
   pre-optimized singles are biased toward no-interp optima).
3. **Results**: single-mode **1 119** (shipped; signs demolish the old
   hand-tuning: sink +6.8, load +7.1, raw +8.0, rigid +7.7, group -3.6,
   freeing +5.3). interp-mode 1 126 after one random+finetune round
   (close, not yet better - candidates were still improving at budget).

1 174 -> **1 119** (-55). Passes correctness (8 seeds) and **all nine**
tiers (244 cyc clear).

---

## Step 25 - dynamic scalar/vector allocation   1 117 cyc   132.2×

Commit `3d6876c`. The static 48-word scalar reserve is gone: ONE vector
free list covers all of scratch as 8-aligned granules; the scalar engine
claims a granule and splits it into 8 free words on demand, and fully-free
granules are reclaimed at CYCLE END (`collect_scalar_garbage`, called by
both schedulers after commit - outside the trial/checkpoint window, so
reclaim needs NO undo support; the entire "Rc" undo family was avoided by
deferral, per the user's simplification). Undo stays self-contained:
claim reversal only when all sibling words are still free
(`_granule_free == VLEN-1`), else plain word return.

Profile findings: scalar usage is flat ~30-32 words everywhere (peaks
don't coincide with the vec peak, but ~5 granules are always pinned);
allocation already consolidates to the 5-granule minimum for 33 words (no
pinning pathology); effective pressure peak 190/191 granules. Net gain
over the static reserve: +1 vec granule (peak 186 vs 185 saturated).

Shipped with re-tuned weights (freeing 5.3 -> 4.3, retuned under the new
allocator): 1 119 -> **1 117** (-2). Passes correctness (8 seeds) and
**all nine** tiers (246 cyc clear).

---

## Exploration - valu op-count reduction: tested and parked   (1 117)

*(Superseded by step 26: the op cut is now SHIPPED at 1 110 under interp
weights found by the v2 search. This entry records why it was parked
under the v1 search.)*

Census of the 6 521 valu slots: hash (1 954 fma + irreducible xor/shift),
entry XOR, 3-op addr update - all at proven minimums (12-slot hash, xor
non-affine so no fma-fold, `&1` doubles as path bit). Dead ends: wrap
write via flow add_imm (immediate needs runtime forest_p).

The one real candidate: level-0 `nv = tree0 ^ 0` is 64 pure-copy valu
ops; the entry XOR can read tree0_vec directly. Floor -~ ~1 077, BUT
after full re-training (random + finetune) the schedule landed at
**1 127 - worse than 1 117**. The copies act as a *frontier pacemaker*:
their commits stagger entry-XOR readiness, and without them the ramp-up
floods (193 ready nodes, deadlock at C=152 with the old weights). Verdict:
scheduler deficiency, not a flaw in the op cut - per discussion, interp /
K=N with a properly trained score function may realize the ~10-cyc cut.

Preserved as the `LEVEL0_DIRECT_TREE0` toggle (default OFF); the
re-trained candidates live in `_weights_found_l0.json` /
`_weights_refined_l0.json` (best 1 127: sink 7.9, load 5.2, raw -9.4,
rigid 7.7, group 3.6, freeing 1.1).

Dev tooling: `sweep_rollout.py` (weight/K sweep), `diag_deadlock.py`
(live-tag autopsy at the wall), `diag_liveness.py` (committed-liveness
profile; peak 220 at big scratch = scheduler artifact, not a capacity
proof). dev_tests/ (14 unit tests): randomized rollback equivalence for
all three structures, golden greedy-equivalence (trials=1 reproduces
regalloc.schedule byte-for-byte), determinism, allocator hygiene,
deadlock detection.

---

## Step 26 - L0-direct shipped: interp weights via trimmed/base+tilt search   1 110 cyc   133.1×

Commits (this step). The parked LEVEL0_DIRECT_TREE0 op cut (64 pure-copy
valu ops deleted; entry XOR reads tree0_vec directly; compute floor
65090 -> 64578 lane-slots = 1085 -> 1077) is now SHIPPED, beating the
L0-off 1117 by 7. Three parts:

1. **Parallel trainer** (`train_weights.py` v2 infra): candidate evals
   farmed to a multiprocessing pool (default min(28, cpu-4) workers;
   ~25x throughput, ~15 evals/s). random = embarrassingly parallel;
   finetune = steepest-neighbor descent (whole dims x DELTAS neighborhood
   in one parallel batch, best applied) replacing sequential Gauss-Seidel.

2. **Failure mechanism of L0-on, measured** (`diag_underfill.py`, new dev
   tool - classifies valu-underfilled cycles as register-pressure vs
   work-starvation): the 1123 interp schedule's 46-cycle slack is
   starvation-dominated (61 cycles, 102 slots), NOT pressure (17, 21).
   The starvation clusters at C~750-900 = the round-15 gather wall: a
   gather feeds at 4 cyc/group (8 loads / 2 ports) but its hash drains at
   2.3 cyc/group (14 valu / 6), so synchronized group arrivals force valu
   starvation arithmetically. L0-on's efficient early schedule removes
   the copy-op metronome that kept groups phase-spread. (User's doubt
   confirmed: no scheduler permutation recovers a feed-rate wall felt
   100+ cycles after the decisions that caused it.)

3. **Search-space v2** (from trend analysis of 4050+ trained completes):
   - Signature of winners: sink+, load+, freeing+ (98-100%), group mildly
     +, raw DEAD (corr +/-0.04). Note (sink+,group+) IS a diagonal
     wavefront (late groups lead by slope w_group/w_sink) - phase
     spreading was already in the searched space; serializing (sink-)
     regions capped at 1149.
   - Changes: raw dropped (fixed 0); sign-biased per-prop bounds
     (sink/freeing [0,10], load [-3,10], group [-4,10], rigid [-3,6]);
     interp reparameterized as base+tilt (w_late = base+tilt/2, w_early =
     base-tilt/2) - the signal lives in base (corr -0.33/-0.44), tilt is
     weak; two-stage finetune deltas (coarse then +/-0.25 polish).
   - Completion rate 15% -> 50%; 1127-equivalent found in 29s (was 536s).

The new basin (found at #3998): EARLY serializes early groups
(group -4.8) + pushes far-from-sink work; LATE drops group entirely;
freeing strong (~7-8) throughout. Random 1114, finetune 1111, fine polish
1110 (freeing_base 7.4 -> 7.65).

Shipped: `LEVEL0_DIRECT_TREE0=True`, `ROLLOUT_SORT_FUNCS =
[make_interp_greedy(INTERP_W_LATE, INTERP_W_EARLY)]` (decoded in
perf_takehome.py). 1110 = floor 1077 + 33 slack. `REGALLOC_WEIGHTS` kept
for the greedy fallback path only.

Passes correctness (8 seeds) and **all nine** tiers (253 cyc clear).
Remaining slack vs floor: 33 cyc (ramp-up latency ~4, round-15 feed wall,
tail drain). Next levers unchanged in kind: K=N with a trained scorer
(still untrained), op-count (the dedup lever - fewer than 4096 hashes).

---

## Step 27 - borrowed op-cuts + weight retrain on the new DAG   1 100 cyc   134.3×

Commits `8201f33` (the cuts), `e918243` (prior-art doc), (this step's
retrain). Two op-count cuts borrowed from rubinownz111's 1063-cycle
solution (toggles + attribution in perf_takehome.py):

- **FUSE_HASH_STAGES_23**: stages 2+3 = (33a + (C2+C3)) ^ (16896a +
  (C2<<9)) = fma+fma+xor (3 ops, was 4). -512 vec ops.
- **WRAP_ROOT_C5_DEFER**: round-10 stage 5 drops ^C5 (-32 vec ops),
  repaired free at round 11's entry XOR via precomputed tree0^C5.

The combined DAG DEADLOCKED at 1536 under the step-26 weights (C=399,
free_vec=0). Weight retrain with the v2 trainer (freeing bound widened
to [0,16] - the step-26 winner sat at 8.4, near the old cap):

- random 1200s (9 713 samples, 4 944 completes): 1 110
- finetune 696s (coarse converged 580s, fine polish): **1 100**
- Winner (base/tilt): sink 7.0/2.5, load 4.9/2.3, rigid -1.0/2.9,
  group -3.6/0.6, freeing 2.3/4.5. Decoded: EARLY sink 5.75, load 3.75,
  rigid -2.45, group -3.9, freeing 0.05; LATE sink 8.25, load 6.05,
  rigid 0.45, group -3.3, freeing 4.55. Structure shift vs step 26:
  group serialization now THROUGHOUT (-3.9/-3.3, not early-only);
  freeing much lighter (0.05/4.55 vs 6.9/8.4).
- K-invariant: K=1/3/6 all 1 100 (the trained candidate wins every
  trial); shipped at K=1.

Oracle fix (same class as the existing skips): the round-11 node_val
DebugVCompare compared nv_op (= tree0^C5, the repair) against reference
tree0 - an intentional mismatch, like the already-skipped round-11 val
and wrap-round hashed_val compares. Added the same guard. Oracle
artifact, not a dataflow bug (final mem matched throughout).

Per-engine bounds now: load 1 063 (BINDING), valu 1 055, alu 823.
1 100 = load floor + 37. Follow-on: level-4 partial preloading (lever
0(3)) deletes the round-15 gather wall AND relieves the load floor.

Correctness: _check (1536) CORRECT, _check_big (4096) CORRECT,
submission tests (8 seeds) 1 100. All nine tiers (263 clear).
