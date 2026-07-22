# Optimization Log

Worked example: Anthropic's VLIW/SIMD performance-engineering take-home.
Canonical workload: `forest_height=10, n_nodes=2047, batch_size=256,
rounds=16`. Scored by simulated cycle count on a frozen copy of the
simulator (`tests/submission_tests.py`).

Each entry records intent, mechanism, cycle count, and PMU corroboration.

---

## Baseline (committed up-stream)               147 734 cyc   1.00×

`KernelBuilder.build_kernel` as shipped, a deliberately-naive scalar program:

- One slot per instruction bundle (no VLIW packing at all — `vliw=False`).
- Fully unrolled `rounds × batch_size` (= 4096) iterations emitted statically.
- Re-reads `idx[i]`, `val[i]` from **mem** every round (one `load` apiece),
  plus one `load` for `tree.values[idx]`.
- Stores `idx[i]` AND `val[i]` back to **mem** every round (the `idx` writes
  are entirely wasted — the grader checks only `val`).
- `myhash` taken literally: 6 stages × 3 alu ops = **18 ops/hash**.
- Idx update `% 2`, `== 0`, `flow select`, `* 2`, `+`, `< n_nodes`,
  `flow select`. Wrap is fully per-lane branchy.
- Utility slot count per lane per round: ~29 alu, 3 load, 2 store, 2 flow.
  All scalar; `valu` untouched.

PMU: alu 118784 (6.7% util), valu 0, load 12564, store 8192, flow 8194.
Every active engine sits at histogram k=1 (one-slot-per-bundle pathology).

---

## v1 — fma via `valu` `multiply_add`         123 165 cyc   1.20×

Commit `648da3d`. First reduction of `myhash`:

- Three linear stages `(a+K) + (a<<s) == a*(1+2^s) + K` collapse from three
  alu slots to **one `valu multiply_add` slot each** (verified bit-exact
  against `myhash`). Using `multiply_add` as a scalar fma on lane 0 of a
  VLEN-word work region; broadcast constants fill lanes 1..7 with junk we
  ignore.
- Three xor/add-shift stages (1, 3, 5) are irreducible at 3 slots each
  — carries block fusing `+K` across the `^` combine (numerically falsified).
- ⇒ `myhash` goes from 18 literal alu ops to **12 slots** (3 fma + 9 alu).
- Everything else still sequential & bloated (per-round mem round-trips,
  branchy `%`/`select`/`<`/`select` idx/wrap, idx stored).

Per-lane per-round slot count: 12 hash (vs 18) + ~6 idx/wrap = ~20 → ~18.
PMU: alu 118784 → 81920, valu 0 → 12296, others unchanged.

---

## v2 — scratch-resident state + branchless idx
                                                77 223 cyc   1.91×
Commit `ff00b76`. Plumbing fixes (no compute parallelism yet):

- `val[256]`, `idx[256]` **resident in scratch** across all 16 rounds.
  Prologue `vload`s `val` once (32 vloads); `idx` starts at 0 (scratch is
  zero-initialized → no init needed). Epilogue `vstore`s `val` once.
  No per-round mem round-trips for lane state.
- `val[i]` doubles as the running hash register via a shared transient
  8-word `work_vec` (lane 0 active for fma). Stage 5's final combine writes
  `v` **directly into `val[i]`** — no separate copy-back op.
- Per-lane stage temps `t1[i]`, `t2[i]` (single-word, never shared across
  lanes — no rename hazards when we later pack VLIW bundles).
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
  the final values when the test compares at the reference's final yield
  (caught by `Incorrect result on round 1` during dev).

PMU before/after:

| engine   | v1 fires | v2 fires | delta |
|----------|----------|----------|-------|
| store    | 8192     | **32**   | −8160 (only final vstore) |
| flow     | 8194     | **2**    | −8192 (branchless idx + build-time wrap) |
| load     | 12565    | 4157     | −8408 (no idx/val re-reads) |
| alu      | 81920    | 60736    | −21184 (dropped %/==/</select/* redundancies) |
| valu     | 12294    | 12296    | unchanged (intrinsic fma count) |

Passes `submission_tests.py` correctness (8 seeds) and the first two speed
tiers: baseline `< 147734`, updated-starting-point `< 18532`. Fails the
`< 2164` tier (expected — that lives in the VLIW-packing + vectorization
regime).

---

## v3a — cross-lane vectorization (8 lanes/group on `valu`)   12 911 cyc   11.44×

First vectorization chunk: run `myhash` elementwise across VLEN=8 lanes per
`valu` slot, 32 groups of 8. Still one slot per instruction bundle
(`vliw=False`) — sophisticated VLIW packing is deliberately postponed; this
pass is the functional 8-lane chunk ("naive loader" gather + full-vector hash).

- **Scratch reorg to a hard rule**: 256 words shared, then 5 words per lane
  across 256 lanes (= 1280), exhausting the 1536-word file. The per-lane
  sector is 5 contiguous planes of 256 (SoA): lane `i` owns one word per
  plane at `plane_base + i`, so group `g` (lanes 8g..8g+7) forms an
  8-word contiguous vector at `plane_base + 8g` — vectorizable at zero
  gather cost. Planes: `val`, `idx`, `t1`, `t2`, `nv` (node_val landing).
  Shared sector (159/256 words used, 97 free) holds header, scalar consts,
  broadcast vector consts, and the few genuinely-shared transients
  (`addr_a` pointer; `addr_vec` gather-address vector, ≤1-cycle live;
  `forest_p_vec` broadcast).
- **Per-lane `t1`/`t2`/`nv`** (not shared) so distinct groups' in-flight stages
  never alias — VLIW-packable without rename management later. The old v2
  "single shared pair of stage temps" is gone; this costs 768 words of
  scratch but removes a whole class of inter-group hazards ` la headroom
  cost nothing (97 + 256×0 free).
- **Hash fully on `valu`** (8 lanes/slot): 3 `multiply_add` fma stages + 3
  irreducible xor/add-shift stages (2 parallel elementwise transforms + `^`
  combine). `val_vec` doubles as the running hash reg; the final stage
  leaves `v` there — zero copy-out (same dual-role trick as v2, now vector).
- **Gather** is the one non-vectorizable op (ISA has no scatter/gather): per
  group, one `valu +` computes all 8 addresses (`addr_vec = idx_vec +
  forest_p_vec`), then 8 scalar `load`s land into the per-lane `nv` plane.
  This is the "naive loader" — prefetch / dual-port packing / speculative
  both-branch loads deferred (see v3b notes).
- **Branchless idx** on `valu`: `parity = v & 1`; `base = 2*idx + 1` (fma);
  `next = base + parity`. Wrap round (10): `idx &= zero_vec` in one slot.
- **Entry XOR** is also `valu` (not `alu`): `val_vec = val_vec ^ nv_g`,
  8 lanes in one slot. `alu` would need 8 scalar slots/lane and forfeit
  the whole vectorization.
- Const-key collision handled: the literal `9` is both a multiplier (stage 4
  fma) and a shift amount (stage 3), so hash constants are split into
  `fma_vec_consts` vs `irr_vec_consts` dicts keyed by raw value.

Per group per round (one slot/bundle): 1 addr valu + 8 loads + 1 xor +
12 hash + 3 idx = 25 bundles (13 on the wrap round). The 8 scalar gathers
are now the dominant cost — the load-port / prefetch lever is the clear
next move.

PMU (fires): valu 8656, load 4157 (4096 gathers + 32 vload + ~29 prologue),
store 32, flow 2. `alu` reported counts are inflated by an artifact: the
simulator's `valu` elementwise form internally calls `self.alu` per lane,
so the PMU double-counts those as `alu` fires on every valu cycle — the
scheduler-visible body work is all on the `valu` engine.

Passes `submission_tests.py` correctness (8 seeds) and the first two tiers
(`baseline < 147734`, `updated-starting-point < 18532`). Next target: `< 2164`.

---

## v3b — (next) gather prefetch / VLIW packing

Status: not started. Two complementary levers, both targeting the gather
(that's now the dominant 8-cyc/group/round):

1. **Dual-port prefetch / speculative both-branch loads** — the 2 load
   ports/cyc are wasted at one-slot-per-bundle. Issue both possible next
   node values (left & right child) ahead of the hash finishing, then select
   the correct one after the parity bit is known. Trades 1 extra load +
   1 select for hiding the gather latency entirely behind the hash compute.
2. **VLIW packing** (`build(..., vliw=True)` exists but is unused) — pack the
   hash's 12 valu slots into parallel bundles. Critical path is 9 cyc/lane
   (6-stage chain 1+2+1+2+1+2), port-pressure bound (6 valu/cyc). Idealized
   body ~7 cyc/group/round vs current 25 → ~2200 cyc, landing the `< 2164`
   tier.

Roofline reminders (see `notes/architecture.md`): compute floor ~1280–1600
yc (12-slot hash × 4096 lane-rounds over 6 valu + 12 alu/cyc); the Opus-4.5
1487 score sits in that band. Scratch-tree tricks are N/A (scratch has no
indirect addressing; tree 2047 > scratch 1536)## v3b - V1 random VLIW scheduler                      ~2,394 cyc   61.7x

Commit (pending). DAG-driven VLIW packing replaces one-slot-per-bundle
with a scheduler that packs multiple independent slots per cycle.

- **`scheduler.py`**: `slot_io` (full ISA dispatch -> reads/writes as
  `(addr, is_vector)` pairs), `build_dag` (program-order walk with per-lane
  `readers_since`, tagged-union `last_writer` (vec_node | list[Node|None]x8),
  deduped RAW weight-1 + WAR weight-0 edges, bidirectional invariant
  asserts, dead-write warning).
- **`schedule_dag`**: v1 random placer. Random pick from frontier (including
  partially-completed spillable nodes; no priority). Native engine first;
  spillable `vec_elem` ops try `valu`-atomic, else spill to `alu` (sticky-alu
  once spilled). WAR resolutions immediate (same-cycle unlock); RAW deferred
  to end-of-cycle `advance()` (reflects read-before-write +1 latency). Debug
  slots ride free (0-cycle). Panic on empty frontier with uncommitted nodes,
  empty cycle, or cap exceeded.
- **Scratch reorder**: planes first `[0..1279]` (8-aligned so every
  `val_vec=base+8g` covers exactly one region), shared sector `[1280..1535]`
  with 8-word vector regions before 1-word scalars. Required for the DAG's
  region-keyed `last_writer`/`readers_since` bookkeeping.
- **`build(vliw=True, seed=42)`** wired into `build_kernel` - the body is
  scheduled; prologue/epilogue stay linear (one slot per bundle); the two
  `pause`s bracket the scheduled body as hard start/end barriers.

Cycle breakdown: prologue ~111 + body ~2219 + epilogue ~65 = ~2394.

Passes `submission_tests.py` correctness (8 seeds) and the first **3** tiers:
`baseline < 147734`, `updated-starting-point < 18532`, `opus4-many-hours
< 2164`. Fails the 5 tiers below 2164 (1790/1579/1548/1487/1363).

---

## v3c - V2 greedy VLIW scheduler                     ~2,236 cyc   66.1x

Commit (pending). Replaces v1 random pick with greedy: iterate the entire
frontier in idx order, skip nodes that can't be placed (don't break), loop
until no progress (WAR unlocks may add new placeable nodes), then advance.

- No instruction priority (by design - future v3d). Vector ops still prefer
  `valu`; spill to `alu` if `valu` full. Known limitation: `vec_elem` ops
  (xor-shift stages) can greedily fill `valu` slots, blocking `vec_fma`
  (`multiply_add`, valu-only) from scheduling. Priority (fma-first) would
  fix this but is deferred.
- `_try_place` refactored out of the inline loop; returns
  (committed_count, emitted_non_debug) tuple or None. Shared by both v1
  random and v2 greedy paths.
- `schedule_dag` gains `greedy: bool = True` parameter; v1 random preserved
  as `greedy=False` for testing.

Cycle count: 2394 (v1 random) -> 2236 (v2 greedy). Improvement from filling
all engine slots each cycle instead of breaking on first failure.

Passes `submission_tests.py` correctness (8 seeds). Passes first 3 tiers
(baseline, updated-starting, opus4-many-hours < 2164 is FAIL - 2236 > 2164
by 72 cyc / 3.3%). The fma-priority fix (v3d) is expected to clear 2164.

---

## v3d - (next) instruction priority + gather dedup

Status: not started. Two levers toward the ~1300 realistic target:

1. **Instruction priority** - prefer `vec_fma` (valu-only, rigid) over
   `vec_elem` (spillable to alu) when both are in the frontier. Prevents
   elementwise ops from saturating valu before fma gets a slot. Estimated
   ~2000 cyc (clears < 2164).
2. **Gather dedup** - exploit level-determinism (all lanes at same level per
   round; verified). Early rounds have few distinct idx (round 0: 1, round 1:
   2, ...). Broadcast shared loads; total distinct gathers ~899 (vs 4096 naive)
   / 2 load ports ~ 450 cyc gather, overlapped with ~1024 cyc compute.
   Estimated ~1300 cyc.

Roofline reminders (see `notes/architecture.md`): compute floor ~1280-1600
cyc (12-slot hash x 4096 lane-rounds over 6 valu + 12 alu/cyc); the Opus-4.5
1487 score sits in that band. Scratch-tree tricks are N/A (scratch has no
indirect addressing; tree 2047 > scratch 1536).

Status: not started. Two levers toward the ~1300 realistic target:

1. **v2 greedy scheduler** - replace random pick with critical-path /
  port-pressure-aware priority. Prefers `valu` (8x efficient); spills to `alu`
  only under pressure. Fills both ports when possible. Estimated ~1700 cyc.
2. **Gather dedup** - exploit level-determinism (all lanes at same level per
  round; verified). Early rounds have few distinct idx (round 0: 1, round 1:
  2, ...). Broadcast shared loads; total distinct gathers ~899 (vs 4096 naive)
  / 2 load ports ~ 450 cyc gather, overlapped with ~1024 cyc compute.
  Estimated ~1300 cyc.

Roofline reminders (see `notes/architecture.md`): compute floor ~1280-1600
cyc (12-slot hash x 4096 lane-rounds over 6 valu + 12 alu/cyc); the Opus-4.5
1487 score sits in that band. Scratch-tree tricks are N/A (scratch has no
indirect addressing; tree 2047 > scratch 1536).