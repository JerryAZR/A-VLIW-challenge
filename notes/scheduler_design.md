# VLIW Scheduler — Design

Status: **designed, not yet working.** The grouped-vector kernel (`build_kernel`,
v3a, 12 911 cyc, one slot per bundle) is the current committed implementation.
This doc captures the DAG-scheduler design we converged on, and the open bug
blocking it, so the next session can pick up cleanly.

## Why a scheduler (vs manual pipelining)

The body is ~17.8k slots (16 rounds × 32 groups × ~35 slots/group, incl. debug)
across three independent parallelism dimensions:

1. **within-group** — the 12-slot hash DAG (critical path 9 cyc).
2. **across-group** — 32 groups are fully independent (disjoint per-lane SoA
   planes); this is the bulk of the `valu` parallelism.
3. **across-round** — round r+1's gather can overlap round r's hash tail.

Hand-pipelining 17.8k ops across all three is error-prone; one mis-counted
dependency is a silent wrong answer. A scheduler handles them uniformly via
dependency edges, and `build(slots, vliw=True)` is the existing hook for it.

## The machine's dependency model (the key insight)

The machine is read-before-write within a cycle: all slots read **pre-cycle**
state, all writes commit at **end of cycle**. From this, exactly two edge kinds
are needed — **no WAW edges**:

- **RAW** (producer W → consumer R): R must be ≥ W's cycle + 1. Edge weight **1**.
- **WAR** (reader R → next writer W' of the same addr): W' must be ≥ R's cycle.
  Same cycle is **safe** (R reads the old pre-cycle value, W' commits the new
  one at end-of-cycle). Edge weight **0**.
- **WAW**: not needed. Any same-address write pair is bridged by a transitive
  RAW path *through the intervening reader* (the next writer reads something
  that traces back to this writer), so the two writes can never be co-scheduled.
  If there is **no** intervening reader, the first write is **dead** and is
  eliminated outright (dead-write elimination) — that is the entirety of WAW
  handling.

This was the subtle part: WAR is *not* "no constraint" — it's a weight-0
ordering edge (W' ≥ R). With RAW(1) + WAR(0), the chain
`W → R (RAW+1) → W' (WAR+0)` forces `W' ≥ W+1`, so consecutive writes to the
same addr are always strictly ordered, never same-cycle, never out of order.
Confirmed by auditing every multiply-written scratch addr in the v3a IR
(`addr_vec`, `val[i]`, `idx[i]`, `t1[i]`/`t2[i]`, `nv[i]`) — each is bridged by
RAW through the round/group carry chain.

## DAG construction

- **Node** = one logical slot. Each node carries:
  `reads`/`writes` (sets of individual scratch addresses, vector ranges
  expanded to 8 lanes), `lanes_total`, `engine_kind`/`engine_options`/`atomic`
  (realisation info — see partial schedules), `in_edges`/`out_edges`
  (lists of `(src_or_dst, weight)`), `unresolved` (count of in-edges whose
  source hasn't committed), `ready_cycle`, `lanes_done`, `engine_choice`,
  `commit_cycle`.
- **`slot_io(engine, slot) → (reads, writes)`** — a ~40-line dispatch over
  every ISA form (alu, valu×3, load×4, store×2, flow, debug) that returns the
  scratch addresses read/written. Memory accesses are ignored for dependency
  purposes (the body only reads the read-only tree; scratch has no indirect
  reads, so scratch-only deps suffice).
- **Program-order walk** with `last_writer[addr]` and `readers_since[addr]`:
  - RAW: each read of addr → edge from `last_writer[addr]`, weight 1.
  - WAR: each new write of addr → edges from every reader-of-addr-since-the-last-
    write to this new write, weight 0; then reset `readers_since[addr]`.
  - Self-edges (a node reads and writes the same addr, e.g. `v ^= x`) are
    skipped — the slot reads old & writes new atomically.
- **Dead-write elimination** (post-pass): a single-destination writer with no
  reader in `(this writer, next writer of the same addr)` is dropped along with
  its edges. (No-op on the v3a IR — every write is read — but kept for
  generality / future IRs.)

## Partial schedules (the second key idea)

The body is mostly 8-lane vector ops. Normally each realises as **one `valu`
slot** (all 8 lanes, clean). But under `valu` pressure (6/cyc cap), a
**spillable** vector node can instead be realised as **`alu` scalar slots**
(one per lane, since `alu` supports the same binary ops and the per-lane SoA
planes make each lane's operand a contiguous word). And it can **split across
cycles**: 4 `alu` lanes this cycle + 4 next.

Node realisation classification (`_classify`):

| kind | engine_options | atomic | lanes_total | notes |
|---|---|---|---|---|
| `fma_rigid` | `[valu]` | valu ✓ | 8 | `multiply_add` — **no scalar fma in the ISA**, so rigid; never spills |
| `elem_spill` | `[valu, alu]` | valu ✓, alu ✗ | 8 | elementwise `^`,`<<`,`>>`,`+`,`&` — spills to alu, splittable |
| `load` | `[load]` | load ✓ | 1 | scalar gather — already pre-split (8 separate nodes) |
| `debug` | `[debug]` | debug ✓ | 0 | read-only, free (0 slots, 0-cycle bundle) |

("loads need that" — the 8 scalar gathers are the canonical pre-split form;
"some alus could use that" — only the non-fma elementwise hash slots can spill
to alu; fma can't.) Sticky engine: once a node starts on `alu` it finishes on
`alu` (`valu` is fixed-8-lane, can't do a partial 4). A `lanes_done` counter
tracks partial completion; the node commits (fires its RAW/WAR relax into
consumers) only when `lanes_done == lanes_total`, at the cycle its last lane
landed. (Alternative realisation: node replacement — swap one vector node for 8
scalar alu sub-nodes; equivalent, the counter is just less node-bloat.)

## Frontier + partial schedules (efficient scheduling)

Maintain, per not-yet-placed node, a `ready_cycle` lower bound =
`max over in-edges (u→v, w) of (cycle[u] + w)` over placed sources. When a node
is placed, **relax** `ready_cycle` along each out-edge into its consumers (take
the running max). This per-node bound *is* the partial schedule — it tightens
monotonically as producers land. No global re-scan per cycle.

- `frontier` = nodes with all in-edges placed and `ready_cycle ≤ C`.
- `pending[rc]` = nodes with all in-edges placed but `ready_cycle == rc > C`
  (waiting out the +1 latency); bucketed by `ready_cycle` so promotion is
  O(newly-due).
- `in_flight` = partially-completed spillable nodes (0 < `lanes_done` <
  `lanes_total`), retried each cycle until full.
- Per cycle: one shuffle of `frontier ∪ in_flight`; place lanes (respecting
  per-engine caps); a **bounded follow-up** for nodes newly unlocked this cycle
  by same-cycle WAR (weight-0) edges. O(E) total work across the schedule.

## Scheduler policy

- **v1: random valid scheduler first.** Pick randomly among all legal
  realisations (valu-atomic vs alu-split, and for alu-split a random lane count
  1..free) — this stresses the spill/partial code paths and the DAG edges,
  catching bugs early. Correctness gate is free and strong: the dev
  `Tests.test_kernel_cycles` (runs with `enable_debug=True`, every `vcompare`
  asserts against the reference trace) plus `submission_tests.py` (8 seeds).
  Expect a correct-but-terrible cycle count.
- v2: greedy / critical-path (ALAP-level priority) — a one-line priority swap;
  prefer `valu` (8× efficient), spill to `alu` only under pressure.

## Wiring

`build(slots, vliw=True, seed=None)` → `build_dag(slots)` → `schedule_dag(...)`
→ list of bundles (`dict[engine, list[slot]]`), one per scheduled cycle.
`build_kernel` routes the body through `vliw=True`; prologue/epilogue stay
linear (one slot per bundle); the two `pause`s bracket the scheduled body as
hard start/end barriers.

**No bubbles needed:** the machine's cycle count = number of non-debug bundles
emitted. Every RAW pair is ≥1 bundle apart by construction (a consumer is never
ready until its producer is emitted, and adjacency supplies the minimum +1;
interleaved work only increases the gap). Skipped scheduled cycles just mean
fewer interleaved bundles, never fewer than +1.

**Safety asserts** (dev-only, cheap): per-cycle write-addr collision check
(two same-cycle writers to one scratch addr = a missing-edge bug; the machine's
dict-last-wins would otherwise hide it as a silent wrong answer), and a
ready-invariant check in `relax` (a node whose `unresolved` hits 0 must have
every in-edge source committed).

## Open bug (blocking)

The scheduler as drafted hits a ready-invariant violation almost immediately:
a consumer node's `unresolved` counter reaches 0 while one of its RAW sources
(a gather load) is still uncommitted, so it gets placed too early and collides
with another writer on the same scratch address.

Concrete failing case (seed=1): node 1061 (an entry-XOR `valu` node,
`('^', 399, 399, 1423)`) reads `val[399..406]` from producer 1060 (8 lanes → 8
RAW edges) **and** `nv[1423..1430]` from 8 gather loads 1051–1058 (1 RAW edge
each). So `1061.in_edges` has 16 entries (8 from 1060 + 8 from the loads) and
`unresolved` should be 16. But when 1060 commits, `1061.unresolved` is already
at 8, so 1060's relax (8 decrements) brings it to 0 — yet the 8 loads are
still uncommitted. The in-edge count (16) and the `unresolved` count (8)
disagree for this vector consumer.

The `slot_io` RAW/edge-append and the `unresolved += 1` are in the same
`if a in last_writer:` branch, so for scalar reasoning they should agree. The
discrepancy suggests either an edge being added to `in_edges` without
incrementing `unresolved` (a path-merge bug in `build_dag`), or `unresolved`
being decremented by an out-edge whose matching in-edge was never counted
(an asymmetry between a producer's `out_edges` and the consumer's `in_edges`).
The duplicate-edge situation (each vector op adds 8 identical RAW edges from
the same producer, one per lane) is a likely confound — it inflates both counts
8× but they should still agree; need to verify they actually do under all
read/write patterns (e.g. a node reading the same vector via two operands, or
a vector read where some lanes' `last_writer` differs).

**Next step when resuming:** add an assertion in `build_dag` that for every
node, `unresolved == len(in_edges_with_resolved_source)` holds at construction
time, and that for every edge `(src→dst)` the counts match at both endpoints
(`#(dst in src.out_edges) == #(src in dst.in_edges)`). That will localise
whether the asymmetry is introduced at build time or by the (empty-on-this-IR)
dead-write pass. Deduplicating edges per (src, dst, weight) at build time is
also worth doing — it removes the 8× bloat and any confounds it brings, and
shrinks the graph ~8× (the `unresolved` counter then counts *distinct* source
nodes, which is the conceptually cleaner model).

## Estimated impact once working

- v1 random: correct, large cycle count (stress test).
- v2 greedy VLIW packing, no gather dedup: ~2100 cyc (256 scalar gathers/round
  ÷ 2 load ports = 128 cyc/round × 16 = 2048 cyc gather-bound) — borderline
  `< 2164`.
- + gather dedup (coincident-idx broadcast, an IR transformation feeding the
  scheduler, ~830 distinct tree values total → ~415 cyc of loads + ~500–700
  cyc distribute): ~600–800 cyc gather + ~864 cyc hash compute → ~1500–1700
  cyc, the 1579/1487 band. (See `architecture.md` roofline.)

---

# V1 random scheduler — pinned spec (implementation-bound)

## Node classification (`_classify`)

| kind         | atomic | lanes_total | native_engine | spill?              |
|--------------|--------|-------------|---------------|---------------------|
| `alu_scalar` | yes    | 1           | alu           | no (already scalar) |
| `load`       | yes    | 1           | load          | no (one word)       |
| `store`      | yes    | 1           | store         | no                  |
| `flow`       | yes    | 1           | flow          | no                  |
| `debug`      | yes    | **0**       | (free / rides along) | no            |
| `vec_fma`    | yes    | 1           | **valu**      | **NO** (no scalar fma in the ISA) |
| `vec_elem`   | **no** | 8           | valu          | yes — spills to alu, splittable |

`atomic = False` only for `vec_elem` (the xor/add-shift elementwise stages of
the hash; these are the ops that can run on either `valu` or `alu`). `vec_fma`
is atomic because `multiply_add` has no scalar form on `alu` — picked as one
`valu` slot or break the cycle.

`lanes_total = 0` for debug: debug emits no slot, consumes no port, "completes"
the moment it's placed and just commits. A cycle that emits only debug slots is
an empty cycle (`debug` rides along free in cycles that already have non-debug
work) → panic per the "no bubbles" rule.

## Spill rule (simplified — sticky alu, no self-RW carve-out)

Once a `vec_elem` node has spilled any lanes to `alu`, **all remaining lanes
must go to alu** on subsequent cycles — no re-issue on `valu`. Rationale: a
self-RW vector op (`dest ∈ reads`, e.g. our fma-on-`val_vec` or the entry
`val ^= node_val`) sees its source corrupted after the first partial-alu write
(already-written lanes hold the NEW value; valu would re-read them as inputs
and compute the wrong result). Avoids per-node self-RW tracking by making
stickiness universal. Non-self-RW (e.g. `t1 = val ^ K1`) could be safely
refilled on valu, but it wastes compute and adds a classifier; pre-emptively
forbidden by the sticky rule.

## V1 placement algorithm

```
schedule_dag(nodes, frontier, *, seed=None) -> list[Instruction]:

  cap = sequential body cycle count
        (run build(body, vliw=False) through the simulator; record machine.cycle.
         The unscheduled body is the worst case: every slot its own bundle.
         Exceeding this cap means the placer has regressed below the baseline —
         bug, panic.)

  bundles = []
  C = 0
  while not all nodes committed:
    # Per-cycle queue: random pick from the frontier (including partially-
    # completed nodes — they sit in frontier with lanes_done > 0).
    while frontier non-empty:
      n = random.choice(frontier)           # uniform, no priority
      
      if n already committed: discard; continue
      
      # Compute take (lanes this placement can grab).
      if n.kind == 'debug':
        commit(n)                            # no slot emitted, no port consumed
        continue                             # does NOT break the cycle loop; debug commit
                                               may unlock same-cycle WAR children for non-debug placement
      elif lanes_total == 1 (atomic):        # alu/load/store/flow/vec_fma
        eng = native_engine
        if free[eng] == 0:
          break                              # no retry; advance cycle
        emit 1 (eng, n.slot); free[eng] -= 1
        n.lanes_done = lanes_total
        commit(n)
      else:                                   # vec_elem (spillable)
        if n.lanes_done > 0:                  # partial → sticky alu (no re-spill to valu)
          take = min(8 - n.lanes_done, free.alu)
          if take == 0: break
          emit take alu slots (dest+lane, a+lane, b+lane, ...)
          n.lanes_done += take; free.alu -= take
          if n.lanes_done == 8: commit(n)    # else leave in frontier (partial)
        elif free.valu > 0:                   # fresh & valu available → atomic
          emit 1 ('valu', n.slot)
          n.lanes_done = 8; free.valu -= 1
          commit(n)
        else:                                  # fresh, valu saturated → spill
          take = min(8, free.alu)
          if take == 0: break
          emit take alu slots (lane-split from original vec slot tuple)
          n.lanes_done += take; free.alu -= take
          n.engine_choice = 'alu'             # sticky marker
          if n.lanes_done == 8: commit(n)    # possible only if alu_free >= 8
      # never implicit break — explicit in each path've done above
      
    # Cycle done — check for empty cycle:
    if this cycle emitted zero non-debug slots:
      panic("empty cycle — scheduler stuck; cyclic DAG or counter bug")
    # Apply end-of-cycle RAW resolutions:
    advance()
    # Record this cycle's bundle:
    bundles.append(this_cycle_bundle)        # dict[engine, list[slot]]
    C += 1
    if C > cap: panic("scheduler regressed below unscheduled baseline")
  return bundles
```

## `commit(n)` semantics

- `n.committed = True; n.commit_cycle = C; n.in_frontier = False; frontier.discard(n)`
- For each `(dst_idx, w) in n.out_edges`:
  - `w == 0` (WAR, same-cycle-safe): `dst.war_blockers -= 1`. If now `dst.raw_blockers + dst.war_blockers == 0` and not `dst.in_frontier`: add to frontier.
  - `w == 1` (RAW, deferred): `dst.incoming += 1`. Apply at `advance()`.

A node that completes same-cycle fires WAR into its children immediately — so a
WAR-only child can become frontier-eligible *this same cycle* (the inner while
will see it next iteration).

## `advance()` — at end of every cycle

```
advance():
  for every node n with n.incoming > 0:
    n.raw_blockers -= n.incoming
    n.incoming = 0
    if n.raw_blockers + n.war_blockers == 0 and not n.in_frontier and not n.committed:
      frontier.add(n.idx); n.in_frontier = True
```

This is the only place `raw_blockers` decreases.

## Panic conditions (any one = bug, raise)

1. Frontier empty while some node is uncommitted (**cyclic DAG or wrong counts**).
2. About to emit a bundle with zero non-debug slots (**stuck** / would never land because debug rides along free).
3. `C > cap` (regressed below unscheduled baseline).

## Three todos this version does NOT do (future v2+)

- No priority (no critical-path, no port-pressure-aware greedy). Random pick.
- No "use alu for spillable vector work even when valu is free" — partial spills stuck on alu forever; valu idle-side filling deferred to v2.
- No gather-sharing / broadcast IR transformation (the gather-wall still bites; land ~2100 cyc with naive gather).
